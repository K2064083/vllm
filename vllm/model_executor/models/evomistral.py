from typing import Optional, List, Iterable, Set, Tuple, Type, Union

import torch
from torch import nn

from vllm.config import LoRAConfig, VllmConfig, CacheConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from .llama import LlamaModel as _LlamaModel, LlamaDecoderLayer, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE, LogitsProcessor, get_sampler
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter, make_layers, maybe_prefix,)

class EvoMistralModel(_LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: Type[LlamaDecoderLayer] = LlamaDecoderLayer,
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        config = vllm_config.model_config.hf_config
        self.input_scales = nn.Parameter(
            data=torch.zeros(config.num_hops).float(), requires_grad=False
        )
        self.input_layers = nn.Parameter(
            data=torch.zeros(config.num_hops).int(), requires_grad=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        print("lenlen",len(self.input_layers),len(kv_caches))
        for idx, layer_ix in enumerate(self.input_layers):
            layer = self.layers[layer_ix]
            scale = self.input_scales[idx].to(hidden_states.device)
            hidden_states, residual = layer(
                positions,
                hidden_states * scale,
                kv_caches[idx],  # index into kv_caches based on hop index
                attn_metadata,
                residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class EvoMistralForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # Reuses most of LlamaForCausalLM's configurations
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    mistral_mapping = {  # Add mistral mapping here if needed
        # ...
    }


    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.lora_config = vllm_config.lora_config
        self.model = self._init_model(vllm_config=vllm_config,
                                      prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            unpadded_vocab_size = self.config.vocab_size
            if self.lora_config:
                unpadded_vocab_size += self.lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE if not self.lora_config else
                    self.lora_config.lora_vocab_padding_size
                ),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self.config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(self.config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(unpadded_vocab_size,
                                                    self.config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (self.model.
                                                make_empty_intermediate_tensors)

    def _init_model(self, vllm_config: VllmConfig, prefix: str = ""):
        return EvoMistralModel(vllm_config=vllm_config, prefix=prefix)

    # Reuses the rest of the methods from LlamaForCausalLM (forward, compute_logits, sample, load_weights, load_kv_cache_scales, maybe_remap_mistral)
    # ... (copy the remaining methods from the new llama.py)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors,
                                  inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    # add for rewrite weights
    def rewrite_weights(self, state_dict: dict) -> Set[str]:
        weights = []
        handled_params = set()  # 処理済みのパラメータ名を格納するセット
        for name, param in state_dict.items():
            if "input_scales" in name:
                weights.append(("model.input_scales", param))
                continue
            if "input_layers" in name:
                weights.append(("model.input_layers", param))
                continue
            if "embed_tokens" in name:
                weights.append(("model.embed_tokens.weight", param))
                continue
            # パラメータ名を正規化し、Mistral のマッピングを適用
            name, param = self.maybe_remap_mistral(name, param)
            weights.append((name, param))

        loader = AutoWeightsLoader(self, skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None)) # 重みをすべて上書きするためskip_prefixesはNone
        return loader.load_weights(weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight

