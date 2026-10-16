from typing import ClassVar

import torch as t
from huggingface_hub.dataclasses import strict
from torch import nn
from transformers import (
    Cache,
    DynamicCache,
    GenerationMixin,
    PreTrainedConfig,
    PreTrainedModel,
    RopeParameters,
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeMLP,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import (
    TransformersKwargs,
    can_return_tuple,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import capture_outputs


@strict
class DenseOlmoeConfig(PreTrainedConfig):
    model_type = "dense_olmoe"
    keys_to_ignore_at_inference: ClassVar = ["past_key_values"]

    base_model_tp_plan: ClassVar = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
    }

    vocab_size: int = 50304
    hidden_size: int = 2048
    # `OlmoeConfig` has `intermediate_size` 2048 by default
    # This is a bug, it should be 1024
    intermediate_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 50279
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    clip_qkv: float | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


class DenseOlmoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DenseOlmoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OlmoeAttention(config, layer_idx)  # type: ignore
        self.mlp = OlmoeMLP(config)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: t.Tensor,
        attention_mask: t.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[t.Tensor, t.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> t.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DenseOlmoePreTrainedModel(PreTrainedModel):
    config: DenseOlmoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DenseOlmoeDecoderLayer"]  # noqa: RUF012
    _skip_keys_device_placement = ["past_key_values"]  # noqa: RUF012
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {  # noqa: RUF012
        "hidden_states": DenseOlmoeDecoderLayer,
        "attentions": OlmoeAttention,
    }

    _supports_attention_backend = True
    _can_compile_fullgraph = True


class DenseOlmoeModel(DenseOlmoePreTrainedModel):
    def __init__(self, config: DenseOlmoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [DenseOlmoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config)  # type: ignore
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: t.LongTensor | None = None,
        attention_mask: t.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: t.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                t.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(  # diff with mixtral: no sliding
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class DenseOlmoeForCausalLM(DenseOlmoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # noqa: RUF012
    _tp_plan = {"lm_head": "colwise_gather_output"}  # noqa: RUF012
    _pp_plan = {"lm_head": ("hidden_states", "logits")}  # noqa: RUF012
    _fsdp_plan = {"lm_head": "keep_full_weight"}  # noqa: RUF012

    def __init__(self, config: DenseOlmoeConfig):
        super().__init__(config)
        self.model = DenseOlmoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: t.LongTensor | None = None,
        attention_mask: t.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: t.FloatTensor | None = None,
        labels: t.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | t.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        assert hidden_states is not None

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
