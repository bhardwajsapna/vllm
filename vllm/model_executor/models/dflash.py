# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash model for block diffusion speculative decoding.

A transformer decoder that supports bidirectional attention in denoising mode,
used as a draft model for DFlash speculative decoding. Reuses standard
transformer components (similar to Qwen3/Llama architecture).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .utils import maybe_prefix


class DFlashAttention(nn.Module):
    """Multi-head attention for DFlash that supports bidirectional mode."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        if positions is not None:
            q, k = self.rotary_emb(positions, q, k)

        # Reshape for attention
        batch_tokens = q.shape[0]
        q = q.view(batch_tokens, self.num_heads, self.head_dim)
        k = k.view(batch_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(batch_tokens, self.num_kv_heads, self.head_dim)

        # Simple scaled dot-product attention (for draft model forward)
        # In practice, the attention backend handles the actual computation
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2),
            k.unsqueeze(0).transpose(1, 2),
            v.unsqueeze(0).transpose(1, 2),
            scale=self.scale,
        )
        attn_output = attn_output.transpose(1, 2).squeeze(0)
        attn_output = attn_output.reshape(batch_tokens, -1)

        output, _ = self.o_proj(attn_output)
        return output


class DFlashMLP(nn.Module):
    """MLP layer for DFlash model."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        output, _ = self.down_proj(self.act_fn(gate) * up)
        return output


class DFlashDecoderLayer(nn.Module):
    """Single transformer decoder layer for DFlash."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int | None = None,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = DFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = DFlashMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashModel(nn.Module):
    """DFlash transformer model (decoder stack)."""

    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=getattr(
                        config, "num_key_value_heads", config.num_attention_heads
                    ),
                    intermediate_size=config.intermediate_size,
                    head_dim=getattr(config, "head_dim", None),
                    max_position_embeddings=getattr(
                        config, "max_position_embeddings", 8192
                    ),
                    rope_theta=getattr(config, "rope_theta", 10000.0),
                    rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=getattr(config, "rms_norm_eps", 1e-6),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class DFlashForCausalLM(nn.Module):
    """DFlash model for causal language modeling (draft model).

    Used as a draft model in DFlash block diffusion speculative decoding.
    Supports bidirectional attention during denoising steps.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        self.model = DFlashModel(config, prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, config.vocab_size, logit_scale
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the DFlash model.

        Args:
            input_ids: Input token IDs.
            positions: Position IDs (optional, used for RoPE).
            hidden_states: Pre-computed hidden states (e.g., from target model).
                If provided, skips the embedding layer.

        Returns:
            Hidden states from the model.
        """
        return self.model(input_ids, positions, hidden_states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: Model hidden states.

        Returns:
            Logits tensor of shape [num_tokens, vocab_size].
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
