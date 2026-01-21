# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash model implementation for Block Diffusion speculative decoding.

DFlash uses block diffusion to generate multiple draft tokens in parallel,
conditioned on target model hidden states.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .utils import maybe_prefix


class DFlashNoiseSchedule(nn.Module):
    """Noise schedule for the diffusion process.

    Supports cosine, linear, and sqrt schedules.
    """

    def __init__(self, num_steps: int, schedule_type: str = "cosine"):
        super().__init__()
        self.num_steps = num_steps
        self.schedule_type = schedule_type

        # Precompute alpha schedule
        if schedule_type == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
        elif schedule_type == "linear":
            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, num_steps)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)
            alpha_bar = torch.cat([torch.tensor([1.0]), alpha_bar])
        elif schedule_type == "sqrt":
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            alpha_bar = 1 - torch.sqrt(steps / num_steps + 0.0001)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule_type}")

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer(
            "sqrt_alpha_bar", torch.sqrt(alpha_bar)
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar)
        )

    def get_noise_level(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get noise levels for timestep t."""
        return self.sqrt_alpha_bar[t], self.sqrt_one_minus_alpha_bar[t]


class DFlashAttention(nn.Module):
    """Attention module for DFlash that attends to context and noisy tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            hidden_states: Query states [batch_size, seq_len, hidden_size]
            context: Key/value states for cross-attention, or None for self-attention
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Self-attention if no context provided
        if context is None:
            context = hidden_states

        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(context)
        v, _ = self.v_proj(context)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        output, _ = self.o_proj(attn_output)

        return output


class DFlashMLP(nn.Module):
    """MLP module for DFlash decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        down_input = self.act_fn(gate) * up
        output, _ = self.down_proj(down_input)
        return output


class DFlashDecoderLayer(nn.Module):
    """Decoder layer for DFlash with self-attention, cross-attention, and MLP."""

    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention on noisy tokens
        self.self_attn = DFlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.self_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Cross-attention to context (target hidden states)
        self.cross_attn = DFlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
        )
        self.cross_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Feed-forward network
        self.mlp = DFlashMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # Cross-attention with residual
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, context)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashModel(nn.Module):
    """DFlash model for Block Diffusion speculative decoding.

    This model generates draft tokens using iterative denoising, conditioned
    on hidden states from the target model.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.num_diffusion_steps = config.num_diffusion_steps

        # Noise schedule
        self.noise_schedule = DFlashNoiseSchedule(
            num_steps=config.num_diffusion_steps,
            schedule_type=config.noise_schedule,
        )

        # Token embedding for the draft tokens
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Position embedding for draft positions
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.hidden_size,
        )

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection to vocab
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        # Logits processor
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def _get_timestep_embedding(
        self, timesteps: torch.Tensor, dim: int
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        context_hidden_states: torch.Tensor,
        noisy_tokens: torch.Tensor | None = None,
        timestep: int | None = None,
    ) -> torch.Tensor:
        """Forward pass for denoising.

        Args:
            context_hidden_states: Hidden states from target model [batch_size, hidden_size]
            noisy_tokens: Current noisy token embeddings [batch_size, block_size, hidden_size]
            timestep: Current diffusion timestep

        Returns:
            Denoised hidden states [batch_size, block_size, hidden_size]
        """
        batch_size = context_hidden_states.shape[0]
        device = context_hidden_states.device

        # Expand context to have sequence dimension if needed
        if context_hidden_states.dim() == 2:
            context_hidden_states = context_hidden_states.unsqueeze(1)

        # Initialize noisy tokens if not provided (for initial noise)
        if noisy_tokens is None:
            noisy_tokens = torch.randn(
                batch_size, self.block_size, self.hidden_size,
                device=device, dtype=context_hidden_states.dtype
            )

        # Add position embeddings
        positions = torch.arange(self.block_size, device=device)
        pos_emb = self.position_embedding(positions)
        hidden_states = noisy_tokens + pos_emb.unsqueeze(0)

        # Add timestep embedding
        if timestep is not None:
            t_emb = self._get_timestep_embedding(
                torch.tensor([timestep], device=device),
                self.hidden_size
            )
            t_emb = self.timestep_embedding(t_emb)
            hidden_states = hidden_states + t_emb.unsqueeze(1)

        # Apply decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, context_hidden_states)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: [batch_size, block_size, hidden_size]

        Returns:
            logits: [batch_size, block_size, vocab_size]
        """
        # Reshape for logits processor
        batch_size, block_size, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        logits = self.logits_processor(self.lm_head, hidden_states_flat)

        if logits is not None:
            logits = logits.view(batch_size, block_size, -1)

        return logits

    @torch.inference_mode()
    def generate_draft_tokens(
        self,
        context_hidden_states: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate draft tokens using iterative denoising.

        Args:
            context_hidden_states: Hidden states from target model [batch_size, hidden_size]
            seed: Optional random seed for reproducibility. If provided, the same
                  seed with the same inputs will produce identical draft tokens.

        Returns:
            draft_token_ids: [batch_size, block_size]
        """
        batch_size = context_hidden_states.shape[0]
        device = context_hidden_states.device
        dtype = context_hidden_states.dtype

        # Set seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        # Start with random noise
        x_t = torch.randn(
            batch_size, self.block_size, self.hidden_size,
            device=device, dtype=dtype,
            generator=generator,
        )

        # Iterative denoising
        for t in reversed(range(self.num_diffusion_steps)):
            # Predict denoised hidden states
            predicted_x0 = self.forward(
                context_hidden_states,
                noisy_tokens=x_t,
                timestep=t,
            )

            if t > 0:
                # Get noise schedule parameters
                sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t = \
                    self.noise_schedule.get_noise_level(t)
                sqrt_alpha_bar_t_prev, sqrt_one_minus_alpha_bar_t_prev = \
                    self.noise_schedule.get_noise_level(t - 1)

                # DDPM sampling step
                # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0 + sqrt(1 - alpha_bar_{t-1}) * noise
                noise = torch.randn(
                    x_t.shape, device=device, dtype=dtype,
                    generator=generator,
                )
                x_t = sqrt_alpha_bar_t_prev * predicted_x0 + \
                      sqrt_one_minus_alpha_bar_t_prev * noise
            else:
                x_t = predicted_x0

        # Compute logits and get token ids
        logits = self.compute_logits(x_t)

        if logits is None:
            # Return empty tensor for non-primary ranks
            return torch.zeros(batch_size, self.block_size, dtype=torch.long, device=device)

        draft_token_ids = logits.argmax(dim=-1)

        return draft_token_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle weight name mappings
            name = name.replace("model.", "")

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
