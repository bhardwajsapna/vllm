# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash configuration for Block Diffusion speculative decoding."""

import os

from transformers import PretrainedConfig


class DFlashConfig(PretrainedConfig):
    """Configuration class for DFlash (Block Diffusion for Flash Speculative Decoding).

    DFlash uses block diffusion to generate multiple draft tokens in parallel,
    conditioned on target model hidden states.
    """

    model_type = "dflash"

    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        block_size: int = 8,
        num_diffusion_steps: int = 8,
        noise_schedule: str = "cosine",
        num_attention_heads: int = 32,
        num_hidden_layers: int = 2,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        **kwargs,
    ):
        """Initialize DFlash configuration.

        Args:
            hidden_size: Size of the hidden states from the target model.
            vocab_size: Size of the vocabulary.
            block_size: Number of tokens to generate in parallel (draft length).
            num_diffusion_steps: Number of denoising steps in the diffusion process.
            noise_schedule: Type of noise schedule ("cosine", "linear", "sqrt").
            num_attention_heads: Number of attention heads in decoder layers.
            num_hidden_layers: Number of decoder layers.
            intermediate_size: Size of the MLP intermediate layer.
            max_position_embeddings: Maximum sequence length.
            rms_norm_eps: Epsilon for RMS normalization.
        """
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_schedule = noise_schedule
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.max_seq_len = int(2**20)

        if "architectures" not in kwargs:
            kwargs["architectures"] = ["DFlashModel"]

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "DFlashConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)

    @property
    def num_lookahead_tokens(self) -> int:
        """Return the number of draft tokens (block_size)."""
        return self.block_size

    @num_lookahead_tokens.setter
    def num_lookahead_tokens(self, num_lookahead_tokens: int) -> None:
        self.block_size = num_lookahead_tokens
