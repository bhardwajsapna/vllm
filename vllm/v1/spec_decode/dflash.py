# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash proposer for Block Diffusion speculative decoding."""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class DFlashProposer:
    """DFlash proposer for generating draft tokens using block diffusion.

    DFlash generates multiple draft tokens in parallel using iterative
    denoising, conditioned on target model hidden states.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.hidden_size = (
            vllm_config.speculative_config.draft_model_config.get_hidden_size()
        )
        self.dtype = vllm_config.model_config.dtype

        # Get DFlash-specific config
        draft_config = vllm_config.speculative_config.draft_model_config.hf_config
        self.block_size = getattr(draft_config, "block_size", 8)
        self.num_diffusion_steps = getattr(draft_config, "num_diffusion_steps", 8)

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Generate draft tokens using block diffusion.

        Args:
            target_hidden_states: Hidden states from target model
                [batch_size, hidden_size] or [num_tokens, hidden_size]
            sampling_metadata: Sampling parameters and metadata

        Returns:
            draft_tokens: Tensor of draft token ids [batch_size, block_size]
        """
        # Use the model's built-in generation method
        draft_tokens = self.model.generate_draft_tokens(target_hidden_states)

        return draft_tokens

    def load_model(self, target_model: nn.Module) -> None:
        """Load the DFlash draft model.

        Args:
            target_model: The target model (used for potential weight sharing)
        """
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("dflash_head"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.speculative_config.draft_model_config,
            )

        assert not (
            is_mixture_of_experts(self.model)
            and self.vllm_config.parallel_config.enable_eplb
        ), "EPLB for DFlash is not supported"

        # Optionally share embeddings with target model
        self._maybe_share_embeddings(target_model)

    def _maybe_share_embeddings(self, target_model: nn.Module) -> None:
        """Share token embeddings with target model if compatible.

        This reduces memory usage when the draft and target models use
        the same vocabulary.
        """
        # Check if target model has embed_tokens
        if not hasattr(target_model, "model"):
            return

        target_inner = target_model.model
        if not hasattr(target_inner, "embed_tokens"):
            return

        # Check if vocab sizes match
        target_vocab_size = target_inner.embed_tokens.num_embeddings
        draft_vocab_size = self.model.embed_tokens.num_embeddings

        if target_vocab_size == draft_vocab_size:
            # Share the embedding weights
            self.model.embed_tokens.weight = target_inner.embed_tokens.weight
            logger.info(
                "Sharing token embeddings between target and DFlash draft model"
            )

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        """Perform a dummy forward pass for warmup.

        Args:
            num_tokens: Number of tokens for the dummy batch
        """
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(hidden_states)
