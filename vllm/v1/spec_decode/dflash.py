# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash proposer for block diffusion speculative decoding.

DFlash uses discrete masked diffusion to generate draft tokens in parallel
through iterative denoising. Instead of autoregressive token-by-token
generation, it initializes a block of MASK tokens and refines them over
multiple denoising steps, achieving significant speedup.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.sample.metadata import SamplingMetadata

logger = init_logger(__name__)


class DFlashProposer:
    """DFlash proposer using iterative denoising for draft token generation.

    The denoising process:
    1. Initialize a block of MASK tokens
    2. For each denoising step:
       a. Run forward pass with context + current draft block
       b. Sample tokens for masked positions
       c. Re-mask low-confidence positions (except on final step)
    3. Return final draft tokens
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens
        )
        self.dflash_block_size = self.speculative_config.dflash_block_size
        self.num_denoising_steps = (
            self.speculative_config.dflash_num_denoising_steps
        )
        self.mask_token_id: int = -1  # Will be set in load_model

        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.hidden_size = (
            self.speculative_config.draft_model_config.get_hidden_size()
        )
        self.dtype = vllm_config.model_config.dtype

        # Compute the keep schedule: fraction of tokens to keep at each step
        # Monotonically increases from step 0 to step N-1 (final step keeps all)
        self.keep_schedule = self._compute_keep_schedule()

    def _compute_keep_schedule(self) -> list[float]:
        """Compute the fraction of tokens to keep at each denoising step.

        Returns a monotonically increasing schedule from a small fraction
        to 1.0 (keep all) on the final step.
        """
        n = self.num_denoising_steps
        if n <= 1:
            return [1.0]
        schedule = []
        for i in range(n):
            # Linear schedule: keep_ratio increases from 1/n to 1.0
            keep_ratio = (i + 1) / n
            schedule.append(keep_ratio)
        return schedule

    def _detect_mask_token_id(self) -> int:
        """Detect the mask token ID from the tokenizer or config.

        Looks for common mask token names: <|MASK|>, <mask>, [MASK].
        Falls back to the configured value or raises an error.
        """
        configured = self.speculative_config.dflash_mask_token_id
        if configured is not None:
            return configured

        # Try to get from tokenizer
        try:
            tokenizer = self.vllm_config.model_config.get_tokenizer()
            # Check common mask token names
            for name in ["<|MASK|>", "<mask>", "[MASK]", "<MASK>"]:
                token_ids = tokenizer.encode(name, add_special_tokens=False)
                if len(token_ids) == 1:
                    return token_ids[0]
            # Check if tokenizer has a mask_token attribute
            if (
                hasattr(tokenizer, "mask_token_id")
                and tokenizer.mask_token_id is not None
            ):
                return tokenizer.mask_token_id
        except Exception:
            pass

        raise ValueError(
            "Could not auto-detect mask token ID for DFlash. "
            "Please set `dflash_mask_token_id` in speculative_config."
        )

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Generate draft tokens using iterative denoising.

        Args:
            target_hidden_states: Hidden states from the target model for the
                last token of each sequence. Shape: [batch_size, hidden_size]
            sampling_metadata: Sampling parameters for the batch.

        Returns:
            draft_token_ids: Generated draft tokens.
                Shape: [batch_size, num_speculative_tokens]
        """
        batch_size = target_hidden_states.shape[0]
        num_spec_tokens = self.num_speculative_tokens

        # Initialize draft block with MASK tokens
        draft_tokens = torch.full(
            (batch_size, num_spec_tokens),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )

        # Iterative denoising loop
        for step in range(self.num_denoising_steps):
            is_final_step = step == self.num_denoising_steps - 1
            keep_ratio = self.keep_schedule[step]

            # Prepare input: embed current draft tokens
            input_ids = draft_tokens.view(-1)  # [batch_size * num_spec_tokens]

            # Forward pass through draft model
            num_tokens = input_ids.shape[0]
            with set_forward_context(
                None, self.vllm_config, num_tokens=num_tokens
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=None,
                    hidden_states=target_hidden_states,
                )

            # Compute logits for draft positions
            logits = self.model.compute_logits(hidden_states)
            # logits shape: [batch_size * num_spec_tokens, vocab_size]
            logits = logits.view(batch_size, num_spec_tokens, -1)

            # Sample tokens for all positions
            # Use argmax (greedy) for draft token generation
            sampled_tokens = logits.argmax(dim=-1)
            # sampled_tokens shape: [batch_size, num_spec_tokens]

            if is_final_step:
                # Final step: accept all sampled tokens
                draft_tokens = sampled_tokens
            else:
                # Intermediate step: apply confidence-based re-masking
                draft_tokens = self._denoise_step(
                    sampled_tokens, logits, keep_ratio
                )

        return draft_tokens

    def _denoise_step(
        self,
        sampled_tokens: torch.Tensor,
        logits: torch.Tensor,
        keep_ratio: float,
    ) -> torch.Tensor:
        """Apply confidence-based re-masking for intermediate denoising steps.

        Keeps the top `keep_ratio` fraction of tokens by confidence
        (max probability), and re-masks the rest.

        Args:
            sampled_tokens: Sampled token IDs. [batch_size, num_spec_tokens]
            logits: Logits from the model. [batch_size, num_spec_tokens, vocab_size]
            keep_ratio: Fraction of tokens to keep (not re-mask).

        Returns:
            Updated draft tokens with low-confidence positions re-masked.
        """
        batch_size, num_spec_tokens = sampled_tokens.shape

        # Compute confidence as max probability
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [batch_size, num_spec_tokens]

        # Determine how many tokens to keep per sequence
        num_keep = max(1, int(keep_ratio * num_spec_tokens))

        # Get top-k confident positions
        _, top_indices = confidence.topk(num_keep, dim=-1)

        # Create mask: True for positions to keep
        keep_mask = torch.zeros_like(sampled_tokens, dtype=torch.bool)
        keep_mask.scatter_(1, top_indices, True)

        # Keep high-confidence tokens, re-mask low-confidence ones
        result = torch.where(
            keep_mask, sampled_tokens,
            torch.full_like(sampled_tokens, self.mask_token_id)
        )
        return result

    def load_model(self, target_model: nn.Module) -> None:
        """Load the DFlash draft model.

        Args:
            target_model: The target model (used for embedding sharing).
        """
        from vllm.compilation.backends import set_model_tag

        logger.info(
            "Loading DFlash draft model: %s",
            self.speculative_config.draft_model_config.model,
        )

        with set_model_tag("dflash_draft"):
            self.model = get_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
            )

        # Detect mask token ID
        self.mask_token_id = self._detect_mask_token_id()
        logger.info("DFlash mask token ID: %d", self.mask_token_id)

    @torch.inference_mode()
    def dummy_run(self, num_tokens: int) -> None:
        """Warmup forward pass for the draft model."""
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        input_ids = torch.zeros(
            (num_tokens,), dtype=torch.long, device=self.device
        )
        with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):
            self.model(
                input_ids=input_ids,
                positions=None,
                hidden_states=hidden_states[:num_tokens],
            )
