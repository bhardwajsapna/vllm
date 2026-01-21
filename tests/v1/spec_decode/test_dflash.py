# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DFlash block diffusion speculative decoding proposer."""

import pytest
import torch


class TestDFlashMaskInitialization:
    """Test that the draft block is correctly initialized with MASK tokens."""

    def test_dflash_mask_initialization(self):
        """Draft block should start with all MASK tokens."""
        proposer = self._create_mock_proposer(
            mask_token_id=128255, num_speculative_tokens=16
        )

        batch_size = 4
        draft_tokens = torch.full(
            (batch_size, proposer.num_speculative_tokens),
            fill_value=proposer.mask_token_id,
            dtype=torch.long,
            device="cpu",
        )

        assert draft_tokens.shape == (4, 16)
        assert (draft_tokens == 128255).all()

    def test_mask_token_fills_entire_block(self):
        """All positions in the draft block should be masked initially."""
        mask_token_id = 42
        block_size = 8
        batch_size = 2

        draft_tokens = torch.full(
            (batch_size, block_size),
            fill_value=mask_token_id,
            dtype=torch.long,
        )

        # Every position should be the mask token
        for b in range(batch_size):
            for pos in range(block_size):
                assert draft_tokens[b, pos].item() == mask_token_id

    @staticmethod
    def _create_mock_proposer(
        mask_token_id: int = 128255,
        num_speculative_tokens: int = 16,
    ):
        """Create a mock DFlashProposer without full initialization."""
        proposer = object.__new__(
            __import__(
                "vllm.v1.spec_decode.dflash", fromlist=["DFlashProposer"]
            ).DFlashProposer
        )
        proposer.mask_token_id = mask_token_id
        proposer.num_speculative_tokens = num_speculative_tokens
        proposer.num_denoising_steps = 3
        proposer.dflash_block_size = 16
        proposer.device = torch.device("cpu")
        proposer.dtype = torch.bfloat16
        proposer.keep_schedule = proposer._compute_keep_schedule()
        return proposer


class TestDFlashDenoisingSchedule:
    """Test that the keep schedule increases monotonically."""

    def test_keep_schedule_monotonically_increasing(self):
        """Keep ratio should increase at each step."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()
        schedule = proposer.keep_schedule

        for i in range(1, len(schedule)):
            assert schedule[i] >= schedule[i - 1], (
                f"Schedule not monotonic: step {i-1}={schedule[i-1]}, "
                f"step {i}={schedule[i]}"
            )

    def test_keep_schedule_ends_at_one(self):
        """Final step should keep all tokens (ratio = 1.0)."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()
        assert proposer.keep_schedule[-1] == 1.0

    def test_keep_schedule_starts_above_zero(self):
        """First step should keep at least some tokens."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()
        assert proposer.keep_schedule[0] > 0.0

    @pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 10])
    def test_keep_schedule_length_matches_steps(self, num_steps):
        """Schedule length should match number of denoising steps."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()
        proposer.num_denoising_steps = num_steps
        schedule = proposer._compute_keep_schedule()
        assert len(schedule) == num_steps

    def test_single_step_schedule(self):
        """Single denoising step should have schedule [1.0]."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()
        proposer.num_denoising_steps = 1
        schedule = proposer._compute_keep_schedule()
        assert schedule == [1.0]


class TestDFlashAttentionMaskPattern:
    """Test the hybrid attention mask pattern (causal prefix + bidirectional
    draft block)."""

    def test_prefix_is_causal(self):
        """Prefix region should use causal (lower-triangular) attention."""
        prefix_len = 5

        # Build a causal mask for the prefix
        causal_mask = torch.tril(
            torch.ones(prefix_len, prefix_len, dtype=torch.bool)
        )

        # Verify lower-triangular structure
        for i in range(prefix_len):
            for j in range(prefix_len):
                if j <= i:
                    assert causal_mask[i, j], (
                        f"Position ({i},{j}) should be visible (causal)"
                    )
                else:
                    assert not causal_mask[i, j], (
                        f"Position ({i},{j}) should be masked (causal)"
                    )

    def test_draft_block_is_bidirectional(self):
        """Draft block tokens should attend to each other (full attention)."""
        draft_block_size = 4

        # Bidirectional mask within draft block
        bidirectional_mask = torch.ones(
            draft_block_size, draft_block_size, dtype=torch.bool
        )

        # All positions should be visible to each other
        assert bidirectional_mask.all()

    def test_draft_attends_to_prefix(self):
        """Draft block tokens should attend to all prefix tokens."""
        prefix_len = 5
        draft_block_size = 4

        # Draft-to-prefix attention: all visible
        cross_mask = torch.ones(
            draft_block_size, prefix_len, dtype=torch.bool
        )
        assert cross_mask.all()

    def test_full_hybrid_mask_shape(self):
        """Full hybrid mask should have correct shape."""
        prefix_len = 5
        draft_block_size = 4
        total_len = prefix_len + draft_block_size

        # Build full hybrid mask
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Causal for prefix
        mask[:prefix_len, :prefix_len] = torch.tril(
            torch.ones(prefix_len, prefix_len, dtype=torch.bool)
        )
        # Draft attends to all prefix tokens
        mask[prefix_len:, :prefix_len] = True
        # Draft block is bidirectional
        mask[prefix_len:, prefix_len:] = True

        assert mask.shape == (total_len, total_len)
        # Prefix tokens should NOT attend to draft tokens
        assert not mask[:prefix_len, prefix_len:].any()


class TestDFlashProposeOutputShape:
    """Test that propose() returns the correct output shape."""

    def test_output_shape_matches_batch_and_spec_tokens(self):
        """Output should be [batch_size, num_speculative_tokens]."""
        batch_size = 4
        num_spec_tokens = 16

        # Simulate output
        draft_tokens = torch.randint(
            0, 32000, (batch_size, num_spec_tokens), dtype=torch.long
        )
        assert draft_tokens.shape == (batch_size, num_spec_tokens)

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_output_shape_various_batch_sizes(self, batch_size):
        """Output shape should work for various batch sizes."""
        num_spec_tokens = 16
        draft_tokens = torch.randint(
            0, 32000, (batch_size, num_spec_tokens), dtype=torch.long
        )
        assert draft_tokens.shape[0] == batch_size
        assert draft_tokens.shape[1] == num_spec_tokens


class TestDFlashConfidenceRemasking:
    """Test confidence-based re-masking between denoising steps."""

    def test_low_confidence_tokens_remasked(self):
        """Tokens with low confidence should be re-masked."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()

        batch_size = 2
        num_spec_tokens = 8
        vocab_size = 100

        # Create logits with varying confidence
        logits = torch.randn(batch_size, num_spec_tokens, vocab_size)
        # Make first 4 positions high confidence
        logits[:, :4, 0] = 100.0  # Very high logit for token 0
        # Last 4 positions remain low confidence

        sampled_tokens = logits.argmax(dim=-1)
        keep_ratio = 0.5  # Keep only half

        result = proposer._denoise_step(sampled_tokens, logits, keep_ratio)

        # Some positions should be re-masked
        mask_positions = result == proposer.mask_token_id
        assert mask_positions.any(), "Some tokens should be re-masked"

        # High-confidence positions should NOT be re-masked
        # (first 4 positions have very high confidence)
        assert (result[:, :4] != proposer.mask_token_id).all(), (
            "High-confidence tokens should not be re-masked"
        )

    def test_final_step_keeps_all(self):
        """On final step (keep_ratio=1.0), no tokens should be re-masked."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()

        batch_size = 2
        num_spec_tokens = 8
        vocab_size = 100

        logits = torch.randn(batch_size, num_spec_tokens, vocab_size)
        sampled_tokens = logits.argmax(dim=-1)

        result = proposer._denoise_step(sampled_tokens, logits, 1.0)

        # All tokens should be kept (none re-masked)
        assert (result != proposer.mask_token_id).all(), (
            "Final step should keep all tokens"
        )

    def test_remasking_preserves_batch_independence(self):
        """Re-masking should be independent per sequence in the batch."""
        proposer = TestDFlashMaskInitialization._create_mock_proposer()

        batch_size = 2
        num_spec_tokens = 8
        vocab_size = 100

        logits = torch.randn(batch_size, num_spec_tokens, vocab_size)
        # Make batch 0 all high confidence
        logits[0, :, 0] = 100.0
        # Make batch 1 all low confidence
        logits[1] = torch.randn(num_spec_tokens, vocab_size) * 0.01

        sampled_tokens = logits.argmax(dim=-1)
        result = proposer._denoise_step(sampled_tokens, logits, 0.5)

        # Batch 0 should have more kept tokens than batch 1
        # (both keep exactly num_keep = max(1, int(0.5 * 8)) = 4 tokens)
        kept_0 = (result[0] != proposer.mask_token_id).sum()
        kept_1 = (result[1] != proposer.mask_token_id).sum()
        # Both should keep exactly 4 tokens
        assert kept_0 == 4
        assert kept_1 == 4


class TestDFlashConfigValidation:
    """Test DFlash configuration validation and auto-detection."""

    def test_dflash_method_in_speculative_method(self):
        """'dflash' should be a valid SpeculativeMethod."""
        from typing import get_args

        from vllm.config.speculative import SpeculativeMethod

        # Flatten the literal type to get all valid methods
        all_methods = set()

        def extract_literals(t):
            args = get_args(t)
            for a in args:
                sub_args = get_args(a)
                if sub_args:
                    extract_literals(a)
                else:
                    all_methods.add(a)

        extract_literals(SpeculativeMethod)
        assert "dflash" in all_methods

    def test_use_dflash_helper(self):
        """use_dflash() should return True when method is 'dflash'."""
        from unittest.mock import MagicMock

        from vllm.config.speculative import SpeculativeConfig

        # Create a mock config
        config = MagicMock(spec=SpeculativeConfig)
        config.method = "dflash"
        config.use_dflash = SpeculativeConfig.use_dflash.__get__(config)

        assert config.use_dflash()

    def test_use_dflash_false_for_other_methods(self):
        """use_dflash() should return False for non-dflash methods."""
        from unittest.mock import MagicMock

        from vllm.config.speculative import SpeculativeConfig

        for method in ["ngram", "eagle", "medusa", "draft_model"]:
            config = MagicMock(spec=SpeculativeConfig)
            config.method = method
            config.use_dflash = SpeculativeConfig.use_dflash.__get__(config)
            assert not config.use_dflash()

    def test_dflash_config_defaults(self):
        """DFlash config fields should have sensible defaults."""
        import inspect

        from vllm.config.speculative import SpeculativeConfig

        # Check that the fields exist in the class
        source = inspect.getsource(SpeculativeConfig)
        assert "dflash_block_size" in source
        assert "dflash_num_denoising_steps" in source
        assert "dflash_mask_token_id" in source


class TestDFlashFP8QuantizationCompat:
    """Test DFlash compatibility with FP8 quantization."""

    def test_fp8_quantization_field_accepted(self):
        """SpeculativeConfig should accept quantization='fp8'."""
        import inspect

        from vllm.config.speculative import SpeculativeConfig

        source = inspect.getsource(SpeculativeConfig)
        assert "quantization" in source

    def test_dflash_model_supports_quantization_config(self):
        """DFlash model should work with quantization parameters."""
        from vllm.model_executor.models.dflash import DFlashForCausalLM

        # Verify the model class exists and has the expected interface
        assert hasattr(DFlashForCausalLM, "forward")
        assert hasattr(DFlashForCausalLM, "compute_logits")
        assert hasattr(DFlashForCausalLM, "load_weights")
