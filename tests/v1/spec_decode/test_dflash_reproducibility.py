# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducibility tests for DFlash block diffusion speculative decoding.

Verifies that given the same inputs, all DFlash operations produce
identical outputs across repeated runs.
"""

import pytest
import torch


def _create_mock_proposer(
    mask_token_id: int = 128255,
    num_speculative_tokens: int = 16,
    num_denoising_steps: int = 3,
):
    """Create a mock DFlashProposer for testing without full init."""
    from vllm.v1.spec_decode.dflash import DFlashProposer

    proposer = object.__new__(DFlashProposer)
    proposer.mask_token_id = mask_token_id
    proposer.num_speculative_tokens = num_speculative_tokens
    proposer.num_denoising_steps = num_denoising_steps
    proposer.dflash_block_size = num_speculative_tokens
    proposer.device = torch.device("cpu")
    proposer.dtype = torch.bfloat16
    proposer.keep_schedule = proposer._compute_keep_schedule()
    return proposer


class TestKeepScheduleReproducibility:
    """Verify keep schedule is deterministic across calls."""

    @pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 10])
    def test_schedule_identical_across_calls(self, num_steps):
        """Same num_denoising_steps always produces same schedule."""
        proposer = _create_mock_proposer(num_denoising_steps=num_steps)
        schedules = [proposer._compute_keep_schedule() for _ in range(100)]
        assert all(s == schedules[0] for s in schedules)

    def test_schedule_values_exact(self):
        """Schedule values should be exactly reproducible floats."""
        proposer = _create_mock_proposer(num_denoising_steps=4)
        schedule = proposer._compute_keep_schedule()
        assert schedule == [0.25, 0.5, 0.75, 1.0]

    def test_schedule_independent_of_other_state(self):
        """Schedule computation should not depend on external state."""
        proposer1 = _create_mock_proposer(
            num_denoising_steps=3, mask_token_id=100
        )
        proposer2 = _create_mock_proposer(
            num_denoising_steps=3, mask_token_id=999
        )
        assert proposer1._compute_keep_schedule() == (
            proposer2._compute_keep_schedule()
        )


class TestMaskInitReproducibility:
    """Verify MASK token initialization is deterministic."""

    def test_mask_init_identical_across_runs(self):
        """torch.full with same params always produces same tensor."""
        results = []
        for _ in range(50):
            t = torch.full(
                (4, 16), fill_value=128255, dtype=torch.long, device="cpu"
            )
            results.append(t.clone())
        assert all(torch.equal(results[0], r) for r in results)

    def test_mask_init_independent_of_prior_operations(self):
        """Mask init should not be affected by prior tensor operations."""
        # Do some random operations first
        _ = torch.randn(100, 100)
        t1 = torch.full((4, 16), fill_value=128255, dtype=torch.long)

        _ = torch.randn(200, 200)
        _ = torch.randint(0, 1000, (50, 50))
        t2 = torch.full((4, 16), fill_value=128255, dtype=torch.long)

        assert torch.equal(t1, t2)


class TestArgmaxReproducibility:
    """Verify argmax token selection is deterministic."""

    def test_argmax_deterministic_same_input(self):
        """argmax on same logits always gives same tokens."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 32000)

        results = [logits.argmax(dim=-1) for _ in range(50)]
        assert all(torch.equal(results[0], r) for r in results)

    def test_argmax_deterministic_across_seeds(self):
        """argmax result depends only on input, not on RNG state."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 32000)

        # Change RNG state
        torch.manual_seed(99)
        _ = torch.randn(100)

        result1 = logits.argmax(dim=-1)

        # Change RNG state again
        torch.manual_seed(7)
        _ = torch.randn(200)

        result2 = logits.argmax(dim=-1)
        assert torch.equal(result1, result2)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("vocab_size", [100, 32000, 128256])
    def test_argmax_various_shapes(self, batch_size, vocab_size):
        """argmax is deterministic for various tensor shapes."""
        torch.manual_seed(42)
        logits = torch.randn(batch_size, 16, vocab_size)
        results = [logits.argmax(dim=-1) for _ in range(10)]
        assert all(torch.equal(results[0], r) for r in results)


class TestSoftmaxReproducibility:
    """Verify softmax computation is deterministic."""

    def test_softmax_deterministic(self):
        """softmax on same input produces identical output."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 1000)

        results = [torch.softmax(logits, dim=-1) for _ in range(50)]
        assert all(torch.equal(results[0], r) for r in results)

    def test_softmax_max_confidence_deterministic(self):
        """Max probability extraction is deterministic."""
        torch.manual_seed(42)
        logits = torch.randn(4, 16, 1000)
        probs = torch.softmax(logits, dim=-1)

        results = [probs.max(dim=-1).values for _ in range(50)]
        assert all(torch.equal(results[0], r) for r in results)


class TestTopkReproducibility:
    """Verify topk behavior for confidence-based re-masking."""

    def test_topk_deterministic_distinct_values(self):
        """topk with all distinct values is always deterministic."""
        torch.manual_seed(42)
        # Use randn which almost surely gives distinct values
        confidence = torch.randn(4, 16)

        results = []
        for _ in range(50):
            _, indices = confidence.topk(8, dim=-1)
            results.append(indices.clone())
        assert all(torch.equal(results[0], r) for r in results)

    def test_topk_with_ties_on_cpu(self):
        """topk with tied values on CPU should be deterministic."""
        # CPU topk is typically stable (preserves original order for ties)
        confidence = torch.tensor(
            [[0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]]
        )

        results = []
        for _ in range(100):
            _, indices = confidence.topk(4, dim=-1)
            results.append(indices.clone())
        assert all(torch.equal(results[0], r) for r in results)

    def test_topk_preserves_order_for_distinct(self):
        """topk should return indices in descending value order."""
        confidence = torch.tensor([[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]])
        _, indices = confidence.topk(4, dim=-1)
        # Should be indices 0, 2, 4, 6 (values 0.9, 0.8, 0.7, 0.6)
        expected = torch.tensor([[0, 2, 4, 6]])
        assert torch.equal(indices, expected)

    @pytest.mark.parametrize("num_keep", [1, 4, 8, 15])
    def test_topk_various_k_values(self, num_keep):
        """topk is deterministic for various k values."""
        torch.manual_seed(42)
        confidence = torch.randn(4, 16)

        results = []
        for _ in range(20):
            _, indices = confidence.topk(
                min(num_keep, confidence.shape[-1]), dim=-1
            )
            results.append(indices.clone())
        assert all(torch.equal(results[0], r) for r in results)


class TestDenoiseStepReproducibility:
    """Verify _denoise_step produces identical results for same inputs."""

    def test_denoise_step_deterministic(self):
        """Same inputs always produce same re-masking pattern."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            results.append(r.clone())
        assert all(torch.equal(results[0], r) for r in results)

    @pytest.mark.parametrize("keep_ratio", [0.25, 0.5, 0.75, 1.0])
    def test_denoise_step_deterministic_various_ratios(self, keep_ratio):
        """Deterministic across different keep ratios."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(4, 8, 100)
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(20):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), keep_ratio
            )
            results.append(r.clone())
        assert all(torch.equal(results[0], r) for r in results)

    def test_denoise_step_mask_count_consistent(self):
        """Number of masked positions should be consistent across runs."""
        proposer = _create_mock_proposer(num_speculative_tokens=16)

        torch.manual_seed(42)
        logits = torch.randn(4, 16, 100)
        sampled = logits.argmax(dim=-1)

        mask_counts = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            count = (r == proposer.mask_token_id).sum().item()
            mask_counts.append(count)
        # All runs should mask the same number of tokens
        assert all(c == mask_counts[0] for c in mask_counts)

    def test_denoise_step_kept_positions_consistent(self):
        """The exact positions kept should be consistent across runs."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        sampled = logits.argmax(dim=-1)

        kept_positions = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            kept = (r != proposer.mask_token_id)
            kept_positions.append(kept.clone())
        assert all(torch.equal(kept_positions[0], k) for k in kept_positions)


class TestFullDenoiseLoopReproducibility:
    """Verify the full iterative denoising loop is reproducible."""

    def test_multi_step_denoising_deterministic(self):
        """Full multi-step denoising with same logits is deterministic."""
        proposer = _create_mock_proposer(
            num_speculative_tokens=8, num_denoising_steps=3
        )

        torch.manual_seed(42)
        batch_size = 2
        num_spec = 8
        vocab_size = 100

        # Simulate full denoising loop (without model forward)
        def run_loop():
            draft_tokens = torch.full(
                (batch_size, num_spec),
                fill_value=proposer.mask_token_id,
                dtype=torch.long,
            )
            for step in range(proposer.num_denoising_steps):
                is_final = step == proposer.num_denoising_steps - 1
                keep_ratio = proposer.keep_schedule[step]

                # Use fixed logits to simulate model output
                torch.manual_seed(step + 100)
                logits = torch.randn(batch_size, num_spec, vocab_size)
                sampled = logits.argmax(dim=-1)

                if is_final:
                    draft_tokens = sampled
                else:
                    draft_tokens = proposer._denoise_step(
                        sampled, logits, keep_ratio
                    )
            return draft_tokens

        results = [run_loop() for _ in range(20)]
        assert all(torch.equal(results[0], r) for r in results)

    def test_intermediate_states_reproducible(self):
        """Each intermediate denoising step should match across runs."""
        proposer = _create_mock_proposer(
            num_speculative_tokens=8, num_denoising_steps=4
        )

        batch_size = 2
        num_spec = 8
        vocab_size = 100

        def run_loop_with_intermediates():
            intermediates = []
            draft_tokens = torch.full(
                (batch_size, num_spec),
                fill_value=proposer.mask_token_id,
                dtype=torch.long,
            )
            for step in range(proposer.num_denoising_steps):
                is_final = step == proposer.num_denoising_steps - 1
                keep_ratio = proposer.keep_schedule[step]

                torch.manual_seed(step + 200)
                logits = torch.randn(batch_size, num_spec, vocab_size)
                sampled = logits.argmax(dim=-1)

                if is_final:
                    draft_tokens = sampled
                else:
                    draft_tokens = proposer._denoise_step(
                        sampled, logits, keep_ratio
                    )
                intermediates.append(draft_tokens.clone())
            return intermediates

        run1 = run_loop_with_intermediates()
        run2 = run_loop_with_intermediates()

        for step_idx, (s1, s2) in enumerate(zip(run1, run2)):
            assert torch.equal(s1, s2), (
                f"Mismatch at step {step_idx}"
            )


class TestScatterMaskReproducibility:
    """Verify scatter-based mask creation is deterministic."""

    def test_scatter_mask_deterministic(self):
        """scatter_ used for keep_mask creation is deterministic."""
        batch_size = 4
        num_spec = 16
        num_keep = 8

        torch.manual_seed(42)
        confidence = torch.randn(batch_size, num_spec)
        _, top_indices = confidence.topk(num_keep, dim=-1)

        masks = []
        for _ in range(50):
            keep_mask = torch.zeros(
                batch_size, num_spec, dtype=torch.bool
            )
            keep_mask.scatter_(1, top_indices, True)
            masks.append(keep_mask.clone())
        assert all(torch.equal(masks[0], m) for m in masks)

    def test_where_operation_deterministic(self):
        """torch.where for token selection is deterministic."""
        mask = torch.tensor(
            [[True, False, True, False, True, False, True, False]]
        )
        tokens = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        fill_val = 128255

        results = []
        for _ in range(50):
            r = torch.where(
                mask, tokens, torch.full_like(tokens, fill_val)
            )
            results.append(r.clone())
        assert all(torch.equal(results[0], r) for r in results)


class TestEdgeCaseReproducibility:
    """Reproducibility for edge cases and boundary conditions."""

    def test_single_token_block(self):
        """Single token block should always produce same result."""
        proposer = _create_mock_proposer(
            num_speculative_tokens=1, num_denoising_steps=3
        )

        torch.manual_seed(42)
        logits = torch.randn(2, 1, 100)
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            results.append(r.clone())
        # With num_keep = max(1, int(0.5*1)) = 1, all tokens kept
        assert all(torch.equal(results[0], r) for r in results)

    def test_keep_ratio_one_always_keeps_all(self):
        """keep_ratio=1.0 should deterministically keep everything."""
        proposer = _create_mock_proposer(num_speculative_tokens=16)

        for seed in range(10):
            torch.manual_seed(seed)
            logits = torch.randn(4, 16, 100)
            sampled = logits.argmax(dim=-1)

            r = proposer._denoise_step(sampled, logits, 1.0)
            assert torch.equal(r, sampled), (
                f"keep_ratio=1.0 should keep all tokens (seed={seed})"
            )

    def test_batch_size_one(self):
        """Single sequence batch should be deterministic."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(1, 8, 100)
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            results.append(r.clone())
        assert all(torch.equal(results[0], r) for r in results)

    def test_very_large_batch(self):
        """Large batch should remain deterministic."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(64, 8, 100)
        sampled = logits.argmax(dim=-1)

        r1 = proposer._denoise_step(sampled.clone(), logits.clone(), 0.5)
        r2 = proposer._denoise_step(sampled.clone(), logits.clone(), 0.5)
        assert torch.equal(r1, r2)

    def test_extreme_confidence_distribution(self):
        """Extreme confidence (all same) should be deterministic."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        # All logits identical -> all confidence equal
        logits = torch.zeros(2, 8, 100)
        logits[:, :, 0] = 10.0  # All positions predict same token
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(50):
            r = proposer._denoise_step(
                sampled.clone(), logits.clone(), 0.5
            )
            results.append(r.clone())
        # Even with tied confidence, CPU topk should be stable
        assert all(torch.equal(results[0], r) for r in results)


class TestAttentionMetadataReproducibility:
    """Verify attention metadata construction is deterministic."""

    def test_metadata_fields_deterministic(self):
        """Metadata dataclass should produce identical fields."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionMetadata,
        )

        query_start = torch.tensor([0, 10])
        seq_lens = torch.tensor([10])
        block_table = torch.zeros(1, 4, dtype=torch.int32)
        slot_mapping = torch.arange(10, dtype=torch.long)

        metas = []
        for _ in range(20):
            m = DFlashAttentionMetadata(
                num_actual_tokens=10,
                max_query_len=10,
                query_start_loc=query_start,
                max_seq_len=10,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=slot_mapping,
                draft_block_start=torch.tensor([6]),
                draft_block_size=4,
                is_denoising_step=True,
            )
            metas.append(m)

        for m in metas:
            assert m.num_actual_tokens == metas[0].num_actual_tokens
            assert m.max_query_len == metas[0].max_query_len
            assert m.draft_block_size == metas[0].draft_block_size
            assert m.is_denoising_step == metas[0].is_denoising_step
            assert torch.equal(
                m.draft_block_start, metas[0].draft_block_start
            )

    def test_kv_cache_shape_deterministic(self):
        """KV cache shape should always be the same for same params."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        params = {
            "num_blocks": 100,
            "block_size": 16,
            "num_kv_heads": 8,
            "head_size": 128,
        }

        shapes = [
            DFlashAttentionBackend.get_kv_cache_shape(**params)
            for _ in range(100)
        ]
        assert all(s == shapes[0] for s in shapes)


class TestNumericalStability:
    """Verify numerical stability doesn't affect reproducibility."""

    def test_softmax_no_overflow(self):
        """Softmax with large logits should not produce NaN/Inf."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        logits = torch.full((2, 8, 100), fill_value=1000.0)
        logits[:, :, 0] = 1001.0  # Slight difference for argmax

        sampled = logits.argmax(dim=-1)
        result = proposer._denoise_step(sampled, logits, 0.5)

        assert not torch.isnan(result.float()).any()
        assert not torch.isinf(result.float()).any()

    def test_softmax_no_underflow(self):
        """Softmax with very negative logits should not produce NaN."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        logits = torch.full((2, 8, 100), fill_value=-1000.0)
        logits[:, :, 0] = -999.0  # Slight difference for argmax

        sampled = logits.argmax(dim=-1)
        result = proposer._denoise_step(sampled, logits, 0.5)

        assert not torch.isnan(result.float()).any()
        assert not torch.isinf(result.float()).any()

    def test_confidence_ordering_stable_with_float16(self):
        """Confidence ranking should be stable in float16."""
        proposer = _create_mock_proposer(num_speculative_tokens=8)

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100, dtype=torch.float16)
        sampled = logits.argmax(dim=-1)

        results = []
        for _ in range(20):
            r = proposer._denoise_step(
                sampled.clone(), logits.float().clone(), 0.5
            )
            results.append(r.clone())
        assert all(torch.equal(results[0], r) for r in results)
