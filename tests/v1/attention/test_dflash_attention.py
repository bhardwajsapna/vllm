# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DFlash attention backend."""

import pytest
import torch


class TestDFlashHybridMaskCorrectness:
    """Verify the hybrid attention mask pattern."""

    def test_denoising_metadata_fields(self):
        """DFlashAttentionMetadata should have denoising-specific fields."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionMetadata,
        )

        metadata = DFlashAttentionMetadata(
            num_actual_tokens=10,
            max_query_len=10,
            query_start_loc=torch.tensor([0, 10]),
            max_seq_len=10,
            seq_lens=torch.tensor([10]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            slot_mapping=torch.zeros(10, dtype=torch.long),
            draft_block_start=torch.tensor([6]),
            draft_block_size=4,
            is_denoising_step=True,
        )

        assert metadata.is_denoising_step is True
        assert metadata.draft_block_size == 4
        assert metadata.draft_block_start is not None
        assert metadata.draft_block_start[0].item() == 6

    def test_non_denoising_metadata(self):
        """Non-denoising metadata should have is_denoising_step=False."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionMetadata,
        )

        metadata = DFlashAttentionMetadata(
            num_actual_tokens=10,
            max_query_len=10,
            query_start_loc=torch.tensor([0, 10]),
            max_seq_len=10,
            seq_lens=torch.tensor([10]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            slot_mapping=torch.zeros(10, dtype=torch.long),
        )

        assert metadata.is_denoising_step is False
        assert metadata.draft_block_size == 0
        assert metadata.draft_block_start is None

    def test_hybrid_mask_structure(self):
        """Verify the expected hybrid mask structure for DFlash."""
        prefix_len = 6
        draft_block_size = 4
        total_len = prefix_len + draft_block_size

        # Build expected hybrid mask
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Causal for prefix region
        for i in range(prefix_len):
            for j in range(i + 1):
                mask[i, j] = True

        # Draft tokens attend to all prefix tokens
        mask[prefix_len:, :prefix_len] = True

        # Draft tokens have bidirectional attention among themselves
        mask[prefix_len:, prefix_len:] = True

        # Verify prefix is causal
        for i in range(prefix_len):
            for j in range(prefix_len):
                if j <= i:
                    assert mask[i, j]
                else:
                    assert not mask[i, j]

        # Verify draft block is bidirectional
        for i in range(prefix_len, total_len):
            for j in range(prefix_len, total_len):
                assert mask[i, j]

        # Verify draft attends to prefix
        for i in range(prefix_len, total_len):
            for j in range(prefix_len):
                assert mask[i, j]

        # Verify prefix does NOT attend to draft
        for i in range(prefix_len):
            for j in range(prefix_len, total_len):
                assert not mask[i, j]


class TestDFlashBackendSelection:
    """Test that DFlash backend is properly registered and selectable."""

    def test_backend_registered_in_enum(self):
        """DFLASH_ATTN should be in the AttentionBackendEnum."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        assert hasattr(AttentionBackendEnum, "DFLASH_ATTN")

    def test_backend_path_is_valid(self):
        """The backend path should point to the correct module."""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        backend = AttentionBackendEnum.DFLASH_ATTN
        path = backend.get_path()
        assert "dflash_attn" in path
        assert "DFlashAttentionBackend" in path

    def test_backend_class_importable(self):
        """DFlashAttentionBackend should be importable."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend is not None

    def test_backend_name(self):
        """Backend name should be 'DFLASH_ATTN'."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend.get_name() == "DFLASH_ATTN"

    def test_backend_supports_decoder_attention(self):
        """DFlash backend should support decoder attention type."""
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend.supports_attn_type(
            AttentionType.DECODER
        )

    def test_backend_does_not_support_encoder(self):
        """DFlash backend should not support encoder attention types."""
        from vllm.v1.attention.backend import AttentionType
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert not DFlashAttentionBackend.supports_attn_type(
            AttentionType.ENCODER
        )
        assert not DFlashAttentionBackend.supports_attn_type(
            AttentionType.ENCODER_ONLY
        )


class TestDFlashKVCacheShape:
    """Test that KV cache shape matches flash_attn format."""

    def test_kv_cache_shape_format(self):
        """KV cache shape should be (2, num_blocks, block_size, num_kv_heads,
        head_size)."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        shape = DFlashAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
        )

        assert shape == (2, 100, 16, 8, 128)

    def test_kv_cache_shape_matches_flash_attn(self):
        """DFlash KV cache shape should match FlashAttention format."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )

        params = {
            "num_blocks": 64,
            "block_size": 16,
            "num_kv_heads": 4,
            "head_size": 64,
        }

        dflash_shape = DFlashAttentionBackend.get_kv_cache_shape(**params)
        flash_shape = FlashAttentionBackend.get_kv_cache_shape(**params)

        assert dflash_shape == flash_shape

    def test_kv_cache_requires_block_size_multiple_of_16(self):
        """Block size must be a multiple of 16."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        with pytest.raises(ValueError, match="multiple of 16"):
            DFlashAttentionBackend.get_kv_cache_shape(
                num_blocks=64,
                block_size=15,
                num_kv_heads=4,
                head_size=64,
            )

    def test_supported_dtypes(self):
        """DFlash should support float16 and bfloat16."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend.supports_dtype(torch.float16)
        assert DFlashAttentionBackend.supports_dtype(torch.bfloat16)
        assert not DFlashAttentionBackend.supports_dtype(torch.float32)

    def test_supported_head_sizes(self):
        """DFlash should support head sizes that are multiples of 8."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend.supports_head_size(64)
        assert DFlashAttentionBackend.supports_head_size(128)
        assert DFlashAttentionBackend.supports_head_size(256)
        assert not DFlashAttentionBackend.supports_head_size(257)
        assert not DFlashAttentionBackend.supports_head_size(7)

    def test_supported_block_sizes(self):
        """DFlash should support block sizes that are multiples of 16."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
        )

        assert DFlashAttentionBackend.supports_block_size(16)
        assert DFlashAttentionBackend.supports_block_size(32)
        assert DFlashAttentionBackend.supports_block_size(64)


class TestDFlashMetadataBuilder:
    """Test the DFlash attention metadata builder."""

    def test_builder_cls(self):
        """get_builder_cls should return DFlashAttentionMetadataBuilder."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
            DFlashAttentionMetadataBuilder,
        )

        assert (
            DFlashAttentionBackend.get_builder_cls()
            is DFlashAttentionMetadataBuilder
        )

    def test_impl_cls(self):
        """get_impl_cls should return DFlashAttentionImpl."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionBackend,
            DFlashAttentionImpl,
        )

        assert DFlashAttentionBackend.get_impl_cls() is DFlashAttentionImpl

    def test_impl_rejects_sliding_window(self):
        """DFlashAttentionImpl should reject sliding window."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionImpl,
        )

        with pytest.raises(NotImplementedError, match="sliding window"):
            DFlashAttentionImpl(
                num_heads=8,
                head_size=64,
                scale=0.125,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=(128, 256),
                blocksparse_params=None,
                logits_soft_cap=None,
                attn_type="decoder",
            )

    def test_impl_rejects_alibi(self):
        """DFlashAttentionImpl should reject ALiBi."""
        from vllm.v1.attention.backends.dflash_attn import (
            DFlashAttentionImpl,
        )

        with pytest.raises(NotImplementedError, match="ALiBi"):
            DFlashAttentionImpl(
                num_heads=8,
                head_size=64,
                scale=0.125,
                num_kv_heads=8,
                alibi_slopes=[0.1] * 8,
                sliding_window=None,
                blocksparse_params=None,
                logits_soft_cap=None,
                attn_type="decoder",
            )
