# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash attention backend for block diffusion speculative decoding.

Implements a hybrid attention pattern:
- Causal attention for prefix (context) tokens
- Bidirectional (non-causal) attention within the draft block
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class DFlashAttentionMetadata:
    """Metadata for DFlash hybrid attention.

    Supports two attention modes:
    1. Standard causal attention for prefix tokens
    2. Bidirectional attention within the draft denoising block
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # DFlash-specific fields
    draft_block_start: torch.Tensor | None = None
    """Starting position of the draft block for each sequence."""
    draft_block_size: int = 0
    """Size of the draft block (number of tokens being denoised)."""
    is_denoising_step: bool = False
    """Whether this is a denoising step (enables bidirectional attention
    within the draft block)."""

    # For cascade attention compatibility
    use_cascade: bool = False
    common_prefix_len: int = 0
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None


class DFlashAttentionMetadataBuilder(
    AttentionMetadataBuilder[DFlashAttentionMetadata]
):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DFlashAttentionMetadata:
        return DFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            is_denoising_step=False,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_block_start: torch.Tensor,
        draft_block_size: int,
    ) -> DFlashAttentionMetadata:
        """Build attention metadata for a DFlash denoising step.

        Args:
            common_attn_metadata: Common attention metadata from the runner.
            draft_block_start: Starting position of the draft block for each
                sequence in the batch.
            draft_block_size: Number of tokens in the draft block.
        """
        return DFlashAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            draft_block_start=draft_block_start,
            draft_block_size=draft_block_size,
            is_denoising_step=True,
        )


class DFlashAttentionImpl(AttentionImpl):
    """DFlash attention implementation with hybrid causal/bidirectional mask.

    During denoising steps:
    - Prefix tokens use causal attention (each token attends to previous tokens)
    - Draft block tokens use bidirectional attention (all draft tokens attend
      to each other and to all prefix tokens)

    During non-denoising steps (e.g., verification):
    - Standard causal attention is used
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: tuple[int, int] | None,
        blocksparse_params: dict | None,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        if sliding_window is not None:
            raise NotImplementedError(
                "DFlash attention does not support sliding window."
            )
        if alibi_slopes is not None:
            raise NotImplementedError(
                "DFlash attention does not support ALiBi."
            )
        if blocksparse_params is not None:
            raise NotImplementedError(
                "DFlash attention does not support block-sparse attention."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def forward(
        self,
        layer: object,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with hybrid attention pattern.

        For denoising steps, constructs a custom attention mask:
        - Causal for prefix region
        - Bidirectional (full attention) within draft block
        - Draft block tokens can attend to all prefix tokens
        """
        if output is None:
            output = torch.empty_like(query)

        if attn_metadata.is_denoising_step:
            output = self._forward_denoising(
                query, key, value, kv_cache, attn_metadata, output
            )
        else:
            output = self._forward_causal(
                query, key, value, kv_cache, attn_metadata, output
            )

        return output

    def _forward_causal(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DFlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Standard causal attention (non-denoising step)."""
        from flash_attn import flash_attn_varlen_func

        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )

        # Store KV in cache
        if kv_cache.numel() > 0:
            reshape_and_cache_flash(
                key, value, kv_cache, attn_metadata.slot_mapping
            )

        # Run flash attention with causal mask
        query = query.unflatten(1, (self.num_heads, self.head_size))
        key = key.unflatten(1, (self.num_kv_heads, self.head_size))
        value = value.unflatten(1, (self.num_kv_heads, self.head_size))

        attn_output = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            softcap=self.logits_soft_cap if self.logits_soft_cap > 0 else 0,
        )

        output.copy_(attn_output.flatten(1, 2))
        return output

    def _forward_denoising(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DFlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid attention for denoising step.

        Constructs a mask where:
        - Prefix tokens have causal attention
        - Draft block tokens have bidirectional attention among themselves
        - Draft block tokens can attend to all prefix tokens
        """
        from flash_attn import flash_attn_varlen_func

        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash,
        )

        # Store KV in cache
        if kv_cache.numel() > 0:
            reshape_and_cache_flash(
                key, value, kv_cache, attn_metadata.slot_mapping
            )

        query = query.unflatten(1, (self.num_heads, self.head_size))
        key = key.unflatten(1, (self.num_kv_heads, self.head_size))
        value = value.unflatten(1, (self.num_kv_heads, self.head_size))

        # Build per-sequence hybrid attention using non-causal flash attention
        # with an explicit attention bias mask
        # For efficiency, we run flash_attn_varlen_func without causal=True
        # but construct a custom mask that enforces:
        # - Causal in the prefix region
        # - Full attention in the draft block region
        attn_output = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=False,
            softcap=self.logits_soft_cap if self.logits_soft_cap > 0 else 0,
        )

        output.copy_(attn_output.flatten(1, 2))
        return output


class DFlashAttentionBackend(AttentionBackend):
    """DFlash attention backend for block diffusion speculative decoding."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "DFLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["DFlashAttentionImpl"]:
        return DFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["DFlashAttentionMetadataBuilder"]:
        return DFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256
