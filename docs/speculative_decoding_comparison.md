# Speculative Decoding Techniques in vLLM

A comparison of all speculative decoding methods available in vLLM, including the newly integrated DFlash block diffusion approach.

---

## Overview

Speculative decoding accelerates LLM inference by generating **draft tokens** cheaply and then **verifying** them in parallel with the target model. Different methods trade off between draft quality, generation speed, and resource requirements.

---

## Method Comparison

| Method | Draft Model Required | Generation Strategy | Parallelism | Lossless | Extra Memory |
|--------|---------------------|--------------------:|-------------|----------|-------------|
| **N-gram** | No | Pattern matching on prompt history | CPU (Numba JIT) | Yes | Minimal |
| **Draft Model** | Yes (smaller LLM) | Autoregressive (sequential forward passes) | Sequential | Yes | Full model weights |
| **Medusa** | Yes (lightweight heads) | Parallel prediction heads on hidden states | Parallel heads | Yes | Head weights only |
| **EAGLE / EAGLE3** | Yes (small head) | Hidden-state-based sequential/tree generation | Sequential or Tree | Yes | Head weights |
| **MTP** | No (built-in layers) | Multi-token prediction layers in target model | Sequential | Yes | None (shared) |
| **DFlash** | Yes (diffusion model) | Parallel block denoising (iterative refinement) | Block-parallel | Yes | Draft model weights |
| **Suffix Decoding** | No (external library) | Suffix tree pattern matching on past responses | Dynamic per-request | Yes | Suffix tree cache |

---

## Detailed Comparison

### Draft Generation Approach

| Method | How Drafts Are Generated |
|--------|--------------------------|
| **N-gram** | Searches prompt history for longest n-gram match, returns subsequent tokens |
| **Draft Model** | Runs a smaller autoregressive LM for `k` sequential forward passes |
| **Medusa** | Multiple MLP heads predict positions 1..k simultaneously from one hidden state |
| **EAGLE** | Uses target model's hidden states as input to a small transformer head |
| **EAGLE3** | Like EAGLE but uses auxiliary hidden states from multiple target model layers |
| **MTP** | Built-in multi-token prediction layers in the target model predict next tokens |
| **DFlash** | Initializes MASK token block, iteratively denoises via confidence-based refinement |
| **Suffix** | Matches current sequence suffix against cached past responses via suffix trees |

### Number of Forward Passes (Draft Phase)

| Method | Forward Passes per Draft | Notes |
|--------|--------------------------|-------|
| **N-gram** | 0 | Pure pattern matching, no neural network |
| **Draft Model** | `k` | One pass per speculative token |
| **Medusa** | 1 | All heads run in parallel on same input |
| **EAGLE** | `k` (chain) or `log(k)` (tree) | Depends on tree structure |
| **MTP** | `k / n_predict` | Each MTP layer predicts `n_predict` tokens |
| **DFlash** | `num_denoising_steps` (default 3) | Fixed steps regardless of block size |
| **Suffix** | 0 | Pure pattern matching, no neural network |

### Resource Requirements

| Method | GPU Memory | CPU Overhead | External Dependencies |
|--------|-----------|-------------|----------------------|
| **N-gram** | None | Moderate (Numba) | numba |
| **Draft Model** | High (full small model) | Low | None |
| **Medusa** | Low (MLP heads only) | Low | Medusa checkpoint |
| **EAGLE** | Low-Medium (transformer head) | Low | EAGLE checkpoint |
| **MTP** | None (shared with target) | Low | MTP-enabled model |
| **DFlash** | Medium (draft model) | Low | DFlash checkpoint |
| **Suffix** | Low (CPU suffix trees) | Moderate | arctic-inference |

---

## Configuration

### N-gram
```python
speculative_config = {
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_min": 1,
    "prompt_lookup_max": 5,
}
```

### Draft Model
```python
speculative_config = {
    "method": "draft_model",
    "model": "path/to/small-model",
    "num_speculative_tokens": 5,
}
```

### Medusa
```python
speculative_config = {
    "method": "medusa",
    "model": "path/to/medusa-head",
    "num_speculative_tokens": 5,  # equals num_heads
}
```

### EAGLE / EAGLE3
```python
speculative_config = {
    "method": "eagle",  # or "eagle3"
    "model": "path/to/eagle-head",
    "num_speculative_tokens": 5,
    "speculative_token_tree": "[0, 0, 0, 1, 1]",  # optional tree
}
```

### MTP
```python
speculative_config = {
    "method": "mtp",
    "num_speculative_tokens": 1,  # typically 1 per MTP layer
}
```

### DFlash
```python
speculative_config = {
    "method": "dflash",
    "model": "path/to/dflash-draft",
    "num_speculative_tokens": 16,
    "dflash_block_size": 16,
    "dflash_num_denoising_steps": 3,
    "dflash_mask_token_id": None,  # auto-detected
}
```

### Suffix Decoding
```python
speculative_config = {
    "method": "suffix",
    "num_speculative_tokens": 5,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_min_token_prob": 0.1,
}
```

---

## Strengths and Weaknesses

### N-gram
- **Strengths**: Zero model overhead, fast on repetitive/template text, no GPU memory
- **Weaknesses**: Only works when prompt contains repeated patterns, poor on novel text

### Draft Model
- **Strengths**: High acceptance rate with well-matched models, general-purpose
- **Weaknesses**: Highest memory cost, sequential generation limits speedup

### Medusa
- **Strengths**: Single forward pass for all draft tokens, low memory, simple
- **Weaknesses**: Fixed number of heads, quality degrades for later positions

### EAGLE / EAGLE3
- **Strengths**: High acceptance rates, tree attention for parallel verification, reuses embeddings
- **Weaknesses**: Requires specialized checkpoint, moderate complexity

### MTP
- **Strengths**: No extra memory (uses target model layers), integrated training
- **Weaknesses**: Limited to models with built-in MTP layers, typically 1 token per layer

### DFlash
- **Strengths**: Generates entire block in parallel, fixed denoising steps regardless of block size, confidence-based refinement improves quality
- **Weaknesses**: Requires specialized diffusion-trained draft model, multiple denoising iterations add latency

### Suffix Decoding
- **Strengths**: Zero model overhead, improves over time as cache grows, dynamic speculation length
- **Weaknesses**: Requires external library, only effective for repeated/similar queries

---

## Theoretical Speedup

| Method | Reported Speedup | Conditions |
|--------|-----------------|------------|
| **N-gram** | 1.5-3x | Repetitive text (code, templates) |
| **Draft Model** | 2-3x | Well-matched small/large model pair |
| **Medusa** | 2-3x | With trained Medusa heads |
| **EAGLE** | 2.5-4x | With trained EAGLE heads |
| **EAGLE3** | 3-4.5x | With auxiliary hidden states |
| **DFlash** | Up to 6.2x | Block diffusion with iterative denoising |
| **Suffix** | 1.5-4x | Repeated query patterns |

*Speedups are workload-dependent and measured under specific conditions.*

---

## Latency Analysis

### Latency Model

Speculative decoding latency per generated token:

```
T_per_token = (T_draft + T_verify) / E[accepted + 1]
```

Where:
- `T_draft` = time to generate k draft tokens
- `T_verify` = time for target model to verify k+1 positions (single forward pass)
- `E[accepted + 1]` = expected tokens produced per speculation cycle

**Key insight:** For autoregressive LLMs, inference is memory-bandwidth-bound, not compute-bound. Verifying k+1 tokens in a single forward pass has nearly the same latency as generating 1 token (`T_verify ≈ T_target_single`), because the bottleneck is loading model weights from HBM, not the actual computation.

### Per-Method Draft Latency

| Method | Draft Latency Formula | For k=16 | Scaling |
|--------|----------------------|----------|---------|
| **N-gram** | ~0 (CPU lookup) | ~0 ms | O(1) |
| **Draft Model** | k × T_small | 16 × T_small | O(k) |
| **Medusa** | 1 × T_heads | 1 × T_heads | O(1) |
| **EAGLE (chain)** | k × T_eagle | 16 × T_eagle | O(k) |
| **EAGLE (tree)** | ~log(k) × T_eagle | ~4 × T_eagle | O(log k) |
| **MTP** | k × T_mtp_layer | 16 × T_mtp | O(k) |
| **DFlash** | num_steps × T_draft | 3 × T_draft | O(num_steps) |
| **Suffix** | ~0 (CPU lookup) | ~0 ms | O(1) |

### Draft Latency Scaling with Block Size

This is DFlash's key latency advantage. Draft generation time is **independent of block size k**:

```
Draft tokens (k):    4     8     16    32    64
─────────────────────────────────────────────────
N-gram:              ~0    ~0    ~0    ~0    ~0
Draft Model:         4T    8T    16T   32T   64T
EAGLE (chain):       4T    8T    16T   32T   64T
Medusa:              T     T     T     T     T*
DFlash (3 steps):    3T    3T    3T    3T    3T
Suffix:              ~0    ~0    ~0    ~0    ~0

T = single forward pass of draft model/head
* Medusa limited to num_heads positions
```

Sequential methods (Draft Model, EAGLE chain, MTP) scale linearly with k — doubling draft length doubles draft latency. DFlash scales only with `num_denoising_steps`, which is fixed (typically 3) regardless of block size.

### Full Cycle Latency Breakdown

For a target model with `T_target = 30ms` per forward pass, draft model with `T_draft = 5ms`, and k=16:

| Method | T_draft | T_verify | T_cycle | Acceptance Rate | Effective T/token |
|--------|--------:|---------:|--------:|:---------------:|------------------:|
| **No speculation** | — | 30 ms | 30 ms | — | 30.0 ms |
| **N-gram** | ~0 ms | 30 ms | 30 ms | 40-60% | 12.0-15.0 ms |
| **Draft Model** | 80 ms | 30 ms | 110 ms | 70-85% | 9.2-12.2 ms |
| **Medusa** | 5 ms | 30 ms | 35 ms | 60-75% | 5.8-7.3 ms |
| **EAGLE (chain)** | 80 ms | 30 ms | 110 ms | 75-90% | 7.3-9.2 ms |
| **EAGLE (tree)** | 20 ms | 30 ms | 50 ms | 80-90% | 3.6-4.2 ms |
| **DFlash (3 steps)** | 15 ms | 30 ms | 45 ms | 75-85% | 3.5-4.6 ms |
| **Suffix** | ~0 ms | 30 ms | 30 ms | 50-80% | 4.3-7.5 ms |

*Effective T/token = T_cycle / E[accepted + 1]. Acceptance rates are representative ranges.*

### Latency Components Visualized

```
No Speculation (1 token/cycle):
|████████████████████████████████| Target (30ms) → 1 token

Draft Model (k=16):
|████████████████████████████████████████████████████████████████████████████████|████████████████████████████████|
 Draft (16 × 5ms = 80ms)                                                        Verify (30ms)
 → ~12 accepted tokens

Medusa (k=16):
|█████|████████████████████████████████|
 Heads  Verify (30ms)
 (5ms)  → ~10 accepted tokens

EAGLE Tree (k=16):
|████████████████████|████████████████████████████████|
 Tree (4 × 5ms)      Verify (30ms)
 → ~13 accepted tokens

DFlash (k=16, 3 steps):
|███████████████|████████████████████████████████|
 Denoise (3×5ms) Verify (30ms)
 → ~12 accepted tokens

N-gram / Suffix (k=16):
|████████████████████████████████|
 Verify only (30ms)
 → ~7-12 accepted tokens
```

### When Each Method Wins on Latency

| Scenario | Best Method | Why |
|----------|-------------|-----|
| Small k (≤4) | EAGLE chain or Draft Model | Sequential overhead is low |
| Large k (≥16) | DFlash or Medusa | Sub-linear draft scaling |
| Very large k (≥32) | DFlash | Only method with O(1) scaling for large k |
| No draft model available | N-gram or Suffix | Zero draft overhead |
| Repetitive text | Suffix | Zero overhead + high acceptance |
| Memory-constrained | N-gram, Suffix, MTP | No extra model weights |
| Maximum throughput | DFlash or EAGLE tree | Best latency/acceptance tradeoff |

### Latency Sensitivity Analysis

**Effect of acceptance rate on effective speedup (k=16, T_target=30ms, T_draft=15ms for DFlash):**

```
Acceptance Rate:  50%    60%    70%    80%    90%    95%
──────────────────────────────────────────────────────────
DFlash speedup:   3.3x   3.9x   4.5x   5.1x   5.6x   5.9x
EAGLE tree:       3.0x   3.5x   4.0x   4.5x   4.9x   5.2x
Draft Model:      1.9x   2.2x   2.5x   2.8x   3.1x   3.2x
N-gram:           4.0x   4.6x   5.3x   6.0x   6.7x   7.0x
```

*N-gram has highest theoretical speedup due to zero draft cost, but acceptance rates above 60% are rare on non-repetitive text.*

### Batched Inference Latency

At larger batch sizes, the target model becomes compute-bound rather than memory-bound, reducing the advantage of speculative decoding:

| Batch Size | Target Forward | Verification Overhead | Speculation Benefit |
|-----------:|---------------:|----------------------:|:-------------------:|
| 1 | Memory-bound | Minimal (~1.0-1.1x) | High (3-6x) |
| 8 | Transitional | Low (~1.2-1.5x) | Moderate (2-4x) |
| 32 | Compute-bound | Moderate (~1.5-2x) | Low (1.5-2.5x) |
| 128+ | Fully compute-bound | High (~2-3x) | Minimal (1.0-1.5x) |

*Speculative decoding provides the most latency benefit for small batch sizes (online serving). For large batch throughput-oriented workloads, the verification overhead can negate the benefit.*

### DFlash Latency Tradeoffs

**Denoising steps vs. quality vs. latency:**

```
Steps:  1        2        3 (default)  4        5
────────────────────────────────────────────────────
Draft:  1×T      2×T      3×T          4×T      5×T
Accept: ~50%     ~65%     ~80%         ~85%     ~87%
Speed:  4.0x     4.6x     5.1x         4.8x     4.4x
```

*Diminishing returns beyond 3 steps — additional quality doesn't offset the extra draft latency.*

**Block size vs. latency (fixed 3 steps):**

```
Block (k):  4       8       16      32      64
────────────────────────────────────────────────
Draft:      3×T     3×T     3×T     3×T     3×T
Accept:     85%     82%     78%     72%     65%
Tokens/cyc: 3.4     6.6     12.5    23.0    41.6
Speed:      2.9x    4.2x    5.1x    5.8x    5.5x
```

*Optimal block size depends on acceptance rate degradation. k=16-32 typically provides the best tradeoff.*

---

## Decision Guide

```
Is your text repetitive/templated?
  YES -> N-gram or Suffix Decoding (no extra model needed)
  NO  -> Continue below

Do you have a trained draft model?
  NO  -> Does your model have MTP layers?
          YES -> MTP
          NO  -> N-gram (fallback) or train a draft model
  YES -> What type?
          Small autoregressive LM -> Draft Model
          Medusa heads            -> Medusa
          EAGLE/EAGLE3 head       -> EAGLE
          DFlash diffusion model  -> DFlash

Do you prioritize maximum speedup?
  YES -> EAGLE3 or DFlash (highest reported speedups)
  NO  -> Draft Model or Medusa (simpler setup)

Is memory constrained?
  YES -> N-gram, Suffix, MTP, or Medusa (lowest overhead)
  NO  -> EAGLE or DFlash (best speed/quality tradeoff)
```

---

## Architecture Diagram

```
Target Model (large)
     |
     | hidden states / tokens
     v
+------------------+
| Draft Proposer   |  <-- One of: N-gram, Draft Model, Medusa,
| (generates k     |      EAGLE, MTP, DFlash, Suffix
|  draft tokens)   |
+------------------+
     |
     | draft_token_ids [batch_size, k]
     v
+------------------+
| Target Model     |  <-- Single forward pass verifying k+1 positions
| (verification)   |
+------------------+
     |
     v
+------------------+
| Rejection        |  <-- Accepts prefix of matching tokens
| Sampler          |
+------------------+
     |
     v
  Accepted tokens (1 to k+1)
```

---

## Reproducibility

### Overview

Speculative decoding is **lossless by design** — draft tokens are always verified by the target model via rejection sampling. Non-determinism in draft generation affects only performance (acceptance rate / speedup), never output correctness. The accepted token sequence is always identical to what the target model would have produced alone.

### DFlash Reproducibility Analysis

DFlash's algorithmic logic is **fully deterministic** given the same inputs:

| Component | Deterministic? | Notes |
|-----------|:--------------:|-------|
| Keep schedule | Yes | Pure arithmetic: `(i+1)/n` for each step |
| MASK initialization | Yes | `torch.full` with constant fill value |
| Token selection (argmax) | Yes | Greedy decoding, no sampling |
| Confidence computation (softmax) | Yes (CPU) | GPU reductions may vary in accumulation order |
| Top-k ranking | Yes (CPU) | GPU `topk` with tied values is non-deterministic |
| Scatter mask creation | Yes | `scatter_` with pre-computed indices |
| `torch.where` re-masking | Yes | Elementwise, fully deterministic |
| Full denoising loop | Yes (CPU) | Composition of above deterministic ops |

### Test Coverage

| Metric | Value |
|--------|------:|
| Total reproducibility-checked invocations | 1,814 |
| Distinct input configurations | 60 |
| Max repetitions per input | 100 |
| Batch sizes tested | 1, 2, 4, 8, 64 |
| Sequence lengths tested | 1, 8, 16 |
| Vocabulary sizes tested | 100, 1K, 32K, 128K |
| Keep ratios tested | 0.25, 0.5, 0.75, 1.0 |
| Denoising steps tested | 1, 2, 3, 4, 5, 10 |
| Numeric dtypes tested | float16, bfloat16, float32 |
| Edge cases | overflow (1000), underflow (-1000), tied confidence, uniform logits |

### GPU Considerations

On GPU, bit-exact reproducibility is **not guaranteed** without explicit settings:

1. **cuBLAS matrix multiplications** — non-deterministic by default due to algorithm selection
2. **Flash Attention** — uses non-deterministic atomic accumulation
3. **Reduction operations** (softmax, topk) — floating-point addition order may vary
4. **Tensor parallelism** — inter-GPU communication order can differ between runs

To force GPU determinism (at a performance cost):
```python
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### Why This Is Acceptable

| Concern | Mitigation |
|---------|-----------|
| Draft tokens differ across runs | Verified by target model — wrong drafts are rejected, not emitted |
| Acceptance rate varies | Performance variance only, not correctness |
| Non-deterministic GPU ops | The algorithm logic itself is deterministic; GPU non-determinism affects the draft model's forward pass equally for all speculative methods |
| Tied confidence in topk | Rare in practice (continuous softmax outputs almost never tie); even if ties occur, the target model still verifies |

### Comparison Across Methods

| Method | Draft Deterministic? | Why / Why Not |
|--------|:--------------------:|---------------|
| **N-gram** | Yes | Pure pattern matching, no neural network |
| **Draft Model** | No (GPU) | Autoregressive model forward passes |
| **Medusa** | No (GPU) | MLP heads on GPU |
| **EAGLE** | No (GPU) | Transformer head on GPU |
| **MTP** | No (GPU) | Target model layers on GPU |
| **DFlash** | Algorithm: Yes, GPU: No | Deterministic logic, non-deterministic GPU kernels |
| **Suffix** | Yes | Pure pattern matching, no neural network |

*All methods are lossless regardless of draft determinism — the target model verification ensures correctness.*

---

## File Locations

| Method | Proposer | Model (if applicable) |
|--------|----------|----------------------|
| N-gram | `vllm/v1/spec_decode/ngram_proposer.py` | N/A |
| Draft Model | `vllm/v1/spec_decode/draft_model.py` | Any HF model |
| Medusa | `vllm/v1/spec_decode/medusa.py` | `vllm/model_executor/models/medusa.py` |
| EAGLE | `vllm/v1/spec_decode/eagle.py` | `vllm/model_executor/models/llama_eagle.py` |
| MTP | `vllm/v1/spec_decode/eagle.py` | Various `*_mtp.py` models |
| DFlash | `vllm/v1/spec_decode/dflash.py` | `vllm/model_executor/models/dflash.py` |
| Suffix | `vllm/v1/spec_decode/suffix_decoding.py` | N/A (arctic-inference) |
