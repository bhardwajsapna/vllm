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
