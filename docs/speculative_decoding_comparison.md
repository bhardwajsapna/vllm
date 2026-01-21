# Speculative Decoding Techniques Comparison in vLLM

This document provides a comprehensive comparison of the speculative decoding methods available in vLLM, including the newly added DFlash (Block Diffusion) method.

## Overview

Speculative decoding is a technique to accelerate LLM inference by using a faster "draft" mechanism to generate candidate tokens, which are then verified by the target model in parallel. This can significantly improve throughput while maintaining output quality.

---

## Methods at a Glance

| Method | Type | Draft Mechanism | Parallel Generation | Model Required |
|--------|------|-----------------|---------------------|----------------|
| **N-gram** | Pattern-based | Prompt lookup | No | None |
| **Medusa** | Head-based | Multiple prediction heads | Yes | Medusa heads |
| **EAGLE/EAGLE3** | Model-based | Auxiliary draft model | Tree-based | EAGLE checkpoint |
| **MTP** | Native | Built-in prediction layers | Yes | Native support |
| **MLP Speculator** | Head-based | Simple MLP networks | Tree-based | MLP weights |
| **Suffix Decoding** | Cache-based | Suffix tree matching | No | None (cache only) |
| **DFlash** | Diffusion-based | Block diffusion denoising | Yes | DFlash model |
| **Draft Model** | Model-based | Smaller LLM | No | Any smaller LLM |

---

## Detailed Comparison

### 1. N-gram (Prompt Lookup)

**How it works:** Searches the prompt history for matching n-gram patterns and proposes the tokens that followed those patterns.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Very Fast (CPU-based) |
| **Memory Overhead** | Minimal |
| **Acceptance Rate** | Variable (depends on prompt patterns) |
| **Best Use Case** | Prompts with repetitive patterns |

**Configuration:**
```python
speculative_config = {
    "method": "ngram",
    "num_speculative_tokens": 5,
    "prompt_lookup_min": 5,
    "prompt_lookup_max": 5,
}
```

**Pros:**
- Zero model overhead
- No GPU memory required
- Works with any model

**Cons:**
- Only effective with repetitive content
- Cannot leverage semantic understanding

---

### 2. Medusa

**How it works:** Attaches multiple parallel prediction heads to the target model's hidden states. Each head independently predicts the next token.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Fast |
| **Memory Overhead** | Moderate (small heads) |
| **Acceptance Rate** | High |
| **Best Use Case** | General purpose speculation |

**Configuration:**
```python
speculative_config = {
    "method": "medusa",
    "model": "path/to/medusa-checkpoint",
    "num_speculative_tokens": 4,
}
```

**Pros:**
- Lightweight compared to full draft models
- Parallel generation is efficient
- Good accuracy

**Cons:**
- Requires trained Medusa heads
- Model-specific checkpoints needed

---

### 3. EAGLE / EAGLE3

**How it works:** Uses an optimized auxiliary draft model architecture that leverages target model hidden states. EAGLE3 additionally uses intermediate layer outputs for better predictions.

| Aspect | EAGLE | EAGLE3 |
|--------|-------|--------|
| **Generation Speed** | Fast | Fast |
| **Memory Overhead** | Moderate-High | Moderate-High |
| **Acceptance Rate** | Very High (80-90%+) | Very High |
| **Hidden State Usage** | Final layer | Multiple layers |

**Configuration:**
```python
# EAGLE
speculative_config = {
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_speculative_tokens": 4,
}

# EAGLE3
speculative_config = {
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 4,
}
```

**Pros:**
- Excellent acceptance rates
- Tree-based verification improves throughput
- Well-established with available checkpoints

**Cons:**
- Limited to supported architectures (Llama, Qwen, MiniCPM)
- Requires specific EAGLE checkpoints

---

### 4. MTP (Multi-Token Prediction)

**How it works:** Native support in certain model architectures (DeepSeek-V3, Qwen3, GLM-4) that predict multiple tokens per forward pass.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Very Fast |
| **Memory Overhead** | Low (uses target model) |
| **Acceptance Rate** | Good |
| **Best Use Case** | Supported model architectures |

**Supported Models:**
- DeepSeek-V3/V32
- Qwen3-Next
- GLM-4 MOE
- ERNIE
- PanguAlpha Ultra

**Configuration:**
```python
speculative_config = {
    "method": "mtp",
    "num_speculative_tokens": 4,  # Must be divisible by n_predict
}
```

**Pros:**
- No additional model loading
- Efficient native implementation
- Lower memory overhead

**Cons:**
- Only works with specific model families
- Architecture-dependent

---

### 5. MLP Speculator

**How it works:** Uses simple MLP networks to predict multiple candidate tokens from target model hidden states with tree-based candidate generation.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Very Fast |
| **Memory Overhead** | Minimal |
| **Acceptance Rate** | Moderate |
| **Best Use Case** | Lightweight setups |

**Configuration:**
```python
speculative_config = {
    "method": "mlp_speculator",
    "model": "path/to/mlp-speculator",
    "num_speculative_tokens": 3,
}
```

**Pros:**
- Very lightweight
- Small parameter overhead

**Cons:**
- Only supports tensor parallel size = 1
- Limited documentation

---

### 6. Suffix Decoding

**How it works:** Builds suffix trees from cached responses and uses statistical pattern matching to predict continuations.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Medium |
| **Memory Overhead** | Low-Moderate (tree cache) |
| **Acceptance Rate** | Variable (cache-dependent) |
| **Best Use Case** | Long-running services with response caching |

**Configuration:**
```python
speculative_config = {
    "method": "suffix",
    "num_speculative_tokens": 24,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000,
    "suffix_decoding_min_token_prob": 0.1,
}
```

**Pros:**
- No model training needed
- Learns from actual outputs
- Adapts to workload patterns

**Cons:**
- Requires Arctic Inference library
- Cold start performance is poor
- Best for services with repeated queries

---

### 7. DFlash (Block Diffusion) - NEW

**How it works:** Uses iterative denoising (diffusion process) to generate an entire block of draft tokens in parallel, conditioned on target model hidden states.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Moderate (multiple diffusion steps) |
| **Memory Overhead** | Moderate |
| **Acceptance Rate** | TBD (new method) |
| **Best Use Case** | Experimental/Research |

**Configuration:**
```python
speculative_config = {
    "method": "dflash",
    "model": "path/to/dflash-checkpoint",
    "num_speculative_tokens": 8,
}
```

**Key Parameters (in model config):**
- `block_size`: Number of tokens per block (default: 8)
- `num_diffusion_steps`: Denoising iterations (default: 8)
- `noise_schedule`: "cosine", "linear", or "sqrt"

**Pros:**
- Parallel block generation
- Novel diffusion-based approach
- Can share embeddings with target model

**Cons:**
- Newer method with limited testing
- Requires trained DFlash models
- Multiple diffusion steps add latency

---

### 8. Draft Model

**How it works:** Uses any smaller/faster language model to generate draft tokens sequentially, which are then verified by the target model.

| Aspect | Details |
|--------|---------|
| **Generation Speed** | Slow (sequential) |
| **Memory Overhead** | High (full model) |
| **Acceptance Rate** | Good (model-dependent) |
| **Best Use Case** | When smaller version of target exists |

**Configuration:**
```python
speculative_config = {
    "method": "draft_model",
    "model": "meta-llama/Llama-2-7b",  # Smaller than target
    "num_speculative_tokens": 5,
    "draft_tensor_parallel_size": 1,
}
```

**Pros:**
- Most flexible approach
- Can use any smaller model
- Well-understood approach

**Cons:**
- Highest memory overhead
- Slowest draft generation
- Requires matching vocabulary sizes

---

## Performance Comparison Matrix

| Method | Memory | Latency | Throughput Gain | Setup Complexity |
|--------|--------|---------|-----------------|------------------|
| N-gram | Low | Low | Low-Medium | Very Easy |
| Medusa | Medium | Low | Medium-High | Medium |
| EAGLE | Medium-High | Low | High | Medium |
| EAGLE3 | Medium-High | Low | Very High | Medium |
| MTP | Low | Very Low | Medium | Easy (if supported) |
| MLP Speculator | Low | Very Low | Medium | Easy |
| Suffix | Low-Medium | Medium | Variable | Easy |
| DFlash | Medium | Medium | TBD | Medium |
| Draft Model | High | High | Medium | Complex |

---

## Decision Guide

### Choose **N-gram** if:
- Your prompts have repetitive patterns
- You want zero model overhead
- Memory is constrained

### Choose **Medusa** if:
- You want a good balance of speed and accuracy
- Medusa heads are available for your model
- General-purpose speculation is needed

### Choose **EAGLE/EAGLE3** if:
- You need maximum throughput
- Your model is supported (Llama, Qwen, etc.)
- High acceptance rates are critical

### Choose **MTP** if:
- Your model natively supports multi-token prediction
- You want minimal overhead
- Using DeepSeek, Qwen3, or similar

### Choose **Suffix Decoding** if:
- Running a long-lived service
- Queries have repeating patterns over time
- Cold start latency is acceptable

### Choose **DFlash** if:
- Experimenting with novel approaches
- Interested in diffusion-based generation
- Research/prototyping scenarios

### Choose **Draft Model** if:
- A smaller version of your target model exists
- Flexibility is more important than efficiency
- Other methods aren't available for your model

---

## Usage Example

```python
from vllm import LLM

# Example: Using DFlash speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "dflash",
        "model": "path/to/dflash-checkpoint",
        "num_speculative_tokens": 8,
    },
)

# Example: Using EAGLE
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 4,
    },
)

# Example: Using N-gram (no additional model)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_min": 5,
        "prompt_lookup_max": 5,
    },
)

output = llm.generate("Your prompt here")
```

---

## References

- [Medusa Paper](https://arxiv.org/abs/2401.10774)
- [EAGLE Paper](https://arxiv.org/abs/2401.15077)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Speculative Decoding Overview](https://arxiv.org/abs/2302.01318)
