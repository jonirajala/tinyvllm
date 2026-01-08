# tinyvllm Roadmap

Future improvements based on research of vLLM, SGLang, TensorRT-LLM, and llama.cpp.

**Current Performance:** ~5 tok/s (9% of theoretical 55 tok/s max)

---

## Phase 1: Quick Wins (10-30% improvement)

### 1.1 Buffer Reuse ✅ DONE
**Impact:** 10-20% | **Effort:** Low | **Source:** vLLM V1 "Persistent Batch"

vLLM V1's biggest optimization: cache input tensors and only apply diffs each step.
- Pre-allocate decode buffers at engine init
- Use `Tensor.assign()` to update in-place
- Avoid Tensor.zeros() allocation each step

**Implemented:** `_bt_buffer_data` and `_ctx_buffer_data` in `engine.py`

### 1.2 Object Pooling
**Impact:** 10-15% | **Effort:** Low | **Source:** vLLM saw 24% improvement

Pool and reuse Python objects:
- Request, Sequence, SchedulerOutput with `.reset()` methods
- Reduces GC pressure and allocation overhead

### 1.3 Batched Sampling
**Impact:** 5-10% | **Effort:** Low

Sample all sequences in batch together instead of loop.

### 1.4 Incremental Block Allocation
**Impact:** 30-50% memory savings | **Effort:** Low | **Source:** [vLLM PagedAttention](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

Current approach pre-allocates all blocks upfront (wastes 60-80% memory).
vLLM allocates blocks incrementally as tokens are generated.

- Use `get_slot()` instead of pre-allocating full sequence length
- Only grab new block when current block is full
- Enables more concurrent sequences with same memory

**Already implemented:** `block_manager.get_slot()` exists but unused.
**TODO:** Refactor engine to use incremental allocation.

---

## Phase 2: Chunked Prefill (Major Feature)

### 2.1 Chunked Prefill
**Impact:** +50% throughput, better latency | **Effort:** Medium | **Source:** [vLLM](https://docs.vllm.ai/en/stable/configuration/optimization/), [Sarathi paper](https://arxiv.org/pdf/2308.16369)

**What it does:**
- Split long prompts into chunks (512-8192 tokens)
- Mix prefill chunks with decode in same batch
- Prefill is compute-bound, decode is memory-bound → better GPU utilization

**Implementation:**
1. Scheduler prioritizes decode requests first
2. Fill remaining batch budget with prefill chunks
3. Track partial prefill progress per sequence
4. Flatten to 1D query tensor (not 2D batch)

**Why it matters:**
- Prevents head-of-line blocking from long prompts
- vLLM V1 has it always enabled
- Up to 30% TTFT reduction

### 2.2 Batched Prefill
**Impact:** Higher throughput | **Effort:** Medium

Process multiple prefill sequences together when lengths similar.

---

## Phase 3: Prefix Caching

### 3.1 Automatic Prefix Caching (APC)
**Impact:** 2-5x for repeated prefixes | **Effort:** Medium | **Source:** [vLLM](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

**What it does:**
- Cache KV blocks for common prompt prefixes
- Reuse cached KV on cache hit instead of recomputing
- LRU eviction when memory full

**vLLM V1 insight:** Near-zero overhead even with 0% hit rate (enabled by default).

### 3.2 RadixAttention (SGLang approach)
**Impact:** Better for dynamic patterns | **Effort:** High | **Source:** [SGLang](https://lmsys.org/blog/2024-01-17-sglang/)

**What it does:**
- Store KV cache in radix tree (prefix tree with sequence edges)
- Automatic prefix discovery at runtime
- Works for both prompts AND generation results

**When to use:**
- APC: Predictable, structured caching patterns
- RadixAttention: Unpredictable, dynamic conversations

**Use cases:** Chat (system prompts), few-shot prompting, agents, tree-of-thought.

---

## Phase 4: Speculative Decoding

### 4.1 Draft Model Speculation
**Impact:** 2-3x throughput | **Effort:** High | **Source:** [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)

**How it works:**
1. Small draft model generates N candidate tokens (fast)
2. Main model verifies all N tokens in single forward pass
3. Accept matching tokens, reject where they diverge
4. Net win if draft acceptance rate > 1/N

**Current SOTA:** Eagle3 - uses hidden states from 3 verifier layers as input.

**vLLM support:** Eagle1, Eagle3 with CUDA graphs, acceptance rate metrics.

### 4.2 Self-Speculation (Medusa/EAGLE)
**Impact:** 2x, no separate model | **Effort:** Medium

Add small prediction heads to main model instead of separate draft model.

---

## Phase 5: API Server

### 5.1 HTTP Server with Streaming
**Impact:** Real-world usability | **Effort:** Medium

```
POST /generate
POST /v1/completions (OpenAI compat)
POST /v1/chat/completions (OpenAI compat)
```

- Server-sent events (SSE) for token streaming
- Request queuing and cancellation
- Async request handling

### 5.2 Structured Output
**Impact:** Better usability | **Effort:** Medium

- JSON schema enforcement
- Grammar-based constrained decoding
- Stop string matching (beyond EOS)

---

## Phase 6: Advanced Optimizations

### 6.1 Quantization (when tinygrad supports)
**Impact:** 2-4x memory, possibly faster | **Effort:** Medium | **Source:** llama.cpp, TensorRT-LLM

**Types:**
- **INT8 weights:** tinygrad has embedding support (PR #9273)
- **INT4 weights:** tinygrad "immediate goal"
- **KV cache INT8:** 50% memory reduction
- **k-quants (llama.cpp):** Q4_K_M, Q5_K_S, etc.

### 6.2 Persistent KV Cache
**Impact:** Faster restarts | **Effort:** Low

Save/load KV cache to disk for common prefixes.

### 6.3 Kernel Fusion Investigation
**Impact:** 10-20% | **Effort:** Medium

Check what tinygrad auto-fuses with `DEBUG=4`:
- RMSNorm + Linear
- RoPE + Q/K projection
- Attention + Output projection

---

## Phase 7: Multi-GPU & Scaling

### 7.1 Tensor Parallelism
**Impact:** Larger models | **Effort:** High | **Source:** vLLM, TensorRT-LLM

Split model layers across GPUs:
- Each GPU holds fraction of weights
- All-reduce for combining outputs

### 7.2 Disaggregated Prefill/Decode
**Impact:** Better latency control | **Effort:** Very High | **Source:** [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf), [vLLM](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

**What it does:**
- Separate GPU pools for prefill vs decode
- Prefill GPUs: optimize for TTFT
- Decode GPUs: optimize for ITL (inter-token latency)
- Transfer KV cache via RDMA

**When useful:** Large scale, strict latency SLAs. Not for small deployments.

### 7.3 Pipeline Parallelism
**Impact:** Very large models | **Effort:** High

Different layers on different GPUs, pipeline micro-batches.

---

## Phase 8: Memory Management

### 8.1 Preemption & Recompute
**Impact:** Handle memory pressure | **Effort:** Medium

When OOM:
- Preempt lowest priority sequence
- Free its KV blocks
- Recompute later when memory available

### 8.2 CPU Offload
**Impact:** Longer contexts | **Effort:** High

- Swap inactive KV blocks to CPU
- Async transfers to hide latency
- Bring back when needed

### 8.3 KV Cache Compression
**Impact:** 2-4x longer contexts | **Effort:** Medium

- Quantize KV cache to INT8
- Or use lower precision (FP8 on supported hardware)

---

## Phase 9: Model Support

### 9.1 More Architectures
**Effort:** Medium per model

- Mistral/Mixtral (sliding window attention, MoE)
- Qwen
- Phi
- Gemma

### 9.2 Multimodal
**Effort:** High

- Vision encoders (CLIP, SigLIP)
- Image token handling
- Audio models

---

## Not Planned / Deferred

| Feature | Reason |
|---------|--------|
| Custom Metal/CUDA kernels | TinyJit + tinygrad is simpler and competitive |
| Manual quantization kernels | Wait for native tinygrad support |
| Complex priority scheduling | Overkill for current scale |
| Disaggregated serving | Only useful at large scale with RDMA |

---

## Priority Matrix

```
QUICK WINS (do first):
├── 1.1 Buffer reuse                     ~15% ✅ DONE
├── 1.2 Object pooling                   ~10%
├── 1.3 Batched sampling                 ~5%
└── 1.4 Incremental block allocation     ~40% memory

HIGH IMPACT (do next):
├── 2.1 Chunked prefill                  ~50% throughput
├── 3.1 Prefix caching (APC)             2-5x for cache hits
└── 5.1 HTTP server                      usability

MEDIUM IMPACT (later):
├── 4.1 Speculative decoding             2-3x
├── 6.1 Quantization                     2-4x memory
└── 5.2 Structured output                usability

ADVANCED (someday):
├── 7.x Multi-GPU                        larger models
├── 8.x Memory management                longer contexts
└── 9.x More models                      broader support
```

---

## References

- [vLLM V1 Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
- [vLLM Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [DistServe: Disaggregated Prefill/Decode](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
