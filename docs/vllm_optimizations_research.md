# vLLM Optimization Research

A comprehensive analysis of vLLM's performance optimizations and their applicability to tinyvllm.

**Research Date:** January 2025
**Sources:** vLLM v0.6.0 blog, vLLM docs, DeepWiki analysis

---

## Table of Contents

1. [Performance Bottleneck Analysis](#1-performance-bottleneck-analysis)
2. [Memory Bandwidth Optimizations](#2-memory-bandwidth-optimizations)
3. [CPU Overhead Reduction](#3-cpu-overhead-reduction)
4. [Attention Optimizations](#4-attention-optimizations)
5. [Scheduling Optimizations](#5-scheduling-optimizations)
6. [Quantization Support](#6-quantization-support)
7. [CUDA-Specific Optimizations](#7-cuda-specific-optimizations)
8. [Comparison with tinyvllm](#8-comparison-with-tinyvllm)
9. [Implementation Priority](#9-implementation-priority)

---

## 1. Performance Bottleneck Analysis

### vLLM's Profiling Findings (v0.6.0)

For Llama 3 8B on 1 H100 GPU, vLLM found:

| Component | Time % |
|-----------|--------|
| HTTP API Server | 33% |
| Scheduling | 29% |
| **GPU Execution** | **38%** |

**Key Insight:** Only 38% of time was spent on actual GPU work. The rest was CPU overhead.

### Decode vs Prefill Characteristics

| Phase | Characteristic | Bottleneck |
|-------|---------------|------------|
| Prefill | Process entire prompt at once | Compute-bound |
| Decode | Generate 1 token at a time | Memory-bandwidth-bound |

For decode, the bottleneck is loading model weights from GPU memory:
```
Per token: Load ~2-7GB of weights → Compute → Output 1 token
Theoretical max = Memory Bandwidth / Model Size
Example: 120 GB/s / 2.2 GB = 55 tok/s (TinyLlama on M4)
```

---

## 2. Memory Bandwidth Optimizations

### 2.1 Weight Quantization

vLLM's most impactful optimization for memory-bound decode.

**Supported Formats:**
- **FP8 (E4M3/E5M2):** 2x compression, minimal quality loss
- **INT8:** 2x compression, requires calibration
- **INT4 (GPTQ/AWQ):** 4x compression, slight quality loss
- **MXFP4:** 4x compression, newer format

**Implementation Details:**
```
FP16 weights: 2 bytes/param → Load 2.2 GB for TinyLlama
INT8 weights: 1 byte/param → Load 1.1 GB (2x faster)
INT4 weights: 0.5 bytes/param → Load 0.55 GB (4x faster)
```

**vLLM Kernels:**
- **Marlin kernels:** Optimized INT4/FP8 matmul for Ampere+
- **CUTLASS 3.x:** For Hopper/Blackwell GPUs
- **DeepGEMM:** Block-wise FP8 with E8M0 scales

**Expected Speedup:** 2-4x throughput improvement

### 2.2 KV Cache Quantization

Separate from weight quantization - reduces KV cache memory.

- FP8 KV cache: 2x memory reduction
- INT8 KV cache: 2x memory reduction
- Enables longer contexts without OOM

### 2.3 Fused Kernels

Reduce memory round-trips by combining operations:

```
Without fusion:
  Load X → RMSNorm → Store → Load → Linear → Store
  (4 memory operations)

With fusion:
  Load X → RMSNorm+Linear → Store
  (2 memory operations)
```

**Common fusions in vLLM:**
- RMSNorm + Linear projection
- RoPE + Q/K projection
- SiLU + element-wise multiply (SwiGLU)
- Attention + softmax + value aggregation

---

## 3. CPU Overhead Reduction

### 3.1 API Server Separation (PR #6883)

**Problem:** HTTP server and vLLM engine competed for Python's GIL.

**Solution:** Separate processes connected via ZMQ sockets.

```
Before:
  [HTTP Server + vLLM Engine] (single process, GIL contention)

After:
  [HTTP Server] <--ZMQ--> [vLLM Engine] (separate processes)
```

### 3.2 Object Pooling (PR #7162)

**Problem:** Python object allocation/deallocation overhead during request lifecycle.

**Solution:** Pre-allocate and reuse request objects.

```python
# Before: allocate every time
request = Request(prompt, params)  # malloc
process(request)
del request  # free

# After: pool and reuse
request = request_pool.acquire()
request.reset(prompt, params)
process(request)
request_pool.release(request)  # no free, just return to pool
```

**Result:** 24% throughput improvement

### 3.3 Fast Path for Simple Sampling (PR #7117)

Bypass complex sampling logic for greedy decoding:

```python
# Complex path (temperature, top-p, top-k, penalties, etc.)
def sample_complex(logits, params):
    # ... many operations

# Fast path (greedy)
def sample_greedy(logits):
    return logits.argmax()  # single operation
```

### 3.4 Async Output Processing (PRs #7049, #7921, #8050)

**Problem:** Detokenization blocks GPU execution.

**Solution:** Process step N-1 output while GPU runs step N.

```
Before:
  GPU[step N] → wait → CPU[detokenize N] → GPU[step N+1]

After:
  GPU[step N] ←――――――――――――――――――――――――→ GPU[step N+1]
              CPU[detokenize N-1]
```

**Result:** 8.7% TPOT improvement on Llama 70B

### 3.5 Non-Blocking Data Transfers (PR #7172)

Launch CPU→GPU copies without waiting:

```python
# Before: blocking
tensor.to(device)  # waits for completion

# After: non-blocking
tensor.to(device, non_blocking=True)  # returns immediately
# GPU can start working on previous data while new data transfers
```

---

## 4. Attention Optimizations

### 4.1 FlashAttention

**Core Innovation:** Tiled attention that never materializes full attention matrix.

```
Standard Attention:
  S = Q @ K.T          # [seq, seq] matrix - O(n²) memory
  P = softmax(S)       # [seq, seq] matrix
  O = P @ V            # Output

FlashAttention:
  for tile in tiles:
    # Compute attention for tile only
    # Never store full [seq, seq] matrix
    # O(1) memory regardless of sequence length
```

**Memory Savings:**
- 10x at seq_len=2K
- 20x at seq_len=4K

**vLLM Integration:**
- `flash_attn_varlen_func`: Variable-length prefill
- `flash_attn_with_kvcache`: Decode with KV cache

### 4.2 PagedAttention V1 vs V2

| Version | Strategy | Best For |
|---------|----------|----------|
| V1 | Single kernel | Short sequences, lower overhead |
| V2 | Two-stage (reduce + aggregate) | Long sequences |

vLLM auto-selects based on sequence length.

### 4.3 FlashInfer Backend

Alternative attention backend optimized for:
- Hopper/Blackwell architectures
- Advanced KV cache management
- MoE (Mixture of Experts) models

---

## 5. Scheduling Optimizations

### 5.1 Multi-Step Scheduling (PR #7000)

**Problem:** CPU scheduling overhead between every decode step.

**Solution:** Schedule N steps at once, let GPU run without interruption.

```
Before (single-step):
  CPU[schedule] → GPU[decode] → CPU[schedule] → GPU[decode] → ...

After (multi-step, N=8):
  CPU[schedule] → GPU[decode×8] → CPU[schedule] → GPU[decode×8] → ...
```

**Trade-off:**
- Pro: 28% throughput improvement (Llama 70B)
- Con: Higher TTFT at low request rates (new requests wait for batch)

**Configuration:** `--num-scheduler-steps 8`

### 5.2 Chunked Prefill

**Problem:** Long prompts block other requests during prefill.

**Solution:** Break prefill into chunks, interleave with decode.

```
Before:
  [Prefill 10K tokens]―――――――――――――――――→ [Decode requests wait]

After:
  [Prefill chunk 1] [Decode] [Prefill chunk 2] [Decode] ...
```

**Benefits:**
- 30% TTFT reduction
- 1.4x ITL improvement
- Better GPU utilization (mix compute + memory bound)

### 5.3 Iteration-Level Scheduling

vLLM schedules at granularity of individual tokens, not full sequences:

```
Static batching:
  Wait for all sequences in batch to finish → Start new batch

Continuous batching:
  Sequence finishes → Immediately fill slot with new request
```

### 5.4 Disaggregated Prefill

Separate prefill and decode to different GPU pools:

```
Prefill Pool (compute-optimized):
  Process prompts, transfer KV cache to decode pool

Decode Pool (memory-optimized):
  Generate tokens using transferred KV cache
```

---

## 6. Quantization Support

### 6.1 Weight Quantization Methods

| Method | Bits | Requires Calibration | Quality |
|--------|------|---------------------|---------|
| FP8 | 8 | No | Excellent |
| INT8 | 8 | Yes (simple) | Very Good |
| GPTQ | 4 | Yes (data) | Good |
| AWQ | 4 | Yes (activation-aware) | Good |
| GGUF | 2-8 | Varies | Varies |

### 6.2 vLLM Quantization Kernels

**Marlin Kernels:**
- Optimized for INT4/FP8 on Ampere+
- 4-bit weights packed into INT32
- Dequantize during matmul

**CUTLASS 3.x:**
- For Hopper (SM90+)
- Native FP8 tensor core support
- Block-wise scaling

### 6.3 Activation Quantization

Beyond weight quantization:
- FP8 activations (E4M3)
- INT8 activations
- Reduces memory for intermediate tensors

---

## 7. CUDA-Specific Optimizations

### 7.1 CUDA Graphs

**Problem:** Kernel launch overhead (~10μs per kernel) adds up.

**Solution:** Capture entire decode loop as a graph, replay without launch overhead.

```python
# Capture phase (once)
with torch.cuda.graph(graph):
    output = model.decode(input)

# Replay phase (every decode step)
graph.replay()  # Near-zero launch overhead
```

**Challenges:**
- Fixed tensor sizes (use batch bucketing)
- Memory pre-allocation required
- Can't use dynamic control flow

### 7.2 Custom CUDA Kernels

vLLM implements specialized kernels for:
- Paged attention (gather from scattered blocks)
- Cache reshape and copy
- Sampling (top-k, top-p on GPU)
- RMS normalization (fused)
- RoPE (fused with projection)

### 7.3 Architecture-Specific Tuning

| Architecture | Optimizations |
|--------------|---------------|
| Ampere (A100) | FP16 tensor cores, async copy |
| Ada (4090) | FP8 tensor cores |
| Hopper (H100) | FP8 native, TMA, warp specialization |
| Blackwell | FP4, larger tensor cores |

---

## 8. Comparison with tinyvllm

### Features Already Implemented

| Feature | vLLM | tinyvllm | Status |
|---------|------|----------|--------|
| PagedAttention | ✅ | ✅ | Phase 2 done |
| Continuous Batching | ✅ | ✅ | Phase 3 done |
| Block-based KVCache | ✅ | ✅ | Phase 4 done |
| Custom Attention Kernel | ✅ CUDA | ✅ Metal | Phase 5 done |
| SIMD Optimizations | ✅ | ✅ | Phase 6.2 done |

### Features Planned but Not Implemented

| Feature | vLLM | tinyvllm Phase | Priority |
|---------|------|----------------|----------|
| Prefix Caching | ✅ | 7.1 | Medium |
| Speculative Decoding | ✅ | 7.2 | High |
| KV Cache Quantization | ✅ | 7.4 | Medium |
| Chunked Prefill | ✅ | 6.4 | Medium |
| Multi-GPU | ✅ | 8 | Low |

### Features Missing from tinyvllm Phases

| Feature | vLLM | Impact | Notes |
|---------|------|--------|-------|
| **Weight Quantization** | ✅ INT4/8, FP8 | **CRITICAL** | 2-4x speedup |
| **CUDA Graphs** | ✅ | High | 1.3-2x speedup |
| **FlashAttention** | ✅ | High | 2-4x prefill |
| **Multi-Step Scheduling** | ✅ | High | 28% throughput |
| **Async Output Processing** | ✅ | Medium | 8.7% improvement |
| Object Pooling | ✅ | Medium | 24% throughput |
| Fused Kernels | ✅ | Medium | Fewer memory ops |
| torch.compile | ✅ | Medium | JIT optimization |

---

## 9. Implementation Priority

Based on tinyvllm's current bottleneck (memory bandwidth limited decode):

### Tier 1: Critical (2-4x improvement each)

1. **Weight Quantization (INT8/INT4)**
   - Reduces weight memory by 2-4x
   - Directly addresses memory bandwidth bottleneck
   - Effort: Medium (new kernels needed)

2. **Speculative Decoding**
   - Generate 3-4 tokens per weight load
   - Uses existing infrastructure
   - Effort: Medium (draft model integration)

### Tier 2: High Impact (1.3-2x improvement)

3. **Multi-Step Scheduling**
   - Reduces CPU scheduling overhead
   - Simple scheduler modification
   - Effort: Low

4. **FlashAttention for Prefill**
   - Faster prefill, enables longer context
   - Complex kernel implementation
   - Effort: High

5. **CUDA Graphs** (CUDA backend only)
   - Eliminates kernel launch overhead
   - Requires fixed batch sizes
   - Effort: Medium

### Tier 3: Medium Impact (10-30% improvement)

6. **Async Output Processing**
   - Overlap detokenize with GPU
   - Effort: Low

7. **Object Pooling**
   - Reduce Python allocation overhead
   - Effort: Low

8. **Fused Kernels**
   - RMSNorm+Linear, RoPE+Projection
   - Effort: Medium (new kernels)

---

## References

- [vLLM v0.6.0 Performance Update](https://blog.vllm.ai/2024/09/05/perf-update.html)
- [vLLM 2024 Retrospective and 2025 Vision](https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision.html)
- [vLLM Performance Optimizations - DeepWiki](https://deepwiki.com/vllm-project/vllm/5-performance-optimizations)
- [vLLM Optimization Docs](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Continuous Batching - Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
