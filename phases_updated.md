
# tinyvllm Implementation Roadmap (Priority-Ordered)

**Current Status:** 1.7 tok/s single / 3.6 tok/s batched vs 55 tok/s theoretical (6.6% utilization)
**Progress:** Phase 7 complete (+42% single, +33% batched vs Phase 6.2)
**Primary Bottleneck:** Python→GPU scheduling overhead, NOT kernel speed

---

## Phase 1: Foundation ✅

1.1 Token Generation Loop ✅
- Greedy decoding
- KV cache (naive, pre-allocated)
- Stop conditions (EOS, max length)

1.2 Model Loading ✅
- Load LLaMA weights (safetensors)
- LLaMA architecture (attention, FFN, RMSNorm)
- RoPE positional embeddings

1.3 Sampling ✅
- Temperature, top-p, top-k
- Basic CLI interface

---

## Phase 2: Paged Attention ✅

2.1 Block Manager ✅
- Block allocator (malloc/free for KV blocks)
- Free list management
- Block reference counting

2.2 Paged KV Cache ✅
- Non-contiguous block storage
- Logical → physical block mapping
- Allocate on demand, free when done

2.3 Paged Attention Kernel ✅
- Gather K,V from scattered blocks
- Compute attention with block tables

---

## Phase 3: Continuous Batching ✅

3.1 Request Queue ✅
- Add request (prompt + params)
- Request states (waiting, running, finished)

3.2 Scheduler ✅
- Dynamic batch composition
- Add new requests when slots free
- Remove finished requests immediately

3.3 Batch Execution ✅
- Mixed prefill + decode handling
- Update KV cache per sequence

---

## Phase 4: Block-based KVCache ✅

4.1 Block Tensors ✅
- Pre-allocated block tensors (tinygrad compatible)
- Tensor.stack() for gathering blocks

4.2 BlockManager Integration ✅
- Connect to Scheduler (can_allocate checks)
- Memory tracking and OOM prevention

4.3 Batched Forward Pass ✅
- Process multiple sequences in single forward

---

## Phase 5: Custom Metal Kernel ✅

5.1 Metal Kernel ✅
- Fused paged attention (gather + matmul in one kernel)
- Direct block addressing without copy
- tinygrad MetalProgram API

5.2 Kernel Integration ✅
- Auto-detect backend
- Fallback to portable Tensor.stack()

---

## Phase 6: Metal Kernel Optimizations ✅ (Partial)

6.1 SIMD Optimizations ✅
- 32-thread threadgroups (single simdgroup)
- SIMD shuffle reductions (simd_sum)
- simd_broadcast_first for exp()

6.2 Memory Access ✅
- Vectorized float4 loads
- FP16 support

6.3 Online Softmax ✅
- Single-pass attention (vs two-pass)
- Better memory locality

6.4 Attempted but Removed ❌
- Buffer pooling ❌ - tensors can't be recycled
- simdgroup_async_copy ❌ - API deprecated by Apple

---

## Phase 7: Critical Optimizations (10-20x potential)

7.1 TinyJit for Decode Loop ✅
**Impact:** 1.5-2x speedup | **Status:** Completed
- Applied @TinyJit decorator to JIT-compatible forward pass
- Pad batch to fixed max_batch_size (JIT requires fixed shapes)
- Uses pure tinygrad ops for attention (JIT-compatible)
- Benchmark results: 1.56-1.82x faster than non-JIT
- Files: `tinyvllm/engine/jit_decode.py`, `tinyvllm/kernels/paged_attention_jit.py`
- Enable with: `LLMEngine(..., use_jit=True)`

7.2 Weight Quantization INT8 ❌
**Impact:** Theoretical 2x, Actual 4x SLOWER | **Status:** Investigated, not viable with current approach

**Why it doesn't work:**
1. **Unfused approach (tinygrad ops):** `w_int8.cast(fp16) * scale` creates separate kernels for cast, multiply, then matmul. Extra memory round-trips make it 3x slower than FP16.

2. **Fused Metal kernel:** We implemented a custom kernel that loads INT8, dequantizes in registers, computes matmul. The kernel itself is 6x faster than FP16 in isolation.

3. **Python overhead kills it:** Custom kernels don't integrate with tinygrad's lazy evaluation. Each of 155 linear layers requires:
   - Input tensor realize: ~1.75ms
   - Output allocation: ~0.48ms
   - Buffer fetching: ~0.36ms
   - Kernel execution: ~0.23ms (fast!)
   - **Total: ~2.8ms/layer × 155 layers = 434ms/token**

4. **Result:** 0.30 tok/s INT8 vs 1.22 tok/s FP16 = 4x slower

**What would fix it:**
- Native tinygrad INT8 matmul support (fused with lazy graph)
- Or: Write entire model forward pass as single Metal kernel (impractical)

7.3 Multi-Step Scheduling ✅
**Impact:** Minimal with simple scheduler, future-proofs for complex scheduling | **Status:** Completed
- Added `num_scheduler_steps` parameter to LLMEngine (default=1)
- GPU runs N decode iterations per step() call, amortizing scheduler overhead
- After N steps, checks for finished sequences and new requests
- Benchmark results: With current simple scheduler, <5% improvement (scheduler not the bottleneck)
- Future value: Will help when we add priority scheduling, preemption, prefix caching, SLA-based scheduling
- Trade-off: higher TTFT at low load (N=1 for latency, N=4+ for throughput)
- Usage: `--num-scheduler-steps 4` or `LLMEngine(..., num_scheduler_steps=4)`

7.4 Reduce Python→GPU Copies ✅
**Impact:** ~15% speedup | **Status:** Completed
- Build block_tables and context_lens tensors ONCE per decode step (not 22x per layer)
- Added `decode_attention_with_tensors()` for direct tensor path
- Benchmark: 1.52 tok/s (optimized) vs 1.32 tok/s (baseline) = 15% improvement
- Files: `engine.py`, `llama.py`, `attention_utils.py`
- Note: Less than expected 2-3x because other bottlenecks dominate (kernel execution, weight reads)

---

## Phase 8: Performance Tuning (30-50% additional)

8.1 Pre-allocated Buffers
**Impact:** 10-20% | **Status:** Not started
- Pre-allocate decode input/output buffers at engine init
- Reuse instead of Tensor.zeros() each step
- Note: Works best combined with TinyJit (7.1)

8.2 Async Output Processing
**Impact:** 8-10% | **Status:** Not started
- Detokenize previous step while GPU computes current step ← overlap CPU/GPU
- Background thread with queue for pending outputs
- Return results via callback or poll

8.3 Object Pooling (CPU)
**Impact:** 10-15% | **Status:** Not started
- Pre-allocate Request, Sequence, SchedulerOutput objects
- Reuse instead of alloc/free each step ← vLLM saw 24% improvement
- Add .reset() method to clear state

8.4 Weight Quantization INT4
**Impact:** 2x more (4x total vs FP16) | **Status:** Not started
- Per-group scales (group_size=128 typical)
- GPTQ/AWQ style quantization
- Requires calibration dataset
- Slight quality loss (~1-3% perplexity)

8.5 Sampling Optimizations
**Impact:** 10-20% | **Status:** Not started
- Remove .realize().tolist() sync points ← easy, high impact
- Batched sampling across sequences (currently single token at a time)
- Top-p/top-k on GPU (currently converts to CPU list)
- Fast path for greedy sampling (direct argmax)

---

## Phase 9: Feature Additions

9.1 Flash Attention for Prefill
**Impact:** 2-4x prefill speedup | **Status:** Not started
**Prerequisite:** Online softmax ✅ (completed in 6.3)
- Tiled attention with online softmax for prefill phase
- Process Q, K, V in tiles (8x8 or 16x16 for M-series)
- Use threadgroup memory for tile storage
- O(1) memory regardless of sequence length (vs O(n²))
- Enables longer context windows (16K+)

9.2 API Server
**Impact:** Enables real usage | **Status:** Not started
- FastAPI or simple HTTP server
- POST /generate endpoint
- Server-sent events (SSE) for streaming
- Token-by-token streaming

9.3 OpenAI API Compatibility
**Impact:** Drop-in replacement | **Status:** Not started
- POST /v1/completions
- POST /v1/chat/completions
- Response format matching

9.4 Prefill Optimization
**Status:** Not started
- Chunked prefill (don't block on long prompts) vLLM does this - singel decode prefill loop
- Batched prefill (process multiple prefill sequences together) 

---

## Phase 10: Advanced Features

10.1 Speculative Decoding
**Impact:** 2-3x throughput | **Status:** Not started
- Draft model integration (small fast model proposes tokens)
- Parallel verification with main model
- Token acceptance/rejection logic
- Requires draft model weights

10.2 Prefix Caching
**Impact:** Variable (use-case dependent) | **Status:** Not started
- Hash prompt prefixes
- Reuse KV cache for common prefixes
- Cache eviction policy (LRU)
- Cache hit/miss tracking

10.3 KV Cache Quantization
**Impact:** 2-4x memory reduction | **Status:** Not started
- Support int8/float16 KV cache storage
- Dequantization in attention kernel
- Enables longer sequences with same memory

10.4 Kernel Fusion
**Impact:** 10-20% | **Status:** Not started
- RMSNorm + Linear projection
- RoPE + Q/K projection
- Verify what tinygrad auto-fuses with DEBUG=4

10.5 Generation Control
**Status:** Not started
- Stop string matching (beyond EOS token)
- Constrained decoding (JSON schema, grammar)
- Logit processors plugin system

10.6 Metrics & Monitoring
**Status:** Not started
- Per-step timing (prefill time, decode time, sample time)
- Memory profiling (used, free, fragmentation)
- Cache hit rate (for prefix caching)

---

## Phase 11: Platform Expansion

11.1 CUDA Kernel
**Impact:** NVIDIA platform support | **Status:** Not started
- Fused paged attention for CUDA
- Use tinygrad's CUDAProgram API
- CUDA graphs (capture and replay decode loop)

11.2 Multi-GPU
**Impact:** Large model scaling | **Status:** Not started
- Tensor parallelism (split layers across GPUs)
- Sequence parallelism (different sequences on different GPUs)
- Pipeline parallelism (different layers on different GPUs)

---

## Phase 12: Memory Management (Advanced)

12.1 LRU Eviction
**Status:** Not started
- Evict least recently used blocks when memory full

12.2 Memory Defragmentation
**Status:** Not started
- Compact blocks to prevent external fragmentation

12.3 CPU Swap
**Status:** Not started
- Move blocks to CPU when GPU full
- Swap back when needed

---

## Deferred/Not Recommended ❌

### simdgroup_async_copy ❌
**Status:** Research complete - NOT RECOMMENDED
- API deprecated/blocked by Apple (macOS 26 Beta 4 disabled __asm workaround)
- Causes undefined behavior with multi-simdgroup sync
- Alternatives: standard tiled attention, simd_matrix intrinsics

### Buffer Pooling ❌
**Status:** Removed
- Tensors returned to caller can't be recycled back to pool
- TinyJit is better solution (caches entire kernel graph)

### Tiled Attention for Decode ❌
**Status:** Deferred
- Decode is memory-bandwidth-bound, not compute-bound
- Minimal gain from tiling in decode phase
- Focus on prefill (9.1) where tiling helps

---

## Quick Reference

```
✅ COMPLETED (Phase 1-7.4):
   1: Foundation
   2: Paged Attention
   3: Continuous Batching
   4: Block-based KVCache
   5: Custom Metal Kernel
   6: Metal Kernel Optimizations (online softmax, SIMD)
   7.1: TinyJit for decode       → 1.5-2x ✅
   7.3: Multi-step scheduling    → <5% (future-proofs) ✅
   7.4: Reduce Python→GPU copies → 15% ✅

🎯 NEXT (Phase 8 - Performance Tuning):
   8.1-8.5: Buffers, async output, object pooling, INT4, sampling

📦 FEATURES (Phase 9):
   9.1: Flash Attention prefill
   9.2: API Server
   9.3: OpenAI compatibility

🔮 ADVANCED (Phase 10-12):
   10: Speculative decoding, prefix caching, KV quantization
   11: CUDA kernel, multi-GPU
   12: Memory management (eviction, defrag, swap)

❌ SKIP:
   simdgroup_async_copy (deprecated)
   Buffer pooling (doesn't work)
   7.2: INT8 quantization (Python overhead makes it 4x slower)
```

