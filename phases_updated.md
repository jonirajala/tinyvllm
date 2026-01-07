
# tinyvllm Implementation Roadmap (Priority-Ordered)

**Current Status:** 1.7 tok/s single / 3.6 tok/s batched vs 55 tok/s theoretical (6.6% utilization)
**Progress:** Phase 8.1 complete (Flash Attention: 1.4-2.2x prefill speedup)
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

8.1 Flash Attention for Prefill ✅
**Impact:** 1.4-2.2x prefill speedup | **Status:** Completed
**Prerequisite:** Online softmax ✅ (completed in 6.3)
- Tiled attention with online softmax for prefill phase
- Process Q, K, V in 8x8 tiles
- Custom Metal kernel with threadgroup memory for tile storage
- O(1) memory regardless of sequence length (vs O(n²))
- Enables longer context windows (16K+)
- **Benchmark results (isolated attention):**
  - 32 tokens: 1.59x speedup (2.76ms → 1.73ms)
  - 64 tokens: 1.46x speedup (2.79ms → 1.92ms)
  - 128 tokens: 1.39x speedup (2.74ms → 1.97ms)
  - 256 tokens: 2.22x speedup (4.16ms → 1.87ms)
  - 512 tokens: 1.62x speedup (3.07ms → 1.89ms)
- **Files:** `kernels/flash_attention_metal.py`, `kernels/flash_attention_tinygrad.py`, `attention_utils.py`
- **Features:** GQA support, causal masking, auto-selects Metal or tinygrad fallback

8.2 Pre-allocated Buffers
**Impact:** 10-20% | **Status:** Not started
- Pre-allocate decode input/output buffers at engine init
- Reuse instead of Tensor.zeros() each step
- Note: Works best combined with TinyJit (7.1)

8.3 Async Output Processing
**Impact:** 8-10% | **Status:** Not started
- Detokenize previous step while GPU computes current step ← overlap CPU/GPU
- Background thread with queue for pending outputs
- Return results via callback or poll

8.4 Object Pooling (CPU)
**Impact:** 10-15% | **Status:** Not started
- Pre-allocate Request, Sequence, SchedulerOutput objects
- Reuse instead of alloc/free each step ← vLLM saw 24% improvement
- Add .reset() method to clear state

8.5 Weight Quantization INT4 ❌
**Impact:** 2x more (4x total vs FP16) | **Status:** Won't work (same issue as INT8)
- Same problem as INT8: custom kernels can't integrate with tinygrad lazy graph
- Each layer needs realize() calls that kill performance
- Would need native tinygrad INT4 support
- Per-group scales (group_size=128 typical)
- GPTQ/AWQ style quantization

8.6 Sampling Optimizations
**Impact:** 5-10% remaining | **Status:** Partially done
- ✅ Top-k/top-p moved to CPU (faster than tinygrad's GPU topk which uses O(n log² n) bitonic sort)
- Greedy fast path could help slightly
- Batched sampling across sequences (currently single token at a time)
- Remove remaining .realize().tolist() sync points

8.7 Deferred KV Writes with Metal Kernel Support
**Impact:** 13x faster KV writes (47ms → 3.5ms) | **Status:** Requires Metal kernel update
- **Goal:** Reduce KV cache write overhead by batching writes across all 22 layers
- **Approach:** Defer K/V writes during forward pass, collect all layers' K/V, write once at end
- **Blocker:** Metal kernel doesn't support current_k/current_v params
  - Without it, falls back to tinygrad kernel (6.7x slower than Metal)
  - Net result with tinygrad fallback: 141ms slower per token (save 43ms writes, lose 185ms attention)
- **Solution:** Update Metal kernel to accept current_k/current_v buffers
  - Add extra buffer parameters to kernel
  - Concatenate current K/V with cached K/V in-kernel before attention
- **Prerequisite:** Unified KVCache tensor structure (already implemented, stashed)

8.8 Reduce Tinygrad Operations per Forward Pass
**Impact:** Potentially significant | **Status:** Not started
- Each tinygrad tensor operation builds a lazy graph node
- More ops = more graph overhead, even if kernels are fast
- Investigate fusing multiple ops into single custom kernels
- Profile with DEBUG=4 to see what tinygrad auto-fuses
- Consider rewriting hot paths as single Metal kernels (like attention)
- Trade-off: Custom kernels lose tinygrad's lazy evaluation benefits

8.9 Decode Kernel: Block-wise Parallelization
**Impact:** Large for long contexts | **Status:** Not started
- Current decode kernel has O(ctx_len) sequential loop per head
- Each head processes tokens one at a time (memory bandwidth bound)
- **Goal:** Parallelize across tokens with block-wise reductions
- **Approach:**
  - Split context into blocks (e.g., 64 tokens each)
  - Each threadgroup processes one block, computes partial softmax
  - Final reduction across blocks to combine partial results
  - Similar to vLLM's PagedAttention V2 algorithm
- **Complexity:** High - requires partial softmax accumulation across blocks
- **Prerequisite:** Verify tinygrad JIT doesn't already beat this

8.10 Decode Kernel: K/V Prefetching & Unrolling
**Impact:** Medium | **Status:** Not started
- Current kernel loads one K/V position per iteration
- Apple GPUs benefit from coalesced memory access patterns
- **Goal:** Process small window of positions per iteration
- **Approach:**
  - Unroll inner loop to process 2-4 tokens at once
  - Use threadgroup memory to stage K/V tiles
  - Prefetch next positions while computing current
- **Trade-off:** More register pressure vs better memory throughput

8.11 Prefill Kernel: Double-Buffering
**Impact:** Medium | **Status:** Not started
- Current prefill kernel: load tile → barrier → compute → barrier
- Memory latency not hidden during compute phase
- **Goal:** Overlap loading next tile while computing current tile
- **Approach:**
  - Allocate two K/V tile buffers (K0/V0, K1/V1)
  - While computing on buffer 0, load next tile into buffer 1
  - Swap buffers each iteration (software pipelining)
- **Trade-off:** 2x threadgroup memory usage
- **Note:** TILE_KV=16 already hurt performance, so memory pressure is a concern

---

## Phase 9: Feature Additions

9.1 API Server
**Impact:** Enables real usage | **Status:** Not started
- FastAPI or simple HTTP server
- POST /generate endpoint
- Server-sent events (SSE) for streaming
- Token-by-token streaming

9.2 OpenAI API Compatibility
**Impact:** Drop-in replacement | **Status:** Not started
- POST /v1/completions
- POST /v1/chat/completions
- Response format matching

9.3 Prefill Optimization
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
✅ COMPLETED (Phase 1-8.1):
   1: Foundation
   2: Paged Attention
   3: Continuous Batching
   4: Block-based KVCache
   5: Custom Metal Kernel
   6: Metal Kernel Optimizations (online softmax, SIMD)
   7.1: TinyJit for decode       → 1.5-2x ✅
   7.3: Multi-step scheduling    → <5% (future-proofs) ✅
   7.4: Reduce Python→GPU copies → 15% ✅
   8.1: Flash Attention prefill  → 1.4-2.2x ✅

🎯 NEXT (Phase 8 - Performance Tuning):
   8.2-8.6: Buffers, async output, object pooling, INT4, sampling
   8.7: Deferred KV writes (needs Metal kernel update)
   8.8: Reduce tinygrad ops per forward pass

📦 FEATURES (Phase 9):
   9.1: API Server
   9.2: OpenAI compatibility

🔮 ADVANCED (Phase 10-12):
   10: Speculative decoding, prefix caching, KV quantization
   11: CUDA kernel, multi-GPU
   12: Memory management (eviction, defrag, swap)

❌ SKIP:
   simdgroup_async_copy (deprecated)
   Buffer pooling (doesn't work)
   7.2: INT8 quantization (Python overhead makes it 4x slower)
```

