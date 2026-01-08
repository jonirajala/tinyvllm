
# tinyvllm Implementation Roadmap (Priority-Ordered)

**Current Status:** ~4 tok/s batched with TinyJit optimization
**Progress:** Phase 8.3 complete + JIT decode working (4x speedup)
**Architecture:** Pure tinygrad with TinyJit compilation (no custom kernels)

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

2.3 Paged Attention (tinygrad) ✅
- Gather K,V from scattered blocks using tinygrad ops
- Compute attention with block tables
- No custom kernels needed

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
- Tensor indexing for gathering blocks

4.2 BlockManager Integration ✅
- Connect to Scheduler (can_allocate checks)
- Memory tracking and OOM prevention

4.3 Batched Forward Pass ✅
- Process multiple sequences in single forward

---

## Phase 5: Custom Metal Kernels ❌ REMOVED

**Status:** Removed in favor of tinygrad + TinyJit approach

**History:**
- Originally implemented fused paged attention Metal kernels
- Custom kernels achieved good standalone performance
- However, TinyJit + pure tinygrad ops proved faster overall:
  - Better integration with tinygrad's lazy evaluation
  - No Python↔Metal synchronization overhead
  - JIT compiles entire decode loop into single optimized kernel

**Deleted files:**
- `paged_decode_attention_metal.py`
- `flash_prefill_attention_metal.py`
- Entire `tinyvllm/kernels/` folder

**Current approach:** Pure tinygrad attention ops compiled via TinyJit

---

## Phase 6: Metal Kernel Optimizations ❌ SUPERSEDED

**Status:** No longer applicable - custom kernels removed

These optimizations (SIMD, online softmax, etc.) were for the custom Metal kernels.
With TinyJit approach, tinygrad handles low-level optimization automatically.

---

## Phase 7: Critical Optimizations

7.1 TinyJit for Decode Loop ✅ WORKING
**Impact:** 4x speedup | **Status:** Completed

**Implementation:**
- `@TinyJit` decorator compiles entire decode forward pass
- JIT function created per-engine (not cached on model) for determinism
- Factory pattern: `model.create_jit_decode()` returns fresh JIT function
- Engine stores and reuses its own JIT instance

**Key insight:** JIT must be per-engine, not per-model:
- Same model reused across multiple engines needs independent JIT state
- Otherwise, different engines interfere with each other's cached graphs
- Solution: `LLMEngine.__init__` creates its own `_jit_decode_fn`

**Files:** `tinyvllm/model/llama.py`, `tinyvllm/core/engine.py`

7.2 Weight Quantization INT8 ❌
**Impact:** Theoretical 2x, Actual 4x SLOWER | **Status:** Not viable

**Why it doesn't work:**
1. Custom kernels can't integrate with tinygrad lazy graph
2. Each layer needs realize() calls that kill performance
3. Would need native tinygrad INT8 support

7.3 Multi-Step Scheduling ✅
**Impact:** Minimal with simple scheduler, future-proofs for complex scheduling | **Status:** Completed
- Added `num_scheduler_steps` parameter to LLMEngine (default=1)
- GPU runs N decode iterations per step() call, amortizing scheduler overhead
- Usage: `--num-scheduler-steps 4` or `LLMEngine(..., num_scheduler_steps=4)`

7.4 Reduce Python→GPU Copies ✅
**Impact:** ~15% speedup | **Status:** Completed
- Build block_tables and context_lens tensors ONCE per decode step (not per layer)
- Tensors passed directly to attention functions

---

## Phase 8: Performance Tuning

8.1 Flash Attention for Prefill ✅
**Impact:** 1.4-2.2x prefill speedup | **Status:** Completed
- Tiled attention with online softmax for prefill phase
- Pure tinygrad implementation (no custom Metal kernel)
- Process Q, K, V in tiles for memory efficiency
- GQA support, causal masking
- **Files:** `tinyvllm/model/llama.py` (integrated directly)

8.2 Pre-allocated Buffers
**Impact:** 10-20% | **Status:** Not started
- Pre-allocate decode input/output buffers at engine init
- Reuse instead of Tensor.zeros() each step

8.3 Async Output Processing ✅
**Impact:** ~3% measured | **Status:** Completed
- Detokenize previous step while GPU computes current step
- Background thread with queue for pending outputs
- **Files:** `tinyvllm/core/output_processor.py`, `tinyvllm/core/engine.py`
- **Usage:** `LLMEngine(..., async_output=True)`

8.4 Object Pooling (CPU)
**Impact:** 10-15% | **Status:** Not started
- Pre-allocate Request, Sequence, SchedulerOutput objects
- Reuse instead of alloc/free each step

8.5 Weight Quantization INT4 ❌
**Impact:** Not viable | **Status:** Same issue as INT8
- Custom kernels can't integrate with tinygrad lazy graph
- Would need native tinygrad INT4 support

8.6 Sampling Optimizations
**Impact:** 5-10% remaining | **Status:** Partially done
- ✅ Top-k/top-p moved to CPU (faster than tinygrad's GPU topk)
- Batched sampling across sequences could help slightly

---

## Phase 9: Feature Additions

9.1 API Server
**Impact:** Enables real usage | **Status:** Not started
- FastAPI or simple HTTP server
- POST /generate endpoint
- Server-sent events (SSE) for streaming

9.2 OpenAI API Compatibility
**Impact:** Drop-in replacement | **Status:** Not started
- POST /v1/completions
- POST /v1/chat/completions

9.3 Prefill Optimization
**Status:** Not started
- Chunked prefill (don't block on long prompts)
- Batched prefill (process multiple prefill sequences together)

---

## Phase 10: Advanced Features

10.1 Speculative Decoding
**Impact:** 2-3x throughput | **Status:** Not started
- Draft model integration
- Parallel verification with main model

10.2 Prefix Caching
**Impact:** Variable | **Status:** Not started
- Hash prompt prefixes
- Reuse KV cache for common prefixes

10.3 KV Cache Quantization
**Impact:** 2-4x memory reduction | **Status:** Not started
- Support int8/float16 KV cache storage
- Enables longer sequences

10.4 Kernel Fusion
**Impact:** 10-20% | **Status:** Not started
- RMSNorm + Linear projection
- RoPE + Q/K projection
- Verify what tinygrad auto-fuses with DEBUG=4

---

## Phase 11: Platform Expansion

11.1 CUDA Support
**Impact:** NVIDIA platform | **Status:** Not started
- tinygrad already supports CUDA backend
- May need CUDA-specific optimizations

11.2 Multi-GPU
**Impact:** Large model scaling | **Status:** Not started
- Tensor parallelism
- Pipeline parallelism

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

---

## Deferred/Not Recommended ❌

### Custom Metal Kernels ❌
**Status:** Removed
- TinyJit + pure tinygrad proved faster
- Better integration with lazy evaluation
- Less Python↔Metal synchronization overhead

### Weight Quantization (INT8/INT4) ❌
**Status:** Not viable
- Custom kernels can't integrate with tinygrad lazy graph
- Would need native tinygrad quantization support

### Buffer Pooling ❌
**Status:** Removed
- Tensors returned to caller can't be recycled
- Tinygrad's lazy evaluation handles caching

---

## Quick Reference

```
✅ COMPLETED:
   Phase 1: Foundation
   Phase 2: Paged Attention (tinygrad ops)
   Phase 3: Continuous Batching
   Phase 4: Block-based KVCache
   Phase 7.1: TinyJit decode loop     → 4x speedup ✅
   Phase 7.3: Multi-step scheduling   → future-proofs ✅
   Phase 7.4: Reduce Python→GPU copies → 15% ✅
   Phase 8.1: Flash Attention prefill → 1.4-2.2x ✅
   Phase 8.3: Async output processing → ~3% ✅

🎯 NEXT PRIORITIES:
   8.2: Pre-allocated buffers
   8.4: Object pooling
   8.6: Sampling optimizations
   9.1: API Server

📦 FEATURES (Phase 9):
   9.1: API Server
   9.2: OpenAI compatibility
   9.3: Prefill optimization

🔮 ADVANCED (Phase 10-12):
   10: Speculative decoding, prefix caching, KV quantization
   11: CUDA support, multi-GPU
   12: Memory management (eviction, defrag, swap)

❌ REMOVED/SKIPPED:
   Phase 5-6: Custom Metal kernels (replaced by TinyJit)
   7.2/8.5: INT8/INT4 quantization (not viable)
   Buffer pooling (doesn't work with tinygrad)
```

## Code Structure

```
tinyvllm/
├── core/           # Engine, scheduler, KV cache, sampling
│   ├── engine.py   # LLMEngine with JIT decode
│   ├── scheduler.py
│   ├── kv_cache.py
│   ├── block_manager.py
│   ├── sampling.py
│   ├── sequence.py
│   └── output_processor.py
├── model/          # LLaMA model with attention
│   ├── llama.py    # Model + paged/flash attention
│   ├── weights.py
│   └── tokenizer.py
└── main.py         # CLI entry point
```
