Implementation Plans for TinyVLLM Optimizations

## Progress Tracking

### Plan 1: Batch KV Cache Writes - ✅ IMPLEMENTED
**Date**: 2025-01-07
**Result**: Partial improvement
- __setitem__ calls: 1738 → 1320 (-24%)
- Scheduling overhead: 6607ms → 6280ms (-5%)
- Total kernel launches: 7766 → 7348 (-5.4%)
- Time saved: ~367ms

**Analysis**: Batching helps for prefill (multi-token), but decode still does
1 token per layer × 22 layers × 2 (K+V) = 44 setitem ops per decode step.
The main benefit is for prefill phase. Decode still dominates.

### Plan 2: Fuse RMSNorm Operations - ❌ ABANDONED (Regression)
**Date**: 2025-01-07
**Result**: Custom kernel is 2x SLOWER than tinygrad ops
- WITH fused Metal kernel: 23268 ms (0.9 tok/s)
- WITHOUT fused kernel: 12058 ms (1.7 tok/s)

**Root Cause**: Custom Metal kernel launch overhead exceeds the benefit of fusing ops.
Tinygrad already fuses simple element-wise ops (pow, mean, rsqrt, mul) efficiently.

**Lesson Learned**: Custom kernels only help when:
1. The operation is complex enough that kernel time dominates launch overhead
2. The kernel can batch many operations together (like paged attention)
3. Tinygrad cannot fuse the operations efficiently

**Status**: All changes reverted. Plan 3 (SwiGLU) likely has same issue - skipping.

### Plan 3: Fuse SwiGLU Activation - ⏭️ SKIPPED
Same kernel launch overhead issue as Plan 2. Skipped.

### Plan 4: JIT/Graph Caching - ⏭️ SKIPPED
**Issue**: TinyJit doesn't work with custom Metal kernels - they bypass tinygrad's graph.
Custom kernels run via Metal APIs directly, not through tinygrad's lazy execution.
Would need to choose: pure tinygrad + JIT OR custom kernels without JIT.

### Plan 5: Direct Metal Buffer Writes for KV Cache - ❌ ABANDONED
**Date**: 2025-01-07
**Issue**: Custom kernels call `.realize()` which forces scheduling of pending lazy ops.
When KV cache uses `__setitem__`, it creates lazy ops. Custom kernel's `realize()` then
triggers scheduling of these writes, adding ~7ms overhead per kernel call.

**Attempted Solutions**:

**Attempt 1: Sync per write** - FAILED
- Bypass scheduler by writing directly to Metal buffer contents
- Problem: Requires `synchronize()` before each buffer access
- Result: 2% slower than standard writes (sync overhead negates benefit)

**Attempt 2: Sync once, copy all (sync-once approach)** - PARTIALLY WORKED
- Realize all K/V tensors, sync ONCE, then do all direct copies
- Isolated benchmark: **53x faster** (2180ms → 41ms for 1100 writes)
- But model structure requires sync per layer (22 layers = 22 syncs)
- End-to-end result: Still slower than lazy writes (670ms vs 632ms TPOT)

**Root Cause Analysis**:
The transformer architecture processes layers sequentially. Each layer:
1. Computes new K/V tensors
2. Writes to KV cache
3. Runs attention
4. Proceeds to next layer

For sync-once to work, we'd need to batch ALL layer operations and sync once
at the end. But that requires restructuring the entire forward pass, which is
impractical.

**Key Insight**: Tinygrad's lazy evaluation handles the scheduling efficiently.
The "overhead" we measured was actually correct synchronization that ensures
data consistency. Attempting to bypass it just shifts the sync elsewhere.

**Lesson Learned**: Direct buffer writes only help when:
1. You can batch writes across ALL layers with a single sync (53x speedup)
2. But this requires major architectural changes to batch layer processing
3. Per-layer or per-write sync negates any benefit

**Status**: Abandoned. Standard lazy writes are optimal for current architecture.

### Plan 6: TinyJit for Decode Forward - ✅ IMPLEMENTED (4x Speedup!)
**Date**: 2026-01-07

**Problem Identified**: Profiling with DEBUG=2 revealed:
- Total kernel time: 2612ms (27.4%)
- **Scheduling overhead: 6920ms (72.6%)** ← This is the real bottleneck!
- 1559 scheduling batches at 4.44ms average each
- Kernel breakdown: linear 52%, silu 21%, __add__ 8%, __setitem__ 5%

**Root Cause**: Each `.realize()` call triggers tinygrad's scheduler which:
1. Builds computation graph
2. Linearizes operations
3. Allocates buffers
4. Launches kernels

With 1559 scheduling batches for 10 tokens, that's ~150 schedule calls per token!

**Solution: TinyJit**
TinyJit caches the compiled graph and reuses it, eliminating scheduling overhead.

**Test Results**:
1. **Simple matmul**: JIT 0.07ms vs Non-JIT 0.61ms → **8.7x speedup**
2. **FFN (22 layers)**: JIT 37.8ms vs Non-JIT 78.7ms → **2.1x speedup**
3. **Full decode (no KV writes)**: JIT 48.5ms vs Non-JIT 426ms → **8.8x speedup**
4. **Full decode (with KV writes)**: JIT 131ms vs Non-JIT 700ms → **5.3x speedup**

**Key Challenge**: KV cache writes use `__setitem__` which:
- Cannot be JITted (forces immediate realize)
- Takes ~30-120ms per decode step
- Becomes the new bottleneck after JIT is applied

**Implementation Plan**:
1. Create JIT-compatible decode function:
   - Use pure tinygrad paged attention (not custom Metal kernel)
   - Separate KV writes from compute graph
   - JIT: embedding → 22 layers (attention + FFN) → output
   - Non-JIT: KV cache writes after JIT returns

2. Code structure:
```python
@TinyJit
def jit_decode_forward(token, cos, sin, block_tables, context_lens):
    h = model.tok_embeddings(token)
    k_outputs, v_outputs = [], []
    for layer in model.layers:
        # attention + FFN (pure tinygrad ops)
        ...
        k_outputs.append(k)
        v_outputs.append(v)
    return model.output(model.norm(h)), k_outputs, v_outputs

def decode_step(token, pos):
    logits, k_all, v_all = jit_decode_forward(token, cos, sin, ...)
    # Write K/V outside JIT
    for layer_idx in range(n_layers):
        kv_cache.write_kv_batch(layer_idx, block_id, offset, k_all[layer_idx], v_all[layer_idx])
    return logits
```

**Expected Impact**: 4-5x speedup (700ms → 140-180ms per decode step)

**Trade-offs**:
- Must use pure tinygrad attention, not custom Metal kernel
- KV writes remain a bottleneck (~30-50ms per step)
- First decode step has JIT compilation overhead

**FINAL RESULTS (Implemented)**:
- **4.06x speedup achieved!**
- JIT: 8178ms for 50 tokens = **6.11 tok/s**
- Non-JIT: 33218ms for 50 tokens = **1.51 tok/s**

**Implementation Details**:
- Added `decode_jit()` method to Llama class (`llama.py`)
- Added `_create_jit_decode_forward()` to create cached JIT function
- Updated `LLMEngine` with `use_jit=True` flag (default enabled)
- Block tables padded to fixed size (64 blocks) for JIT compatibility

**Files Modified**:
- `tinyvllm/model/llama.py`: Added JIT decode methods
- `tinyvllm/engine/engine.py`: Added use_jit parameter

**Status**: ✅ IMPLEMENTED AND WORKING

---

  Plan 1: Batch KV Cache Writes (HIGH PRIORITY - Biggest Impact)

  Problem: _write_kv_to_blocks() loops token-by-token, creating N kernel launches per layer (1738 __setitem__ calls observed).

  Current Code (llama.py:160-176):
  for i in range(seq_len):
      pos = start_pos + i
      block_idx = pos // block_size
      offset = pos % block_size
      ...
      kv_cache.write_kv(layer_idx, block_id, offset, k[i], v[i])

  Implementation Steps:

  1. Group tokens by block - Partition tokens that fall in the same block
    - File: tinyvllm/model/llama.py
    - Replace loop with block-aware batching
  2. Use slice assignment for contiguous tokens
    - If tokens span offset 0 to N within a block, use single k_cache[block_id, 0:N] = k_batch
    - File: tinyvllm/core/kv_cache.py - already has write_kv_batch() but unused
  3. Handle block boundaries
    - When tokens span multiple blocks, batch within each block separately
    - Only 1-2 kernel launches per block instead of per token

  New Code Structure:
  def _write_kv_to_blocks_batched(self, kv_cache, block_manager, layer_idx, seq_id, k, v, start_pos):
      seq_len = k.shape[0]
      block_size = block_manager.block_size
      block_table = block_manager.get_block_table(seq_id)

      # Group by block
      pos = start_pos
      token_idx = 0
      while token_idx < seq_len:
          block_idx = pos // block_size
          offset = pos % block_size
          # How many tokens fit in this block?
          tokens_in_block = min(block_size - offset, seq_len - token_idx)

          # Allocate block if needed (layer 0 only)
          if block_idx >= len(block_table) and layer_idx == 0:
              # ... allocation logic (unchanged)

          block_id = block_table[block_idx]
          # Batch write!
          kv_cache.write_kv_batch(layer_idx, block_id, offset,
                                  k[token_idx:token_idx+tokens_in_block],
                                  v[token_idx:token_idx+tokens_in_block])

          token_idx += tokens_in_block
          pos += tokens_in_block

  Expected Impact: Reduce __setitem__ calls from ~1738 to ~100-200 (depending on block size). Should cut scheduling overhead by 50%+.

  Files to Modify:
  - tinyvllm/model/llama.py: _write_kv_to_blocks() → _write_kv_to_blocks_batched()

  ---
  Plan 2: Fuse RMSNorm Operations (MEDIUM PRIORITY)

  Problem: RMSNorm creates 3 separate kernels: pow → mean → rsqrt → mul (900 rsqrt calls, 900 mul calls)

  Current Code (llama.py:23-26):
  def __call__(self, x: Tensor) -> Tensor:
      rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
      return x * rms * self.weight

  Implementation Steps:

  1. Create fused RMSNorm kernel for Metal
    - File: tinyvllm/kernels/rmsnorm_metal.py (new)
    - Single kernel: compute RMS and normalize in one pass
  2. Kernel implementation:
  kernel void rmsnorm_fused(
      device const float* input,
      device const float* weight,
      device float* output,
      constant int& dim,
      constant float& eps,
      uint tid [[thread_position_in_grid]]
  ) {
      // Each thread handles one row
      int row = tid;
      float sum_sq = 0.0f;
      for (int i = 0; i < dim; i++) {
          float val = input[row * dim + i];
          sum_sq += val * val;
      }
      float rms = rsqrt(sum_sq / dim + eps);
      for (int i = 0; i < dim; i++) {
          output[row * dim + i] = input[row * dim + i] * rms * weight[i];
      }
  }

  3. Integration:
    - Modify RMSNorm.__call__() to use fused kernel on Metal
    - Fallback to tinygrad ops on other devices

  Expected Impact: Reduce kernel count by ~1800 (900 rsqrt + 900 mul → 44 fused). ~3% GPU time savings plus scheduling overhead reduction.

  Files to Create/Modify:
  - tinyvllm/kernels/rmsnorm_metal.py (new)
  - tinyvllm/kernels/__init__.py: add export
  - tinyvllm/model/llama.py: update RMSNorm

  ---
  Plan 3: Fuse SwiGLU Activation (MEDIUM PRIORITY)

  Problem: SwiGLU does w2(silu(w1(x)) * w3(x)) with separate kernels for silu and element-wise multiply. (440 silu calls = 19% of GPU time)

  Current Code (llama.py:187-189):
  def __call__(self, x: Tensor) -> Tensor:
      return self.w2(self.w1(x).silu() * self.w3(x))

  Implementation Steps:

  1. Fuse w1+silu+w3+mul into single operation
    - Can't easily fuse the matmuls, but can fuse silu(a) * b
  2. Create fused silu_mul kernel:
    - File: tinyvllm/kernels/silu_mul_metal.py (new)

  kernel void silu_mul_fused(
      device const float* a,  // w1(x) output
      device const float* b,  // w3(x) output
      device float* out,
      uint tid [[thread_position_in_grid]]
  ) {
      float val_a = a[tid];
      float silu_a = val_a / (1.0f + exp(-val_a));  // silu = x * sigmoid(x)
      out[tid] = silu_a * b[tid];
  }

  3. Modify FeedForward:
  def __call__(self, x: Tensor) -> Tensor:
      gate = self.w1(x)
      up = self.w3(x)
      # Fused silu_mul instead of gate.silu() * up
      hidden = silu_mul_fused(gate, up)
      return self.w2(hidden)

  Expected Impact: Reduce silu+mul to single kernel. ~5% GPU time savings.

  Files to Create/Modify:
  - tinyvllm/kernels/silu_mul_metal.py (new)
  - tinyvllm/kernels/__init__.py: add export
  - tinyvllm/model/llama.py: update FeedForward

  ---
  Plan 4: Reduce Graph Compilation Overhead (HIGH PRIORITY)

  Problem: 3.67ms average scheduling time per batch. Tinygrad recompiles the graph repeatedly.

  Implementation Steps:

  1. Use @TinyJit decorator for hot paths
    - Tinygrad's JIT caches compiled kernels
  2. Wrap decode step:
  from tinygrad import TinyJit

  class Llama:
      @TinyJit
      def _decode_step_jit(self, h, cos, sin, kv_cache_k, kv_cache_v, ...):
          # Core decode computation that gets JIT-compiled
          for layer_idx, layer in enumerate(self.layers):
              h = layer(h, cos, sin, ...)
          return self.output(self.norm(h))

  3. Challenges:
    - JIT requires static shapes - decode is batch=variable but seq=1
    - KV cache updates may break JIT (dynamic indexing)
    - May need to restructure to separate "JIT-able" and "dynamic" parts
  4. Alternative: Manual kernel caching:
    - Cache compiled Metal programs in paged_decode_attention_metal.py
    - Already partially done with PagedAttentionOnline.get_instance()
    - Extend to other kernels

  Expected Impact: Could reduce scheduling overhead from 3.67ms to <1ms per batch (2-3x speedup).

  Files to Modify:
  - tinyvllm/model/llama.py: add JIT decorators
  - tinyvllm/engine/engine.py: restructure for JIT compatibility

  ---
  Plan 5: Create Decode-Only Benchmark (LOW PRIORITY - Diagnostic)

  Problem: Current profile includes model loading and prefill. Need isolated decode metrics.

  Implementation Steps:

  1. Create benchmark script:
    - File: benchmarks/bench_decode.py (new)

  """Benchmark decode-only performance."""
  import time
  from tinygrad import Tensor, Device

  def benchmark_decode(engine, num_tokens=100, warmup=10):
      # Add a prompt and prefill
      engine.add_request("Hello world", SamplingParams(max_tokens=num_tokens))

      # Warmup
      for _ in range(warmup):
          engine.step()

      # Benchmark decode steps
      Device[Device.DEFAULT].synchronize()
      start = time.perf_counter()

      tokens_generated = 0
      while engine.has_unfinished():
          outputs = engine.step()
          tokens_generated += len(outputs) if outputs else 1

      Device[Device.DEFAULT].synchronize()
      elapsed = time.perf_counter() - start

      print(f"Tokens: {tokens_generated}")
      print(f"Time: {elapsed*1000:.2f} ms")
      print(f"Tokens/sec: {tokens_generated/elapsed:.2f}")
      print(f"ms/token: {elapsed*1000/tokens_generated:.2f}")

  2. Add per-step timing:
    - Instrument engine.step() to report prefill vs decode time
    - Track kernel count per step

  Files to Create:
  - benchmarks/bench_decode.py (new)
  - benchmarks/__init__.py (new)

  ---
  Priority Order

  | Priority | Optimization          | Expected Impact           | Effort |
  |----------|-----------------------|---------------------------|--------|
  | 1        | Batch KV Cache Writes | 50%+ scheduling reduction | Low    |
  | 2        | JIT/Graph Caching     | 2-3x scheduling speedup   | Medium |
  | 3        | Fuse SwiGLU           | 5-10% GPU time            | Medium |
  | 4        | Fuse RMSNorm          | 3-5% GPU time             | Medium |
  | 5        | Decode Benchmark      | Diagnostic only           | Low    |

  Recommendation: Start with Plan 1 (Batch KV Cache Writes) as it's the lowest effort with highest impact. The current token-by-token loop is the primary source of kernel explosion.

  ---
