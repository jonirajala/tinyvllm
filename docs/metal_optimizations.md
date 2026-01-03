# Metal Kernel Optimization Research

Phase 6 optimization techniques for Apple Silicon Metal kernels.
Based on research from Metal FlashAttention, MetalQwen3, Apple WWDC, and academic papers.

## Sources

- [Metal FlashAttention](https://github.com/philipturner/metal-flash-attention) - FlashAttention port to Metal
- [MetalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) - Full transformer on Metal
- [Apple Metal Best Practices](https://developer.apple.com/videos/play/tech-talks/111373/) - WWDC tech talk
- [ThunderMittens for MLX](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx) - Hazy Research

---

## 1. SIMD and Threadgroup Optimization

### SIMD Width
- Apple GPUs have **32 threads per simdgroup** (SIMD width = 32)
- Query at runtime: `threads_per_simdgroup` in shader or `threadExecutionWidth` in API

### Threadgroup Sizing Strategy
```
CURRENT (suboptimal):
  global_size=(n_heads, batch_size, 1)
  local_size=(1, 1, 1)  ← only 1 thread per threadgroup!

OPTIMIZED:
  global_size=(n_heads * 32, batch_size, 1)
  local_size=(32, 1, 1)  ← full simdgroup
```

### Why 32 Threads Matters
- **Single simdgroup = no threadgroup barriers needed**
- `threadgroup_barrier()` is expensive, `simdgroup_barrier()` is fast
- If threadgroup fits in one SIMD, use simdgroup operations exclusively

### SIMD Shuffle Operations
Instead of threadgroup memory for reductions:
```metal
// SLOW: threadgroup memory
threadgroup float shared[32];
shared[tid] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
// ... reduction loop

// FAST: SIMD shuffle
float sum = simd_sum(value);  // Hardware reduction
```

Available SIMD functions:
- `simd_sum()`, `simd_max()`, `simd_min()` - reductions
- `simd_shuffle()`, `simd_shuffle_xor()` - data exchange
- `simd_broadcast()` - broadcast from one lane

---

## 2. Memory Access Patterns

### 16-Byte Alignment
- Align all threadgroup memory allocations to 16 bytes
- Align memory accesses to 16 bytes for coalesced access
- Use `float4` (16 bytes) instead of 4x `float` (4 bytes each)

### Vectorized Loads
```metal
// SLOW: scalar loads
float a = data[idx];
float b = data[idx + 1];
float c = data[idx + 2];
float d = data[idx + 3];

// FAST: vector load
float4 vec = *((device float4*)(data + idx));
```

### simdgroup_async_copy (A14+ / M1+)
Overlaps memory loads with compute - undocumented but powerful:
```metal
// Async copy from device to threadgroup memory
simdgroup_async_copy(threadgroup_ptr, device_ptr, num_elements);
simdgroup_async_copy_wait();  // Wait for completion
```

### Unified Memory Advantage
- Apple Silicon shares memory between CPU and GPU
- No explicit CPU↔GPU transfers needed
- Can keep buffers persistent across kernel calls

---

## 3. Precision and Register Optimization

### Use half Precision
```metal
// SLOW: 32-bit (uses 2x registers)
float value = input[idx];
float result = value * scale;

// FAST: 16-bit (half the registers)
half value = input[idx];
half result = value * scale;
```

Benefits:
- 2x register capacity → higher occupancy
- Faster arithmetic instructions
- Sufficient precision for attention (< 1% quality loss)

### Register Pressure Management
- M-series: ~128 registers per thread max
- With 32 threads/simdgroup: 4096 registers per simdgroup
- **Intentional spilling**: Sometimes better to spill to memory with optimized access patterns than reduce parallelism

### Occupancy Tuning
```metal
// Control via max threads per threadgroup
[[max_total_threads_per_threadgroup(256)]]
kernel void attention_kernel(...) { }
```

Trade-off: More threads = more latency hiding, but more register pressure.

---

## 4. Tiled Attention (Flash Attention Style)

### Block Tiling Strategy
Metal FlashAttention uses 3D blocking:
- **Parallelization dimension**: 16-32 (across heads/batches)
- **Traversal dimension**: 80-128 (along sequence)
- **Head dimension**: varies (64, 128)

### Tile Sizes for Apple Silicon
From ThunderMittens research:
- **8×8 base tiles** optimal for M-series (128 register limit)
- Larger tiles (16×16) need careful register management

### Two-Pass Softmax (Current Implementation)
```metal
// Pass 1: Find max
for (pos = 0; pos < context_len; pos++) {
    score = dot(Q, K[pos]);
    max_score = max(max_score, score);
}

// Pass 2: Softmax + weighted sum
for (pos = 0; pos < context_len; pos++) {
    score = dot(Q, K[pos]);
    weight = exp(score - max_score);
    sum += weight;
    output += weight * V[pos];
}
```

### Online Softmax (Better)
Single pass with running statistics:
```metal
float max_score = -INFINITY;
float sum_exp = 0.0;
float4 output = 0.0;

for (pos = 0; pos < context_len; pos++) {
    float score = dot(Q, K[pos]);
    float new_max = max(max_score, score);

    // Rescale previous accumulations
    float scale = exp(max_score - new_max);
    sum_exp = sum_exp * scale + exp(score - new_max);
    output = output * scale + exp(score - new_max) * V[pos];

    max_score = new_max;
}
output /= sum_exp;
```

---

## 5. Buffer and Command Management

### Buffer Pooling
```cpp
// BAD: Allocate every call
MTLBuffer* buffer = [device newBufferWithLength:size options:...];

// GOOD: Reuse from pool
MTLBuffer* buffer = bufferPool.acquire(size);
// ... use buffer ...
bufferPool.release(buffer);
```

### Command Buffer Batching
```cpp
// BAD: One command buffer per operation
for (op in operations) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    // encode op
    [cmd commit];
    [cmd waitUntilCompleted];  // Sync!
}

// GOOD: Batch operations
id<MTLCommandBuffer> cmd = [queue commandBuffer];
for (op in operations) {
    // encode all ops
}
[cmd commit];
[cmd waitUntilCompleted];  // Single sync
```

### Persistent Buffers
For KV cache that persists across tokens:
```cpp
// Allocate once at init
k_cache_buffer = [device newBufferWithLength:...];
v_cache_buffer = [device newBufferWithLength:...];

// Reuse across all forward passes (no reallocation)
```

---

## 6. Quantization (Future)

### INT8 Quantization (Q8_0)
MetalQwen3 approach:
- 64-element quantization groups (matches SIMD width well)
- Per-group scale factor
- < 1% quality loss vs FP32

```metal
struct Q8_0Block {
    half scale;           // Scale factor
    char elements[64];    // Quantized values
};

// Dequantize on the fly
float dequantize(Q8_0Block block, int idx) {
    return float(block.scale) * float(block.elements[idx]);
}
```

---

## 7. Current Kernel Analysis (M4 10-Core)

Analysis of tinyvllm's Phase 5 Metal kernel against Apple M4 10-core GPU capabilities.

### M4 10-Core GPU Specs

| Spec | Value |
|------|-------|
| GPU Cores | 10 |
| ALUs per core | ~128 |
| Total ALUs | ~1280 |
| SIMD width | 32 threads |
| Theoretical FP32 | ~4.1 TFLOPS |
| Memory bandwidth | ~120 GB/s |
| Max threads/core | ~256+ |

### Current Kernel Bottlenecks

**1. Thread Utilization (Critical - ~1.25% used)**

```python
# paged_attention_metal.py
global_size=(n_heads, batch_size, 1)  # e.g., (32, 1, 1) for Llama 3.2 1B
local_size=(1, 1, 1)                  # Only 1 thread per threadgroup!
```

- Llama 3.2 1B has 32 heads
- Batch=1: 32 threads launched
- M4 can run 10 × 256 = 2560+ threads concurrently
- **Thread utilization: 32 / 2560 ≈ 1.25%**

**2. SIMD Waste (Critical - ~3% efficiency)**

- 1 thread per threadgroup means 31 of 32 SIMD lanes sit idle
- Each simdgroup executes in lockstep, but only lane 0 does useful work
- **SIMD efficiency: 1/32 ≈ 3%**

**3. Register Pressure (High)**

```metal
float out_acc[128];  // 128 × 4 bytes = 512 bytes = 128 registers
```

- Uses all ~128 available registers just for the accumulator
- No headroom for loop unrolling or prefetching
- May cause register spilling to slower memory

**4. Memory Access Pattern (Medium)**

```metal
// Current: scalar loads (4 bytes)
for (int d = 0; d < head_dim; d++) {
    score += query[q_base + d] * k_cache[k_base + d];
}

// Optimal: vectorized loads (16 bytes)
for (int d = 0; d < head_dim; d += 4) {
    float4 q = *((device float4*)(query + q_base + d));
    float4 k = *((device float4*)(k_cache + k_base + d));
    score += dot(q, k);
}
```

- Scalar float loads waste 75% of memory bus width
- No 16-byte alignment guarantees

**5. Two-Pass Softmax (Medium)**

- Current kernel reads entire KV cache twice (find max, then compute)
- Online softmax would read once, halving memory traffic

### Theoretical vs Actual Performance

For decode with context_len=1000, head_dim=64, n_heads=32:

```
FLOPs per token:
  = n_heads × 2_passes × context_len × head_dim × 2_ops (multiply + add)
  = 32 × 2 × 1000 × 64 × 2
  = 8.2 million FLOPs

Theoretical time at 4.1 TFLOPS:
  = 8.2M / 4.1T = 2 microseconds

Actual time: Much higher due to underutilization
```

### Efficiency Summary

| Metric | Current | Optimal | Waste |
|--------|---------|---------|-------|
| Thread utilization | ~1.25% | 80%+ | 97% |
| SIMD efficiency | ~3% | 100% | 97% |
| Memory loads | scalar | float4 | 75% |
| KV cache reads | 2x | 1x | 50% |

### Estimated Speedup from Phase 6

| Optimization | Expected Gain |
|--------------|---------------|
| 32-thread threadgroups | 10-32x |
| SIMD shuffle reductions | 2-4x |
| Vectorized float4 loads | 2-4x |
| Online softmax | 1.5-2x |
| half precision | 1.5-2x |
| **Combined** | **10-30x** |

The current kernel leaves ~97% of M4 GPU compute unused. Phase 6 optimizations target this gap.

---

## 8. Implementation Priority (Phase 6.2)

### Quick Wins (implement first)
1. **Pre-allocate output buffer** - trivial change
2. **Cache Metal buffer references** - avoid uop traversal
3. **Reuse block table tensor** - avoid allocation per call
4. **half precision** - change dtypes, big win

### Medium Effort
5. **32-thread threadgroups** - rewrite kernel launch
6. **SIMD reductions** - replace threadgroup barriers
7. **Vectorized float4 loads** - modify inner loops
8. **16-byte alignment** - adjust buffer layouts

### High Effort
9. **Online softmax** - algorithmic change
10. **Tiled attention** - major kernel rewrite
11. **simdgroup_async_copy** - async memory pipeline
12. **JIT compilation** - runtime kernel generation

---

## 9. Benchmarking

### Metrics to Track
- Tokens/second (throughput)
- Time-to-first-token (latency)
- GPU ALU utilization (target: >80%)
- Memory bandwidth utilization
- Register spill rate

### M4 10-Core Theoretical Maximums

| Resource | Theoretical Max | Unit |
|----------|-----------------|------|
| FP32 Compute | 4.1 | TFLOPS |
| FP16 Compute | 8.2 | TFLOPS |
| Memory Bandwidth | 120 | GB/s |
| Concurrent Threads | 2560+ | threads |
| SIMD Lanes | 1280 | lanes (10 cores × 4 simdgroups × 32) |

### Current Kernel Utilization (Phase 5) - Measured

**TinyLlama 1.1B** (32 heads, head_dim=64, 22 layers, FP16 = 2.2 GB)

Theoretical maximum (memory-bound): **55 tok/s** (120 GB/s ÷ 2.2 GB)

| Metric | Theoretical | Measured | Utilization |
|--------|-------------|----------|-------------|
| Thread count | 2560 | 32 | **1.25%** |
| SIMD efficiency | 100% | 3% | **3%** |
| Decode throughput | 55 tok/s | 1.4 tok/s | **2.6%** |
| Batched throughput (5 req) | 55 tok/s | 3.1 tok/s | **5.6%** |
| TTFT | ~18 ms | 942 ms | **~2%** |
| TPOT | ~18 ms | 722 ms | **~2.5%** |

### Measured Performance Summary

| Benchmark | Tokens/sec | Utilization |
|-----------|------------|-------------|
| single_request | 1.4 | 2.6% |
| concurrent_5 | 3.1 | 5.6% |
| best_scaling (4 req) | 2.5 | 4.6% |

### Decode Attention Performance Targets

| Metric | Current (measured) | Phase 6 Target | Improvement |
|--------|-------------------|----------------|-------------|
| Tokens/sec (batch=1) | 1.4 | 30+ | 21x |
| Tokens/sec (batch=5) | 3.1 | 40+ | 13x |
| TPOT | 722 ms | ~30 ms | 24x |
| Utilization | 2.6% | >50% | 19x |

### Compute-Bound vs Memory-Bound Analysis

For decode attention (1 query token, context_len K positions):

```
Memory reads per token:
  Q: n_heads × head_dim × 4 bytes = 32 × 64 × 4 = 8 KB
  K: context_len × n_kv_heads × head_dim × 4 = K × 8 × 64 × 4 = 2K KB
  V: context_len × n_kv_heads × head_dim × 4 = K × 8 × 64 × 4 = 2K KB
  Total: ~4K KB per token (for context_len = K)

FLOPs per token:
  Q·K: 2 × context_len × n_heads × head_dim = 2 × K × 32 × 64 = 4K FLOPs
  softmax: ~3 × context_len × n_heads = 3 × K × 32 = 0.1K FLOPs
  attn×V: 2 × context_len × n_heads × head_dim = 4K FLOPs
  Total: ~8K FLOPs per token

Arithmetic intensity = FLOPs / Bytes = 8K / 4K = 2 FLOPs/byte
```

At 2 FLOPs/byte, decode attention is **memory-bound** on M4:
- Memory ceiling: 120 GB/s × 2 = 240 GFLOPS achievable
- Compute ceiling: 4100 GFLOPS available
- **Memory is the bottleneck** → optimize for bandwidth, not raw compute

### Reference Performance
- Metal FlashAttention: 4400 gigainstructions/sec on M1 Max (83% ALU utilization)
- MetalQwen3: 75 tokens/sec (2.1x vs CPU), 10x faster TTFT
- vLLM on A100: ~3000 tokens/sec decode (batch=32)

### Profiling Tools
- Xcode Metal System Trace
- Metal Performance HUD
- `MTLCounterSampleBuffer` for programmatic profiling
- `Instrument.app` → Metal System Trace template
