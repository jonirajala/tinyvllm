# Phase 4: KV Cache Optimization for Tinygrad

## Current State (Phase 2)

In Phase 2, we use a simple **list-based KV cache** where each sequence stores its K/V tensors in Python lists. This works but has performance limitations.

```python
# Phase 2: Simple list-based (current)
self.k_cache[seq_id].append(k_tensor)  # Append each token's K/V
```

## The Problem

Real paged attention requires:
1. **Pre-allocated memory pool** - One big tensor for all K/V blocks
2. **Scattered writes** - Write to arbitrary block_id + offset
3. **Gathered reads** - Read from multiple non-contiguous blocks

Tinygrad doesn't support scattered writes to tensor slices:
```python
# This FAILS in tinygrad:
self.k_cache[layer_idx, block_id, offset, :, :] = k
# RuntimeError: setitem target needs to be contiguous
```

## Why Custom Kernels in Production

Production systems (vLLM) use custom CUDA kernels because:

1. **Fused gather + attention** - Read from scattered blocks during matmul, no intermediate copy
2. **No memory allocation** - Kernel reads directly from block addresses
3. **Optimal memory access** - Coalesced reads tuned for GPU architecture

```
Without custom kernel:
  gather_to_contiguous()  → O(context_len) copy
  matmul(Q, K)            → attention computation

With custom kernel:
  paged_attention(Q, K_cache, block_table)  → fused, no copy
```

## Tinygrad-Native Solution (Phase 4)

Since tinygrad supports multiple backends (CPU, CUDA, Metal, AMD, WebGPU), we want a portable solution without backend-specific kernels.

### Key Insight

Tinygrad DOES support:
1. **Slice assignment on realized tensors** - If tensor is `.contiguous().realize()`, writes work
2. **Fancy indexing for reads** - `tensor[block_ids]` gathers blocks
3. **Tensor.stack()** - Combine multiple tensors

### The Approach

Store each block as a **separate realized tensor**:

```python
class KVCache:
    def __init__(self, num_layers, num_blocks, block_size, n_kv_heads, head_dim, dtype):
        # Each block is a separate contiguous realized tensor
        self.k_blocks = [
            [Tensor.zeros(block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
             for _ in range(num_blocks)]
            for _ in range(num_layers)
        ]
        self.v_blocks = [
            [Tensor.zeros(block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
             for _ in range(num_blocks)]
            for _ in range(num_layers)
        ]

    def write_kv(self, layer_idx, block_id, offset, k, v):
        # Direct assignment works on realized tensors
        self.k_blocks[layer_idx][block_id][offset] = k
        self.k_blocks[layer_idx][block_id] = self.k_blocks[layer_idx][block_id].realize()

        self.v_blocks[layer_idx][block_id][offset] = v
        self.v_blocks[layer_idx][block_id] = self.v_blocks[layer_idx][block_id].realize()

    def read_kv_blocks(self, layer_idx, block_ids):
        # Stack requested blocks into contiguous tensor
        k = Tensor.stack(*[self.k_blocks[layer_idx][i] for i in block_ids])
        v = Tensor.stack(*[self.v_blocks[layer_idx][i] for i in block_ids])
        return k, v  # Shape: [num_blocks, block_size, n_kv_heads, head_dim]
```

### Why This Works

1. **Each block is contiguous** - Separate tensor, not a slice of big tensor
2. **`.realize()` allocates memory** - Makes it a real buffer that accepts writes
3. **Stack for reading** - Gathers blocks into format attention needs

### Tradeoffs

| Aspect | One Big Tensor | Separate Blocks (Phase 4) |
|--------|---------------|---------------------------|
| Write | Needs custom kernel | Works in pure tinygrad |
| Read | Fancy index (view) | Stack (copy) |
| Memory layout | Optimal | Slightly fragmented |
| Portability | Backend-specific | All tinygrad backends |
| Complexity | High | Low |

### Performance Considerations

**Write performance**: Good
- Direct assignment to realized tensor is fast
- One kernel launch per write

**Read performance**: Acceptable
- `Tensor.stack()` creates a copy
- O(blocks * block_size) memory copy per attention
- For small/medium context lengths, this is fine

**When to optimize further**:
- Context length > 4096 tokens
- High throughput requirements (>100 req/s)
- Then consider backend-specific kernels

## Implementation Steps (Phase 4)

1. **Replace list-based cache with block tensors**
   - Pre-allocate blocks as separate realized tensors
   - Update `write_kv` to use direct assignment
   - Update `read_kv_blocks` to use `Tensor.stack`

2. **Update attention_utils.py**
   - `gather_kv_from_blocks` works with stacked tensor
   - Handle partial blocks (last block may not be full)

3. **Benchmark**
   - Compare memory usage vs list-based
   - Measure write/read latency
   - Test on different backends (Metal, CUDA)

4. **Optional: Backend-specific optimization**
   - If needed, write Metal kernel for Mac
   - Or CUDA kernel for NVIDIA
   - Use tinygrad's `MetalProgram` / `CUDAProgram`

## Example: Metal Kernel (Advanced)

For maximum performance on Mac, you could write a custom Metal kernel:

```python
from tinygrad.runtime.ops_metal import MetalProgram

kernel_code = """
#include <metal_stdlib>
using namespace metal;

kernel void paged_attention_write(
    device float* cache [[buffer(0)]],
    device const float* kv [[buffer(1)]],
    constant int& layer_idx [[buffer(2)]],
    constant int& block_id [[buffer(3)]],
    constant int& offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Calculate destination index in cache
    int cache_idx = layer_idx * BLOCKS * BLOCK_SIZE * HEADS * DIM
                  + block_id * BLOCK_SIZE * HEADS * DIM
                  + offset * HEADS * DIM
                  + gid;
    cache[cache_idx] = kv[gid];
}
"""

# Compile and use
prg = MetalProgram("paged_attention_write", kernel_code)
```

But this is only needed if the tinygrad-native solution is too slow for your use case.

## Summary

| Phase | Approach | Performance | Portability |
|-------|----------|-------------|-------------|
| Phase 2 | List-based | Slow | All backends |
| Phase 4 | Separate realized blocks | Good | All backends |
| Future | Custom kernels | Optimal | Per-backend |

The Phase 4 approach gives us 80% of the performance with 20% of the complexity, and works on any tinygrad backend.
