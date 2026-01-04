"""Metal implementation of fused paged attention.

Phase 5: Fused kernel that reads directly from block tables during attention,
avoiding the Tensor.stack() overhead of gathering blocks first.

Phase 6.2: Optimizations:
- Cache Metal buffer references for KV cache (avoid uop tree traversal)
- Half precision (FP16) kernel variant for 2x register efficiency

Note: Metal only supports float32 and float16. Models with bfloat16 weights
are converted to float32 at load time. Native bfloat16 will be supported
when CUDA backend is implemented (CUDA has native bf16 support).
"""

from typing import List, Optional, Dict, Any
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram


def get_metal_buffer(tensor: Tensor):
    """Get the underlying Metal buffer from a tensor, traversing the uop tree if needed.
    It finds the GPU memory pointer where the tensor's data lives.
    """
    tensor = tensor.contiguous().realize()

    def find_buffer(uop):
        if hasattr(uop, 'realized') and uop.realized is not None:
            return uop.realized
        if hasattr(uop, 'src'):
            for s in uop.src:
                result = find_buffer(s)
                if result is not None:
                    return result
        return None

    buf = find_buffer(tensor.uop)
    if buf is None:
        raise RuntimeError(f"Could not find realized buffer for tensor with shape {tensor.shape}")
    return buf


"""
                      ┌─────────────────────────────────────────┐
                      │         METAL GPU THREADS               │
                      │  Thread (0,0)  Thread (1,0) ... (31,0)  │  ← Sequence 0
                      │  Thread (0,1)  Thread (1,1) ... (31,1)  │  ← Sequence 1
                      │       ...                               │
                      └─────────────────────────────────────────┘
                                        │
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Each thread (head_idx, batch_idx):                                          │
  │                                                                             │
  │ 1. Look up which KV head to use (GQA)                                       │
  │    kv_head = head_idx / (n_heads / n_kv_heads)                              │
  │                                                                             │
  │ 2. First pass - find max score:                                             │
  │    for pos in 0..context_len:                                               │
  │        block_id = block_table[pos / 16]     ←── Paged lookup!               │
  │        offset = pos % 16                                                    │
  │        K = k_cache[block_id, offset, kv_head, :]                            │
  │        score = Q · K / √d                                                   │
  │        max_score = max(max_score, score)                                    │
  │                                                                             │
  │ 3. Second pass - softmax + weighted V:                                      │
  │    for pos in 0..context_len:                                               │
  │        block_id = block_table[pos / 16]                                     │
  │        K = k_cache[block_id, offset, kv_head, :]                            │
  │        V = v_cache[block_id, offset, kv_head, :]                            │
  │        weight = exp(Q · K / √d - max_score)                                 │
  │        sum_exp += weight                                                    │
  │        out_acc += weight * V                                                │
  │                                                                             │
  │ 4. Normalize and write:                                                     │
  │    output[batch, head, :] = out_acc / sum_exp                               │
  └─────────────────────────────────────────────────────────────────────────────┘
"""

# Metal kernel source for paged attention
# Each thread handles one query head, loops over KV positions
PAGED_ATTENTION_KERNEL = """
#include <metal_stdlib>
using namespace metal;

// Paged attention kernel for single sequence decode
// Query: [n_heads, head_dim] (single token)
// K/V cache: [num_blocks, block_size, n_kv_heads, head_dim]
// Block table: [max_blocks_per_seq]
// Output: [n_heads, head_dim]
kernel void paged_attention_decode(
    device const float* query [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& context_len [[buffer(5)]],
    constant int& block_size [[buffer(6)]],
    constant int& n_heads [[buffer(7)]],
    constant int& n_kv_heads [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    uint head_idx [[thread_position_in_grid]]
) {
    if ((int)head_idx >= n_heads) return;

    // GQA: map query head to KV head
    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    // Registers for accumulation
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: find max score for numerical stability
    for (int pos = 0; pos < context_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = block_table[block_idx];

        // K layout: [num_blocks, block_size, n_kv_heads, head_dim]
        int k_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        int q_base = head_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += query[q_base + d] * k_cache[k_base + d];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and weighted sum
    // Use thread-local array for output accumulation
    float out_acc[128];  // Assuming head_dim <= 128
    for (int d = 0; d < head_dim; d++) {
        out_acc[d] = 0.0f;
    }

    for (int pos = 0; pos < context_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        int q_base = head_idx * head_dim;

        // Recompute score
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += query[q_base + d] * k_cache[kv_base + d];
        }
        score = exp(score * scale - max_score);
        sum_exp += score;

        // Accumulate weighted value
        for (int d = 0; d < head_dim; d++) {
            out_acc[d] += score * v_cache[kv_base + d];
        }
    }

    // Write normalized output
    int out_base = head_idx * head_dim;
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        output[out_base + d] = out_acc[d] * inv_sum;
    }
}

// Batched version for multiple sequences in decode
// Each threadgroup handles one sequence
kernel void paged_attention_batched(
    device const float* queries [[buffer(0)]],      // [batch, n_heads, head_dim]
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],   // [batch, max_blocks]
    device const int* context_lens [[buffer(4)]],   // [batch]
    device float* outputs [[buffer(5)]],            // [batch, n_heads, head_dim]
    constant int& batch_size [[buffer(6)]],
    constant int& max_blocks [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& n_heads [[buffer(9)]],
    constant int& n_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]  // (head_idx, batch_idx)
) {
    int head_idx = tid.x;
    int batch_idx = tid.y;

    if (head_idx >= n_heads || batch_idx >= batch_size) return;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) return;

    // GQA mapping
    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    // Get this sequence's block table
    device const int* seq_block_table = block_tables + batch_idx * max_blocks;

    // Query and output offsets for this sequence/head
    int q_offset = (batch_idx * n_heads + head_idx) * head_dim;
    int out_offset = q_offset;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: find max
    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int k_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += queries[q_offset + d] * k_cache[k_base + d];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: softmax + weighted sum
    float out_acc[128];
    for (int d = 0; d < head_dim; d++) {
        out_acc[d] = 0.0f;
    }

    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += queries[q_offset + d] * k_cache[kv_base + d];
        }
        score = exp(score * scale - max_score);
        sum_exp += score;

        for (int d = 0; d < head_dim; d++) {
            out_acc[d] += score * v_cache[kv_base + d];
        }
    }

    // Write output
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        outputs[out_offset + d] = out_acc[d] * inv_sum;
    }
}
"""

# Half-precision (float16) version of the kernel for 2x register efficiency
PAGED_ATTENTION_KERNEL_FP16 = """
#include <metal_stdlib>
using namespace metal;

// Half-precision paged attention kernel for single sequence decode
kernel void paged_attention_decode_fp16(
    device const half* query [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant int& context_len [[buffer(5)]],
    constant int& block_size [[buffer(6)]],
    constant int& n_heads [[buffer(7)]],
    constant int& n_kv_heads [[buffer(8)]],
    constant int& head_dim [[buffer(9)]],
    uint head_idx [[thread_position_in_grid]]
) {
    if ((int)head_idx >= n_heads) return;

    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: find max score
    for (int pos = 0; pos < context_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = block_table[block_idx];

        int k_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        int q_base = head_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += float(query[q_base + d]) * float(k_cache[k_base + d]);
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: softmax + weighted sum (use float for accumulation precision)
    float out_acc[128];
    for (int d = 0; d < head_dim; d++) {
        out_acc[d] = 0.0f;
    }

    for (int pos = 0; pos < context_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        int q_base = head_idx * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += float(query[q_base + d]) * float(k_cache[kv_base + d]);
        }
        score = exp(score * scale - max_score);
        sum_exp += score;

        for (int d = 0; d < head_dim; d++) {
            out_acc[d] += score * float(v_cache[kv_base + d]);
        }
    }

    // Write normalized output as half
    int out_base = head_idx * head_dim;
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        output[out_base + d] = half(out_acc[d] * inv_sum);
    }
}

// Half-precision batched version
kernel void paged_attention_batched_fp16(
    device const half* queries [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device half* outputs [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    constant int& max_blocks [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& n_heads [[buffer(9)]],
    constant int& n_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int head_idx = tid.x;
    int batch_idx = tid.y;

    if (head_idx >= n_heads || batch_idx >= batch_size) return;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) return;

    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    device const int* seq_block_table = block_tables + batch_idx * max_blocks;

    int q_offset = (batch_idx * n_heads + head_idx) * head_dim;
    int out_offset = q_offset;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // First pass: find max
    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int k_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += float(queries[q_offset + d]) * float(k_cache[k_base + d]);
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: softmax + weighted sum
    float out_acc[128];
    for (int d = 0; d < head_dim; d++) {
        out_acc[d] = 0.0f;
    }

    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += float(queries[q_offset + d]) * float(k_cache[kv_base + d]);
        }
        score = exp(score * scale - max_score);
        sum_exp += score;

        for (int d = 0; d < head_dim; d++) {
            out_acc[d] += score * float(v_cache[kv_base + d]);
        }
    }

    // Write output as half
    float inv_sum = 1.0f / sum_exp;
    for (int d = 0; d < head_dim; d++) {
        outputs[out_offset + d] = half(out_acc[d] * inv_sum);
    }
}
"""


class PagedAttentionMetal:
    """Metal implementation of fused paged attention.

    Phase 6.2: Optimized with buffer reuse and Metal buffer caching.
    """

    _instance: Optional['PagedAttentionMetal'] = None
    _program_batched: Optional[MetalProgram] = None
    _program_batched_fp16: Optional[MetalProgram] = None

    def __init__(self):
        self._compiled = False

        # Phase 6.2: Cache Metal buffer refs for KV cache tensors (keyed by tensor id)
        # KV cache tensors are persistent, so caching avoids uop tree traversal
        self._kv_buf_cache: Dict[int, Any] = {}

    @classmethod
    def get_instance(cls) -> 'PagedAttentionMetal':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_compiled(self):
        """Compile kernels on first use."""
        if self._compiled:
            return

        device = MetalDevice('METAL')
        compiler = MetalCompiler()

        # Compile FP32 kernel
        lib_fp32 = compiler.compile(PAGED_ATTENTION_KERNEL)
        PagedAttentionMetal._program_batched = MetalProgram(device, 'paged_attention_batched', lib_fp32)

        # Compile FP16 kernel
        lib_fp16 = compiler.compile(PAGED_ATTENTION_KERNEL_FP16)
        PagedAttentionMetal._program_batched_fp16 = MetalProgram(device, 'paged_attention_batched_fp16', lib_fp16)

        self._compiled = True

    def _get_cached_kv_buffer(self, tensor: Tensor) -> Any:
        """Get Metal buffer for KV cache tensor, using cache if available."""
        # Use tensor's id as cache key - KV cache tensors are persistent
        tensor_id = id(tensor)
        if tensor_id not in self._kv_buf_cache:
            self._kv_buf_cache[tensor_id] = get_metal_buffer(tensor)
        return self._kv_buf_cache[tensor_id]

    def batched(
        self,
        queries: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        block_tables: List[List[int]],
        context_lens: List[int],
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int,
    ) -> Tensor:
        """
        Fused paged attention for multiple sequences in decode.

        Phase 6.2: Optimized with buffer reuse and Metal buffer caching.

        Args:
            queries: [batch, 1, n_heads, head_dim]
            k_cache: [num_blocks, block_size, n_kv_heads, head_dim]
            v_cache: [num_blocks, block_size, n_kv_heads, head_dim]
            block_tables: List of block tables, one per sequence
            context_lens: Context length for each sequence
            n_heads: Number of query heads
            n_kv_heads: Number of KV heads
            head_dim: Dimension per head
            block_size: Tokens per block

        Returns:
            output: [batch, 1, n_heads, head_dim]
        """
        self._ensure_compiled()

        batch_size = len(block_tables)
        max_blocks = max(len(bt) for bt in block_tables) if block_tables else 0

        # Determine dtype (FP16 or FP32)
        use_fp16 = queries.dtype == dtypes.float16

        # Allocate output buffer
        output = Tensor.zeros(batch_size, n_heads, head_dim, dtype=queries.dtype).contiguous().realize()

        # Pad block tables to same length and flatten
        padded_tables = []
        for bt in block_tables:
            padded = bt + [0] * (max_blocks - len(bt))
            padded_tables.extend(padded)

        # Write block table data - create tensor directly from list
        # Note: For small tensors like block tables, allocation overhead is minimal
        pt_tensor = Tensor(padded_tables, dtype=dtypes.int32).contiguous().realize()
        pt_buf = get_metal_buffer(pt_tensor)

        # Phase 6.2: Context lengths tensor
        ctx_tensor = Tensor(context_lens, dtype=dtypes.int32).contiguous().realize()
        ctx_buf = get_metal_buffer(ctx_tensor)

        # Get query buffer (new each call since queries change)
        q_buf = get_metal_buffer(queries.reshape(batch_size, n_heads, head_dim))

        # Phase 6.2: Use cached Metal buffer refs for KV cache (persistent across calls)
        k_buf = self._get_cached_kv_buffer(k_cache)
        v_buf = self._get_cached_kv_buffer(v_cache)

        # Get output buffer ref
        out_buf = get_metal_buffer(output)

        # Select kernel based on dtype
        program = self._program_batched_fp16 if use_fp16 else self._program_batched

        # Run batched kernel
        program(
            q_buf._buf,
            k_buf._buf,
            v_buf._buf,
            pt_buf._buf,
            ctx_buf._buf,
            out_buf._buf,
            global_size=(n_heads, batch_size, 1),
            local_size=(1, 1, 1),
            vals=(batch_size, max_blocks, block_size, n_heads, n_kv_heads, head_dim),
            wait=True
        )

        return output.reshape(batch_size, 1, n_heads, head_dim)


# Export with standard name for dispatcher
fused_paged_attention = PagedAttentionMetal.get_instance().batched
