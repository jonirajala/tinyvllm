"""Metal implementation of paged attention with online softmax.

Phase 6.2 High Effort Optimizations:
- Online softmax (single pass instead of two-pass)

Online Softmax Algorithm:
Instead of two passes (find max, then softmax+accumulate), we maintain
running statistics and rescale when a new max is found:

  running_max = -inf, running_sum = 0, running_out = 0
  for each position:
      score = Q · K * scale
      new_max = max(running_max, score)
      rescale = exp(running_max - new_max)
      running_sum = running_sum * rescale + exp(score - new_max)
      running_out = running_out * rescale + exp(score - new_max) * V
      running_max = new_max
  output = running_out / running_sum

Benefits:
- Single pass over KV cache (vs two passes)
- Better memory locality
- Foundation for Flash Attention tiled variant

Kernel Constraints (same as original):
- head_dim must be divisible by 4 (float4 vectorization)
- head_dim must be <= 128 (32 SIMD lanes * 4 floats per lane)
- n_heads must be divisible by n_kv_heads (GQA requirement)
"""

from typing import List, Optional, Dict, Any
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram


def get_metal_buffer(tensor: Tensor):
    """Get the underlying Metal buffer from a tensor, traversing the uop tree if needed."""
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


# =============================================================================
# Online Softmax Kernels
# =============================================================================
# Key change from two-pass kernel:
# - Single loop instead of two loops
# - Maintain running_max, running_sum, running_out
# - Rescale accumulated values when new max is found
# =============================================================================

ONLINE_SOFTMAX_KERNEL_FP32 = """
#include <metal_stdlib>
using namespace metal;

// Online softmax paged attention (FP32) with single-pass algorithm
// Each simdgroup (32 threads) handles one (head, batch) pair
kernel void paged_attention_online_softmax(
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int head_idx = tgid.x;
    int batch_idx = tgid.y;

    if (head_idx >= n_heads || batch_idx >= batch_size) return;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) return;

    // GQA mapping
    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    // Get this sequence's block table
    device const int* seq_block_table = block_tables + batch_idx * max_blocks;

    // Query setup
    int q_base = (batch_idx * n_heads + head_idx) * head_dim;
    device const float4* q_vec = (device const float4*)(queries + q_base);

    int num_vec4 = head_dim / 4;
    bool lane_active = (int)lane < num_vec4;

    // Load query once
    float4 q4 = lane_active ? q_vec[lane] : float4(0.0f);

    // =========================================================================
    // Online softmax state (single pass)
    // =========================================================================
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 running_out = float4(0.0f);

    // =========================================================================
    // Single pass: online softmax + weighted V accumulation
    // Optimization: 1-exp branch - only compute one exp() per token instead of two
    // =========================================================================
    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        device const float4* k_vec = (device const float4*)(k_cache + kv_base);
        device const float4* v_vec = (device const float4*)(v_cache + kv_base);

        // Compute attention score
        float partial = lane_active ? dot(q4, k_vec[lane]) : 0.0f;
        float score = simd_sum(partial) * scale;

        // Online softmax update with 1-exp optimization
        // Instead of 2 exp() calls, use branching to compute only 1
        float rescale, exp_score;
        if (score <= running_max) {
            // Common case: score doesn't exceed max
            rescale = 1.0f;
            exp_score = fast::exp(score - running_max);
        } else {
            // New max found: rescale old values
            rescale = fast::exp(running_max - score);
            exp_score = 1.0f;
            running_max = score;
        }

        // Broadcast scalars to all lanes
        rescale = simd_broadcast_first(rescale);
        exp_score = simd_broadcast_first(exp_score);

        // Update running statistics
        running_sum = running_sum * rescale + exp_score;

        if (lane_active) {
            // Rescale old output and add new weighted V
            running_out = running_out * rescale + exp_score * v_vec[lane];
        }
    }

    // =========================================================================
    // Write normalized output
    // =========================================================================
    if (lane_active) {
        device float4* out_vec = (device float4*)(outputs + q_base);
        float inv_sum = simd_broadcast_first(1.0f / running_sum);
        out_vec[lane] = running_out * inv_sum;
    }
}
"""

ONLINE_SOFTMAX_KERNEL_FP16 = """
#include <metal_stdlib>
using namespace metal;

// Online softmax paged attention (FP16) with single-pass algorithm
kernel void paged_attention_online_softmax_fp16(
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
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    int head_idx = tgid.x;
    int batch_idx = tgid.y;

    if (head_idx >= n_heads || batch_idx >= batch_size) return;

    int ctx_len = context_lens[batch_idx];
    if (ctx_len == 0) return;

    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    device const int* seq_block_table = block_tables + batch_idx * max_blocks;

    int q_base = (batch_idx * n_heads + head_idx) * head_dim;
    device const half4* q_vec = (device const half4*)(queries + q_base);

    int num_vec4 = head_dim / 4;
    bool lane_active = (int)lane < num_vec4;

    // Load query as half4, convert to float4 for computation
    float4 q4 = lane_active ? float4(q_vec[lane]) : float4(0.0f);

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 running_out = float4(0.0f);

    // Single pass with 1-exp optimization
    for (int pos = 0; pos < ctx_len; pos++) {
        int block_idx = pos / block_size;
        int block_offset = pos % block_size;
        int phys_block = seq_block_table[block_idx];

        int kv_base = ((phys_block * block_size + block_offset) * n_kv_heads + kv_head_idx) * head_dim;
        device const half4* k_vec = (device const half4*)(k_cache + kv_base);
        device const half4* v_vec = (device const half4*)(v_cache + kv_base);

        float partial = lane_active ? dot(q4, float4(k_vec[lane])) : 0.0f;
        float score = simd_sum(partial) * scale;

        // 1-exp optimization: only compute one exp() per token
        float rescale, exp_score;
        if (score <= running_max) {
            rescale = 1.0f;
            exp_score = fast::exp(score - running_max);
        } else {
            rescale = fast::exp(running_max - score);
            exp_score = 1.0f;
            running_max = score;
        }

        rescale = simd_broadcast_first(rescale);
        exp_score = simd_broadcast_first(exp_score);

        running_sum = running_sum * rescale + exp_score;

        if (lane_active) {
            running_out = running_out * rescale + exp_score * float4(v_vec[lane]);
        }
    }

    // Write output as half4
    if (lane_active) {
        device half4* out_vec = (device half4*)(outputs + q_base);
        float inv_sum = simd_broadcast_first(1.0f / running_sum);
        out_vec[lane] = half4(running_out * inv_sum);
    }
}
"""


class PagedAttentionOnline:
    """Metal implementation of paged attention with online softmax.

    Phase 6.2 High Effort: Single-pass attention.

    Improvements over two-pass attention:
    - Online softmax: Single pass over KV cache instead of two passes
    - Better memory locality (each K/V loaded once instead of twice)
    - Foundation for Flash Attention tiled variant
    - Pre-allocated buffers to minimize sync points
    """

    _instance: Optional['PagedAttentionOnline'] = None
    _program_fp32: Optional[MetalProgram] = None
    _program_fp16: Optional[MetalProgram] = None

    def __init__(self):
        self._compiled = False
        self._kv_buf_cache: Dict[int, Any] = {}
        # Pre-allocated buffer pools (key: (batch, heads, dim, dtype))
        self._output_pool: Dict[tuple, Tensor] = {}
        self._bt_pool: Dict[tuple, Tensor] = {}
        self._ctx_pool: Dict[tuple, Tensor] = {}

    @classmethod
    def get_instance(cls) -> 'PagedAttentionOnline':
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

        lib_fp32 = compiler.compile(ONLINE_SOFTMAX_KERNEL_FP32)
        PagedAttentionOnline._program_fp32 = MetalProgram(device, 'paged_attention_online_softmax', lib_fp32)

        lib_fp16 = compiler.compile(ONLINE_SOFTMAX_KERNEL_FP16)
        PagedAttentionOnline._program_fp16 = MetalProgram(device, 'paged_attention_online_softmax_fp16', lib_fp16)

        self._compiled = True

    def _get_cached_kv_buffer(self, tensor: Tensor) -> Any:
        """Get Metal buffer for KV cache tensor, using cache if available."""
        tensor_id = id(tensor)
        if tensor_id not in self._kv_buf_cache:
            self._kv_buf_cache[tensor_id] = get_metal_buffer(tensor)
        return self._kv_buf_cache[tensor_id]

    def _get_output_buffer(self, batch_size: int, n_heads: int, head_dim: int, dtype) -> Tensor:
        """Get pre-allocated output buffer, creating if needed."""
        key = (batch_size, n_heads, head_dim, dtype)
        if key not in self._output_pool:
            # Only realize once on first use
            self._output_pool[key] = Tensor.zeros(batch_size, n_heads, head_dim, dtype=dtype).contiguous().realize()
        return self._output_pool[key]

    def _get_bt_buffer(self, size: int) -> Tensor:
        """Get pre-allocated block table buffer."""
        if size not in self._bt_pool:
            self._bt_pool[size] = Tensor.zeros(size, dtype=dtypes.int32).contiguous().realize()
        return self._bt_pool[size]

    def _get_ctx_buffer(self, size: int) -> Tensor:
        """Get pre-allocated context lens buffer."""
        if size not in self._ctx_pool:
            self._ctx_pool[size] = Tensor.zeros(size, dtype=dtypes.int32).contiguous().realize()
        return self._ctx_pool[size]

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
        Fused paged attention with online softmax for decode.

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
        # Safety assertions
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        assert head_dim <= 128, f"head_dim must be <= 128, got {head_dim}"
        assert n_heads % n_kv_heads == 0, f"GQA requires n_heads divisible by n_kv_heads"

        self._ensure_compiled()

        batch_size = len(block_tables)
        max_blocks = max(len(bt) for bt in block_tables) if block_tables else 0

        use_fp16 = queries.dtype == dtypes.float16

        # Reuse pre-allocated output buffer (no realize - already done once)
        output = self._get_output_buffer(batch_size, n_heads, head_dim, queries.dtype)

        # Block tables - create tensor but defer realize
        padded_tables = []
        for bt in block_tables:
            padded = bt + [0] * (max_blocks - len(bt))
            padded_tables.extend(padded)

        # Create tensors from Python lists
        pt_tensor = Tensor(padded_tables, dtype=dtypes.int32).contiguous().realize()
        ctx_tensor = Tensor(context_lens, dtype=dtypes.int32).contiguous().realize()

        pt_buf = get_metal_buffer(pt_tensor)
        ctx_buf = get_metal_buffer(ctx_tensor)

        # Get buffers (queries should already be realized from upstream)
        q_buf = get_metal_buffer(queries.reshape(batch_size, n_heads, head_dim))
        k_buf = self._get_cached_kv_buffer(k_cache)
        v_buf = self._get_cached_kv_buffer(v_cache)
        out_buf = get_metal_buffer(output)

        # Select kernel
        program = self._program_fp16 if use_fp16 else self._program_fp32

        # Dispatch - don't wait unless we need the result immediately
        # The next operation that needs the output will sync
        program(
            q_buf._buf,
            k_buf._buf,
            v_buf._buf,
            pt_buf._buf,
            ctx_buf._buf,
            out_buf._buf,
            global_size=(n_heads, batch_size, 1),
            local_size=(32, 1, 1),
            vals=(batch_size, max_blocks, block_size, n_heads, n_kv_heads, head_dim),
            wait=False  # Async dispatch - sync happens when output is used
        )

        return output.reshape(batch_size, 1, n_heads, head_dim)


# Export with standard name for dispatcher
paged_decode_attention = PagedAttentionOnline.get_instance().batched
