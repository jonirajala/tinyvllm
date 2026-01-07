"""Metal Flash Attention implementation for prefill.

Phase 8.1: Tiled attention with online softmax for O(1) memory.

Unlike decode (single query token), prefill processes multiple queries at once.
This kernel uses:
- Tiling: Process Q in tiles (8 queries per threadgroup)
- Threadgroup memory: K/V tiles loaded once, reused across Q tile
- Online softmax: Running max/sum/output rescaled per KV tile
- GQA support: Map query heads to KV heads

Kernel Constraints:
- head_dim must be divisible by 4 (float4 vectorization)
- head_dim must be <= 128 (register limit)
- n_heads must be divisible by n_kv_heads (GQA requirement)
"""

from typing import Optional, Dict, Any
from tinygrad import Tensor, Device, dtypes
from tinygrad.runtime.ops_metal import MetalDevice, MetalCompiler, MetalProgram


def get_metal_buffer(tensor: Tensor):
    """Get the underlying Metal buffer from a tensor."""
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
# Flash Attention Kernel (FP16)
# =============================================================================
# Key design:
# - Each threadgroup handles one (head, q_tile) pair
# - 32 threads per threadgroup (1 simdgroup), each thread handles one query
# - Balanced K/V loading: all 32 threads participate via linear indexing
# - Vectorized K/V loads using half4 for 4x memory efficiency
# - Per-lane kv_limit for causal masking (reduces divergence)
# - Pre-scaled Q for fewer multiplications
# - fast::exp for cheaper exponentials
# - Online softmax across KV tiles
# =============================================================================

FLASH_ATTENTION_FP16_KERNEL = """
#include <metal_stdlib>
using namespace metal;

// Tile sizes
#define TILE_Q 32     // Queries per Q tile (one per thread)
#define TILE_KV 8     // KV positions per tile

// Flash Attention for prefill with tiled computation
// Each threadgroup processes TILE_Q queries for one head
kernel void flash_attention_prefill_fp16(
    device const half* queries [[buffer(0)]],    // [q_len, n_heads, head_dim]
    device const half* keys [[buffer(1)]],       // [kv_len, n_kv_heads, head_dim]
    device const half* values [[buffer(2)]],     // [kv_len, n_kv_heads, head_dim]
    device half* outputs [[buffer(3)]],          // [q_len, n_heads, head_dim]
    constant int& q_len [[buffer(4)]],
    constant int& kv_len [[buffer(5)]],
    constant int& n_heads [[buffer(6)]],
    constant int& n_kv_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& is_causal [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],  // (head_idx, q_tile_idx)
    uint tid [[thread_index_in_threadgroup]]      // 0-31 within threadgroup
) {
    int head_idx = tgid.x;
    int q_tile_idx = tgid.y;
    int q_start = q_tile_idx * TILE_Q;

    // Early exit if out of bounds
    if (head_idx >= n_heads || q_start >= q_len) return;

    // GQA mapping
    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    // Each thread handles one query (tid maps directly to query)
    int global_q_idx = q_start + tid;
    bool q_valid = (global_q_idx < q_len);

    // Per-lane KV limit for causal masking (reduces divergence in last tile)
    int kv_limit = is_causal ? min(kv_len, global_q_idx + 1) : kv_len;

    // Threadgroup-level KV limit: max KV any thread in this tile can attend to
    // This lets us skip entire tiles that no thread will use
    int kv_limit_group = is_causal ? min(kv_len, q_start + TILE_Q) : kv_len;

    // Threadgroup memory for K/V tiles - vectorized layout [TILE_KV][head_dim/4]
    threadgroup half4 K_tile[8][32];  // TILE_KV=8, max head_dim/4=32
    threadgroup half4 V_tile[8][32];

    // Load query into registers and pre-scale
    int num_vec4 = head_dim / 4;
    float4 q_reg[32];  // Max head_dim / 4 = 32 float4s

    if (q_valid) {
        int q_offset = (global_q_idx * n_heads + head_idx) * head_dim;
        device const half4* q_vec = (device const half4*)(queries + q_offset);
        for (int d = 0; d < num_vec4; d++) {
            q_reg[d] = float4(q_vec[d]) * scale;  // Pre-scale Q
        }
    }

    // Online softmax state per query
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 running_out[32];  // Max head_dim / 4 = 32
    for (int d = 0; d < num_vec4; d++) {
        running_out[d] = float4(0.0f);
    }

    // Process KV in tiles - only up to kv_limit_group (skip useless tiles)
    int num_kv_tiles = (kv_limit_group + TILE_KV - 1) / TILE_KV;

    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        int kv_start = kv_tile_idx * TILE_KV;

        // K/V load: threads 0-7 each load one KV position
        if (tid < 8) {
            int global_kv = kv_start + tid;
            if (global_kv < kv_len) {
                int kv_offset = (global_kv * n_kv_heads + kv_head_idx) * head_dim;
                device const half4* k_vec = (device const half4*)(keys + kv_offset);
                device const half4* v_vec = (device const half4*)(values + kv_offset);
                for (int d = 0; d < num_vec4; d++) {
                    K_tile[tid][d] = k_vec[d];
                    V_tile[tid][d] = v_vec[d];
                }
            } else {
                for (int d = 0; d < num_vec4; d++) {
                    K_tile[tid][d] = half4(0.0f);
                    V_tile[tid][d] = half4(0.0f);
                }
            }
        }

        // simdgroup_barrier since we're exactly 1 SIMD group (32 threads)
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Process each KV position in tile (using per-lane kv_limit)
        if (q_valid) {
            int tile_kv_end = min(kv_start + TILE_KV, kv_limit);
            for (int global_kv = kv_start; global_kv < tile_kv_end; global_kv++) {
                int local_kv = global_kv - kv_start;

                // Compute score: Q dot K (Q already scaled)
                float score = 0.0f;
                for (int d = 0; d < num_vec4; d++) {
                    float4 k4 = float4(K_tile[local_kv][d]);
                    score += dot(q_reg[d], k4);
                }

                // Online softmax update with fast::exp
                float new_max = max(running_max, score);
                float rescale = fast::exp(running_max - new_max);
                float exp_score = fast::exp(score - new_max);

                // Update running statistics
                running_sum = running_sum * rescale + exp_score;

                // Update running output with rescaling
                for (int d = 0; d < num_vec4; d++) {
                    float4 v4 = float4(V_tile[local_kv][d]);
                    running_out[d] = running_out[d] * rescale + exp_score * v4;
                }

                running_max = new_max;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final output
    if (q_valid && running_sum > 0.0f) {
        float inv_sum = 1.0f / running_sum;
        int out_offset = (global_q_idx * n_heads + head_idx) * head_dim;
        device half4* out_vec = (device half4*)(outputs + out_offset);

        for (int d = 0; d < num_vec4; d++) {
            out_vec[d] = half4(running_out[d] * inv_sum);
        }
    }
}
"""

FLASH_ATTENTION_FP32_KERNEL = """
#include <metal_stdlib>
using namespace metal;

// Tile sizes
#define TILE_Q 32     // Queries per Q tile (one per thread)
#define TILE_KV 8     // KV positions per tile

kernel void flash_attention_prefill_fp32(
    device const float* queries [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device float* outputs [[buffer(3)]],
    constant int& q_len [[buffer(4)]],
    constant int& kv_len [[buffer(5)]],
    constant int& n_heads [[buffer(6)]],
    constant int& n_kv_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& is_causal [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    int head_idx = tgid.x;
    int q_tile_idx = tgid.y;
    int q_start = q_tile_idx * TILE_Q;

    if (head_idx >= n_heads || q_start >= q_len) return;

    int n_rep = n_heads / n_kv_heads;
    int kv_head_idx = head_idx / n_rep;

    float scale = 1.0f / sqrt((float)head_dim);

    // Each thread handles one query (tid maps directly to query)
    int global_q_idx = q_start + tid;
    bool q_valid = (global_q_idx < q_len);

    // Per-lane KV limit for causal masking (reduces divergence in last tile)
    int kv_limit = is_causal ? min(kv_len, global_q_idx + 1) : kv_len;

    // Threadgroup-level KV limit: max KV any thread in this tile can attend to
    int kv_limit_group = is_causal ? min(kv_len, q_start + TILE_Q) : kv_len;

    // Threadgroup memory for K/V tiles - vectorized layout [TILE_KV][head_dim/4]
    threadgroup float4 K_tile[8][32];  // TILE_KV=8, max head_dim/4=32
    threadgroup float4 V_tile[8][32];

    int num_vec4 = head_dim / 4;
    float4 q_reg[32];

    // Load query into registers and pre-scale
    if (q_valid) {
        int q_offset = (global_q_idx * n_heads + head_idx) * head_dim;
        device const float4* q_vec = (device const float4*)(queries + q_offset);
        for (int d = 0; d < num_vec4; d++) {
            q_reg[d] = q_vec[d] * scale;  // Pre-scale Q
        }
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 running_out[32];
    for (int d = 0; d < num_vec4; d++) {
        running_out[d] = float4(0.0f);
    }

    // Process KV in tiles - only up to kv_limit_group (skip useless tiles)
    int num_kv_tiles = (kv_limit_group + TILE_KV - 1) / TILE_KV;

    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; kv_tile_idx++) {
        int kv_start = kv_tile_idx * TILE_KV;

        // K/V load: threads 0-7 each load one KV position
        if (tid < 8) {
            int global_kv = kv_start + tid;
            if (global_kv < kv_len) {
                int kv_offset = (global_kv * n_kv_heads + kv_head_idx) * head_dim;
                device const float4* k_vec = (device const float4*)(keys + kv_offset);
                device const float4* v_vec = (device const float4*)(values + kv_offset);
                for (int d = 0; d < num_vec4; d++) {
                    K_tile[tid][d] = k_vec[d];
                    V_tile[tid][d] = v_vec[d];
                }
            } else {
                for (int d = 0; d < num_vec4; d++) {
                    K_tile[tid][d] = float4(0.0f);
                    V_tile[tid][d] = float4(0.0f);
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Process each KV position in tile (using per-lane kv_limit)
        if (q_valid) {
            int tile_kv_end = min(kv_start + TILE_KV, kv_limit);
            for (int global_kv = kv_start; global_kv < tile_kv_end; global_kv++) {
                int local_kv = global_kv - kv_start;

                // Compute score: Q dot K (Q already scaled)
                float score = 0.0f;
                for (int d = 0; d < num_vec4; d++) {
                    score += dot(q_reg[d], K_tile[local_kv][d]);
                }

                // Online softmax update with fast::exp
                float new_max = max(running_max, score);
                float rescale = fast::exp(running_max - new_max);
                float exp_score = fast::exp(score - new_max);

                running_sum = running_sum * rescale + exp_score;

                for (int d = 0; d < num_vec4; d++) {
                    running_out[d] = running_out[d] * rescale + exp_score * V_tile[local_kv][d];
                }

                running_max = new_max;
            }
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (q_valid && running_sum > 0.0f) {
        float inv_sum = 1.0f / running_sum;
        int out_offset = (global_q_idx * n_heads + head_idx) * head_dim;
        device float4* out_vec = (device float4*)(outputs + out_offset);

        for (int d = 0; d < num_vec4; d++) {
            out_vec[d] = running_out[d] * inv_sum;
        }
    }
}
"""


class FlashAttentionMetal:
    """Metal implementation of Flash Attention for prefill.

    Uses tiled computation with online softmax for O(1) memory.
    """

    _instance: Optional['FlashAttentionMetal'] = None
    _program_fp32: Optional[MetalProgram] = None
    _program_fp16: Optional[MetalProgram] = None

    def __init__(self):
        self._compiled = False

    @classmethod
    def get_instance(cls) -> 'FlashAttentionMetal':
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

        lib_fp32 = compiler.compile(FLASH_ATTENTION_FP32_KERNEL)
        FlashAttentionMetal._program_fp32 = MetalProgram(device, 'flash_attention_prefill_fp32', lib_fp32)

        lib_fp16 = compiler.compile(FLASH_ATTENTION_FP16_KERNEL)
        FlashAttentionMetal._program_fp16 = MetalProgram(device, 'flash_attention_prefill_fp16', lib_fp16)

        self._compiled = True

    def prefill(
        self,
        queries: Tensor,   # [1, q_len, n_heads, head_dim]
        keys: Tensor,      # [1, kv_len, n_kv_heads, head_dim]
        values: Tensor,    # [1, kv_len, n_kv_heads, head_dim]
        causal: bool = True,
    ) -> Tensor:
        """Flash Attention for prefill.

        Args:
            queries: [1, q_len, n_heads, head_dim]
            keys: [1, kv_len, n_kv_heads, head_dim]
            values: [1, kv_len, n_kv_heads, head_dim]
            causal: Apply causal masking

        Returns:
            output: [1, q_len, n_heads, head_dim]
        """
        self._ensure_compiled()

        # Remove batch dimension
        queries = queries.squeeze(0)  # [q_len, n_heads, head_dim]
        keys = keys.squeeze(0)        # [kv_len, n_kv_heads, head_dim]
        values = values.squeeze(0)    # [kv_len, n_kv_heads, head_dim]

        q_len, n_heads, head_dim = queries.shape
        kv_len, n_kv_heads, _ = keys.shape

        # Validate constraints
        assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
        assert head_dim <= 128, f"head_dim must be <= 128, got {head_dim}"
        assert n_heads % n_kv_heads == 0, f"GQA requires n_heads divisible by n_kv_heads"

        use_fp16 = queries.dtype == dtypes.float16

        # Allocate output
        output = Tensor.zeros(q_len, n_heads, head_dim, dtype=queries.dtype).contiguous().realize()

        # Get Metal buffers
        q_buf = get_metal_buffer(queries.contiguous().realize())
        k_buf = get_metal_buffer(keys.contiguous().realize())
        v_buf = get_metal_buffer(values.contiguous().realize())
        out_buf = get_metal_buffer(output)

        # Select kernel
        program = self._program_fp16 if use_fp16 else self._program_fp32

        # Dispatch
        # Each threadgroup handles TILE_Q=32 queries for one head
        num_q_tiles = (q_len + 31) // 32

        program(
            q_buf._buf,
            k_buf._buf,
            v_buf._buf,
            out_buf._buf,
            global_size=(n_heads, num_q_tiles, 1),
            local_size=(32, 1, 1),  # 32 threads per threadgroup (1 simdgroup)
            vals=(q_len, kv_len, n_heads, n_kv_heads, head_dim, 1 if causal else 0),
            wait=False
        )

        # Add batch dimension back
        return output.unsqueeze(0)


# Export with standard name
flash_prefill_attention_metal = FlashAttentionMetal.get_instance().prefill
