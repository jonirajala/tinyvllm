"""Custom kernels for paged attention, flash attention, and fused ops.

- Metal: Custom Metal kernels with optimizations
- Other: Pure tinygrad fallback (device-agnostic)

Paged Attention: For decode phase (reading from block-based KV cache)
Flash Attention: For prefill phase (O(1) memory, direct on fresh K/V)
"""

from tinygrad import Device

# Pure tinygrad version (device-agnostic)
from .paged_decode_attention_tinygrad import paged_decode_attention_tinygrad

# Flash attention for prefill (Phase 8.1)
from .flash_prefill_attention_tinygrad import flash_prefill_attention_tinygrad

# Lazy kernel selection - checked at call time, not import time
_paged_decode_kernel = None
_flash_metal_kernel = None


def paged_decode_attention(queries, k_cache, v_cache, block_tables_tensor,
                           context_lens_tensor, n_heads, n_kv_heads,
                           head_dim, block_size, max_context_len=None):
    """Select paged decode kernel based on device (tensor-based API).

    On Metal: uses custom Metal kernel with online softmax.
    On other devices: uses pure tinygrad implementation.
    """
    global _paged_decode_kernel
    device = Device.DEFAULT.split(":")[0].lower()

    if device == "metal":
        if _paged_decode_kernel is None:
            from .paged_decode_attention_metal import PagedAttentionOnline
            _paged_decode_kernel = PagedAttentionOnline.get_instance().batched_tensors

        return _paged_decode_kernel(queries, k_cache, v_cache, block_tables_tensor,
                                    context_lens_tensor, n_heads, n_kv_heads,
                                    head_dim, block_size)
    else:
        return paged_decode_attention_tinygrad(queries, k_cache, v_cache, block_tables_tensor,
                                               context_lens_tensor, n_heads, n_kv_heads,
                                               head_dim, block_size, max_context_len)


def flash_prefill_attention(query, key, value, causal=True):
    """Flash Attention for prefill phase (O(1) memory).

    Selects Metal or tinygrad kernel based on device.
    Input/Output: [1, q_len, n_heads, head_dim]
    """
    global _flash_metal_kernel
    device = Device.DEFAULT.split(":")[0].lower()

    if device == "metal":
        if _flash_metal_kernel is None:
            from .flash_prefill_attention_metal import flash_prefill_attention_metal
            _flash_metal_kernel = flash_prefill_attention_metal
        return _flash_metal_kernel(query, key, value, causal)
    else:
        return flash_prefill_attention_tinygrad(query, key, value, causal)
