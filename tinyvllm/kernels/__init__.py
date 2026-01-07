"""Custom kernels for paged attention and flash attention.

- Metal: Custom Metal kernel with online softmax
- Other: Pure tinygrad fallback (device-agnostic)

Paged Attention: For decode phase (reading from block-based KV cache)
Flash Attention: For prefill phase (O(1) memory, direct on fresh K/V)
"""

from tinygrad import Device

# Pure tinygrad versions (device-agnostic)
from .paged_decode_attention_tinygrad import paged_decode_attention_tinygrad
from .paged_decode_attention_tinygrad import paged_decode_attention_from_lists

# Flash attention for prefill (Phase 8.1)
from .flash_prefill_attention_tinygrad import flash_prefill_attention_tinygrad

# Lazy kernel selection - checked at call time, not import time
_paged_metal_kernel = None
_flash_metal_kernel = None

def paged_decode_attention(*args, **kwargs):
    """Select paged decode kernel based on current device."""
    global _paged_metal_kernel
    device = Device.DEFAULT.split(":")[0].lower()

    if device == "metal":
        if _paged_metal_kernel is None:
            from .paged_decode_attention_metal import paged_decode_attention as metal_impl
            _paged_metal_kernel = metal_impl
        return _paged_metal_kernel(*args, **kwargs)
    else:
        return paged_decode_attention_from_lists(*args, **kwargs)


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
