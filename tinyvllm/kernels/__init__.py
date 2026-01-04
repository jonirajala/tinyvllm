"""Custom kernels for paged attention.

- Metal: Custom Metal kernel with online softmax
- Other: Pure tinygrad fallback (device-agnostic)
"""

from tinygrad import Device

# Pure tinygrad versions (device-agnostic, works with TinyJit)
from .paged_attention_tinygrad import fused_paged_attention_tinygrad
from .paged_attention_tinygrad import fused_paged_attention_from_lists

# Lazy kernel selection - checked at call time, not import time
_metal_kernel = None

def fused_paged_attention(*args, **kwargs):
    """Select kernel based on current device."""
    global _metal_kernel
    device = Device.DEFAULT.split(":")[0].lower()

    if device == "metal":
        if _metal_kernel is None:
            from .paged_attention_metal import fused_paged_attention as metal_impl
            _metal_kernel = metal_impl
        return _metal_kernel(*args, **kwargs)
    else:
        return fused_paged_attention_from_lists(*args, **kwargs)
