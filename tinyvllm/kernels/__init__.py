"""Custom kernels for Phase 5/6 - backend selected by device name convention.

Phase 6.2: Uses online softmax kernel with buffer pooling for Metal.
"""

import importlib
from tinygrad import Device

_device = Device.DEFAULT.split(":")[0].lower()
_module = importlib.import_module(f".paged_attention_{_device}", __package__)
fused_paged_attention = _module.fused_paged_attention
