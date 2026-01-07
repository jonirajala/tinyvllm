"""Attention kernels for tinyvllm.

All kernels use tinygrad Tensor operations for device-agnostic execution.
Works on any tinygrad backend (Metal, CUDA, CPU, etc.)
"""

from .paged_decode_attention import paged_decode_attention
from .flash_prefill_attention import flash_prefill_attention
