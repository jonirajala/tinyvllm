"""KV Cache - Block-based storage for paged attention.

KVCache owns storage and writes. Attention kernels get direct read access
to k_cache/v_cache tensors for efficient batched indexing.
"""

from typing import List, Tuple

from tinygrad import Tensor
from tinygrad.dtype import DType


class KVCache:
    """Block-based KV cache. Shape per layer: [num_blocks, block_size, n_kv_heads, head_dim]."""

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: DType):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.k_cache: List[Tensor] = [
            Tensor.zeros(num_blocks, block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
            for _ in range(num_layers)
        ]
        self.v_cache: List[Tensor] = [
            Tensor.zeros(num_blocks, block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
            for _ in range(num_layers)
        ]

    def write_kv(self, layer_idx: int, block_id: int, start_offset: int, k: Tensor, v: Tensor) -> None:
        """Write K/V tokens to a block. k/v shape: [num_tokens, n_kv_heads, head_dim]."""
        end_offset = start_offset + k.shape[0]
        self.k_cache[layer_idx][block_id, start_offset:end_offset] = k
        self.v_cache[layer_idx][block_id, start_offset:end_offset] = v

    def write_kv(self, layer_idx: int, block_ids: List[int], context_len: int) -> Tuple[Tensor, Tensor]:
        """Read K/V from blocks. Returns tensors of shape [context_len, n_kv_heads, head_dim]."""
        if not block_ids:
            return (
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype),
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype)
            )

        k_stacked = Tensor.stack(*[self.k_cache[layer_idx][bid] for bid in block_ids])
        v_stacked = Tensor.stack(*[self.v_cache[layer_idx][bid] for bid in block_ids])

        total_slots = len(block_ids) * self.block_size
        k_flat = k_stacked.reshape(total_slots, self.n_kv_heads, self.head_dim)
        v_flat = v_stacked.reshape(total_slots, self.n_kv_heads, self.head_dim)

        return k_flat[:context_len], v_flat[:context_len]

    def get_memory_bytes(self) -> int:
        """Calculate total memory used by KV cache in bytes."""
        total_elements = (self.num_layers * self.num_blocks * self.block_size *
                         self.n_kv_heads * self.head_dim * 2)  # K + V
        return total_elements * self.dtype.itemsize
