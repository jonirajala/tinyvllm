"""KV Cache with Block-based Storage.

Phase 4: Pre-allocated block tensors for efficient paged attention.
Each block is a separate realized tensor that accepts direct writes.
"""

from typing import List, Tuple

from tinygrad import Tensor
from tinygrad.dtype import DType


class KVCache:
    """
    Block-based KV cache for Phase 4.

    Storage: Pre-allocated block tensors per layer.
        k_blocks[layer_idx][block_id] = Tensor[block_size, n_kv_heads, head_dim]
        v_blocks[layer_idx][block_id] = Tensor[block_size, n_kv_heads, head_dim]

    Each block is a separate realized tensor, allowing direct slice assignment.
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: DType):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Pre-allocate block tensors - each block is separate realized tensor
        self.k_blocks: List[List[Tensor]] = [
            [Tensor.zeros(block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
             for _ in range(num_blocks)]
            for _ in range(num_layers)
        ]
        self.v_blocks: List[List[Tensor]] = [
            [Tensor.zeros(block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
             for _ in range(num_blocks)]
            for _ in range(num_layers)
        ]

    def write_kv(self, layer_idx: int, block_id: int, offset: int, k: Tensor, v: Tensor):
        """
        Write K/V for a single token to a specific block slot.

        Args:
            layer_idx: Which transformer layer
            block_id: Physical block ID
            offset: Position within block (0 to block_size-1)
            k: Key tensor [n_kv_heads, head_dim]
            v: Value tensor [n_kv_heads, head_dim]
        """
        # Direct assignment works on realized tensors
        self.k_blocks[layer_idx][block_id][offset] = k
        self.k_blocks[layer_idx][block_id] = self.k_blocks[layer_idx][block_id].realize()

        self.v_blocks[layer_idx][block_id][offset] = v
        self.v_blocks[layer_idx][block_id] = self.v_blocks[layer_idx][block_id].realize()

    def write_kv_batch(self, layer_idx: int, block_id: int, start_offset: int, k: Tensor, v: Tensor):
        """
        Write K/V for multiple tokens to a block (for prefill).

        Args:
            layer_idx: Which transformer layer
            block_id: Physical block ID
            start_offset: Starting position within block
            k: Key tensor [num_tokens, n_kv_heads, head_dim]
            v: Value tensor [num_tokens, n_kv_heads, head_dim]
        """
        num_tokens = k.shape[0]
        end_offset = start_offset + num_tokens

        # Write slice
        self.k_blocks[layer_idx][block_id][start_offset:end_offset] = k
        self.k_blocks[layer_idx][block_id] = self.k_blocks[layer_idx][block_id].realize()

        self.v_blocks[layer_idx][block_id][start_offset:end_offset] = v
        self.v_blocks[layer_idx][block_id] = self.v_blocks[layer_idx][block_id].realize()

    def read_kv_blocks(self, layer_idx: int, block_ids: List[int], context_len: int) -> Tuple[Tensor, Tensor]:
        """
        Read K/V from specified blocks.

        Args:
            layer_idx: Which transformer layer
            block_ids: List of physical block IDs (block table)
            context_len: Total tokens to read (handles partial last block)

        Returns:
            k: Tensor [context_len, n_kv_heads, head_dim]
            v: Tensor [context_len, n_kv_heads, head_dim]
        """
        if not block_ids:
            return (
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype),
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype)
            )

        # Stack all blocks: [num_blocks, block_size, n_kv_heads, head_dim]
        k_stacked = Tensor.stack(*[self.k_blocks[layer_idx][bid] for bid in block_ids])
        v_stacked = Tensor.stack(*[self.v_blocks[layer_idx][bid] for bid in block_ids])

        # Reshape to [num_blocks * block_size, n_kv_heads, head_dim]
        total_slots = len(block_ids) * self.block_size
        k_flat = k_stacked.reshape(total_slots, self.n_kv_heads, self.head_dim)
        v_flat = v_stacked.reshape(total_slots, self.n_kv_heads, self.head_dim)

        # Slice to actual context length (handles partial last block)
        k = k_flat[:context_len]
        v = v_flat[:context_len]

        return k, v

    def get_memory_bytes(self) -> int:
        """Calculate total memory used by KV cache in bytes."""
        bytes_per_element = 4  # float32
        # All blocks are pre-allocated
        total_elements = (self.num_layers * self.num_blocks * self.block_size *
                         self.n_kv_heads * self.head_dim * 2)  # K + V
        return total_elements * bytes_per_element
