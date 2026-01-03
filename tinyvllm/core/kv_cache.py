"""KV Cache with Block-based Storage.

Phase 5: Flat tensor storage for efficient kernel access.
All blocks for a layer are stored in a single contiguous tensor,
enabling fused paged attention kernels.
"""

from typing import List, Tuple

from tinygrad import Tensor
from tinygrad.dtype import DType


class KVCache:
    """
    Block-based KV cache with flat tensor storage.

    Phase 5: Stores all blocks in contiguous tensors per layer:
        k_cache[layer_idx] = Tensor[num_blocks, block_size, n_kv_heads, head_dim]
        v_cache[layer_idx] = Tensor[num_blocks, block_size, n_kv_heads, head_dim]

    This allows fused kernels to directly index into blocks without gathering.
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: DType):
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Phase 5: Flat tensor storage - all blocks in single tensor per layer
        # Shape: [num_blocks, block_size, n_kv_heads, head_dim]
        self.k_cache: List[Tensor] = [
            Tensor.zeros(num_blocks, block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
            for _ in range(num_layers)
        ]
        self.v_cache: List[Tensor] = [
            Tensor.zeros(num_blocks, block_size, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
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
        # Phase 5: Write to flat tensor [num_blocks, block_size, n_kv_heads, head_dim]
        self.k_cache[layer_idx][block_id, offset] = k
        self.k_cache[layer_idx] = self.k_cache[layer_idx].realize()

        self.v_cache[layer_idx][block_id, offset] = v
        self.v_cache[layer_idx] = self.v_cache[layer_idx].realize()

    def write_kv_batch(self, layer_idx: int, block_id: int, start_offset: int, k: Tensor, v: Tensor):
        """
        Write K/V for multiple tokens to a block (for prefill).
        in phase 6 we will batch write tokens to the KV cache.

        Args:
            layer_idx: Which transformer layer
            block_id: Physical block ID
            start_offset: Starting position within block
            k: Key tensor [num_tokens, n_kv_heads, head_dim]
            v: Value tensor [num_tokens, n_kv_heads, head_dim]
        """
        end_offset = start_offset + k.shape[0] # start_position + num_tokens

        self.k_cache[layer_idx][block_id, start_offset:end_offset] = k
        self.k_cache[layer_idx] = self.k_cache[layer_idx].realize()

        self.v_cache[layer_idx][block_id, start_offset:end_offset] = v
        self.v_cache[layer_idx] = self.v_cache[layer_idx].realize()

    def get_cache_tensors(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the flat K/V cache tensors for a layer.

        Phase 5: Used by fused kernels for direct block access.

        Args:
            layer_idx: Which transformer layer

        Returns:
            k_cache: Tensor [num_blocks, block_size, n_kv_heads, head_dim]
            v_cache: Tensor [num_blocks, block_size, n_kv_heads, head_dim]
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

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
            print("Warning: read_kv_blocks called with empty block_ids")
            return (
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype),
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype)
            )

        # Phase 5: Index into flat tensors and stack
        k_stacked = Tensor.stack(*[self.k_cache[layer_idx][bid] for bid in block_ids])
        v_stacked = Tensor.stack(*[self.v_cache[layer_idx][bid] for bid in block_ids])

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
