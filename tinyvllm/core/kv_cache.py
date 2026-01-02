"""KV Cache with Paged Storage.

Phase 2: Simple list-based implementation that stores K/V tensors in Python lists.
This works but is not optimized for performance.

TODO Phase 4: Replace with pre-allocated block tensors for better performance.
See docs/phase4_kv_optimization.md for the optimized implementation.
"""

from typing import List, Tuple, Dict

from tinygrad import Tensor
from tinygrad.dtype import DType


class KVCache:
    """
    Simple list-based KV cache for Phase 2.

    Stores K/V tensors in nested Python dicts/lists.
    Each sequence has its own list of K/V tensors.

    This approach:
    - Works with tinygrad's functional style (no scattered writes)
    - Easy to understand and debug
    - Not optimized for performance (Phase 4 will fix this)

    Storage structure:
        k_cache[layer_idx][seq_id] = [k_tensor_pos0, k_tensor_pos1, ...]
        v_cache[layer_idx][seq_id] = [v_tensor_pos0, v_tensor_pos1, ...]

    TODO Phase 4: Replace with pre-allocated block tensors:
        k_blocks[layer_idx][block_id] = Tensor of shape [block_size, n_kv_heads, head_dim]
        See docs/phase4_kv_optimization.md
    """

    def __init__(self, num_layers: int, num_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: DType):
        self.num_layers = num_layers
        self.num_blocks = num_blocks  # Phase 4: Pre-allocated block count
        self.block_size = block_size  # Phase 4: Tokens per block
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Phase 2: Simple list storage per layer per sequence
        # k_cache[layer_idx][seq_id] = list of K tensors (one per position)
        self.k_cache: List[Dict[int, List[Tensor]]] = [{} for _ in range(num_layers)]
        self.v_cache: List[Dict[int, List[Tensor]]] = [{} for _ in range(num_layers)]

    def allocate_sequence(self, seq_id: int):
        """Initialize empty K/V lists for a new sequence."""
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx][seq_id] = []
            self.v_cache[layer_idx][seq_id] = []

    def write_kv(self, layer_idx: int, seq_id: int, k: Tensor, v: Tensor):
        """
        Append K/V for new token position(s).

        Args:
            layer_idx: Which transformer layer
            seq_id: Which sequence
            k: Key tensor of shape [n_kv_heads, head_dim] (single position)
               or [seq_len, n_kv_heads, head_dim] (batch/prefill)
            v: Value tensor of same shape as k

        Note: In Phase 2, we ignore block_id and offset since we're not using blocks.
              We just append to the sequence's list.

        TODO Phase 4: Use block_id and offset for scattered writes:
            self.k_blocks[layer_idx][block_id][offset] = k
        """
        if seq_id not in self.k_cache[layer_idx]:
            self.k_cache[layer_idx][seq_id] = []
            self.v_cache[layer_idx][seq_id] = []

        if k.ndim == 2:
            # Single position: [n_kv_heads, head_dim]
            self.k_cache[layer_idx][seq_id].append(k)
            self.v_cache[layer_idx][seq_id].append(v)
        else:
            # Batch/prefill: [seq_len, n_kv_heads, head_dim]
            for i in range(k.shape[0]):
                self.k_cache[layer_idx][seq_id].append(k[i])
                self.v_cache[layer_idx][seq_id].append(v[i])

    def read_kv(self, layer_idx: int, seq_id: int) -> Tuple[Tensor, Tensor]:
        """
        Read all K/V for a sequence (for attention computation).

        Returns:
            k: Tensor of shape [context_len, n_kv_heads, head_dim]
            v: Tensor of shape [context_len, n_kv_heads, head_dim]

        TODO Phase 4: Use block_table to gather from scattered blocks:
            k = Tensor.stack(*[self.k_blocks[layer_idx][bid] for bid in block_ids])
        """
        k_list = self.k_cache[layer_idx][seq_id]
        v_list = self.v_cache[layer_idx][seq_id]

        if len(k_list) == 0:
            # Return empty tensors if no K/V stored yet
            return (
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype),
                Tensor.zeros(0, self.n_kv_heads, self.head_dim, dtype=self.dtype)
            )

        # Stack all K/V tensors into contiguous tensor
        k = Tensor.stack(*k_list)  # [context_len, n_kv_heads, head_dim]
        v = Tensor.stack(*v_list)
        return k, v

    def get_context_length(self, layer_idx: int, seq_id: int) -> int:
        """Get number of tokens stored for a sequence."""
        if seq_id not in self.k_cache[layer_idx]:
            return 0
        return len(self.k_cache[layer_idx][seq_id])

    def free_sequence(self, seq_id: int):
        """Release K/V storage for a completed sequence."""
        for layer_idx in range(self.num_layers):
            if seq_id in self.k_cache[layer_idx]:
                del self.k_cache[layer_idx][seq_id]
            if seq_id in self.v_cache[layer_idx]:
                del self.v_cache[layer_idx][seq_id]

    def get_memory_bytes(self) -> int:
        """Calculate total memory used by KV cache in bytes."""
        bytes_per_element = 4  # float32
        total = 0
        for layer_idx in range(self.num_layers):
            for seq_id in self.k_cache[layer_idx]:
                n_tokens = len(self.k_cache[layer_idx][seq_id])
                # K + V per token: 2 * n_tokens * n_kv_heads * head_dim * bytes
                total += 2 * n_tokens * self.n_kv_heads * self.head_dim * bytes_per_element
        return total

    def get_num_tokens(self) -> int:
        """Get total tokens stored across all sequences."""
        total = 0
        for layer_idx in range(self.num_layers):
            for seq_id in self.k_cache[layer_idx]:
                total += len(self.k_cache[layer_idx][seq_id])
        # Divide by num_layers since tokens are duplicated per layer
        return total // self.num_layers if self.num_layers > 0 else 0
