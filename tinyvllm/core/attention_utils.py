"""Paged Attention Implementation.

Phase 5: Fused paged attention with custom kernels.
Uses Metal/CUDA kernels when available, falls back to tinygrad ops.

Call chains from engine.py:

  PREFILL (engine.py:217-243) - one sequence at a time, multiple tokens:
    for seq in prefill_seqs:
        self.model()                   # Llama.__call__
            → layer()                  # TransformerBlock.__call__
                → self.attention()     # Attention.__call__
                    → paged_prefill_attention()
                          ├── repeat_kv()           # GQA head expansion
                          ├── create_causal_mask()  # lower triangular mask
                          └── attention()           # actual matmul + softmax

  DECODE (engine.py:245-265) - all sequences batched, 1 token each:
    self.model.batched_decode()        # all sequences at once
        → layer.batched_forward()      # TransformerBlock.batched_forward
            → self.attention.batched_forward()
                → paged_decode_attention()
                      └── paged_decode_attention()  # Metal/CUDA kernel
"""

from typing import List, Optional
import math

from tinygrad import Tensor

from ..kernels import paged_decode_attention as paged_decode_attention_kernel
from ..kernels import flash_prefill_attention
from ..kernels.paged_decode_attention_tinygrad import paged_decode_attention_tinygrad


def create_causal_mask(seq_len: int, start_pos: int = 0) -> Tensor:
    """
    Create a causal attention mask.

    Args:
        seq_len: Length of the query sequence
        start_pos: Starting position (for decode with KV cache)

    Returns:
        mask: Tensor of shape [seq_len, start_pos + seq_len]
              0 for valid positions, -inf for masked positions

    For prefill (start_pos=0, seq_len=5):
        [[0, -inf, -inf, -inf, -inf],
         [0,    0, -inf, -inf, -inf],
         [0,    0,    0, -inf, -inf],
         [0,    0,    0,    0, -inf],
         [0,    0,    0,    0,    0]]

    For decode (start_pos=5, seq_len=1):
        [[0, 0, 0, 0, 0, 0]]  # Can attend to all previous + self
    """
    # Decode: single token can attend to all previous tokens
    if seq_len == 1: return Tensor.zeros(1, start_pos + seq_len)

    # Create lower triangular mask (1 = can attend, 0 = cannot)
    mask = Tensor.ones(seq_len, seq_len).tril(0)

    # Can attend to all cached positions -> concatenate with the mask
    if start_pos > 0: mask = Tensor.ones(seq_len, start_pos).cat(mask, dim=1)

    # Convert to attention mask: 0 for valid, -inf for invalid
    return Tensor.where(mask == 1, Tensor.zeros_like(mask), Tensor.full_like(mask, float('-inf')))


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """
    Repeat KV heads to match query heads (for GQA).

    Args:
        x: Tensor of shape [batch, seq_len, n_kv_heads, head_dim]
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor of shape [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x

    batch, seq_len, n_kv_heads, head_dim = x.shape

    # Expand and repeat
    x = x.reshape(batch, seq_len, n_kv_heads, 1, head_dim)
    x = x.expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    x = x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
    return x


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """
    Standard scaled dot-product attention.

    Args:
        query: [batch, q_len, n_heads, head_dim]
        key: [batch, kv_len, n_heads, head_dim]
        value: [batch, kv_len, n_heads, head_dim]
        mask: Optional mask [batch, 1, q_len, kv_len] or broadcastable
        scale: Scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, q_len, n_heads, head_dim]
    """
    batch, q_len, n_heads, head_dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Transpose for matmul: [batch, n_heads, seq_len, head_dim]
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Compute attention scores: [batch, n_heads, q_len, kv_len]
    scores = (q @ k.transpose(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Softmax and output
    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v  # [batch, n_heads, q_len, head_dim]

    # Transpose back: [batch, q_len, n_heads, head_dim]
    return output.transpose(1, 2)


def paged_prefill_attention(
    query: Tensor,
    kv_cache,  # KVCache instance (block-based)
    block_table: List[int],
    context_len: int,
    layer_idx: int,
    start_pos: int = 0,
) -> Tensor:
    """
    Attention for prefill phase (single sequence, multiple tokens).

    Reads K/V from scattered blocks, handles GQA and causal masking, runs attention.
    Called via: engine → model() → layer() → attention() → paged_prefill_attention()

    Args:
        query: [1, q_len, n_heads, head_dim] - single sequence, q_len tokens
        kv_cache: Block-based KVCache instance
        block_table: List of physical block IDs for this sequence
        context_len: Total number of tokens to read from cache
        layer_idx: Which layer's K/V to use
        start_pos: Starting position for causal mask

    Returns:
        output: [1, q_len, n_heads, head_dim]
    """
    _, q_len, n_heads, head_dim = query.shape

    # Read K/V from scattered blocks
    k, v = kv_cache.read_kv_blocks(layer_idx, block_table, context_len)

    # Add batch dimension: [context_len, n_kv_heads, head_dim] -> [1, context_len, n_kv_heads, head_dim]
    if k.ndim == 3:
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    kv_len = k.shape[1]
    kv_heads = k.shape[2]

    # Handle GQA: repeat KV heads to match query heads
    if kv_heads != n_heads:
        n_rep = n_heads // kv_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

    # Compute effective start_pos from tensor shapes
    effective_start_pos = kv_len - q_len if kv_len > q_len else start_pos

    # Create causal mask
    mask = create_causal_mask(q_len, effective_start_pos)
    mask = mask.reshape(1, 1, q_len, kv_len)

    return attention(query, k, v, mask=mask)


def paged_decode_attention(
    queries: List[Tensor],
    kv_cache,
    block_tables: List[List[int]],
    context_lens: List[int],
    layer_idx: int,
    start_positions: List[int],
) -> Tensor:
    """
    Attention for decode phase (multiple sequences batched, 1 token each).

    Uses fused Metal/CUDA kernel for performance. Each sequence generates one token
    but we batch them together for GPU efficiency.
    Called via: engine → model.batched_decode() → layer.batched_forward() → paged_decode_attention()

    Args:
        queries: List of [1, 1, n_heads, head_dim] tensors, one per sequence
        kv_cache: Block-based KVCache instance
        block_tables: List of block tables, one per sequence
        context_lens: Context length for each sequence
        layer_idx: Which layer
        start_positions: Start position for each sequence (unused, kept for API consistency)

    Returns:
        output: [batch, 1, n_heads, head_dim]
    """
    if not queries:
        return Tensor.zeros(0, 1, 1, 1)

    # Get dimensions from first query
    _, _, n_heads, head_dim = queries[0].shape

    # Phase 5: Use fused kernel
    # Stack queries into single tensor [batch, 1, n_heads, head_dim]
    queries_stacked = Tensor.cat(*queries, dim=0)
    k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)

    return paged_decode_attention_kernel(
        queries_stacked, k_cache, v_cache, block_tables, context_lens,
        n_heads, kv_cache.n_kv_heads, head_dim, kv_cache.block_size
    )


def paged_decode_attention_with_tensors(
    queries: List[Tensor],
    kv_cache,
    block_tables_tensor: Tensor,
    context_lens_tensor: Tensor,
    layer_idx: int,
) -> Tensor:
    """
    Attention for decode phase using pre-built tensors.

    Phase 7.4 optimization: Accepts Tensors directly instead of Python lists,
    eliminating the per-layer list→tensor conversion overhead.

    Called via: engine → model.batched_decode() → layer.batched_forward()
                → attention.batched_forward() → paged_decode_attention_with_tensors()

    Args:
        queries: List of [1, 1, n_heads, head_dim] tensors, one per sequence
        kv_cache: Block-based KVCache instance
        block_tables_tensor: [batch, max_blocks] int32 - pre-built tensor
        context_lens_tensor: [batch] int32 - pre-built tensor
        layer_idx: Which layer

    Returns:
        output: [batch, 1, n_heads, head_dim]
    """
    if not queries:
        return Tensor.zeros(0, 1, 1, 1)

    # Get dimensions from first query
    _, _, n_heads, head_dim = queries[0].shape

    # Stack queries into single tensor [batch, 1, n_heads, head_dim]
    queries_stacked = Tensor.cat(*queries, dim=0)
    k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)

    # Direct call to tensor version - no list→tensor conversion
    return paged_decode_attention_tinygrad(
        queries_stacked, k_cache, v_cache,
        block_tables_tensor, context_lens_tensor,
        n_heads, kv_cache.n_kv_heads, head_dim, kv_cache.block_size
    )
