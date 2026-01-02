"""Paged Attention Implementation.

Phase 4: Block-based KVCache with paged attention.
Supports both single-sequence and batched decode operations.
"""

from typing import List, Optional
import math

from tinygrad import Tensor


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
    total_len = start_pos + seq_len

    if seq_len == 1:
        # Decode: single token can attend to all previous tokens
        return Tensor.zeros(1, total_len)

    # Prefill: causal mask
    # Create lower triangular mask (1 = can attend, 0 = cannot)
    mask = Tensor.ones(seq_len, seq_len).tril(0)

    if start_pos > 0:
        # Can attend to all cached positions
        prefix = Tensor.ones(seq_len, start_pos)
        mask = prefix.cat(mask, dim=1)

    # Convert to attention mask: 0 for valid, -inf for invalid
    # Use where to avoid NaN from 0 * -inf
    mask = Tensor.where(mask == 1, Tensor.zeros_like(mask), Tensor.full_like(mask, float('-inf')))
    return mask


def create_padding_mask(context_lens: List[int], max_len: int) -> Tensor:
    """
    Create mask to hide padding positions beyond each sequence's actual length.

    Phase 5: Used for true batched attention with multiple sequences.

    Args:
        context_lens: Actual length of each sequence in batch
        max_len: Maximum sequence length (padded length)

    Returns:
        mask: Tensor of shape [batch, 1, 1, max_len]
              0 for valid positions, -inf for padding
    """
    batch_size = len(context_lens)

    # Create position indices [0, 1, 2, ..., max_len-1]
    positions = Tensor.arange(max_len).reshape(1, max_len)

    # Create length tensor [len0, len1, ...]
    lengths = Tensor(context_lens).reshape(batch_size, 1)

    # Valid where position < length
    valid = (positions < lengths).float()  # [batch, max_len], 1 = valid, 0 = padding

    # Convert to mask: 0 for valid, -inf for padding
    mask = Tensor.where(valid == 1, Tensor.zeros_like(valid), Tensor.full_like(valid, float('-inf')))
    return mask.reshape(batch_size, 1, 1, max_len)


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
    kv_len = key.shape[1]

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


def paged_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    context_lens: List[int],
    start_pos: int = 0,
) -> Tensor:
    """
    Compute attention with paged KV cache.

    K/V are contiguous tensors from KVCache.read_kv_blocks().
    Handles GQA head expansion, causal masking, and padding masking.

    Args:
        query: [batch, q_len, n_heads, head_dim]
        key: [batch, kv_len, n_kv_heads, head_dim]
        value: [batch, kv_len, n_kv_heads, head_dim]
        context_lens: Actual length of each sequence (for padding mask)
        start_pos: Starting position for causal mask

    Returns:
        output: [batch, q_len, n_heads, head_dim]
    """
    batch, q_len, n_heads, head_dim = query.shape
    _, kv_len, kv_heads, _ = key.shape

    # Handle GQA: repeat KV heads to match query heads
    if kv_heads != n_heads:
        n_rep = n_heads // kv_heads
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)

    # Compute effective start_pos from tensor shapes
    # If kv_len > q_len, there are cached positions we need to account for
    effective_start_pos = kv_len - q_len if kv_len > q_len else start_pos

    # Create causal mask
    causal_mask = create_causal_mask(q_len, effective_start_pos)
    causal_mask = causal_mask.reshape(1, 1, q_len, kv_len)

    # Create padding mask if needed (for batched sequences of different lengths)
    if batch > 1 and len(set(context_lens)) > 1:
        padding_mask = create_padding_mask(context_lens, kv_len)
        mask = causal_mask + padding_mask
    else:
        mask = causal_mask

    # Compute attention
    return attention(query, key, value, mask=mask)


def paged_attention_with_blocks(
    query: Tensor,
    kv_cache,  # KVCache instance (block-based)
    block_table: List[int],
    context_len: int,
    layer_idx: int,
    start_pos: int = 0,
) -> Tensor:
    """
    Compute attention reading from block-based KVCache.

    Args:
        query: [1, q_len, n_heads, head_dim] - single sequence
        kv_cache: Block-based KVCache instance
        block_table: List of physical block IDs for this sequence
        context_len: Total number of tokens to read
        layer_idx: Which layer's K/V to use
        start_pos: Starting position for causal mask

    Returns:
        output: [1, q_len, n_heads, head_dim]
    """
    # Read K/V from scattered blocks
    k, v = kv_cache.read_kv_blocks(layer_idx, block_table, context_len)

    # Add batch dimension: [context_len, n_kv_heads, head_dim] -> [1, context_len, n_kv_heads, head_dim]
    if k.ndim == 3:
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    kv_len = k.shape[1]
    context_lens = [kv_len]

    return paged_attention(query, k, v, context_lens, start_pos)


def batched_paged_attention(
    queries: List[Tensor],
    kv_cache,
    block_tables: List[List[int]],
    context_lens: List[int],
    layer_idx: int,
    start_positions: List[int],
) -> Tensor:
    """
    Batched attention for multiple decode sequences.

    For decode (single token per sequence), compute attention for all sequences
    and stack results. Each sequence reads from its own block table.

    Args:
        queries: List of [1, 1, n_heads, head_dim] tensors, one per sequence
        kv_cache: Block-based KVCache instance
        block_tables: List of block tables, one per sequence
        context_lens: Context length for each sequence
        layer_idx: Which layer
        start_positions: Start position for each sequence

    Returns:
        output: [batch, 1, n_heads, head_dim]
    """
    outputs = []
    for q, bt, ctx_len, start_pos in zip(queries, block_tables, context_lens, start_positions):
        out = paged_attention_with_blocks(q, kv_cache, bt, ctx_len, layer_idx, start_pos)
        outputs.append(out)

    return Tensor.cat(*outputs, dim=0)
