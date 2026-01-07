"""Pure tinygrad paged attention implementation.

Unlike the custom Metal/CUDA kernels which use device-specific APIs,
this implementation uses only tinygrad Tensor operations, making it:
- Device-agnostic (works on any tinygrad backend)

This is the portable fallback that trades some efficiency for compatibility.
"""

import math
from tinygrad import Tensor, dtypes


def paged_decode_attention_tinygrad(
    queries: Tensor,        # [batch, 1, n_heads, head_dim]
    k_cache: Tensor,        # [num_blocks, block_size, n_kv_heads, head_dim]
    v_cache: Tensor,        # [num_blocks, block_size, n_kv_heads, head_dim]
    block_tables: Tensor,   # [batch, max_blocks] int32 - TENSOR not list
    context_lens: Tensor,   # [batch] int32 - TENSOR not list
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context_len: int = None,  # Optional: limit reads to this context length
) -> Tensor:
    """Pure tinygrad paged attention.

    Uses only tinygrad Tensor operations for device-agnostic execution.

    Args:
        queries: [batch, 1, n_heads, head_dim]
        k_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        block_tables: [batch, max_blocks] int32 TENSOR
        context_lens: [batch] int32 TENSOR
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads
        head_dim: Dimension per head
        block_size: Tokens per block
        max_context_len: Optional limit on context (reduces memory reads)

    Returns:
        output: [batch, 1, n_heads, head_dim]
    """
    batch_size = queries.shape[0]
    total_max_blocks = block_tables.shape[1]

    # Determine how many blocks we actually need to read
    if max_context_len is not None:
        # Only read blocks needed for max_context_len
        blocks_needed = (max_context_len + block_size - 1) // block_size
        blocks_needed = min(blocks_needed, total_max_blocks)
        max_context = blocks_needed * block_size
        # Slice block_tables to only needed blocks
        block_tables = block_tables[:, :blocks_needed]
    else:
        blocks_needed = total_max_blocks
        max_context = total_max_blocks * block_size

    # Flatten block indices for gathering: [batch * blocks_needed]
    block_indices = block_tables.flatten()

    # Gather only needed blocks
    # k_cache[block_indices] -> [batch * blocks_needed, block_size, n_kv_heads, head_dim]
    k_gathered = k_cache[block_indices]
    v_gathered = v_cache[block_indices]

    # Reshape to [batch, max_context, n_kv_heads, head_dim]
    k_gathered = k_gathered.reshape(batch_size, max_context, n_kv_heads, head_dim)
    v_gathered = v_gathered.reshape(batch_size, max_context, n_kv_heads, head_dim)

    # Handle GQA: repeat KV heads to match query heads
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k_gathered = k_gathered.unsqueeze(3).expand(
            batch_size, max_context, n_kv_heads, n_rep, head_dim
        ).reshape(batch_size, max_context, n_heads, head_dim)
        v_gathered = v_gathered.unsqueeze(3).expand(
            batch_size, max_context, n_kv_heads, n_rep, head_dim
        ).reshape(batch_size, max_context, n_heads, head_dim)

    # Reshape query: [batch, 1, n_heads, head_dim] -> [batch, n_heads, 1, head_dim]
    q = queries.transpose(1, 2)

    # Transpose K/V: [batch, max_context, n_heads, head_dim] -> [batch, n_heads, max_context, head_dim]
    k = k_gathered.transpose(1, 2)
    v = v_gathered.transpose(1, 2)

    # Compute attention scores: [batch, n_heads, 1, max_context]
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale

    # Create causal mask based on context_lens
    # positions: [1, max_context] -> 0, 1, 2, ...
    positions = Tensor.arange(max_context).reshape(1, max_context)
    # context_lens: [batch, 1]
    ctx_lens_expanded = context_lens.reshape(batch_size, 1)

    # Mask: 1 where position < context_len, 0 otherwise
    valid_mask = (positions < ctx_lens_expanded).cast(dtypes.float32)

    # Convert to attention mask: 0 for valid, -inf for invalid
    # [batch, max_context] -> [batch, 1, 1, max_context]
    attn_mask = (1.0 - valid_mask) * (-1e9)
    attn_mask = attn_mask.reshape(batch_size, 1, 1, max_context)

    # Apply mask and softmax
    scores = scores + attn_mask
    attn_weights = scores.softmax(axis=-1)

    # Compute output: [batch, n_heads, 1, head_dim]
    output = attn_weights @ v

    # Transpose back: [batch, 1, n_heads, head_dim]
    return output.transpose(1, 2)
