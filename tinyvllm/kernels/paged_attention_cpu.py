"""CPU/Fallback implementation of fused paged attention.

Phase 5: Pure tinygrad implementation that works on any backend.
Used when custom kernels are not available.
"""

from typing import List
import math

from tinygrad import Tensor


def paged_attention_cpu(
    query: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_table: List[int],
    context_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> Tensor:
    """
    Paged attention using tinygrad ops (fallback implementation).

    Still reads from scattered blocks but uses tinygrad's gather operation
    instead of Python loops where possible.

    Args:
        query: [1, 1, n_heads, head_dim]
        k_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        block_table: List of physical block IDs
        context_len: Number of tokens in context
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads
        head_dim: Dimension per head
        block_size: Tokens per block

    Returns:
        output: [1, 1, n_heads, head_dim]
    """
    # Gather K/V from blocks into contiguous tensors
    # This is the Tensor.stack approach but optimized
    num_blocks_needed = (context_len + block_size - 1) // block_size

    k_gathered = Tensor.stack(*[k_cache[block_table[i]] for i in range(num_blocks_needed)])
    v_gathered = Tensor.stack(*[v_cache[block_table[i]] for i in range(num_blocks_needed)])

    # Reshape to [context slots, n_kv_heads, head_dim]
    total_slots = num_blocks_needed * block_size
    k_flat = k_gathered.reshape(total_slots, n_kv_heads, head_dim)[:context_len]
    v_flat = v_gathered.reshape(total_slots, n_kv_heads, head_dim)[:context_len]

    # Handle GQA: repeat KV heads
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        # [context_len, n_kv_heads, head_dim] -> [context_len, n_heads, head_dim]
        k_flat = k_flat.unsqueeze(2).expand(context_len, n_kv_heads, n_rep, head_dim)
        k_flat = k_flat.reshape(context_len, n_heads, head_dim)
        v_flat = v_flat.unsqueeze(2).expand(context_len, n_kv_heads, n_rep, head_dim)
        v_flat = v_flat.reshape(context_len, n_heads, head_dim)

    # Reshape query: [1, 1, n_heads, head_dim] -> [1, n_heads, 1, head_dim]
    q = query.reshape(1, n_heads, 1, head_dim)

    # K/V: [context_len, n_heads, head_dim] -> [1, context_len, n_heads, head_dim]
    k = k_flat.unsqueeze(0)
    v = v_flat.unsqueeze(0)

    # Transpose for attention: [1, n_heads, seq_len, head_dim]
    k = k.permute(0, 2, 1, 3)  # [1, n_heads, context_len, head_dim]
    v = v.permute(0, 2, 1, 3)  # [1, n_heads, context_len, head_dim]

    # Compute attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale  # [1, n_heads, 1, context_len]
    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v  # [1, n_heads, 1, head_dim]

    # Reshape to expected output: [1, 1, n_heads, head_dim]
    return output.transpose(1, 2)


def batched_paged_attention_cpu(
    queries: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: List[List[int]],
    context_lens: List[int],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> Tensor:
    """
    Batched paged attention using tinygrad ops (fallback).

    Processes each sequence separately and stacks results.

    Args:
        queries: [batch, 1, n_heads, head_dim]
        k_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, n_kv_heads, head_dim]
        block_tables: List of block tables per sequence
        context_lens: Context length for each sequence
        n_heads: Number of query heads
        n_kv_heads: Number of KV heads
        head_dim: Dimension per head
        block_size: Tokens per block

    Returns:
        output: [batch, 1, n_heads, head_dim]
    """
    outputs = []
    for i, (block_table, ctx_len) in enumerate(zip(block_tables, context_lens)):
        q = queries[i:i+1]  # [1, 1, n_heads, head_dim]
        out = paged_attention_cpu(
            q, k_cache, v_cache, block_table, ctx_len,
            n_heads, n_kv_heads, head_dim, block_size
        )
        outputs.append(out)

    return Tensor.cat(*outputs, dim=0)


# Export with standard name for dispatcher
fused_paged_attention = batched_paged_attention_cpu
