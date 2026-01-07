"""Flash Attention for prefill phase.

Uses tinygrad Tensor operations for device-agnostic execution.
Works on any tinygrad backend (Metal, CUDA, CPU, etc.)

Features:
- GQA support (n_kv_heads → n_heads)
- Causal masking
"""

import math
from tinygrad import Tensor, dtypes


def flash_prefill_attention(
    queries: Tensor,   # [1, q_len, n_heads, head_dim]
    keys: Tensor,      # [1, kv_len, n_kv_heads, head_dim]
    values: Tensor,    # [1, kv_len, n_kv_heads, head_dim]
    causal: bool = True,
) -> Tensor:
    """Flash Attention for prefill phase.

    Args:
        queries: [1, q_len, n_heads, head_dim]
        keys: [1, kv_len, n_kv_heads, head_dim]
        values: [1, kv_len, n_kv_heads, head_dim]
        causal: Apply causal masking (default True)

    Returns:
        output: [1, q_len, n_heads, head_dim]
    """
    # Remove batch dimension
    queries = queries.squeeze(0)  # [q_len, n_heads, head_dim]
    keys = keys.squeeze(0)        # [kv_len, n_kv_heads, head_dim]
    values = values.squeeze(0)    # [kv_len, n_kv_heads, head_dim]

    q_len, n_heads, head_dim = queries.shape
    kv_len, n_kv_heads, _ = keys.shape

    scale = 1.0 / math.sqrt(head_dim)

    # Handle GQA: expand KV heads to match Q heads
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        # [kv_len, n_kv_heads, head_dim] -> [kv_len, n_heads, head_dim]
        keys = keys.unsqueeze(2).expand(
            kv_len, n_kv_heads, n_rep, head_dim
        ).reshape(kv_len, n_heads, head_dim)
        values = values.unsqueeze(2).expand(
            kv_len, n_kv_heads, n_rep, head_dim
        ).reshape(kv_len, n_heads, head_dim)

    # Transpose for batched matmul: [n_heads, seq_len, head_dim]
    q = queries.transpose(0, 1)  # [n_heads, q_len, head_dim]
    k = keys.transpose(0, 1)     # [n_heads, kv_len, head_dim]
    v = values.transpose(0, 1)   # [n_heads, kv_len, head_dim]

    # Compute attention scores: [n_heads, q_len, kv_len]
    # Note: realize() breaks the lazy graph to avoid tinygrad TC optimizer bug
    scores = ((q @ k.transpose(-2, -1)) * scale).realize()

    # Apply causal mask if needed
    if causal:
        # Create causal mask: position i can only attend to positions <= i
        # For prefill, q_positions == kv_positions (0, 1, 2, ...)
        q_positions = Tensor.arange(q_len).reshape(q_len, 1)
        kv_positions = Tensor.arange(kv_len).reshape(1, kv_len)
        # mask is True where we should NOT attend (kv > q)
        mask = (kv_positions > q_positions).cast(dtypes.float32) * (-1e9)
        # Broadcast to [n_heads, q_len, kv_len]
        scores = scores + mask.unsqueeze(0)

    # Softmax and weighted sum
    attn_weights = scores.softmax(axis=-1)  # [n_heads, q_len, kv_len]
    output = attn_weights @ v                # [n_heads, q_len, head_dim]

    # Transpose back and add batch dimension: [1, q_len, n_heads, head_dim]
    return output.transpose(0, 1).unsqueeze(0)
