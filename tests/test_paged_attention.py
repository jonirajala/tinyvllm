"""Tests for paged decode attention.

Tests correctness of paged attention against standard attention,
including GQA support and various batch/context configurations.
"""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import paged_decode_attention


def standard_attention(query, key, value, n_heads, n_kv_heads):
    """Reference standard attention for testing.

    Args:
        query: [batch, 1, n_heads, head_dim]
        key: [batch, context_len, n_kv_heads, head_dim]
        value: [batch, context_len, n_kv_heads, head_dim]

    Returns:
        [batch, 1, n_heads, head_dim]
    """
    batch, _, _, head_dim = query.shape
    _, context_len, _, _ = key.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Handle GQA by repeating K/V heads
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        key = key.reshape(batch, context_len, n_kv_heads, 1, head_dim)
        key = key.expand(batch, context_len, n_kv_heads, n_rep, head_dim)
        key = key.reshape(batch, context_len, n_heads, head_dim)

        value = value.reshape(batch, context_len, n_kv_heads, 1, head_dim)
        value = value.expand(batch, context_len, n_kv_heads, n_rep, head_dim)
        value = value.reshape(batch, context_len, n_heads, head_dim)

    # [batch, n_heads, 1, head_dim] @ [batch, n_heads, head_dim, context_len]
    q = query.transpose(1, 2)  # [batch, n_heads, 1, head_dim]
    k = key.transpose(1, 2)    # [batch, n_heads, context_len, head_dim]
    v = value.transpose(1, 2)  # [batch, n_heads, context_len, head_dim]

    scores = (q @ k.transpose(-2, -1)) * scale  # [batch, n_heads, 1, context_len]
    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v  # [batch, n_heads, 1, head_dim]

    return output.transpose(1, 2)  # [batch, 1, n_heads, head_dim]


def assert_allclose(actual: Tensor, expected: Tensor, rtol: float = 1e-4, atol: float = 1e-4):
    """Assert two tensors are element-wise close."""
    actual_data = actual.realize().flatten().tolist()
    expected_data = expected.realize().flatten().tolist()
    max_diff = 0.0
    for a, e in zip(actual_data, expected_data):
        diff = abs(a - e)
        thresh = atol + rtol * abs(e)
        if diff > thresh:
            max_diff = max(max_diff, diff)
    assert max_diff == 0.0, f"Max diff {max_diff} exceeds tolerance (rtol={rtol}, atol={atol})"


def setup_paged_cache(batch_size, context_len, n_kv_heads, head_dim, block_size):
    """Create K/V cache in paged format for testing."""
    num_blocks_per_seq = (context_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * batch_size + 5  # Extra unused blocks

    # Create cache
    k_cache = Tensor.randn(total_blocks, block_size, n_kv_heads, head_dim).realize()
    v_cache = Tensor.randn(total_blocks, block_size, n_kv_heads, head_dim).realize()

    # Create block tables - each sequence uses consecutive blocks
    block_tables_list = []
    for i in range(batch_size):
        blocks = list(range(i * num_blocks_per_seq, (i + 1) * num_blocks_per_seq))
        block_tables_list.append(blocks)

    max_blocks = max(len(bt) for bt in block_tables_list)
    padded = []
    for bt in block_tables_list:
        padded.extend(bt + [0] * (max_blocks - len(bt)))

    block_tables = Tensor(padded, dtype=dtypes.int32).reshape(batch_size, max_blocks).realize()
    context_lens = Tensor([context_len] * batch_size, dtype=dtypes.int32).realize()

    # Extract contiguous K/V for reference implementation
    k_contiguous = []
    v_contiguous = []
    for i in range(batch_size):
        k_seq = []
        v_seq = []
        for block_idx in block_tables_list[i]:
            k_seq.append(k_cache[block_idx])
            v_seq.append(v_cache[block_idx])
        k_contiguous.append(Tensor.cat(*k_seq, dim=0)[:context_len])
        v_contiguous.append(Tensor.cat(*v_seq, dim=0)[:context_len])

    k_ref = Tensor.stack(*k_contiguous, dim=0)  # [batch, context_len, n_kv_heads, head_dim]
    v_ref = Tensor.stack(*v_contiguous, dim=0)

    return k_cache, v_cache, block_tables, context_lens, k_ref, v_ref


class TestPagedAttentionBasic:
    """Basic correctness tests."""

    def test_output_shape(self):
        """Output shape should be [batch, 1, n_heads, head_dim]."""
        batch, n_heads, n_kv_heads, head_dim = 2, 8, 8, 64
        context_len, block_size = 32, 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, _, _ = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        assert out.shape == (batch, 1, n_heads, head_dim)

    def test_matches_standard_attention(self):
        """Should match standard attention output."""
        batch, n_heads, n_kv_heads, head_dim = 1, 4, 4, 32
        context_len, block_size = 32, 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        std_out = standard_attention(queries, k_ref, v_ref, n_heads, n_kv_heads)

        assert_allclose(paged_out, std_out, rtol=1e-3, atol=1e-3)


class TestPagedAttentionGQA:
    """Tests for Grouped Query Attention support."""

    def test_gqa_4x(self):
        """4x GQA (32 query heads, 8 KV heads)."""
        batch, n_heads, n_kv_heads, head_dim = 1, 32, 8, 64
        context_len, block_size = 32, 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        std_out = standard_attention(queries, k_ref, v_ref, n_heads, n_kv_heads)

        assert_allclose(paged_out, std_out, rtol=1e-3, atol=1e-3)

    def test_gqa_8x(self):
        """8x GQA (32 query heads, 4 KV heads) - TinyLlama config."""
        batch, n_heads, n_kv_heads, head_dim = 1, 32, 4, 64
        context_len, block_size = 64, 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        std_out = standard_attention(queries, k_ref, v_ref, n_heads, n_kv_heads)

        assert_allclose(paged_out, std_out, rtol=1e-3, atol=1e-3)


class TestPagedAttentionBatching:
    """Tests for batched attention."""

    def test_batch_size_4(self):
        """Should handle batch size 4."""
        batch, n_heads, n_kv_heads, head_dim = 4, 8, 8, 32
        context_len, block_size = 32, 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        std_out = standard_attention(queries, k_ref, v_ref, n_heads, n_kv_heads)

        assert_allclose(paged_out, std_out, rtol=1e-3, atol=1e-3)


class TestPagedAttentionContextLengths:
    """Tests for various context lengths."""

    @pytest.mark.parametrize("context_len", [16, 32, 64, 128])
    def test_various_context_lengths(self, context_len):
        """Should work with various context lengths."""
        batch, n_heads, n_kv_heads, head_dim = 1, 4, 4, 32
        block_size = 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        std_out = standard_attention(queries, k_ref, v_ref, n_heads, n_kv_heads)

        assert_allclose(paged_out, std_out, rtol=1e-3, atol=1e-3)

    def test_context_not_multiple_of_block_size(self):
        """Should handle context length not multiple of block size."""
        batch, n_heads, n_kv_heads, head_dim = 1, 4, 4, 32
        context_len, block_size = 25, 16  # 25 is not multiple of 16

        queries = Tensor.randn(batch, 1, n_heads, head_dim).realize()
        k_cache, v_cache, block_tables, context_lens, k_ref, v_ref = setup_paged_cache(
            batch, context_len, n_kv_heads, head_dim, block_size
        )

        paged_out = paged_decode_attention(
            queries, k_cache, v_cache, block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        # Output should be valid (no NaN/Inf)
        assert paged_out.isnan().sum().item() == 0
        assert paged_out.isinf().sum().item() == 0
