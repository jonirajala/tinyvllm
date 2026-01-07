"""Tests for Flash Attention implementation.

Tests correctness of Flash Attention against standard attention,
including causal masking, GQA support, and various sequence lengths.
"""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import flash_prefill_attention as _flash_kernel


def flash_prefill_attention(q, k, v, causal=True):
    """Wrapper that handles batch dimension for tests."""
    out = _flash_kernel(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal)
    return out.squeeze(0)


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


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match query heads (for GQA)."""
    if n_rep == 1:
        return x
    batch, seq_len, n_kv_heads, head_dim = x.shape
    x = x.reshape(batch, seq_len, n_kv_heads, 1, head_dim)
    x = x.expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return x.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


def _attention(query: Tensor, key: Tensor, value: Tensor, mask=None) -> Tensor:
    """Standard scaled dot-product attention."""
    batch, q_len, n_heads, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask

    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v
    return output.transpose(1, 2)


def _create_causal_mask(seq_len: int) -> Tensor:
    """Create causal attention mask."""
    mask = Tensor.ones(seq_len, seq_len).tril(0)
    return Tensor.where(mask == 1, Tensor.zeros_like(mask), Tensor.full_like(mask, float('-inf')))


def standard_attention_with_gqa(queries, keys, values, causal=True):
    """Reference implementation for testing."""
    queries = queries.unsqueeze(0)
    keys = keys.unsqueeze(0)
    values = values.unsqueeze(0)

    _, q_len, n_heads, head_dim = queries.shape
    _, kv_len, n_kv_heads, _ = keys.shape

    k, v = keys, values
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)

    mask = None
    if causal:
        mask = _create_causal_mask(q_len).reshape(1, 1, q_len, q_len)

    return _attention(queries, k, v, mask=mask).squeeze(0)


class TestFlashAttentionBasic:
    def test_output_shape(self):
        q = Tensor.randn(16, 8, 64)
        k = Tensor.randn(16, 8, 64)
        v = Tensor.randn(16, 8, 64)
        out = flash_prefill_attention(q, k, v, causal=True)
        assert out.shape == (16, 8, 64)

    def test_matches_standard_attention_no_gqa(self):
        q = Tensor.randn(32, 4, 32)
        k = Tensor.randn(32, 4, 32)
        v = Tensor.randn(32, 4, 32)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_matches_standard_attention_with_gqa(self):
        q = Tensor.randn(32, 32, 64)
        k = Tensor.randn(32, 8, 64)
        v = Tensor.randn(32, 8, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_non_causal(self):
        q = Tensor.randn(16, 4, 32)
        k = Tensor.randn(16, 4, 32)
        v = Tensor.randn(16, 4, 32)
        std_out = _attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask=None).squeeze(0)
        flash_out = flash_prefill_attention(q, k, v, causal=False)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestCausalMasking:
    def test_causal_mask_first_token(self):
        q_data = [[0.0] * 16] * 2
        q_first = [[1.0] * 16] * 2
        q = Tensor([q_first] + [q_data] * 7)
        k = Tensor.randn(8, 2, 16)
        v = Tensor.eye(8).unsqueeze(1).expand(8, 2, 8)
        out = flash_prefill_attention(q, k, v, causal=True)
        assert out[0, 0, 0].item() > 0.5

    def test_last_token_attends_to_all(self):
        q = Tensor.ones(8, 2, 16)
        k = Tensor.zeros(8, 2, 16)
        v = Tensor.arange(8).reshape(8, 1, 1).expand(8, 2, 16).float()
        out = flash_prefill_attention(q, k, v, causal=True)
        last_token_out = out[-1, 0, 0].item()
        expected = sum(range(8)) / 8
        assert abs(last_token_out - expected) < 1e-2


class TestGQA:
    def test_gqa_2x(self):
        q = Tensor.randn(16, 8, 32)
        k = Tensor.randn(16, 4, 32)
        v = Tensor.randn(16, 4, 32)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_gqa_4x(self):
        q = Tensor.randn(16, 32, 64)
        k = Tensor.randn(16, 8, 64)
        v = Tensor.randn(16, 8, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_gqa_8x(self):
        q = Tensor.randn(16, 32, 64)
        k = Tensor.randn(16, 4, 64)
        v = Tensor.randn(16, 4, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestSequenceLengths:
    @pytest.mark.parametrize("q_len", [1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 128])
    def test_various_lengths(self, q_len):
        q = Tensor.randn(q_len, 4, 32)
        k = Tensor.randn(q_len, 4, 32)
        v = Tensor.randn(q_len, 4, 32)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_single_token(self):
        q = Tensor.randn(1, 8, 64)
        k = Tensor.randn(1, 8, 64)
        v = Tensor.randn(1, 8, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestHeadDimensions:
    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_various_head_dims(self, head_dim):
        q = Tensor.randn(16, 4, head_dim)
        k = Tensor.randn(16, 4, head_dim)
        v = Tensor.randn(16, 4, head_dim)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestDtypes:
    def test_float16(self):
        q = Tensor.randn(32, 4, 32, dtype=dtypes.float16)
        k = Tensor.randn(32, 4, 32, dtype=dtypes.float16)
        v = Tensor.randn(32, 4, 32, dtype=dtypes.float16)
        out = flash_prefill_attention(q, k, v, causal=True)
        assert out.shape == (32, 4, 32)
        assert out.isnan().sum().item() == 0
        assert out.isinf().sum().item() == 0

    def test_float32(self):
        q = Tensor.randn(32, 4, 32, dtype=dtypes.float32)
        k = Tensor.randn(32, 4, 32, dtype=dtypes.float32)
        v = Tensor.randn(32, 4, 32, dtype=dtypes.float32)
        out = flash_prefill_attention(q, k, v, causal=True)
        assert out.dtype == dtypes.float32
        assert out.shape == (32, 4, 32)


class TestTinyLlamaConfig:
    def test_tinyllama_config(self):
        q = Tensor.randn(64, 32, 64)
        k = Tensor.randn(64, 8, 64)
        v = Tensor.randn(64, 8, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_tinyllama_longer_context(self):
        q = Tensor.randn(256, 32, 64)
        k = Tensor.randn(256, 8, 64)
        v = Tensor.randn(256, 8, 64)
        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention(q, k, v, causal=True)
        assert_allclose(flash_out, std_out, rtol=1e-3, atol=1e-3)
