"""Tests for Flash Attention implementation (Phase 8.1).

Tests correctness of Flash Attention against standard attention,
including causal masking, GQA support, and various sequence lengths.
"""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.kernels.flash_prefill_attention_tinygrad import flash_prefill_attention_tinygrad as _flash_kernel
from tinyvllm.core.attention_utils import (
    attention,
    create_causal_mask,
    repeat_kv,
)


def flash_prefill_attention_tinygrad(q, k, v, causal=True):
    """Wrapper that handles batch dimension for tests.

    Tests use [q_len, n_heads, head_dim] format.
    Kernel expects [1, q_len, n_heads, head_dim] format.
    """
    out = _flash_kernel(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal)
    return out.squeeze(0)


def assert_allclose(actual: Tensor, expected: Tensor, rtol: float = 1e-4, atol: float = 1e-4):
    """Assert two tensors are element-wise close (replacement for np.testing.assert_allclose)."""
    actual_data = actual.realize().flatten().tolist()
    expected_data = expected.realize().flatten().tolist()
    max_diff = 0.0
    for a, e in zip(actual_data, expected_data):
        diff = abs(a - e)
        thresh = atol + rtol * abs(e)
        if diff > thresh:
            max_diff = max(max_diff, diff)
    assert max_diff == 0.0, f"Max diff {max_diff} exceeds tolerance (rtol={rtol}, atol={atol})"


def standard_attention_with_gqa(
    queries: Tensor,  # [q_len, n_heads, head_dim]
    keys: Tensor,     # [kv_len, n_kv_heads, head_dim]
    values: Tensor,   # [kv_len, n_kv_heads, head_dim]
    causal: bool = True,
) -> Tensor:
    """Reference implementation using existing attention utilities.

    Tests use [q_len, n_heads, head_dim] format.
    Internally adds batch dimension for attention utilities.
    """
    # Add batch dimension
    queries = queries.unsqueeze(0)  # [1, q_len, n_heads, head_dim]
    keys = keys.unsqueeze(0)
    values = values.unsqueeze(0)

    _, q_len, n_heads, head_dim = queries.shape
    _, kv_len, n_kv_heads, _ = keys.shape

    # Handle GQA
    k = keys
    v = values
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k = repeat_kv(k, n_rep)  # [1, kv_len, n_heads, head_dim]
        v = repeat_kv(v, n_rep)  # [1, kv_len, n_heads, head_dim]

    # Create causal mask if needed
    mask = None
    if causal:
        mask = create_causal_mask(q_len, 0).reshape(1, 1, q_len, q_len)

    # Run standard attention and remove batch dimension
    return attention(queries, k, v, mask=mask).squeeze(0)  # [q_len, n_heads, head_dim]


class TestFlashAttentionBasic:
    """Basic correctness tests for Flash Attention."""

    def test_output_shape(self):
        """Test that output shape matches input query shape."""
        q_len = 16
        n_heads = 8
        n_kv_heads = 8
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        out = flash_prefill_attention_tinygrad(q, k, v, causal=True)
        assert out.shape == (q_len, n_heads, head_dim)

    def test_matches_standard_attention_no_gqa(self):
        """Flash Attention should match standard attention numerically (no GQA)."""
        q_len = 32
        n_heads = 4
        n_kv_heads = 4  # Same as n_heads (no GQA)
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        # Standard attention
        std_out = standard_attention_with_gqa(q, k, v, causal=True)

        # Flash attention
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        # Should be numerically close
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_matches_standard_attention_with_gqa(self):
        """Flash Attention should match standard attention with GQA."""
        q_len = 32
        n_heads = 32
        n_kv_heads = 8  # GQA: 4x repeat
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        # Standard attention
        std_out = standard_attention_with_gqa(q, k, v, causal=True)

        # Flash attention
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        # Should be numerically close
        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_non_causal(self):
        """Test non-causal attention (all positions attend to all)."""
        q_len = 16
        n_heads = 4
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_heads, head_dim)
        v = Tensor.randn(q_len, n_heads, head_dim)

        # Non-causal standard attention
        q_b = q.unsqueeze(0)
        k_b = k.unsqueeze(0)
        v_b = v.unsqueeze(0)
        std_out = attention(q_b, k_b, v_b, mask=None).squeeze(0)

        # Non-causal flash attention
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=False)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestCausalMasking:
    """Tests for causal masking correctness."""

    def test_causal_mask_first_token(self):
        """First token should only attend to itself."""
        q_len = 8
        n_heads = 2
        head_dim = 16

        # Use specific values to verify masking
        # Create q with first token set to 1.0 (use list construction to avoid setitem issue)
        q_data = [[0.0] * head_dim] * n_heads
        q_first = [[1.0] * head_dim] * n_heads
        q = Tensor([q_first] + [q_data] * (q_len - 1))  # [q_len, n_heads, head_dim]

        k = Tensor.randn(q_len, n_heads, head_dim)
        v = Tensor.eye(q_len).unsqueeze(1).expand(q_len, n_heads, q_len)
        # v[i] has 1 at position i, so output should tell us attention pattern

        out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        # First token's output should be v[0] (attended only to position 0)
        # Since v[0,:,:] = [1, 0, 0, 0, 0, 0, 0, 0]
        assert out[0, 0, 0].item() > 0.5  # Should be close to 1.0

    def test_last_token_attends_to_all(self):
        """Last token should attend to all positions."""
        q_len = 8
        n_heads = 2
        head_dim = 16

        # All queries equal
        q = Tensor.ones(q_len, n_heads, head_dim)

        # All keys equal - will have uniform attention weights
        k = Tensor.zeros(q_len, n_heads, head_dim)

        # V has different values per position
        v = Tensor.arange(q_len).reshape(q_len, 1, 1).expand(q_len, n_heads, head_dim).float()

        out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        # Last token should average all positions (0+1+...+7)/8 = 3.5
        last_token_out = out[-1, 0, 0].item()
        expected = sum(range(q_len)) / q_len  # 3.5
        assert abs(last_token_out - expected) < 1e-2, f"Expected {expected}, got {last_token_out}"


class TestGQA:
    """Tests for Grouped Query Attention support."""

    def test_gqa_2x(self):
        """Test GQA with 2x repeat."""
        q_len = 16
        n_heads = 8
        n_kv_heads = 4
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_gqa_4x(self):
        """Test GQA with 4x repeat (TinyLlama style)."""
        q_len = 16
        n_heads = 32
        n_kv_heads = 8
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_gqa_8x(self):
        """Test GQA with 8x repeat."""
        q_len = 16
        n_heads = 32
        n_kv_heads = 4
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestSequenceLengths:
    """Tests for various sequence lengths."""

    @pytest.mark.parametrize("q_len", [1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 128])
    def test_various_lengths(self, q_len):
        """Test various sequence lengths including non-power-of-2."""
        n_heads = 4
        n_kv_heads = 4
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_single_token(self):
        """Test single token (decode-like scenario)."""
        n_heads = 8
        head_dim = 64

        q = Tensor.randn(1, n_heads, head_dim)
        k = Tensor.randn(1, n_heads, head_dim)
        v = Tensor.randn(1, n_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestHeadDimensions:
    """Tests for various head dimensions."""

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_various_head_dims(self, head_dim):
        """Test various head dimensions."""
        q_len = 16
        n_heads = 4
        n_kv_heads = 4

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)


class TestDtypes:
    """Tests for different data types."""

    def test_float16(self):
        """Test with FP16 inputs."""
        q_len = 32
        n_heads = 4
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float16)
        k = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float16)
        v = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float16)

        out = flash_prefill_attention_tinygrad(q, k, v, causal=True)
        # Note: tinygrad may promote to float32 for numerical stability in softmax
        # The important thing is the shape is correct
        assert out.shape == (q_len, n_heads, head_dim)
        # Output should be finite (no NaN/inf)
        assert out.isnan().sum().item() == 0, "Output contains NaN"
        assert out.isinf().sum().item() == 0, "Output contains Inf"

    def test_float32(self):
        """Test with FP32 inputs."""
        q_len = 32
        n_heads = 4
        head_dim = 32

        q = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float32)
        k = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float32)
        v = Tensor.randn(q_len, n_heads, head_dim, dtype=dtypes.float32)

        out = flash_prefill_attention_tinygrad(q, k, v, causal=True)
        assert out.dtype == dtypes.float32
        assert out.shape == (q_len, n_heads, head_dim)


class TestTinyLlamaConfig:
    """Tests with TinyLlama-like configuration."""

    def test_tinyllama_config(self):
        """Test with TinyLlama dimensions (32 heads, 8 KV heads, 64 head_dim)."""
        q_len = 64
        n_heads = 32
        n_kv_heads = 8
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-4, atol=1e-4)

    def test_tinyllama_longer_context(self):
        """Test with longer context (256 tokens)."""
        q_len = 256
        n_heads = 32
        n_kv_heads = 8
        head_dim = 64

        q = Tensor.randn(q_len, n_heads, head_dim)
        k = Tensor.randn(q_len, n_kv_heads, head_dim)
        v = Tensor.randn(q_len, n_kv_heads, head_dim)

        std_out = standard_attention_with_gqa(q, k, v, causal=True)
        flash_out = flash_prefill_attention_tinygrad(q, k, v, causal=True)

        assert_allclose(flash_out, std_out, rtol=1e-3, atol=1e-3)
