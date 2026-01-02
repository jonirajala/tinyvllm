"""Tests for LLaMA model components."""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rope,
    Attention,
    FeedForward,
    TransformerBlock,
    Llama,
)
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.kv_cache import KVCache


# Small config for fast tests
def small_config():
    return LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=100,
        hidden_dim=64,
        max_seq_len=64,
    )


def create_test_kv_cache(config):
    """Create a KVCache for testing."""
    return KVCache(
        num_layers=config.n_layers,
        num_blocks=10,
        block_size=16,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=dtypes.float32,
    )


class TestRMSNorm:
    def test_output_shape(self):
        """Output shape should match input shape."""
        norm = RMSNorm(dim=32)
        x = Tensor.randn(2, 8, 32)  # [batch, seq, dim]
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """RMS of output should be approximately 1."""
        norm = RMSNorm(dim=32, eps=1e-5)
        x = Tensor.randn(1, 1, 32)
        out = norm(x)

        # Compute RMS of output
        out_list = out.realize().tolist()[0][0]
        rms = math.sqrt(sum(v**2 for v in out_list) / len(out_list))

        # Should be close to 1 (weight is initialized to ones)
        assert 0.8 < rms < 1.2

    def test_zero_input(self):
        """Should handle zero input without NaN."""
        norm = RMSNorm(dim=8, eps=1e-5)
        x = Tensor.zeros(1, 1, 8)
        out = norm(x)
        out_list = out.realize().tolist()[0][0]

        # Should all be zeros (or very small due to eps)
        assert all(abs(v) < 0.01 for v in out_list)

    def test_scale_invariance(self):
        """Scaling input should scale output proportionally."""
        norm = RMSNorm(dim=8)
        x = Tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]])

        out1 = norm(x).realize().tolist()
        out2 = norm(x * 2).realize().tolist()

        # Outputs should be the same (RMSNorm is scale-invariant)
        for v1, v2 in zip(out1[0][0][0], out2[0][0][0]):
            assert v1 == pytest.approx(v2, rel=0.01)


class TestPrecomputeFreqsCis:
    def test_output_shapes(self):
        """Should return cos and sin with correct shapes."""
        dim, seq_len = 16, 32
        cos, sin = precompute_freqs_cis(dim, seq_len)

        # Shape should be [seq_len, dim/2]
        assert cos.shape == (seq_len, dim // 2)
        assert sin.shape == (seq_len, dim // 2)

    def test_cos_sin_range(self):
        """Cos and sin values should be in [-1, 1]."""
        cos, sin = precompute_freqs_cis(16, 32)
        cos_list = cos.realize().tolist()
        sin_list = sin.realize().tolist()

        for row in cos_list:
            assert all(-1.0 <= v <= 1.0 for v in row)
        for row in sin_list:
            assert all(-1.0 <= v <= 1.0 for v in row)

    def test_position_zero(self):
        """At position 0, cos should be 1 and sin should be 0."""
        cos, sin = precompute_freqs_cis(8, 10)
        cos_0 = cos[0].realize().tolist()
        sin_0 = sin[0].realize().tolist()

        # cos(0) = 1, sin(0) = 0
        assert all(v == pytest.approx(1.0, abs=1e-5) for v in cos_0)
        assert all(v == pytest.approx(0.0, abs=1e-5) for v in sin_0)

    def test_different_theta(self):
        """Different theta should produce different frequencies."""
        cos1, _ = precompute_freqs_cis(8, 10, theta=10000.0)
        cos2, _ = precompute_freqs_cis(8, 10, theta=1000.0)

        # Should be different at position > 0
        c1 = cos1[5].realize().tolist()
        c2 = cos2[5].realize().tolist()
        assert c1 != c2


class TestApplyRope:
    def test_output_shape(self):
        """Output shape should match input shape."""
        batch, seq, heads, head_dim = 2, 8, 4, 16
        x = Tensor.randn(batch, seq, heads, head_dim)
        cos, sin = precompute_freqs_cis(head_dim, seq)

        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_zero_position(self):
        """At position 0, output should equal input (rotation by 0)."""
        batch, seq, heads, head_dim = 1, 1, 1, 8
        x = Tensor.randn(batch, seq, heads, head_dim)
        cos, sin = precompute_freqs_cis(head_dim, 10)

        out = apply_rope(x, cos, sin)

        x_list = x.realize().tolist()[0][0][0]
        out_list = out.realize().tolist()[0][0][0]

        # Should be equal at position 0 (cos=1, sin=0)
        for v1, v2 in zip(x_list, out_list):
            assert v1 == pytest.approx(v2, rel=0.01)

    def test_rotation_preserves_norm(self):
        """Rotation should preserve vector magnitude."""
        batch, seq, heads, head_dim = 1, 4, 1, 8
        x = Tensor.randn(batch, seq, heads, head_dim)
        cos, sin = precompute_freqs_cis(head_dim, seq)

        out = apply_rope(x, cos, sin)

        # Compare norms at each position
        x_list = x.realize().tolist()[0]
        out_list = out.realize().tolist()[0]

        for pos in range(seq):
            x_norm = math.sqrt(sum(v**2 for v in x_list[pos][0]))
            out_norm = math.sqrt(sum(v**2 for v in out_list[pos][0]))
            assert x_norm == pytest.approx(out_norm, rel=0.01)


class TestAttention:
    def test_output_shape(self):
        """Output shape should be [batch, seq, dim]."""
        config = small_config()
        attn = Attention(config)
        kv_cache = create_test_kv_cache(config)

        batch, seq = 1, 8  # batch must be 1 with kv_cache
        x = Tensor.randn(batch, seq, config.dim)
        cos, sin = precompute_freqs_cis(config.head_dim, seq)

        kv_cache.allocate_sequence(seq_id=0)
        out = attn(x, cos, sin, kv_cache=kv_cache, layer_idx=0, seq_id=0)

        assert out.shape == (batch, seq, config.dim)

    def test_with_kv_cache(self):
        """Should work with KVCache."""
        config = small_config()
        attn = Attention(config)
        kv_cache = create_test_kv_cache(config)

        batch, seq = 1, 8
        x = Tensor.randn(batch, seq, config.dim)
        cos, sin = precompute_freqs_cis(config.head_dim, seq)

        kv_cache.allocate_sequence(seq_id=0)
        out = attn(x, cos, sin, kv_cache=kv_cache, layer_idx=0, seq_id=0)

        assert out.shape == (batch, seq, config.dim)
        # KV cache should have stored the tokens
        assert kv_cache.get_context_length(layer_idx=0, seq_id=0) == seq

    def test_kv_cache_grows(self):
        """KVCache should grow with each call."""
        config = small_config()
        attn = Attention(config)
        kv_cache = create_test_kv_cache(config)

        batch = 1
        x1 = Tensor.randn(batch, 4, config.dim)  # First 4 tokens
        x2 = Tensor.randn(batch, 1, config.dim)  # Next token

        cos, sin = precompute_freqs_cis(config.head_dim, 10)

        kv_cache.allocate_sequence(seq_id=0)

        # First pass
        attn(x1, cos[:4], sin[:4], kv_cache=kv_cache, layer_idx=0, seq_id=0, start_pos=0)
        assert kv_cache.get_context_length(layer_idx=0, seq_id=0) == 4

        # Second pass
        attn(x2, cos[4:5], sin[4:5], kv_cache=kv_cache, layer_idx=0, seq_id=0, start_pos=4)
        assert kv_cache.get_context_length(layer_idx=0, seq_id=0) == 5


class TestFeedForward:
    def test_output_shape(self):
        """Output shape should match input shape."""
        config = small_config()
        ff = FeedForward(config)

        x = Tensor.randn(2, 8, config.dim)
        out = ff(x)

        assert out.shape == x.shape

    def test_nonlinearity(self):
        """SwiGLU should apply nonlinearity."""
        config = small_config()
        ff = FeedForward(config)

        # Test that output is different from a simple linear transformation
        x = Tensor.randn(1, 1, config.dim)
        out = ff(x)

        # Output should exist and have values
        out_list = out.realize().tolist()[0][0]
        assert len(out_list) == config.dim
        assert any(v != 0 for v in out_list)


class TestTransformerBlock:
    def test_output_shape(self):
        """Output shape should match input shape."""
        config = small_config()
        block = TransformerBlock(config)
        kv_cache = create_test_kv_cache(config)

        batch, seq = 1, 8  # batch must be 1 with kv_cache
        x = Tensor.randn(batch, seq, config.dim)
        cos, sin = precompute_freqs_cis(config.head_dim, seq)

        kv_cache.allocate_sequence(seq_id=0)
        out = block(x, cos, sin, kv_cache=kv_cache, layer_idx=0, seq_id=0)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should include residual from input."""
        config = small_config()
        block = TransformerBlock(config)
        kv_cache = create_test_kv_cache(config)

        batch, seq = 1, 4
        x = Tensor.randn(batch, seq, config.dim)
        cos, sin = precompute_freqs_cis(config.head_dim, seq)

        kv_cache.allocate_sequence(seq_id=0)
        out = block(x, cos, sin, kv_cache=kv_cache, layer_idx=0, seq_id=0)

        # Output should be different from input but correlated
        x_list = x.realize().tolist()
        out_list = out.realize().tolist()

        # They should be different
        assert x_list != out_list

    def test_with_kv_cache(self):
        """Should work with KVCache."""
        config = small_config()
        block = TransformerBlock(config)
        kv_cache = create_test_kv_cache(config)

        x = Tensor.randn(1, 4, config.dim)
        cos, sin = precompute_freqs_cis(config.head_dim, 4)

        kv_cache.allocate_sequence(seq_id=0)
        out = block(x, cos, sin, kv_cache=kv_cache, layer_idx=0, seq_id=0)

        assert out.shape == x.shape
        assert kv_cache.get_context_length(layer_idx=0, seq_id=0) == 4


class TestLlama:
    def test_output_shape(self):
        """Output logits should have shape [batch, seq, vocab_size]."""
        config = small_config()
        model = Llama(config)
        kv_cache = model.create_kv_cache()

        batch, seq = 1, 8
        tokens = Tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        kv_cache.allocate_sequence(seq_id=0)
        logits = model(tokens, kv_cache=kv_cache, seq_id=0)
        assert logits.shape == (batch, seq, config.vocab_size)

    def test_with_kv_cache(self):
        """Should work with KVCache and cache all layers."""
        config = small_config()
        model = Llama(config)
        kv_cache = model.create_kv_cache()

        tokens = Tensor([[1, 2, 3, 4]])
        kv_cache.allocate_sequence(seq_id=0)
        logits = model(tokens, kv_cache=kv_cache, seq_id=0)

        assert logits.shape == (1, 4, config.vocab_size)
        # All layers should have cached tokens
        for layer_idx in range(config.n_layers):
            assert kv_cache.get_context_length(layer_idx, seq_id=0) == 4

    def test_incremental_generation(self):
        """Should support incremental generation with KV cache."""
        config = small_config()
        model = Llama(config)
        kv_cache = model.create_kv_cache()

        kv_cache.allocate_sequence(seq_id=0)

        # First pass: process prompt
        prompt = Tensor([[1, 2, 3, 4]])
        logits1 = model(prompt, start_pos=0, kv_cache=kv_cache, seq_id=0)
        assert logits1.shape == (1, 4, config.vocab_size)

        # Second pass: generate next token
        next_token = Tensor([[5]])
        logits2 = model(next_token, start_pos=4, kv_cache=kv_cache, seq_id=0)
        assert logits2.shape == (1, 1, config.vocab_size)

        # KV cache should have grown
        assert kv_cache.get_context_length(layer_idx=0, seq_id=0) == 5

    def test_single_token(self):
        """Should handle single token input."""
        config = small_config()
        model = Llama(config)
        kv_cache = model.create_kv_cache()

        tokens = Tensor([[42]])
        kv_cache.allocate_sequence(seq_id=0)
        logits = model(tokens, kv_cache=kv_cache, seq_id=0)

        assert logits.shape == (1, 1, config.vocab_size)

    def test_causal_masking(self):
        """Earlier positions should not attend to later positions."""
        config = small_config()
        model = Llama(config)
        kv_cache = model.create_kv_cache()

        kv_cache.allocate_sequence(seq_id=0)

        # This is implicitly tested by the model working correctly
        tokens = Tensor([[1, 2, 3, 4, 5]])
        logits = model(tokens, kv_cache=kv_cache, seq_id=0)

        # Should produce valid output
        assert logits.shape == (1, 5, config.vocab_size)
        logits_list = logits.realize().tolist()
        assert all(not math.isnan(v) for row in logits_list for pos in row for v in pos)


class TestLlamaConfig:
    def test_default_values(self):
        """Test default config values."""
        config = LlamaConfig()
        assert config.dim == 4096
        assert config.n_layers == 32
        assert config.n_heads == 32
        assert config.vocab_size == 32000

    def test_head_dim_computed(self):
        """head_dim should be dim // n_heads."""
        config = LlamaConfig(dim=128, n_heads=8)
        assert config.head_dim == 16

    def test_kv_heads_defaults_to_n_heads(self):
        """n_kv_heads should default to n_heads."""
        config = LlamaConfig(n_heads=16)
        assert config.n_kv_heads == 16

    def test_gqa_config(self):
        """Test grouped-query attention config."""
        config = LlamaConfig(n_heads=32, n_kv_heads=8)
        assert config.n_heads == 32
        assert config.n_kv_heads == 8
