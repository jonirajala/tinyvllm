"""Tests for attention_utils (Phase 2 simplified implementation)."""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.core.attention_utils import (
    create_causal_mask,
    create_padding_mask,
    repeat_kv,
    attention,
    paged_attention,
    paged_attention_with_kvcache,
    gather_kv_from_blocks,
)
from tinyvllm.core.kv_cache import KVCache


class TestCreateCausalMask:
    def test_prefill_mask_shape(self):
        mask = create_causal_mask(seq_len=5, start_pos=0)
        assert mask.shape == (5, 5)

    def test_prefill_mask_is_causal(self):
        mask = create_causal_mask(seq_len=4, start_pos=0)

        # Check lower triangle is 0 (can attend)
        # Check upper triangle is -inf (cannot attend)
        for i in range(4):
            for j in range(4):
                val = mask[i, j].item()
                if j <= i:
                    assert val == 0, f"Position ({i},{j}) should be 0"
                else:
                    assert val == float('-inf'), f"Position ({i},{j}) should be -inf"

    def test_decode_mask_shape(self):
        # Single token at position 5
        mask = create_causal_mask(seq_len=1, start_pos=5)
        assert mask.shape == (1, 6)  # Can attend to 6 positions (0-5)

    def test_decode_mask_all_valid(self):
        mask = create_causal_mask(seq_len=1, start_pos=5)

        # Decode: can attend to all previous tokens
        for j in range(6):
            assert mask[0, j].item() == 0, f"Position (0,{j}) should be 0"

    def test_prefill_with_cache(self):
        # 3 tokens in cache, 2 new tokens
        mask = create_causal_mask(seq_len=2, start_pos=3)
        assert mask.shape == (2, 5)

        # First new token can attend to positions 0-3
        assert mask[0, 0].item() == 0  # cached
        assert mask[0, 3].item() == 0  # self
        assert mask[0, 4].item() == float('-inf')  # future

        # Second new token can attend to positions 0-4
        assert mask[1, 4].item() == 0


class TestCreatePaddingMask:
    def test_padding_mask_shape(self):
        mask = create_padding_mask(context_lens=[3, 5], max_len=5)
        assert mask.shape == (2, 1, 1, 5)

    def test_padding_mask_values(self):
        mask = create_padding_mask(context_lens=[3, 5], max_len=5)

        # Sequence 0: length 3, so positions 3,4 are masked
        assert mask[0, 0, 0, 0].item() == 0  # valid
        assert mask[0, 0, 0, 2].item() == 0  # valid
        assert mask[0, 0, 0, 3].item() == float('-inf')  # padding
        assert mask[0, 0, 0, 4].item() == float('-inf')  # padding

        # Sequence 1: length 5, all valid
        for j in range(5):
            assert mask[1, 0, 0, j].item() == 0


class TestRepeatKV:
    def test_no_repeat(self):
        x = Tensor.ones(2, 4, 8, 64)  # [batch, seq, heads, dim]
        out = repeat_kv(x, n_rep=1)
        assert out.shape == (2, 4, 8, 64)

    def test_repeat_2x(self):
        x = Tensor.ones(2, 4, 4, 64)
        out = repeat_kv(x, n_rep=2)
        assert out.shape == (2, 4, 8, 64)

    def test_repeat_4x(self):
        x = Tensor.ones(2, 4, 2, 64)
        out = repeat_kv(x, n_rep=4)
        assert out.shape == (2, 4, 8, 64)

    def test_values_repeated_correctly(self):
        # Create distinct values per head
        # Shape [1, 1, 2, 2]: batch=1, seq_len=1, n_kv_heads=2, head_dim=2
        # Head 0 has values [1, 2], Head 1 has values [3, 4]
        x = Tensor([[[[1, 2], [3, 4]]]])  # [1, 1, 2, 2]
        out = repeat_kv(x, n_rep=2)  # [1, 1, 4, 2]

        # Each head should be repeated
        assert out[0, 0, 0, 0].item() == 1  # head 0 copy 1
        assert out[0, 0, 1, 0].item() == 1  # head 0 copy 2
        assert out[0, 0, 2, 0].item() == 3  # head 1 copy 1
        assert out[0, 0, 3, 0].item() == 3  # head 1 copy 2


class TestAttention:
    def test_output_shape(self):
        batch, q_len, n_heads, head_dim = 2, 4, 8, 64
        kv_len = 6

        q = Tensor.randn(batch, q_len, n_heads, head_dim)
        k = Tensor.randn(batch, kv_len, n_heads, head_dim)
        v = Tensor.randn(batch, kv_len, n_heads, head_dim)

        out = attention(q, k, v)
        assert out.shape == (batch, q_len, n_heads, head_dim)

    def test_with_mask(self):
        batch, seq_len, n_heads, head_dim = 1, 4, 2, 8

        q = Tensor.randn(batch, seq_len, n_heads, head_dim)
        k = Tensor.randn(batch, seq_len, n_heads, head_dim)
        v = Tensor.randn(batch, seq_len, n_heads, head_dim)

        mask = create_causal_mask(seq_len, 0).reshape(1, 1, seq_len, seq_len)
        out = attention(q, k, v, mask=mask)

        assert out.shape == (batch, seq_len, n_heads, head_dim)

    def test_attention_weights_sum_to_one(self):
        # With no mask, softmax should sum to 1
        batch, seq_len, n_heads, head_dim = 1, 4, 1, 8

        q = Tensor.randn(batch, seq_len, n_heads, head_dim)
        k = Tensor.randn(batch, seq_len, n_heads, head_dim)
        v = Tensor.ones(batch, seq_len, n_heads, head_dim)

        out = attention(q, k, v)

        # If V is all ones and attention sums to 1, output should be close to 1
        out_mean = out.mean().item()
        assert 0.9 < out_mean < 1.1


class TestPagedAttention:
    def test_output_shape(self):
        batch, q_len, n_heads, head_dim = 2, 4, 8, 64
        kv_len = 6

        q = Tensor.randn(batch, q_len, n_heads, head_dim)
        k = Tensor.randn(batch, kv_len, n_heads, head_dim)
        v = Tensor.randn(batch, kv_len, n_heads, head_dim)

        out = paged_attention(q, k, v, context_lens=[kv_len, kv_len])
        assert out.shape == (batch, q_len, n_heads, head_dim)

    def test_with_gqa(self):
        """Test grouped query attention (fewer KV heads than query heads)"""
        batch, q_len, n_heads, head_dim = 1, 4, 8, 64
        n_kv_heads = 2
        kv_len = 6

        q = Tensor.randn(batch, q_len, n_heads, head_dim)
        k = Tensor.randn(batch, kv_len, n_kv_heads, head_dim)
        v = Tensor.randn(batch, kv_len, n_kv_heads, head_dim)

        out = paged_attention(q, k, v, context_lens=[kv_len])
        assert out.shape == (batch, q_len, n_heads, head_dim)

    def test_decode_single_token(self):
        batch, n_heads, head_dim = 1, 4, 32
        kv_len = 10

        q = Tensor.randn(batch, 1, n_heads, head_dim)  # Single query token
        k = Tensor.randn(batch, kv_len, n_heads, head_dim)
        v = Tensor.randn(batch, kv_len, n_heads, head_dim)

        out = paged_attention(q, k, v, context_lens=[kv_len], start_pos=kv_len - 1)
        assert out.shape == (batch, 1, n_heads, head_dim)


class TestPagedAttentionWithKVCache:
    def test_basic_usage(self):
        # Setup cache
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=16,
            n_kv_heads=4,
            head_dim=32,
            dtype=dtypes.float32
        )

        seq_id = 0
        cache.allocate_sequence(seq_id)

        # Write some K/V
        for pos in range(5):
            k = Tensor.randn(4, 32)
            v = Tensor.randn(4, 32)
            cache.write_kv(layer_idx=0, seq_id=seq_id, k=k, v=v)

        # Create query
        query = Tensor.randn(1, 1, 4, 32)  # [batch, q_len, heads, dim]

        # Run attention
        out = paged_attention_with_kvcache(
            query=query,
            kv_cache=cache,
            seq_id=seq_id,
            layer_idx=0,
            start_pos=4
        )

        assert out.shape == (1, 1, 4, 32)

    def test_prefill(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=16,
            n_kv_heads=4,
            head_dim=32,
            dtype=dtypes.float32
        )

        seq_id = 0
        cache.allocate_sequence(seq_id)

        # Write K/V for prefill
        for pos in range(10):
            k = Tensor.randn(4, 32)
            v = Tensor.randn(4, 32)
            cache.write_kv(layer_idx=0, seq_id=seq_id, k=k, v=v)

        # Prefill query (all positions)
        query = Tensor.randn(1, 10, 4, 32)

        out = paged_attention_with_kvcache(
            query=query,
            kv_cache=cache,
            seq_id=seq_id,
            layer_idx=0,
            start_pos=0
        )

        assert out.shape == (1, 10, 4, 32)


class TestGatherKVFromBlocks:
    def test_raises_not_implemented(self):
        """Phase 2: This function should raise NotImplementedError"""
        with pytest.raises(NotImplementedError):
            gather_kv_from_blocks(
                block_pool=Tensor.zeros(4, 16, 8, 64),
                block_table=[0, 1, 2],
                context_len=40,
                block_size=16
            )


class TestIntegration:
    def test_full_generation_flow(self):
        """Simulate prefill + decode using KVCache and paged attention"""
        n_layers = 2
        n_heads = 4
        n_kv_heads = 4
        head_dim = 32
        prompt_len = 5
        decode_steps = 3

        cache = KVCache(
            num_layers=n_layers,
            num_blocks=10,
            block_size=16,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtypes.float32
        )

        seq_id = 0
        cache.allocate_sequence(seq_id)

        # Prefill: write K/V for all prompt tokens
        for layer_idx in range(n_layers):
            for pos in range(prompt_len):
                k = Tensor.randn(n_kv_heads, head_dim)
                v = Tensor.randn(n_kv_heads, head_dim)
                cache.write_kv(layer_idx=layer_idx, seq_id=seq_id, k=k, v=v)

        # Prefill attention
        query = Tensor.randn(1, prompt_len, n_heads, head_dim)
        out = paged_attention_with_kvcache(query, cache, seq_id, layer_idx=0, start_pos=0)
        assert out.shape == (1, prompt_len, n_heads, head_dim)

        # Decode: generate tokens one at a time
        for step in range(decode_steps):
            current_pos = prompt_len + step

            # Write new K/V
            for layer_idx in range(n_layers):
                k = Tensor.randn(n_kv_heads, head_dim)
                v = Tensor.randn(n_kv_heads, head_dim)
                cache.write_kv(layer_idx=layer_idx, seq_id=seq_id, k=k, v=v)

            # Decode attention (single token query)
            query = Tensor.randn(1, 1, n_heads, head_dim)
            out = paged_attention_with_kvcache(query, cache, seq_id, layer_idx=0, start_pos=current_pos)
            assert out.shape == (1, 1, n_heads, head_dim)

        # Verify final cache length
        assert cache.get_context_length(layer_idx=0, seq_id=seq_id) == prompt_len + decode_steps
