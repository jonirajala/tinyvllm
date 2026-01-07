"""Tests for attention_utils (Phase 4 block-based implementation)."""

import pytest
import math
from tinygrad import Tensor, dtypes

from tinyvllm.core.attention_utils import (
    create_causal_mask,
    repeat_kv,
    attention,
    paged_prefill_attention,
)
from tinyvllm.core.kv_cache import KVCache
from tinyvllm.core.block_manager import BlockManager


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


class TestPrefillAttention:
    """Tests for paged_prefill_attention with block-based KV cache."""

    def test_basic_usage(self):
        """Test reading K/V from block-based cache."""
        n_layers = 2
        num_blocks = 10
        block_size = 16
        n_kv_heads = 4
        head_dim = 32

        # Create block-based cache
        cache = KVCache(
            num_layers=n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtypes.float32
        )

        # Write some K/V to blocks
        block_table = [0, 1]  # Using blocks 0 and 1
        context_len = 5

        for pos in range(context_len):
            block_idx = pos // block_size
            offset = pos % block_size
            block_id = block_table[block_idx]

            k = Tensor.randn(n_kv_heads, head_dim)
            v = Tensor.randn(n_kv_heads, head_dim)
            cache.write_kv(layer_idx=0, block_id=block_id, offset=offset, k=k, v=v)

        # Create query
        query = Tensor.randn(1, 1, n_kv_heads, head_dim)  # [batch, q_len, heads, dim]

        # Run attention
        out = paged_prefill_attention(
            query=query,
            kv_cache=cache,
            block_table=block_table,
            context_len=context_len,
            layer_idx=0,
            start_pos=context_len - 1,
        )

        assert out.shape == (1, 1, n_kv_heads, head_dim)

    def test_prefill(self):
        """Test prefill with block-based cache."""
        n_layers = 2
        num_blocks = 10
        block_size = 16
        n_kv_heads = 4
        head_dim = 32
        prompt_len = 10

        cache = KVCache(
            num_layers=n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtypes.float32
        )

        # Allocate blocks for prompt
        block_table = [0]  # Single block can hold 16 tokens

        # Write K/V for prefill
        for pos in range(prompt_len):
            block_idx = pos // block_size
            offset = pos % block_size
            block_id = block_table[block_idx]

            k = Tensor.randn(n_kv_heads, head_dim)
            v = Tensor.randn(n_kv_heads, head_dim)
            cache.write_kv(layer_idx=0, block_id=block_id, offset=offset, k=k, v=v)

        # Prefill query (all positions)
        query = Tensor.randn(1, prompt_len, n_kv_heads, head_dim)

        out = paged_prefill_attention(
            query=query,
            kv_cache=cache,
            block_table=block_table,
            context_len=prompt_len,
            layer_idx=0,
            start_pos=0,
        )

        assert out.shape == (1, prompt_len, n_kv_heads, head_dim)


class TestIntegration:
    """Phase 4 integration tests with BlockManager."""

    def test_full_generation_flow(self):
        """Simulate prefill + decode using BlockManager and block-based KVCache."""
        n_layers = 2
        n_heads = 4
        n_kv_heads = 4
        head_dim = 32
        num_blocks = 10
        block_size = 16
        prompt_len = 5
        decode_steps = 3

        # Create BlockManager and KVCache
        block_manager = BlockManager(
            num_gpus=1,
            blocks_per_gpu=num_blocks,
            block_size=block_size,
        )
        cache = KVCache(
            num_layers=n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtypes.float32
        )

        seq_id = 0
        block_manager.allocate_sequence(seq_id, prompt_len)

        # Prefill: write K/V for all prompt tokens
        for pos in range(prompt_len):
            _, block_id, offset = block_manager.get_slot(seq_id)
            for layer_idx in range(n_layers):
                k = Tensor.randn(n_kv_heads, head_dim)
                v = Tensor.randn(n_kv_heads, head_dim)
                cache.write_kv(layer_idx=layer_idx, block_id=block_id, offset=offset, k=k, v=v)
            block_manager.advance_position(seq_id)

        # Prefill attention
        block_table = block_manager.get_block_table(seq_id)
        context_len = block_manager.get_context_length(seq_id)
        query = Tensor.randn(1, prompt_len, n_heads, head_dim)
        out = paged_prefill_attention(
            query, cache, block_table, context_len, layer_idx=0, start_pos=0
        )
        assert out.shape == (1, prompt_len, n_heads, head_dim)

        # Decode: generate tokens one at a time
        for step in range(decode_steps):
            current_pos = block_manager.get_context_length(seq_id)

            # Write new K/V
            _, block_id, offset = block_manager.get_slot(seq_id)
            for layer_idx in range(n_layers):
                k = Tensor.randn(n_kv_heads, head_dim)
                v = Tensor.randn(n_kv_heads, head_dim)
                cache.write_kv(layer_idx=layer_idx, block_id=block_id, offset=offset, k=k, v=v)
            block_manager.advance_position(seq_id)

            # Decode attention (single token query)
            block_table = block_manager.get_block_table(seq_id)
            context_len = block_manager.get_context_length(seq_id)
            query = Tensor.randn(1, 1, n_heads, head_dim)
            out = paged_prefill_attention(
                query, cache, block_table, context_len, layer_idx=0, start_pos=current_pos
            )
            assert out.shape == (1, 1, n_heads, head_dim)

        # Verify final context length
        assert block_manager.get_context_length(seq_id) == prompt_len + decode_steps

        # Cleanup
        block_manager.free_sequence(seq_id)
