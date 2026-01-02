"""Tests for KVCache (Phase 4 block-based implementation)."""

import pytest
from tinygrad import Tensor, dtypes

from tinyvllm.core.kv_cache import KVCache


class TestKVCacheInit:
    def test_init_creates_storage(self):
        cache = KVCache(
            num_layers=4,
            num_blocks=10,
            block_size=16,
            n_kv_heads=8,
            head_dim=64,
            dtype=dtypes.float32
        )

        # Should have block tensors for each layer
        assert len(cache.k_blocks) == 4
        assert len(cache.v_blocks) == 4
        # Each layer should have num_blocks blocks
        assert len(cache.k_blocks[0]) == 10
        assert len(cache.v_blocks[0]) == 10

    def test_init_stores_params(self):
        cache = KVCache(
            num_layers=4,
            num_blocks=10,
            block_size=16,
            n_kv_heads=8,
            head_dim=64,
            dtype=dtypes.float32
        )

        assert cache.num_layers == 4
        assert cache.num_blocks == 10
        assert cache.block_size == 16
        assert cache.n_kv_heads == 8
        assert cache.head_dim == 64

    def test_init_block_shapes(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=4,
            head_dim=32,
            dtype=dtypes.float32
        )

        # Each block should have shape [block_size, n_kv_heads, head_dim]
        for layer_idx in range(2):
            for block_id in range(4):
                assert cache.k_blocks[layer_idx][block_id].shape == (8, 4, 32)
                assert cache.v_blocks[layer_idx][block_id].shape == (8, 4, 32)


class TestWriteKV:
    def test_write_kv_single_token(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        k = Tensor.ones(2, 4)  # [n_kv_heads, head_dim]
        v = Tensor.ones(2, 4) * 2

        cache.write_kv(layer_idx=0, block_id=0, offset=0, k=k, v=v)

        # Read back and verify
        k_read = cache.k_blocks[0][0][0]  # First slot
        assert k_read.sum().item() == 8  # 1 * 2 * 4

    def test_write_kv_multiple_positions(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write to multiple positions in the same block
        for i in range(5):
            k = Tensor.ones(2, 4) * (i + 1)
            v = Tensor.ones(2, 4) * (i + 1)
            cache.write_kv(layer_idx=0, block_id=0, offset=i, k=k, v=v)

        # Verify each position
        for i in range(5):
            k_read = cache.k_blocks[0][0][i]
            assert k_read.sum().item() == (i + 1) * 2 * 4

    def test_write_kv_multiple_blocks(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write to different blocks
        for block_id in range(3):
            k = Tensor.ones(2, 4) * (block_id + 1)
            v = Tensor.ones(2, 4) * (block_id + 1)
            cache.write_kv(layer_idx=0, block_id=block_id, offset=0, k=k, v=v)

        # Verify each block
        for block_id in range(3):
            k_read = cache.k_blocks[0][block_id][0]
            assert k_read.sum().item() == (block_id + 1) * 2 * 4

    def test_write_kv_multiple_layers(self):
        cache = KVCache(
            num_layers=4,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write to all layers
        for layer_idx in range(4):
            k = Tensor.ones(2, 4) * (layer_idx + 1)
            v = Tensor.ones(2, 4) * (layer_idx + 1)
            cache.write_kv(layer_idx=layer_idx, block_id=0, offset=0, k=k, v=v)

        # Verify each layer
        for layer_idx in range(4):
            k_read = cache.k_blocks[layer_idx][0][0]
            assert k_read.sum().item() == (layer_idx + 1) * 2 * 4


class TestWriteKVBatch:
    def test_write_kv_batch_basic(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=16,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write 5 tokens at once
        k = Tensor.ones(5, 2, 4)  # [num_tokens, n_kv_heads, head_dim]
        v = Tensor.ones(5, 2, 4) * 2

        cache.write_kv_batch(layer_idx=0, block_id=0, start_offset=0, k=k, v=v)

        # Verify all 5 positions
        for i in range(5):
            k_read = cache.k_blocks[0][0][i]
            assert k_read.sum().item() == 8  # 1 * 2 * 4

    def test_write_kv_batch_with_offset(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=16,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write 3 tokens starting at offset 5
        k = Tensor.ones(3, 2, 4) * 5
        v = Tensor.ones(3, 2, 4) * 5

        cache.write_kv_batch(layer_idx=0, block_id=0, start_offset=5, k=k, v=v)

        # Verify positions 5, 6, 7 have the values
        for i in range(5, 8):
            k_read = cache.k_blocks[0][0][i]
            assert k_read.sum().item() == 5 * 2 * 4


class TestReadKVBlocks:
    def test_read_single_block(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write 5 tokens
        for i in range(5):
            k = Tensor.ones(2, 4) * (i + 1)
            v = Tensor.ones(2, 4) * (i + 1)
            cache.write_kv(layer_idx=0, block_id=0, offset=i, k=k, v=v)

        # Read back
        block_table = [0]
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=block_table, context_len=5)

        assert k_read.shape == (5, 2, 4)
        assert v_read.shape == (5, 2, 4)

    def test_read_multiple_blocks(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Fill two blocks (8 tokens total)
        # Block 0: positions 0-3
        for i in range(4):
            k = Tensor.ones(2, 4) * (i + 1)
            v = Tensor.ones(2, 4) * (i + 1)
            cache.write_kv(layer_idx=0, block_id=0, offset=i, k=k, v=v)
        # Block 1: positions 4-7
        for i in range(4):
            k = Tensor.ones(2, 4) * (i + 5)
            v = Tensor.ones(2, 4) * (i + 5)
            cache.write_kv(layer_idx=0, block_id=1, offset=i, k=k, v=v)

        # Read back 6 tokens spanning 2 blocks
        block_table = [0, 1]
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=block_table, context_len=6)

        assert k_read.shape == (6, 2, 4)
        assert v_read.shape == (6, 2, 4)

    def test_read_empty_blocks(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Read with empty block list
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=[], context_len=0)

        assert k_read.shape == (0, 2, 4)
        assert v_read.shape == (0, 2, 4)


class TestGetMemoryBytes:
    def test_memory_calculation(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=10,
            block_size=16,
            n_kv_heads=4,
            head_dim=64,
            dtype=dtypes.float32
        )

        # Expected: num_layers * num_blocks * block_size * n_kv_heads * head_dim * 2 (K+V) * 4 (bytes)
        expected = 2 * 10 * 16 * 4 * 64 * 2 * 4
        assert cache.get_memory_bytes() == expected


class TestIntegration:
    def test_full_sequence_lifecycle(self):
        """Simulate a complete sequence with write and read operations"""
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Simulate prefill: write 5 tokens using batch write
        k_prefill = Tensor.arange(40).reshape(5, 2, 4).float()
        v_prefill = Tensor.arange(40).reshape(5, 2, 4).float() + 100

        for layer_idx in range(2):
            cache.write_kv_batch(layer_idx=layer_idx, block_id=0, start_offset=0,
                                k=k_prefill, v=v_prefill)

        # Read back and verify
        block_table = [0]
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=block_table, context_len=5)
        assert k_read.shape == (5, 2, 4)
        assert v_read.shape == (5, 2, 4)

        # Simulate decode: add one more token
        k_decode = Tensor.ones(2, 4) * 999
        v_decode = Tensor.ones(2, 4) * 888

        for layer_idx in range(2):
            cache.write_kv(layer_idx=layer_idx, block_id=0, offset=5, k=k_decode, v=v_decode)

        # Read all 6 tokens
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=block_table, context_len=6)
        assert k_read.shape == (6, 2, 4)

    def test_multiple_blocks_sequence(self):
        """Test sequence that spans multiple blocks"""
        cache = KVCache(
            num_layers=1,
            num_blocks=4,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write 10 tokens (needs 3 blocks with block_size=4)
        block_table = [0, 1, 2]
        for pos in range(10):
            block_idx = pos // 4
            offset = pos % 4
            block_id = block_table[block_idx]
            k = Tensor.ones(2, 4) * (pos + 1)
            v = Tensor.ones(2, 4) * (pos + 1)
            cache.write_kv(layer_idx=0, block_id=block_id, offset=offset, k=k, v=v)

        # Read all 10 tokens
        k_read, v_read = cache.read_kv_blocks(layer_idx=0, block_ids=block_table, context_len=10)
        assert k_read.shape == (10, 2, 4)

        # Verify values
        for pos in range(10):
            expected_sum = (pos + 1) * 2 * 4
            assert k_read[pos].sum().item() == expected_sum

    def test_layer_isolation(self):
        """Verify each layer stores independently"""
        cache = KVCache(
            num_layers=4,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Write different values to each layer
        for layer_idx in range(4):
            k = Tensor.ones(2, 4) * (layer_idx + 1)
            v = Tensor.ones(2, 4) * (layer_idx + 1) * 10
            cache.write_kv(layer_idx=layer_idx, block_id=0, offset=0, k=k, v=v)

        # Verify each layer has correct value
        for layer_idx in range(4):
            k_read, v_read = cache.read_kv_blocks(layer_idx=layer_idx, block_ids=[0], context_len=1)
            expected_k_sum = (layer_idx + 1) * 2 * 4
            expected_v_sum = (layer_idx + 1) * 10 * 2 * 4
            assert k_read.sum().item() == expected_k_sum
            assert v_read.sum().item() == expected_v_sum
