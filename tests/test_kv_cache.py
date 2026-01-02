"""Tests for KVCache (Phase 2 list-based implementation)."""

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

        assert len(cache.k_cache) == 4
        assert len(cache.v_cache) == 4

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

    def test_init_empty(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # No sequences allocated yet
        assert len(cache.k_cache[0]) == 0


class TestAllocateSequence:
    def test_allocate_sequence(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        assert 0 in cache.k_cache[0]
        assert 0 in cache.k_cache[1]
        assert len(cache.k_cache[0]) == 1

    def test_allocate_multiple_sequences(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)
        cache.allocate_sequence(seq_id=1)
        cache.allocate_sequence(seq_id=5)

        assert len(cache.k_cache[0]) == 3


class TestWriteKV:
    def test_write_kv_basic(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        k = Tensor.ones(2, 4)  # [n_kv_heads, head_dim]
        v = Tensor.ones(2, 4) * 2

        cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        assert cache.get_context_length(layer_idx=0, seq_id=0) == 1

    def test_write_kv_multiple_positions(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        for i in range(5):
            k = Tensor.ones(2, 4) * i
            v = Tensor.ones(2, 4) * i
            cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        assert cache.get_context_length(layer_idx=0, seq_id=0) == 5

    def test_write_kv_multiple_layers(self):
        cache = KVCache(
            num_layers=4,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        # Write to all layers for one position
        for layer_idx in range(4):
            k = Tensor.ones(2, 4) * layer_idx
            v = Tensor.ones(2, 4) * layer_idx
            cache.write_kv(layer_idx=layer_idx, seq_id=0, k=k, v=v)

        # Each layer should have 1 position
        for layer_idx in range(4):
            assert cache.get_context_length(layer_idx, seq_id=0) == 1

    def test_write_kv_auto_allocates(self):
        """write_kv should work even if allocate_sequence wasn't called"""
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        k = Tensor.ones(2, 4)
        v = Tensor.ones(2, 4)
        cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        assert cache.get_context_length(layer_idx=0, seq_id=0) == 1


class TestReadKV:
    def test_read_kv_single_position(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        k = Tensor.ones(2, 4) * 5
        v = Tensor.ones(2, 4) * 10
        cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        k_read, v_read = cache.read_kv(layer_idx=0, seq_id=0)

        assert k_read.shape == (1, 2, 4)  # [1 position, n_kv_heads, head_dim]
        assert v_read.shape == (1, 2, 4)
        assert k_read.sum().item() == 40  # 5 * 2 * 4
        assert v_read.sum().item() == 80  # 10 * 2 * 4

    def test_read_kv_multiple_positions(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        # Write 3 positions
        for i in range(3):
            k = Tensor.ones(2, 4) * (i + 1)
            v = Tensor.ones(2, 4) * (i + 1)
            cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        k_read, v_read = cache.read_kv(layer_idx=0, seq_id=0)

        assert k_read.shape == (3, 2, 4)  # [3 positions, n_kv_heads, head_dim]
        # Sum: 1*8 + 2*8 + 3*8 = 48
        assert k_read.sum().item() == 48

    def test_read_kv_empty_sequence(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        k_read, v_read = cache.read_kv(layer_idx=0, seq_id=0)

        assert k_read.shape == (0, 2, 4)
        assert v_read.shape == (0, 2, 4)


class TestFreeSequence:
    def test_free_sequence(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)
        cache.allocate_sequence(seq_id=1)

        k = Tensor.ones(2, 4)
        v = Tensor.ones(2, 4)
        cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)
        cache.write_kv(layer_idx=0, seq_id=1, k=k, v=v)

        assert len(cache.k_cache[0]) == 2

        cache.free_sequence(seq_id=0)

        assert len(cache.k_cache[0]) == 1
        assert 0 not in cache.k_cache[0]
        assert 1 in cache.k_cache[0]


class TestGetContextLength:
    def test_get_context_length(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        assert cache.get_context_length(layer_idx=0, seq_id=0) == 0

        for i in range(5):
            k = Tensor.ones(2, 4)
            v = Tensor.ones(2, 4)
            cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        assert cache.get_context_length(layer_idx=0, seq_id=0) == 5

    def test_get_context_length_nonexistent(self):
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=8,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        assert cache.get_context_length(layer_idx=0, seq_id=99) == 0


class TestIntegration:
    def test_full_sequence_lifecycle(self):
        """Simulate a complete sequence: allocate, write tokens, read, free"""
        cache = KVCache(
            num_layers=2,
            num_blocks=4,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Allocate
        cache.allocate_sequence(seq_id=0)

        # Write 6 tokens (simulating prefill + decode)
        for pos in range(6):
            for layer_idx in range(2):
                k = Tensor.ones(2, 4) * (pos + 1)
                v = Tensor.ones(2, 4) * (pos + 1)
                cache.write_kv(layer_idx=layer_idx, seq_id=0, k=k, v=v)

        # Verify
        assert cache.get_context_length(layer_idx=0, seq_id=0) == 6
        assert cache.get_context_length(layer_idx=1, seq_id=0) == 6

        # Read and check
        k_read, v_read = cache.read_kv(layer_idx=0, seq_id=0)
        assert k_read.shape == (6, 2, 4)

        # Free
        cache.free_sequence(seq_id=0)
        assert len(cache.k_cache[0]) == 0

    def test_multiple_sequences(self):
        """Test multiple concurrent sequences"""
        cache = KVCache(
            num_layers=2,
            num_blocks=10,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        # Allocate 3 sequences
        for seq_id in range(3):
            cache.allocate_sequence(seq_id=seq_id)

        # Write different lengths
        for pos in range(3):  # seq 0 gets 3 tokens
            k = Tensor.ones(2, 4)
            v = Tensor.ones(2, 4)
            cache.write_kv(layer_idx=0, seq_id=0, k=k, v=v)

        for pos in range(5):  # seq 1 gets 5 tokens
            k = Tensor.ones(2, 4)
            v = Tensor.ones(2, 4)
            cache.write_kv(layer_idx=0, seq_id=1, k=k, v=v)

        for pos in range(1):  # seq 2 gets 1 token
            k = Tensor.ones(2, 4)
            v = Tensor.ones(2, 4)
            cache.write_kv(layer_idx=0, seq_id=2, k=k, v=v)

        # Verify lengths
        assert cache.get_context_length(layer_idx=0, seq_id=0) == 3
        assert cache.get_context_length(layer_idx=0, seq_id=1) == 5
        assert cache.get_context_length(layer_idx=0, seq_id=2) == 1

        # Free middle sequence
        cache.free_sequence(seq_id=1)
        assert len(cache.k_cache[0]) == 2
        assert cache.get_context_length(layer_idx=0, seq_id=0) == 3

    def test_layer_isolation(self):
        """Verify each layer stores independently"""
        cache = KVCache(
            num_layers=4,
            num_blocks=4,
            block_size=4,
            n_kv_heads=2,
            head_dim=4,
            dtype=dtypes.float32
        )

        cache.allocate_sequence(seq_id=0)

        # Write different values to each layer
        for layer_idx in range(4):
            k = Tensor.ones(2, 4) * (layer_idx + 1)
            v = Tensor.ones(2, 4) * (layer_idx + 1) * 10
            cache.write_kv(layer_idx=layer_idx, seq_id=0, k=k, v=v)

        # Verify each layer has correct value
        for layer_idx in range(4):
            k_read, v_read = cache.read_kv(layer_idx=layer_idx, seq_id=0)
            expected_k_sum = (layer_idx + 1) * 2 * 4  # value * heads * dim
            expected_v_sum = (layer_idx + 1) * 10 * 2 * 4
            assert k_read.sum().item() == expected_k_sum
            assert v_read.sum().item() == expected_v_sum
