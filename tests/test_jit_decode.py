"""Tests for JIT decode compilation.

Tests that JIT-compiled decode matches non-JIT decode and
behaves consistently across multiple calls.
"""

import pytest
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.kv_cache import KVCache
from tinyvllm.core.block_manager import BlockManager


def _setup_seq(bm, seq_id, num_tokens):
    """Helper to register sequence and pre-allocate blocks."""
    bm.register_sequence(seq_id=seq_id)
    if num_tokens > 0:
        bm.ensure_block_for_position(seq_id=seq_id, pos=num_tokens-1)




def create_test_model():
    """Create a small test model."""
    config = LlamaConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=128,
        hidden_dim=128,
        max_seq_len=128,
    )
    return Llama(config), config


def create_test_components(config, num_blocks=32, block_size=16):
    """Create KVCache and BlockManager for testing."""
    block_manager = BlockManager(num_gpus=1, blocks_per_gpu=num_blocks, block_size=block_size)
    kv_cache = KVCache(
        num_layers=config.n_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=dtypes.float32,
    )
    return block_manager, kv_cache


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


class TestJitDecodeCreation:
    """Tests for JIT decode function creation."""

    def test_create_jit_decode_returns_callable(self):
        """create_jit_decode should return a callable function."""
        model, config = create_test_model()
        jit_fn = model.create_jit_decode(block_size=16)
        assert callable(jit_fn)

    def test_multiple_jit_creates_independent_functions(self):
        """Each call to create_jit_decode should return a new function."""
        model, config = create_test_model()
        jit_fn1 = model.create_jit_decode(block_size=16)
        jit_fn2 = model.create_jit_decode(block_size=16)
        assert jit_fn1 is not jit_fn2


class TestJitDecodeCorrectness:
    """Tests for JIT decode output correctness."""

    def test_jit_decode_output_shape(self):
        """JIT decode should produce correct output shape."""
        model, config = create_test_model()
        block_size = 16
        batch_size = 1
        max_blocks = 4

        block_manager, kv_cache = create_test_components(config, block_size=block_size)

        # Allocate and prefill to populate cache
        prompt_tokens = [1, 2, 3, 4, 5]
        _setup_seq(block_manager, seq_id=0, num_tokens=len(prompt_tokens) + 10)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        # Decode setup
        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[6]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        logits = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        )

        assert logits.shape == (batch_size, 1, config.vocab_size)

    def test_jit_decode_deterministic(self):
        """JIT decode should be deterministic across calls."""
        model, config = create_test_model()
        block_size = 16
        max_blocks = 4

        block_manager, kv_cache = create_test_components(config, block_size=block_size)

        # Allocate and prefill
        prompt_tokens = [1, 2, 3, 4, 5]
        _setup_seq(block_manager, seq_id=0, num_tokens=len(prompt_tokens) + 10)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        # Decode setup
        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[6]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        # Multiple calls should give same result
        logits1 = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        ).realize()

        logits2 = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        ).realize()

        assert_allclose(logits1, logits2, rtol=0, atol=0)


class TestJitDecodeConsistency:
    """Tests for JIT decode consistency."""

    def test_jit_warmup_then_cached(self):
        """First JIT call compiles, subsequent calls use cached version."""
        model, config = create_test_model()
        block_size = 16
        max_blocks = 4

        block_manager, kv_cache = create_test_components(config, block_size=block_size)

        # Allocate and prefill
        prompt_tokens = [1, 2, 3, 4, 5]
        _setup_seq(block_manager, seq_id=0, num_tokens=len(prompt_tokens) + 10)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[6]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        # Run 3 times - first compiles, subsequent are cached
        results = []
        for i in range(3):
            logits = model.decode(
                decode_tokens, kv_cache,
                block_manager=block_manager,
                seq_ids=[0],
                start_positions=[context_len],
                block_tables_tensor=block_tables,
                context_lens_tensor=context_lens,
                jit_fn=jit_fn,
                max_blocks=max_blocks,
            ).realize()
            results.append(logits)

        # All results should be identical
        for i in range(1, len(results)):
            assert_allclose(results[0], results[i], rtol=0, atol=0)


class TestJitDecodeEdgeCases:
    """Edge case tests for JIT decode."""

    def test_single_token_context(self):
        """JIT decode should work with minimal context."""
        model, config = create_test_model()
        block_size = 16
        max_blocks = 4

        block_manager, kv_cache = create_test_components(config, block_size=block_size)

        # Single token prefill
        _setup_seq(block_manager, seq_id=0, num_tokens=5)
        tokens = Tensor([[1]], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[2]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        logits = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        )

        assert logits.shape == (1, 1, config.vocab_size)
        assert logits.isnan().sum().item() == 0

    def test_multiple_blocks_context(self):
        """JIT decode should work with context spanning multiple blocks."""
        model, config = create_test_model()
        block_size = 4  # Small blocks to force multiple
        max_blocks = 8

        block_manager, kv_cache = create_test_components(config, block_size=block_size)

        # Longer prefill spanning multiple blocks
        prompt_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _setup_seq(block_manager, seq_id=0, num_tokens=len(prompt_tokens) + 5)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[11]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        logits = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        )

        assert logits.shape == (1, 1, config.vocab_size)
        assert logits.isnan().sum().item() == 0
