"""Tests for error handling.

Tests that appropriate errors are raised for invalid inputs
and edge cases like OOM scenarios.
"""

import pytest
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.kv_cache import KVCache
from tinyvllm.core.block_manager import BlockManager
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine


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


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False):
        tokens = [hash(c) % (self.vocab_size - 3) + 3 for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens):
        return "".join(chr((t % 26) + ord("a")) for t in tokens if t > 2)


class TestBlockManagerOOM:
    """Tests for out of memory conditions in BlockManager."""

    def test_allocate_exceeds_available_blocks(self):
        """Should raise RuntimeError when requesting more blocks than available."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=2, block_size=16)

        # Request 100 tokens = 7 blocks, but only 2 available
        with pytest.raises(RuntimeError, match="Not enough free blocks"):
            bm.allocate_sequence(seq_id=0, num_tokens=100)

    def test_multiple_allocations_exhaust_memory(self):
        """Should raise RuntimeError when cumulative allocations exceed capacity."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=4, block_size=16)

        # First allocation: 32 tokens = 2 blocks
        bm.allocate_sequence(seq_id=0, num_tokens=32)

        # Second allocation: 32 tokens = 2 blocks (uses remaining)
        bm.allocate_sequence(seq_id=1, num_tokens=32)

        # Third allocation: should fail
        with pytest.raises(RuntimeError, match="Not enough free blocks"):
            bm.allocate_sequence(seq_id=2, num_tokens=32)

    def test_get_slot_oom_when_extending(self):
        """Should raise RuntimeError when extending sequence with no free blocks."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=1, block_size=16)

        # Allocate sequence using the only block
        bm.allocate_sequence(seq_id=0, num_tokens=10)

        # Advance to position 16 (needs second block)
        bm.seq_positions[0] = 16

        with pytest.raises(RuntimeError, match="Out of KV cache memory"):
            bm.get_slot(seq_id=0)


class TestBlockManagerBasic:
    """Tests for basic BlockManager operations."""

    def test_can_allocate_returns_false_when_full(self):
        """can_allocate should return False when not enough space."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=2, block_size=16)

        assert bm.can_allocate(num_tokens=100) is False  # Needs 7 blocks

    def test_can_allocate_returns_true_when_sufficient(self):
        """can_allocate should return True when enough space."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)

        assert bm.can_allocate(num_tokens=100) is True  # Needs 7 blocks, have 10

    def test_free_sequence_allows_reallocation(self):
        """After freeing, blocks should be available again."""
        bm = BlockManager(num_gpus=1, blocks_per_gpu=4, block_size=16)

        # Allocate all 4 blocks
        bm.allocate_sequence(seq_id=0, num_tokens=64)

        # Can't allocate more
        assert bm.can_allocate(num_tokens=16) is False

        # Free the sequence
        bm.free_sequence(seq_id=0)

        # Now can allocate again
        assert bm.can_allocate(num_tokens=16) is True
        bm.allocate_sequence(seq_id=1, num_tokens=16)


class TestEngineErrorHandling:
    """Tests for error handling in LLMEngine."""

    def test_engine_handles_empty_prompt(self):
        """Engine should handle empty prompts gracefully."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer)

        # Empty prompt - should at least have BOS token
        params = SamplingParams(max_tokens=5, temperature=0.0)
        engine.add_request("", params)

        # Should run without crashing
        outputs = list(engine.run())
        assert len(outputs) == 1

    def test_engine_respects_max_tokens(self):
        """Engine should stop at max_tokens."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer)

        params = SamplingParams(max_tokens=3, temperature=0.0)
        engine.add_request("test", params)

        outputs = list(engine.run())
        assert len(outputs) == 1
        # Output should be <= max_tokens (might stop early due to EOS)
        assert len(outputs[0].tokens) <= 3


class TestSamplingParams:
    """Tests for SamplingParams validation."""

    def test_default_params(self):
        """Default params should have sensible values."""
        params = SamplingParams()

        assert params.max_tokens > 0
        assert 0 <= params.temperature <= 2.0

    def test_temperature_zero_for_greedy(self):
        """Temperature 0 should enable greedy decoding."""
        params = SamplingParams(temperature=0.0)

        assert params.temperature == 0.0

    def test_custom_max_tokens(self):
        """Should accept custom max_tokens."""
        params = SamplingParams(max_tokens=100)

        assert params.max_tokens == 100


class TestKVCacheBasic:
    """Tests for basic KVCache operations."""

    def test_cache_shape_correct(self):
        """KVCache tensors should have correct shapes."""
        num_layers = 2
        num_blocks = 10
        block_size = 16
        n_kv_heads = 4
        head_dim = 32

        cache = KVCache(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=dtypes.float32,
        )

        assert len(cache.k_cache) == num_layers
        assert len(cache.v_cache) == num_layers

        for layer_idx in range(num_layers):
            assert cache.k_cache[layer_idx].shape == (num_blocks, block_size, n_kv_heads, head_dim)
            assert cache.v_cache[layer_idx].shape == (num_blocks, block_size, n_kv_heads, head_dim)


class TestModelEdgeCases:
    """Edge case tests for model operations."""

    def test_prefill_single_token(self):
        """Prefill should work with a single token."""
        model, config = create_test_model()
        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=10,
            block_size=16,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        block_manager.allocate_sequence(seq_id=0, num_tokens=1)
        tokens = Tensor([[42]], dtype=dtypes.int32)

        logits = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        assert logits.shape == (1, 1, config.vocab_size)
        assert logits.isnan().sum().item() == 0

    def test_prefill_max_sequence(self):
        """Prefill should work up to max_seq_len."""
        model, config = create_test_model()
        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=20, block_size=16)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=20,
            block_size=16,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        seq_len = 64  # Below max_seq_len of 128
        block_manager.allocate_sequence(seq_id=0, num_tokens=seq_len)
        tokens = Tensor([[i % config.vocab_size for i in range(seq_len)]], dtype=dtypes.int32)

        logits = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        assert logits.shape == (1, seq_len, config.vocab_size)
        assert logits.isnan().sum().item() == 0
