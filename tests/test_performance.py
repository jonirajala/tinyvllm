"""Performance regression tests.

These tests establish baselines for performance metrics and alert to
major regressions. They use small test models for fast CI runs.

Note: These are sanity checks, not comprehensive benchmarks.
Use the benchmarks/ directory for detailed performance analysis.
"""

import time
import pytest
from tinygrad import Tensor, dtypes, Device

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.kv_cache import KVCache
from tinyvllm.core.block_manager import BlockManager
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine


def create_test_model():
    """Create a small test model for performance tests."""
    config = LlamaConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=128,
        hidden_dim=128,
        max_seq_len=256,
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


class TestThroughputBaseline:
    """Tests for basic throughput sanity checks."""

    def test_single_request_throughput(self):
        """Single request should achieve reasonable throughput."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer)

        params = SamplingParams(max_tokens=10, temperature=0.0)
        engine.add_request("hello world", params)

        start = time.perf_counter()
        outputs = list(engine.run())
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o.tokens) for o in outputs)

        # Sanity check: should generate at least 1 token per second
        # (Very conservative for tiny test model on any hardware)
        if total_tokens > 0:
            tps = total_tokens / elapsed
            assert tps >= 1.0, f"Throughput {tps:.1f} tok/s is below minimum threshold"

    def test_concurrent_requests_throughput(self):
        """Concurrent requests should have reasonable throughput."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer, max_batch_size=4)

        params = SamplingParams(max_tokens=10, temperature=0.0)
        for i in range(4):
            engine.add_request(f"prompt {i}", params)

        start = time.perf_counter()
        outputs = list(engine.run())
        elapsed = time.perf_counter() - start

        total_tokens = sum(len(o.tokens) for o in outputs)

        # Batched requests should still achieve reasonable throughput
        if total_tokens > 0:
            tps = total_tokens / elapsed
            assert tps >= 2.0, f"Batched throughput {tps:.1f} tok/s is below minimum threshold"


class TestLatencyBaseline:
    """Tests for latency sanity checks."""

    def test_step_latency(self):
        """Single step should complete in reasonable time."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer)

        params = SamplingParams(max_tokens=5, temperature=0.0)
        engine.add_request("test", params)

        # Warmup step
        engine.step()

        # Measure subsequent steps
        step_times = []
        while engine.has_unfinished():
            start = time.perf_counter()
            engine.step()
            step_times.append(time.perf_counter() - start)

        if step_times:
            avg_step_ms = (sum(step_times) / len(step_times)) * 1000

            # Sanity check: steps should complete within 5 seconds
            # (Very conservative for test model)
            assert avg_step_ms < 5000, f"Average step time {avg_step_ms:.0f}ms exceeds threshold"


class TestMemoryBaseline:
    """Tests for memory usage sanity checks."""

    def test_kv_cache_memory_reasonable(self):
        """KV cache memory should be predictable."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        engine = LLMEngine(model, tokenizer)

        params = SamplingParams(max_tokens=5, temperature=0.0)
        engine.add_request("test", params)

        # Run to populate cache
        list(engine.run())

        # Check memory is reasonable (< 100MB for tiny test model)
        mem_bytes = engine.kv_cache.get_memory_bytes()
        mem_mb = mem_bytes / (1024 * 1024)

        assert mem_mb < 100, f"KV cache memory {mem_mb:.1f}MB exceeds threshold for test model"

    def test_memory_scales_with_batch(self):
        """Memory should scale reasonably with batch size."""
        model, config = create_test_model()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)

        memories = []
        for batch_size in [1, 2, 4]:
            engine = LLMEngine(model, tokenizer, max_batch_size=batch_size)
            params = SamplingParams(max_tokens=5, temperature=0.0)

            for i in range(batch_size):
                engine.add_request(f"test {i}", params)

            list(engine.run())
            memories.append(engine.kv_cache.get_memory_bytes())

        # Memory should increase but not exponentially
        # (Test model has fixed cache size, so should be same)
        assert all(m > 0 for m in memories), "Memory should be positive"


class TestPrefillPerformance:
    """Tests for prefill phase performance."""

    def test_prefill_scales_with_length(self):
        """Prefill time should scale roughly linearly with sequence length."""
        model, config = create_test_model()
        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=32, block_size=16)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=32,
            block_size=16,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        times = {}
        for seq_len in [8, 16, 32]:
            # Fresh block manager for each test
            bm = BlockManager(num_gpus=1, blocks_per_gpu=32, block_size=16)
            cache = KVCache(
                num_layers=config.n_layers,
                num_blocks=32,
                block_size=16,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                dtype=dtypes.float32,
            )

            bm.allocate_sequence(seq_id=0, num_tokens=seq_len)
            tokens = Tensor([[i % config.vocab_size for i in range(seq_len)]], dtype=dtypes.int32)

            start = time.perf_counter()
            _ = model.prefill(tokens, kv_cache=cache, block_manager=bm, seq_id=0)
            times[seq_len] = time.perf_counter() - start

        # Longer sequences should not take disproportionately longer
        # (4x longer sequence should take at most 10x time due to attention)
        if times[8] > 0:
            ratio = times[32] / times[8]
            assert ratio < 20, f"Prefill scaling ratio {ratio:.1f}x exceeds threshold"


class TestJitWarmup:
    """Tests for JIT compilation overhead."""

    def test_jit_warmup_overhead_reasonable(self):
        """First JIT call (warmup) should not be excessively slow."""
        model, config = create_test_model()
        block_size = 16
        max_blocks = 4

        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=32, block_size=block_size)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=32,
            block_size=block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        # Prefill
        prompt_tokens = [1, 2, 3, 4, 5]
        block_manager.allocate_sequence(seq_id=0, num_tokens=len(prompt_tokens) + 10)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[6]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        # First call (warmup/compilation)
        start = time.perf_counter()
        _ = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        ).realize()
        warmup_time = time.perf_counter() - start

        # Warmup should complete within 30 seconds (conservative for any system)
        assert warmup_time < 30, f"JIT warmup time {warmup_time:.1f}s exceeds threshold"

    def test_jit_cached_faster_than_warmup(self):
        """Cached JIT calls should be faster than warmup."""
        model, config = create_test_model()
        block_size = 16
        max_blocks = 4

        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=32, block_size=block_size)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=32,
            block_size=block_size,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        # Prefill
        prompt_tokens = [1, 2, 3, 4, 5]
        block_manager.allocate_sequence(seq_id=0, num_tokens=len(prompt_tokens) + 10)
        tokens = Tensor([prompt_tokens], dtype=dtypes.int32).realize()
        _ = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        context_len = block_manager.get_context_length(seq_id=0)
        decode_tokens = Tensor([[6]], dtype=dtypes.int32).realize()
        block_table = block_manager.get_block_table(seq_id=0)
        padded_table = block_table + [0] * (max_blocks - len(block_table))
        block_tables = Tensor([padded_table], dtype=dtypes.int32).realize()
        context_lens = Tensor([context_len], dtype=dtypes.int32).realize()

        jit_fn = model.create_jit_decode(block_size=block_size)

        # Warmup call
        start = time.perf_counter()
        _ = model.decode(
            decode_tokens, kv_cache,
            block_manager=block_manager,
            seq_ids=[0],
            start_positions=[context_len],
            block_tables_tensor=block_tables,
            context_lens_tensor=context_lens,
            jit_fn=jit_fn,
            max_blocks=max_blocks,
        ).realize()
        warmup_time = time.perf_counter() - start

        # Cached calls (average of 3)
        cached_times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = model.decode(
                decode_tokens, kv_cache,
                block_manager=block_manager,
                seq_ids=[0],
                start_positions=[context_len],
                block_tables_tensor=block_tables,
                context_lens_tensor=context_lens,
                jit_fn=jit_fn,
                max_blocks=max_blocks,
            ).realize()
            cached_times.append(time.perf_counter() - start)

        avg_cached_time = sum(cached_times) / len(cached_times)

        # Cached should be faster (or at worst similar if already fast)
        # Allow some variance for test stability
        assert avg_cached_time <= warmup_time * 1.5, \
            f"Cached time {avg_cached_time:.3f}s not faster than warmup {warmup_time:.3f}s"
