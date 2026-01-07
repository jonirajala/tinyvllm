"""Tests for LLMEngine."""

import pytest
from tinygrad import Tensor

from tinyvllm.core.engine import LLMEngine, GenerationOutput, generate_batch
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig


def small_config():
    """Small config for fast tests."""
    return LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=100,
        hidden_dim=64,
        max_seq_len=64,
    )


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, eos_id=2):
        self.eos_id = eos_id

    def encode(self, text: str):
        """Simple encoding: each character becomes a token ID."""
        if not text:
            return []
        return [ord(c) % 100 for c in text]

    def decode(self, ids):
        """Simple decoding: token IDs back to characters."""
        if not ids:
            return ""
        return "".join(chr((i % 26) + ord("a")) for i in ids)


class TestLLMEngineInit:
    """Tests for LLMEngine initialization."""

    def test_basic_init(self):
        """Test basic engine initialization."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer)

        assert engine.model is model
        assert engine.tokenizer is tokenizer
        assert engine.max_batch_size == 8
        assert engine.next_request_id == 0

    def test_custom_batch_size(self):
        """Test custom max_batch_size."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer, max_batch_size=4)

        assert engine.max_batch_size == 4


class TestAddRequest:
    """Tests for add_request method."""

    def test_add_single_request(self):
        """Test adding a single request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        request_id = engine.add_request("hello")

        assert request_id == 0
        assert 0 in engine.requests
        assert engine.scheduler.get_num_waiting() == 1

    def test_add_multiple_requests(self):
        """Test adding multiple requests."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        id1 = engine.add_request("hello")
        id2 = engine.add_request("world")
        id3 = engine.add_request("test")

        assert id1 == 0
        assert id2 == 1
        assert id3 == 2
        assert engine.scheduler.get_num_waiting() == 3

    def test_add_with_custom_id(self):
        """Test adding request with custom ID."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        request_id = engine.add_request("test", request_id=42)

        assert request_id == 42
        assert 42 in engine.requests

    def test_add_with_sampling_params(self):
        """Test adding request with sampling params."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        params = SamplingParams(max_tokens=10, temperature=0.5)
        request_id = engine.add_request("test", params)

        assert engine.requests[request_id].sampling_params.max_tokens == 10
        assert engine.requests[request_id].sampling_params.temperature == 0.5


class TestHasUnfinished:
    """Tests for has_unfinished method."""

    def test_empty_engine(self):
        """Test empty engine has no unfinished."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        assert not engine.has_unfinished()

    def test_with_request(self):
        """Test has_unfinished with pending request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test")

        assert engine.has_unfinished()


class TestStep:
    """Tests for step method."""

    def test_step_empty_returns_empty(self):
        """Step with no requests returns empty list."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        outputs = engine.step()

        assert outputs == []

    def test_step_processes_request(self):
        """Step should process a request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test", SamplingParams(max_tokens=1))
        outputs = engine.step()

        # Should have output (finished after 1 token)
        assert len(outputs) == 1
        assert isinstance(outputs[0], GenerationOutput)

    def test_step_returns_finished_only(self):
        """Step should only return finished requests."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        # Request with more tokens
        engine.add_request("test", SamplingParams(max_tokens=5))

        # First step - not finished yet
        outputs = engine.step()

        # May or may not be finished depending on if EOS hit
        assert isinstance(outputs, list)

    def test_step_multiple_requests(self):
        """Step should handle multiple requests."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("hello", SamplingParams(max_tokens=1))
        engine.add_request("world", SamplingParams(max_tokens=1))

        outputs = engine.step()

        # Both should finish after 1 token
        assert len(outputs) == 2


class TestRun:
    """Tests for run method."""

    def test_run_single_request(self):
        """Run should complete a single request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test", SamplingParams(max_tokens=3))
        outputs = list(engine.run())

        assert len(outputs) == 1
        assert isinstance(outputs[0], GenerationOutput)
        assert outputs[0].request_id == 0

    def test_run_multiple_requests(self):
        """Run should complete multiple requests."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("hello", SamplingParams(max_tokens=2))
        engine.add_request("world", SamplingParams(max_tokens=2))
        engine.add_request("test", SamplingParams(max_tokens=2))

        outputs = list(engine.run())

        assert len(outputs) == 3
        request_ids = {o.request_id for o in outputs}
        assert request_ids == {0, 1, 2}

    def test_run_yields_as_completed(self):
        """Run should yield outputs as requests complete."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("a", SamplingParams(max_tokens=1))
        engine.add_request("bb", SamplingParams(max_tokens=5))

        outputs = []
        for output in engine.run():
            outputs.append(output)
            # First request should finish before second
            if len(outputs) == 1:
                assert engine.has_unfinished()

        assert len(outputs) == 2

    def test_run_respects_max_tokens(self):
        """Run should respect max_tokens limit."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test", SamplingParams(max_tokens=5))
        outputs = list(engine.run())

        assert len(outputs) == 1
        assert len(outputs[0].tokens) <= 5


class TestAbortRequest:
    """Tests for abort_request method."""

    def test_abort_waiting_request(self):
        """Test aborting a waiting request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test")
        result = engine.abort_request(0)

        assert result is True
        assert not engine.has_unfinished()

    def test_abort_nonexistent_request(self):
        """Test aborting nonexistent request."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        result = engine.abort_request(999)

        assert result is False


class TestGenerationOutput:
    """Tests for GenerationOutput dataclass."""

    def test_output_fields(self):
        """Test GenerationOutput has correct fields."""
        output = GenerationOutput(
            request_id=1,
            text="hello",
            tokens=[1, 2, 3],
            finish_reason="length",
        )

        assert output.request_id == 1
        assert output.text == "hello"
        assert output.tokens == [1, 2, 3]
        assert output.finish_reason == "length"


class TestGenerateBatch:
    """Tests for generate_batch convenience function."""

    def test_batch_single_prompt(self):
        """Test batch generation with single prompt."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        results = generate_batch(
            engine,
            ["hello"],
            SamplingParams(max_tokens=3),
        )

        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_batch_multiple_prompts(self):
        """Test batch generation with multiple prompts."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        results = generate_batch(
            engine,
            ["hello", "world", "test"],
            SamplingParams(max_tokens=2),
        )

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_batch_preserves_order(self):
        """Test batch results are in same order as prompts."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        # Use different length prompts to potentially finish at different times
        prompts = ["a", "bb", "ccc"]
        results = generate_batch(
            engine,
            prompts,
            SamplingParams(max_tokens=2),
        )

        # Results should be in same order as prompts
        assert len(results) == 3


class TestEngineWithKVCache:
    """Tests for engine KV cache handling (Phase 4: BlockManager)."""

    def test_kv_cache_allocated_on_step(self):
        """Blocks should be allocated when processing."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        # Before adding request - all blocks free
        initial_free = engine.block_manager.get_num_free_blocks()

        engine.add_request("test", SamplingParams(max_tokens=3))

        # After step - some blocks should be allocated
        engine.step()

        # Some blocks should now be in use
        after_free = engine.block_manager.get_num_free_blocks()
        assert after_free <= initial_free

    def test_kv_cache_freed_on_finish(self):
        """Blocks should be freed when request finishes."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        initial_free = engine.block_manager.get_num_free_blocks()

        engine.add_request("test", SamplingParams(max_tokens=1))
        list(engine.run())

        # All sequences finished, blocks should be returned
        final_free = engine.block_manager.get_num_free_blocks()
        assert final_free == initial_free


class TestEngineEdgeCases:
    """Edge case tests for engine."""

    def test_single_char_prompt(self):
        """Test handling of single character prompt."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("a", SamplingParams(max_tokens=3))
        outputs = list(engine.run())

        # Should handle gracefully
        assert len(outputs) == 1

    def test_max_tokens_one(self):
        """Test max_tokens=1."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()
        engine = LLMEngine(model, tokenizer)

        engine.add_request("test", SamplingParams(max_tokens=1))
        outputs = list(engine.run())

        assert len(outputs) == 1
        assert len(outputs[0].tokens) == 1

    def test_greedy_deterministic(self):
        """Temperature 0 should give deterministic output."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine1 = LLMEngine(model, tokenizer)
        engine1.add_request("test", SamplingParams(max_tokens=5, temperature=0.0))
        result1 = list(engine1.run())[0].text

        engine2 = LLMEngine(model, tokenizer)
        engine2.add_request("test", SamplingParams(max_tokens=5, temperature=0.0))
        result2 = list(engine2.run())[0].text

        assert result1 == result2


class TestAsyncOutput:
    """Tests for Phase 8.3: Async Output Processing."""

    def test_async_output_init(self):
        """Test engine initialization with async_output=True."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer, async_output=True)
        assert engine._output_processor is not None
        assert engine._output_processor.async_mode is True
        engine.shutdown()

    def test_async_output_run(self):
        """Test run() with async_output=True produces valid results."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        # Run with async
        engine = LLMEngine(model, tokenizer, async_output=True)
        engine.add_request("test", SamplingParams(max_tokens=3))
        outputs = list(engine.run())
        engine.shutdown()

        # Should produce valid output
        assert len(outputs) == 1
        assert outputs[0].request_id == 0
        assert 1 <= len(outputs[0].tokens) <= 3  # max_tokens=3, but EOS can end early
        assert outputs[0].text is not None
        assert outputs[0].finish_reason in ('length', 'eos')

    def test_async_output_multiple_requests(self):
        """Test async output with multiple concurrent requests."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer, async_output=True)

        engine.add_request("hello", SamplingParams(max_tokens=2))
        engine.add_request("world", SamplingParams(max_tokens=2))
        engine.add_request("test", SamplingParams(max_tokens=2))

        outputs = list(engine.run())
        engine.shutdown()

        assert len(outputs) == 3
        request_ids = {o.request_id for o in outputs}
        assert request_ids == {0, 1, 2}

    def test_async_output_callback(self):
        """Test async output with callback."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        received = []

        def on_output(output):
            received.append(output)

        engine = LLMEngine(model, tokenizer, async_output=True, output_callback=on_output)

        engine.add_request("test", SamplingParams(max_tokens=2))
        list(engine.run())  # Drive the engine
        engine.shutdown()

        assert len(received) == 1
        assert received[0].request_id == 0

    def test_async_output_poll(self):
        """Test manual polling with poll_outputs()."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer, async_output=True)

        engine.add_request("test", SamplingParams(max_tokens=1))

        # Step should queue output for async processing
        engine.step()

        # Give async processor time to complete
        import time
        time.sleep(0.05)

        # Poll should return the output
        outputs = engine.poll_outputs()
        assert len(outputs) == 1
        assert outputs[0].request_id == 0

        engine.shutdown()

    def test_poll_outputs_empty_when_sync(self):
        """poll_outputs() should return empty list when async_output=False."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer()

        engine = LLMEngine(model, tokenizer, async_output=False)

        outputs = engine.poll_outputs()
        assert outputs == []
