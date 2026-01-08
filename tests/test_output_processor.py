"""Tests for Phase 8.3: Output Processing (sync and async modes)."""

import time
from unittest.mock import Mock

import pytest

from tinyvllm.core.output_processor import OutputProcessor, PendingOutput


class MockTokenizer:
    """Mock tokenizer for testing."""

    def decode(self, tokens):
        """Simulate decoding with small delay."""
        time.sleep(0.001)  # Simulate decode time
        return f"decoded:{','.join(map(str, tokens))}"


class TestPendingOutput:
    """Test PendingOutput dataclass."""

    def test_create_pending_output(self):
        pending = PendingOutput(
            request_id=42,
            tokens=[1, 2, 3],
            finish_reason='eos'
        )
        assert pending.request_id == 42
        assert pending.tokens == [1, 2, 3]
        assert pending.finish_reason == 'eos'


class TestOutputProcessorSync:
    """Test OutputProcessor in sync mode (default)."""

    def test_sync_mode_no_thread(self):
        """Sync mode should not start a background thread."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=False)

        assert not processor.is_running
        processor.shutdown()  # Should be no-op

    def test_sync_submit_and_poll(self):
        """Submit should decode immediately in sync mode."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=False)

        processor.submit(
            request_id=1,
            tokens=[10, 20, 30],
            finish_reason='eos'
        )

        # Result should be available immediately
        results = processor.poll_all()
        assert len(results) == 1
        assert results[0].request_id == 1
        assert results[0].text == "decoded:10,20,30"

    def test_sync_callback(self):
        """Test callback in sync mode."""
        tokenizer = MockTokenizer()
        received = []

        def on_output(output):
            received.append(output)

        processor = OutputProcessor(tokenizer, async_mode=False, callback=on_output)
        processor.submit(request_id=1, tokens=[1], finish_reason='eos')

        # Callback should be invoked immediately
        assert len(received) == 1
        assert received[0].request_id == 1

        # Polling should return empty (callback mode)
        assert processor.poll_all() == []


class TestOutputProcessorAsync:
    """Test OutputProcessor in async mode."""

    def test_async_mode_starts_thread(self):
        """Async mode should start a background thread."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        assert processor.is_running
        processor.shutdown()

    def test_async_submit_and_poll(self):
        """Submit should queue for background processing."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        processor.submit(
            request_id=1,
            tokens=[10, 20, 30],
            finish_reason='eos'
        )

        # Wait for background processing
        time.sleep(0.05)

        results = processor.poll_all()
        assert len(results) == 1
        assert results[0].request_id == 1
        assert results[0].text == "decoded:10,20,30"

        processor.shutdown()

    def test_async_multiple(self):
        """Queue multiple outputs in async mode."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        for i in range(5):
            processor.submit(
                request_id=i,
                tokens=[i, i + 1],
                finish_reason='length'
            )

        time.sleep(0.1)
        results = processor.poll_all()
        assert len(results) == 5

        request_ids = {r.request_id for r in results}
        assert request_ids == {0, 1, 2, 3, 4}

        processor.shutdown()

    def test_async_callback(self):
        """Test callback in async mode."""
        tokenizer = MockTokenizer()
        received = []

        def on_output(output):
            received.append(output)

        processor = OutputProcessor(tokenizer, async_mode=True, callback=on_output)

        processor.submit(request_id=1, tokens=[1], finish_reason='eos')
        processor.submit(request_id=2, tokens=[2], finish_reason='eos')

        time.sleep(0.1)

        assert len(received) == 2
        assert processor.poll_all() == []

        processor.shutdown()

    def test_async_callback_error_doesnt_crash(self):
        """Callback errors should not crash the processor."""
        tokenizer = MockTokenizer()

        def bad_callback(output):
            raise ValueError("Callback error!")

        processor = OutputProcessor(tokenizer, async_mode=True, callback=bad_callback)

        processor.submit(request_id=1, tokens=[1], finish_reason='eos')
        time.sleep(0.05)

        assert processor.is_running
        processor.shutdown()

    def test_async_drain(self):
        """Test drain waits for pending outputs."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        for i in range(3):
            processor.submit(request_id=i, tokens=[i], finish_reason='eos')

        results = processor.drain()
        assert len(results) == 3

        processor.shutdown()

    def test_token_copy(self):
        """Tokens should be copied to avoid mutation issues."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        original_tokens = [1, 2, 3]
        processor.submit(request_id=1, tokens=original_tokens, finish_reason='eos')

        # Mutate original list
        original_tokens.append(4)
        original_tokens[0] = 999

        time.sleep(0.05)
        result = processor.poll_all()[0]

        assert result.tokens == [1, 2, 3]
        processor.shutdown()

    def test_shutdown(self):
        """Test clean shutdown."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        assert processor.is_running
        processor.shutdown()
        time.sleep(0.02)
        assert not processor.is_running


class TestOutputProcessorIntegration:
    """Integration tests simulating engine usage."""

    def test_simulate_engine_flow(self):
        """Simulate how engine uses the processor."""
        tokenizer = MockTokenizer()
        processor = OutputProcessor(tokenizer, async_mode=True)

        finished_request_ids = []

        for step in range(10):
            if step % 2 == 0:
                processor.submit(
                    request_id=step,
                    tokens=list(range(step, step + 5)),
                    finish_reason='eos'
                )
                finished_request_ids.append(step)

        time.sleep(0.1)
        results = processor.poll_all()

        result_ids = {r.request_id for r in results}
        assert result_ids == set(finished_request_ids)

        processor.shutdown()
