"""Tests for Sequence and Request classes."""

import pytest
import time

from tinyvllm.core.sequence import Sequence, Request, SequenceStatus, SchedulerOutput
from tinyvllm.core.sampling import SamplingParams


class TestSequence:
    """Tests for Sequence dataclass."""

    def test_create_sequence(self):
        """Test basic sequence creation."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        assert seq.seq_id == 0
        assert seq.request_id == 0
        assert seq.prompt_tokens == [1, 2, 3]
        assert seq.output_tokens == []
        assert seq.status == SequenceStatus.WAITING

    def test_get_all_tokens_prompt_only(self):
        """Test get_all_tokens with only prompt."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        assert seq.get_all_tokens() == [1, 2, 3]

    def test_get_all_tokens_with_output(self):
        """Test get_all_tokens with prompt and output."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        seq.output_tokens = [4, 5]
        assert seq.get_all_tokens() == [1, 2, 3, 4, 5]

    def test_get_len(self):
        """Test get_len returns total length."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        assert seq.get_len() == 3
        seq.output_tokens = [4, 5]
        assert seq.get_len() == 5

    def test_get_prompt_len(self):
        """Test get_prompt_len."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3, 4])
        assert seq.get_prompt_len() == 4

    def test_get_output_len(self):
        """Test get_output_len."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        assert seq.get_output_len() == 0
        seq.output_tokens = [4, 5, 6]
        assert seq.get_output_len() == 3

    def test_append_token(self):
        """Test append_token adds to output."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(4)
        assert seq.output_tokens == [4]
        seq.append_token(5)
        assert seq.output_tokens == [4, 5]

    def test_is_finished(self):
        """Test is_finished checks status."""
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        assert not seq.is_finished()
        seq.status = SequenceStatus.RUNNING
        assert not seq.is_finished()
        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished()


class TestRequest:
    """Tests for Request dataclass."""

    def test_create_request(self):
        """Test basic request creation."""
        params = SamplingParams()
        req = Request(
            request_id=0,
            prompt="Hello",
            prompt_tokens=[1, 2, 3],
            sampling_params=params,
        )
        assert req.request_id == 0
        assert req.prompt == "Hello"
        assert req.prompt_tokens == [1, 2, 3]
        assert req.sequence is None

    def test_arrival_time_auto_set(self):
        """Test arrival_time is automatically set."""
        before = time.time()
        req = Request(
            request_id=0,
            prompt="Hello",
            prompt_tokens=[1],
            sampling_params=SamplingParams(),
        )
        after = time.time()
        assert before <= req.arrival_time <= after

    def test_create_sequence(self):
        """Test create_sequence creates and attaches sequence."""
        req = Request(
            request_id=5,
            prompt="Hello",
            prompt_tokens=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        seq = req.create_sequence(seq_id=10)

        assert seq is not None
        assert seq.seq_id == 10
        assert seq.request_id == 5
        assert seq.prompt_tokens == [1, 2, 3]
        assert seq.status == SequenceStatus.RUNNING
        assert req.sequence is seq

    def test_create_sequence_multiple_times(self):
        """Test create_sequence replaces previous sequence."""
        req = Request(
            request_id=0,
            prompt="Hello",
            prompt_tokens=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        seq1 = req.create_sequence(seq_id=1)
        seq2 = req.create_sequence(seq_id=2)

        assert req.sequence is seq2
        assert seq2.seq_id == 2


class TestSchedulerOutput:
    """Tests for SchedulerOutput dataclass."""

    def test_default_empty(self):
        """Test default SchedulerOutput is empty."""
        output = SchedulerOutput()
        assert output.scheduled_seqs == []
        assert output.num_prefill_tokens == 0
        assert output.num_decode_tokens == 0

    def test_with_sequences(self):
        """Test SchedulerOutput with sequences."""
        seq1 = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2])
        seq2 = Sequence(seq_id=1, request_id=1, prompt_tokens=[3, 4, 5])

        output = SchedulerOutput(
            scheduled_seqs=[seq1, seq2],
            num_prefill_tokens=5,
            num_decode_tokens=2,
        )

        assert len(output.scheduled_seqs) == 2
        assert output.num_prefill_tokens == 5
        assert output.num_decode_tokens == 2
