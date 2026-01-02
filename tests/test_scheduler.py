"""Tests for Scheduler class."""

import pytest

from tinyvllm.core.scheduler import Scheduler
from tinyvllm.core.sequence import Request, Sequence, SequenceStatus, SchedulerOutput
from tinyvllm.engine.sampling import SamplingParams


def make_request(request_id: int, prompt_tokens=None) -> Request:
    """Helper to create a test request."""
    if prompt_tokens is None:
        prompt_tokens = [1, 2, 3]
    return Request(
        request_id=request_id,
        prompt="test",
        prompt_tokens=prompt_tokens,
        sampling_params=SamplingParams(),
    )


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scheduler = Scheduler()
        assert scheduler.max_batch_size == 8
        assert scheduler.max_seq_len == 2048
        assert scheduler.waiting == []
        assert scheduler.running == {}
        assert scheduler.next_seq_id == 0

    def test_custom_init(self):
        """Test custom initialization."""
        scheduler = Scheduler(max_batch_size=4, max_seq_len=1024)
        assert scheduler.max_batch_size == 4
        assert scheduler.max_seq_len == 1024


class TestAddRequest:
    """Tests for add_request method."""

    def test_add_single_request(self):
        """Test adding a single request."""
        scheduler = Scheduler()
        req = make_request(0)

        scheduler.add_request(req)

        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0] is req

    def test_add_multiple_requests_fcfs(self):
        """Test requests are added in FCFS order."""
        scheduler = Scheduler()
        req1 = make_request(1)
        req2 = make_request(2)
        req3 = make_request(3)

        scheduler.add_request(req1)
        scheduler.add_request(req2)
        scheduler.add_request(req3)

        assert len(scheduler.waiting) == 3
        assert scheduler.waiting[0].request_id == 1
        assert scheduler.waiting[1].request_id == 2
        assert scheduler.waiting[2].request_id == 3


class TestHasUnfinished:
    """Tests for has_unfinished method."""

    def test_empty_scheduler(self):
        """Test empty scheduler has no unfinished."""
        scheduler = Scheduler()
        assert not scheduler.has_unfinished()

    def test_with_waiting(self):
        """Test has_unfinished with waiting requests."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        assert scheduler.has_unfinished()

    def test_with_running(self):
        """Test has_unfinished with running sequences."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.schedule()  # Move to running
        assert scheduler.has_unfinished()

    def test_after_all_finished(self):
        """Test has_unfinished after all complete."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq_id = batch.scheduled_seqs[0].seq_id
        scheduler.update({seq_id: 100}, [seq_id])
        assert not scheduler.has_unfinished()


class TestSchedule:
    """Tests for schedule method."""

    def test_schedule_empty(self):
        """Test schedule with no requests."""
        scheduler = Scheduler()
        batch = scheduler.schedule()
        assert batch.scheduled_seqs == []

    def test_schedule_single_request(self):
        """Test schedule with single request."""
        scheduler = Scheduler()
        req = make_request(0, [1, 2, 3])
        scheduler.add_request(req)

        batch = scheduler.schedule()

        assert len(batch.scheduled_seqs) == 1
        seq = batch.scheduled_seqs[0]
        assert seq.prompt_tokens == [1, 2, 3]
        assert seq.status == SequenceStatus.RUNNING
        assert seq.seq_id == 0

    def test_schedule_moves_to_running(self):
        """Test schedule moves request to running."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))

        batch = scheduler.schedule()

        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 1
        seq_id = batch.scheduled_seqs[0].seq_id
        assert seq_id in scheduler.running

    def test_schedule_includes_running_sequences(self):
        """Test schedule includes already running sequences."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))

        # First schedule - both become running
        batch1 = scheduler.schedule()
        assert len(batch1.scheduled_seqs) == 2

        # Update with tokens but don't finish
        for seq in batch1.scheduled_seqs:
            scheduler.update({seq.seq_id: 100}, [])

        # Second schedule - running sequences still included
        batch2 = scheduler.schedule()
        assert len(batch2.scheduled_seqs) == 2

    def test_schedule_respects_max_batch_size(self):
        """Test schedule respects max_batch_size."""
        scheduler = Scheduler(max_batch_size=2)
        for i in range(5):
            scheduler.add_request(make_request(i))

        batch = scheduler.schedule()

        assert len(batch.scheduled_seqs) == 2
        assert len(scheduler.waiting) == 3
        assert len(scheduler.running) == 2

    def test_schedule_increments_seq_id(self):
        """Test schedule increments seq_id for each new sequence."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))
        scheduler.add_request(make_request(2))

        batch = scheduler.schedule()

        seq_ids = [seq.seq_id for seq in batch.scheduled_seqs]
        assert seq_ids == [0, 1, 2]
        assert scheduler.next_seq_id == 3

    def test_schedule_adds_waiting_when_room(self):
        """Test schedule adds waiting requests when batch has room."""
        scheduler = Scheduler(max_batch_size=4)

        # Add first request, schedule it
        scheduler.add_request(make_request(0))
        batch1 = scheduler.schedule()
        assert len(batch1.scheduled_seqs) == 1

        # Update without finishing
        scheduler.update({0: 100}, [])

        # Add more requests
        scheduler.add_request(make_request(1))
        scheduler.add_request(make_request(2))

        # Schedule again - should include running + new
        batch2 = scheduler.schedule()
        assert len(batch2.scheduled_seqs) == 3


class TestUpdate:
    """Tests for update method."""

    def test_update_appends_tokens(self):
        """Test update appends tokens to sequences."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq = batch.scheduled_seqs[0]

        scheduler.update({seq.seq_id: 100}, [])

        assert seq.output_tokens == [100]

    def test_update_multiple_tokens(self):
        """Test update with multiple sequences."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))
        batch = scheduler.schedule()

        outputs = {seq.seq_id: seq.seq_id + 100 for seq in batch.scheduled_seqs}
        scheduler.update(outputs, [])

        for seq in batch.scheduled_seqs:
            assert seq.output_tokens == [seq.seq_id + 100]

    def test_update_removes_finished(self):
        """Test update removes finished sequences from running."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq_id = batch.scheduled_seqs[0].seq_id

        assert seq_id in scheduler.running
        scheduler.update({seq_id: 100}, [seq_id])

        assert seq_id not in scheduler.running
        assert not scheduler.has_unfinished()

    def test_update_sets_finished_status(self):
        """Test update sets FINISHED status."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq = batch.scheduled_seqs[0]

        scheduler.update({seq.seq_id: 100}, [seq.seq_id])

        assert seq.status == SequenceStatus.FINISHED

    def test_update_partial_finish(self):
        """Test update with some sequences finishing."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))
        batch = scheduler.schedule()

        # Finish only first sequence
        seq0_id = batch.scheduled_seqs[0].seq_id
        seq1_id = batch.scheduled_seqs[1].seq_id
        scheduler.update({seq0_id: 100, seq1_id: 200}, [seq0_id])

        assert seq0_id not in scheduler.running
        assert seq1_id in scheduler.running
        assert scheduler.has_unfinished()


class TestGetNumWaitingRunning:
    """Tests for get_num_waiting and get_num_running."""

    def test_initial_counts(self):
        """Test initial counts are zero."""
        scheduler = Scheduler()
        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0

    def test_counts_after_add(self):
        """Test counts after adding requests."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))

        assert scheduler.get_num_waiting() == 2
        assert scheduler.get_num_running() == 0

    def test_counts_after_schedule(self):
        """Test counts after scheduling."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))
        scheduler.schedule()

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 2

    def test_counts_after_finish(self):
        """Test counts after finishing."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq_id = batch.scheduled_seqs[0].seq_id
        scheduler.update({seq_id: 100}, [seq_id])

        assert scheduler.get_num_waiting() == 0
        assert scheduler.get_num_running() == 0


class TestAbortRequest:
    """Tests for abort_request method."""

    def test_abort_waiting_request(self):
        """Test aborting a waiting request."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.add_request(make_request(1))

        result = scheduler.abort_request(0)

        assert result is True
        assert scheduler.get_num_waiting() == 1
        assert scheduler.waiting[0].request_id == 1

    def test_abort_running_request(self):
        """Test aborting a running request."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        scheduler.schedule()

        result = scheduler.abort_request(0)

        assert result is True
        assert scheduler.get_num_running() == 0

    def test_abort_nonexistent_request(self):
        """Test aborting a request that doesn't exist."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))

        result = scheduler.abort_request(999)

        assert result is False
        assert scheduler.get_num_waiting() == 1

    def test_abort_already_finished(self):
        """Test aborting already finished request returns False."""
        scheduler = Scheduler()
        scheduler.add_request(make_request(0))
        batch = scheduler.schedule()
        seq_id = batch.scheduled_seqs[0].seq_id
        scheduler.update({seq_id: 100}, [seq_id])

        result = scheduler.abort_request(0)

        assert result is False


class TestFullWorkflow:
    """Integration tests for scheduler workflow."""

    def test_single_request_lifecycle(self):
        """Test complete lifecycle of a single request."""
        scheduler = Scheduler()

        # Add request
        req = make_request(0, [1, 2, 3])
        scheduler.add_request(req)
        assert scheduler.get_num_waiting() == 1

        # Schedule (prefill)
        batch = scheduler.schedule()
        assert len(batch.scheduled_seqs) == 1
        seq = batch.scheduled_seqs[0]
        assert scheduler.get_num_running() == 1

        # Generate tokens
        for i in range(5):
            scheduler.update({seq.seq_id: 100 + i}, [])
            batch = scheduler.schedule()
            assert len(batch.scheduled_seqs) == 1

        assert seq.output_tokens == [100, 101, 102, 103, 104]

        # Finish
        scheduler.update({seq.seq_id: 200}, [seq.seq_id])
        assert not scheduler.has_unfinished()

    def test_multiple_requests_lifecycle(self):
        """Test multiple requests with staggered completion."""
        scheduler = Scheduler(max_batch_size=3)

        # Add 3 requests
        for i in range(3):
            scheduler.add_request(make_request(i))

        # Schedule all
        batch = scheduler.schedule()
        assert len(batch.scheduled_seqs) == 3

        # Generate 2 tokens for all
        for _ in range(2):
            outputs = {seq.seq_id: 100 for seq in scheduler.running.values()}
            scheduler.update(outputs, [])

        # Finish first request
        first_seq_id = list(scheduler.running.keys())[0]
        outputs = {seq_id: 100 for seq_id in scheduler.running.keys()}
        scheduler.update(outputs, [first_seq_id])

        assert scheduler.get_num_running() == 2

        # Add new request
        scheduler.add_request(make_request(10))
        batch = scheduler.schedule()
        assert len(batch.scheduled_seqs) == 3  # 2 running + 1 new
