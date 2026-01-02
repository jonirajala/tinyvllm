"""Scheduler for continuous batching.

The scheduler decides which sequences to process in each step.
It manages two queues:
- waiting: Requests that haven't started (need prefill)
- running: Sequences currently generating (need decode)

Key concepts:

1. FCFS (First-Come-First-Served): Earlier requests get priority
2. Prefill vs Decode:
   - Prefill: Process entire prompt at once (expensive)
   - Decode: Generate one token (cheap, done repeatedly)
3. Batching: Combine multiple sequences into one GPU call

Scheduling loop:
    1. Check if any running sequences finished
    2. Move finished to done, free their resources
    3. Check if we can add waiting requests (memory available?)
    4. Return batch of sequences to process

Example:
    scheduler = Scheduler(max_batch_size=8, max_seq_len=2048)

    scheduler.add_request(request1)
    scheduler.add_request(request2)

    while scheduler.has_unfinished():
        output = scheduler.schedule()
        # output.scheduled_seqs contains sequences to process
        # Process them, then call scheduler.update() with results
"""

from typing import Dict, List

from .sequence import Request, Sequence, SequenceStatus, SchedulerOutput


class Scheduler:
    """
    Manages request queues and batch composition.

    The scheduler is responsible for:
    1. Managing waiting queue (requests not yet started)
    2. Managing running queue (sequences generating)
    3. Deciding which sequences to batch together
    4. Coordinating with BlockManager for memory

    Attributes:
        max_batch_size: Maximum sequences in one batch
        max_seq_len: Maximum sequence length allowed
        waiting: Queue of requests waiting to start
        running: Dict of seq_id -> Sequence currently generating
        next_seq_id: Counter for assigning sequence IDs

    Example:
        scheduler = Scheduler(max_batch_size=8)
        scheduler.add_request(request)

        output = scheduler.schedule()
        for seq in output.scheduled_seqs:
            # Process sequence
            pass
    """

    def __init__(self, max_batch_size: int = 8, max_seq_len: int = 2048):
        """
        Initialize scheduler.

        Args:
            max_batch_size: Max sequences per batch
            max_seq_len: Max tokens per sequence (Phase 5: length limiting)
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len  # Phase 5: Will enforce length limits
        self.waiting = []
        self.running = {}
        self.next_seq_id = 0

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the waiting queue.

        Args:
            request: The request to add

        """
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        """
        Check if there are any unfinished requests.

        Returns:
            True if waiting or running queues are non-empty
        """
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> SchedulerOutput:
        """
        Select sequences for the next batch.

        This is the core scheduling logic:
        1. All running sequences get a decode slot (they need to continue)
        2. If batch has room, add waiting requests (prefill them)
        3. Return the batch

        Returns:
            SchedulerOutput with sequences to process
        """
        batch = SchedulerOutput()

        # 1. All running sequences get a decode slot
        batch.scheduled_seqs = list(self.running.values())

        # 2. If batch has room, add waiting requests (prefill)
        while len(batch.scheduled_seqs) < self.max_batch_size and self.waiting:
            request = self.waiting.pop(0)
            sequence = request.create_sequence(self.next_seq_id)
            self.next_seq_id += 1
            self.running[sequence.seq_id] = sequence
            batch.scheduled_seqs.append(sequence)

        return batch


    def update(
        self,
        seq_outputs: Dict[int, int],
        finished_seqs: List[int],
    ) -> None:
        """
        Update sequences after a batch step.

        Called after engine processes a batch. Updates sequence state
        and handles finished sequences.

        Args:
            seq_outputs: Dict of seq_id -> generated token
            finished_seqs: List of seq_ids that finished (EOS or max_len)

        """
        for seq_id, token in seq_outputs.items():
            if seq_id in self.running:
                self.running[seq_id].append_token(token)

        for seq_id in finished_seqs:
            if seq_id in self.running:
                self.running[seq_id].status = SequenceStatus.FINISHED
                del self.running[seq_id]

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running sequences."""
        return len(self.running)

    def abort_request(self, request_id: int) -> bool:
        """
        Abort a request (remove from waiting or running).

        Args:
            request_id: ID of request to abort

        Returns:
            True if request was found and aborted

        """
        # Check waiting queue
        for i, req in enumerate(self.waiting):
            if req.request_id == request_id:
                self.waiting.pop(i)
                return True

        # Check running dict (keyed by seq_id, need to find by request_id)
        for seq_id, seq in list(self.running.items()):
            if seq.request_id == request_id:
                del self.running[seq_id]
                return True

        return False
