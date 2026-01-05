"""Scheduler for continuous batching with BlockManager integration.

The scheduler decides which sequences to process in each step.
It manages two queues:
- waiting: Requests that haven't started (need prefill)
- running: Sequences currently generating (need decode)

Phase 4: Integrates with BlockManager for memory-aware scheduling.
"""

from typing import Dict, List, Optional

from .sequence import Request, Sequence, SequenceStatus, SchedulerOutput
from .block_manager import BlockManager


class Scheduler:
    """
    Manages request queues and batch composition with memory awareness.

    Phase 4: Uses BlockManager to check memory before scheduling.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        block_manager: Optional[BlockManager] = None,
    ):
        """
        Initialize scheduler.

        Args:
            max_batch_size: Max sequences per batch
            max_seq_len: Max tokens per sequence
            block_manager: BlockManager for memory tracking (Phase 4)
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.block_manager = block_manager
        self.waiting: List[Request] = []
        self.running: Dict[int, Sequence] = {}
        self.next_seq_id = 0

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue."""
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        """Check if there are any unfinished requests."""
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> SchedulerOutput:
        """
        Select sequences for the next batch.

        Phase 4: Checks BlockManager.can_allocate before adding new requests.

        Returns:
            SchedulerOutput with sequences to process
        """
        batch = SchedulerOutput()

        # 1. All running sequences get a decode slot
        batch.scheduled_seqs = list(self.running.values())

        # 2. If batch has room, add waiting requests (prefill)
        while len(batch.scheduled_seqs) < self.max_batch_size and self.waiting:
            request = self.waiting[0]
            num_tokens = len(request.prompt_tokens)

            # Phase 4: Check if we have memory for this sequence
            if self.block_manager is not None:
                if not self.block_manager.can_allocate(num_tokens):
                    # No memory available, stop adding new requests
                    break
                # Allocate blocks for this sequence
                self.block_manager.allocate_sequence(self.next_seq_id, num_tokens)

            # Remove from waiting and create sequence
            self.waiting.pop(0)
            sequence = request.create_sequence(self.next_seq_id)
            self.next_seq_id += 1
            self.running[sequence.seq_id] = sequence
            batch.scheduled_seqs.append(sequence)

        return batch

    def update(self, finished_seqs: List[int]) -> None:
        """
        Update scheduler after decode step(s).

        Removes finished sequences and frees their blocks.
        Note: Tokens are appended directly to sequences in engine.step(),
        so this method only handles cleanup.

        Args:
            finished_seqs: List of seq_ids that finished
        """
        for seq_id in finished_seqs:
            if seq_id in self.running:
                self.running[seq_id].status = SequenceStatus.FINISHED
                del self.running[seq_id]
                if self.block_manager is not None:
                    self.block_manager.free_sequence(seq_id)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running sequences."""
        return len(self.running)

    def abort_request(self, request_id: int) -> bool:
        """
        Abort a request (remove from waiting or running).

        Returns:
            True if request was found and aborted
        """
        # Check waiting queue
        for i, req in enumerate(self.waiting):
            if req.request_id == request_id:
                self.waiting.pop(i)
                return True

        # Check running dict
        for seq_id, seq in list(self.running.items()):
            if seq.request_id == request_id:
                del self.running[seq_id]
                if self.block_manager is not None:
                    self.block_manager.free_sequence(seq_id)
                return True

        return False

