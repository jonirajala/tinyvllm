"""Scheduler for continuous batching with BlockManager integration."""

from typing import Dict, List, Optional

from .sequence import Request, Sequence, SequenceStatus, SchedulerOutput
from .block_manager import BlockManager
from .pool import ObjectPool


class Scheduler:
    """Manages request queues and batch composition with memory awareness."""

    def __init__(self, max_batch_size: int, block_manager: BlockManager, max_seq_len: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.block_manager = block_manager
        self.waiting: List[Request] = []
        self.running: Dict[int, Sequence] = {}
        self.next_seq_id = 0
        self._batch_pool: ObjectPool[SchedulerOutput] = ObjectPool(SchedulerOutput)
        self._seq_pool: ObjectPool[Sequence] = ObjectPool(lambda: Sequence(0, 0, []))
        self._last_batch: Optional[SchedulerOutput] = None

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue."""
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        """Check if there are any unfinished requests."""
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule(self) -> SchedulerOutput:
        """Select sequences for the next batch."""
        if self._last_batch:
            self._batch_pool.release(self._last_batch.reset())

        batch = self._batch_pool.acquire()
        batch.reset()
        batch.scheduled_seqs = list(self.running.values())

        while len(batch.scheduled_seqs) < self.max_batch_size and self.waiting:
            request = self.waiting[0]
            num_tokens = len(request.prompt_tokens)

            if not self.block_manager.can_allocate(num_tokens):
                break
            self.block_manager.allocate_sequence(self.next_seq_id, num_tokens)

            self.waiting.pop(0)
            sequence = self._seq_pool.acquire()
            sequence.reset(self.next_seq_id, request.request_id, request.prompt_tokens)
            request.sequence = sequence
            self.next_seq_id += 1
            self.running[sequence.seq_id] = sequence
            batch.scheduled_seqs.append(sequence)

        self._last_batch = batch
        return batch

    def update(self, finished_seqs: List[int]) -> None:
        """Remove finished sequences and free their blocks."""
        for seq_id in finished_seqs:
            if seq_id in self.running:
                seq = self.running[seq_id]
                seq.status = SequenceStatus.FINISHED
                del self.running[seq_id]
                self.block_manager.free_sequence(seq_id)
                self._seq_pool.release(seq)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running sequences."""
        return len(self.running)

    def abort_request(self, request_id: int) -> bool:
        """Abort a request. Returns True if found and aborted."""
        for i, req in enumerate(self.waiting):
            if req.request_id == request_id:
                self.waiting.pop(i)
                return True

        for seq_id, seq in self.running.items():
            if seq.request_id == request_id:
                del self.running[seq_id]
                self.block_manager.free_sequence(seq_id)
                self._seq_pool.release(seq)
                return True

        return False
