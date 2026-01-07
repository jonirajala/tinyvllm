"""Output processing - sync or async detokenization.

Phase 8.3: Unified output processor that handles both modes.

Sync mode (default): Decode immediately when submit() is called.
Async mode: Queue tokens, decode on background thread while GPU continues.

Example:
    processor = OutputProcessor(tokenizer, async_mode=True)
    processor.submit(request_id=1, tokens=[1,2,3], finish_reason='eos')
    outputs = processor.poll_all()
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from .engine import GenerationOutput


@dataclass
class PendingOutput:
    """Output waiting for detokenization."""
    request_id: int
    tokens: List[int]
    finish_reason: str


class OutputProcessor:
    """
    Unified output processor - handles both sync and async modes.

    Sync mode: Decode immediately in submit(), no background thread.
    Async mode: Queue tokens, decode on background thread.

    Args:
        tokenizer: Tokenizer with decode(tokens) -> str method
        async_mode: If True, use background thread for decoding
        callback: Optional callback for completed outputs
    """

    def __init__(
        self,
        tokenizer,
        async_mode: bool = False,
        callback: Optional[Callable[[GenerationOutput], None]] = None,
    ):
        self.tokenizer = tokenizer
        self.async_mode = async_mode
        self.callback = callback
        self._results: List[GenerationOutput] = []

        if async_mode:
            self._pending_queue: queue.Queue[PendingOutput] = queue.Queue()
            self._results_lock = threading.Lock()
            self._running = True
            self._thread = threading.Thread(
                target=self._process_loop,
                name="OutputProcessor",
                daemon=True
            )
            self._thread.start()

    def submit(
        self,
        request_id: int,
        tokens: List[int],
        finish_reason: str
    ) -> None:
        """
        Submit completed sequence for processing.

        Sync mode: Decodes immediately and stores result.
        Async mode: Queues for background processing.
        """
        if self.async_mode:
            self._pending_queue.put(PendingOutput(
                request_id=request_id,
                tokens=tokens.copy(),
                finish_reason=finish_reason
            ))
        else:
            # Sync: decode immediately
            from .engine import GenerationOutput
            output = GenerationOutput(
                request_id=request_id,
                text=self.tokenizer.decode(tokens),
                tokens=tokens.copy(),
                finish_reason=finish_reason
            )
            if self.callback:
                self.callback(output)
            else:
                self._results.append(output)

    def poll_all(self) -> List[GenerationOutput]:
        """Get all available results."""
        if self.async_mode:
            with self._results_lock:
                outputs = self._results.copy()
                self._results.clear()
                return outputs
        else:
            outputs = self._results.copy()
            self._results.clear()
            return outputs

    def drain(self, timeout: float = 1.0) -> List[GenerationOutput]:
        """Wait for pending work and return all results."""
        if self.async_mode:
            deadline = time.time() + timeout
            while not self._pending_queue.empty() and time.time() < deadline:
                time.sleep(0.001)
            time.sleep(0.005)
        return self.poll_all()

    def shutdown(self) -> None:
        """Stop background thread if running."""
        if self.async_mode:
            self._running = False
            self._thread.join(timeout=1.0)

    def _process_loop(self) -> None:
        """Background thread for async mode."""
        from .engine import GenerationOutput

        while self._running:
            try:
                pending = self._pending_queue.get(timeout=0.001)
                output = GenerationOutput(
                    request_id=pending.request_id,
                    text=self.tokenizer.decode(pending.tokens),
                    tokens=pending.tokens,
                    finish_reason=pending.finish_reason
                )
                if self.callback:
                    try:
                        self.callback(output)
                    except Exception:
                        pass
                else:
                    with self._results_lock:
                        self._results.append(output)
            except queue.Empty:
                continue

    @property
    def is_running(self) -> bool:
        """Check if background thread is running (async mode only)."""
        return self.async_mode and self._thread.is_alive()


# Backwards compatibility alias
AsyncOutputProcessor = OutputProcessor
