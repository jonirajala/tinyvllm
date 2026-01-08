"""Output Processor - Sync or async detokenization."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from .engine import GenerationOutput


@dataclass
class PendingOutput:
    """Output waiting for detokenization."""
    request_id: int
    tokens: List[int]
    finish_reason: str


class OutputProcessor:
    """Handles detokenization. Sync mode decodes immediately, async uses background thread."""

    def __init__(
        self,
        tokenizer: Any,
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
            self._thread = threading.Thread(
                target=self._process_loop,
                name="OutputProcessor",
                daemon=True
            )
            self._thread.start()

    def submit(self, request_id: int, tokens: List[int], finish_reason: str) -> None:
        """Submit completed sequence for processing."""
        if self.async_mode:
            self._pending_queue.put(PendingOutput(
                request_id=request_id,
                tokens=tokens.copy(),
                finish_reason=finish_reason
            ))
        else:
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
        """Get and clear all available results."""
        if self.async_mode:
            with self._results_lock:
                outputs = self._results.copy()
                self._results.clear()
                return outputs
        outputs = self._results.copy()
        self._results.clear()
        return outputs

    def drain(self) -> List[GenerationOutput]:
        """Wait for pending work and return all results."""
        if self.async_mode:
            self._pending_queue.join()
        return self.poll_all()

    def shutdown(self) -> None:
        """Stop background thread if running."""
        if self.async_mode and self._thread.is_alive():
            self._pending_queue.put(None)
            self._thread.join(timeout=1.0)

    def _process_loop(self) -> None:
        """Background thread loop for async mode."""
        from .engine import GenerationOutput

        while True:
            pending = self._pending_queue.get()
            if pending is None:
                break
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
            self._pending_queue.task_done()

    @property
    def is_running(self) -> bool:
        """Check if background thread is running."""
        return self.async_mode and self._thread.is_alive()
