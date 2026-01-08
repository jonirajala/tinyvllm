"""Sequence and Request state management for continuous batching."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import time


class SequenceStatus(Enum):
    """Lifecycle states for a sequence."""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """Tracks state for a single generation sequence."""
    seq_id: int
    request_id: int
    prompt_tokens: List[int]
    output_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    def get_all_tokens(self) -> List[int]:
        """Get all tokens (prompt + generated)."""
        return self.prompt_tokens + self.output_tokens

    def get_len(self) -> int:
        """Get total sequence length."""
        return len(self.prompt_tokens) + len(self.output_tokens)

    def get_prompt_len(self) -> int:
        """Get prompt length."""
        return len(self.prompt_tokens)

    def get_output_len(self) -> int:
        """Get number of generated tokens."""
        return len(self.output_tokens)

    def append_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.output_tokens.append(token_id)

    def is_finished(self) -> bool:
        """Check if sequence has finished."""
        return self.status == SequenceStatus.FINISHED


@dataclass
class Request:
    """User-facing request with metadata."""
    request_id: int
    prompt: str
    prompt_tokens: List[int]
    sampling_params: 'SamplingParams'
    arrival_time: float = field(default_factory=time.time)
    sequence: Optional[Sequence] = None

    def create_sequence(self, seq_id: int) -> Sequence:
        """Create a Sequence for this request when scheduled."""
        self.sequence = Sequence(
            seq_id=seq_id,
            request_id=self.request_id,
            prompt_tokens=self.prompt_tokens,
            status=SequenceStatus.RUNNING,
        )
        return self.sequence


@dataclass
class SchedulerOutput:
    """Output from scheduler for one step."""
    scheduled_seqs: List[Sequence] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
