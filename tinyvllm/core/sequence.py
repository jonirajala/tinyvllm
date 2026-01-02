"""Sequence and Request state management for continuous batching.

This module defines the core data structures for tracking requests through
the generation pipeline:

- SequenceStatus: Enum for sequence lifecycle states
- Sequence: Tracks a single generation (tokens, position, KV cache mapping)
- Request: User-facing request with callback support

Lifecycle:
    1. User calls engine.add_request(prompt, params)
    2. Request created with status=WAITING
    3. Scheduler picks request, creates Sequence, status=RUNNING
    4. Engine generates tokens, appending to Sequence
    5. On EOS or max_tokens, status=FINISHED
    6. Callback fired, KV cache freed
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
import time


class SequenceStatus(Enum):
    """Lifecycle states for a sequence."""
    WAITING = auto()    # In queue, not yet scheduled
    RUNNING = auto()    # Currently generating tokens
    FINISHED = auto()   # Generation complete (EOS or max_tokens)


@dataclass
class Sequence:
    """
    Tracks state for a single generation sequence.

    A Sequence represents one prompt being processed. It tracks:
    - The tokens (prompt + generated)
    - Position in generation
    - Mapping to KV cache

    Attributes:
        seq_id: Unique identifier (used as key in KVCache)
        request_id: ID of the Request that owns this sequence
        prompt_tokens: Original prompt token IDs
        output_tokens: Generated token IDs (appended during generation)
        status: Current lifecycle state

    Example:
        seq = Sequence(seq_id=0, request_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(4)
        seq.append_token(5)
        print(seq.get_all_tokens())  # [1, 2, 3, 4, 5]
        print(seq.get_len())  # 5
    """
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
        """Append a newly generated token."""
        self.output_tokens.append(token_id)

    def is_finished(self) -> bool:
        """Check if sequence has finished generating."""
        return self.status == SequenceStatus.FINISHED


@dataclass
class Request:
    """
    User-facing request with metadata.

    Attributes:
        request_id: Unique identifier for this request
        prompt: Original prompt string
        prompt_tokens: Tokenized prompt
        sampling_params: Temperature, top_p, etc.
        arrival_time: When request was added (Phase 5: FCFS scheduling)
        sequence: The Sequence tracking generation state (None until scheduled)
    """
    request_id: int
    prompt: str
    prompt_tokens: List[int]
    sampling_params: 'SamplingParams'  # Forward reference
    arrival_time: float = field(default_factory=time.time)  # Phase 5: FCFS priority
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
    """
    Output from scheduler for one step.

    Contains the sequences to process in this batch and any
    memory operations needed.

    Attributes:
        scheduled_seqs: Sequences to run in this batch
        num_prefill_tokens: Phase 5 - Total tokens needing prefill (new prompts)
        num_decode_tokens: Phase 5 - Total tokens needing decode (1 per running seq)
    """
    scheduled_seqs: List[Sequence] = field(default_factory=list)
    num_prefill_tokens: int = 0  # Phase 5: Batch statistics for chunked prefill
    num_decode_tokens: int = 0   # Phase 5: Batch statistics for scheduling
    # Phase 4: Memory swapping (commented until implemented)
    # blocks_to_swap_in: Dict[int, int] = field(default_factory=dict)
    # blocks_to_swap_out: Dict[int, int] = field(default_factory=dict)
