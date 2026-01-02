"""LLM Engine - Main orchestrator for continuous batching.

The engine ties everything together:
- Model (LLaMA)
- KVCache (paged attention storage)
- Scheduler (batch composition)
- Tokenizer (encode/decode)

It runs the main generation loop:
    while has_requests:
        batch = scheduler.schedule()
        outputs = model.forward(batch)
        scheduler.update(outputs)
        yield finished_results

Example usage:
    engine = LLMEngine(model, tokenizer)

    # Add requests (non-blocking)
    engine.add_request("Write a poem", request_id=0)
    engine.add_request("Hello world", request_id=1)

    # Process until done
    for output in engine.run():
        print(f"Request {output.request_id}: {output.text}")

Or step-by-step:
    engine.add_request("Hello", request_id=0)

    while engine.has_unfinished():
        outputs = engine.step()
        for out in outputs:
            print(out.text)
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from tinygrad import Tensor

from ..model.llama import Llama
from ..core.scheduler import Scheduler
from ..core.sequence import Request, Sequence
from ..engine.sampling import SamplingParams, sample_token


@dataclass
class GenerationOutput:
    """
    Output from a completed generation.

    Attributes:
        request_id: Which request this is for
        text: Generated text
        tokens: Generated token IDs
        finish_reason: Why generation stopped ('eos', 'length', 'abort')
    """
    request_id: int
    text: str
    tokens: List[int]
    finish_reason: str


class LLMEngine:
    """
    Main engine for LLM inference with continuous batching.

    The engine manages the full lifecycle:
    1. Accept requests via add_request()
    2. Schedule batches via internal Scheduler
    3. Run model forward pass
    4. Sample tokens
    5. Return completed generations

    Attributes:
        model: The LLaMA model
        tokenizer: Tokenizer for encode/decode
        scheduler: Scheduler for batch management
        kv_cache: Shared KVCache for all sequences
        next_request_id: Counter for request IDs

    Example:
        engine = LLMEngine(model, tokenizer, max_batch_size=8)

        engine.add_request("Hello", SamplingParams(max_tokens=50))
        engine.add_request("Write code", SamplingParams(temperature=0.7))

        for output in engine.run():
            print(output.text)
    """

    def __init__(
        self,
        model: Llama,
        tokenizer,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
    ):
        """
        Initialize the engine.

        Args:
            model: LLaMA model instance
            tokenizer: Tokenizer with encode/decode methods
            max_batch_size: Maximum sequences per batch
            max_seq_len: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.scheduler = Scheduler(max_batch_size)
        self.kv_cache = model.create_kv_cache()
        self.next_request_id = 0

        # Mappings
        self.requests: Dict[int, Request] = {}  # request_id -> Request



    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[int] = None,
    ) -> int:
        """
        Add a new generation request.

        Args:
            prompt: Input prompt string
            sampling_params: Sampling parameters (default: greedy)
            request_id: Optional custom ID (auto-assigned if None)

        Returns:
            The request ID
        """
        if request_id is None:
            request_id = self.next_request_id
            self.next_request_id += 1

        if sampling_params is None:
            sampling_params = SamplingParams()

        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=self.tokenizer.encode(prompt),
            sampling_params=sampling_params,
        )

        self.requests[request_id] = request
        self.scheduler.add_request(request)
        return request_id

    def step(self) -> List[GenerationOutput]:
        """
        Run one generation step.

        This is the core engine loop iteration:
        1. Get batch from scheduler
        2. Prepare inputs (handle prefill vs decode)
        3. Run model forward
        4. Sample next tokens
        5. Update scheduler
        6. Return any finished outputs

        Returns:
            List of GenerationOutput for requests that finished this step
        """
        batch = self.scheduler.schedule()
        if not batch.scheduled_seqs:
            return []

        finished_outputs = []
        seq_outputs = {}      # seq_id -> new token
        finished_seqs = []    # seq_ids that finished

        for seq in batch.scheduled_seqs:
            request = self.requests[seq.request_id]

            # Determine if prefill or decode
            if seq.get_output_len() == 0:
                # PREFILL: first time, process full prompt
                self.kv_cache.allocate_sequence(seq.seq_id)
                input_ids = Tensor([seq.prompt_tokens])  # [1, prompt_len]
                start_pos = 0
            else:
                # DECODE: continuing, process just last token
                last_token = seq.output_tokens[-1]
                input_ids = Tensor([[last_token]])  # [1, 1]
                start_pos = seq.get_len() - 1

            # Run model
            logits = self.model(input_ids, start_pos=start_pos,
                               kv_cache=self.kv_cache, seq_id=seq.seq_id)

            # Sample next token
            next_token = sample_token(
                logits[0, -1, :],
                request.sampling_params,
                seq.get_all_tokens()
            )

            seq_outputs[seq.seq_id] = next_token

            # Check if finished
            finish_reason = self._check_finished(seq.seq_id, next_token, seq, request)
            if finish_reason:
                finished_seqs.append(seq.seq_id)
                finished_outputs.append(GenerationOutput(
                    request_id=request.request_id,
                    text=self.tokenizer.decode(seq.output_tokens + [next_token]),
                    tokens=seq.output_tokens + [next_token],
                    finish_reason=finish_reason
                ))
                self.kv_cache.free_sequence(seq.seq_id)

        # Update scheduler with results
        self.scheduler.update(seq_outputs, finished_seqs)

        return finished_outputs


    def run(self) -> Iterator[GenerationOutput]:
        """
        Run until all requests complete.

        Yields:
            GenerationOutput for each completed request
        """
        while self.scheduler.has_unfinished():
            yield from self.step()

    def has_unfinished(self) -> bool:
        """
        Check if there are pending requests.

        Returns:
            True if waiting or running requests exist
        """
        return self.scheduler.has_unfinished()

    def abort_request(self, request_id: int) -> bool:
        """
        Abort a request.

        Args:
            request_id: ID of request to abort

        Returns:
            True if request was found and aborted
        """
        return self.scheduler.abort_request(request_id)

    def _check_finished(
        self,
        seq_id: int,
        token: int,
        sequence: Sequence,
        request: Request,
    ) -> Optional[str]:
        """
        Check if sequence should finish.

        Args:
            seq_id: Sequence ID
            token: Just-generated token
            sequence: The sequence
            request: The request

        Returns:
            Finish reason ('eos', 'length') or None if not finished
        """
        if token == self.tokenizer.eos_id:
            return 'eos'
        if sequence.get_output_len() + 1 >= request.sampling_params.max_tokens:
            return 'length'
        return None


def generate_batch(
    engine: LLMEngine,
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
) -> List[str]:
    """
    Generate completions for multiple prompts.

    Convenience function for batch generation.

    Args:
        engine: LLMEngine instance
        prompts: List of prompts
        sampling_params: Shared sampling params

    Returns:
        List of generated texts (in same order as prompts)
    """
    request_ids = [engine.add_request(p, sampling_params) for p in prompts]
    results = {out.request_id: out.text for out in engine.run()}
    return [results[rid] for rid in request_ids]
