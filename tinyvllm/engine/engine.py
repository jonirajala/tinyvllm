"""LLM Engine - Main orchestrator for continuous batching.

The engine ties everything together:
- Model (LLaMA)
- KVCache (paged attention storage)
- BlockManager (memory allocation)
- Scheduler (batch composition)
- Tokenizer (encode/decode)

Phase 4: Integrates BlockManager for memory-aware scheduling and
block-based KV cache storage.

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

from tinygrad import Tensor, dtypes

from ..model.llama import Llama
from ..core.scheduler import Scheduler
from ..core.sequence import Request, Sequence
from ..core.block_manager import BlockManager
from ..core.kv_cache import KVCache
from ..engine.sampling import SamplingParams, sample_tokens


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
        num_blocks: int = 100,
        block_size: int = 16,
    ):
        """
        Initialize the engine.

        Args:
            model: LLaMA model instance
            tokenizer: Tokenizer with encode/decode methods
            max_batch_size: Maximum sequences per batch
            num_blocks: Number of KV cache blocks (Phase 4)
            block_size: Tokens per block (Phase 4)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.block_size = block_size

        # Phase 4: Create BlockManager for memory allocation
        self.block_manager = BlockManager(
            num_gpus=1,
            blocks_per_gpu=num_blocks,
            block_size=block_size,
        )

        # Phase 4: Create block-based KVCache
        self.kv_cache = KVCache(
            num_layers=model.config.n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            dtype=dtypes.float32,
        )

        # Phase 4: Scheduler with BlockManager integration
        self.scheduler = Scheduler(max_batch_size, block_manager=self.block_manager)
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
        2. Separate prefill vs decode sequences
        3. Process prefill one-by-one, decode in batch
        4. Sample next tokens
        5. Update scheduler
        6. Return any finished outputs

        Phase 4: Uses BlockManager for slot allocation and batched decode.

        Returns:
            List of GenerationOutput for requests that finished this step
        """
        batch = self.scheduler.schedule()
        if not batch.scheduled_seqs:
            return []

        finished_outputs = []
        seq_outputs = {}      # seq_id -> new token
        finished_seqs = []    # seq_ids that finished

        # Separate prefill and decode sequences
        prefill_seqs = []
        decode_seqs = []

        for seq in batch.scheduled_seqs:
            if seq.get_output_len() == 0:
                prefill_seqs.append(seq)
            else:
                decode_seqs.append(seq)

        # Process prefill sequences one-by-one (different lengths)
        for seq in prefill_seqs:
            request = self.requests[seq.request_id]
            input_ids = Tensor([seq.prompt_tokens])
            logits = self.model(
                input_ids,
                start_pos=0,
                kv_cache=self.kv_cache,
                block_manager=self.block_manager,
                seq_id=seq.seq_id
            )
            # Sample next token
            next_token = sample_tokens(
                logits[0, -1, :],
                request.sampling_params,
                seq.get_all_tokens()
            )[0]
            seq_outputs[seq.seq_id] = next_token

            finish_reason = self._check_finished(next_token, seq, request)
            if finish_reason:
                finished_seqs.append(seq.seq_id)
                finished_outputs.append(GenerationOutput(
                    request_id=request.request_id,
                    text=self.tokenizer.decode(seq.output_tokens + [next_token]),
                    tokens=seq.output_tokens + [next_token],
                    finish_reason=finish_reason
                ))

        # Process decode sequences in batch (all have 1 token)
        if decode_seqs:
            tokens_list = []
            seq_ids = []
            start_positions = []

            for seq in decode_seqs:
                last_token = seq.output_tokens[-1]
                tokens_list.append(last_token)
                seq_ids.append(seq.seq_id)
                start_positions.append(seq.get_len() - 1)

            # Batched forward pass
            input_ids = Tensor(tokens_list).reshape(len(decode_seqs), 1)
            logits = self.model.batched_decode(
                input_ids,
                kv_cache=self.kv_cache,
                block_manager=self.block_manager,
                seq_ids=seq_ids,
                start_positions=start_positions
            )

            # Sample tokens for entire batch at once
            batch_logits = logits[:, 0, :]  # [batch, vocab_size]
            params_list = [self.requests[s.request_id].sampling_params for s in decode_seqs]
            seen_tokens_batch = [s.get_all_tokens() for s in decode_seqs]
            next_tokens = sample_tokens(batch_logits, params_list, seen_tokens_batch)

            # Process results
            for i, seq in enumerate(decode_seqs):
                next_token = next_tokens[i]
                request = self.requests[seq.request_id]
                seq_outputs[seq.seq_id] = next_token

                finish_reason = self._check_finished(next_token, seq, request)
                if finish_reason:
                    finished_seqs.append(seq.seq_id)
                    finished_outputs.append(GenerationOutput(
                        request_id=request.request_id,
                        text=self.tokenizer.decode(seq.output_tokens + [next_token]),
                        tokens=seq.output_tokens + [next_token],
                        finish_reason=finish_reason
                    ))

        # Update scheduler with results (also frees blocks for finished seqs)
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
        token: int,
        sequence: Sequence,
        request: Request,
    ) -> Optional[str]:
        """
        Check if sequence should finish.

        Args:
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
