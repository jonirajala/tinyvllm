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
from typing import Dict, Iterator, List, Optional, Tuple

from tinygrad import Tensor, dtypes

from ..model.llama import Llama
from ..core.scheduler import Scheduler
from ..core.sequence import Request, Sequence
from ..core.block_manager import BlockManager
from ..core.kv_cache import KVCache
from ..engine.sampling import SamplingParams, sample_tokens
from ..engine.jit_decode import JitDecoder


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
        use_jit: bool = False,
        num_scheduler_steps: int = 1,
    ):
        """
        Initialize the engine.

        Args:
            model: LLaMA model instance
            tokenizer: Tokenizer with encode/decode methods
            max_batch_size: Maximum sequences per batch
            num_blocks: Number of KV cache blocks (Phase 4)
            block_size: Tokens per block (Phase 4)
            use_jit: Enable JIT compilation for decode (Phase 7.1)
            num_scheduler_steps: Decode steps per scheduler cycle (Phase 7.3)
                1 = best latency (default), 4-8 = better throughput
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.use_jit = use_jit
        self.num_scheduler_steps = num_scheduler_steps

        # Phase 4: Create BlockManager for memory allocation
        self.block_manager = BlockManager(
            num_gpus=1,
            blocks_per_gpu=num_blocks,
            block_size=block_size,
        )

        # Phase 4: Create block-based KVCache (uses model dtype for FP16 support)
        self.kv_cache = KVCache(
            num_layers=model.config.n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            dtype=model.config.dtype,
        )

        # Phase 4: Scheduler with BlockManager integration
        self.scheduler = Scheduler(max_batch_size, block_manager=self.block_manager)
        self.next_request_id = 0

        # Mappings
        self.requests: Dict[int, Request] = {}  # request_id -> Request

        # Phase 7.1: JIT decoder (optional)
        self.jit_decoder: Optional[JitDecoder] = None
        if use_jit:
            # max_context_len: typical prompt (50-100) + max_tokens (50-100)
            # Default 256 is reasonable for most use cases
            max_context = min(256, num_blocks * block_size)
            self.jit_decoder = JitDecoder(
                model=model,
                kv_cache=self.kv_cache,
                max_batch_size=max_batch_size,
                max_context_len=max_context,
            )



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
        Run one or more generation steps.

        Phase 7.3: Multi-step scheduling - runs num_scheduler_steps decode
        iterations before returning, amortizing scheduler overhead.

        This is the core engine loop iteration:
        1. Get batch from scheduler (once per step() call)
        2. Process prefill sequences one-by-one
        3. Run multi-step decode loop
        4. Update scheduler (once at end)
        5. Return finished outputs

        Returns:
            List of GenerationOutput for requests that finished this step
        """
        batch = self.scheduler.schedule()
        if not batch.scheduled_seqs: return []

        finished_outputs, finished_seqs = [], []

        # Separate prefill and decode sequences
        prefill_seqs = [s for s in batch.scheduled_seqs if s.get_output_len() == 0]
        decode_seqs = [s for s in batch.scheduled_seqs if s.get_output_len() > 0]

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
                logits[:, -1, :],  # [1, vocab_size]
                [request.sampling_params],
                [seq.get_all_tokens()]
            )[0]

            seq.append_token(next_token)

            finish_reason = self._check_finished(next_token, seq, request)
            if finish_reason:
                finished_seqs.append(seq.seq_id)
                finished_outputs.append(GenerationOutput(
                    request_id=request.request_id,
                    text=self.tokenizer.decode(seq.output_tokens),
                    tokens=seq.output_tokens.copy(),
                    finish_reason=finish_reason
                ))
            # Sequence stays in self.running - will be scheduled for decode on next step()

        # Multi-step decode loop (Phase 7.3) - only for sequences already in decode phase
        for step_idx in range(self.num_scheduler_steps):
            if not decode_seqs: break

            # Prepare batch
            tokens_list = [s.output_tokens[-1] for s in decode_seqs]
            seq_ids = [s.seq_id for s in decode_seqs]
            start_positions = [s.get_len() - 1 for s in decode_seqs]

            # Phase 7.4: Build block_tables and context_lens tensors once
            # (eliminates 32x per-layer Python list → Tensor conversions)
            bt_tensor, ctx_tensor = self._prepare_batch_tensors(seq_ids, start_positions)

            # Batched forward pass - use JIT decoder if enabled
            if self.use_jit and self.jit_decoder is not None:
                logits = self.jit_decoder.decode(
                    block_manager=self.block_manager,
                    tokens_list=tokens_list,
                    seq_ids=seq_ids,
                    start_positions=start_positions
                )
            else:
                input_ids = Tensor(tokens_list).reshape(len(decode_seqs), 1)
                logits = self.model.batched_decode(
                    input_ids,
                    kv_cache=self.kv_cache,
                    block_manager=self.block_manager,
                    seq_ids=seq_ids,
                    start_positions=start_positions,
                    block_tables_tensor=bt_tensor,
                    context_lens_tensor=ctx_tensor,
                )

            # Sample tokens for entire batch at once
            batch_logits = logits[:, 0, :]  # [batch, vocab_size]
            params_list = [self.requests[s.request_id].sampling_params for s in decode_seqs]
            seen_tokens_batch = [s.get_all_tokens() for s in decode_seqs]
            next_tokens = sample_tokens(batch_logits, params_list, seen_tokens_batch)

            # Process results
            still_active = []
            for seq, next_token in zip(decode_seqs, next_tokens):
                seq.append_token(next_token)
                if finish_reason := self._check_finished(next_token, seq, self.requests[seq.request_id]):
                    finished_seqs.append(seq.seq_id)
                    finished_outputs.append(GenerationOutput(
                        request_id=seq.request_id,
                        text=self.tokenizer.decode(seq.output_tokens),
                        tokens=seq.output_tokens.copy(),
                        finish_reason=finish_reason
                    ))
                else:
                    still_active.append(seq)
            decode_seqs = still_active

        # Update scheduler (tokens already appended, just handle finished)
        self.scheduler.update(finished_seqs)

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

        Note: Called AFTER token is appended to sequence (Phase 7.3 change).

        Args:
            token: Just-generated token (already appended)
            sequence: The sequence (with token already appended)
            request: The request

        Returns:
            Finish reason ('eos', 'length') or None if not finished
        """
        if token == self.tokenizer.eos_id:
            return 'eos'
        # Token already appended, so check get_output_len() directly
        if sequence.get_output_len() >= request.sampling_params.max_tokens:
            return 'length'
        return None

    def _prepare_batch_tensors(
        self,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tuple[Tensor, Tensor]:
        """Build block_tables and context_lens tensors for batched decode.

        Phase 7.4: Build these once per step instead of per-layer.
        Eliminates 32x redundant Python list → Tensor conversions.

        Args:
            seq_ids: List of sequence IDs in the batch
            start_positions: Start position for each sequence

        Returns:
            block_tables_tensor: [batch, max_blocks] int32
            context_lens_tensor: [batch] int32
        """
        batch_size = len(seq_ids)

        # Determine max blocks needed (use actual block table sizes)
        block_tables_raw = [self.block_manager.get_block_table(sid) for sid in seq_ids]
        max_blocks = max(len(bt) for bt in block_tables_raw) if block_tables_raw else 1

        # Pad block tables and build context lens
        block_tables_padded = []
        context_lens = []

        for bt, start_pos in zip(block_tables_raw, start_positions):
            block_tables_padded.append(bt + [0] * (max_blocks - len(bt)))
            context_lens.append(start_pos + 1)

        # Flatten and create tensors
        bt_flat = [b for row in block_tables_padded for b in row]
        bt_tensor = Tensor(bt_flat, dtype=dtypes.int32).reshape(batch_size, max_blocks)
        ctx_tensor = Tensor(context_lens, dtype=dtypes.int32)

        return bt_tensor, ctx_tensor


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
