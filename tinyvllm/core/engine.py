"""LLM Engine - orchestrates model, KV cache, scheduler, and tokenizer."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple

from tinygrad import Tensor, dtypes

from .scheduler import Scheduler
from .sequence import Request, Sequence
from .block_manager import BlockManager
from .kv_cache import KVCache
from .sampling import SamplingParams, sample_tokens
from .output_processor import OutputProcessor

if TYPE_CHECKING:
    from ..model.llama import Llama


@dataclass
class GenerationOutput:
    """Output from a completed generation."""
    request_id: int
    text: str
    tokens: List[int]
    finish_reason: str  # 'eos', 'length', or 'abort'


class LLMEngine:
    """Main engine for LLM inference with continuous batching."""

    def __init__(
        self,
        model: "Llama",
        tokenizer: Any,
        max_batch_size: int = 8,
        num_blocks: int = 100,
        block_size: int = 16,
        num_scheduler_steps: int = 1,
        async_output: bool = False,
        output_callback: Optional[Callable[[GenerationOutput], None]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.num_scheduler_steps = num_scheduler_steps

        self.block_manager = BlockManager(
            num_gpus=1,
            blocks_per_gpu=num_blocks,
            block_size=block_size,
        )
        self.kv_cache = KVCache(
            num_layers=model.config.n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.head_dim,
            dtype=model.config.dtype,
        )
        self.scheduler = Scheduler(max_batch_size, block_manager=self.block_manager)
        self.next_request_id = 0
        self.requests: Dict[int, Request] = {}
        self._output_processor = OutputProcessor(
            tokenizer=self.tokenizer,
            async_mode=async_output,
            callback=output_callback,
        )

        # JIT requires fixed shapes - cache separate function per batch size
        self._jit_decode_fns: Dict[int, object] = {}
        self._jit_max_blocks = 64

        # Pre-allocated buffers to avoid allocation each step
        self._bt_buffer_data = [0] * (max_batch_size * self._jit_max_blocks)
        self._ctx_buffer_data = [0] * max_batch_size

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[int] = None,
    ) -> int:
        """Add a generation request. Returns request_id."""
        if request_id is None:
            request_id = self.next_request_id
            self.next_request_id += 1
        elif request_id in self.requests: raise ValueError(f"Request ID {request_id} already exists")
        elif request_id >= self.next_request_id: self.next_request_id = request_id + 1

        if sampling_params is None: sampling_params = SamplingParams()

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
        """Run one generation step. Returns finished outputs."""
        batch = self.scheduler.schedule()
        if not batch.scheduled_seqs: return []

        finished_seqs = []
        prefill_seqs, decode_seqs = [], []
        for s in batch.scheduled_seqs:
            (prefill_seqs if s.get_output_len() == 0 else decode_seqs).append(s)

        # Prefill: process one-by-one (variable lengths)
        for seq in prefill_seqs:
            request = self.requests[seq.request_id]
            logits = self.model.prefill(
                tokens=Tensor([seq.prompt_tokens]),
                kv_cache=self.kv_cache,
                block_manager=self.block_manager,
                seq_id=seq.seq_id,
            )
            next_token = sample_tokens(
                logits=logits[:, -1, :],
                params=[request.sampling_params],
                seen_tokens=[seq.get_all_tokens()]
            )[0]
            seq.append_token(next_token)

            if finish_reason := self._check_finished(seq, request):
                finished_seqs.append(seq.seq_id)
                self._output_processor.submit(
                    request_id=request.request_id,
                    tokens=seq.output_tokens,
                    finish_reason=finish_reason
                )

        # Decode: batched multi-step loop
        for _ in range(self.num_scheduler_steps):
            if not decode_seqs: break

            batch_size = len(decode_seqs)
            seq_ids = [s.seq_id for s in decode_seqs]
            start_positions = [s.get_len() - 1 for s in decode_seqs]
            bt_tensor, ctx_tensor = self._prepare_batch_tensors(seq_ids, start_positions)

            if batch_size not in self._jit_decode_fns:
                self._jit_decode_fns[batch_size] = self.model.create_jit_decode(block_size=self.block_size)
            jit_fn = self._jit_decode_fns[batch_size]

            logits = self.model.decode(
                tokens=Tensor([s.output_tokens[-1] for s in decode_seqs]).reshape(batch_size, 1),
                kv_cache=self.kv_cache,
                block_manager=self.block_manager,
                seq_ids=seq_ids,
                start_positions=start_positions,
                block_tables_tensor=bt_tensor,
                context_lens_tensor=ctx_tensor,
                jit_fn=jit_fn,
                max_blocks=self._jit_max_blocks,
            )

            next_tokens = sample_tokens(
                logits=logits[:, 0, :],
                params=[self.requests[s.request_id].sampling_params for s in decode_seqs],
                seen_tokens=[s.get_all_tokens() for s in decode_seqs],
            )

            still_active = []
            for seq, next_token in zip(decode_seqs, next_tokens):
                seq.append_token(next_token)
                if finish_reason := self._check_finished(seq, self.requests[seq.request_id]):
                    finished_seqs.append(seq.seq_id)
                    self._output_processor.submit(
                        request_id=seq.request_id,
                        tokens=seq.output_tokens,
                        finish_reason=finish_reason
                    )
                else:
                    still_active.append(seq)
            decode_seqs = still_active

        self.scheduler.update(finished_seqs)
        return self._output_processor.poll_all()


    def run(self) -> Iterator[GenerationOutput]:
        """Run until all requests complete. Yields outputs as they finish."""
        while self.scheduler.has_unfinished():
            yield from self.step()
        yield from self._output_processor.drain()

    def has_unfinished(self) -> bool:
        """Check if there are pending requests."""
        return self.scheduler.has_unfinished()

    def abort_request(self, request_id: int) -> bool:
        """Abort a request. Returns True if found and aborted."""
        return self.scheduler.abort_request(request_id)

    def poll_outputs(self) -> List[GenerationOutput]:
        """Poll for completed async outputs."""
        return self._output_processor.poll_all()

    def shutdown(self) -> None:
        """Clean shutdown of engine resources."""
        self._output_processor.shutdown()

    def _check_finished(self, sequence: Sequence, request: Request) -> Optional[str]:
        """Check if sequence should finish. Returns finish reason or None."""
        if sequence.output_tokens[-1] == self.tokenizer.eos_id: return 'eos'
        if sequence.get_output_len() >= request.sampling_params.max_tokens: return 'length'
        return None

    def _prepare_batch_tensors(
        self,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tuple[Tensor, Tensor]:
        """Build block_tables and context_lens tensors for batched decode."""
        batch_size = len(seq_ids)
        max_blocks = self._jit_max_blocks
        bt_data = self._bt_buffer_data
        ctx_data = self._ctx_buffer_data

        for i, (sid, start_pos) in enumerate(zip(seq_ids, start_positions)):
            bt = self.block_manager.get_block_table(sid)
            offset = i * max_blocks
            for j in range(max_blocks):
                bt_data[offset + j] = bt[j] if j < len(bt) else 0
            ctx_data[i] = start_pos + 1

        bt_tensor = Tensor(bt_data[:batch_size * max_blocks], dtype=dtypes.int32).reshape(batch_size, max_blocks)
        ctx_tensor = Tensor(ctx_data[:batch_size], dtype=dtypes.int32)
        return bt_tensor, ctx_tensor


def generate_batch(
    engine: LLMEngine,
    prompts: List[str],
    sampling_params: Optional[SamplingParams] = None,
) -> List[str]:
    """Generate completions for multiple prompts. Returns texts in input order."""
    request_ids = [engine.add_request(p, sampling_params) for p in prompts]
    results = {out.request_id: out.text for out in engine.run()}
    return [results[rid] for rid in request_ids]
