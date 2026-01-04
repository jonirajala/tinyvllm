⏺ Phase 1: Foundation

  Goal

  Generate text with a single LLaMA model, one request at a time.

  ---
  Files

  tinyvllm/
  ├── model/
  │   ├── llama.py        # LLaMA architecture
  │   ├── tokenizer.py    # Text ↔ tokens
  │   └── weights.py      # Load safetensors
  ├── engine/
  │   ├── generate.py     # Generation loop
  │   └── sampling.py     # Token selection
  └── main.py             # CLI

  ---
  Components

  1.1 Token Generation Loop

  The autoregressive process:
  1. Convert prompt to tokens
  2. Prefill: Run entire prompt through model once, cache all K,V
  3. Decode: Generate one token at a time, reusing cached K,V
  4. Stop when EOS token or max length reached
  5. Convert tokens back to text

  Key concept - KV Cache: Without it, you recompute attention for ALL previous tokens at each step (O(n²)). With it, you only compute for the new token (O(n)).

  1.2 Model Loading

  - Load config (dimensions, layers, heads)
  - Create LLaMA architecture:
    - Embedding layer (token → vector)
    - N transformer blocks (attention + FFN + norms)
    - Output layer (vector → vocabulary probabilities)
  - Load weights from safetensors file
  - Apply RoPE (rotary position emb eddings)

  1.3 Sampling

  Convert model output (logits) to next token:
  - Temperature: Higher = more random
  - Top-k: Only consider k most likely tokens
  - Top-p (nucleus): Only consider tokens until cumulative probability > p
  - Greedy: Just pick the highest probability

  ---
  After Phase 1 You Can

  ✅ Load any LLaMA model
  ✅ Generate text with various sampling strategies
  ✅ Stream tokens as generated
  ✅ Basic CLI usage

  Limitations

  ❌ One request at a time
  ❌ Pre-allocates maximum memory (wasteful)
  ❌ Can't serve multiple users
  ❌ No API server

  ---
  Phase 2: Paged Attention

  Goal

  Replace wasteful pre-allocated KV cache with on-demand block allocation.

  ---
  Files

  tinyvllm/
  ├── model/
  │   ├── llama.py
  │   ├── tokenizer.py
  │   ├── weights.py
  │   └── paged_attention.py   # NEW
  ├── core/                     # NEW
  │   ├── block_manager.py     # NEW
  │   ├── kv_cache.py          # NEW
  │   └── sequence.py          # NEW
  ├── engine/
  │   ├── generate.py          # Modified
  │   └── sampling.py
  └── main.py

  ---
  Components

  2.1 Block Manager

  Memory allocator for KV cache blocks (like malloc/free):
  - Maintains pool of free blocks
  - allocate() → returns a free block ID
  - free(block_id) → returns block to pool
  - Tracks how many blocks available

  2.2 Paged KV Cache

  Stores K,V in non-contiguous blocks:
  - Physical storage: Large GPU tensor holding all blocks
  - Each block holds K,V for N tokens (e.g., 16)
  - Blocks allocated on-demand as sequence grows
  - Blocks freed when sequence completes

  Visual:
  GPU Memory (1000 blocks × 16 tokens each):
  ┌────────┬────────┬────────┬────────┬─────┐
  │Block 0 │Block 1 │Block 2 │Block 3 │ ... │
  └────────┴────────┴────────┴────────┴─────┘

  Sequence A (45 tokens): blocks [5, 12, 77]
  Sequence B (20 tokens): blocks [3, 88]

  No wasted memory!

  2.3 Sequence State

  Tracks each request:
  - Prompt tokens
  - Generated tokens
  - Block table (which physical blocks hold this sequence's K,V)
  - Methods to append token, allocate new block if needed, free all blocks

  2.4 Paged Attention

  Modified attention that reads from scattered blocks:
  - Input: query, block table, sequence length
  - Gathers K,V from non-contiguous blocks using block table
  - Computes standard attention
  - Returns output

  Difference from normal attention:
  - Normal: K,V stored contiguously [token0, token1, token2, ...]
  - Paged: K,V scattered across blocks, gathered via block table lookup

  2.5 Modified Generation Loop

  Same as Phase 1, but:
  - Creates Sequence object with block allocation
  - Uses paged attention instead of normal
  - Allocates new blocks as tokens generated
  - Frees blocks when done

  ---
  After Phase 2 You Can

  ✅ Everything from Phase 1
  ✅ Memory-efficient KV cache (near-zero waste)
  ✅ Support much longer sequences
  ✅ Foundation ready for multi-user

  Comparison

  |                 | Phase 1 (Naive)  | Phase 2 (Paged)    |
  |-----------------|------------------|--------------------|
  | Memory per seq  | Pre-allocate max | Allocate as needed |
  | Waste           | ~90%             | ~0%                |
  | Max sequence    | Limited          | Much longer        |
  | Concurrent seqs | 1                | 1 (ready for more) |

  ---
  Still Missing (Phase 3)

  ❌ Multiple concurrent requests
  ❌ Continuous batching
  ❌ Dynamic scheduling

  ---
  Summary

  | Phase | What You Build      | Lines | Key Concept                         |
  |-------|---------------------|-------|-------------------------------------|
  | 1     | Basic inference     | ~400  | KV cache, autoregressive loop       |
  | 2     | Paged attention     | ~400  | Block allocation, memory efficiency |
  | 3     | Continuous batching | ~300  | Multi-request scheduling            |

  Phase 2 is the core innovation - it's what makes vLLM special. Understanding block-based memory management for KV cache is understanding vLLM.




## Whats already in tinygrad


What You Can Reuse for tinyvLLM

  Use directly (no changes needed)

  - SimpleTokenizer - Complete BPE tokenizer
  - Transformer.from_gguf() - Load any GGUF model
  - apply_rope() - RoPE implementation
  - TransformerBlock - Full attention + FFN

  Modify for Phase 2

  - _attention() method - Currently uses naive KV cache, replace with paged attention
  - cache_kv - Replace with PagedKVCache

  ---
  Phase 1 → Already Done!

  You can run tinygrad's LLM right now:

  # Run LLaMA 3.2 1B
  echo "Hello" | python -m tinygrad.apps.llm

  # Or with server
  python -m tinygrad.apps.llm --serve

  ---
  For tinyvLLM, You Only Need to Add

  Phase 2: Paged Attention

  NEW files to create:
  ├── core/
  │   ├── block_manager.py    # Block allocation
  │   ├── kv_cache.py         # Paged storage
  │   └── sequence.py         # Request tracking

  MODIFY:
  └── TransformerBlock._attention()  # Use paged attention

  Phase 3: Continuous Batching

  NEW files:
  ├── core/
  │   └── scheduler.py        # Batch multiple requests

  MODIFY:
  └── Transformer.generate()  # Handle multiple sequences

  ---
  Summary

  | Phase   | tinygrad Status | Your Work                        |
  |---------|-----------------|----------------------------------|
  | Phase 1 | ✅ 100% done    | Just study llm.py                |
  | Phase 2 | ❌ Not done     | Add paged attention (~400 lines) |
  | Phase 3 | ❌ Not done     | Add scheduler (~300 lines)       |

  You can skip Phase 1 entirely and start directly on Phase 2 using tinygrad's existing code as your foundation!

  ---
  Quick Start Path

  1. Study tinygrad/apps/llm.py (especially _attention() method)
  2. Create PagedKVCache class to replace self.cache_kv
  3. Create BlockManager for allocation
  4. Modify _attention() to use paged attention
  5. Add Scheduler for continuous batching

  tinygrad gives you the model, tokenizer, and basic generation for free. You just add the vLLM innovations on top.





Two types of request:
1. Prefill
    - First forward pass where the whole prompt goes in
    - Compute bound
2. Decode
    - Only the newest generated token is forward passed
    - Memory-bandwidth-bound

---

## Full Implementation Roadmap

### Phase 1: Foundation (~400 lines)

1.1 Basic Inference Loop
- Token generation loop (greedy)
- KV cache (naive, pre-allocated)
- Sampling (temperature, top-p, top-k)
- Stop conditions (EOS, max length)

1.2 Model Loading
- Load LLaMA weights (safetensors)
- LLaMA architecture (attention, FFN, RMSNorm)
- RoPE positional embeddings
- Tokenizer (use tiktoken or sentencepiece)

1.3 Basic API
- Simple Python interface
- generate(prompt) → string
- generate_stream(prompt) → iterator

---

### Phase 2: Paged Attention (~400 lines)

2.1 Block Manager
- Block allocator (malloc/free for KV blocks)
- Free list management
- Block reference counting
- Sequence → block table mapping

2.2 Paged KV Cache
- Physical block storage (GPU tensor)
- Logical → physical mapping
- Allocate on demand (as tokens generated)
- Free blocks when sequence done

2.3 Paged Attention Kernel
- Gather K,V from scattered blocks
- Compute attention with block tables
- Handle variable sequence lengths
- (Optional) CUDA kernel for speed

---

### Phase 3: Continuous Batching (~300 lines)

3.1 Request Queue
- Add request (prompt + params)
- Request states (waiting, running, finished)
- Request metadata (arrival time, tokens generated)
- Callbacks for completion

3.2 Scheduler
- Select requests for next batch
- Dynamic batch composition
- Add new requests when slots free
- Remove finished requests immediately

3.3 Batch Execution
- Prepare batched inputs
- Handle mixed prefill + decode
- Update KV cache per sequence
- Dispatch results to requests

---

### Phase 4: Block-based KVCache (~200 lines)

Portable tinygrad-native solution. BlockManager already exists with tests.

4.1 KV Cache with Block Tensors
- Replace list-based KV cache with pre-allocated block tensors
- Each block is a separate realized tensor (tinygrad compatible)
- Tensor.stack() for gathering blocks during attention
- See docs/phase4_kv_optimization.md for details

4.2 Integrate BlockManager
- Connect BlockManager to Scheduler (can_allocate checks)
- Connect BlockManager to KVCache (block-based storage)
- Scheduler calls block_manager.allocate_sequence() on prefill
- Scheduler calls block_manager.free_sequence() on finish

4.3 Batched Forward Pass
- Process multiple sequences in single forward pass
- Pad sequences to same length or use attention mask
- Real throughput improvement from GPU parallelism

4.4 Memory Tracking
- Track blocks per sequence
- Memory budget enforcement
- OOM prevention via can_allocate checks

---

### Phase 5: Custom Kernels (~200 lines)

Metal kernel for Apple Silicon. CUDA kernel deferred to Phase 6.

5.1 Metal Kernel (Apple Silicon)
- Fused paged attention (gather + matmul in one kernel)
- Direct block addressing without copy
- Use tinygrad's MetalProgram API

5.2 Kernel Integration
- Auto-detect backend and select kernel
- Fallback to portable Tensor.stack() if no kernel
- Benchmark: measure speedup vs portable solution

---

### Phase 6: Optimizations (~300 lines)

6.1 Sampling Optimization (QUICK WINS)
- Remove .realize().tolist() sync points  ← easy, high impact
- Batched sampling across sequences (currently single token at a time)  --- not yet implemented since we have different sampling params
- Top-p/top-k on GPU (currently converts to CPU list)

6.2 Metal Kernel Optimizations (INCREMENTAL)
See docs/metal_optimizations.md for detailed research.

Quick wins:
- Pre-allocate output buffer once (reuse instead of Tensor.zeros() each call)
- Cache Metal buffer references (avoid uop tree traversal every call)
- Reuse block table tensor (overwrite values instead of new Tensor each call)
- Reuse context_lens tensor (pre-allocate [max_batch] tensor)
- half precision (float16 instead of float32 - 2x register efficiency)

Medium effort:
- 32-thread threadgroups (one simdgroup, avoid threadgroup barriers) ✅
- SIMD shuffle reductions (simd_sum instead of shared memory) ✅
- Vectorized float4 loads (16-byte aligned memory access) ✅
- simd_broadcast_first for exp() (eliminates 31 redundant exp() per position) ✅
- Safety assertions for kernel constraints (head_dim, GQA divisibility) ✅
- Loop unrolling (unroll head_dim loops for instruction-level parallelism) -- not needed, dynamic ctx_len
- Command buffer batching (multiple ops per submit) -- not completed, requires tinygrad changes

High effort:
- Online softmax (single pass instead of two-pass) ✅
- Buffer pooling (reusable GPU memory pools) ✅
- Tiled attention (Flash Attention style blocking) -- deferred to Phase 7.x (prefill focus)
- simdgroup_async_copy (overlap compute and memory loads, M1+) -- deferred to Phase 6.2.1

### Phase 6.2.1: simdgroup_async_copy (Future Research)
**Status:** Deferred - undocumented Metal API requires research

Metal's simdgroup_async_copy provides async memory transfers from device to threadgroup memory,
similar to CUDA's cp.async. Benefits:
- Overlap memory loads with compute (double-buffering)
- Hide memory latency during tile loading

Implementation requires:
- Research undocumented API (available since A14/M1)
- May require AIR (Apple IR) compilation workarounds
- Most effective when combined with tiled attention

6.3 Decode Optimization
- KV cache write batching (batch realizes instead of per-token)
- Fused kernels (RMSNorm + Linear, RoPE + Projection)
- Memory-efficient attention

6.4 Prefill Optimization (COMPLEX)
- Chunked prefill (don't block on long prompts)
- Batched prefill (process multiple prefill sequences together, currently one-by-one)
- Flash attention for prefill

6.5 Memory Optimization (ADVANCED)
- LRU eviction (least recently used)
- Memory defragmentation/compaction (prevent external fragmentation)
- CPU swap (move blocks to CPU when GPU full)
- Attention mask caching (reuse masks for common patterns)

6.6 CUDA Kernel (LAST)
- Fused paged attention for CUDA
- Optimized memory access patterns
- Use tinygrad's CUDAProgram API
- CUDA graphs (capture and replay decode loop)

6.7 Weight Quantization (CRITICAL - from vLLM research)
**Impact: 2-4x throughput improvement**
**Rationale:** Decode is memory-bandwidth-bound. Loading 2.2GB weights per token limits throughput to ~55 tok/s on M4. Quantization directly reduces data to load.

INT8 Quantization:
- Per-channel or per-tensor scale factors
- 2x compression (2.2GB → 1.1GB)
- Minimal quality loss (<1% perplexity)
- Dequantize on-the-fly during matmul

INT4 Quantization (GPTQ/AWQ style):
- Per-group scales (group_size=128 typical)
- 4x compression (2.2GB → 0.55GB)
- Requires calibration dataset
- Slight quality loss (~1-3% perplexity)

Implementation:
```python
class QuantizedLinear:
    def __init__(self, weight_int8: Tensor, scale: Tensor):
        self.weight = weight_int8  # [out, in] int8
        self.scale = scale         # [out] or [out, in//group_size]

    def __call__(self, x: Tensor) -> Tensor:
        # Dequantize during matmul for memory efficiency
        w_fp = self.weight.cast(dtypes.float16) * self.scale
        return x @ w_fp.T
```

Metal kernel for fused dequant+matmul:
- Load INT8 weights, dequantize in registers
- Compute matmul with FP16 accumulation
- Avoid materializing full FP16 weight tensor

6.8 Multi-Step Scheduling (HIGH IMPACT - from vLLM research)
**Impact: 20-30% throughput improvement**
**Rationale:** CPU scheduling overhead between every decode step wastes time. Batch multiple steps together.

Before (single-step):
```
CPU[schedule] → GPU[decode] → CPU[schedule] → GPU[decode] → ...
~10ms overhead between each step
```

After (multi-step, N=8):
```
CPU[schedule] → GPU[decode×8] → CPU[schedule] → GPU[decode×8] → ...
Overhead amortized over 8 steps
```

Implementation:
- Add `num_scheduler_steps` parameter to engine
- Scheduler returns batch for N steps
- GPU runs N decode iterations without returning to scheduler
- After N steps, check for finished sequences and new requests

Trade-offs:
- Pro: Less CPU overhead, higher throughput
- Con: New requests wait up to N steps for scheduling (higher TTFT at low load)
- Configurable: N=1 for latency-sensitive, N=8+ for throughput

6.9 FlashAttention for Prefill (HIGH IMPACT - from vLLM research)
**Impact: 2-4x prefill speedup, enables longer context**
**Rationale:** Standard attention materializes O(n²) attention matrix. FlashAttention uses O(1) memory via tiling.

Standard Attention Memory:
```
seq_len=2K: 2K × 2K × 4 bytes = 16 MB per head
seq_len=8K: 8K × 8K × 4 bytes = 256 MB per head
```

FlashAttention Memory:
```
Any seq_len: ~constant (tile size only)
```

Algorithm (simplified):
```
for q_tile in Q_tiles:
    running_max = -inf
    running_sum = 0
    running_output = 0

    for kv_tile in KV_tiles:
        # Compute attention for this tile
        scores = q_tile @ kv_tile.T

        # Online softmax update
        new_max = max(running_max, scores.max())
        scale = exp(running_max - new_max)

        running_sum = running_sum * scale + exp(scores - new_max).sum()
        running_output = running_output * scale + softmax(scores) @ v_tile
        running_max = new_max

    output_tile = running_output / running_sum
```

Implementation for Metal:
- Adapt Metal FlashAttention (github.com/philipturner/metal-flash-attention)
- Use simdgroup_matrix for tile operations
- Block sizes: 8×8 or 16×16 tiles for M-series

6.10 Async Output Processing (MEDIUM IMPACT - from vLLM research)
**Impact: 8-10% throughput improvement**
**Rationale:** Detokenization blocks GPU. Process previous step's output while GPU computes current step.

Before:
```
GPU[step N] → wait → CPU[detokenize N] → GPU[step N+1]
              ↑ GPU idle
```

After:
```
GPU[step N] ――――――――――――――――――――――――→ GPU[step N+1]
            CPU[detokenize N-1]
            ↑ overlapped
```

Implementation:
```python
class AsyncOutputProcessor:
    def __init__(self):
        self.pending_outputs = queue.Queue()
        self.worker = threading.Thread(target=self._process_loop)
        self.worker.start()

    def submit(self, tokens: List[int], callback):
        """Submit tokens for async detokenization."""
        self.pending_outputs.put((tokens, callback))

    def _process_loop(self):
        while True:
            tokens, callback = self.pending_outputs.get()
            text = self.tokenizer.decode(tokens)
            callback(text)
```

Engine modification:
- Don't wait for detokenization in step()
- Submit to async processor
- Return results via callback or poll

6.11 CPU Overhead Reduction (MEDIUM IMPACT - from vLLM research)
**Impact: 20-30% throughput improvement (combined)**

Object Pooling:
- Pre-allocate Request, Sequence, SchedulerOutput objects
- Reuse instead of alloc/free each step
- vLLM saw 24% improvement from this alone

```python
class ObjectPool:
    def __init__(self, factory, size=100):
        self.pool = [factory() for _ in range(size)]
        self.available = list(range(size))

    def acquire(self):
        if self.available:
            idx = self.available.pop()
            return self.pool[idx]
        return self.factory()  # Fallback

    def release(self, obj):
        obj.reset()  # Clear state
        self.available.append(self.pool.index(obj))
```

Fast Path for Greedy Sampling:
- Skip temperature/top-p/penalties when not needed
- Direct argmax for greedy decoding

```python
def sample_tokens(logits, params):
    if params.is_greedy:  # Fast path
        return logits.argmax(dim=-1)
    else:  # Full sampling
        return _sample_with_params(logits, params)
```

Non-blocking Tensor Operations:
- Use async CPU→GPU transfers where possible
- Avoid unnecessary .realize() calls
- Batch realize operations

---

### Phase 7: Advanced Features (Optional, ~400 lines)

7.1 Prefix Caching
- Hash prompt prefixes
- Reuse KV cache for common prefixes
- Cache eviction policy
- Cache hit/miss tracking

7.2 Speculative Decoding
- Draft model integration
- Parallel verification
- Token acceptance/rejection
- Speedup measurement

7.3 Multi-Sequence (Beam Search)
- Fork sequences (share prefix blocks)
- Sequence log probability tracking (needed for scoring)
- Beam scoring and pruning
- Merge/prune beams
- Copy-on-write for blocks

7.4 KV Cache Quantization
- Support int8/float16 KV cache storage
- Dequantization in attention kernel
- 2-4x memory reduction for KV cache

7.5 Request Scheduling
- Priority scheduling (not just FCFS)
- Request preemption (pause low-priority, resume high-priority)
- Adaptive batch size based on memory/queue depth

7.6 Generation Control
- Stop string matching (beyond EOS token)
- Constrained decoding (JSON schema, grammar)
- Logit processors plugin system

7.7 Metrics & Monitoring
- Per-step timing (prefill time, decode time, sample time)
- Token acceptance rate (for speculative decoding)
- Cache hit rate (for prefix caching)
- Memory profiling (used, free, fragmentation)

7.8 Flash Attention for Prefill (HIGH IMPACT)
**Impact: 2-4x prefill speedup, enables longer context**
**Prerequisite:** Online softmax (completed in Phase 6.2)

Tiled attention with online softmax for prefill phase:
- Process Q, K, V in tiles (block_size × block_size)
- Use threadgroup memory for tile storage
- Online softmax to combine tile results without storing full attention matrix

Algorithm:
```
for each Q tile:
    running_max, running_sum, running_out = -inf, 0, 0
    for each KV tile:
        load K_tile, V_tile to threadgroup memory
        scores = Q_tile @ K_tile.T * scale
        # Online softmax update
        new_max = max(running_max, scores.max())
        rescale = exp(running_max - new_max)
        running_sum = running_sum * rescale + sum(exp(scores - new_max))
        running_out = running_out * rescale + softmax(scores) @ V_tile
        running_max = new_max
    output_tile = running_out / running_sum
```

Benefits:
- O(1) memory regardless of sequence length (vs O(n²) for standard attention)
- Faster prefill for long prompts (4K+ tokens)
- Foundation for longer context windows (16K+)

7.9 Framework Overhead Reduction (CRITICAL - from Phase 6.2 profiling) - check profiling.md
**Impact: 10-50x throughput improvement potential**
**Status:** Research phase - identified as primary bottleneck

Profiling with tinygrad DEBUG=2 revealed the main performance gap is NOT GPU kernel speed,
but framework overhead between kernel calls.

Current bottlenecks identified:

| Bottleneck | Measured Impact | Root Cause |
|------------|-----------------|------------|
| Python→Metal copies | 87ms for 131MB (1.5 GB/s) | Slow unified memory path |
| Scheduling overhead | 2-20ms per batch | Python interprets kernel graph each step |
| No JIT caching | Repeats work | `TinyJit` not used in hot path |
| Many small ops | 6-12μs each | Kernel launch latency dominates |

**Current state:** 1.9 tok/s achieved vs 55 tok/s theoretical (3.5% utilization)

7.9.1 TinyJit for Decode Loop
Apply tinygrad's JIT decorator to cache kernel graphs:

```python
from tinygrad.engine.jit import TinyJit

class Llama:
    @TinyJit
    def batched_decode_jit(self, tokens, kv_cache, ...):
        # JIT requires fixed input shapes
        # Pad batch to max_batch_size
        return self._batched_decode_impl(tokens, kv_cache, ...)
```

Challenges:
- TinyJit requires fixed input shapes (variable batch size breaks it)
- Solution: Pad to fixed batch size, mask unused slots
- Warmup run required to capture kernel graph

Expected impact: 2-5x speedup (eliminates per-step scheduling overhead)

7.9.2 Reduce Python→GPU Data Copies
Model weights should stay on GPU, but may be re-copied:

Investigation needed:
- Profile which tensors are being copied (weights vs activations?)
- Ensure weights are `.realize()`d once at load time
- Check if block_tables/context_lens can stay on GPU

7.9.3 Kernel Fusion
Many small operations (6-12μs each) could be fused:

Candidates for fusion:
- RMSNorm + Linear projection
- RoPE + Q/K projection
- Softmax + V accumulation (done via online softmax)

tinygrad may auto-fuse some of these - need to verify with DEBUG=4.

7.9.4 Pre-allocated Buffers
Avoid per-step tensor allocation:

```python
class LLMEngine:
    def __init__(self):
        # Pre-allocate decode buffers
        self.input_buffer = Tensor.zeros(max_batch, 1).realize()
        self.output_buffer = Tensor.zeros(max_batch, 1, vocab_size).realize()
```

7.9.5 Metal System Trace
For deeper GPU analysis, use Xcode Instruments:
- Metal System Trace template
- Shows GPU utilization, kernel occupancy, memory bandwidth
- Identifies GPU-side bottlenecks vs CPU-side

---

### Phase 8: Multi-GPU Support (Optional, ~200 lines)

8.1 Tensor Parallelism
- Split model layers across GPUs
- All-reduce for layer outputs
- Device placement for weights

8.2 Sequence Parallelism
- Different sequences on different GPUs
- KVCache per GPU
- Load balancing across GPUs

8.3 Pipeline Parallelism
- Different layers on different GPUs
- Micro-batching for pipeline efficiency
- Async communication between stages

---

### Phase 9: API Server (~200 lines)

9.1 HTTP Server
- FastAPI or simple HTTP server
- POST /generate endpoint
- Request validation
- Async handling

9.2 Streaming
- Server-sent events (SSE)
- Token-by-token streaming
- Partial response handling

9.3 OpenAI Compatibility (Optional)
- POST /v1/completions
- POST /v1/chat/completions
- Response format matching
- Usage statistics

---

## File Structure

```
tinyvllm/
├── __init__.py
├── model/
│   ├── llama.py           # LLaMA architecture
│   ├── attention.py       # Paged attention
│   └── sampling.py        # Token sampling
├── core/
│   ├── block_manager.py   # Block allocation
│   ├── kv_cache.py        # Paged KV cache
│   ├── scheduler.py       # Continuous batching
│   └── sequence.py        # Request/sequence state
├── engine/
│   ├── engine.py          # Main engine loop
│   └── async_engine.py    # Async wrapper
├── api/
│   ├── server.py          # HTTP server
│   └── openai.py          # OpenAI compat
└── utils/
    ├── memory.py          # Memory tracking
    └── tokenizer.py       # Tokenizer wrapper
```

---

## Key Data Structures

```python
@dataclass
class Block:
    block_id: int
    ref_count: int = 0

@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: list[int]
    output_tokens: list[int]
    block_table: list[int]      # Logical → physical block mapping
    status: Literal["waiting", "running", "finished"]

@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    blocks_to_swap_in: dict[int, int]   # CPU → GPU
    blocks_to_swap_out: dict[int, int]  # GPU → CPU
    blocks_to_copy: dict[int, int]      # For beam search

@dataclass
class GenerateRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] = None
    stream: bool = False
```

---

## Milestones & Tests

| Phase | Milestone | Test |
|-------|-----------|------|
| 1 | Basic generation works | Generate "Hello" → coherent output |
| 2 | Paged attention works | Output identical with/without paging |
| 3 | Continuous batching works | 10 concurrent requests complete correctly |
| 4 | Batched forward improves throughput | >2x tokens/sec vs Phase 3 |
| 4 | BlockManager integration | Memory tracked, OOM prevented |
| 5 | Custom kernels faster | >1.5x speedup vs Tensor.stack() |
| 6 | Chunked prefill works | Long prompts don't block other requests |
| 6 | Batched sampling works | Sampling 16 sequences at once |
| 6.7 | **Weight quantization works** | **INT8 model loads, >1.5x throughput, <2% quality loss** |
| 6.8 | Multi-step scheduling works | N=8 steps per schedule, >20% throughput gain |
| 6.9 | FlashAttention prefill works | 2K+ context prefill in <1s, O(1) memory |
| 6.10 | Async output processing works | Detokenization overlaps with GPU compute |
| 6.11 | Object pooling works | No allocations in hot path, >10% throughput |
| 7 | Prefix caching works | Repeated prefixes hit cache |
| 7 | KV cache quantization works | int8 KV with <1% quality loss |
| 7 | Stop strings work | Generation stops at custom string |
| 7.2 | **Speculative decoding works** | **Draft model + verify, >2x throughput** |
| 8 | Multi-GPU works | Model runs across 2+ GPUs |
| 9 | API works | curl POST /generate returns response |
| 9 | Streaming works | Tokens arrive incrementally via SSE |

---

## Dependencies (Minimal)

Required:
- tinygrad
- safetensors (weight loading)
- tiktoken (tokenizer, or sentencepiece)

Optional:
- fastapi (API server)
- uvicorn (ASGI server)
- triton (custom kernels)

