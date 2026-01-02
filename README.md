# Tiny vllm
- 1000 loc limit
- Uses tinygrad

## what it vllm

vLLM is an LLM inference engine that serves many users simultaneously by treating GPU memory like virtual memory.
Used when need to serve llm to multiple users

  ---
  The Problem

  LLM generates tokens one at a time.
  Each token needs to "remember" all previous tokens.
  This memory (KV cache) is HUGE and grows with each token.

  Naive approach: Pre-allocate max memory per user
  → 90% of GPU memory wasted on tokens never generated
  → Can only serve ~4 users on an 80GB GPU

  ---
  vLLM's Solution

  1. PagedAttention: Allocate memory in small blocks, on-demand
     → Like OS virtual memory pages
     → No waste, 10-24x more users

  2. Continuous Batching: Don't wait for slow requests
     → When one finishes, immediately add another
     → GPU never idle



## Reason
  ---
  3. Edge / Embedded Deployment

  ┌─────────────────────────────────────────────────┐
  │           Edge Device (Jetson, etc.)            │
  │                                                 │
  │  Resources: Limited GPU memory, limited CPU     │
  │                                                 │
  │  vLLM: Heavy dependencies, designed for A100    │
  │        → Won't run or runs poorly               │
  │                                                 │
  │  tinyvLLM: Minimal dependencies                 │
  │        → Runs on small GPU                      │
  │        → Custom optimizations for hardware      │
  └─────────────────────────────────────────────────┘

  Use cases:
  - On-device AI assistant
  - Robotics
  - IoT with local LLM

  ---
  4. Custom Hardware / Backends

  ┌─────────────────────────────────────────────────┐
  │        Novel Hardware (TPU, FPGA, etc.)         │
  │                                                 │
  │  vLLM: Tightly coupled to CUDA                  │
  │        → Porting is massive effort              │
  │                                                 │
  │  tinyvLLM: Clean abstractions                   │
  │        → Implement PagedAttention for TPU       │
  │        → Or your custom accelerator             │
  │        → Or tinygrad backend!                   │
  └─────────────────────────────────────────────────┘

  Use case: tinyvLLM + tinygrad = run on ANY backend tinygrad supports

  # tinyvLLM on tinygrad
  from tinygrad import Tensor, Device

  Device.DEFAULT = "METAL"  # or "CUDA", "AMD", "WEBGPU"

  class TinyVLLMEngine:
      def __init__(self):
          self.kv_cache = PagedKVCache()  # Uses tinygrad Tensors

      # Now runs on any tinygrad backend!

  ---
  5. Single-User / Personal Use

  ┌─────────────────────────────────────────────────┐
  │           Personal LLM Server                    │
  │                                                 │
  │  You: Running LLaMA on your RTX 4090            │
  │       Just for yourself, maybe a few friends    │
  │                                                 │
  │  vLLM: Overkill, complex setup                  │
  │        Multi-GPU features you don't need        │
  │                                                 │
  │  tinyvLLM: Simple, starts in seconds            │
  │        Perfect for 1-10 concurrent users        │
  │        Easy to customize                        │
  └─────────────────────────────────────────────────┘

## Struct


  ---
  Phase 1: Foundation (~400 lines)

  1.1 Basic Inference Loop

  □ Token generation loop (greedy)
  □ KV cache (naive, pre-allocated)
  □ Sampling (temperature, top-p, top-k)
  □ Stop conditions (EOS, max length)

  1.2 Model Loading

  □ Load LLaMA weights (safetensors)
  □ LLaMA architecture (attention, FFN, RMSNorm)
  □ RoPE positional embeddings
  □ Tokenizer (use tiktoken or sentencepiece)

  1.3 Basic API

  □ Simple Python interface
  □ generate(prompt) → string
  □ generate_stream(prompt) → iterator

  ---
  Phase 2: Paged Attention (~400 lines)

  2.1 Block Manager

  □ Block allocator (malloc/free for KV blocks)
  □ Free list management
  □ Block reference counting
  □ Sequence → block table mapping

  2.2 Paged KV Cache

  □ Physical block storage (GPU tensor)
  □ Logical → physical mapping
  □ Allocate on demand (as tokens generated)
  □ Free blocks when sequence done

  2.3 Paged Attention Kernel

  □ Gather K,V from scattered blocks
  □ Compute attention with block tables
  □ Handle variable sequence lengths
  □ (Optional) CUDA kernel for speed

  ---
  Phase 3: Continuous Batching (~300 lines)

  3.1 Request Queue

  □ Add request (prompt + params)
  □ Request states (waiting, running, finished)
  □ Request metadata (arrival time, tokens generated)
  □ Callbacks for completion

  3.2 Scheduler

  □ Select requests for next batch
  □ Dynamic batch composition
  □ Add new requests when slots free
  □ Remove finished requests immediately

  3.3 Batch Execution

  □ Prepare batched inputs
  □ Handle mixed prefill + decode
  □ Update KV cache per sequence
  □ Dispatch results to requests

  ---
  Phase 4: Memory Management & KV Cache Optimization (~200 lines)

  Note: BlockManager (block_manager.py) already exists with tests.
        Currently unused - KVCache manages its own lists.
        This phase integrates BlockManager for proper memory control.

  4.1 KV Cache Optimization (Tinygrad-native)

  □ Replace list-based KV cache with pre-allocated block tensors
  □ Use separate realized tensors per block (tinygrad compatible)
  □ Tensor.stack() for gathering blocks during attention
  □ See docs/phase4_kv_optimization.md for details

  4.2 Integrate BlockManager

  □ Connect BlockManager to Scheduler (can_allocate checks)
  □ Connect BlockManager to KVCache (block-based storage)
  □ Scheduler calls block_manager.allocate_sequence() on prefill
  □ Scheduler calls block_manager.free_sequence() on finish

  4.3 Memory Tracking

  □ Track GPU memory usage
  □ Track blocks per sequence
  □ Memory budget enforcement
  □ OOM prevention

  4.4 Eviction Policy

  □ LRU eviction (least recently used)
  □ Preemption (pause low-priority requests)
  □ Recomputation (evict, recompute later)

  4.5 CPU Swap (Optional)

  □ Swap blocks to CPU when GPU full
  □ Swap back when needed
  □ Async transfers

  ---
  Phase 5: Optimizations (~300 lines)

  5.1 Prefill Optimization

  □ Chunked prefill (don't block on long prompts)
  □ Parallel prefill for batch
  □ Flash attention for prefill

  5.2 Decode Optimization

  □ CUDA graphs (capture and replay)
  □ Fused kernels (RMSNorm + attention)
  □ Memory-efficient attention

  5.3 Sampling Optimization

  □ Batched sampling
  □ Top-p/top-k on GPU
  □ Avoid CPU-GPU sync

  ---
  Phase 6: API Server (~200 lines)

  6.1 HTTP Server

  □ FastAPI or simple HTTP server
  □ POST /generate endpoint
  □ Request validation
  □ Async handling

  6.2 Streaming

  □ Server-sent events (SSE)
  □ Token-by-token streaming
  □ Partial response handling

  6.3 OpenAI Compatibility (Optional)

  □ POST /v1/completions
  □ POST /v1/chat/completions
  □ Response format matching
  □ Usage statistics

  ---
  Phase 7: Advanced Features (Optional, ~400 lines)

  7.1 Prefix Caching

  □ Hash prompt prefixes
  □ Reuse KV cache for common prefixes
  □ Cache eviction policy
  □ Cache hit/miss tracking

  7.2 Speculative Decoding

  □ Draft model integration
  □ Parallel verification
  □ Token acceptance/rejection
  □ Speedup measurement

  7.3 Multi-Sequence (Beam Search)

  □ Fork sequences (share prefix blocks)
  □ Track multiple hypotheses
  □ Merge/prune beams
  □ Copy-on-write for blocks

  ---
  Phase 8: Multi-GPU Support (Optional, ~200 lines)

  8.1 Tensor Parallelism

  □ Split model layers across GPUs
  □ All-reduce for layer outputs
  □ Device placement for weights

  8.2 Sequence Parallelism

  □ Different sequences on different GPUs
  □ KVCache per GPU
  □ Load balancing across GPUs

  8.3 Pipeline Parallelism

  □ Different layers on different GPUs
  □ Micro-batching for pipeline efficiency
  □ Async communication between stages

  ---
  Implementation Order (Recommended)

  Week 1: Phase 1 (Foundation)
          → Get basic generation working

  Week 2: Phase 2 (Paged Attention)
          → Core innovation, most learning

  Week 3: Phase 3 (Continuous Batching)
          → Handle multiple requests

  Week 4: Phase 4 + 5 (Memory + Optimizations)
          → Make it practical

  Week 5: Phase 6 (API)
          → Make it usable

  Week 6+: Phase 7 (Advanced)
          → Nice-to-haves

  ---
  File Structure

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

  ---
  Key Data Structures

  # Core structures to implement:

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

  ---
  Milestones & Tests

  □ Milestone 1: Generate "Hello" → "Hello, world!"
    Test: Single request, greedy decoding

  □ Milestone 2: Paged attention matches naive
    Test: Output identical with/without paging

  □ Milestone 3: 10 concurrent requests
    Test: All complete correctly, no OOM

  □ Milestone 4: Throughput > naive batching
    Test: Measure tokens/sec improvement

  □ Milestone 5: Memory efficiency
    Test: 2x more concurrent requests than naive

  □ Milestone 6: API works
    Test: curl POST returns response

  □ Milestone 7: Streaming works
    Test: Tokens arrive incrementally

  ---
  Metrics to Track

  @dataclass
  class Metrics:
      # Throughput
      tokens_per_second: float
      requests_per_second: float

      # Latency
      time_to_first_token: float    # TTFT
      time_per_output_token: float  # TPOT
      total_latency: float

      # Memory
      gpu_memory_used: int
      kv_cache_blocks_used: int
      kv_cache_blocks_free: int

      # Efficiency
      batch_size_avg: float
      gpu_utilization: float
      cache_hit_rate: float  # For prefix caching

  ---
  Dependencies (Minimal)

  # Required
  tinygrad
  safetensors          # Weight loading
  tiktoken             # Tokenizer (or sentencepiece)

  # Optional
  fastapi              # API server
  uvicorn              # ASGI server
  triton               # Custom kernels

