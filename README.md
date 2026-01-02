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

See [phases.md](phases.md) for implementation roadmap.

## Benchmark Results

Run: `python -m benchmarks.bench_throughput`

### Phase 3 (Continuous Batching - No Batched Forward)

Config: dim=64, n_layers=4, n_heads=4, max_tokens=20, 5 concurrent requests

**METAL (Apple Silicon)**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 2.2 | 0.11 |
| sequential_5 | 4.5 | 0.23 |
| concurrent_5 | 4.7 | 0.23 |

Concurrent vs Sequential: **1.04x** speedup

**CPU**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 1.1 | 0.05 |
| sequential_5 | 3.6 | 0.18 |
| concurrent_5 | 5.1 | 0.25 |

Concurrent vs Sequential: **1.39x** speedup

Note: Current implementation processes one sequence per step.
Real batching gains will come in Phase 4 with batched forward passes.

