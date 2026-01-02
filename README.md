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
| single_request | 2.8 | 0.14 |
| sequential_5 | 4.3 | 0.21 |
| concurrent_5 | 4.4 | 0.22 |

Concurrent vs Sequential: **1.02x** speedup

**CPU**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 1.8 | 0.09 |
| sequential_5 | 3.9 | 0.20 |
| concurrent_5 | 4.7 | 0.23 |

Concurrent vs Sequential: **1.19x** speedup

Note: Current implementation processes one sequence per step.
Real batching gains will come in Phase 4 with batched forward passes.

### Memory Usage (Phase 3)

Run: `python -m benchmarks.bench_memory`

Model: dim=64, layers=4, kv_heads=4, head_dim=16
Theoretical KV memory per token: 2.0 KB (2 × layers × kv_heads × head_dim × 4 bytes)

| Sequences | Tokens | KV Cache Memory | Bytes/Token |
|-----------|--------|-----------------|-------------|
| 1 | 18 | 36.0 KB | 2048 |
| 2 | 36 | 72.0 KB | 2048 |
| 5 | 90 | 180.0 KB | 2048 |
| 10 | 144 | 288.0 KB | 2048 |

Note: Memory scales linearly with tokens. Phase 4 paged attention will add
block-based allocation for better memory management under pressure.

### Latency (Phase 3)

Run: `python -m benchmarks.bench_latency`

**METAL**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 176 | 242 | 2356 |
| 17 | 10 | 380 | 328 | 3335 |
| 51 | 10 | 873 | 756 | 7673 |

Summary: TTFT=98ms, TPOT=191ms, E2E=2774ms (15 tokens)

**CPU**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 238 | 425 | 4064 |
| 17 | 10 | 465 | 548 | 5401 |
| 51 | 10 | 1071 | 1184 | 11722 |

Summary: TTFT=95ms, TPOT=180ms, E2E=2617ms (15 tokens)

### Scalability (Phase 3)

Run: `python -m benchmarks.bench_scalability`

**METAL - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 4.3 | 100% |
| 2 | 5.5 | 64% |
| 4 | 6.1 | 35% |
| 8 | 5.5 | 16% |
| 16 | 4.6 | 7% |

**CPU - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 3.8 | 100% |
| 2 | 6.1 | 80% |
| 4 | 5.8 | 38% |
| 8 | 5.4 | 18% |
| 16 | 4.5 | 7% |

Note: Efficiency = (actual throughput) / (baseline × n_requests).
Drops sharply because we process one sequence per step.
Phase 4 batched forward passes should maintain higher efficiency at scale.

