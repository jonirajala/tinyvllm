# Tiny vllm
- 1000 loc limit
- Uses tinygrad

python  -m tinyvllm.main --model ./models/tinyllama --prompt "The capital of France is" --max-tokens 20 --device metal

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

### Phase 4 (Block-based KVCache + Batched Forward)

Config: dim=64, n_layers=4, n_heads=4, max_tokens=20, 5 concurrent requests

**METAL (Apple Silicon)**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 4.5 | 0.22 |
| sequential_5 | 2.4 | 0.12 |
| concurrent_5 | 6.6 | 0.33 |

Concurrent vs Sequential: **2.82x** speedup

**CPU**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 4.3 | 0.22 |
| sequential_5 | 2.4 | 0.12 |
| concurrent_5 | 6.0 | 0.30 |

Concurrent vs Sequential: **2.47x** speedup

### Memory Usage (Phase 4)

Run: `python -m benchmarks.bench_memory`

Model: dim=64, layers=4, kv_heads=4, head_dim=16
Theoretical KV memory per token: 2.0 KB

| Sequences | Tokens | KV Cache Memory | Bytes/Token |
|-----------|--------|-----------------|-------------|
| 1 | 18 | 64.0 KB | 3640 |
| 2 | 36 | 128.0 KB | 3640 |
| 5 | 90 | 320.0 KB | 3640 |
| 10 | 144 | 512.0 KB | 3640 |

Note: Block-based allocation has ~1.8x overhead vs theoretical due to
block granularity (16 tokens/block). Memory is pre-allocated for efficiency.

### Latency (Phase 4)

Run: `python -m benchmarks.bench_latency`

**METAL**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 633 | 200 | 2429 |
| 17 | 10 | 2360 | 148 | 3689 |
| 51 | 10 | 5746 | 191 | 7462 |

Summary: TTFT=825ms, TPOT=134ms, E2E=2695ms (15 tokens)

**CPU**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 859 | 230 | 2927 |
| 17 | 10 | 4085 | 166 | 5578 |
| 51 | 10 | 11780 | 319 | 14651 |

Summary: TTFT=835ms, TPOT=150ms, E2E=2931ms (15 tokens)

### Scalability (Phase 4)

Run: `python -m benchmarks.bench_scalability`

**METAL - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 4.0 | 100% |
| 2 | 5.1 | 63% |
| 4 | 5.9 | 37% |
| 8 | 5.8 | 18% |
| 16 | 6.0 | 9% |

**CPU - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 3.3 | 100% |
| 2 | 4.9 | 73% |
| 4 | 5.5 | 41% |
| 8 | 4.7 | 18% |
| 16 | 5.1 | 10% |

Phase 4 improvements:
- Block-based KVCache with pre-allocated tensors
- BlockManager for memory slot allocation
- Batched decode forward pass (multiple sequences in one call)
- Result: **>2x throughput improvement** over sequential processing

### Phase 5 (Custom Fused Kernels)

Config: dim=64, n_layers=4, n_heads=4, max_tokens=20, 5 concurrent requests

**METAL (Apple Silicon) - With Fused Paged Attention Kernel**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 10.2 | 0.51 |
| sequential_5 | 12.7 | 0.64 |
| concurrent_5 | 20.9 | 1.04 |

Concurrent vs Sequential: **1.64x** speedup

**CPU (Fallback tinygrad implementation)**
| Benchmark | Tokens/sec | Requests/sec |
|-----------|------------|--------------|
| single_request | 4.8 | 0.24 |
| sequential_5 | 11.1 | 0.66 |
| concurrent_5 | 5.5 | 0.33 |

Note: CPU fallback uses standard tinygrad ops without custom kernels.

### Memory Usage (Phase 5)

| Sequences | Tokens | KV Cache Memory | Bytes/Token |
|-----------|--------|-----------------|-------------|
| 1 | 18 | 64.0 KB | 3640 |
| 2 | 36 | 128.0 KB | 3640 |
| 5 | 90 | 320.0 KB | 3640 |
| 10 | 144 | 512.0 KB | 3640 |

Note: Memory usage unchanged from Phase 4 (same block-based allocation).

### Latency (Phase 5)

**METAL - With Fused Kernel**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 311 | 65 | 893 |
| 17 | 10 | 1096 | 66 | 1689 |
| 51 | 10 | 2977 | 77 | 3667 |

Summary: TTFT=197ms, TPOT=66ms, E2E=1117ms (15 tokens)

**CPU**
| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |
|------------|--------|-----------|-----------|----------|
| 3 | 10 | 453 | 152 | 1825 |
| 17 | 10 | 1864 | 80 | 2581 |
| 51 | 10 | 5294 | 231 | 7375 |

Summary: TTFT=197ms, TPOT=69ms, E2E=1162ms (15 tokens)

### Scalability (Phase 5)

**METAL - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 11.3 | 100% |
| 2 | 17.8 | 79% |
| 4 | 22.8 | 51% |
| 8 | 24.6 | 27% |
| 16 | 25.0 | 14% |

**CPU - Throughput vs Concurrent Requests**
| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 8.7 | 100% |
| 2 | 11.3 | 65% |
| 4 | 13.0 | 37% |
| 8 | 12.9 | 19% |
| 16 | 11.8 | 8% |

### Phase 5 vs Phase 4 Comparison

| Metric | Phase 4 (Metal) | Phase 5 (Metal) | Improvement |
|--------|-----------------|-----------------|-------------|
| Single-request throughput | 4.5 tok/s | 10.2 tok/s | **2.3x** |
| Concurrent (5 req) | 6.6 tok/s | 20.9 tok/s | **3.2x** |
| TPOT | 134 ms | 66 ms | **2x faster** |
| TTFT | 825 ms | 197 ms | **4.2x faster** |
| Max throughput (16 req) | 6.0 tok/s | 25.0 tok/s | **4.2x** |

Phase 5 improvements:
- Custom Metal kernel for fused paged attention
- Direct block table indexing (no Tensor.stack() overhead)
- Flat tensor storage in KVCache for efficient kernel access
- Backend dispatcher for Metal/CPU selection

### Real Model: TinyLlama 1.1B

Run with: `DEVICE=METAL GPU=M4_10CORE python -m benchmarks.bench_throughput --model models/tinyllama`

**Hardware:** Apple M4 10-core GPU (120 GB/s memory bandwidth)

**Model:** TinyLlama 1.1B (2048 dim, 22 layers, FP16 weights = 2.2 GB)

**Theoretical Maximum (Memory-Bound):** 55 tok/s

#### Throughput

| Benchmark | Tokens/sec | Utilization | Headroom |
|-----------|------------|-------------|----------|
| single_request | 1.4 | 2.6% | 97% |
| sequential_5 | 1.3 | 2.5% | 98% |
| concurrent_5 | 3.1 | 5.6% | 94% |

Concurrent vs Sequential: **2.26x** speedup

#### Latency

| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | 942 ms |
| TPOT (Time Per Output Token) | 722 ms |
| Decode throughput | 1.4 tok/s |

#### Memory

| Context Length | KV Cache per Sequence |
|----------------|----------------------|
| 512 | 11.0 MB |
| 1024 | 22.0 MB |
| 2048 | 44.0 MB |
| 4096 | 88.0 MB |

#### Scalability

| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 1.2 | 100% |
| 2 | 1.8 | 74% |
| 4 | 2.5 | 50% |

#### Analysis

Current utilization is ~5% of theoretical maximum (55 tok/s memory-bound limit).
The 18x gap is due to:
- Single-thread Metal kernel (using 1.25% of GPU threads)
- SIMD inefficiency (1/32 lanes active = 3% efficiency)
- Scalar memory loads (75% bandwidth waste)
- Two-pass softmax (2x KV cache reads)

See [docs/metal_optimizations.md](docs/metal_optimizations.md) for Phase 6 optimization roadmap targeting 10-30x speedup.

