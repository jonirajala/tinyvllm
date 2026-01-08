# tinyvllm Benchmarks

Benchmark results for tinyvllm on Apple Silicon.

---

# TinyJit Phase (Latest - 2026-01-07)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Device | Apple M4 10-core GPU |
| Model | TinyLlama 1.1B (FP16) |
| Model Size | 2.05 GB |
| Dimensions | 2048 dim, 22 layers, 32 heads |
| Date | 2026-01-07 |

**Key changes:**
- Removed custom Metal kernels in favor of pure tinygrad
- TinyJit compilation for decode loop (4x speedup)
- Per-engine JIT caching for determinism
- Simplified codebase (deleted kernels/ folder)

## Theoretical Limits

| Metric | Value |
|--------|-------|
| GPU Memory Bandwidth | 120 GB/s |
| Max tok/s (memory-bound) | 55 tok/s |
| Min TPOT | 18.3 ms |

---

## Throughput

| Benchmark | Tokens/sec | Utilization | vs Phase 7.4 |
|-----------|------------|-------------|--------------|
| Single request | 4.9 | 8.9% | **+188%** |
| Sequential (5 requests) | 4.7 | 8.7% | **+176%** |
| Concurrent (5 requests) | 4.7 | 8.6% | **+31%** |

**Concurrent vs Sequential speedup: 1.00x** (JIT warmup dominates)

### Context Length Scaling

| Context Length | Tokens | Time (s) | Tok/sec |
|----------------|--------|----------|---------|
| 32 | 10 | 4.03 | 2.5 |
| 128 | 10 | 7.35 | 1.4 |
| 256 | 10 | 10.06 | 1.0 |
| 512 | 10 | 11.25 | 0.9 |

---

## Latency

| Metric | Value | vs Phase 7.4 |
|--------|-------|--------------|
| TTFT (Time to First Token) | 670 ms | **-19%** |
| TPOT (Time Per Output Token) | 238 ms | **-59%** |
| Decode throughput | 4.2 tok/s | **+147%** |
| E2E (15 tokens) | 4019 ms | **-55%** |

### Per-Prompt Latency

| Prompt | Tokens | TTFT (ms) | TPOT (ms) | tok/s |
|--------|--------|-----------|-----------|-------|
| Hello | 10 | 636 | 237 | 4.2 |
| How are | 10 | 656 | 247 | 4.1 |
| Tell me | 10 | 683 | 255 | 3.9 |

### Context Length Scaling

| Context | TTFT (ms) | TPOT (ms) | Decode tok/s |
|---------|-----------|-----------|--------------|
| 32 | 1394 | 372 | 2.7 |
| 128 | 3481 | 350 | 2.9 |
| 256 | 5703 | 293 | 3.4 |
| 512 | 6395 | 300 | 3.3 |

---

## Memory

| Metric | Value |
|--------|-------|
| Model weights | 2.05 GB |
| KV memory per token | 22.0 KB |

### KV Cache Scaling

| Sequences | Tokens | KV Cache |
|-----------|--------|----------|
| 1 | 8 | 352.0 KB |
| 2 | 16 | 704.0 KB |
| 5 | 40 | 1.7 MB |

---

## Scalability

### Throughput vs Concurrent Requests

| Requests | Tok/sec | Efficiency | vs Phase 7.4 |
|----------|---------|------------|--------------|
| 1 | 3.4 | 100.0% | **+113%** |
| 2 | 4.1 | 60.8% | **+86%** |
| 4 | 4.7 | 34.8% | **+62%** |

**Best throughput: 4.7 tok/s at 4 concurrent requests**

---

## TinyJit Phase vs Phase 7.4 Summary

| Metric | Phase 7.4 | TinyJit | Change |
|--------|-----------|---------|--------|
| Single request | 1.7 tok/s | 4.9 tok/s | **+188%** |
| Concurrent (5) | 3.6 tok/s | 4.7 tok/s | **+31%** |
| TTFT | 828 ms | 670 ms | **-19%** |
| TPOT | 574 ms | 238 ms | **-59%** |
| Decode throughput | 1.7 tok/s | 4.2 tok/s | **+147%** |
| Best scalability | 2.9 tok/s | 4.7 tok/s | **+62%** |
| E2E (15 tokens) | 8857 ms | 4019 ms | **-55%** |

**Key insights:**
- TinyJit provides massive decode speedup (2.4x TPOT reduction)
- Single request throughput nearly tripled (1.7 → 4.9 tok/s)
- Concurrent speedup less dramatic because JIT warmup amortizes better with batching
- Pure tinygrad approach simpler and faster than custom Metal kernels

---

# Phase 7.4 (Previous - 2026-01-06)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Device | Apple M4 10-core GPU |
| Model | TinyLlama 1.1B (FP16) |
| Model Size | 2.05 GB |
| Dimensions | 2048 dim, 22 layers, 32 heads |
| Date | 2026-01-06 |

**Note:** Phase 8 deferred KV writes was attempted but stashed - falls back to slower tinygrad kernel, net negative.

## Theoretical Limits

| Metric | Value |
|--------|-------|
| GPU Memory Bandwidth | 120 GB/s |
| Max tok/s (memory-bound) | 55 tok/s |
| Min TPOT | 18.3 ms |

---

## Throughput

| Benchmark | Tokens/sec | Utilization | vs Phase 7 |
|-----------|------------|-------------|------------|
| Single request | 1.7 | 3.2% | same |
| Sequential (5 requests) | 1.7 | 3.2% | same |
| Concurrent (5 requests) | 3.6 | 6.6% | same |

**Concurrent vs Sequential speedup: 2.06x**

### Context Length Scaling

| Context Length | Tokens | Time (s) | Tok/sec |
|----------------|--------|----------|---------|
| 32 | 10 | 12.03 | 0.8 |
| 128 | 10 | 29.61 | 0.3 |
| 256 | 10 | 52.77 | 0.2 |
| 512 | 10 | 99.77 | 0.1 |

---

## Latency

| Metric | Value | vs Phase 7 |
|--------|-------|------------|
| TTFT (Time to First Token) | 828 ms | **-45%** |
| TPOT (Time Per Output Token) | 574 ms | **-13%** |
| Decode throughput | 1.7 tok/s | **+13%** |
| E2E (15 tokens) | 8857 ms | **-12%** |

### Per-Prompt Latency

| Prompt | Tokens | TTFT (ms) | TPOT (ms) | tok/s |
|--------|--------|-----------|-----------|-------|
| Hello | 10 | 695 | 565 | 1.8 |
| How are | 10 | 1250 | 566 | 1.8 |
| Tell me | 10 | 1144 | 576 | 1.7 |

### Context Length Scaling

| Context | TTFT (ms) | TPOT (ms) | Decode tok/s |
|---------|-----------|-----------|--------------|
| 32 | 5426 | 601 | 1.7 |
| 128 | 20118 | 632 | 1.6 |
| 256 | 40348 | 642 | 1.6 |
| 512 | 75477 | 623 | 1.6 |

---

## Memory

| Metric | Value |
|--------|-------|
| Model weights | 2.05 GB |
| KV memory per token | 22.0 KB |

### KV Cache Scaling

| Sequences | Tokens | KV Cache |
|-----------|--------|----------|
| 1 | 8 | 352.0 KB |
| 2 | 16 | 704.0 KB |
| 5 | 40 | 1.7 MB |

---

## Scalability

### Throughput vs Concurrent Requests

| Requests | Tok/sec | Efficiency | vs Phase 7 |
|----------|---------|------------|------------|
| 1 | 1.6 | 100.0% | same |
| 2 | 2.2 | 68.7% | same |
| 4 | 2.9 | 46.4% | **+4%** |

**Best throughput: 2.9 tok/s at 4 concurrent requests**

---

## Timing Breakdown

| Operation | Median (ms) | % of Total |
|-----------|-------------|------------|
| Decode forward | 434.22 | 78.3% |
| Prefill | 2057.08 | 19.5% |
| Sampling | 11.48 | 2.2% |
| Tokenization | 0.05 | 0.0% |
| Detokenization | 0.06 | 0.0% |

**Decode throughput: 2.3 tok/s**

---

## Phase 7.4 vs Phase 7 Summary

| Metric | Phase 7 | Phase 7.4 | Change |
|--------|---------|-----------|--------|
| Single request | 1.7 tok/s | 1.7 tok/s | same |
| Concurrent (5) | 3.6 tok/s | 3.6 tok/s | same |
| TTFT | 1502 ms | 828 ms | **-45%** |
| TPOT | 657 ms | 574 ms | **-13%** |
| Decode throughput | 1.5 tok/s | 1.7 tok/s | **+13%** |
| Best scalability | 2.8 tok/s | 2.9 tok/s | **+4%** |
| Decode forward | 733 ms | 434 ms | **-41%** |

**Key insight:** Decode forward time dropped significantly (733→434ms, -41%), but overall throughput stayed similar due to other bottlenecks (prefill still dominates).

---

# Phase 7 (Previous)

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Device | Apple M4 10-core GPU |
| Model | TinyLlama 1.1B (FP16) |
| Model Size | 2.05 GB |
| Dimensions | 2048 dim, 22 layers, 32 heads |
| Date | 2026-01-06 |

**Phase 7 includes:**
- 7.3 Multi-step scheduling
- 7.4 Reduce Python→GPU copies

## Theoretical Limits

| Metric | Value |
|--------|-------|
| GPU Memory Bandwidth | 120 GB/s |
| Max tok/s (memory-bound) | 55 tok/s |
| Min TPOT | 18.3 ms |

---

## Throughput

| Benchmark | Tokens/sec | Utilization | vs Phase 6.2 |
|-----------|------------|-------------|--------------|
| Single request | 1.7 | 3.0% | **+42%** |
| Sequential (5 requests) | 1.7 | 3.1% | **+42%** |
| Concurrent (5 requests) | 3.6 | 6.6% | **+33%** |

**Concurrent vs Sequential speedup: 2.12x**

### Context Length Scaling

| Context Length | Tokens | Time (s) | Tok/sec |
|----------------|--------|----------|---------|
| 32 | 10 | 12.54 | 0.8 |
| 128 | 10 | 31.07 | 0.3 |
| 256 | 10 | 52.08 | 0.2 |
| 512 | 10 | 319.50 | 0.0 |

---

## Latency

| Metric | Value | vs Phase 6.2 |
|--------|-------|--------------|
| TTFT (Time to First Token) | 1502 ms | - |
| TPOT (Time Per Output Token) | 657 ms | **-30%** |
| Decode throughput | 1.5 tok/s | **+36%** |
| E2E (14 tokens) | 10059 ms | - |

### Per-Prompt Latency

| Prompt | Tokens | TTFT (ms) | TPOT (ms) | tok/s |
|--------|--------|-----------|-----------|-------|
| Hello | 9 | 1304 | 649 | 1.5 |
| How are you | 9 | 2442 | 628 | 1.6 |
| Tell me a story | 9 | 2147 | 631 | 1.6 |

### Context Length Scaling

| Context | TTFT (ms) | TPOT (ms) | Decode tok/s |
|---------|-----------|-----------|--------------|
| 32 | 7015 | 649 | 1.5 |
| 128 | 33550 | 633 | 1.6 |
| 256 | 67749 | 1295 | 0.8 |
| 512 | 208856 | 1186 | 0.8 |

---

## Memory

| Metric | Value |
|--------|-------|
| Model weights | 2.05 GB |
| KV memory per token | 22.0 KB |

### KV Cache Scaling

| Sequences | Tokens | KV Cache |
|-----------|--------|----------|
| 1 | 9 | 352.0 KB |
| 2 | 18 | 704.0 KB |
| 5 | 45 | 1.7 MB |

### Memory Projections (per sequence)

| Context Length | KV Cache |
|----------------|----------|
| 512 | 11.0 MB |
| 1024 | 22.0 MB |
| 2048 | 44.0 MB |
| 4096 | 88.0 MB |

---

## Scalability

### Throughput vs Concurrent Requests

| Requests | Tok/sec | Efficiency | vs Phase 6.2 |
|----------|---------|------------|--------------|
| 1 | 1.5 | 100.0% | **+36%** |
| 2 | 2.1 | 71.1% | **+40%** |
| 4 | 2.8 | 47.0% | **+47%** |

**Best throughput: 2.8 tok/s at 4 concurrent requests**

---

## Timing Breakdown

| Operation | Median (ms) | % of Total |
|-----------|-------------|------------|
| Decode forward | 733.04 | 72.5% |
| Prefill | 5068.68 | 26.4% |
| Sampling | 10.89 | 1.1% |
| Tokenization | 0.07 | 0.0% |
| Detokenization | 1.07 | 0.0% |

**Decode throughput: 1.4 tok/s**

---

## Phase 7 Improvements Summary

| Metric | Phase 6.2 | Phase 7 | Improvement |
|--------|-----------|---------|-------------|
| Single request | 1.2 tok/s | 1.7 tok/s | **+42%** |
| Concurrent (5) | 2.7 tok/s | 3.6 tok/s | **+33%** |
| TPOT | 939 ms | 657 ms | **-30%** |
| Decode throughput | 1.1 tok/s | 1.5 tok/s | **+36%** |
| Best scalability | 1.9 tok/s | 2.8 tok/s | **+47%** |

---

# Phase 6.2
## Test Configuration

| Parameter | Value |
|-----------|-------|
| Device | Apple M4 10-core GPU |
| Model | TinyLlama 1.1B (FP16) |
| Model Size | 2.05 GB |
| Dimensions | 2048 dim, 22 layers, 32 heads |
| Date | 2025-01-04 |

## Theoretical Limits

| Metric | Value |
|--------|-------|
| GPU Memory Bandwidth | 120 GB/s |
| Max tok/s (memory-bound) | 55 tok/s |
| Min TPOT | 18.3 ms |

---

## Throughput

| Benchmark | Tokens/sec | Utilization |
|-----------|------------|-------------|
| Single request | 1.2 | 2.2% |
| Sequential (5 requests) | 1.2 | 2.2% |
| Concurrent (5 requests) | 2.7 | 4.9% |

**Concurrent vs Sequential speedup: 2.21x**

### Context Length Scaling

| Context Length | Tokens | Time (s) | Tok/sec |
|----------------|--------|----------|---------|
| 32 | 10 | 15.32 | 0.7 |
| 128 | 10 | 38.07 | 0.3 |
| 256 | 10 | 62.74 | 0.2 |
| 512 | 10 | 127.27 | 0.1 |

---

## Latency

| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | 1079 ms |
| TPOT (Time Per Output Token) | 939 ms |
| Decode throughput | 1.1 tok/s |
| E2E (15 tokens) | 14325 ms |

### Per-Prompt Latency

| Prompt | Tokens | TTFT (ms) | TPOT (ms) | tok/s |
|--------|--------|-----------|-----------|-------|
| Hello | 10 | 1025 | 1096 | 0.9 |
| How are you | 10 | 2411 | 946 | 1.1 |
| Tell me a story | 10 | 1881 | 923 | 1.1 |

### Context Length Scaling

| Context | TTFT (ms) | TPOT (ms) | Decode tok/s |
|---------|-----------|-----------|--------------|
| 32 | 8447 | 980 | 1.0 |
| 128 | 28802 | 869 | 1.2 |
| 256 | 56006 | 947 | 1.1 |
| 512 | 95757 | 917 | 1.1 |

---

## Memory

| Metric | Value |
|--------|-------|
| Model weights | 2.05 GB |
| KV memory per token | 22.0 KB |

### KV Cache Scaling

| Sequences | Tokens | KV Cache |
|-----------|--------|----------|
| 1 | 8 | 352.0 KB |
| 2 | 16 | 704.0 KB |
| 5 | 40 | 1.7 MB |

### Memory Projections (per sequence)

| Context Length | KV Cache |
|----------------|----------|
| 512 | 11.0 MB |
| 1024 | 22.0 MB |
| 2048 | 44.0 MB |
| 4096 | 88.0 MB |

---

## Scalability

### Throughput vs Concurrent Requests

| Requests | Tok/sec | Efficiency |
|----------|---------|------------|
| 1 | 1.1 | 100.0% |
| 2 | 1.5 | 68.4% |
| 4 | 1.9 | 44.4% |

**Best throughput: 1.9 tok/s at 4 concurrent requests**

---

## Timing Breakdown

| Operation | Median (ms) | % of Total |
|-----------|-------------|------------|
| Decode forward | 520.37 | 75.3% |
| Prefill | 3084.57 | 23.5% |
| Sampling | 7.88 | 1.2% |
| Tokenization | 0.07 | 0.0% |
| Detokenization | 0.13 | 0.0% |

**Decode throughput: 1.9 tok/s**

---

## Running Benchmarks

```bash
# All benchmarks with real model
DEVICE=METAL ./venv/bin/python -m benchmarks.bench_throughput --model models/tinyllama
DEVICE=METAL ./venv/bin/python -m benchmarks.bench_latency --model models/tinyllama
DEVICE=METAL ./venv/bin/python -m benchmarks.bench_memory --model models/tinyllama
DEVICE=METAL ./venv/bin/python -m benchmarks.bench_scalability --model models/tinyllama
DEVICE=METAL ./venv/bin/python -m benchmarks.bench_breakdown --model models/tinyllama

# Quick tests with tiny model (no --model flag)
./venv/bin/python -m benchmarks.bench_throughput
```
