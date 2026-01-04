# tinyvllm Benchmarks

Benchmark results for tinyvllm on Apple Silicon.

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
