# TinyVLLM

Minimal vLLM implementation in ~1000 lines using tinygrad.

## Features

- PagedAttention with block-based KV cache
- Continuous batching
- JIT-compiled decode path
- Multi-backend support (Metal, CUDA, CPU)

## Usage

```bash
# Basic generation
python -m tinyvllm.main --model ./models/tinyllama --prompt "Hello" --max-tokens 50

# With sampling params
python -m tinyvllm.main --model ./models/tinyllama --prompt "The capital of France is" \
    --temperature 0.8 --top-k 40 --top-p 0.95

# Specify device
python -m tinyvllm.main --model ./models/tinyllama --prompt "Hello" --device metal
```

## Benchmarks

```bash
# Throughput
DEVICE=METAL python -m benchmarks.bench_throughput --model models/tinyllama

# Latency
DEVICE=METAL python -m benchmarks.bench_latency --model models/tinyllama

# Run tests
python -m pytest tests/ -v
```

## Requirements

- Python 3.10+
- tinygrad
- sentencepiece (for tokenizer)
