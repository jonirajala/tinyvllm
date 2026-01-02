"""Benchmark: Measure tokens/sec throughput.

Run with:
  python -m benchmarks.bench_throughput          # default device
  DEVICE=CPU python -m benchmarks.bench_throughput  # CPU
  DEVICE=METAL python -m benchmarks.bench_throughput  # Metal
"""

import os
import time
from dataclasses import dataclass
from typing import List

from tinygrad import Tensor, Device

# Allow setting device via env var
if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine


@dataclass
class BenchmarkResult:
    name: str
    num_requests: int
    total_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    requests_per_sec: float


class MockTokenizer:
    """Mock tokenizer for benchmarks."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False):
        tokens = [hash(c) % (self.vocab_size - 3) + 3 for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens):
        return "".join(chr((t % 26) + ord("a")) for t in tokens if t > 2)


def create_model(dim=64, n_layers=4, n_heads=4, vocab_size=256):
    """Create a small model for benchmarking."""
    config = LlamaConfig(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        hidden_dim=dim * 4,
        max_seq_len=512,
    )
    return Llama(config), config


def bench_single_request(model, tokenizer, prompt: str, max_tokens: int) -> BenchmarkResult:
    """Benchmark single request generation."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model, tokenizer)
    engine.add_request(prompt, params)

    start = time.perf_counter()
    outputs = list(engine.run())
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.tokens) for o in outputs)

    return BenchmarkResult(
        name="single_request",
        num_requests=1,
        total_tokens=total_tokens,
        elapsed_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed if elapsed > 0 else 0,
        requests_per_sec=1 / elapsed if elapsed > 0 else 0,
    )


def bench_concurrent_requests(
    model, tokenizer, prompts: List[str], max_tokens: int
) -> BenchmarkResult:
    """Benchmark concurrent request generation."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model, tokenizer)

    for prompt in prompts:
        engine.add_request(prompt, params)

    start = time.perf_counter()
    outputs = list(engine.run())
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.tokens) for o in outputs)

    return BenchmarkResult(
        name=f"concurrent_{len(prompts)}",
        num_requests=len(prompts),
        total_tokens=total_tokens,
        elapsed_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed if elapsed > 0 else 0,
        requests_per_sec=len(prompts) / elapsed if elapsed > 0 else 0,
    )


def bench_sequential_requests(
    model, tokenizer, prompts: List[str], max_tokens: int
) -> BenchmarkResult:
    """Benchmark sequential (one at a time) request generation."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    total_tokens = 0

    start = time.perf_counter()
    for prompt in prompts:
        engine = LLMEngine(model, tokenizer)
        engine.add_request(prompt, params)
        outputs = list(engine.run())
        total_tokens += sum(len(o.tokens) for o in outputs)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=f"sequential_{len(prompts)}",
        num_requests=len(prompts),
        total_tokens=total_tokens,
        elapsed_sec=elapsed,
        tokens_per_sec=total_tokens / elapsed if elapsed > 0 else 0,
        requests_per_sec=len(prompts) / elapsed if elapsed > 0 else 0,
    )


def print_result(result: BenchmarkResult):
    print(f"\n{result.name}:")
    print(f"  Requests:    {result.num_requests}")
    print(f"  Tokens:      {result.total_tokens}")
    print(f"  Time:        {result.elapsed_sec:.3f}s")
    print(f"  Tokens/sec:  {result.tokens_per_sec:.1f}")
    print(f"  Requests/sec:{result.requests_per_sec:.2f}")


def run_benchmarks():
    print("=" * 50)
    print("tinyvllm Throughput Benchmark")
    print("=" * 50)
    print(f"Device: {Device.DEFAULT}")

    # Create model
    print("\nCreating model...")
    model, config = create_model(dim=64, n_layers=4, n_heads=4, vocab_size=256)
    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    max_tokens = 20
    prompts = [
        "Hello world",
        "How are you",
        "Write code",
        "Test prompt",
        "Another one",
    ]

    # Warmup - multiple runs to ensure JIT is warmed
    print("Warming up (3 runs)...")
    for _ in range(3):
        engine = LLMEngine(model, tokenizer)
        engine.add_request("warmup prompt", SamplingParams(max_tokens=10, temperature=0.0))
        list(engine.run())

    results = []

    # Single request
    print("\nRunning single request benchmark...")
    r1 = bench_single_request(model, tokenizer, "Hello world", max_tokens)
    results.append(r1)
    print_result(r1)

    # Sequential requests (baseline)
    print("\nRunning sequential requests benchmark...")
    r2 = bench_sequential_requests(model, tokenizer, prompts, max_tokens)
    results.append(r2)
    print_result(r2)

    # Concurrent requests (continuous batching)
    print("\nRunning concurrent requests benchmark...")
    r3 = bench_concurrent_requests(model, tokenizer, prompts, max_tokens)
    results.append(r3)
    print_result(r3)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    if r2.tokens_per_sec > 0:
        speedup = r3.tokens_per_sec / r2.tokens_per_sec
        print(f"\nConcurrent vs Sequential speedup: {speedup:.2f}x")

    print("\n| Benchmark | Tokens/sec | Requests/sec |")
    print("|-----------|------------|--------------|")
    for r in results:
        print(f"| {r.name:18} | {r.tokens_per_sec:10.1f} | {r.requests_per_sec:12.2f} |")

    return results


if __name__ == "__main__":
    run_benchmarks()
