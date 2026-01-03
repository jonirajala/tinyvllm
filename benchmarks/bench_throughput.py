"""Benchmark: Measure tokens/sec throughput.

Run with:
  python -m benchmarks.bench_throughput                    # tiny test model
  python -m benchmarks.bench_throughput --model models/tinyllama  # real model
  DEVICE=METAL GPU=M4_10CORE python -m benchmarks.bench_throughput --model models/tinyllama
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from tinygrad import Tensor, Device

# Allow setting device via env var
if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig, load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine
from benchmarks.gpu_specs import get_gpu_specs, TheoreticalLimits, GPUSpecs


@dataclass
class BenchmarkResult:
    name: str
    num_requests: int
    total_tokens: int
    elapsed_sec: float
    tokens_per_sec: float
    requests_per_sec: float
    utilization_pct: Optional[float] = None  # vs theoretical max


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
    if result.utilization_pct is not None:
        print(f"  Utilization: {result.utilization_pct:.1f}% of theoretical max")


def run_benchmarks(model_path: Optional[str] = None):
    print("=" * 50)
    print("tinyvllm Throughput Benchmark")
    print("=" * 50)
    print(f"Device: {Device.DEFAULT}")

    # Get GPU specs for theoretical limits
    gpu_name = os.environ.get("GPU", "M4_10CORE")
    gpu = get_gpu_specs(gpu_name)
    print(f"GPU specs: {gpu.name}")

    # Create or load model
    if model_path:
        print(f"\nLoading model from {model_path}...")
        config, weights = load_llama_weights(Path(model_path))
        model = create_llama(config, weights)
        tokenizer = load_tokenizer(model_path)
        # Estimate total params from weights
        total_params = sum(w.numel() for w in weights.values())
        # Determine bytes per param from dtype
        sample_weight = next(iter(weights.values()))
        bytes_per_param = sample_weight.dtype.itemsize
        print(f"Model: {config.dim} dim, {config.n_layers} layers, {total_params/1e9:.2f}B params")
    else:
        print("\nCreating tiny test model...")
        model, config = create_model(dim=64, n_layers=4, n_heads=4, vocab_size=256)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        total_params = 0  # Will be estimated from architecture
        bytes_per_param = 4.0  # FP32

    max_tokens = 20
    prompts = [
        "Hello world",
        "How are you",
        "Write code",
        "Test prompt",
        "Another one",
    ]

    # Calculate theoretical limits for this workload
    # Estimate average context length during decode
    avg_context_len = 50 if model_path else 20  # longer for real models
    theoretical = TheoreticalLimits(
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        context_len=avg_context_len,
        batch_size=1,
        gpu=gpu,
        # Full model parameters for realistic estimates
        n_layers=config.n_layers,
        dim=config.dim,
        vocab_size=config.vocab_size,
        total_params=total_params,
        bytes_per_param=bytes_per_param,
    )

    print("\n" + "=" * 50)
    print("Theoretical Limits")
    print("=" * 50)
    print(theoretical.report())

    # Warmup - multiple runs to ensure JIT is warmed
    print("\n" + "=" * 50)
    print("Benchmarks")
    print("=" * 50)
    print("\nWarming up (3 runs)...")
    for _ in range(3):
        engine = LLMEngine(model, tokenizer)
        engine.add_request("warmup prompt", SamplingParams(max_tokens=10, temperature=0.0))
        list(engine.run())

    results = []

    # Single request
    print("\nRunning single request benchmark...")
    r1 = bench_single_request(model, tokenizer, "Hello world", max_tokens)
    r1.utilization_pct = theoretical.utilization_full_model(r1.tokens_per_sec)
    results.append(r1)
    print_result(r1)

    # Sequential requests (baseline)
    print("\nRunning sequential requests benchmark...")
    r2 = bench_sequential_requests(model, tokenizer, prompts, max_tokens)
    r2.utilization_pct = theoretical.utilization_full_model(r2.tokens_per_sec)
    results.append(r2)
    print_result(r2)

    # Concurrent requests (continuous batching)
    print("\nRunning concurrent requests benchmark...")
    # Update theoretical for batched workload
    theoretical_batched = TheoreticalLimits(
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        context_len=avg_context_len,
        batch_size=len(prompts),
        gpu=gpu,
        n_layers=config.n_layers,
        dim=config.dim,
        vocab_size=config.vocab_size,
        total_params=total_params,
        bytes_per_param=bytes_per_param,
    )
    r3 = bench_concurrent_requests(model, tokenizer, prompts, max_tokens)
    r3.utilization_pct = theoretical_batched.utilization_full_model(r3.tokens_per_sec)
    results.append(r3)
    print_result(r3)

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    if r2.tokens_per_sec > 0:
        speedup = r3.tokens_per_sec / r2.tokens_per_sec
        print(f"\nConcurrent vs Sequential speedup: {speedup:.2f}x")

    print("\n| Benchmark          | Tokens/sec | Util % | Headroom |")
    print("|--------------------|------------|--------|----------|")
    for r in results:
        headroom = f"{100 - r.utilization_pct:.0f}%" if r.utilization_pct else "N/A"
        # Show more precision for very small utilization values
        if r.utilization_pct and r.utilization_pct < 0.01:
            util = f"{r.utilization_pct:.4f}%"
        elif r.utilization_pct:
            util = f"{r.utilization_pct:.2f}%"
        else:
            util = "N/A"
        print(f"| {r.name:18} | {r.tokens_per_sec:10.1f} | {util:>8} | {headroom:>8} |")

    # Theoretical max summary
    best_tps = max(r.tokens_per_sec for r in results)
    util_pct = theoretical.utilization_full_model(best_tps)
    print(f"\nTheoretical max (full model): {theoretical.max_tokens_per_sec_full_model:,.0f} tok/s")
    print(f"Best achieved: {best_tps:,.1f} tok/s")
    if util_pct < 0.01:
        print(f"Utilization: {util_pct:.4f}%")
    else:
        print(f"Utilization: {util_pct:.2f}%")
    print(f"Potential speedup: {theoretical.max_tokens_per_sec_full_model / best_tps:.1f}x")

    if config.dim < 512:
        print("\nNOTE: Small test model (dim={}, {:.1f} MB) - framework overhead dominates.".format(
            config.dim, theoretical.model_bytes / 1e6))
        print("      Use --model models/tinyllama for realistic benchmarks.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--model", type=str, help="Path to model directory (e.g., models/tinyllama)")
    args = parser.parse_args()
    run_benchmarks(model_path=args.model)
