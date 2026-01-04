#!/usr/bin/env python3
"""Benchmark JIT vs non-JIT decode performance.

Phase 7.1: TinyJit for Decode Loop

Compares:
- Non-JIT: Custom Metal kernel (MetalProgram)
- JIT: Pure tinygrad ops (JIT-compatible)

Usage:
    DEVICE=METAL ./venv/bin/python -m benchmarks.bench_jit --model models/tinyllama
"""

import os
import argparse
import time
from statistics import mean, stdev

from tinygrad import Device

# Allow setting device via env var
if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig, load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.engine.engine import LLMEngine
from tinyvllm.engine.sampling import SamplingParams


def benchmark_generation(
    engine: LLMEngine,
    prompt: str,
    max_tokens: int,
    num_runs: int = 5,
) -> dict:
    """Benchmark generation speed."""
    times = []
    tokens_generated = []

    for i in range(num_runs):
        # Add request
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))

        # Time generation
        start = time.perf_counter()
        outputs = list(engine.run())
        end = time.perf_counter()

        elapsed = end - start
        num_tokens = len(outputs[0].tokens)
        times.append(elapsed)
        tokens_generated.append(num_tokens)

        print(f"  Run {i+1}: {num_tokens} tokens in {elapsed:.3f}s ({num_tokens/elapsed:.1f} tok/s)")

    # Skip first run (warmup)
    times = times[1:]
    tokens_generated = tokens_generated[1:]

    return {
        "mean_time": mean(times),
        "std_time": stdev(times) if len(times) > 1 else 0,
        "mean_tokens": mean(tokens_generated),
        "tok_per_sec": mean(tokens_generated) / mean(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark JIT vs non-JIT decode")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today?", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    args = parser.parse_args()

    print(f"Device: {Device.DEFAULT}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs}")
    print()

    # Load model
    print("Loading model...")
    config, weights = load_llama_weights(args.model)
    model = create_llama(config, weights)
    tokenizer = load_tokenizer(args.model)

    # Benchmark non-JIT
    print("\n=== Non-JIT (Custom Metal Kernel) ===")
    engine_non_jit = LLMEngine(model, tokenizer, use_jit=False)
    results_non_jit = benchmark_generation(
        engine_non_jit, args.prompt, args.max_tokens, args.runs
    )

    # Benchmark JIT
    print("\n=== JIT (Pure Tinygrad Ops) ===")
    engine_jit = LLMEngine(model, tokenizer, use_jit=True)
    results_jit = benchmark_generation(
        engine_jit, args.prompt, args.max_tokens, args.runs
    )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Non-JIT: {results_non_jit['tok_per_sec']:.2f} tok/s (±{results_non_jit['std_time']*1000:.1f}ms)")
    print(f"JIT:     {results_jit['tok_per_sec']:.2f} tok/s (±{results_jit['std_time']*1000:.1f}ms)")

    speedup = results_jit['tok_per_sec'] / results_non_jit['tok_per_sec']
    if speedup > 1:
        print(f"\nJIT is {speedup:.2f}x faster")
    else:
        print(f"\nNon-JIT is {1/speedup:.2f}x faster")


if __name__ == "__main__":
    main()
