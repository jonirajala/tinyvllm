"""Benchmark: Measure scalability (throughput vs batch size, max concurrent).

Run with:
  python -m benchmarks.bench_scalability                    # tiny test model
  python -m benchmarks.bench_scalability --model models/tinyllama  # real model
  DEVICE=METAL GPU=M4_10CORE python -m benchmarks.bench_scalability --model models/tinyllama
"""

import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from tinygrad import Tensor, Device

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig, load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine
from benchmarks.gpu_specs import get_gpu_specs, TheoreticalLimits


class MockTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False):
        tokens = [hash(c) % (self.vocab_size - 3) + 3 for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens):
        return "".join(chr((t % 26) + ord("a")) for t in tokens if t > 2)


def measure_throughput(model, tokenizer, n_requests: int, max_tokens: int) -> dict:
    """Measure throughput for n concurrent requests."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model, tokenizer, max_batch_size=n_requests)

    for i in range(n_requests):
        engine.add_request(f"Test prompt {i}", params)

    start = time.perf_counter()
    outputs = list(engine.run())
    elapsed = time.perf_counter() - start

    total_tokens = sum(len(o.tokens) for o in outputs)

    return {
        "n_requests": n_requests,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "requests_per_sec": n_requests / elapsed if elapsed > 0 else 0,
        "avg_tokens_per_request": total_tokens / n_requests if n_requests > 0 else 0,
    }


def measure_throughput_avg(model, tokenizer, n_requests: int, max_tokens: int, runs: int = 2) -> dict:
    """Measure throughput averaged over multiple runs."""
    results = []
    for _ in range(runs):
        results.append(measure_throughput(model, tokenizer, n_requests, max_tokens))

    avg_tok_sec = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_elapsed = sum(r["elapsed_sec"] for r in results) / len(results)

    return {
        "n_requests": n_requests,
        "total_tokens": results[0]["total_tokens"],
        "elapsed_sec": avg_elapsed,
        "tokens_per_sec": avg_tok_sec,
        "requests_per_sec": n_requests / avg_elapsed if avg_elapsed > 0 else 0,
    }


def run_scalability_benchmark(model_path: Optional[str] = None):
    print("=" * 50)
    print("tinyvllm Scalability Benchmark")
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
        total_params = sum(w.numel() for w in weights.values())
        sample_weight = next(iter(weights.values()))
        bytes_per_param = sample_weight.dtype.itemsize
        print(f"Model: {config.dim} dim, {config.n_layers} layers, {total_params/1e9:.2f}B params")
    else:
        print("\nCreating tiny test model...")
        config = LlamaConfig(
            dim=64, n_layers=4, n_heads=4, n_kv_heads=4,
            vocab_size=256, hidden_dim=256, max_seq_len=512
        )
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        total_params = 0
        bytes_per_param = 4.0

    print(f"Config: dim={config.dim}, layers={config.n_layers}, heads={config.n_heads}")

    # Calculate theoretical limits
    avg_context_len = 50 if model_path else 20
    theoretical = TheoreticalLimits(
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        context_len=avg_context_len,
        batch_size=1,
        gpu=gpu,
        n_layers=config.n_layers,
        dim=config.dim,
        vocab_size=config.vocab_size,
        total_params=total_params,
        bytes_per_param=bytes_per_param,
    )

    print("\n" + "=" * 50)
    print("Theoretical Limits")
    print("=" * 50)
    print(f"Full model max (batch=1): {theoretical.max_tokens_per_sec_full_model:,.0f} tok/s ({theoretical.full_model_bottleneck}-bound)")

    # Warmup - run multiple times with various batch sizes
    print("\n" + "=" * 50)
    print("Benchmarks")
    print("=" * 50)
    print("\nWarming up (5 runs)...")
    for i in range(5):
        engine = LLMEngine(model, tokenizer, max_batch_size=4)
        for j in range(min(i + 1, 4)):
            engine.add_request(f"warmup {j}", SamplingParams(max_tokens=10, temperature=0.0))
        list(engine.run())

    # Throughput vs batch size
    print("\n### Throughput vs Concurrent Requests (avg of 2 runs) ###")
    print("| Requests | Tokens | Time (s) | Tok/sec | Req/sec | Util % |")
    print("|----------|--------|----------|---------|---------|--------|")

    max_tokens = 10 if model_path else 15
    request_counts = [1, 2, 4] if model_path else [1, 2, 4, 8, 16]
    results = []
    for n_req in request_counts:
        # Calculate theoretical for this batch size
        theo_batched = TheoreticalLimits(
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            context_len=avg_context_len,
            batch_size=n_req,
            gpu=gpu,
            n_layers=config.n_layers,
            dim=config.dim,
            vocab_size=config.vocab_size,
            total_params=total_params,
            bytes_per_param=bytes_per_param,
        )
        result = measure_throughput_avg(model, tokenizer, n_req, max_tokens, runs=2)
        result["utilization"] = theo_batched.utilization_full_model(result["tokens_per_sec"])
        results.append(result)
        util_str = f"{result['utilization']:.1f}" if result['utilization'] >= 0.1 else f"{result['utilization']:.2f}"
        print(f"| {result['n_requests']:8} | {result['total_tokens']:6} | {result['elapsed_sec']:8.2f} | {result['tokens_per_sec']:7.1f} | {result['requests_per_sec']:7.2f} | {util_str:>5}% |")

    # Calculate scaling efficiency
    print("\n### Scaling Efficiency ###")
    if results:
        baseline = results[0]["tokens_per_sec"]
        print("| Requests | Throughput | vs 1 req | Ideal | Efficiency |")
        print("|----------|------------|----------|-------|------------|")
        for r in results:
            actual = r["tokens_per_sec"]
            ideal = baseline * r["n_requests"]
            efficiency = (actual / ideal * 100) if ideal > 0 else 0
            print(f"| {r['n_requests']:8} | {actual:10.1f} | {actual/baseline:8.2f}x | {r['n_requests']:5}x | {efficiency:9.1f}% |")

    # Memory scaling
    print("\n### Memory vs Concurrent Requests ###")
    print("| Requests | KV Cache Memory |")
    print("|----------|-----------------|")

    mem_request_counts = [1, 2, 4] if model_path else [1, 2, 4, 8, 16]
    for n_req in mem_request_counts:
        engine = LLMEngine(model, tokenizer, max_batch_size=n_req)
        params = SamplingParams(max_tokens=10, temperature=0.0)

        for i in range(n_req):
            prompt = f"Hello {i}" if model_path else f"Test {i}"
            engine.add_request(prompt, params)

        # Run a few steps
        steps = 5 if model_path else 8
        for _ in range(steps):
            if not engine.has_unfinished():
                break
            engine.step()

        mem_bytes = engine.kv_cache.get_memory_bytes()
        if mem_bytes > 1024 * 1024:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
        else:
            mem_str = f"{mem_bytes / 1024:.1f} KB"
        print(f"| {n_req:8} | {mem_str:>15} |")


    # Summary
    if model_path:
        print("\n### Summary ###")
        best_result = max(results, key=lambda r: r["tokens_per_sec"])
        print(f"Best throughput: {best_result['tokens_per_sec']:.1f} tok/s at {best_result['n_requests']} concurrent requests")
        print(f"Theoretical max: {theoretical.max_tokens_per_sec_full_model:,.0f} tok/s")
        print(f"Utilization: {best_result['utilization']:.1f}%")

    if config.dim < 512:
        print("\nNOTE: Small test model - use --model models/tinyllama for realistic benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scalability benchmark")
    parser.add_argument("--model", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_scalability_benchmark(model_path=args.model)
