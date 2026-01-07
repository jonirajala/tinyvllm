"""Benchmark: Measure latency metrics (TTFT, TPOT, E2E).

Run with:
  python -m benchmarks.bench_latency                    # tiny test model
  python -m benchmarks.bench_latency --model models/tinyllama  # real model
  DEVICE=METAL GPU=M4_10CORE python -m benchmarks.bench_latency --model models/tinyllama
"""

import os
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from tinygrad import Tensor, Device

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig, load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine
from benchmarks.gpu_specs import get_gpu_specs, TheoreticalLimits


@dataclass
class LatencyResult:
    ttft_ms: float  # Time to first token
    tpot_ms: float  # Time per output token (average)
    e2e_ms: float   # End-to-end latency
    num_tokens: int
    decode_tok_per_sec: Optional[float] = None  # Decode throughput
    utilization_pct: Optional[float] = None     # vs theoretical max


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


def measure_latency_detailed(model, tokenizer, prompt: str, max_tokens: int, runs: int = 3) -> LatencyResult:
    """Measure latency with per-token timing, using median of multiple runs."""
    all_ttft = []
    all_tpot = []
    all_e2e = []
    num_tokens = 0

    for _ in range(runs):
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        engine = LLMEngine(model, tokenizer)
        engine.add_request(prompt, params)

        token_times = []
        start = time.perf_counter()
        first_token_time = None

        step_count = 0
        while engine.has_unfinished():
            step_start = time.perf_counter()
            outputs = engine.step()
            step_end = time.perf_counter()

            step_count += 1
            token_times.append((step_end - step_start) * 1000)

            if first_token_time is None and step_count == 1:
                first_token_time = (step_end - start) * 1000

            if outputs:
                break

        end = time.perf_counter()

        ttft = first_token_time if first_token_time else 0
        decode_times = token_times[1:] if len(token_times) > 1 else []
        tpot = sum(decode_times) / len(decode_times) if decode_times else 0
        e2e = (end - start) * 1000

        all_ttft.append(ttft)
        all_tpot.append(tpot)
        all_e2e.append(e2e)
        num_tokens = len(token_times)

    # Use median for stability
    all_ttft.sort()
    all_tpot.sort()
    all_e2e.sort()
    mid = len(all_ttft) // 2

    return LatencyResult(
        ttft_ms=all_ttft[mid],
        tpot_ms=all_tpot[mid],
        e2e_ms=all_e2e[mid],
        num_tokens=num_tokens
    )


def run_latency_benchmark(model_path: Optional[str] = None):
    print("=" * 50)
    print("tinyvllm Latency Benchmark")
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

    # Calculate theoretical limits
    avg_context_len = 50 if model_path else 15
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
    print(f"Full model max: {theoretical.max_tokens_per_sec_full_model:,.0f} tok/s ({theoretical.full_model_bottleneck}-bound)")
    print(f"Min TPOT: {1000 / theoretical.max_tokens_per_sec_full_model:.1f} ms")

    # Warmup
    print("\n" + "=" * 50)
    print("Benchmarks")
    print("=" * 50)
    print("\nWarming up (2 runs)...")
    for _ in range(2):
        engine = LLMEngine(model, tokenizer)
        engine.add_request("warmup", SamplingParams(max_tokens=5, temperature=0.0))
        list(engine.run())

    # Single request latency
    print("\n### Single Request Latency ###")
    print("| Prompt | Tokens | TTFT (ms) | TPOT (ms) | tok/s | Util % |")
    print("|--------|--------|-----------|-----------|-------|--------|")

    prompts = [("Hi", 10), ("Hello world", 10), ("A" * 20, 10)] if not model_path else [
        ("Hello", 10), ("How are you today?", 10), ("Tell me a story", 10)
    ]

    for prompt, max_tok in prompts:
        result = measure_latency_detailed(model, tokenizer, prompt, max_tok)
        decode_tps = 1000 / result.tpot_ms if result.tpot_ms > 0 else 0
        util = theoretical.utilization_full_model(decode_tps)
        print(f"| {prompt[:8]:8} | {result.num_tokens:6} | {result.ttft_ms:9.0f} | {result.tpot_ms:9.0f} | {decode_tps:5.1f} | {util:5.1f}% |")

    # Summary
    print("\n### Summary ###")
    result = measure_latency_detailed(model, tokenizer, "Hello world", 15)
    decode_tps = 1000 / result.tpot_ms if result.tpot_ms > 0 else 0
    util = theoretical.utilization_full_model(decode_tps)

    print(f"TTFT (Time to First Token): {result.ttft_ms:.0f} ms")
    print(f"TPOT (Time Per Output Token): {result.tpot_ms:.0f} ms")
    print(f"Decode throughput: {decode_tps:.1f} tok/s")
    print(f"E2E: {result.e2e_ms:.0f} ms for {result.num_tokens} tokens")

    print("\n### vs Theoretical Max ###")
    print(f"Theoretical max: {theoretical.max_tokens_per_sec_full_model:,.0f} tok/s")
    print(f"Achieved: {decode_tps:.1f} tok/s")
    print(f"Utilization: {util:.1f}%")
    if decode_tps > 0:
        print(f"Potential speedup: {theoretical.max_tokens_per_sec_full_model / decode_tps:.1f}x")

    if config.dim < 512:
        print("\nNOTE: Small test model - use --model models/tinyllama for realistic benchmarks.")

    # Context length scaling
    print("\n" + "=" * 50)
    print("Context Length Scaling")
    print("=" * 50)

    context_lengths = [32, 128, 256, 512] if model_path else [16, 64, 128, 256]

    def generate_prompt(target_tokens: int) -> str:
        if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size == 256:
            return "A" * target_tokens
        else:
            base = "The quick brown fox jumps over the lazy dog. "
            return (base * (target_tokens * 4 // len(base) + 1))[:target_tokens * 4]

    print("\n| Context | TTFT (ms) | TPOT (ms) | Decode tok/s |")
    print("|---------|-----------|-----------|--------------|")

    for ctx_len in context_lengths:
        try:
            prompt = generate_prompt(ctx_len)
            result = measure_latency_detailed(model, tokenizer, prompt, max_tokens=10)
            decode_tps = 1000 / result.tpot_ms if result.tpot_ms > 0 else 0
            print(f"| {ctx_len:>7} | {result.ttft_ms:>9.0f} | {result.tpot_ms:>9.0f} | {decode_tps:>12.1f} |")
        except Exception as e:
            print(f"| {ctx_len:>7} | ERROR: {str(e)[:30]} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--model", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_latency_benchmark(model_path=args.model)
