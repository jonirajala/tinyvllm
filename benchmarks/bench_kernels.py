"""Benchmark: Compare tinygrad vs Metal kernels for attention.

Compares:
- Paged decode attention: tinygrad ops vs custom Metal kernel
- Flash prefill attention: tinygrad ops vs custom Metal kernel

Run with:
  DEVICE=METAL python -m benchmarks.bench_kernels --model models/tinyllama
"""

import os
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from tinygrad import Tensor, Device, dtypes

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.weights import load_llama_weights, LlamaConfig


@dataclass
class BenchResult:
    name: str
    time_ms: float
    speedup: float = 1.0


def benchmark_fn(fn, warmup: int = 3, runs: int = 10) -> float:
    """Benchmark a function, return median time in ms."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        if hasattr(result, 'realize'):
            result.realize()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        if hasattr(result, 'realize'):
            result.realize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def bench_paged_decode_attention(
    batch_size: int = 4,
    n_heads: int = 32,
    n_kv_heads: int = 8,
    head_dim: int = 64,
    context_len: int = 128,
    block_size: int = 16,
) -> Tuple[BenchResult, BenchResult]:
    """Benchmark paged decode attention: tinygrad vs Metal."""
    from tinyvllm.kernels.paged_decode_attention_tinygrad import paged_decode_attention_tinygrad
    from tinyvllm.kernels.paged_decode_attention_metal import paged_decode_attention as metal_impl

    # Setup
    num_blocks = (context_len + block_size - 1) // block_size
    total_blocks = num_blocks * batch_size + 10  # Extra blocks

    queries = Tensor.randn(batch_size, 1, n_heads, head_dim).realize()
    k_cache = Tensor.randn(total_blocks, block_size, n_kv_heads, head_dim).realize()
    v_cache = Tensor.randn(total_blocks, block_size, n_kv_heads, head_dim).realize()

    # Block tables: each sequence uses consecutive blocks
    block_tables_list = []
    for i in range(batch_size):
        bt = list(range(i * num_blocks, (i + 1) * num_blocks))
        block_tables_list.append(bt)

    # Pad block tables
    max_blocks = max(len(bt) for bt in block_tables_list)
    padded = []
    for bt in block_tables_list:
        padded.extend(bt + [0] * (max_blocks - len(bt)))
    block_tables = Tensor(padded, dtype=dtypes.int32).reshape(batch_size, max_blocks).realize()
    context_lens = Tensor([context_len] * batch_size, dtype=dtypes.int32).realize()

    # Benchmark tinygrad version
    def run_tinygrad():
        return paged_decode_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

    tinygrad_time = benchmark_fn(run_tinygrad)

    # Benchmark Metal version
    def run_metal():
        return metal_impl(
            queries, k_cache, v_cache,
            block_tables_list, [context_len] * batch_size,
            n_heads, n_kv_heads, head_dim, block_size
        )

    metal_time = benchmark_fn(run_metal)

    speedup = tinygrad_time / metal_time if metal_time > 0 else 0

    return (
        BenchResult("tinygrad", tinygrad_time),
        BenchResult("Metal", metal_time, speedup)
    )


def bench_flash_prefill_attention(
    seq_len: int = 128,
    n_heads: int = 32,
    head_dim: int = 64,
) -> Tuple[BenchResult, BenchResult]:
    """Benchmark flash prefill attention: tinygrad vs Metal."""
    from tinyvllm.kernels.flash_prefill_attention_tinygrad import flash_prefill_attention_tinygrad
    from tinyvllm.kernels.flash_prefill_attention_metal import flash_prefill_attention_metal

    # Setup: [1, seq_len, n_heads, head_dim]
    query = Tensor.randn(1, seq_len, n_heads, head_dim).realize()
    key = Tensor.randn(1, seq_len, n_heads, head_dim).realize()
    value = Tensor.randn(1, seq_len, n_heads, head_dim).realize()

    # Benchmark tinygrad version
    def run_tinygrad():
        return flash_prefill_attention_tinygrad(query, key, value, causal=True)

    tinygrad_time = benchmark_fn(run_tinygrad)

    # Benchmark Metal version
    def run_metal():
        return flash_prefill_attention_metal(query, key, value, causal=True)

    metal_time = benchmark_fn(run_metal)

    speedup = tinygrad_time / metal_time if metal_time > 0 else 0

    return (
        BenchResult("tinygrad", tinygrad_time),
        BenchResult("Metal", metal_time, speedup)
    )


def bench_full_forward_batch(model, tokenizer, config, batch_size: int = 1, num_tokens: int = 20, use_metal: bool = True) -> Tuple[float, float]:
    """Benchmark full model forward pass with batching, return (prefill_ms, decode_ms_per_step)."""
    import tinyvllm.kernels as kernels
    from tinyvllm.engine.engine import LLMEngine
    from tinyvllm.engine.sampling import SamplingParams

    # Force kernel selection by overriding the dispatcher's cached kernel
    if use_metal:
        # Use Metal kernels
        from tinyvllm.kernels.paged_decode_attention_metal import PagedAttentionOnline
        from tinyvllm.kernels.flash_prefill_attention_metal import flash_prefill_attention_metal
        kernels._paged_decode_kernel = PagedAttentionOnline.get_instance().batched_tensors
        kernels._flash_metal_kernel = flash_prefill_attention_metal
    else:
        # Use tinygrad kernels
        kernels._paged_decode_kernel = kernels.paged_decode_attention_tinygrad
        kernels._flash_metal_kernel = kernels.flash_prefill_attention_tinygrad

    engine = LLMEngine(model, tokenizer)
    params = SamplingParams(max_tokens=num_tokens, temperature=0.0)

    # Add multiple requests
    prompts = ["Hello world", "How are you", "Tell me a story", "What is AI",
               "Explain quantum", "Write code", "Help me", "Good morning"]
    for i in range(batch_size):
        engine.add_request(prompts[i % len(prompts)], params)

    times = []
    while engine.has_unfinished():
        start = time.perf_counter()
        engine.step()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    # Prefill is first batch_size steps, decode is the rest
    prefill_ms = sum(times[:batch_size]) if len(times) >= batch_size else sum(times)
    decode_times = times[batch_size:] if len(times) > batch_size else []
    decode_ms = sum(decode_times) / len(decode_times) if decode_times else 0

    return prefill_ms, decode_ms


def run_benchmarks(model_path: Optional[str] = None):
    print("=" * 60)
    print("Kernel Benchmark: tinygrad vs Metal")
    print("=" * 60)
    print(f"Device: {Device.DEFAULT}")

    if Device.DEFAULT.split(":")[0].lower() != "metal":
        print("\nWARNING: Not running on Metal. Metal benchmarks will fail.")
        print("Run with: DEVICE=METAL python -m benchmarks.bench_kernels --model models/tinyllama")
        return

    # Load model config or use defaults
    model = None
    tokenizer = None
    if model_path:
        print(f"\nLoading model from {model_path}...")
        from tinyvllm.model.llama import create_llama
        from tinyvllm.model.tokenizer import load_tokenizer
        config, weights = load_llama_weights(Path(model_path))
        model = create_llama(config, weights)
        tokenizer = load_tokenizer(model_path)
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = config.head_dim
        n_layers = config.n_layers
        print(f"Model: {config.dim} dim, {n_layers} layers")
        print(f"Attention: {n_heads} heads, {n_kv_heads} KV heads, {head_dim} head_dim")
    else:
        print("\nNo model specified, using default config (TinyLlama-like)")
        n_heads = 32
        n_kv_heads = 4
        head_dim = 64
        n_layers = 22
        config = None
        print(f"Attention: {n_heads} heads, {n_kv_heads} KV heads, {head_dim} head_dim")

    block_size = 16

    # Paged Decode Attention benchmarks
    print("\n" + "=" * 60)
    print("Paged Decode Attention (decode phase)")
    print("=" * 60)

    decode_configs = [
        # (batch_size, context_len)
        (1, 32),
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (4, 128),
        (8, 128),
    ]

    print(f"\nConfig: {n_heads} heads, {n_kv_heads} KV heads, {head_dim} head_dim")
    print("\n| Batch | Context | tinygrad (ms) | Metal (ms) | Speedup |")
    print("|-------|---------|---------------|------------|---------|")

    for batch, ctx_len in decode_configs:
        try:
            tg, metal = bench_paged_decode_attention(
                batch_size=batch,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                context_len=ctx_len,
                block_size=block_size,
            )
            print(f"| {batch:5} | {ctx_len:7} | {tg.time_ms:13.2f} | {metal.time_ms:10.2f} | {metal.speedup:6.2f}x |")
        except Exception as e:
            print(f"| {batch:5} | {ctx_len:7} | ERROR: {str(e)[:30]} |")

    # Flash Prefill Attention benchmarks
    print("\n" + "=" * 60)
    print("Flash Prefill Attention (prefill phase)")
    print("=" * 60)

    prefill_configs = [32, 64, 128, 256, 512]

    print(f"\nConfig: {n_heads} heads, {head_dim} head_dim")
    print("\n| Seq Len | tinygrad (ms) | Metal (ms) | Speedup |")
    print("|---------|---------------|------------|---------|")

    for seq_len in prefill_configs:
        try:
            tg, metal = bench_flash_prefill_attention(
                seq_len=seq_len,
                n_heads=n_heads,
                head_dim=head_dim,
            )
            print(f"| {seq_len:7} | {tg.time_ms:13.2f} | {metal.time_ms:10.2f} | {metal.speedup:6.2f}x |")
        except Exception as e:
            print(f"| {seq_len:7} | ERROR: {str(e)[:30]} |")

    # Full model forward pass benchmark
    if model is not None and tokenizer is not None:
        print("\n" + "=" * 60)
        print("Full Model Forward Pass: tinygrad vs Metal kernels")
        print("=" * 60)

        # Warmup both paths
        print("\nWarming up...")
        for _ in range(2):
            bench_full_forward_batch(model, tokenizer, config, batch_size=1, num_tokens=5, use_metal=True)
            bench_full_forward_batch(model, tokenizer, config, batch_size=1, num_tokens=5, use_metal=False)

        num_runs = 3
        num_tokens = 20
        batch_sizes = [1, 2, 4, 8]

        print(f"\nRunning {num_runs} iterations per config, {num_tokens} tokens each...")
        print("\n| Batch | tinygrad (ms) | Metal (ms) | tinygrad tok/s | Metal tok/s | Speedup |")
        print("|-------|---------------|------------|----------------|-------------|---------|")

        for batch_size in batch_sizes:
            tg_times = []
            metal_times = []

            for _ in range(num_runs):
                # tinygrad kernels
                _, tg_dec = bench_full_forward_batch(model, tokenizer, config, batch_size=batch_size, num_tokens=num_tokens, use_metal=False)
                tg_times.append(tg_dec)

                # Metal kernels
                _, m_dec = bench_full_forward_batch(model, tokenizer, config, batch_size=batch_size, num_tokens=num_tokens, use_metal=True)
                metal_times.append(m_dec)

            avg_tg = sum(tg_times) / len(tg_times)
            avg_m = sum(metal_times) / len(metal_times)
            tg_tps = batch_size * 1000 / avg_tg if avg_tg > 0 else 0
            m_tps = batch_size * 1000 / avg_m if avg_m > 0 else 0
            speedup = avg_tg / avg_m if avg_m > 0 else 0

            print(f"| {batch_size:5} | {avg_tg:13.1f} | {avg_m:10.1f} | {tg_tps:14.2f} | {m_tps:11.2f} | {speedup:6.2f}x |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel benchmark: tinygrad vs Metal")
    parser.add_argument("--model", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_benchmarks(model_path=args.model)
