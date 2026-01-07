#!/usr/bin/env python3
"""
Benchmark breakdown: Time and memory for each step of the inference pipeline.

This benchmark identifies bottlenecks by measuring:
1. Tokenization (encode)
2. Prefill (process prompt, fill KV cache)
3. Decode loop:
   - Attention (paged attention kernel)
   - FFN (feed-forward layers)
   - Sampling (token selection)
4. Detokenization (decode)

Usage:
    python benchmarks/bench_breakdown.py                    # Use tiny test model
    python benchmarks/bench_breakdown.py --model models/tinyllama  # Use real model
"""

import argparse
import time
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from contextlib import contextmanager

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tinygrad import Tensor, Device, dtypes

# Timing utilities
@dataclass
class TimingStats:
    """Accumulates timing statistics for a named operation."""
    name: str
    times: List[float] = field(default_factory=list)

    def record(self, elapsed_ms: float):
        self.times.append(elapsed_ms)

    @property
    def total_ms(self) -> float:
        return sum(self.times)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0

    @property
    def median_ms(self) -> float:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
        return sorted_times[n//2]

    @property
    def min_ms(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_ms(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def std_ms(self) -> float:
        if len(self.times) < 2:
            return 0
        avg = self.avg_ms
        variance = sum((t - avg) ** 2 for t in self.times) / len(self.times)
        return variance ** 0.5


class BreakdownProfiler:
    """Profiles each step of the inference pipeline."""

    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
        self._start_time: Optional[float] = None
        self._current_op: Optional[str] = None

    def start(self, name: str):
        """Start timing an operation."""
        # Ensure any GPU ops are done before timing
        Tensor.realize
        self._current_op = name
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop timing and record."""
        if self._current_op is None:
            return
        elapsed = (time.perf_counter() - self._start_time) * 1000
        if self._current_op not in self.stats:
            self.stats[self._current_op] = TimingStats(self._current_op)
        self.stats[self._current_op].record(elapsed)
        self._current_op = None

    @contextmanager
    def time(self, name: str):
        """Context manager for timing."""
        self.start(name)
        try:
            yield
        finally:
            self.stop()

    def report(self) -> str:
        """Generate a timing report using median for stability."""
        lines = []
        # Use median-based total for more stable results
        total_median = sum(s.median_ms * s.count for s in self.stats.values())

        lines.append("=" * 80)
        lines.append("TIMING BREAKDOWN (median values for stability)")
        lines.append("=" * 80)
        lines.append(f"{'Operation':<25} {'Count':>6} {'Median':>10} {'Std':>8} {'Total':>10} {'%':>7}")
        lines.append("-" * 80)

        # Sort by total time descending
        sorted_stats = sorted(self.stats.values(), key=lambda s: s.median_ms * s.count, reverse=True)

        for stat in sorted_stats:
            total_ms = stat.median_ms * stat.count
            pct = (total_ms / total_median * 100) if total_median > 0 else 0
            lines.append(f"{stat.name:<25} {stat.count:>6} {stat.median_ms:>10.2f} {stat.std_ms:>8.2f} {total_ms:>10.1f} {pct:>6.1f}%")

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<25} {'':<6} {'':<10} {'':<8} {total_median:>10.1f}")
        lines.append("=" * 80)

        return "\n".join(lines)


def create_test_model():
    """Create a tiny model for testing."""
    from tinyvllm.model.llama import Llama
    from tinyvllm.model.weights import LlamaConfig

    config = LlamaConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        hidden_dim=128,
        max_seq_len=512,
    )
    return Llama(config), config


def create_mock_tokenizer(vocab_size: int = 256):
    """Create a mock tokenizer for testing."""
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.eos_id = 2
            self.bos_id = 1

        def encode(self, text: str) -> List[int]:
            # Simple encoding: ASCII values mod vocab_size
            return [ord(c) % self.vocab_size for c in text]

        def decode(self, tokens: List[int]) -> str:
            # Simple decoding
            return "".join(chr(t % 128) if t < 128 else "?" for t in tokens)

    return MockTokenizer(vocab_size)


def load_real_model(model_path: str):
    """Load a real model from disk."""
    from tinyvllm.model.llama import Llama
    from tinyvllm.model.weights import load_llama_weights
    from tinyvllm.model.tokenizer import load_tokenizer

    config, weights = load_llama_weights(Path(model_path))
    model = Llama(config)
    model.load_weights(weights)

    tokenizer = load_tokenizer(Path(model_path))

    return model, config, tokenizer


def benchmark_inference_breakdown(
    model,
    config,
    tokenizer,
    prompt: str = "Hello, how are you today?",
    max_tokens: int = 20,
    warmup_runs: int = 3,
    benchmark_runs: int = 3,
) -> BreakdownProfiler:
    """
    Benchmark each step of inference with detailed timing.
    """
    from tinyvllm.core.kv_cache import KVCache
    from tinyvllm.core.block_manager import BlockManager
    from tinyvllm.core.sampling import SamplingParams, sample_tokens

    profiler = BreakdownProfiler()

    # Setup
    block_size = 16
    num_blocks = 100

    block_manager = BlockManager(
        num_gpus=1,
        blocks_per_gpu=num_blocks,
        block_size=block_size,
    )

    kv_cache = KVCache(
        num_layers=config.n_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=config.dtype,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Warmup (simple forward pass without KV cache)
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        tokens = tokenizer.encode(prompt)[:10]  # Short prompt for warmup
        input_ids = Tensor([tokens])
        # Use a temporary setup for warmup
        warmup_bm = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=block_size)
        warmup_kv = KVCache(
            num_layers=config.n_layers, num_blocks=10, block_size=block_size,
            n_kv_heads=config.n_kv_heads, head_dim=config.head_dim, dtype=config.dtype
        )
        warmup_bm.allocate_sequence(seq_id=999, num_tokens=10)
        _ = model(input_ids, start_pos=0, kv_cache=warmup_kv, block_manager=warmup_bm, seq_id=999).realize()
        warmup_bm.free_sequence(999)

    print(f"Running benchmark with prompt: '{prompt[:50]}...'")
    print(f"Generating {max_tokens} tokens\n")

    # =========================================================================
    # TOKENIZATION
    # =========================================================================
    with profiler.time("1. Tokenization (encode)"):
        prompt_tokens = tokenizer.encode(prompt)

    # Allocate sequence in block manager
    seq_id = 0
    block_manager.allocate_sequence(seq_id, num_tokens=len(prompt_tokens) + max_tokens)

    # =========================================================================
    # PREFILL
    # =========================================================================
    with profiler.time("2. Prefill (full prompt)"):
        input_ids = Tensor([prompt_tokens])
        logits = model(
            input_ids,
            start_pos=0,
            kv_cache=kv_cache,
            block_manager=block_manager,
            seq_id=seq_id
        )
        logits.realize()

    # Sample first token
    with profiler.time("3. Sampling"):
        next_token = sample_tokens(logits[:, -1, :], [sampling_params], [prompt_tokens])[0]

    generated_tokens = [next_token]

    # Update position
    for _ in range(len(prompt_tokens)):
        block_manager.advance_position(seq_id)
    block_manager.advance_position(seq_id)

    # =========================================================================
    # DECODE LOOP
    # =========================================================================
    for i in range(max_tokens - 1):
        # Check for EOS
        if next_token == tokenizer.eos_id:
            break

        # --- Decode forward pass (includes attention + FFN) ---
        with profiler.time("4. Decode forward"):
            input_ids = Tensor([[next_token]])
            start_pos = len(prompt_tokens) + len(generated_tokens) - 1
            logits = model(
                input_ids,
                start_pos=start_pos,
                kv_cache=kv_cache,
                block_manager=block_manager,
                seq_id=seq_id
            )
            logits.realize()

        # --- Sampling ---
        with profiler.time("3. Sampling"):
            all_tokens = prompt_tokens + generated_tokens
            next_token = sample_tokens(logits[:, -1, :], [sampling_params], [all_tokens])[0]

        generated_tokens.append(next_token)
        block_manager.advance_position(seq_id)

    # =========================================================================
    # DETOKENIZATION
    # =========================================================================
    with profiler.time("5. Detokenization (decode)"):
        output_text = tokenizer.decode(generated_tokens)

    # Cleanup
    block_manager.free_sequence(seq_id)

    return profiler, generated_tokens, output_text


def benchmark_decode_components(
    model,
    config,
    tokenizer,
    context_len: int = 100,
    decode_steps: int = 20,
) -> BreakdownProfiler:
    """
    Detailed breakdown of decode step components:
    - Embedding lookup
    - Attention (per layer)
    - FFN (per layer)
    - Output projection
    """
    from tinyvllm.core.kv_cache import KVCache
    from tinyvllm.core.block_manager import BlockManager

    profiler = BreakdownProfiler()

    block_size = 16
    num_blocks = 100

    block_manager = BlockManager(num_gpus=1, blocks_per_gpu=num_blocks, block_size=block_size)
    kv_cache = KVCache(
        num_layers=config.n_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=config.dtype,
    )

    # Setup: prefill some context
    seq_id = 0
    block_manager.allocate_sequence(seq_id, num_tokens=context_len + decode_steps)

    # Create dummy context tokens
    context_tokens = [1] * context_len
    input_ids = Tensor([context_tokens])

    # Prefill
    print(f"Prefilling {context_len} tokens...")
    _ = model(input_ids, start_pos=0, kv_cache=kv_cache, block_manager=block_manager, seq_id=seq_id).realize()
    for _ in range(context_len):
        block_manager.advance_position(seq_id)

    print(f"Benchmarking {decode_steps} decode steps...\n")

    # Decode steps with component timing
    for step in range(decode_steps):
        token = Tensor([[1]])  # Dummy token
        start_pos = context_len + step

        # Time each component separately by instrumenting the model
        # Note: This requires access to model internals

        # Embedding
        with profiler.time("Embedding"):
            h = model.tok_embeddings(token)
            h.realize()

        # RoPE frequencies
        with profiler.time("RoPE freqs"):
            freqs_cos = model.cos[start_pos:start_pos+1]
            freqs_sin = model.sin[start_pos:start_pos+1]

        # Transformer blocks
        for layer_idx, block in enumerate(model.layers):
            # Attention
            with profiler.time(f"Layer {layer_idx} Attention"):
                # RMSNorm
                h_norm = block.attention_norm(h)
                # Attention forward (signature: x, cos, sin, kv_cache, block_manager, layer_idx, seq_id, start_pos)
                attn_out = block.attention(
                    h_norm, freqs_cos, freqs_sin,
                    kv_cache, block_manager, layer_idx, seq_id, start_pos
                )
                attn_out.realize()

            # Residual + FFN
            with profiler.time(f"Layer {layer_idx} FFN"):
                h = h + attn_out
                h_norm = block.ffn_norm(h)
                ffn_out = block.feed_forward(h_norm)
                h = h + ffn_out
                h.realize()

        # Output
        with profiler.time("Output projection"):
            h = model.norm(h)
            logits = model.output(h)
            logits.realize()

        block_manager.advance_position(seq_id)

    return profiler


def get_memory_info():
    """Get current memory usage info."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0}


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference breakdown")
    parser.add_argument("--model", type=str, default=None, help="Path to model directory")
    parser.add_argument("--prompt", type=str, default="Hello, how are you today? I would like to know more about",
                        help="Prompt to use")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs for averaging")
    parser.add_argument("--detailed", action="store_true", help="Show per-layer breakdown")
    args = parser.parse_args()

    print("=" * 80)
    print("tinyvllm Inference Breakdown Benchmark")
    print("=" * 80)
    print(f"Device: {Device.DEFAULT}")
    print(f"Runs: {args.runs} (using median for stability)")

    # Load model
    if args.model:
        print(f"\nLoading model from {args.model}...")
        model, config, tokenizer = load_real_model(args.model)
        print(f"Model: {config.dim} dim, {config.n_layers} layers")
    else:
        print("\nUsing tiny test model...")
        model, config = create_test_model()
        tokenizer = create_mock_tokenizer(config.vocab_size)
        print(f"Model: {config.dim} dim, {config.n_layers} layers")

    # Run benchmark multiple times and aggregate
    print("\n" + "=" * 80)
    print("HIGH-LEVEL BREAKDOWN")
    print("=" * 80)

    # Aggregate profiler across runs
    aggregate_profiler = BreakdownProfiler()
    tokens = None
    output = None

    for run in range(args.runs):
        print(f"\nRun {run + 1}/{args.runs}...")
        profiler, tokens, output = benchmark_inference_breakdown(
            model, config, tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
        # Merge timing stats
        for name, stat in profiler.stats.items():
            if name not in aggregate_profiler.stats:
                aggregate_profiler.stats[name] = TimingStats(name)
            for t in stat.times:
                aggregate_profiler.stats[name].record(t)

    print(f"\nGenerated {len(tokens)} tokens")
    print(f"Output: {output[:100]}...")
    print()
    print(aggregate_profiler.report())

    # Calculate derived metrics using median
    decode_stat = aggregate_profiler.stats.get("4. Decode forward", TimingStats(""))
    if decode_stat.count > 0:
        median_decode = decode_stat.median_ms
        print(f"\nDerived Metrics (median):")
        print(f"  Decode latency: {median_decode:.2f} ms/token (std: {decode_stat.std_ms:.2f})")
        print(f"  Decode throughput: {1000 / median_decode:.1f} tok/s")

    # Detailed per-layer breakdown
    if args.detailed:
        print("\n" + "=" * 80)
        print("PER-LAYER BREAKDOWN (decode only)")
        print("=" * 80)

        detail_profiler = benchmark_decode_components(
            model, config, tokenizer,
            context_len=50,
            decode_steps=10,
        )
        print()
        print(detail_profiler.report())

        # Aggregate by component type
        print("\n" + "=" * 80)
        print("AGGREGATED BY COMPONENT")
        print("=" * 80)

        aggregated = {}
        for name, stat in detail_profiler.stats.items():
            if "Attention" in name:
                key = "Attention (all layers)"
            elif "FFN" in name:
                key = "FFN (all layers)"
            else:
                key = name

            if key not in aggregated:
                aggregated[key] = 0
            aggregated[key] += stat.median_ms * stat.count

        total = sum(aggregated.values())
        print(f"{'Component':<30} {'Time (ms)':>10} {'%':>8}")
        print("-" * 50)
        for name, time_ms in sorted(aggregated.items(), key=lambda x: -x[1]):
            pct = time_ms / total * 100 if total > 0 else 0
            print(f"{name:<30} {time_ms:>10.1f} {pct:>7.1f}%")

if __name__ == "__main__":
    main()
