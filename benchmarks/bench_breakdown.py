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
    def min_ms(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_ms(self) -> float:
        return max(self.times) if self.times else 0


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
        """Generate a timing report."""
        lines = []
        total = sum(s.total_ms for s in self.stats.values())

        lines.append("=" * 70)
        lines.append("TIMING BREAKDOWN")
        lines.append("=" * 70)
        lines.append(f"{'Operation':<30} {'Total (ms)':>10} {'Count':>8} {'Avg (ms)':>10} {'%':>8}")
        lines.append("-" * 70)

        # Sort by total time descending
        sorted_stats = sorted(self.stats.values(), key=lambda s: s.total_ms, reverse=True)

        for stat in sorted_stats:
            pct = (stat.total_ms / total * 100) if total > 0 else 0
            lines.append(f"{stat.name:<30} {stat.total_ms:>10.1f} {stat.count:>8} {stat.avg_ms:>10.2f} {pct:>7.1f}%")

        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<30} {total:>10.1f}")
        lines.append("=" * 70)

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
    warmup_runs: int = 2,
) -> BreakdownProfiler:
    """
    Benchmark each step of inference with detailed timing.
    """
    from tinyvllm.core.kv_cache import KVCache
    from tinyvllm.core.block_manager import BlockManager
    from tinyvllm.engine.sampling import SamplingParams, sample_tokens

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
        next_token = sample_tokens(logits[0, -1, :], sampling_params, prompt_tokens)[0]

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
            next_token = sample_tokens(logits[0, -1, :], sampling_params, all_tokens)[0]

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
    parser.add_argument("--detailed", action="store_true", help="Show per-layer breakdown")
    args = parser.parse_args()

    print("=" * 70)
    print("tinyvllm Inference Breakdown Benchmark")
    print("=" * 70)
    print(f"Device: {Device.DEFAULT}")

    # Memory before loading
    mem_before = get_memory_info()
    print(f"Memory before: {mem_before['rss_mb']:.1f} MB")

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

    # Memory after loading
    mem_after = get_memory_info()
    print(f"Memory after load: {mem_after['rss_mb']:.1f} MB (+{mem_after['rss_mb'] - mem_before['rss_mb']:.1f} MB)")

    # Run main benchmark
    print("\n" + "=" * 70)
    print("HIGH-LEVEL BREAKDOWN")
    print("=" * 70)

    profiler, tokens, output = benchmark_inference_breakdown(
        model, config, tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )

    print(f"\nGenerated {len(tokens)} tokens")
    print(f"Output: {output[:100]}...")
    print()
    print(profiler.report())

    # Calculate derived metrics
    total_time = sum(s.total_ms for s in profiler.stats.values())
    decode_time = profiler.stats.get("4. Decode forward", TimingStats("")).total_ms
    num_decode_steps = profiler.stats.get("4. Decode forward", TimingStats("")).count

    if num_decode_steps > 0:
        print(f"\nDerived Metrics:")
        print(f"  Total time: {total_time:.1f} ms")
        print(f"  Decode steps: {num_decode_steps}")
        print(f"  Avg decode latency: {decode_time / num_decode_steps:.2f} ms/token")
        print(f"  Decode throughput: {num_decode_steps / (decode_time / 1000):.1f} tok/s")

    # Detailed per-layer breakdown
    if args.detailed:
        print("\n" + "=" * 70)
        print("PER-LAYER BREAKDOWN (decode only)")
        print("=" * 70)

        detail_profiler = benchmark_decode_components(
            model, config, tokenizer,
            context_len=50,
            decode_steps=10,
        )
        print()
        print(detail_profiler.report())

        # Aggregate by component type
        print("\n" + "=" * 70)
        print("AGGREGATED BY COMPONENT")
        print("=" * 70)

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
            aggregated[key] += stat.total_ms

        total = sum(aggregated.values())
        print(f"{'Component':<30} {'Time (ms)':>10} {'%':>8}")
        print("-" * 50)
        for name, time_ms in sorted(aggregated.items(), key=lambda x: -x[1]):
            pct = time_ms / total * 100 if total > 0 else 0
            print(f"{name:<30} {time_ms:>10.1f} {pct:>7.1f}%")

    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    # Identify bottleneck
    sorted_stats = sorted(profiler.stats.items(), key=lambda x: -x[1].total_ms)
    top = sorted_stats[0] if sorted_stats else None

    if top:
        print(f"\nPrimary bottleneck: {top[0]}")
        print(f"  Time: {top[1].total_ms:.1f} ms ({top[1].total_ms / total_time * 100:.1f}% of total)")

        if "Prefill" in top[0]:
            print("\n  Suggestion: Prefill is dominant. For long prompts, consider:")
            print("    - Chunked prefill (process prompt in chunks)")
            print("    - Flash attention for prefill")
            print("    - Prefix caching for repeated prompts")
        elif "Decode" in top[0]:
            print("\n  Suggestion: Decode is dominant. Consider:")
            print("    - Speculative decoding (draft + verify)")
            print("    - Model quantization (INT8/INT4)")
            print("    - Larger batch sizes for throughput")
        elif "Sampling" in top[0]:
            print("\n  Suggestion: Sampling is slow. Consider:")
            print("    - GPU-based sampling (avoid CPU sync)")
            print("    - Batched sampling")


if __name__ == "__main__":
    main()
