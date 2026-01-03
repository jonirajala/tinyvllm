"""Benchmark: Measure KV cache memory usage.

Run with:
  python -m benchmarks.bench_memory                    # tiny test model
  python -m benchmarks.bench_memory --model models/tinyllama  # real model
"""

import os
import argparse
from pathlib import Path
from typing import Optional

from tinygrad import Device

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig, load_llama_weights
from tinyvllm.model.tokenizer import load_tokenizer
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine


class MockTokenizer:
    def __init__(self, vocab_size=100):
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


def format_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    elif b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f} MB"
    else:
        return f"{b / (1024 * 1024 * 1024):.2f} GB"


def run_memory_benchmark(model_path: Optional[str] = None):
    print("=" * 50)
    print("tinyvllm Memory Benchmark")
    print("=" * 50)
    print(f"Device: {Device.DEFAULT}")

    # Create or load model
    if model_path:
        print(f"\nLoading model from {model_path}...")
        config, weights = load_llama_weights(Path(model_path))
        model = create_llama(config, weights)
        tokenizer = load_tokenizer(model_path)
        total_params = sum(w.numel() for w in weights.values())
        sample_weight = next(iter(weights.values()))
        kv_dtype_size = sample_weight.dtype.itemsize
        print(f"Model: {config.dim} dim, {config.n_layers} layers, {total_params/1e9:.2f}B params")
    else:
        print("\nCreating tiny test model...")
        config = LlamaConfig(
            dim=64, n_layers=4, n_heads=4, n_kv_heads=4,
            vocab_size=256, hidden_dim=256, max_seq_len=512
        )
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        kv_dtype_size = 4  # FP32

    print(f"Config: dim={config.dim}, layers={config.n_layers}, kv_heads={config.n_kv_heads}, head_dim={config.head_dim}")

    # Calculate theoretical memory per token
    # KV cache: 2 (K+V) * n_layers * n_kv_heads * head_dim * dtype_size
    bytes_per_token = 2 * config.n_layers * config.n_kv_heads * config.head_dim * kv_dtype_size
    print(f"KV memory per token: {format_bytes(bytes_per_token)}")

    # Model weights size
    if model_path:
        model_size = sum(w.numel() * w.dtype.itemsize for w in weights.values())
        print(f"Model weights: {format_bytes(model_size)}")

    # Show memory scaling
    print("\n" + "=" * 50)
    print("KV Cache Memory Scaling")
    print("=" * 50)

    print(f"\n| Sequences | Tokens | KV Cache | Bytes/Token |")
    print(f"|-----------|--------|----------|-------------|")

    n_seqs_list = [1, 2, 5] if model_path else [1, 2, 5, 10]
    max_tokens = 10 if model_path else 15

    for n_seqs in n_seqs_list:
        engine = LLMEngine(model, tokenizer)
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        for i in range(n_seqs):
            prompt = f"Hello {i}" if model_path else f"Prompt {i}"
            engine.add_request(prompt, params)

        # Run steps to accumulate tokens
        steps = 5 if model_path else 10
        for _ in range(steps):
            if not engine.has_unfinished():
                break
            engine.step()

        # Count tokens from BlockManager context lengths
        n_tokens = sum(
            engine.block_manager.get_context_length(seq_id)
            for seq_id in engine.block_manager.block_tables.keys()
        )
        # Memory = allocated blocks * block_size * per-token storage
        num_used_blocks = engine.block_manager.blocks_per_gpu - engine.block_manager.get_num_free_blocks()
        mem_bytes = num_used_blocks * engine.block_manager.block_size * bytes_per_token
        bytes_per = mem_bytes // n_tokens if n_tokens > 0 else 0
        print(f"| {n_seqs:9} | {n_tokens:6} | {format_bytes(mem_bytes):8} | {bytes_per:11} |")

    # Memory projections
    print("\n" + "=" * 50)
    print("Memory Projections")
    print("=" * 50)

    for ctx_len in [512, 1024, 2048, 4096]:
        kv_mem = ctx_len * bytes_per_token
        print(f"Context {ctx_len}: {format_bytes(kv_mem)} KV cache per sequence")

    if config.dim < 512:
        print("\nNOTE: Small test model - use --model models/tinyllama for realistic benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory benchmark")
    parser.add_argument("--model", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_memory_benchmark(model_path=args.model)
