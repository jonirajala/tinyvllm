"""Benchmark: Measure latency metrics (TTFT, TPOT, E2E).

Run with: python -m benchmarks.bench_latency
"""

import os
import time
from dataclasses import dataclass
from typing import List

from tinygrad import Tensor, Device

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine


@dataclass
class LatencyResult:
    ttft_ms: float  # Time to first token
    tpot_ms: float  # Time per output token (average)
    e2e_ms: float   # End-to-end latency
    num_tokens: int


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


def measure_latency(model, tokenizer, prompt: str, max_tokens: int) -> LatencyResult:
    """Measure latency for a single request."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine = LLMEngine(model, tokenizer)
    engine.add_request(prompt, params)

    token_times = []
    start = time.perf_counter()

    while engine.has_unfinished():
        step_start = time.perf_counter()
        outputs = engine.step()
        step_end = time.perf_counter()
        token_times.append(step_end - step_start)

        if outputs:
            break

    end = time.perf_counter()

    ttft = token_times[0] * 1000 if token_times else 0
    tpot = (sum(token_times[1:]) / len(token_times[1:]) * 1000) if len(token_times) > 1 else 0
    e2e = (end - start) * 1000

    return LatencyResult(
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_ms=e2e,
        num_tokens=len(token_times)
    )


def measure_latency_detailed(model, tokenizer, prompt: str, max_tokens: int) -> LatencyResult:
    """Measure latency with per-token timing."""
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

    return LatencyResult(
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_ms=e2e,
        num_tokens=len(token_times)
    )


def run_latency_benchmark():
    print("=" * 50)
    print("tinyvllm Latency Benchmark")
    print("=" * 50)
    print(f"Device: {Device.DEFAULT}")

    config = LlamaConfig(
        dim=64, n_layers=4, n_heads=4, n_kv_heads=4,
        vocab_size=256, hidden_dim=256, max_seq_len=512
    )
    model = Llama(config)
    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    print(f"Model: dim={config.dim}, layers={config.n_layers}")

    # Warmup - run multiple times to ensure JIT is warmed
    print("\nWarming up (3 runs)...")
    for _ in range(3):
        engine = LLMEngine(model, tokenizer)
        engine.add_request("warmup prompt here", SamplingParams(max_tokens=10, temperature=0.0))
        list(engine.run())

    # Single request latency
    print("\n### Single Request Latency ###")
    print("| Prompt Len | Tokens | TTFT (ms) | TPOT (ms) | E2E (ms) |")
    print("|------------|--------|-----------|-----------|----------|")

    for prompt, max_tok in [("Hi", 10), ("Hello world test", 10), ("A" * 50, 10)]:
        result = measure_latency_detailed(model, tokenizer, prompt, max_tok)
        prompt_len = len(tokenizer.encode(prompt))
        print(f"| {prompt_len:10} | {result.num_tokens:6} | {result.ttft_ms:9.1f} | {result.tpot_ms:9.1f} | {result.e2e_ms:8.1f} |")

    # Latency vs output length
    print("\n### Latency vs Output Length ###")
    print("| Max Tokens | Actual | TTFT (ms) | TPOT (ms) | E2E (ms) |")
    print("|------------|--------|-----------|-----------|----------|")

    for max_tok in [5, 10, 20]:
        result = measure_latency_detailed(model, tokenizer, "Test prompt", max_tok)
        print(f"| {max_tok:10} | {result.num_tokens:6} | {result.ttft_ms:9.1f} | {result.tpot_ms:9.1f} | {result.e2e_ms:8.1f} |")

    # Summary
    print("\n### Summary ###")
    result = measure_latency_detailed(model, tokenizer, "Hello world", 15)
    print(f"TTFT (Time to First Token): {result.ttft_ms:.1f} ms")
    print(f"TPOT (Time Per Output Token): {result.tpot_ms:.1f} ms")
    print(f"E2E (End-to-End): {result.e2e_ms:.1f} ms for {result.num_tokens} tokens")


if __name__ == "__main__":
    run_latency_benchmark()
