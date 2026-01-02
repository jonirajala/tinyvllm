"""Benchmark: Measure scalability (throughput vs batch size, max concurrent).

Run with: python -m benchmarks.bench_scalability
"""

import os
import time
from dataclasses import dataclass

from tinygrad import Tensor, Device

if "DEVICE" in os.environ:
    Device.DEFAULT = os.environ["DEVICE"]

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.engine.sampling import SamplingParams
from tinyvllm.engine.engine import LLMEngine


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


def run_scalability_benchmark():
    print("=" * 50)
    print("tinyvllm Scalability Benchmark")
    print("=" * 50)
    print(f"Device: {Device.DEFAULT}")

    config = LlamaConfig(
        dim=64, n_layers=4, n_heads=4, n_kv_heads=4,
        vocab_size=256, hidden_dim=256, max_seq_len=512
    )
    model = Llama(config)
    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    print(f"Model: dim={config.dim}, layers={config.n_layers}")

    # Warmup - run multiple times with various batch sizes
    print("\nWarming up (5 runs)...")
    for i in range(5):
        engine = LLMEngine(model, tokenizer, max_batch_size=4)
        for j in range(min(i + 1, 4)):
            engine.add_request(f"warmup {j}", SamplingParams(max_tokens=10, temperature=0.0))
        list(engine.run())

    # Throughput vs batch size
    print("\n### Throughput vs Concurrent Requests (avg of 2 runs) ###")
    print("| Requests | Tokens | Time (s) | Tok/sec | Req/sec |")
    print("|----------|--------|----------|---------|---------|")

    max_tokens = 15
    results = []
    for n_req in [1, 2, 4, 8, 16]:
        result = measure_throughput_avg(model, tokenizer, n_req, max_tokens, runs=2)
        results.append(result)
        print(f"| {result['n_requests']:8} | {result['total_tokens']:6} | {result['elapsed_sec']:8.2f} | {result['tokens_per_sec']:7.1f} | {result['requests_per_sec']:7.2f} |")

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

    for n_req in [1, 2, 4, 8, 16]:
        engine = LLMEngine(model, tokenizer, max_batch_size=n_req)
        params = SamplingParams(max_tokens=10, temperature=0.0)

        for i in range(n_req):
            engine.add_request(f"Test {i}", params)

        # Run a few steps
        for _ in range(8):
            if not engine.has_unfinished():
                break
            engine.step()

        mem_bytes = engine.kv_cache.get_memory_bytes()
        mem_kb = mem_bytes / 1024
        print(f"| {n_req:8} | {mem_kb:13.1f} KB |")


if __name__ == "__main__":
    run_scalability_benchmark()
