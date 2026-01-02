"""Benchmark: Measure KV cache memory usage.

Run with: python -m benchmarks.bench_memory
"""

from tinyvllm.model.llama import Llama
from tinyvllm.model.weights import LlamaConfig
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
    else:
        return f"{b / (1024 * 1024):.2f} MB"


def run_memory_benchmark():
    print("=" * 50)
    print("tinyvllm Memory Benchmark")
    print("=" * 50)

    config = LlamaConfig(dim=64, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=256, hidden_dim=256, max_seq_len=512)
    print(f"\nModel: dim={config.dim}, layers={config.n_layers}, kv_heads={config.n_kv_heads}, head_dim={config.head_dim}")

    # Calculate theoretical memory per token
    bytes_per_token = 2 * config.n_layers * config.n_kv_heads * config.head_dim * 4  # K+V, float32
    print(f"Theoretical KV memory per token: {format_bytes(bytes_per_token)}")

    # Show memory scaling
    print("\n" + "=" * 50)
    print("Memory Scaling")
    print("=" * 50)

    model = Llama(config)
    tokenizer = MockTokenizer(vocab_size=config.vocab_size)

    print(f"\n| Sequences | Tokens | KV Cache Memory | Bytes/Token |")
    print(f"|-----------|--------|-----------------|-------------|")

    for n_seqs in [1, 2, 5, 10]:
        engine = LLMEngine(model, tokenizer)
        params = SamplingParams(max_tokens=15, temperature=0.0)

        for i in range(n_seqs):
            engine.add_request(f"Prompt {i}", params)

        # Run 10 steps to accumulate tokens
        for _ in range(10):
            if not engine.has_unfinished():
                break
            engine.step()

        # Phase 4: Count tokens from BlockManager context lengths
        n_tokens = sum(
            engine.block_manager.get_context_length(seq_id)
            for seq_id in engine.block_manager.block_tables.keys()
        )
        # Memory is the allocated blocks * block_size * per-token storage
        num_used_blocks = engine.block_manager.blocks_per_gpu - engine.block_manager.get_num_free_blocks()
        mem_bytes = num_used_blocks * engine.block_manager.block_size * bytes_per_token
        bytes_per = mem_bytes // n_tokens if n_tokens > 0 else 0
        print(f"| {n_seqs:9} | {n_tokens:6} | {format_bytes(mem_bytes):15} | {bytes_per:11} |")


if __name__ == "__main__":
    run_memory_benchmark()
