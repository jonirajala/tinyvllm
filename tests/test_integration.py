"""Integration tests for the full inference pipeline."""

import pytest
from tinygrad import Tensor, dtypes

from tinyvllm.model.llama import Llama, create_llama
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.sampling import SamplingParams
from tinyvllm.core.engine import LLMEngine, generate_batch


# Small config for fast tests
def small_config():
    return LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=100,
        hidden_dim=64,
        max_seq_len=64,
    )


class MockTokenizer:
    """Mock tokenizer for unit tests."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False):
        if not text:
            return []
        # Simple encoding: hash each char to token ID
        tokens = [hash(c) % (self.vocab_size - 3) + 3 for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens):
        # Simple decoding: map to letters
        return "".join(chr((t % 26) + ord("a")) for t in tokens if t > 2)


class TestEndToEndPipeline:
    """Test the full pipeline from prompt to output."""

    def test_full_pipeline_with_mock(self):
        """Test complete pipeline with mock tokenizer and random weights."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=10, temperature=0.8)

        engine = LLMEngine(model, tokenizer)
        engine.add_request("Hello world", params)
        outputs = list(engine.run())

        assert len(outputs) == 1
        assert isinstance(outputs[0].text, str)

    def test_greedy_decoding_deterministic(self):
        """Greedy decoding should produce deterministic results."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=5, temperature=0.0)

        engine1 = LLMEngine(model, tokenizer)
        engine1.add_request("Same prompt", params)
        result1 = list(engine1.run())[0].text

        engine2 = LLMEngine(model, tokenizer)
        engine2.add_request("Same prompt", params)
        result2 = list(engine2.run())[0].text

        assert result1 == result2

    def test_kv_cache_consistency(self):
        """KV cache should produce consistent results."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=10, temperature=0.0)

        # Run twice - should get same result
        engine1 = LLMEngine(model, tokenizer)
        engine1.add_request("Test", params)
        result1 = list(engine1.run())[0].text

        engine2 = LLMEngine(model, tokenizer)
        engine2.add_request("Test", params)
        result2 = list(engine2.run())[0].text

        assert result1 == result2

    def test_various_temperatures(self):
        """Test generation at different temperatures."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)

        for temp in [0.0, 0.5, 1.0, 1.5]:
            params = SamplingParams(max_tokens=5, temperature=temp)
            engine = LLMEngine(model, tokenizer)
            engine.add_request("Test", params)
            outputs = list(engine.run())
            assert len(outputs) == 1

    def test_various_sampling_params(self):
        """Test generation with various sampling configurations."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)

        configs = [
            {"temperature": 0.0, "top_k": 0, "top_p": 1.0},  # Greedy
            {"temperature": 1.0, "top_k": 10, "top_p": 1.0},  # Top-k only
            {"temperature": 1.0, "top_k": 0, "top_p": 0.9},  # Top-p only
            {"temperature": 0.8, "top_k": 40, "top_p": 0.95},  # Combined
            {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.2},
        ]

        for cfg in configs:
            params = SamplingParams(max_tokens=5, **cfg)
            engine = LLMEngine(model, tokenizer)
            engine.add_request("Test input", params)
            outputs = list(engine.run())
            assert len(outputs) == 1

    def test_batch_generation(self):
        """Test batch generation with multiple prompts."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=5, temperature=0.0)

        engine = LLMEngine(model, tokenizer)
        results = generate_batch(engine, ["Hello", "World", "Test"], params)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestModelWeightLoading:
    """Test weight loading and model creation (Phase 4)."""

    def test_create_model_without_weights(self):
        """Model should work with random weights."""
        config = small_config()
        model = Llama(config)

        # Phase 4: Create BlockManager and KVCache
        from tinyvllm.core.block_manager import BlockManager
        from tinyvllm.core.kv_cache import KVCache

        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=10,
            block_size=16,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        tokens = Tensor([[1, 2, 3, 4]])
        block_manager.allocate_sequence(seq_id=0, num_tokens=4)
        logits = model.prefill(tokens, start_pos=0, kv_cache=kv_cache,
                               block_manager=block_manager, seq_id=0)

        assert logits.shape == (1, 4, config.vocab_size)
        # Check BlockManager has tracked context length
        assert block_manager.get_context_length(seq_id=0) == 4

    def test_create_model_with_weights(self):
        """Test create_llama with weight dict."""
        config = small_config()

        # Create mock weights
        weights = {
            "model.embed_tokens.weight": Tensor.randn(config.vocab_size, config.dim),
            "model.norm.weight": Tensor.ones(config.dim),
            "lm_head.weight": Tensor.randn(config.vocab_size, config.dim),
        }

        # Add layer weights
        for i in range(config.n_layers):
            prefix = f"model.layers.{i}."
            weights[f"{prefix}self_attn.q_proj.weight"] = Tensor.randn(
                config.n_heads * config.head_dim, config.dim
            )
            weights[f"{prefix}self_attn.k_proj.weight"] = Tensor.randn(
                config.n_kv_heads * config.head_dim, config.dim
            )
            weights[f"{prefix}self_attn.v_proj.weight"] = Tensor.randn(
                config.n_kv_heads * config.head_dim, config.dim
            )
            weights[f"{prefix}self_attn.o_proj.weight"] = Tensor.randn(
                config.dim, config.n_heads * config.head_dim
            )
            weights[f"{prefix}mlp.gate_proj.weight"] = Tensor.randn(config.hidden_dim, config.dim)
            weights[f"{prefix}mlp.up_proj.weight"] = Tensor.randn(config.hidden_dim, config.dim)
            weights[f"{prefix}mlp.down_proj.weight"] = Tensor.randn(config.dim, config.hidden_dim)
            weights[f"{prefix}input_layernorm.weight"] = Tensor.ones(config.dim)
            weights[f"{prefix}post_attention_layernorm.weight"] = Tensor.ones(config.dim)

        model = create_llama(config, weights)

        # Phase 4: Create BlockManager and KVCache
        from tinyvllm.core.block_manager import BlockManager
        from tinyvllm.core.kv_cache import KVCache

        block_manager = BlockManager(num_gpus=1, blocks_per_gpu=10, block_size=16)
        kv_cache = KVCache(
            num_layers=config.n_layers,
            num_blocks=10,
            block_size=16,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=dtypes.float32,
        )

        tokens = Tensor([[1, 2, 3]])
        block_manager.allocate_sequence(seq_id=0, num_tokens=3)
        logits = model.prefill(tokens, kv_cache=kv_cache, block_manager=block_manager, seq_id=0)

        assert logits.shape == (1, 3, config.vocab_size)


class TestTokenizerIntegration:
    """Test tokenizer with model."""

    def test_encode_decode_roundtrip_mock(self):
        """Encode and decode should work together."""
        tokenizer = MockTokenizer()

        text = "Hello"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        # Mock tokenizer won't perfectly roundtrip, but should produce output
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_tokenizer_with_model_vocab(self):
        """Tokenizer output should be valid for model vocab size."""
        config = small_config()
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)

        tokens = tokenizer.encode("Test text")

        # All tokens should be valid indices
        assert all(0 <= t < config.vocab_size for t in tokens)


class TestEdgeCases:
    """Test edge cases in the pipeline."""

    def test_single_token_generation(self):
        """Should handle generating just one token."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=1, temperature=0.0)

        engine = LLMEngine(model, tokenizer)
        engine.add_request("Test", params)
        outputs = list(engine.run())

        assert len(outputs) == 1
        assert len(outputs[0].tokens) == 1

    def test_longer_prompt(self):
        """Should handle prompts with multiple tokens."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=5, temperature=0.8)

        # Keep within max_seq_len (64 tokens)
        longer_prompt = "This is a longer test prompt"
        engine = LLMEngine(model, tokenizer)
        engine.add_request(longer_prompt, params)
        outputs = list(engine.run())

        assert len(outputs) == 1

    def test_multiple_concurrent_requests(self):
        """Test handling multiple requests concurrently."""
        config = small_config()
        model = Llama(config)
        tokenizer = MockTokenizer(vocab_size=config.vocab_size)
        params = SamplingParams(max_tokens=3, temperature=0.0)

        engine = LLMEngine(model, tokenizer)
        engine.add_request("First", params)
        engine.add_request("Second", params)
        engine.add_request("Third", params)

        outputs = list(engine.run())

        assert len(outputs) == 3
        request_ids = {o.request_id for o in outputs}
        assert request_ids == {0, 1, 2}
