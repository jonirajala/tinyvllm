"""Tests for memory auto-configuration."""

import pytest
from tinyvllm.model.weights import LlamaConfig
from tinyvllm.core.memory import get_gpu_memory, estimate_model_memory, auto_num_blocks


def small_config():
    return LlamaConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=100, hidden_dim=64, max_seq_len=64)


class TestGetGPUMemory:
    def test_returns_positive_or_none(self):
        mem = get_gpu_memory()
        assert mem is None or mem > 0

    def test_reasonable_range_if_detected(self):
        mem = get_gpu_memory()
        if mem is not None:
            assert 1 * 1024**3 <= mem <= 1024 * 1024**3


class TestEstimateModelMemory:
    def test_returns_positive(self):
        assert estimate_model_memory(small_config()) > 0

    def test_scales_with_layers(self):
        config1 = LlamaConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=4, vocab_size=100, hidden_dim=64)
        config2 = LlamaConfig(dim=32, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=100, hidden_dim=64)
        assert estimate_model_memory(config2) > estimate_model_memory(config1)


class TestAutoNumBlocks:
    def test_returns_reasonable_value(self):
        num_blocks = auto_num_blocks(small_config())
        assert 10 <= num_blocks <= 500

    def test_respects_max_blocks(self):
        num_blocks = auto_num_blocks(small_config(), max_blocks=50)
        assert num_blocks <= 50


class TestEngineIntegration:
    def test_engine_auto_config(self):
        from tinyvllm.model.llama import Llama
        from tinyvllm.core.engine import LLMEngine
        from tinyvllm.core.sampling import SamplingParams

        class MockTokenizer:
            vocab_size, bos_id, eos_id, pad_id = 100, 1, 2, 0
            def encode(self, text, add_bos=True, add_eos=False):
                return [self.bos_id] + [hash(c) % 97 + 3 for c in text] if add_bos else [hash(c) % 97 + 3 for c in text]
            def decode(self, tokens):
                return "".join(chr((t % 26) + ord("a")) for t in tokens if t > 2)

        model = Llama(small_config())
        engine = LLMEngine(model, MockTokenizer(), num_blocks=None)
        engine.add_request("Test", SamplingParams(max_tokens=3, temperature=0.0))
        outputs = list(engine.run())
        assert len(outputs) == 1

    def test_engine_manual_blocks(self):
        from tinyvllm.model.llama import Llama
        from tinyvllm.core.engine import LLMEngine

        class MockTokenizer:
            vocab_size, bos_id, eos_id, pad_id = 100, 1, 2, 0
            def encode(self, text, add_bos=True, add_eos=False):
                return [self.bos_id]
            def decode(self, tokens):
                return ""

        model = Llama(small_config())
        engine = LLMEngine(model, MockTokenizer(), num_blocks=50)
        assert engine.block_manager.blocks_per_gpu == 50
