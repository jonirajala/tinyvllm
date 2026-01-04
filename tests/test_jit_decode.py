"""Tests for JIT decode implementation (Phase 7.1)."""

import pytest
from tinygrad import Tensor, dtypes

from tinyvllm.kernels.paged_attention_tinygrad import fused_paged_attention_tinygrad


class TestJitPagedAttention:
    """Test JIT-compatible paged attention kernel."""

    def test_basic_attention(self):
        """Test basic attention computation."""
        batch_size = 2
        n_heads = 4
        n_kv_heads = 2
        head_dim = 64
        block_size = 16
        num_blocks = 10
        max_blocks = 4

        # Create test inputs
        queries = Tensor.randn(batch_size, 1, n_heads, head_dim).realize()
        k_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim).realize()
        v_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim).realize()

        # Block tables: each sequence uses different blocks
        block_tables = Tensor([[0, 1, 0, 0], [2, 3, 0, 0]], dtype=dtypes.int32).realize()
        context_lens = Tensor([20, 25], dtype=dtypes.int32).realize()

        # Run attention
        output = fused_paged_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        # Check output shape
        assert output.shape == (batch_size, 1, n_heads, head_dim)

        # Realize and check output is not all zeros
        output = output.realize()
        max_val = output.abs().max().realize().item()
        assert max_val > 0

    def test_gqa_head_expansion(self):
        """Test GQA (grouped query attention) head expansion."""
        batch_size = 1
        n_heads = 8
        n_kv_heads = 2  # 4x expansion
        head_dim = 32
        block_size = 8
        num_blocks = 5
        max_blocks = 2

        queries = Tensor.randn(batch_size, 1, n_heads, head_dim)
        k_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim)
        v_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim)
        block_tables = Tensor([[0, 1]], dtype=dtypes.int32)
        context_lens = Tensor([10], dtype=dtypes.int32)

        output = fused_paged_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        assert output.shape == (batch_size, 1, n_heads, head_dim)

    def test_single_sequence(self):
        """Test with single sequence."""
        batch_size = 1
        n_heads = 4
        n_kv_heads = 4
        head_dim = 64
        block_size = 16
        num_blocks = 5
        max_blocks = 2

        queries = Tensor.randn(batch_size, 1, n_heads, head_dim)
        k_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim)
        v_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim)
        block_tables = Tensor([[0, 1]], dtype=dtypes.int32)
        context_lens = Tensor([20], dtype=dtypes.int32)

        output = fused_paged_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens,
            n_heads, n_kv_heads, head_dim, block_size
        )

        assert output.shape == (batch_size, 1, n_heads, head_dim)

    def test_masking(self):
        """Test that context length masking works."""
        batch_size = 2
        n_heads = 2
        n_kv_heads = 2
        head_dim = 32
        block_size = 8
        num_blocks = 5
        max_blocks = 2

        queries = Tensor.randn(batch_size, 1, n_heads, head_dim).realize()
        k_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim).realize()
        v_cache = Tensor.randn(num_blocks, block_size, n_kv_heads, head_dim).realize()
        block_tables = Tensor([[0, 1], [2, 3]], dtype=dtypes.int32).realize()

        # Different context lengths - should produce different outputs
        context_lens_1 = Tensor([5, 10], dtype=dtypes.int32).realize()
        context_lens_2 = Tensor([10, 5], dtype=dtypes.int32).realize()

        output_1 = fused_paged_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens_1,
            n_heads, n_kv_heads, head_dim, block_size
        ).realize()

        output_2 = fused_paged_attention_tinygrad(
            queries, k_cache, v_cache,
            block_tables, context_lens_2,
            n_heads, n_kv_heads, head_dim, block_size
        ).realize()

        # Outputs should be different due to different masking
        diff = (output_1 - output_2).abs().max().realize().item()
        assert diff > 0.01, "Outputs should differ with different context lengths"


class TestJitDecoder:
    """Test JitDecoder integration."""

    @pytest.fixture
    def setup_model(self):
        """Set up a minimal model for testing."""
        from tinyvllm.model.weights import LlamaConfig
        from tinyvllm.model.llama import Llama
        from tinyvllm.core.kv_cache import KVCache
        from tinyvllm.core.block_manager import BlockManager
        from tinyvllm.engine.jit_decode import JitDecoder

        # Minimal config for testing (head_dim = dim / n_heads = 256 / 4 = 64)
        config = LlamaConfig(
            dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            hidden_dim=512,
            max_seq_len=128,
            norm_eps=1e-5,
            rope_theta=10000.0,
        )

        model = Llama(config)
        kv_cache = KVCache(
            num_layers=2,
            num_blocks=20,
            block_size=16,
            n_kv_heads=2,
            head_dim=config.head_dim,
            dtype=config.dtype,
        )
        block_manager = BlockManager(
            num_gpus=1,
            blocks_per_gpu=20,
            block_size=16,
        )
        jit_decoder = JitDecoder(
            model=model,
            kv_cache=kv_cache,
            max_batch_size=4,
            max_context_len=160,  # 10 blocks * 16 tokens/block
        )

        return model, kv_cache, block_manager, jit_decoder

    def test_jit_decoder_single_sequence(self, setup_model):
        """Test JitDecoder with single sequence."""
        model, kv_cache, block_manager, jit_decoder = setup_model

        # Allocate sequence with 5 initial tokens
        seq_id = 0
        block_manager.allocate_sequence(seq_id, num_tokens=5)

        # Simulate having processed some tokens (prefill)
        # Write some dummy KV for context
        for layer_idx in range(model.config.n_layers):
            block_table = block_manager.get_block_table(seq_id)
            for pos in range(5):
                block_idx = pos // block_manager.block_size
                offset = pos % block_manager.block_size
                block_id = block_table[block_idx]
                k = Tensor.randn(model.config.n_kv_heads, model.config.head_dim)
                v = Tensor.randn(model.config.n_kv_heads, model.config.head_dim)
                kv_cache.write_kv(layer_idx, block_id, offset, k, v)

        # Update block manager position
        block_manager.seq_positions[seq_id] = 5

        # Decode
        logits = jit_decoder.decode(
            block_manager=block_manager,
            tokens_list=[42],  # dummy token
            seq_ids=[seq_id],
            start_positions=[5],
        )

        # Check output shape
        assert logits.shape == (1, 1, model.config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
