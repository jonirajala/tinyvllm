"""JIT-compiled decode for tinyvllm.

Phase 7.1: TinyJit for Decode Loop

This module provides JIT-accelerated decode using TinyJit.

Two modes:
1. JIT Attention Only: Uses pure tinygrad attention kernel (JIT-compatible)
2. Full TinyJit: Wraps the compute-heavy forward pass with @TinyJit

The key insight: The custom Metal kernel bypasses tinygrad's JIT.
By using pure tinygrad ops and @TinyJit, we cache the kernel graph.
"""

from typing import List, Optional, Callable

from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from ..model.llama import Llama, apply_rope
from ..core.kv_cache import KVCache
from ..core.block_manager import BlockManager


class JitDecoder:
    """JIT-accelerated decoder.

    Uses TinyJit to cache the kernel graph for the compute-heavy
    attention + FFN operations.

    The trade-off:
    - First 2 calls: capture kernel graph (slower)
    - Subsequent calls: replay cached graph (faster)

    For decode-heavy workloads, JIT benefits outweigh warmup overhead.
    """

    def __init__(
        self,
        model: Llama,
        kv_cache: KVCache,
        max_batch_size: int = 8,
        max_context_len: int = 256,  # More reasonable default for decode
    ):
        """Initialize JIT decoder.

        Args:
            model: LLaMA model instance
            kv_cache: KVCache instance
            max_batch_size: Maximum batch size for fixed-shape JIT
            max_context_len: Maximum context length (determines memory reads).
                            Set this based on expected prompt + output length.
                            Default 256 = 16 blocks * 16 tokens/block.
        """
        self.model = model
        self.kv_cache = kv_cache
        self.max_batch = max_batch_size
        self.max_context_len = max_context_len
        # Calculate blocks needed for max_context_len
        self.max_blocks = (max_context_len + kv_cache.block_size - 1) // kv_cache.block_size

        # Model config
        self.n_heads = model.config.n_heads
        self.n_kv_heads = model.config.n_kv_heads
        self.head_dim = model.config.head_dim
        self.dim = model.config.dim
        self.block_size = kv_cache.block_size
        self.n_layers = model.config.n_layers

        # JIT-compiled forward function (created on first use)
        self._jit_forward: Optional[Callable] = None
        self._warmup_done = False

    def _create_jit_forward(self):
        """Create a JIT-compiled forward function.

        The JIT function computes attention + FFN for all layers.
        K/V cache reads are done via tensor indexing (JIT-compatible).
        """
        model = self.model
        kv_cache = self.kv_cache
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        block_size = self.block_size

        from ..kernels import fused_paged_attention_tinygrad

        @TinyJit
        def jit_forward(
            h: Tensor,                  # [max_batch, 1, dim]
            block_tables: Tensor,       # [max_batch, max_blocks]
            context_lens: Tensor,       # [max_batch]
        ) -> Tensor:
            """JIT-compiled forward pass."""
            # RoPE (same for all decode)
            cos = model.cos[0:1]
            sin = model.sin[0:1]

            for layer_idx, layer in enumerate(model.layers):
                # Attention norm
                normed = layer.attention_norm(h)

                # Q projection (K/V already written to cache)
                q = layer.attention.wq(normed).reshape(
                    self.max_batch, 1, n_heads, head_dim
                )
                q = apply_rope(q, cos, sin)

                # Get KV cache tensors
                k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)

                # JIT-compatible attention
                attn_out = fused_paged_attention_tinygrad(
                    q, k_cache, v_cache,
                    block_tables, context_lens,
                    n_heads, n_kv_heads, head_dim, block_size
                )

                # Output projection + residual
                attn_out = attn_out.reshape(self.max_batch, 1, -1)
                h = h + layer.attention.wo(attn_out)

                # FFN with residual
                h = h + layer.feed_forward(layer.ffn_norm(h))

            # Output projection
            return model.output(model.norm(h))

        return jit_forward

    def decode(
        self,
        block_manager: BlockManager,
        tokens_list: List[int],
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tensor:
        """JIT-accelerated batched decode.

        Uses TinyJit to cache the kernel graph after warmup.

        Args:
            block_manager: BlockManager instance
            tokens_list: List of token IDs (one per sequence)
            seq_ids: List of sequence IDs
            start_positions: Start position for each sequence

        Returns:
            logits: [batch, 1, vocab_size]
        """
        # Create JIT function on first use
        if self._jit_forward is None:
            self._jit_forward = self._create_jit_forward()

        batch_size = len(seq_ids)

        # Pad tokens to max_batch (JIT needs fixed shapes)
        tokens_padded = tokens_list + [0] * (self.max_batch - batch_size)
        tokens = Tensor(tokens_padded, dtype=dtypes.int32).reshape(self.max_batch, 1)

        # Embedding
        h = self.model.tok_embeddings(tokens)

        # RoPE for KV
        cos = self.model.cos[0:1]
        sin = self.model.sin[0:1]

        # Write K/V to blocks BEFORE JIT forward (needs Python flexibility)
        for layer_idx, layer in enumerate(self.model.layers):
            # Compute K/V for this layer
            normed = layer.attention_norm(h)
            k = layer.attention.wk(normed).reshape(
                self.max_batch, 1, self.n_kv_heads, self.head_dim
            )
            v = layer.attention.wv(normed).reshape(
                self.max_batch, 1, self.n_kv_heads, self.head_dim
            )
            k = apply_rope(k, cos, sin)

            # Write only actual sequences (not padding)
            k_realized = k.realize()
            v_realized = v.realize()
            for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
                self._write_kv_single(
                    layer_idx, seq_id, start_pos, block_manager,
                    k_realized[i, 0], v_realized[i, 0]
                )

        # Realize KV cache after all writes (single sync instead of per-write)
        for layer_idx in range(self.n_layers):
            k_cache, v_cache = self.kv_cache.get_cache_tensors(layer_idx)
            k_cache.realize()
            v_cache.realize()

        # Prepare block tables tensor (fixed shape for JIT)
        bt_tensor, ctx_tensor = self._prepare_tensors_padded(
            block_manager, seq_ids, start_positions
        )

        # JIT forward pass (cached after warmup)
        h = h.contiguous().realize()
        bt_tensor = bt_tensor.contiguous().realize()
        ctx_tensor = ctx_tensor.contiguous().realize()

        logits = self._jit_forward(h, bt_tensor, ctx_tensor)

        # Advance positions
        for seq_id in seq_ids:
            block_manager.advance_position(seq_id)

        # Return only actual batch (remove padding)
        return logits[:batch_size]

    def _prepare_tensors_padded(
        self,
        block_manager: BlockManager,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> tuple:
        """Prepare block tables and context lens as padded tensors."""
        batch_size = len(seq_ids)

        # Initialize with zeros (padding)
        block_tables = [[0] * self.max_blocks for _ in range(self.max_batch)]
        context_lens = [1] * self.max_batch  # Default 1 to avoid issues

        for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
            bt = block_manager.get_block_table(seq_id)
            block_tables[i][:len(bt)] = bt
            context_lens[i] = start_pos + 1  # +1 for current token

        bt_tensor = Tensor(block_tables, dtype=dtypes.int32)
        ctx_tensor = Tensor(context_lens, dtype=dtypes.int32)

        return bt_tensor, ctx_tensor

    def _write_kv_single(
        self,
        layer_idx: int,
        seq_id: int,
        start_pos: int,
        block_manager: BlockManager,
        k: Tensor,
        v: Tensor,
    ) -> None:
        """Write single K/V token to blocks."""
        block_table = block_manager.get_block_table(seq_id)

        block_idx = start_pos // self.block_size
        offset = start_pos % self.block_size

        # Allocate new block if needed (first layer only)
        if block_idx >= len(block_table) and layer_idx == 0:
            gpu_id = block_manager.get_gpu_for_seq(seq_id)
            if len(block_manager.free_blocks[gpu_id]) == 0:
                raise RuntimeError("Out of KV cache memory!")
            new_block = block_manager.free_blocks[gpu_id].pop()
            block_manager.ref_counts[gpu_id][new_block] = 1
            block_manager.block_tables[seq_id].append(new_block)
            block_table = block_manager.get_block_table(seq_id)

        block_id = block_table[block_idx]
        self.kv_cache.write_kv(layer_idx, block_id, offset, k, v)
