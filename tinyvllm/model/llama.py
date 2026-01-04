"""LLaMA model implementation using tinygrad.


"""

from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding, Linear

from .weights import LlamaConfig
from ..core.kv_cache import KVCache
from ..core.block_manager import BlockManager
from ..core.attention_utils import prefill_attention, decode_attention



class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[Tensor, Tensor]:
    """Precompute rotary position embedding frequencies."""
    # Compute frequencies: theta_i = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2).float() / dim))
    # Compute position * frequency
    t = Tensor.arange(seq_len).float()
    freqs = t.unsqueeze(1) * freqs.unsqueeze(0)  # [seq_len, dim/2]
    # Return cos and sin
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings to queries or keys."""
    # x: [batch, seq, heads, head_dim]
    # cos, sin: [seq, head_dim/2]
    batch, seq, heads, head_dim = x.shape

    # Reshape x to pairs: [batch, seq, heads, head_dim/2, 2]
    x_reshape = x.reshape(batch, seq, heads, head_dim // 2, 2)
    x0, x1 = x_reshape[:, :, :, :, 0], x_reshape[:, :, :, :, 1]

    # Get cos/sin for this sequence length
    cos = cos[:seq].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim/2]
    sin = sin[:seq].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim/2]

    # Apply rotation: (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # Interleave back
    out = Tensor.stack([out0, out1], dim=-1).reshape(batch, seq, heads, head_dim)
    return out


class Attention:
    """Multi-head attention with RoPE and KV cache support.

    Phase 4: Uses BlockManager for slot allocation and block-based KVCache.
    """

    def __init__(self, config: LlamaConfig):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

        # Projections
        self.wq = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = Linear(config.n_heads * config.head_dim, config.dim, bias=False)

    def __call__(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        layer_idx: int,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Forward pass with block-based KVCache.

        Phase 4: Uses BlockManager to get slot for writing K/V.
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Phase 4: Write K/V to blocks using BlockManager slots
        # For prefill: write all tokens at once
        # For decode: write single token
        self._write_kv_to_blocks(
            kv_cache, block_manager, layer_idx, seq_id, k[0], v[0], start_pos
        )

        # Get block table for attention
        block_table = block_manager.get_block_table(seq_id)
        # Context length is start_pos + current tokens being processed
        context_len = start_pos + seq_len

        # Compute attention reading from scattered blocks
        out = prefill_attention(
            q, kv_cache, block_table, context_len, layer_idx, start_pos
        )

        # Project output
        out = out.reshape(batch, seq_len, -1)
        return self.wo(out)

    def _write_kv_to_blocks(
        self,
        kv_cache: KVCache,
        block_manager: BlockManager,
        layer_idx: int,
        seq_id: int,
        k: Tensor,
        v: Tensor,
        start_pos: int,
    ) -> None:
        """Write K/V to block-based cache using BlockManager slots.

        Args:
            kv_cache: Block-based KV cache
            block_manager: For slot allocation
            layer_idx: Which layer
            seq_id: Sequence ID
            k: Keys [seq_len, n_kv_heads, head_dim]
            v: Values [seq_len, n_kv_heads, head_dim]
            start_pos: Starting position in sequence
        """
        seq_len = k.shape[0]
        block_size = block_manager.block_size
        block_table = block_manager.get_block_table(seq_id)

        for i in range(seq_len):
            # Calculate position for this token
            pos = start_pos + i
            block_idx = pos // block_size
            offset = pos % block_size

            # Check if we need a new block (only allocate in first layer)
            if block_idx >= len(block_table) and layer_idx == 0:
                # Need to allocate a new block
                gpu_id = block_manager.get_gpu_for_seq(seq_id)
                if len(block_manager.free_blocks[gpu_id]) == 0:
                    raise RuntimeError("Out of KV cache memory!")
                new_block = block_manager.free_blocks[gpu_id].pop()
                block_manager.ref_counts[gpu_id][new_block] = 1
                block_manager.block_tables[seq_id].append(new_block)
                # Refresh block_table reference
                block_table = block_manager.get_block_table(seq_id)

            # Get physical block ID from block table
            block_id = block_table[block_idx]

            # Write single token K/V to the block
            kv_cache.write_kv(layer_idx, block_id, offset, k[i], v[i])

    def batched_forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        layer_idx: int,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tensor:
        """Batched forward for decode (single token per sequence).

        Args:
            x: [batch, 1, dim] - one token per sequence
            cos, sin: RoPE embeddings
            kv_cache: Block-based KV cache
            block_manager: For slot allocation
            layer_idx: Which layer
            seq_ids: List of sequence IDs
            start_positions: Start position for each sequence

        Returns:
            output: [batch, 1, dim]
        """
        batch, seq_len, _ = x.shape
        assert seq_len == 1, "Batched forward only for decode (seq_len=1)"

        # Project to Q, K, V - batched
        q = self.wq(x).reshape(batch, 1, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, 1, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, 1, self.n_kv_heads, self.head_dim)

        # Apply RoPE - batched
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write K/V and compute attention for each sequence
        queries = []
        block_tables = []
        context_lens = []

        for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
            # Write this sequence's K/V to its blocks
            self._write_kv_to_blocks(
                kv_cache, block_manager, layer_idx, seq_id,
                k[i], v[i], start_pos
            )

            # Collect info for batched attention
            queries.append(q[i:i+1])  # Keep batch dim
            block_tables.append(block_manager.get_block_table(seq_id))
            context_lens.append(start_pos + 1)

        # Batched attention
        out = decode_attention(
            queries, kv_cache, block_tables, context_lens,
            layer_idx, start_positions
        )

        # Project output - batched
        out = out.reshape(batch, 1, -1)
        return self.wo(out)


class FeedForward:
    """SwiGLU feed-forward network."""

    def __init__(self, config: LlamaConfig):
        self.w1 = Linear(config.dim, config.hidden_dim, bias=False)  # Gate
        self.w2 = Linear(config.hidden_dim, config.dim, bias=False)  # Down
        self.w3 = Linear(config.dim, config.hidden_dim, bias=False)  # Up

    def __call__(self, x: Tensor) -> Tensor:
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.w2(self.w1(x).silu() * self.w3(x))


class TransformerBlock:
    """Single transformer block with attention and FFN."""

    def __init__(self, config: LlamaConfig):
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def __call__(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        layer_idx: int,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Forward pass through transformer block."""
        # Attention with residual
        h = self.attention(
            self.attention_norm(x), cos, sin, kv_cache, block_manager,
            layer_idx, seq_id, start_pos
        )
        x = x + h
        # FFN with residual
        return x + self.feed_forward(self.ffn_norm(x))

    def batched_forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        layer_idx: int,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tensor:
        """Batched forward for decode (single token per sequence)."""
        # Attention with residual - batched
        h = self.attention.batched_forward(
            self.attention_norm(x), cos, sin, kv_cache, block_manager,
            layer_idx, seq_ids, start_positions
        )
        x = x + h
        # FFN with residual - batched (naturally handles batch dim)
        return x + self.feed_forward(self.ffn_norm(x))


class Llama:
    """LLaMA model.

    Phase 4: Uses BlockManager for slot allocation and block-based KVCache.
    """

    def __init__(self, config: LlamaConfig):
        self.config = config

        # Token embedding
        self.tok_embeddings = Embedding(config.vocab_size, config.dim)

        # Transformer blocks
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]

        # Output
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = Linear(config.dim, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len * 2, config.rope_theta
        )

    def __call__(
        self,
        tokens: Tensor,
        start_pos: int = 0,
        kv_cache: KVCache = None,
        block_manager: BlockManager = None,
        seq_id: int = 0,
    ) -> Tensor:
        """
        Forward pass through the model.

        Phase 4: Uses BlockManager for slot allocation.

        Args:
            tokens: Input token IDs [batch, seq_len]
            start_pos: Position in sequence (for generation with cache)
            kv_cache: KVCache instance for paged attention
            block_manager: BlockManager for slot allocation (Phase 4)
            seq_id: Sequence ID for KVCache

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        # Get RoPE for this position range
        cos = self.cos[start_pos : start_pos + seq_len]
        sin = self.sin[start_pos : start_pos + seq_len]

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache, block_manager, layer_idx, seq_id, start_pos)

        # Phase 4: After all layers, advance position for each token processed
        if block_manager is not None:
            for _ in range(seq_len):
                block_manager.advance_position(seq_id)

        # Output projection
        return self.output(self.norm(h))

    def batched_decode(
        self,
        tokens: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        seq_ids: List[int],
        start_positions: List[int],
    ) -> Tensor:
        """
        Batched decode forward pass for multiple sequences.

        Phase 4: Process multiple decode sequences in one forward pass.
        Each sequence generates one token but we batch the computation.

        Args:
            tokens: Input token IDs [batch, 1] - one token per sequence
            kv_cache: KVCache instance for paged attention
            block_manager: BlockManager for slot allocation
            seq_ids: List of sequence IDs
            start_positions: Start position for each sequence

        Returns:
            logits: Output logits [batch, 1, vocab_size]
        """
        batch, seq_len = tokens.shape
        assert seq_len == 1, "batched_decode only for single token decode"

        h = self.tok_embeddings(tokens)

        # All decode positions use the same relative position (0) for RoPE
        # But each sequence has a different absolute position
        # We use start_position for each sequence
        cos = self.cos[0:1]  # Position 0 relative to current token
        sin = self.sin[0:1]

        # Forward through layers with batched decode
        for layer_idx, layer in enumerate(self.layers):
            h = layer.batched_forward(
                h, cos, sin, kv_cache, block_manager,
                layer_idx, seq_ids, start_positions
            )

        # Advance positions for all sequences
        for seq_id in seq_ids:
            block_manager.advance_position(seq_id)

        # Output projection - batched
        return self.output(self.norm(h))

    def load_weights(self, weights: Dict[str, Tensor]):
        """Load pretrained weights into the model.

        Weights are cast to config.dtype (auto-detected or explicit).
        Uses replace() to change dtype, not assign() which preserves target dtype.
        """
        # Map weight names to model attributes
        weight_map = self._build_weight_map()
        target_dtype = self.config.dtype

        for name, tensor in weights.items():
            target_name = name
            if name not in weight_map:
                # Try common name variations
                target_name = self._map_weight_name(name)

            if target_name in weight_map:
                target = weight_map[target_name]
                # Use replace() to change both data and dtype
                target.replace(tensor.cast(target_dtype))

    def _build_weight_map(self) -> Dict[str, Tensor]:
        """Build mapping from weight names to tensors."""
        weight_map = {}

        # Embeddings
        weight_map["model.embed_tokens.weight"] = self.tok_embeddings.weight
        weight_map["tok_embeddings.weight"] = self.tok_embeddings.weight

        # Layers
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            alt_prefix = f"layers.{i}."

            for p in [prefix, alt_prefix]:
                # Attention
                weight_map[f"{p}self_attn.q_proj.weight"] = layer.attention.wq.weight
                weight_map[f"{p}self_attn.k_proj.weight"] = layer.attention.wk.weight
                weight_map[f"{p}self_attn.v_proj.weight"] = layer.attention.wv.weight
                weight_map[f"{p}self_attn.o_proj.weight"] = layer.attention.wo.weight
                weight_map[f"{p}attention.wq.weight"] = layer.attention.wq.weight
                weight_map[f"{p}attention.wk.weight"] = layer.attention.wk.weight
                weight_map[f"{p}attention.wv.weight"] = layer.attention.wv.weight
                weight_map[f"{p}attention.wo.weight"] = layer.attention.wo.weight

                # FFN
                weight_map[f"{p}mlp.gate_proj.weight"] = layer.feed_forward.w1.weight
                weight_map[f"{p}mlp.down_proj.weight"] = layer.feed_forward.w2.weight
                weight_map[f"{p}mlp.up_proj.weight"] = layer.feed_forward.w3.weight
                weight_map[f"{p}feed_forward.w1.weight"] = layer.feed_forward.w1.weight
                weight_map[f"{p}feed_forward.w2.weight"] = layer.feed_forward.w2.weight
                weight_map[f"{p}feed_forward.w3.weight"] = layer.feed_forward.w3.weight

                # Norms
                weight_map[f"{p}input_layernorm.weight"] = layer.attention_norm.weight
                weight_map[f"{p}post_attention_layernorm.weight"] = layer.ffn_norm.weight
                weight_map[f"{p}attention_norm.weight"] = layer.attention_norm.weight
                weight_map[f"{p}ffn_norm.weight"] = layer.ffn_norm.weight

        # Output
        weight_map["model.norm.weight"] = self.norm.weight
        weight_map["norm.weight"] = self.norm.weight
        weight_map["lm_head.weight"] = self.output.weight
        weight_map["output.weight"] = self.output.weight

        return weight_map

    def _map_weight_name(self, name: str) -> str:
        """Map various weight naming conventions to standard names."""
        # Handle different naming conventions
        replacements = [
            ("transformer.", "model."),
            ("h.", "layers."),
            ("attn.", "self_attn."),
        ]
        for old, new in replacements:
            name = name.replace(old, new)
        return name


def create_llama(config: LlamaConfig, weights: Optional[Dict[str, Tensor]] = None) -> Llama:
    """Create a LLaMA model, optionally loading weights."""
    model = Llama(config)
    if weights is not None:
        model.load_weights(weights)
    return model
