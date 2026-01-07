"""LLaMA model implementation using tinygrad."""

from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding, Linear

from .weights import LlamaConfig
from ..core.kv_cache import KVCache
from ..core.block_manager import BlockManager
from ..kernels import flash_prefill_attention
from ..kernels import paged_decode_attention as paged_decode_attention_kernel



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
    """Apply rotary position embeddings to queries or keys.

    Args:
        x: [batch, seq, heads, head_dim]
        cos, sin: Either [seq, head_dim/2] for prefill (same positions for all batch)
                  or [batch, seq, head_dim/2] for decode (different positions per batch)
    """
    batch, seq, heads, head_dim = x.shape

    # Reshape x to pairs: [batch, seq, heads, head_dim/2, 2]
    x_reshape = x.reshape(batch, seq, heads, head_dim // 2, 2)
    x0, x1 = x_reshape[:, :, :, :, 0], x_reshape[:, :, :, :, 1]

    # Handle both prefill (2D) and decode (3D) cos/sin shapes
    if len(cos.shape) == 2:
        # Prefill: [seq, head_dim/2] -> [1, seq, 1, head_dim/2]
        cos = cos[:seq].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq].unsqueeze(0).unsqueeze(2)
    else:
        # Decode: [batch, seq, head_dim/2] -> [batch, seq, 1, head_dim/2]
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    # Apply rotation: (x0 + i*x1) * (cos + i*sin) = (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # Interleave back
    out = Tensor.stack([out0, out1], dim=-1).reshape(batch, seq, heads, head_dim)
    return out


class Attention:
    """Multi-head attention with RoPE and KV cache support."""

    def __init__(self, config: LlamaConfig):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

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
        seq_ids: List[int],
        start_positions: List[int],
        block_tables_tensor: Tensor = None,
        context_lens_tensor: Tensor = None,
    ) -> Tensor:
        """Unified attention for both prefill and decode.

        Prefill (block_tables_tensor=None): Uses flash attention on fresh K/V.
        Decode (block_tables_tensor provided): Uses paged attention from KV cache.

        Args:
            x: Input tensor [batch, seq_len, dim]
            cos, sin: RoPE frequencies
            kv_cache: KVCache instance
            block_manager: BlockManager for slot allocation
            layer_idx: Which transformer layer
            seq_ids: List of sequence IDs (length 1 for prefill, batch for decode)
            start_positions: Start position for each sequence
            block_tables_tensor: [batch, max_blocks] for decode, None for prefill
            context_lens_tensor: [batch] for decode, None for prefill
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write K/V to cache
        for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
            self._write_kv_to_blocks(kv_cache, block_manager, layer_idx, seq_id, k[i], v[i], start_pos)

        # Attention: flash for prefill, paged for decode
        # Prefill: block_tables_tensor is None (use flash attention on fresh K/V)
        # Decode: block_tables_tensor provided (use paged attention from cache)
        if block_tables_tensor is None:
            out = flash_prefill_attention(q, k, v, causal=True)
        else:
            k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)
            out = paged_decode_attention_kernel(
                q, k_cache, v_cache, block_tables_tensor, context_lens_tensor,
                self.n_heads, kv_cache.n_kv_heads, self.head_dim, kv_cache.block_size
            )

        return self.wo(out.reshape(batch, seq_len, -1))

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
        """Write K/V to block-based cache."""
        seq_len = k.shape[0]
        block_size = block_manager.block_size
        block_table = block_manager.get_block_table(seq_id)

        for i in range(seq_len):
            pos = start_pos + i
            block_idx = pos // block_size
            offset = pos % block_size

            # Allocate new block if needed (only in first layer)
            if block_idx >= len(block_table) and layer_idx == 0:
                gpu_id = block_manager.get_gpu_for_seq(seq_id)
                if len(block_manager.free_blocks[gpu_id]) == 0:
                    raise RuntimeError("Out of KV cache memory!")
                new_block = block_manager.free_blocks[gpu_id].pop()
                block_manager.ref_counts[gpu_id][new_block] = 1
                block_manager.block_tables[seq_id].append(new_block)
                block_table = block_manager.get_block_table(seq_id)

            block_id = block_table[block_idx]
            kv_cache.write_kv(layer_idx, block_id, offset, k[i], v[i])


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
        seq_ids: List[int],
        start_positions: List[int],
        block_tables_tensor: Tensor = None,
        context_lens_tensor: Tensor = None,
    ) -> Tensor:
        """Forward pass through transformer block (prefill or decode)."""
        h = self.attention(
            self.attention_norm(x), cos, sin, kv_cache, block_manager,
            layer_idx, seq_ids, start_positions,
            block_tables_tensor, context_lens_tensor
        )
        x = x + h
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

        # Precompute RoPE frequencies and realize to GPU
        cos, sin = precompute_freqs_cis(
            config.head_dim, config.max_seq_len * 2, config.rope_theta
        )
        self.cos, self.sin = cos.realize(), sin.realize()

    def prefill(
        self,
        tokens: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Prefill: process prompt tokens and populate KV cache.

        Args:
            tokens: Input token IDs [batch, seq_len]
            kv_cache: KVCache instance
            block_manager: BlockManager for slot allocation
            seq_id: Sequence ID
            start_pos: Position in sequence (default 0)

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        cos = self.cos[start_pos : start_pos + seq_len]
        sin = self.sin[start_pos : start_pos + seq_len]

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache, block_manager, layer_idx,
                      seq_ids=[seq_id], start_positions=[start_pos])

        for _ in range(seq_len):
            block_manager.advance_position(seq_id)

        return self.output(self.norm(h))

    def decode(
        self,
        tokens: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        seq_ids: List[int],
        start_positions: List[int],
        block_tables_tensor: Tensor,
        context_lens_tensor: Tensor,
    ) -> Tensor:
        """Decode: batched single-token generation.

        Args:
            tokens: Input token IDs [batch, 1] - one token per sequence
            kv_cache: KVCache instance
            block_manager: BlockManager for slot allocation
            seq_ids: List of sequence IDs
            start_positions: Start position for each sequence
            block_tables_tensor: [batch, max_blocks] int32
            context_lens_tensor: [batch] int32

        Returns:
            logits: Output logits [batch, 1, vocab_size]
        """
        batch, seq_len = tokens.shape
        assert seq_len == 1, "Decode only for single token per sequence"

        h = self.tok_embeddings(tokens)

        # Gather RoPE for each sequence's actual position
        cos = self.cos[start_positions].unsqueeze(1)  # [batch, 1, head_dim/2]
        sin = self.sin[start_positions].unsqueeze(1)

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache, block_manager, layer_idx,
                      seq_ids, start_positions, block_tables_tensor, context_lens_tensor)

        for seq_id in seq_ids:
            block_manager.advance_position(seq_id)

        return self.output(self.norm(h))

    def load_weights(self, weights: Dict[str, Tensor]):
        """Load pretrained weights into the model.

        Weights are cast to config.dtype (auto-detected or explicit).
        Directly assigns tensors to break any lazy computation graph links.
        """
        # Map weight names to (parent_object, attr_name) for direct assignment
        weight_map = self._build_weight_assignment_map()
        target_dtype = self.config.dtype

        for name, tensor in weights.items():
            target_name = name
            if name not in weight_map:
                # Try common name variations
                target_name = self._map_weight_name(name)

            if target_name in weight_map:
                parent, attr = weight_map[target_name]
                # Weights are already realized from safe_load
                # Only cast if dtype doesn't match (avoid unnecessary ops on realized tensors)
                if tensor.dtype != target_dtype:
                    tensor = tensor.cast(target_dtype).realize()
                setattr(parent, attr, tensor)

    def _build_weight_assignment_map(self) -> Dict[str, tuple]:
        """Build mapping from weight names to (parent_object, attr_name) for direct assignment."""
        weight_map = {}

        # Embeddings
        weight_map["model.embed_tokens.weight"] = (self.tok_embeddings, "weight")
        weight_map["tok_embeddings.weight"] = (self.tok_embeddings, "weight")

        # Layers
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            alt_prefix = f"layers.{i}."

            for p in [prefix, alt_prefix]:
                # Attention
                weight_map[f"{p}self_attn.q_proj.weight"] = (layer.attention.wq, "weight")
                weight_map[f"{p}self_attn.k_proj.weight"] = (layer.attention.wk, "weight")
                weight_map[f"{p}self_attn.v_proj.weight"] = (layer.attention.wv, "weight")
                weight_map[f"{p}self_attn.o_proj.weight"] = (layer.attention.wo, "weight")
                weight_map[f"{p}attention.wq.weight"] = (layer.attention.wq, "weight")
                weight_map[f"{p}attention.wk.weight"] = (layer.attention.wk, "weight")
                weight_map[f"{p}attention.wv.weight"] = (layer.attention.wv, "weight")
                weight_map[f"{p}attention.wo.weight"] = (layer.attention.wo, "weight")

                # FFN
                weight_map[f"{p}mlp.gate_proj.weight"] = (layer.feed_forward.w1, "weight")
                weight_map[f"{p}mlp.down_proj.weight"] = (layer.feed_forward.w2, "weight")
                weight_map[f"{p}mlp.up_proj.weight"] = (layer.feed_forward.w3, "weight")
                weight_map[f"{p}feed_forward.w1.weight"] = (layer.feed_forward.w1, "weight")
                weight_map[f"{p}feed_forward.w2.weight"] = (layer.feed_forward.w2, "weight")
                weight_map[f"{p}feed_forward.w3.weight"] = (layer.feed_forward.w3, "weight")

                # Norms
                weight_map[f"{p}input_layernorm.weight"] = (layer.attention_norm, "weight")
                weight_map[f"{p}post_attention_layernorm.weight"] = (layer.ffn_norm, "weight")
                weight_map[f"{p}attention_norm.weight"] = (layer.attention_norm, "weight")
                weight_map[f"{p}ffn_norm.weight"] = (layer.ffn_norm, "weight")

        # Output
        weight_map["model.norm.weight"] = (self.norm, "weight")
        weight_map["norm.weight"] = (self.norm, "weight")
        weight_map["lm_head.weight"] = (self.output, "weight")
        weight_map["output.weight"] = (self.output, "weight")

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
