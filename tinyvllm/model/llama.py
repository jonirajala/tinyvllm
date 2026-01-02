"""LLaMA model implementation using tinygrad."""

from typing import Dict, Optional, Tuple

from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding, Linear

from .weights import LlamaConfig
from ..core.kv_cache import KVCache
from ..core.attention_utils import paged_attention_with_kvcache


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
    """Multi-head attention with RoPE and KV cache support."""

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
        layer_idx: int,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Forward pass with KVCache."""
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write new K, V to cache, then compute attention
        kv_cache.write_kv(layer_idx, seq_id, k[0], v[0])
        out = paged_attention_with_kvcache(q, kv_cache, seq_id, layer_idx, start_pos)

        # Project output
        out = out.reshape(batch, seq_len, -1)
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
        layer_idx: int,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Forward pass through transformer block."""
        # Attention with residual
        h = self.attention(
            self.attention_norm(x), cos, sin, kv_cache, layer_idx, seq_id, start_pos
        )
        x = x + h
        # FFN with residual
        return x + self.feed_forward(self.ffn_norm(x))


class Llama:
    """LLaMA model."""

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
        seq_id: int = 0,
    ) -> Tensor:
        """
        Forward pass through the model.

        Args:
            tokens: Input token IDs [batch, seq_len]
            start_pos: Position in sequence (for generation with cache)
            kv_cache: KVCache instance for paged attention
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
            h = layer(h, cos, sin, kv_cache, layer_idx, seq_id, start_pos)

        # Output projection
        return self.output(self.norm(h))

    def create_kv_cache(self, dtype=dtypes.float32) -> KVCache:
        """Create a KVCache sized for this model."""
        return KVCache(
            num_layers=self.config.n_layers,
            num_blocks=100,  # Placeholder, not used in Phase 2
            block_size=16,   # Placeholder, not used in Phase 2
            n_kv_heads=self.config.n_kv_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
        )

    def load_weights(self, weights: Dict[str, Tensor]):
        """Load pretrained weights into the model."""
        # Map weight names to model attributes
        weight_map = self._build_weight_map()

        for name, tensor in weights.items():
            if name in weight_map:
                target = weight_map[name]
                target.assign(tensor.cast(target.dtype))
            else:
                # Try common name variations
                mapped_name = self._map_weight_name(name)
                if mapped_name in weight_map:
                    target = weight_map[mapped_name]
                    target.assign(tensor.cast(target.dtype))

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
