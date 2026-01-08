"""LLaMA model implementation using tinygrad."""

import math
from typing import Dict, List, Optional, Tuple

from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn import Embedding, Linear

from .weights import LlamaConfig
from ..core.kv_cache import KVCache
from ..core.block_manager import BlockManager


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = Tensor.ones(dim)

    def __call__(self, x: Tensor) -> Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[Tensor, Tensor]:
    """Precompute rotary position embedding frequencies."""
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2).float() / dim))
    t = Tensor.arange(seq_len).float()
    freqs = t.unsqueeze(1) * freqs.unsqueeze(0)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embeddings.

    x: [batch, seq, heads, head_dim]
    cos, sin: [seq, head_dim/2] for prefill or [batch, seq, head_dim/2] for decode
    """
    batch, seq, heads, head_dim = x.shape

    x_reshape = x.reshape(batch, seq, heads, head_dim // 2, 2)
    x0, x1 = x_reshape[:, :, :, :, 0], x_reshape[:, :, :, :, 1]

    if len(cos.shape) == 2:
        cos = cos[:seq].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq].unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    return Tensor.stack([out0, out1], dim=-1).reshape(batch, seq, heads, head_dim)


def flash_prefill_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    causal: bool = True,
) -> Tensor:
    """Flash Attention for prefill. Input shapes: [1, seq, heads, head_dim]."""
    queries = queries.squeeze(0)
    keys = keys.squeeze(0)
    values = values.squeeze(0)

    q_len, n_heads, head_dim = queries.shape
    kv_len, n_kv_heads, _ = keys.shape
    scale = 1.0 / math.sqrt(head_dim)

    # GQA: expand KV heads to match Q heads
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        keys = keys.unsqueeze(2).expand(kv_len, n_kv_heads, n_rep, head_dim).reshape(kv_len, n_heads, head_dim)
        values = values.unsqueeze(2).expand(kv_len, n_kv_heads, n_rep, head_dim).reshape(kv_len, n_heads, head_dim)

    q = queries.transpose(0, 1)
    k = keys.transpose(0, 1)
    v = values.transpose(0, 1)

    # realize() breaks lazy graph to avoid tinygrad TC optimizer bug
    scores = ((q @ k.transpose(-2, -1)) * scale).realize()

    if causal:
        q_positions = Tensor.arange(q_len).reshape(q_len, 1)
        kv_positions = Tensor.arange(kv_len).reshape(1, kv_len)
        mask = (kv_positions > q_positions).cast(dtypes.float32) * (-1e9)
        scores = scores + mask.unsqueeze(0)

    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v
    return output.transpose(0, 1).unsqueeze(0)


def paged_decode_attention(
    queries: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_context_len: int = None,
) -> Tensor:
    """Paged attention for decode. Gathers K/V from block-based cache."""
    batch_size = queries.shape[0]
    total_max_blocks = block_tables.shape[1]

    if max_context_len is not None:
        blocks_needed = min((max_context_len + block_size - 1) // block_size, total_max_blocks)
        max_context = blocks_needed * block_size
        block_tables = block_tables[:, :blocks_needed]
    else:
        max_context = total_max_blocks * block_size

    block_indices = block_tables.flatten()
    k_gathered = k_cache[block_indices].reshape(batch_size, max_context, n_kv_heads, head_dim)
    v_gathered = v_cache[block_indices].reshape(batch_size, max_context, n_kv_heads, head_dim)

    # GQA
    if n_kv_heads != n_heads:
        n_rep = n_heads // n_kv_heads
        k_gathered = k_gathered.unsqueeze(3).expand(batch_size, max_context, n_kv_heads, n_rep, head_dim).reshape(batch_size, max_context, n_heads, head_dim)
        v_gathered = v_gathered.unsqueeze(3).expand(batch_size, max_context, n_kv_heads, n_rep, head_dim).reshape(batch_size, max_context, n_heads, head_dim)

    q = queries.transpose(1, 2)
    k = k_gathered.transpose(1, 2)
    v = v_gathered.transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale

    positions = Tensor.arange(max_context).reshape(1, max_context)
    valid_mask = (positions < context_lens.reshape(batch_size, 1)).cast(dtypes.float32)
    attn_mask = (1.0 - valid_mask) * (-1e9)
    scores = scores + attn_mask.reshape(batch_size, 1, 1, max_context)

    attn_weights = scores.softmax(axis=-1)
    output = attn_weights @ v
    return output.transpose(1, 2)


class Attention:
    """Multi-head attention with RoPE and KV cache."""

    def __init__(self, config: LlamaConfig):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim

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
    ) -> Tensor:
        """Prefill attention. Decode uses JIT path in Llama.decode()."""
        batch, seq_len, _ = x.shape

        q = self.wq(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
            self._write_kv_to_blocks(kv_cache, block_manager, layer_idx, seq_id, k[i], v[i], start_pos)

        out = flash_prefill_attention(q, k, v, causal=True)

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
        """Write K/V to block-based cache. Groups tokens by block for batched writes."""
        seq_len = k.shape[0]
        block_size = block_manager.block_size
        block_table = block_manager.get_block_table(seq_id)

        token_idx = 0
        pos = start_pos

        while token_idx < seq_len:
            block_idx = pos // block_size
            offset = pos % block_size
            tokens_in_block = min(block_size - offset, seq_len - token_idx)

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

            k_batch = k[token_idx:token_idx + tokens_in_block]
            v_batch = v[token_idx:token_idx + tokens_in_block]
            kv_cache.write_kv(layer_idx, block_id, offset, k_batch, v_batch)

            token_idx += tokens_in_block
            pos += tokens_in_block


class FeedForward:
    """SwiGLU feed-forward network."""

    def __init__(self, config: LlamaConfig):
        self.w1 = Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = Linear(config.dim, config.hidden_dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))


class TransformerBlock:
    """Single transformer block."""

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
    ) -> Tensor:
        """Prefill forward. Decode uses JIT path in Llama.decode()."""
        h = self.attention(
            self.attention_norm(x), cos, sin, kv_cache, block_manager,
            layer_idx, seq_ids, start_positions
        )
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x))


class Llama:
    """LLaMA model with paged KV cache."""

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.tok_embeddings = Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = Linear(config.dim, config.vocab_size, bias=False)

        cos, sin = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, config.rope_theta)
        self.cos, self.sin = cos.realize(), sin.realize()

    def prefill(
        self,
        tokens: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        seq_id: int,
        start_pos: int = 0,
    ) -> Tensor:
        """Process prompt tokens and populate KV cache. Returns logits [batch, seq_len, vocab]."""
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

    def create_jit_decode(self, block_size: int = 16):
        """Create JIT-compiled decode function. Caller should cache the result."""
        config = self.config
        layers = self.layers
        tok_embeddings = self.tok_embeddings
        norm = self.norm
        output = self.output

        @TinyJit
        def jit_decode_forward(
            tokens: Tensor,
            cos: Tensor,
            sin: Tensor,
            k_caches: List[Tensor],
            v_caches: List[Tensor],
            block_tables: Tensor,
            context_lens: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            """Returns (logits, k_all, v_all) for cache writeback."""
            batch = tokens.shape[0]
            h = tok_embeddings(tokens)

            k_outputs = []
            v_outputs = []

            for layer_idx, layer in enumerate(layers):
                x_norm = layer.attention_norm(h)

                q = layer.attention.wq(x_norm).reshape(batch, 1, config.n_heads, config.head_dim)
                k = layer.attention.wk(x_norm).reshape(batch, 1, config.n_kv_heads, config.head_dim)
                v = layer.attention.wv(x_norm).reshape(batch, 1, config.n_kv_heads, config.head_dim)

                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

                k_outputs.append(k.squeeze(1))
                v_outputs.append(v.squeeze(1))

                attn_out = paged_decode_attention(
                    q, k_caches[layer_idx], v_caches[layer_idx],
                    block_tables, context_lens,
                    config.n_heads, config.n_kv_heads, config.head_dim, block_size
                )

                attn_out = layer.attention.wo(attn_out.reshape(batch, 1, -1))
                h = h + attn_out
                h = h + layer.feed_forward(layer.ffn_norm(h))

            logits = output(norm(h))
            k_all = Tensor.stack(*k_outputs)
            v_all = Tensor.stack(*v_outputs)

            return logits, k_all, v_all

        return jit_decode_forward

    def decode(
        self,
        tokens: Tensor,
        kv_cache: KVCache,
        block_manager: BlockManager,
        seq_ids: List[int],
        start_positions: List[int],
        block_tables_tensor: Tensor,
        context_lens_tensor: Tensor,
        jit_fn,
        max_blocks: int = 64,
    ) -> Tensor:
        """Batched single-token decode. Returns logits [batch, 1, vocab]."""
        batch = tokens.shape[0]

        pos_tensor = Tensor(start_positions)
        cos = self.cos[pos_tensor].unsqueeze(1)
        sin = self.sin[pos_tensor].unsqueeze(1)

        k_caches = [kv_cache.k_cache[i] for i in range(self.config.n_layers)]
        v_caches = [kv_cache.v_cache[i] for i in range(self.config.n_layers)]

        # Pad block_tables to fixed size for JIT
        current_blocks = block_tables_tensor.shape[1]
        if current_blocks < max_blocks:
            padding = Tensor.zeros(batch, max_blocks - current_blocks, dtype=dtypes.int32)
            block_tables_padded = block_tables_tensor.cat(padding, dim=1)
        else:
            block_tables_padded = block_tables_tensor[:, :max_blocks]

        logits, k_all, v_all = jit_fn(
            tokens, cos, sin, k_caches, v_caches,
            block_tables_padded, context_lens_tensor
        )

        # Write K/V to cache (outside JIT)
        for i, (seq_id, start_pos) in enumerate(zip(seq_ids, start_positions)):
            block_idx = start_pos // block_manager.block_size
            offset = start_pos % block_manager.block_size
            block_table = block_manager.get_block_table(seq_id)

            if block_idx >= len(block_table):
                gpu_id = block_manager.get_gpu_for_seq(seq_id)
                if len(block_manager.free_blocks[gpu_id]) == 0:
                    raise RuntimeError("Out of KV cache memory!")
                new_block = block_manager.free_blocks[gpu_id].pop()
                block_manager.ref_counts[gpu_id][new_block] = 1
                block_manager.block_tables[seq_id].append(new_block)
                block_table = block_manager.get_block_table(seq_id)

            block_id = block_table[block_idx]

            for layer_idx in range(self.config.n_layers):
                k = k_all[layer_idx, i:i+1]
                v = v_all[layer_idx, i:i+1]
                kv_cache.write_kv(layer_idx, block_id, offset, k, v)

        for seq_id in seq_ids:
            block_manager.advance_position(seq_id)

        return logits

    def load_weights(self, weights: Dict[str, Tensor]) -> None:
        """Load pretrained weights. Casts to config.dtype."""
        weight_map = self._build_weight_map()
        target_dtype = self.config.dtype

        for name, tensor in weights.items():
            target_name = name if name in weight_map else self._map_weight_name(name)

            if target_name in weight_map:
                parent, attr = weight_map[target_name]
                if tensor.dtype != target_dtype:
                    tensor = tensor.cast(target_dtype).realize()
                setattr(parent, attr, tensor)

    def _build_weight_map(self) -> Dict[str, tuple]:
        """Build weight name -> (parent, attr) mapping."""
        weight_map = {}

        weight_map["model.embed_tokens.weight"] = (self.tok_embeddings, "weight")
        weight_map["tok_embeddings.weight"] = (self.tok_embeddings, "weight")

        for i, layer in enumerate(self.layers):
            for prefix in [f"model.layers.{i}.", f"layers.{i}."]:
                weight_map[f"{prefix}self_attn.q_proj.weight"] = (layer.attention.wq, "weight")
                weight_map[f"{prefix}self_attn.k_proj.weight"] = (layer.attention.wk, "weight")
                weight_map[f"{prefix}self_attn.v_proj.weight"] = (layer.attention.wv, "weight")
                weight_map[f"{prefix}self_attn.o_proj.weight"] = (layer.attention.wo, "weight")
                weight_map[f"{prefix}attention.wq.weight"] = (layer.attention.wq, "weight")
                weight_map[f"{prefix}attention.wk.weight"] = (layer.attention.wk, "weight")
                weight_map[f"{prefix}attention.wv.weight"] = (layer.attention.wv, "weight")
                weight_map[f"{prefix}attention.wo.weight"] = (layer.attention.wo, "weight")

                weight_map[f"{prefix}mlp.gate_proj.weight"] = (layer.feed_forward.w1, "weight")
                weight_map[f"{prefix}mlp.down_proj.weight"] = (layer.feed_forward.w2, "weight")
                weight_map[f"{prefix}mlp.up_proj.weight"] = (layer.feed_forward.w3, "weight")
                weight_map[f"{prefix}feed_forward.w1.weight"] = (layer.feed_forward.w1, "weight")
                weight_map[f"{prefix}feed_forward.w2.weight"] = (layer.feed_forward.w2, "weight")
                weight_map[f"{prefix}feed_forward.w3.weight"] = (layer.feed_forward.w3, "weight")

                weight_map[f"{prefix}input_layernorm.weight"] = (layer.attention_norm, "weight")
                weight_map[f"{prefix}post_attention_layernorm.weight"] = (layer.ffn_norm, "weight")
                weight_map[f"{prefix}attention_norm.weight"] = (layer.attention_norm, "weight")
                weight_map[f"{prefix}ffn_norm.weight"] = (layer.ffn_norm, "weight")

        weight_map["model.norm.weight"] = (self.norm, "weight")
        weight_map["norm.weight"] = (self.norm, "weight")
        weight_map["lm_head.weight"] = (self.output, "weight")
        weight_map["output.weight"] = (self.output, "weight")

        return weight_map

    def _map_weight_name(self, name: str) -> str:
        """Map weight naming conventions."""
        for old, new in [("transformer.", "model."), ("h.", "layers."), ("attn.", "self_attn.")]:
            name = name.replace(old, new)
        return name


def create_llama(config: LlamaConfig, weights: Optional[Dict[str, Tensor]] = None) -> Llama:
    """Create a LLaMA model, optionally loading weights."""
    model = Llama(config)
    if weights is not None:
        model.load_weights(weights)
    return model
