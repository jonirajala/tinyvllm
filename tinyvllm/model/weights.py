"""Weight loading utilities for LLaMA models."""

from pathlib import Path
from typing import Any, Dict, Tuple
import json

from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import safe_load


def load_safetensors(path: Path) -> Dict[str, Tensor]:
    """Load weights from safetensors file and move to default device."""
    weights = safe_load(str(path))
    device = Device.DEFAULT
    return {name: tensor.to(device).realize() for name, tensor in weights.items()}


class LlamaConfig:
    """Configuration for LLaMA model."""

    def __init__(
        self,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = None,
        vocab_size: int = 32000,
        hidden_dim: int = None,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dtype: str = "float32",
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim or 4 * dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.head_dim = dim // n_heads
        self.dtype = dtypes.float16 if dtype == "float16" else dtypes.float32

    @classmethod
    def from_dict(cls, d: Dict[str, Any], dtype: str = "float32") -> "LlamaConfig":
        """Create config from dictionary."""
        return cls(
            dim=d.get("hidden_size", d.get("dim", 4096)),
            n_layers=d.get("num_hidden_layers", d.get("n_layers", 32)),
            n_heads=d.get("num_attention_heads", d.get("n_heads", 32)),
            n_kv_heads=d.get("num_key_value_heads", d.get("n_kv_heads")),
            vocab_size=d.get("vocab_size", 32000),
            hidden_dim=d.get("intermediate_size", d.get("hidden_dim")),
            norm_eps=d.get("rms_norm_eps", d.get("norm_eps", 1e-5)),
            max_seq_len=d.get("max_position_embeddings", d.get("max_seq_len", 2048)),
            rope_theta=d.get("rope_theta", 10000.0),
            dtype=dtype,
        )

    @classmethod
    def from_json(cls, path: Path, dtype: str = "float32") -> "LlamaConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f), dtype=dtype)


def _detect_dtype(weights: Dict[str, Tensor], dtype: str) -> str:
    """Detect or validate dtype. Auto: float16 weights → float16, else float32."""
    first_weight = next(iter(weights.values()))
    weight_dtype = first_weight.dtype

    if dtype == "auto":
        return "float16" if weight_dtype == dtypes.float16 else "float32"

    if dtype == "float16" and weight_dtype != dtypes.float16:
        raise ValueError(
            f"Cannot use dtype='float16' with {weight_dtype} weights. "
            f"This would lose precision. Use dtype='float32' or dtype='auto'."
        )

    return dtype


def load_llama_weights(model_path: Path, dtype: str = "auto") -> Tuple[LlamaConfig, Dict[str, Tensor]]:
    """Load LLaMA weights from directory or safetensors file."""
    model_path = Path(model_path)

    if model_path.is_dir():
        weights = {}
        for sf_path in sorted(model_path.glob("*.safetensors")):
            weights.update(load_safetensors(sf_path))

        if not weights:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        dtype = _detect_dtype(weights, dtype)

        config_path = model_path / "config.json"
        if config_path.exists():
            config = LlamaConfig.from_json(config_path, dtype=dtype)
        else:
            raise FileNotFoundError(f"No config.json found in {model_path}")

        return config, weights

    elif model_path.suffix == ".safetensors":
        weights = load_safetensors(model_path)
        dtype = _detect_dtype(weights, dtype)

        config_path = model_path.parent / "config.json"
        if config_path.exists():
            config = LlamaConfig.from_json(config_path, dtype=dtype)
        else:
            raise FileNotFoundError(f"No config.json found for {model_path}")
        return config, weights

    else:
        raise ValueError(f"Unsupported model format: {model_path}")
