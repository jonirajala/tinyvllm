"""Weight loading utilities for LLaMA models using tinygrad."""

from pathlib import Path
from typing import Dict, Any
import json
import struct

from tinygrad import Tensor, dtypes


def load_safetensors(path: Path) -> Dict[str, Tensor]:
    """Load weights from a safetensors file."""
    with open(path, "rb") as f:
        # Read header size (first 8 bytes, little-endian uint64)
        header_size = struct.unpack("<Q", f.read(8))[0]
        # Read and parse header JSON
        header = json.loads(f.read(header_size))
        # Calculate data start offset
        data_start = 8 + header_size

        weights = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]

            # Map safetensors dtype to tinygrad dtype
            dtype_map = {
                "F32": dtypes.float32,
                "F16": dtypes.float16,
                "BF16": dtypes.bfloat16,
                "I32": dtypes.int32,
                "I64": dtypes.int64,
            }
            dtype = dtype_map.get(dtype_str, dtypes.float32)

            # Read tensor data
            f.seek(data_start + offsets[0])
            num_bytes = offsets[1] - offsets[0]
            data = f.read(num_bytes)

            # Create tensor
            weights[name] = Tensor(data, dtype=dtype).reshape(shape)

        return weights


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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LlamaConfig":
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
        )

    @classmethod
    def from_json(cls, path: Path) -> "LlamaConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def load_llama_weights(model_path: Path) -> tuple[LlamaConfig, Dict[str, Tensor]]:
    """
    Load LLaMA weights from a directory or file.

    Supports:
    - Directory with config.json and .safetensors files
    - Single .safetensors file (requires separate config)
    """
    model_path = Path(model_path)

    if model_path.is_dir():
        # Load config
        config_path = model_path / "config.json"
        if config_path.exists():
            config = LlamaConfig.from_json(config_path)
        else:
            raise FileNotFoundError(f"No config.json found in {model_path}")

        # Load weights from safetensors files
        weights = {}
        for sf_path in sorted(model_path.glob("*.safetensors")):
            weights.update(load_safetensors(sf_path))

        if not weights:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        return config, weights

    elif model_path.suffix == ".safetensors":
        # Single file - need config in same directory
        weights = load_safetensors(model_path)
        config_path = model_path.parent / "config.json"
        if config_path.exists():
            config = LlamaConfig.from_json(config_path)
        else:
            raise FileNotFoundError(f"No config.json found for {model_path}")
        return config, weights

    else:
        raise ValueError(f"Unsupported model format: {model_path}")
