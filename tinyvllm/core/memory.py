"""Auto-configure KV cache size based on available GPU memory."""

import subprocess
from tinygrad import Device


def get_gpu_memory() -> int | None:
    """Get GPU memory in bytes. Returns None if detection fails."""
    device = Device.DEFAULT
    try:
        if device == "METAL":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(result.stdout.strip()) if result.returncode == 0 else None
        elif device == "CUDA":
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=5)
            return int(float(result.stdout.strip().split("\n")[0]) * 1024**2) if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError): pass
    return None


def estimate_model_memory(config) -> int:
    """Estimate model memory from config parameters."""
    d, h, kv, ff, L, V = config.dim, config.n_heads, config.n_kv_heads, config.hidden_dim, config.n_layers, config.vocab_size
    head_dim = d // h
    params = V * d * 2 + L * (d * h * head_dim + 2 * d * kv * head_dim + h * head_dim * d + 3 * d * ff + 2 * d) + d
    return params * config.dtype.itemsize


def auto_num_blocks(config, block_size: int = 16, gpu_utilization: float = 0.9, max_blocks: int = 500) -> int:
    """Calculate optimal num_blocks based on available GPU memory."""
    gpu_mem = get_gpu_memory()
    if gpu_mem is None:
        print("KV cache: 100 blocks (GPU detection failed)")
        return 100

    model_mem = estimate_model_memory(config)
    available = int(gpu_mem * gpu_utilization) - model_mem
    block_mem = block_size * config.n_kv_heads * (config.dim // config.n_heads) * 2 * config.n_layers * config.dtype.itemsize
    num_blocks = max(10, min(max_blocks, available // block_mem))

    print(f"KV cache: {num_blocks} blocks ({num_blocks * block_mem / 1024**2:.0f} MB), GPU: {gpu_mem / 1024**3:.0f} GB")
    return num_blocks
