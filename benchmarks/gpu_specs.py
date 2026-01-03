"""GPU theoretical specifications for benchmark comparisons.

Provides hardware limits to calculate utilization percentages.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSpecs:
    """Hardware specifications for a GPU."""
    name: str
    fp32_tflops: float          # Theoretical FP32 compute (TFLOPS)
    fp16_tflops: float          # Theoretical FP16 compute (TFLOPS)
    memory_bandwidth_gbps: float  # Memory bandwidth (GB/s)
    simd_width: int             # Threads per SIMD group
    max_threads: int            # Max concurrent threads


# Known GPU specifications
GPU_SPECS = {
    # Apple Silicon
    "M4_10CORE": GPUSpecs(
        name="Apple M4 10-core GPU",
        fp32_tflops=4.4,  # Theoretical; measured ~2.9 TFLOPS
        fp16_tflops=8.8,
        memory_bandwidth_gbps=120,
        simd_width=32,
        max_threads=2560,
    ),
    "M4_PRO": GPUSpecs(
        name="Apple M4 Pro",
        fp32_tflops=8.0,
        fp16_tflops=16.0,
        memory_bandwidth_gbps=273,
        simd_width=32,
        max_threads=5120,
    ),
    "M3_10CORE": GPUSpecs(
        name="Apple M3 10-core GPU",
        fp32_tflops=4.1,
        fp16_tflops=8.2,
        memory_bandwidth_gbps=100,
        simd_width=32,
        max_threads=2560,
    ),
    "M1_8CORE": GPUSpecs(
        name="Apple M1 8-core GPU",
        fp32_tflops=2.6,
        fp16_tflops=5.2,
        memory_bandwidth_gbps=68,
        simd_width=32,
        max_threads=2048,
    ),
    # Default for unknown
    "DEFAULT": GPUSpecs(
        name="Unknown GPU",
        fp32_tflops=4.0,
        fp16_tflops=8.0,
        memory_bandwidth_gbps=100,
        simd_width=32,
        max_threads=2048,
    ),
}


def get_gpu_specs(gpu_name: Optional[str] = None) -> GPUSpecs:
    """Get GPU specs by name, or return default."""
    if gpu_name and gpu_name.upper() in GPU_SPECS:
        return GPU_SPECS[gpu_name.upper()]
    return GPU_SPECS["DEFAULT"]


@dataclass
class TheoreticalLimits:
    """Theoretical performance limits for a given workload.

    NOTE: These limits are for the ATTENTION KERNEL only, not the full model.
    Full model performance includes FFN, embeddings, and other ops which
    typically dominate compute. Use these numbers to evaluate attention
    kernel efficiency, not end-to-end model performance.
    """

    # Workload parameters
    n_heads: int
    n_kv_heads: int
    head_dim: int
    context_len: int
    batch_size: int

    # Hardware specs
    gpu: GPUSpecs

    # Optional: model parameters for full-model estimates
    n_layers: int = 1
    dim: int = 0  # Model dimension (0 = use n_heads * head_dim)
    vocab_size: int = 0  # For embedding/output projection
    total_params: int = 0  # Total model parameters (for memory-bound calc)
    bytes_per_param: float = 4.0  # 4.0=FP32, 2.0=FP16, 1.0=Q8, 0.5=Q4

    @property
    def model_dim(self) -> int:
        return self.dim if self.dim > 0 else self.n_heads * self.head_dim

    @property
    def attention_flops_per_token(self) -> float:
        """FLOPs for decode attention only (1 query token)."""
        # Q·K: 2 * context_len * n_heads * head_dim (multiply-add)
        qk_flops = 2 * self.context_len * self.n_heads * self.head_dim
        # softmax: ~3 ops per element (exp, sum, div)
        softmax_flops = 3 * self.context_len * self.n_heads
        # attn * V: 2 * context_len * n_heads * head_dim
        av_flops = 2 * self.context_len * self.n_heads * self.head_dim
        # Two-pass doubles Q·K (current implementation)
        return 2 * qk_flops + softmax_flops + av_flops

    @property
    def flops_per_token(self) -> float:
        """FLOPs needed for decode attention (1 query token)."""
        return self.attention_flops_per_token

    @property
    def full_model_flops_per_token(self) -> float:
        """Rough estimate of full model FLOPs per token (decode).

        Includes: attention, FFN (2x MLP), embeddings, RMSNorm.
        FFN typically dominates (~60-70% of compute).
        """
        dim = self.model_dim
        # Attention (already calculated)
        attn = self.attention_flops_per_token * self.n_layers
        # FFN: 2 * (2 * dim * 4*dim) for up/down projections per layer
        # Using 4x multiplier for hidden_dim (typical for LLaMA)
        ffn_per_layer = 2 * 2 * dim * (4 * dim)
        ffn = ffn_per_layer * self.n_layers
        # RMSNorm: ~2 * dim per norm, 2 norms per layer
        norm = 4 * dim * self.n_layers
        # Output projection: dim * vocab_size
        output = 2 * dim * self.vocab_size if self.vocab_size > 0 else 0
        return attn + ffn + norm + output

    @property
    def bytes_per_token(self) -> float:
        """Memory bytes read for decode attention (1 query token)."""
        # Q: n_heads * head_dim * 4 bytes
        q_bytes = self.n_heads * self.head_dim * 4
        # K: context_len * n_kv_heads * head_dim * 4 bytes
        k_bytes = self.context_len * self.n_kv_heads * self.head_dim * 4
        # V: context_len * n_kv_heads * head_dim * 4 bytes
        v_bytes = self.context_len * self.n_kv_heads * self.head_dim * 4
        # Two-pass reads K twice
        return q_bytes + 2 * k_bytes + v_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte - determines compute vs memory bound."""
        return self.flops_per_token / self.bytes_per_token

    @property
    def is_memory_bound(self) -> bool:
        """True if workload is limited by memory bandwidth, not compute."""
        # Ridge point: where memory and compute limits meet
        ridge_point = self.gpu.fp32_tflops * 1e12 / (self.gpu.memory_bandwidth_gbps * 1e9)
        return self.arithmetic_intensity < ridge_point

    @property
    def max_tokens_per_sec_compute(self) -> float:
        """Max tokens/sec if compute-bound."""
        flops_available = self.gpu.fp32_tflops * 1e12
        return flops_available / self.flops_per_token

    @property
    def max_tokens_per_sec_memory(self) -> float:
        """Max tokens/sec if memory-bound."""
        bandwidth_bytes = self.gpu.memory_bandwidth_gbps * 1e9
        return bandwidth_bytes / self.bytes_per_token

    @property
    def max_tokens_per_sec(self) -> float:
        """Theoretical max tokens/sec (min of compute and memory limits)."""
        return min(self.max_tokens_per_sec_compute, self.max_tokens_per_sec_memory)

    @property
    def max_tokens_per_sec_full_model_compute(self) -> float:
        """Theoretical max tokens/sec for full model (compute-bound)."""
        if self.full_model_flops_per_token <= 0:
            return 0
        flops_available = self.gpu.fp32_tflops * 1e12
        return flops_available / self.full_model_flops_per_token

    @property
    def total_model_params(self) -> int:
        """Estimate total model parameters from architecture."""
        if self.total_params > 0:
            return self.total_params
        dim = self.model_dim
        # Attention: Wq, Wk, Wv, Wo per layer
        attn_params = 4 * dim * dim * self.n_layers
        # FFN: up, gate, down (3 projections, 4x hidden)
        ffn_params = 3 * dim * (4 * dim) * self.n_layers
        # Embeddings
        embed_params = dim * self.vocab_size if self.vocab_size > 0 else 0
        return attn_params + ffn_params + embed_params

    @property
    def model_bytes(self) -> float:
        """Model size in bytes (for memory-bound calculation)."""
        return self.total_model_params * self.bytes_per_param

    @property
    def max_tokens_per_sec_full_model_memory(self) -> float:
        """Theoretical max tokens/sec for full model (memory-bound).

        LLM decode is memory-bound: each token reads all model weights.
        This is typically the REAL bottleneck, not compute.
        """
        if self.model_bytes <= 0:
            return 0
        bandwidth_bytes = self.gpu.memory_bandwidth_gbps * 1e9
        return bandwidth_bytes / self.model_bytes

    @property
    def max_tokens_per_sec_full_model(self) -> float:
        """Theoretical max tokens/sec for full model (min of compute and memory)."""
        compute = self.max_tokens_per_sec_full_model_compute
        memory = self.max_tokens_per_sec_full_model_memory
        if compute <= 0:
            return memory
        if memory <= 0:
            return compute
        return min(compute, memory)

    @property
    def full_model_bottleneck(self) -> str:
        """Which resource limits full model performance."""
        compute = self.max_tokens_per_sec_full_model_compute
        memory = self.max_tokens_per_sec_full_model_memory
        if memory < compute:
            return "memory"
        return "compute"

    @property
    def bottleneck(self) -> str:
        """Which resource is the bottleneck."""
        return "memory" if self.is_memory_bound else "compute"

    def utilization(self, actual_tokens_per_sec: float) -> float:
        """Calculate utilization percentage given actual performance (attention only)."""
        return (actual_tokens_per_sec / self.max_tokens_per_sec) * 100

    def utilization_full_model(self, actual_tokens_per_sec: float) -> float:
        """Calculate utilization percentage for full model estimate."""
        if self.max_tokens_per_sec_full_model <= 0:
            return 0
        return (actual_tokens_per_sec / self.max_tokens_per_sec_full_model) * 100

    def report(self, actual_tokens_per_sec: Optional[float] = None) -> str:
        """Generate a report of theoretical limits and utilization."""
        lines = [
            f"GPU: {self.gpu.name}",
            f"Workload: {self.n_heads} heads, head_dim={self.head_dim}, ctx={self.context_len}, batch={self.batch_size}",
        ]

        if self.n_layers > 1 or self.vocab_size > 0:
            lines.append(f"Model: {self.n_layers} layers, dim={self.model_dim}, vocab={self.vocab_size}")

        lines.extend([
            "",
            "Attention Kernel Limits:",
            f"  FLOPs/token:      {self.flops_per_token / 1e6:.2f} M",
            f"  Bytes/token:      {self.bytes_per_token / 1e3:.2f} KB",
            f"  Arithmetic int.:  {self.arithmetic_intensity:.2f} FLOPs/byte",
            f"  Bottleneck:       {self.bottleneck.upper()}",
            f"  Max tok/s:        {self.max_tokens_per_sec:,.0f}",
        ])

        if self.full_model_flops_per_token > 0:
            # Format precision label
            prec_labels = {4.0: "FP32", 2.0: "FP16", 1.0: "Q8", 0.5: "Q4"}
            prec = prec_labels.get(self.bytes_per_param, f"{self.bytes_per_param}B/param")
            lines.extend([
                "",
                f"Full Model Estimate ({prec}):",
                f"  Model size:       {self.model_bytes / 1e6:.1f} MB ({self.total_model_params/1e6:.1f}M params)",
                f"  FLOPs/token:      {self.full_model_flops_per_token / 1e6:.2f} M",
                f"  Max (compute):    {self.max_tokens_per_sec_full_model_compute:,.0f} tok/s",
                f"  Max (memory):     {self.max_tokens_per_sec_full_model_memory:,.0f} tok/s",
                f"  Max (effective):  {self.max_tokens_per_sec_full_model:,.0f} tok/s ({self.full_model_bottleneck}-bound)",
            ])

        if actual_tokens_per_sec is not None:
            util_attn = self.utilization(actual_tokens_per_sec)
            lines.extend([
                "",
                "Actual Performance:",
                f"  Measured:         {actual_tokens_per_sec:,.1f} tok/s",
            ])

            if self.max_tokens_per_sec_full_model > 0:
                util_full = self.utilization_full_model(actual_tokens_per_sec)
                lines.extend([
                    f"  vs Full Model:    {util_full:.1f}% utilized",
                    f"  Potential:        {self.max_tokens_per_sec_full_model / actual_tokens_per_sec:.1f}x speedup",
                ])
            else:
                lines.extend([
                    f"  vs Attention:     {util_attn:.4f}% utilized",
                ])

        return "\n".join(lines)
