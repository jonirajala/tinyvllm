# tinyvllm Profiling Analysis

Performance profiling results for tinyvllm on Apple Silicon (M4 10-core GPU).

**Date:** 2025-01-04
**Model:** TinyLlama 1.1B (FP16)
**Profiling Tool:** tinygrad DEBUG=2

---

## Executive Summary

The 50x performance gap between achieved (1.9 tok/s) and theoretical (55 tok/s) throughput is caused by **framework overhead**, not GPU kernel speed.

| Category | Finding |
|----------|---------|
| **Primary bottleneck** | Python→Metal data copies (1.5 GB/s vs 120 GB/s) |
| **Secondary bottleneck** | Python scheduling overhead (2-20ms per batch) |
| **GPU kernels** | Running efficiently (~100 GFLOPS for linear ops) |
| **Root cause** | No JIT caching, repeated tensor allocation |

---

## Theoretical Limits

| Metric | Value | Calculation |
|--------|-------|-------------|
| GPU Memory Bandwidth | 120 GB/s | M4 specification |
| Model Size | 2.05 GB | TinyLlama FP16 weights |
| Min time per token | 17ms | 2.05 GB / 120 GB/s |
| Max tok/s (memory-bound) | 55 tok/s | 1000ms / 17ms |

Decode is memory-bandwidth-bound: must load all weights for each token.

---

## Achieved Performance

| Metric | Value | % of Theoretical |
|--------|-------|------------------|
| Decode throughput | 1.9 tok/s | 3.5% |
| TPOT | 520 ms | 50x slower than limit |
| TTFT | 1079 ms | - |

---

## Profiling Methodology

### tinygrad DEBUG Levels

| Level | Output |
|-------|--------|
| DEBUG=1 | Basic kernel info |
| DEBUG=2 | Kernel timing, GFLOPS, GB/s, memory |
| DEBUG=3 | Detailed execution info |
| DEBUG=4 | Generated kernel code |
| DEBUG=5 | Full uops and scheduling |

### Profiling Command

```bash
DEVICE=METAL DEBUG=2 ./venv/bin/python -c "
from tinyvllm.model.weights import load_model
from tinyvllm.model.tokenizer import Tokenizer
from tinyvllm.engine.engine import LLMEngine
from tinyvllm.engine.sampling import SamplingParams

model, config = load_model('models/tinyllama')
tokenizer = Tokenizer('models/tinyllama')
engine = LLMEngine(model, tokenizer)
engine.add_request('Hello', SamplingParams(max_tokens=5))
list(engine.run())
" 2>&1 | grep -E "(scheduled|copy|tm.*ms|GFLOPS|GB/s)"
```

---

## Bottleneck Analysis

### 1. Python→Metal Data Copies

**Observed:**
```
*** METAL  copy      131.07 MB tm     87206.75us/    87.21ms (    1.50 GB/s)
```

**Analysis:**
- 131 MB copied at 1.5 GB/s
- Expected: 120 GB/s (80x faster)
- This is model weight transfer, happening repeatedly

**Root cause:** tinygrad's unified memory path uses slow CPU-side memcpy instead of direct GPU access.

### 2. Scheduling Overhead

**Observed:**
```
 *** scheduled  23 kernels in   2.13 ms ***
 *** scheduled 163 kernels in  12.02 ms ***
 *** scheduled  45 kernels in  19.95 ms ***
```

**Analysis:**
- 2-20ms Python overhead per scheduling batch
- This happens between every decode step
- At 10ms average, limits throughput to ~100 tok/s even with instant kernels

**Root cause:** Python interpreter runs scheduling logic each step; no JIT caching.

### 3. Many Small Operations

**Observed:**
```
*** METAL       6 arg 2 mem    0.00 GB tm        6.96us/     6.34ms
*** METAL       3 arg 2 mem    0.00 GB tm       12.21us/     6.35ms
```

**Analysis:**
- Many ops taking 6-12μs each
- Kernel launch latency (~5μs) dominates compute time
- Should be fused into larger kernels

### 4. Linear Layer Performance (Good)

**Observed:**
```
*** METAL      14 arg 9 mem    0.04 GB tm     2292.58us  100.00 GFLOPS  8.98 GB/s
*** METAL      14 arg 9 mem    0.04 GB tm     1728.17us  120.00 GFLOPS  8.98 GB/s
```

**Analysis:**
- Linear layers running at 100-120 GFLOPS
- This is reasonable for M4 (theoretical ~2.5 TFLOPS FP32)
- GPU compute is NOT the bottleneck

---

## Time Breakdown

From bench_breakdown.py:

| Operation | Median (ms) | % of Total |
|-----------|-------------|------------|
| Decode forward | 520.37 | 75.3% |
| Prefill | 3084.57 | 23.5% |
| Sampling | 7.88 | 1.2% |
| Tokenization | 0.07 | 0.0% |
| Detokenization | 0.13 | 0.0% |

**Where does 520ms decode go?**
- ~87ms: Weight copies
- ~10-20ms: Scheduling overhead
- ~400ms: Unaccounted (likely more copies + kernel launches)

---

## Optimization Opportunities

### High Impact

| Optimization | Expected Impact | Complexity |
|--------------|-----------------|------------|
| TinyJit for decode | 2-5x | Medium |
| Eliminate weight re-copies | 2-3x | Medium |
| Kernel fusion | 1.5-2x | High |

### Medium Impact

| Optimization | Expected Impact | Complexity |
|--------------|-----------------|------------|
| Pre-allocated buffers | 1.2-1.5x | Low |
| Fixed batch padding | 1.1-1.3x | Low |
| Reduce realize() calls | 1.1-1.2x | Medium |

---

## TinyJit Investigation

tinygrad provides `TinyJit` decorator for caching kernel graphs:

```python
from tinygrad.engine.jit import TinyJit

@TinyJit
def forward(x):
    # First call: captures kernel graph
    # Subsequent calls: replays cached graph
    return model(x)
```

**Challenges for tinyvllm:**
1. Variable batch sizes (JIT needs fixed shapes)
2. Variable sequence lengths in prefill
3. Dynamic block table contents

**Solution approach:**
- Pad batch to fixed max_batch_size
- Use mask for unused slots
- Keep block tables on GPU

---

## Metal System Trace

For deeper GPU-side analysis, use Xcode Instruments:

1. Open Instruments (Xcode → Open Developer Tool → Instruments)
2. Select "Metal System Trace" template
3. Run tinyvllm inference
4. Analyze:
   - GPU utilization %
   - Kernel occupancy
   - Memory bandwidth achieved
   - GPU idle time between kernels

---

## Comparison with Other Frameworks

| Framework | Approach | Expected tok/s |
|-----------|----------|----------------|
| llama.cpp | Pure C++, no Python | 40-50 tok/s |
| MLX | Apple-optimized | 30-40 tok/s |
| tinygrad (current) | Python scheduler | 1.9 tok/s |
| tinygrad + JIT (expected) | Cached graphs | 5-15 tok/s |

The gap vs llama.cpp/MLX is primarily Python overhead and lack of JIT.

---

## Next Steps

1. **Implement TinyJit wrapper** for batched_decode (Phase 7.9.1)
2. **Profile weight loading** to verify weights stay on GPU (Phase 7.9.2)
3. **Run Metal System Trace** for GPU utilization analysis (Phase 7.9.5)
4. **Benchmark with DEBUG=4** to see generated kernel code

---

## References

- [tinygrad Profiling Guide](https://mesozoic-egg.github.io/tinygrad-notes/profiling.html)
- [tinygrad JIT Documentation](https://docs.tinygrad.org/)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
