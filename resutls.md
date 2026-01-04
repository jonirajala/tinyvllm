phase 6.2 Medium effort:

======================================================================
tinyvllm Inference Breakdown Benchmark
======================================================================
Device: METAL
Memory before: 0.0 MB

Loading model from models/tinyllama...
Model: 2048 dim, 22 layers
Memory after load: 0.0 MB (+0.0 MB)

======================================================================
HIGH-LEVEL BREAKDOWN
======================================================================
Warming up (2 runs)...
Running benchmark with prompt: 'Hello, how are you today? I would like to know mor...'
Generating 10 tokens


Generated 10 tokens
Output: the new.
I'm a little bit...

======================================================================
TIMING BREAKDOWN
======================================================================
Operation                      Total (ms)    Count   Avg (ms)        %
----------------------------------------------------------------------
4. Decode forward                  6746.5        9     749.61    66.2%
2. Prefill (full prompt)           3161.0        1    3160.99    31.0%
3. Sampling                         257.6       10      25.76     2.5%
5. Detokenization (decode)           31.6        1      31.64     0.3%
1. Tokenization (encode)              0.2        1       0.15     0.0%
----------------------------------------------------------------------
TOTAL                             10196.9
======================================================================

Derived Metrics:
  Total time: 10196.9 ms
  Decode steps: 9
  Avg decode latency: 749.61 ms/token
  Decode throughput: 1.3 tok/s

======================================================================
PER-LAYER BREAKDOWN (decode only)
======================================================================
Prefilling 50 tokens...
Benchmarking 10 decode steps...


======================================================================
TIMING BREAKDOWN
======================================================================
Operation                      Total (ms)    Count   Avg (ms)        %
----------------------------------------------------------------------
Layer 0 Attention                   778.3       10      77.83    12.9%
Layer 19 Attention                  226.8       10      22.68     3.8%
Layer 13 Attention                  206.7       10      20.67     3.4%
Layer 9 Attention                   206.0       10      20.60     3.4%
Embedding                           203.7       10      20.37     3.4%
Layer 15 Attention                  183.6       10      18.36     3.0%
Layer 14 Attention                  183.3       10      18.33     3.0%
Layer 18 Attention                  182.4       10      18.24     3.0%
Layer 1 Attention                   180.6       10      18.06     3.0%
Layer 21 Attention                  178.7       10      17.87     3.0%
Layer 20 Attention                  177.4       10      17.74     2.9%
Layer 10 Attention                  177.4       10      17.74     2.9%
Layer 11 Attention                  176.8       10      17.68     2.9%
Layer 12 Attention                  176.7       10      17.67     2.9%
Layer 3 Attention                   176.4       10      17.64     2.9%
Layer 2 Attention                   175.9       10      17.59     2.9%
Layer 7 Attention                   175.8       10      17.58     2.9%
Layer 16 Attention                  175.3       10      17.53     2.9%
Layer 17 Attention                  175.0       10      17.50     2.9%
Layer 5 Attention                   174.5       10      17.45     2.9%
Layer 4 Attention                   174.1       10      17.41     2.9%
Layer 6 Attention                   173.7       10      17.37     2.9%
Layer 8 Attention                   172.4       10      17.24     2.9%
Layer 6 FFN                          86.0       10       8.60     1.4%
Layer 8 FFN                          77.2       10       7.72     1.3%
Layer 0 FFN                          72.5       10       7.25     1.2%
Layer 9 FFN                          51.3       10       5.13     0.9%
Layer 2 FFN                          51.3       10       5.13     0.8%
Layer 11 FFN                         51.2       10       5.12     0.8%
Layer 14 FFN                         50.9       10       5.09     0.8%
Layer 17 FFN                         50.7       10       5.07     0.8%
Layer 10 FFN                         50.6       10       5.06     0.8%
Layer 21 FFN                         50.5       10       5.05     0.8%
Layer 1 FFN                          50.5       10       5.05     0.8%
Layer 12 FFN                         50.4       10       5.04     0.8%
Layer 15 FFN                         50.3       10       5.03     0.8%
Layer 20 FFN                         50.2       10       5.02     0.8%
Layer 3 FFN                          49.9       10       4.99     0.8%
Layer 19 FFN                         49.9       10       4.99     0.8%
Layer 18 FFN                         49.9       10       4.99     0.8%
Layer 7 FFN                          49.8       10       4.98     0.8%
Layer 5 FFN                          49.7       10       4.97     0.8%
Layer 4 FFN                          49.7       10       4.97     0.8%
Layer 13 FFN                         49.6       10       4.96     0.8%
Layer 16 FFN                         49.6       10       4.96     0.8%
Output projection                    31.4       10       3.14     0.5%
RoPE freqs                            1.0       10       0.10     0.0%
----------------------------------------------------------------------
TOTAL                              6035.7
======================================================================

======================================================================
AGGREGATED BY COMPONENT
======================================================================
Component                       Time (ms)        %
--------------------------------------------------
Attention (all layers)             4608.0    76.3%
FFN (all layers)                   1191.6    19.7%
Embedding                           203.7     3.4%
Output projection                    31.4     0.5%
RoPE freqs                            1.0     0.0%

======================================================================
BOTTLENECK ANALYSIS
======================================================================

Primary bottleneck: 4. Decode forward
  Time: 6746.5 ms (66.2% of total)

  Suggestion: Decode is dominant. Consider:
    - Speculative decoding (draft + verify)
    - Model quantization (INT8/INT4)
    - Larger batch sizes for throughput

