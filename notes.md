



VLLM ENGINE
- vllm config
- processor
    - turns raw input to engine request (val, tokenization and processign)
- engine core client (in no async case = enginecore)
    - which serves the shit
- output processosr
    - converts raw engine output to user output


Engine core itself contains
- Model executor
    - drives forward passes to the model (workers etc)
- Structured output manager
    - for guided decoding
- scheduler
    - decides which requests go into the next engine step
    - it further containss
        - policy settting (fcfc or priority)
    - waiting and running queues
    - kv cache manager (in cpu)
        - the heart of paged attention 
        - contains free block queue - a pool of available kv-cahche blocks
            - blocks serve as the indexing structure that maap tokens to thei computed kv cache blocks
            - The actual blocks are in gpu memory




Why we need paged attention?

1. Padding waste

Contiguous KV cache allocates the full context size / max_seq_len sized block of memory.
- With 7b param llama with 32 layers this takes 16gb of memory. 

Of course with singel user this might not be that bit of a problem but when we need to serve multiple users, we dont watn to allocate 16gb of memory for every user. Especially since in most cases the full context is not used.

2. Variable lenght growht

If we allocate just x amount of space we might overflow if -> reallocation causes copying of huge tensors

3. Batching many request

If every request has allocated huge mem block, prioritizing and payusing request becomes very costly

4. GPU memory fragmentation 
if you try to be clever:
- allocate variable tensors
- free them dynamically
- GPU memory becomes fragmented:


So what is paged attention?

Key idea
Instead of:
[token0][token1][token2]...

Use:
[BLOCK][BLOCK][BLOCK]...

Each block holds e.g. 16 tokens.

For example:
Request A (1000 tokens) → 63 blocks
Request B (30 tokens)  → 2 blocks
Request C (500 tokens) → 32 blocks


When B grows from 30 → 31 tokens:
- Already inside its second block
- No new allocation
- No copying

When it crosses block boundary:
- Allocate one more block
- Append block ID to block table


PagedAttention sacrifices memory contiguity and kernel simplicity to gain scalability, preemption, and multi-user efficiency.

Downsides of paged attention:
- Non-contiguous memory reads
- Indirect access via block tables
- Worse L2 cache hit rate

In contiguous kv cache attention can just read the whole caceh from memroy but in paged attention:
1. Read block table
2. Compute physical address
3. Jump memory
4. Handle partial blocks
5. Loop over blocks

-> Slightly lower raw TFLOPs utilization




  GPU Memory Hierarchy

  Fastest → Slowest:

  ┌─────────────────────────────┐
  │  Registers (per thread)     │  ~KB, fastest
  ├─────────────────────────────┤
  │  Shared Memory / L1 Cache   │  ~100KB per SM
  ├─────────────────────────────┤
  │  L2 Cache                   │  ~MB (shared across GPU)
  ├─────────────────────────────┤
  │  Global Memory (VRAM/HBM)   │  GBs ← YOUR TENSOR LIVES HERE
  └─────────────────────────────┘

  Where Your KV Cache Tensor Lives

  - NVIDIA GPU: VRAM (e.g., 8GB-80GB of GDDR6/HBM)
  - Apple Silicon: Unified Memory (shared with CPU)
  - AMD GPU: VRAM

  What Happens During Attention

  1. Your big tensor sits in Global Memory (VRAM)
  2. When you run attention, the GPU:
    - Loads chunks into L2 cache automatically
    - Then into Shared Memory / L1
    - Then into Registers for actual compute
  3. Results write back through the hierarchy




                      ┌─────────────────────┐
                      │   BlockManager      │  ← Runs on CPU
                      │   (no tensors)      │
                      │                     │
                      │ "GPU 0: blocks 0-99 │
                      │  GPU 1: blocks 0-99 │
                      │  GPU 2: blocks 0-99"│
                      └──────────┬──────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │ KVBlockPool  │    │ KVBlockPool  │    │ KVBlockPool  │
      │   (GPU 0)    │    │   (GPU 1)    │    │   (GPU 2)    │
      │  [tensors]   │    │  [tensors]   │    │  [tensors]   │
      └──────────────┘    └──────────────┘    └──────────────┘


  ┌─────────────────────────────────────────────────────────┐
  │                    BlockManager                          │
  │                    (CPU, shared)                         │
  │                                                          │
  │  - Tracks which blocks are free on EACH GPU             │
  │  - Decides which GPU gets new sequences                 │
  │  - No tensors, just dicts and lists                     │
  └─────────────────────────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                  ▼
  ┌──────────────────────┐          ┌──────────────────────┐
  │   KVCache (GPU 0)    │          │   KVCache (GPU 1)    │
  │                      │          │                      │
  │  - k_cache tensor    │          │  - k_cache tensor    │
  │  - v_cache tensor    │          │  - v_cache tensor    │
  │  - write_kv()        │          │  - write_kv()        │
  │  - read_kv()         │          │  - read_kv()         │
  └──────────────────────┘          └──────────────────────┘
