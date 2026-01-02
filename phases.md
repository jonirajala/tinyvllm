⏺ Phase 1: Foundation

  Goal

  Generate text with a single LLaMA model, one request at a time.

  ---
  Files

  tinyvllm/
  ├── model/
  │   ├── llama.py        # LLaMA architecture
  │   ├── tokenizer.py    # Text ↔ tokens
  │   └── weights.py      # Load safetensors
  ├── engine/
  │   ├── generate.py     # Generation loop
  │   └── sampling.py     # Token selection
  └── main.py             # CLI

  ---
  Components

  1.1 Token Generation Loop

  The autoregressive process:
  1. Convert prompt to tokens
  2. Prefill: Run entire prompt through model once, cache all K,V
  3. Decode: Generate one token at a time, reusing cached K,V
  4. Stop when EOS token or max length reached
  5. Convert tokens back to text

  Key concept - KV Cache: Without it, you recompute attention for ALL previous tokens at each step (O(n²)). With it, you only compute for the new token (O(n)).

  1.2 Model Loading

  - Load config (dimensions, layers, heads)
  - Create LLaMA architecture:
    - Embedding layer (token → vector)
    - N transformer blocks (attention + FFN + norms)
    - Output layer (vector → vocabulary probabilities)
  - Load weights from safetensors file
  - Apply RoPE (rotary position emb eddings)

  1.3 Sampling

  Convert model output (logits) to next token:
  - Temperature: Higher = more random
  - Top-k: Only consider k most likely tokens
  - Top-p (nucleus): Only consider tokens until cumulative probability > p
  - Greedy: Just pick the highest probability

  ---
  After Phase 1 You Can

  ✅ Load any LLaMA model
  ✅ Generate text with various sampling strategies
  ✅ Stream tokens as generated
  ✅ Basic CLI usage

  Limitations

  ❌ One request at a time
  ❌ Pre-allocates maximum memory (wasteful)
  ❌ Can't serve multiple users
  ❌ No API server

  ---
  Phase 2: Paged Attention

  Goal

  Replace wasteful pre-allocated KV cache with on-demand block allocation.

  ---
  Files

  tinyvllm/
  ├── model/
  │   ├── llama.py
  │   ├── tokenizer.py
  │   ├── weights.py
  │   └── paged_attention.py   # NEW
  ├── core/                     # NEW
  │   ├── block_manager.py     # NEW
  │   ├── kv_cache.py          # NEW
  │   └── sequence.py          # NEW
  ├── engine/
  │   ├── generate.py          # Modified
  │   └── sampling.py
  └── main.py

  ---
  Components

  2.1 Block Manager

  Memory allocator for KV cache blocks (like malloc/free):
  - Maintains pool of free blocks
  - allocate() → returns a free block ID
  - free(block_id) → returns block to pool
  - Tracks how many blocks available

  2.2 Paged KV Cache

  Stores K,V in non-contiguous blocks:
  - Physical storage: Large GPU tensor holding all blocks
  - Each block holds K,V for N tokens (e.g., 16)
  - Blocks allocated on-demand as sequence grows
  - Blocks freed when sequence completes

  Visual:
  GPU Memory (1000 blocks × 16 tokens each):
  ┌────────┬────────┬────────┬────────┬─────┐
  │Block 0 │Block 1 │Block 2 │Block 3 │ ... │
  └────────┴────────┴────────┴────────┴─────┘

  Sequence A (45 tokens): blocks [5, 12, 77]
  Sequence B (20 tokens): blocks [3, 88]

  No wasted memory!

  2.3 Sequence State

  Tracks each request:
  - Prompt tokens
  - Generated tokens
  - Block table (which physical blocks hold this sequence's K,V)
  - Methods to append token, allocate new block if needed, free all blocks

  2.4 Paged Attention

  Modified attention that reads from scattered blocks:
  - Input: query, block table, sequence length
  - Gathers K,V from non-contiguous blocks using block table
  - Computes standard attention
  - Returns output

  Difference from normal attention:
  - Normal: K,V stored contiguously [token0, token1, token2, ...]
  - Paged: K,V scattered across blocks, gathered via block table lookup

  2.5 Modified Generation Loop

  Same as Phase 1, but:
  - Creates Sequence object with block allocation
  - Uses paged attention instead of normal
  - Allocates new blocks as tokens generated
  - Frees blocks when done

  ---
  After Phase 2 You Can

  ✅ Everything from Phase 1
  ✅ Memory-efficient KV cache (near-zero waste)
  ✅ Support much longer sequences
  ✅ Foundation ready for multi-user

  Comparison

  |                 | Phase 1 (Naive)  | Phase 2 (Paged)    |
  |-----------------|------------------|--------------------|
  | Memory per seq  | Pre-allocate max | Allocate as needed |
  | Waste           | ~90%             | ~0%                |
  | Max sequence    | Limited          | Much longer        |
  | Concurrent seqs | 1                | 1 (ready for more) |

  ---
  Still Missing (Phase 3)

  ❌ Multiple concurrent requests
  ❌ Continuous batching
  ❌ Dynamic scheduling

  ---
  Summary

  | Phase | What You Build      | Lines | Key Concept                         |
  |-------|---------------------|-------|-------------------------------------|
  | 1     | Basic inference     | ~400  | KV cache, autoregressive loop       |
  | 2     | Paged attention     | ~400  | Block allocation, memory efficiency |
  | 3     | Continuous batching | ~300  | Multi-request scheduling            |

  Phase 2 is the core innovation - it's what makes vLLM special. Understanding block-based memory management for KV cache is understanding vLLM.




## Whats already in tinygrad


What You Can Reuse for tinyvLLM

  Use directly (no changes needed)

  - SimpleTokenizer - Complete BPE tokenizer
  - Transformer.from_gguf() - Load any GGUF model
  - apply_rope() - RoPE implementation
  - TransformerBlock - Full attention + FFN

  Modify for Phase 2

  - _attention() method - Currently uses naive KV cache, replace with paged attention
  - cache_kv - Replace with PagedKVCache

  ---
  Phase 1 → Already Done!

  You can run tinygrad's LLM right now:

  # Run LLaMA 3.2 1B
  echo "Hello" | python -m tinygrad.apps.llm

  # Or with server
  python -m tinygrad.apps.llm --serve

  ---
  For tinyvLLM, You Only Need to Add

  Phase 2: Paged Attention

  NEW files to create:
  ├── core/
  │   ├── block_manager.py    # Block allocation
  │   ├── kv_cache.py         # Paged storage
  │   └── sequence.py         # Request tracking

  MODIFY:
  └── TransformerBlock._attention()  # Use paged attention

  Phase 3: Continuous Batching

  NEW files:
  ├── core/
  │   └── scheduler.py        # Batch multiple requests

  MODIFY:
  └── Transformer.generate()  # Handle multiple sequences

  ---
  Summary

  | Phase   | tinygrad Status | Your Work                        |
  |---------|-----------------|----------------------------------|
  | Phase 1 | ✅ 100% done    | Just study llm.py                |
  | Phase 2 | ❌ Not done     | Add paged attention (~400 lines) |
  | Phase 3 | ❌ Not done     | Add scheduler (~300 lines)       |

  You can skip Phase 1 entirely and start directly on Phase 2 using tinygrad's existing code as your foundation!

  ---
  Quick Start Path

  1. Study tinygrad/apps/llm.py (especially _attention() method)
  2. Create PagedKVCache class to replace self.cache_kv
  3. Create BlockManager for allocation
  4. Modify _attention() to use paged attention
  5. Add Scheduler for continuous batching

  tinygrad gives you the model, tokenizer, and basic generation for free. You just add the vLLM innovations on top.





Two types of request:
1. Prefill
    - First forward pass where the whole prompt goes in
    - Compute bound
2. Decode
    - Only the newest generated token is forward passed
    - Memory-bandwidth-bound

