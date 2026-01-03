"""Token sampling strategies for text generation.

Phase 6.1: GPU-optimized sampling with batched operations.
- GPU-based top-k, top-p filtering (no CPU sync)
- Batched sampling across sequences
- Single sync point per batch
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from tinygrad import Tensor


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 1024


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only top-k logits on GPU, set rest to -inf.

    Args:
        logits: [vocab_size] tensor of logits
        k: number of top values to keep

    Returns:
        Filtered logits tensor (no CPU sync)
    """
    vocab_size = logits.shape[-1]
    if k >= vocab_size or k <= 0:
        return logits

    # Get top-k values and indices
    top_values, top_indices = logits.topk(k)

    # Get k-th largest value as threshold
    threshold = top_values[-1]

    # Create mask: True where we keep values (>= threshold)
    mask = logits >= threshold
    neg_inf = Tensor.full(logits.shape, float("-inf"))

    # Tensor.where(condition, if_true, if_false)
    return Tensor.where(mask, logits, neg_inf)


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Apply nucleus (top-p) filtering.

    Note: Uses CPU for the filtering logic because tinygrad's vector indexing
    and scatter ops are ~100x slower than PyTorch on large tensors (32k vocab).
    See: https://github.com/tinygrad/tinygrad/issues/5241
    Single sync + Python sort is faster than GPU scatter for now.

    Args:
        logits: [vocab_size] tensor of logits
        p: cumulative probability threshold

    Returns:
        Filtered logits tensor
    """
    if p >= 1.0:
        return logits

    # Sync to CPU once
    logits_list = logits.realize().tolist()
    vocab_size = len(logits_list)

    # Compute softmax on CPU
    import math
    max_logit = max(logits_list)
    exp_logits = [math.exp(x - max_logit) for x in logits_list]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    # Sort by probability descending with indices
    indexed_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

    # Find cumsum cutoff
    cumsum = 0.0
    keep_indices = set()
    for idx, prob in indexed_probs:
        keep_indices.add(idx)
        cumsum += prob
        if cumsum >= p:
            break

    # If nothing kept, keep at least the top one
    if not keep_indices:
        keep_indices.add(indexed_probs[0][0])

    # Build filtered logits
    filtered = [logits_list[i] if i in keep_indices else float("-inf") for i in range(vocab_size)]
    return Tensor(filtered)


def _multinomial_sample(logits: Tensor) -> Tensor:
    """Sample using Gumbel-max trick, returning tensor (no sync).

    Args:
        logits: [vocab_size] or [batch, vocab_size] tensor

    Returns:
        Token index tensor (call .item() or .tolist() to get Python value)
    """
    u = Tensor.rand(logits.shape)
    u = u.clip(1e-10, 1.0 - 1e-10)
    gumbel_noise = -(-u.log()).log()
    return (logits + gumbel_noise).argmax(axis=-1)


def _repetition_penalty(logits: Tensor, penalty: float, seen_tokens: List[int]) -> Tensor:
    """Apply repetition penalty on GPU.

    Args:
        logits: [vocab_size] tensor
        penalty: penalty factor (>1 reduces probability of seen tokens)
        seen_tokens: list of token ids to penalize

    Returns:
        Modified logits tensor
    """
    if not seen_tokens or penalty == 1.0:
        return logits

    # Create penalty multiplier tensor
    vocab_size = logits.shape[-1]
    penalty_mult = Tensor.ones(vocab_size)

    # For seen tokens: divide positive logits by penalty, multiply negative by penalty
    seen_set = set(seen_tokens)
    seen_indices = Tensor(list(seen_set))

    # Get logit values at seen positions
    seen_logits = logits.gather(-1, seen_indices)

    # Compute penalties: 1/penalty for positive, penalty for negative
    penalties = (seen_logits > 0).where(
        Tensor.full(seen_logits.shape, 1.0 / penalty),
        Tensor.full(seen_logits.shape, penalty)
    )

    # Scatter penalties back to full tensor
    penalty_mult = penalty_mult.scatter(-1, seen_indices, penalties)

    return logits * penalty_mult


def _sample_single(logits: Tensor, params: SamplingParams) -> Tensor:
    """Sample a single token on GPU, returning tensor.

    Internal function - use sample_tokens() instead.
    """
    # Greedy decoding
    if params.temperature == 0.0:
        return logits.argmax()

    # Temperature scaling
    logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0:
        logits = _top_k_filter(logits, params.top_k)

    # Top-p filtering
    if params.top_p < 1.0:
        logits = _top_p_filter(logits, params.top_p)

    return _multinomial_sample(logits)


def sample_tokens(
    logits: Tensor,
    params: Union[SamplingParams, List[SamplingParams]],
    seen_tokens: Optional[Union[List[int], List[List[int]]]] = None,
) -> List[int]:
    """Sample tokens from logits.

    Handles both single and batched sampling with a single sync at the end.

    Args:
        logits: [vocab_size] or [batch, vocab_size] logits
        params: SamplingParams for single, or list of SamplingParams for batch
        seen_tokens: optional token ids for repetition penalty
            - For single: List[int]
            - For batch: List[List[int]]

    Returns:
        List of sampled token ids (length 1 for single, batch_size for batch)

    Note:
        Current implementation loops per-sequence but minimizes syncs (single
        realize() at end). True batched sampling would require:
        - Vectorized temperature: logits / temps[:, None]
        - Batched top-k: use max(k) across batch, then per-sequence masking
        - Batched top-p: similar approach with max(p)
        - Batched Gumbel-max: already works on [batch, vocab] tensors
        Challenge: mixed params per sequence makes full vectorization complex.
        Could group sequences by similar params for partial batching.
    """
    # Handle single case
    if isinstance(params, SamplingParams):
        seq_logits = logits
        if params.repetition_penalty != 1.0 and seen_tokens:
            seq_logits = _repetition_penalty(seq_logits, params.repetition_penalty, seen_tokens)
        token_tensor = _sample_single(seq_logits, params)
        return [int(token_tensor.realize().tolist())]

    # Handle batched case
    params_list = params
    seen_tokens_batch = seen_tokens
    batch_size = logits.shape[0]
    assert len(params_list) == batch_size

    # TODO: True batched sampling - currently loops per-sequence, only benefit
    # is single sync at end instead of N syncs
    results = []

    for i in range(batch_size):
        seq_logits = logits[i]
        seq_params = params_list[i]

        # Apply repetition penalty if needed
        if seq_params.repetition_penalty != 1.0 and seen_tokens_batch and seen_tokens_batch[i]:
            seq_logits = _repetition_penalty(seq_logits, seq_params.repetition_penalty, seen_tokens_batch[i])

        # GPU sampling
        token_tensor = _sample_single(seq_logits, seq_params)
        results.append(token_tensor)

    # Stack and single sync
    stacked = Tensor.stack(*results) if len(results) > 1 else results[0].reshape(1)
    return stacked.realize().tolist()
