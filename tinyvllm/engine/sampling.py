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


def _multinomial_sample(logits: Tensor) -> int:
    """Sample using Gumbel-max trick.

    Args:
        logits: [vocab_size] tensor

    Returns:
        Sampled token index
    """
    u = Tensor.rand(logits.shape)
    u = u.clip(1e-10, 1.0 - 1e-10)
    gumbel_noise = -(-u.log()).log()
    return int((logits + gumbel_noise).argmax(axis=-1).item())


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


def _sample_single(logits: Tensor, params: SamplingParams) -> int:
    """Sample a single token.

    TODO: Replace with true batched sampling - vectorize temperature/top_k/top_p
    across batch dimension to avoid per-sequence loop in sample_tokens().
    """
    if params.temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits / params.temperature
    if params.top_k > 0:
        logits = _top_k_filter(logits, params.top_k)
    if params.top_p < 1.0:
        logits = _top_p_filter(logits, params.top_p)

    return _multinomial_sample(logits)


def sample_tokens(
    logits: Tensor,
    params: List[SamplingParams],
    seen_tokens: Optional[List[List[int]]] = None,
) -> List[int]:
    """Sample tokens from logits (batched).

    TODO: True batched sampling - vectorize the loop below. Currently loops
    per-sequence because params can differ. Could group by similar params
    or use masked operations for temperature/top_k/top_p.

    Args:
        logits: [batch, vocab_size] logits
        params: list of SamplingParams, one per sequence
        seen_tokens: optional list of token id lists for repetition penalty

    Returns:
        List of sampled token ids (length = batch_size)
    """
    batch_size = logits.shape[0]
    assert len(params) == batch_size

    results = []
    for i in range(batch_size):
        seq_logits, seq_params = logits[i], params[i]
        if seq_params.repetition_penalty != 1.0 and seen_tokens and seen_tokens[i]:
            seq_logits = _repetition_penalty(seq_logits, seq_params.repetition_penalty, seen_tokens[i])
        results.append(_sample_single(seq_logits, seq_params))
    return results

