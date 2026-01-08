"""Token sampling strategies for text generation."""

import math
from dataclasses import dataclass
from typing import List, Optional

from tinygrad import Tensor


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 1024


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only top-k logits, set rest to -inf. Uses CPU (faster than tinygrad GPU topk)."""
    if k >= logits.shape[-1] or k <= 0:
        return logits

    logits_list = logits.realize().tolist()
    threshold = sorted(logits_list, reverse=True)[k - 1]
    filtered = [v if v >= threshold else float("-inf") for v in logits_list]
    return Tensor(filtered)


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Apply nucleus (top-p) filtering. Uses CPU (faster than tinygrad GPU scatter)."""
    if p >= 1.0:
        return logits

    logits_list = logits.realize().tolist()

    # Softmax
    max_logit = max(logits_list)
    exp_logits = [math.exp(x - max_logit) for x in logits_list]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    # Keep tokens until cumulative prob >= p
    indexed_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    cumsum = 0.0
    keep_indices = set()
    for idx, prob in indexed_probs:
        keep_indices.add(idx)
        cumsum += prob
        if cumsum >= p:
            break

    if not keep_indices:
        keep_indices.add(indexed_probs[0][0])

    filtered = [logits_list[i] if i in keep_indices else float("-inf") for i in range(len(logits_list))]
    return Tensor(filtered)


def _multinomial_sample(logits: Tensor) -> int:
    """Sample using Gumbel-max trick."""
    u = Tensor.rand(logits.shape).clip(1e-10, 1.0 - 1e-10)
    gumbel_noise = -(-u.log()).log()
    return int((logits + gumbel_noise).argmax(axis=-1).item())


def _repetition_penalty(logits: Tensor, penalty: float, seen_tokens: List[int]) -> Tensor:
    """Apply repetition penalty. Divides positive logits, multiplies negative by penalty."""
    if not seen_tokens or penalty == 1.0:
        return logits

    penalty_mult = Tensor.ones(logits.shape[-1])
    seen_indices = Tensor(list(set(seen_tokens)))
    seen_logits = logits.gather(-1, seen_indices)

    penalties = (seen_logits > 0).where(
        Tensor.full(seen_logits.shape, 1.0 / penalty),
        Tensor.full(seen_logits.shape, penalty)
    )
    penalty_mult = penalty_mult.scatter(-1, seen_indices, penalties)

    return logits * penalty_mult


def _sample_single(logits: Tensor, params: SamplingParams) -> int:
    """Sample a single token. TODO: vectorize for true batched sampling."""
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
    """Sample tokens from logits. TODO: vectorize loop for true batched sampling."""
    batch_size = logits.shape[0]
    assert len(params) == batch_size

    results = []
    for i in range(batch_size):
        seq_logits, seq_params = logits[i], params[i]
        if seq_params.repetition_penalty != 1.0 and seen_tokens and seen_tokens[i]:
            seq_logits = _repetition_penalty(seq_logits, seq_params.repetition_penalty, seen_tokens[i])
        results.append(_sample_single(seq_logits, seq_params))
    return results
