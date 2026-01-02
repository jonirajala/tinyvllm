"""Token sampling strategies for text generation."""

from dataclasses import dataclass
from typing import List
import math

from tinygrad import Tensor


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 1024


def sample_token(logits: Tensor, params: SamplingParams, seen_tokens: List[int]) -> int:
    # Apply repetition penalty
    if params.repetition_penalty != 1.0 and seen_tokens:
        logits = _repetition_penalty(logits, params.repetition_penalty, seen_tokens)

    # Greedy decoding for temperature = 0
    if params.temperature == 0.0:
        return int(logits.argmax().realize().tolist())

    # Temperature scaling
    logits = logits / params.temperature

    # Top-k filtering
    if params.top_k > 0:
        logits = _top_k_filter(logits, params.top_k)

    # Top-p filtering
    if params.top_p < 1.0:
        return _sample_with_top_p(logits, params.top_p)

    return _multinomial_sample(logits)


def _repetition_penalty(logits: Tensor, penalty: float, seen_tokens: List[int]) -> Tensor:
    """Penalize tokens that have already appeared."""
    logits_list = logits.realize().tolist()
    seen_set = set(seen_tokens)

    # Build penalty multiplier
    penalty_mult = [1.0] * len(logits_list)

    for token in seen_set:
        val = logits_list[token]
        if val > 0:
            penalty_mult[token] = 1.0 / penalty
        else:
            penalty_mult[token] = penalty

    return logits * Tensor(penalty_mult)


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only top-k logits, set rest to -inf."""
    logits_list = logits.realize().tolist()
    vocab_size = len(logits_list)

    if k >= vocab_size:
        return logits

    # Find threshold (k-th largest value)
    sorted_vals = sorted(logits_list, reverse=True)
    threshold = sorted_vals[k - 1]

    # Mask values below threshold
    result = [v if v >= threshold else float("-inf") for v in logits_list]
    return Tensor(result)


def _sample_with_top_p(logits: Tensor, p: float) -> int:
    """Sample with top-p filtering, returning original token index."""
    logits_list = logits.realize().tolist()

    # Compute softmax manually
    max_logit = max(logits_list)
    exp_logits = [math.exp(x - max_logit) for x in logits_list]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    # Sort by probability descending, keeping track of original indices
    indexed_probs = [(prob, idx) for idx, prob in enumerate(probs)]
    indexed_probs.sort(reverse=True, key=lambda x: x[0])

    # Find cumsum cutoff
    cumsum = 0.0
    keep_indices = []
    for prob, idx in indexed_probs:
        if cumsum < p:
            keep_indices.append(idx)
            cumsum += prob
        else:
            break

    # If nothing kept, keep at least the top one
    if not keep_indices:
        keep_indices = [indexed_probs[0][1]]

    # Filter logits - set non-kept to -inf
    keep_set = set(keep_indices)
    filtered = [logits_list[i] if i in keep_set else float("-inf") for i in range(len(logits_list))]

    return _multinomial_sample(Tensor(filtered))


def _multinomial_sample(logits: Tensor) -> int:
    """Sample from probability distribution using Gumbel-max trick."""
    # Gumbel-max: argmax(logits + Gumbel_noise) ~ Categorical(softmax(logits))
    # Gumbel noise = -log(-log(U)) where U ~ Uniform(0,1)
    u = Tensor.rand(logits.shape)
    u = u.clip(1e-10, 1.0 - 1e-10)
    gumbel_noise = -(-u.log()).log()

    return int((logits + gumbel_noise).argmax().realize().tolist())
