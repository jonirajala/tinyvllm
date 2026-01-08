"""Token sampling strategies for text generation."""

import math
from dataclasses import dataclass
from typing import List, Optional  # Optional used in sample_tokens signature

from tinygrad import Tensor

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    max_tokens: int = 1024


@dataclass
class SamplingMetadata:
    """Sampling parameters aggregated with optimization flags."""
    temperatures: List[float]
    top_ks: List[int]
    top_ps: List[float]
    repetition_penalties: List[float]

    all_greedy: bool       # Skip random sampling path
    needs_penalties: bool  # Skip repetition penalty loop

    @classmethod
    def from_params(cls, params: List[SamplingParams]) -> 'SamplingMetadata':
        """Construct metadata from list of SamplingParams."""
        temps = [p.temperature for p in params]
        top_ks = [p.top_k for p in params]
        top_ps = [p.top_p for p in params]
        rep_pens = [p.repetition_penalty for p in params]

        return cls(
            temperatures=temps,
            top_ks=top_ks,
            top_ps=top_ps,
            repetition_penalties=rep_pens,
            all_greedy=all(t < _SAMPLING_EPS for t in temps),
            needs_penalties=any(rp != 1.0 for rp in rep_pens),
        )


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep only top-k logits, set rest to -inf."""
    if k >= logits.shape[-1] or k <= 0:
        return logits
    logits_list = logits.realize().tolist()
    threshold = sorted(logits_list, reverse=True)[k - 1]
    return Tensor([v if v >= threshold else float("-inf") for v in logits_list])


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Apply nucleus (top-p) filtering."""
    if p >= 1.0:
        return logits

    logits_list = logits.realize().tolist()
    max_logit = max(logits_list)
    exp_logits = [math.exp(x - max_logit) for x in logits_list]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]

    indexed_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    cumsum, keep_indices = 0.0, set()
    for idx, prob in indexed_probs:
        keep_indices.add(idx)
        cumsum += prob
        if cumsum >= p:
            break
    if not keep_indices:
        keep_indices.add(indexed_probs[0][0])

    return Tensor([logits_list[i] if i in keep_indices else float("-inf")
                   for i in range(len(logits_list))])


def _apply_repetition_penalty(
    logits: Tensor,
    metadata: SamplingMetadata,
    seen_tokens: List[List[int]],
) -> Tensor:
    """Apply repetition penalty. Loops per-sequence (variable seen_tokens)."""
    if not metadata.needs_penalties:
        return logits

    batch_size = logits.shape[0]
    results = []
    for i in range(batch_size):
        seq_logits = logits[i]
        penalty = metadata.repetition_penalties[i]
        seq_seen = seen_tokens[i] if seen_tokens else []

        if penalty != 1.0 and seq_seen:
            # Apply penalty: divide positive logits, multiply negative
            penalty_mult = Tensor.ones(seq_logits.shape[-1])
            seen_indices = Tensor(list(set(seq_seen)))
            seen_logits = seq_logits.gather(-1, seen_indices)
            penalties = (seen_logits > 0).where(
                Tensor.full(seen_logits.shape, 1.0 / penalty),
                Tensor.full(seen_logits.shape, penalty)
            )
            penalty_mult = penalty_mult.scatter(-1, seen_indices, penalties)
            seq_logits = seq_logits * penalty_mult

        results.append(seq_logits)
    return Tensor.stack(results)


def _apply_temperature(logits: Tensor, metadata: SamplingMetadata) -> Tensor:
    """Apply temperature scaling. logits: [batch, vocab]"""
    if metadata.all_greedy:
        return logits
    temp = Tensor(metadata.temperatures).unsqueeze(1).clip(min_=_SAMPLING_EPS)
    return logits / temp


def _gumbel_sample(logits: Tensor) -> Tensor:
    """Sample using Gumbel-max trick. logits: [batch, vocab] -> [batch]"""
    u = Tensor.rand(logits.shape).clip(1e-10, 1.0 - 1e-10)
    gumbel_noise = -(-u.log()).log()
    return (logits + gumbel_noise).argmax(axis=-1)


def _sample_mixed_strategy(logits: Tensor, metadata: SamplingMetadata) -> Tensor:
    """Sample combining greedy and random strategies per-sequence."""
    if metadata.all_greedy:
        return logits.argmax(axis=-1)

    greedy_tokens = logits.argmax(axis=-1)
    random_tokens = _gumbel_sample(logits)
    is_greedy = Tensor(metadata.temperatures) < _SAMPLING_EPS
    return is_greedy.where(greedy_tokens, random_tokens)


def _apply_filtering(logits: Tensor, metadata: SamplingMetadata) -> Tensor:
    """Apply top-k and top-p filtering. Loops per-row (CPU-based sorting)."""
    batch_size = logits.shape[0]

    # Skip if all sequences disable top-k (k=0 or k >= vocab)
    needs_top_k = any(0 < k < logits.shape[-1] for k in metadata.top_ks)
    if needs_top_k:
        logits = Tensor.stack([_top_k_filter(logits[i], metadata.top_ks[i])
                               for i in range(batch_size)])

    # Skip if all sequences disable top-p (p >= 1.0)
    needs_top_p = any(p < 1.0 for p in metadata.top_ps)
    if needs_top_p:
        logits = Tensor.stack([_top_p_filter(logits[i], metadata.top_ps[i])
                               for i in range(batch_size)])

    return logits


def sample_tokens(
    logits: Tensor,
    params: List[SamplingParams],
    seen_tokens: Optional[List[List[int]]] = None,
) -> List[int]:
    """Sample tokens from logits."""
    batch_size = logits.shape[0]
    assert len(params) == batch_size

    metadata = SamplingMetadata.from_params(params)

    if seen_tokens and metadata.needs_penalties:
        logits = _apply_repetition_penalty(logits, metadata, seen_tokens)

    logits = _apply_temperature(logits, metadata)
    logits = _apply_filtering(logits, metadata)
    sampled = _sample_mixed_strategy(logits, metadata)

    return sampled.tolist()
