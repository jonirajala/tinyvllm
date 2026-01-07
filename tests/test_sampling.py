"""Tests for sampling functions."""

import pytest
from tinygrad import Tensor

from tinyvllm.core.sampling import (
    SamplingParams,
    sample_tokens,
    _top_k_filter,
    _top_p_filter,
    _multinomial_sample,
    _repetition_penalty,
)


class TestSamplingParams:
    def test_default_values(self):
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.top_k == 40
        assert params.top_p == 0.95
        assert params.repetition_penalty == 1.0
        assert params.max_tokens == 1024

    def test_custom_values(self):
        params = SamplingParams(temperature=0.7, top_k=50, top_p=0.9)
        assert params.temperature == 0.7
        assert params.top_k == 50
        assert params.top_p == 0.9


class TestTopKFilter:
    """Tests for top-k filtering."""

    def test_keeps_top_k(self):
        """Should keep top k values, set rest to -inf."""
        logits = Tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        result = _top_k_filter(logits, k=3)
        result_list = result.realize().tolist()

        # Top 3 are indices 1 (5.0), 4 (4.0), 2 (3.0)
        assert result_list[1] == pytest.approx(5.0)
        assert result_list[4] == pytest.approx(4.0)
        assert result_list[2] == pytest.approx(3.0)

        # Rest should be -inf
        assert result_list[0] == float("-inf")
        assert result_list[3] == float("-inf")

    def test_k_larger_than_vocab(self):
        """If k >= vocab_size, return unchanged."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _top_k_filter(logits, k=10)
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)
        assert result_list[2] == pytest.approx(3.0)

    def test_k_equals_1(self):
        """k=1 should keep only the maximum."""
        logits = Tensor([1.0, 5.0, 3.0])
        result = _top_k_filter(logits, k=1)
        result_list = result.realize().tolist()
        assert result_list[1] == pytest.approx(5.0)
        assert result_list[0] == float("-inf")
        assert result_list[2] == float("-inf")

    def test_k_zero_returns_unchanged(self):
        """k=0 should return unchanged."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _top_k_filter(logits, k=0)
        result_list = result.realize().tolist()
        assert result_list == pytest.approx([1.0, 2.0, 3.0])

    def test_ties_at_threshold(self):
        """When multiple values tie at threshold, keep all of them."""
        logits = Tensor([1.0, 3.0, 3.0, 3.0, 2.0])  # three 3.0s
        result = _top_k_filter(logits, k=2)
        result_list = result.realize().tolist()
        # All 3.0s should be kept (threshold is 3.0)
        assert result_list[1] == pytest.approx(3.0)
        assert result_list[2] == pytest.approx(3.0)
        assert result_list[3] == pytest.approx(3.0)


class TestTopPFilter:
    """Tests for top-p (nucleus) filtering."""

    def test_high_p_keeps_all(self):
        """p=1.0 should keep all tokens."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _top_p_filter(logits, p=1.0)
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)
        assert result_list[2] == pytest.approx(3.0)

    def test_keeps_at_least_one(self):
        """Even with very low p, should keep at least one token."""
        logits = Tensor([1.0, 2.0, 10.0])  # token 2 is dominant
        result = _top_p_filter(logits, p=0.01)
        result_list = result.realize().tolist()

        # At least one should not be -inf
        non_inf_count = sum(1 for v in result_list if v != float("-inf"))
        assert non_inf_count >= 1

    def test_low_p_restricts_tokens(self):
        """Low p should filter out low-probability tokens."""
        logits = Tensor([0.0, 0.0, 10.0, 0.0])  # token 2 is ~99.9% probability
        result = _top_p_filter(logits, p=0.5)
        result_list = result.realize().tolist()

        # Token 2 should be kept, others might be filtered
        assert result_list[2] == pytest.approx(10.0)


class TestMultinomialSample:
    """Tests for multinomial sampling."""

    def test_returns_int(self):
        """Should return an int."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _multinomial_sample(logits)
        assert isinstance(result, int)

    def test_returns_valid_index(self):
        """Should return an index within vocab range."""
        logits = Tensor([1.0, 2.0, 3.0])
        idx = _multinomial_sample(logits)
        assert 0 <= idx < 3

    def test_higher_logits_more_likely(self):
        """Higher logits should be sampled more often."""
        logits = Tensor([0.0, 0.0, 10.0])  # token 2 is much more likely
        samples = [_multinomial_sample(logits) for _ in range(50)]
        # Token 2 should dominate
        assert samples.count(2) >= 40

    def test_uniform_logits_give_variety(self):
        """Equal logits should give roughly uniform sampling."""
        logits = Tensor([0.0, 0.0, 0.0, 0.0])
        samples = [_multinomial_sample(logits) for _ in range(100)]
        # Should see multiple different tokens
        assert len(set(samples)) >= 3

    def test_handles_negative_inf(self):
        """Should handle -inf logits (masked tokens)."""
        logits = Tensor([float("-inf"), float("-inf"), 5.0, float("-inf")])
        samples = [_multinomial_sample(logits) for _ in range(20)]
        # Only token 2 should be sampled
        assert all(s == 2 for s in samples)


class TestRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_no_penalty_when_1(self):
        """Penalty of 1.0 should not change logits."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _repetition_penalty(logits, 1.0, [0, 1])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)
        assert result_list[2] == pytest.approx(3.0)

    def test_empty_seen_tokens(self):
        """Empty seen_tokens should not change logits."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _repetition_penalty(logits, 2.0, [])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)
        assert result_list[2] == pytest.approx(3.0)

    def test_penalizes_seen_tokens(self):
        """Should penalize seen tokens."""
        logits = Tensor([2.0, 4.0, 1.0])
        result = _repetition_penalty(logits, 2.0, [0, 1])
        result_list = result.realize().tolist()

        # Positive logits should be divided by penalty
        assert result_list[0] == pytest.approx(1.0)  # 2.0 / 2.0
        assert result_list[1] == pytest.approx(2.0)  # 4.0 / 2.0
        assert result_list[2] == pytest.approx(1.0)  # unchanged


class TestSampleTokens:
    """Tests for sample_tokens function."""

    # Batch size 1 tests
    def test_single_greedy(self):
        """Temperature 0 should always return argmax."""
        logits = Tensor([[1.0, 5.0, 3.0]])
        params = [SamplingParams(temperature=0.0, top_k=0, top_p=1.0)]

        for _ in range(10):
            result = sample_tokens(logits, params)
            assert result == [1]

    def test_single_respects_top_k(self):
        """With top_k, should only sample from top k tokens."""
        logits = Tensor([[1.0, 2.0, 10.0, 3.0]])  # token 2 is highest
        params = [SamplingParams(temperature=1.0, top_k=1, top_p=1.0)]

        for _ in range(10):
            result = sample_tokens(logits, params)
            assert result == [2]

    def test_single_respects_top_p(self):
        """With low top_p, should restrict to high-probability tokens."""
        logits = Tensor([[0.0, 0.0, 10.0, 0.0]])  # token 2 dominant
        params = [SamplingParams(temperature=1.0, top_k=0, top_p=0.5)]

        samples = [sample_tokens(logits, params)[0] for _ in range(10)]
        assert samples.count(2) >= 8

    def test_single_with_repetition_penalty(self):
        """Should apply repetition penalty."""
        logits = Tensor([[10.0, 1.0, 1.0]])  # token 0 is dominant
        params = [SamplingParams(temperature=0.0, repetition_penalty=100.0)]

        # With heavy penalty on token 0, should pick another
        result = sample_tokens(logits, params, [[0]])
        assert result[0] != 0

    def test_batch_size_1_returns_list(self):
        """Batch size 1 should return list of length 1."""
        logits = Tensor([[1.0, 2.0, 3.0]])
        params = [SamplingParams(temperature=0.0)]
        result = sample_tokens(logits, params)

        assert isinstance(result, list)
        assert len(result) == 1

    # Batched tests
    def test_batch_multiple_sequences(self):
        """Should sample for all sequences in batch."""
        logits = Tensor([
            [1.0, 5.0, 3.0],  # seq 0: argmax=1
            [10.0, 2.0, 3.0],  # seq 1: argmax=0
        ])
        params_list = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)]
        result = sample_tokens(logits, params_list)

        assert len(result) == 2
        assert result[0] == 1  # seq 0 argmax
        assert result[1] == 0  # seq 1 argmax

    def test_batch_mixed_params(self):
        """Should handle different params per sequence."""
        logits = Tensor([
            [1.0, 5.0, 3.0],  # seq 0
            [1.0, 2.0, 10.0],  # seq 1
        ])
        params_list = [
            SamplingParams(temperature=0.0, top_k=0),
            SamplingParams(temperature=0.0, top_k=1),
        ]
        result = sample_tokens(logits, params_list)

        assert len(result) == 2
        assert result[0] == 1  # argmax of seq 0
        assert result[1] == 2  # argmax of seq 1

    def test_batch_with_repetition_penalty(self):
        """Should apply repetition penalty when provided."""
        logits = Tensor([[10.0, 1.0, 1.0]])  # token 0 is dominant
        params_list = [SamplingParams(temperature=0.0, repetition_penalty=100.0)]
        seen_tokens_batch = [[0]]  # penalize token 0

        result = sample_tokens(logits, params_list, seen_tokens_batch)

        # With heavy penalty on token 0, should pick another
        assert result[0] != 0

    def test_batch_returns_list_of_ints(self):
        """Should return Python list of integers."""
        logits = Tensor([[1.0, 2.0, 3.0]])
        params_list = [SamplingParams(temperature=0.0)]
        result = sample_tokens(logits, params_list)

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
