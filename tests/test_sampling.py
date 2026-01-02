"""Tests for sampling functions."""

import pytest
from tinygrad import Tensor

from tinyvllm.engine.sampling import (
    SamplingParams,
    sample_token,
    _repetition_penalty,
    _top_k_filter,
    _sample_with_top_p,
    _multinomial_sample,
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


class TestRepetitionPenalty:
    def test_no_penalty_when_1(self):
        """Penalty of 1.0 should not change logits."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _repetition_penalty(logits, 1.0, [0, 1])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)

    def test_positive_logits_divided(self):
        """Positive logits should be divided by penalty."""
        logits = Tensor([2.0, 4.0, 1.0])
        result = _repetition_penalty(logits, 2.0, [0, 1])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)  # 2.0 / 2.0
        assert result_list[1] == pytest.approx(2.0)  # 4.0 / 2.0
        assert result_list[2] == pytest.approx(1.0)  # unchanged

    def test_negative_logits_multiplied(self):
        """Negative logits should be multiplied by penalty (more negative)."""
        logits = Tensor([-2.0, 4.0, 1.0])
        result = _repetition_penalty(logits, 2.0, [0])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(-4.0)  # -2.0 * 2.0
        assert result_list[1] == pytest.approx(4.0)   # unchanged

    def test_unseen_tokens_unchanged(self):
        """Tokens not in seen_tokens should not be affected."""
        logits = Tensor([1.0, 2.0, 3.0, 4.0])
        result = _repetition_penalty(logits, 2.0, [0])
        result_list = result.realize().tolist()
        assert result_list[2] == pytest.approx(3.0)
        assert result_list[3] == pytest.approx(4.0)

    def test_empty_seen_tokens(self):
        """Empty seen_tokens should not change logits."""
        logits = Tensor([1.0, 2.0, 3.0])
        result = _repetition_penalty(logits, 2.0, [])
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(1.0)
        assert result_list[1] == pytest.approx(2.0)
        assert result_list[2] == pytest.approx(3.0)

    def test_duplicate_seen_tokens(self):
        """Duplicate tokens should only be penalized once."""
        logits = Tensor([4.0, 2.0, 3.0])
        result = _repetition_penalty(logits, 2.0, [0, 0, 0])  # token 0 repeated
        result_list = result.realize().tolist()
        assert result_list[0] == pytest.approx(2.0)  # 4.0 / 2.0, not 4.0 / 8.0


class TestTopKFilter:
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

    def test_ties_at_threshold(self):
        """When multiple values tie at threshold, keep all of them."""
        logits = Tensor([1.0, 3.0, 3.0, 3.0, 2.0])  # three 3.0s
        result = _top_k_filter(logits, k=2)
        result_list = result.realize().tolist()
        # All 3.0s should be kept (threshold is 3.0)
        assert result_list[1] == pytest.approx(3.0)
        assert result_list[2] == pytest.approx(3.0)
        assert result_list[3] == pytest.approx(3.0)


class TestTopPSampling:
    def test_returns_valid_index(self):
        """Should return an index within vocab range."""
        logits = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        idx = _sample_with_top_p(logits, p=0.9)
        assert 0 <= idx < 5

    def test_high_p_can_sample_any(self):
        """With p=0.99, almost all tokens are candidates."""
        logits = Tensor([1.0, 1.0, 1.0, 1.0])
        samples = [_sample_with_top_p(logits, p=0.99) for _ in range(20)]
        # Should have some variety
        assert len(set(samples)) > 1

    def test_low_p_restricts_to_top(self):
        """With very low p, should mostly sample top tokens."""
        logits = Tensor([0.0, 0.0, 10.0, 0.0])  # token 2 is dominant
        samples = [_sample_with_top_p(logits, p=0.5) for _ in range(10)]
        # Most samples should be token 2
        assert samples.count(2) >= 8

    def test_very_low_p_keeps_at_least_one(self):
        """Even with p close to 0, should keep at least one token."""
        logits = Tensor([1.0, 2.0, 3.0])
        idx = _sample_with_top_p(logits, p=0.001)
        assert 0 <= idx < 3  # Should still return valid index


class TestMultinomialSample:
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


class TestSampleToken:
    def test_greedy_with_temp_0(self):
        """Temperature 0 should always return argmax."""
        logits = Tensor([1.0, 5.0, 3.0])
        params = SamplingParams(temperature=0.0, top_k=0, top_p=1.0)

        for _ in range(10):
            assert sample_token(logits, params, []) == 1

    def test_respects_top_k(self):
        """With top_k, should only sample from top k tokens."""
        logits = Tensor([1.0, 2.0, 10.0, 3.0])  # token 2 is highest
        params = SamplingParams(temperature=1.0, top_k=1, top_p=1.0)

        for _ in range(10):
            assert sample_token(logits, params, []) == 2

    def test_respects_top_p(self):
        """With low top_p, should restrict to high-probability tokens."""
        logits = Tensor([0.0, 0.0, 10.0, 0.0])  # token 2 dominant
        params = SamplingParams(temperature=1.0, top_k=0, top_p=0.5)

        samples = [sample_token(logits, params, []) for _ in range(10)]
        assert samples.count(2) >= 8

    def test_respects_repetition_penalty(self):
        """Repetition penalty should reduce probability of seen tokens."""
        logits = Tensor([10.0, 1.0, 1.0])
        params = SamplingParams(temperature=0.0, repetition_penalty=100.0, top_k=0, top_p=1.0)

        # Without penalty: token 0 wins
        assert sample_token(logits, SamplingParams(temperature=0.0), []) == 0

        # With heavy penalty on token 0: others should win
        result = sample_token(logits, params, [0])
        assert result != 0

    def test_temperature_affects_randomness(self):
        """Higher temperature should increase sampling variety."""
        logits = Tensor([3.0, 2.0, 1.0])  # token 0 is most likely

        # Low temperature - should mostly pick token 0
        params_low = SamplingParams(temperature=0.1, top_k=0, top_p=1.0)
        samples_low = [sample_token(logits, params_low, []) for _ in range(20)]

        # High temperature - should have more variety
        params_high = SamplingParams(temperature=2.0, top_k=0, top_p=1.0)
        samples_high = [sample_token(logits, params_high, []) for _ in range(20)]

        # Low temp should be less varied than high temp
        assert len(set(samples_low)) <= len(set(samples_high)) or samples_low.count(0) > samples_high.count(0)

    def test_disabled_features(self):
        """Test with all optional features disabled."""
        logits = Tensor([1.0, 2.0, 3.0])
        params = SamplingParams(
            temperature=1.0,
            top_k=0,       # disabled
            top_p=1.0,     # disabled
            repetition_penalty=1.0  # disabled
        )
        idx = sample_token(logits, params, [])
        assert 0 <= idx < 3
