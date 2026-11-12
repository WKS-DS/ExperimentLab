"""Tests for core.power_analysis."""

import itertools
import math

import pytest
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

from core.power_analysis import calculate_power, calculate_sample_size, cohens_h


class TestCalculateSampleSize:
    def test_known_value(self):
        """5% baseline, 10% lift (5.0% -> 5.5%), alpha 0.05, 80% power."""
        assert calculate_sample_size(0.05, 0.10) == 31_218

    def test_close_to_evan_miller_calculator(self):
        """https://www.evanmiller.org/ab-testing/sample-size.html gives 31,234.

        That calculator uses a slightly different variance approximation, so
        the answers agree to within 0.1% rather than exactly.
        """
        assert calculate_sample_size(0.05, 0.10) == pytest.approx(31_234, rel=0.001)

    def test_matches_statsmodels_across_inputs(self):
        """Matches statsmodels' numerical solver over 300 input combinations.

        A two-sided test detects a real lift exactly when a one-sided test at
        alpha / 2 does. statsmodels' two-sided power also counts significant
        results in the wrong direction, so the reference is the one-sided solver.
        """
        for baseline, lift, alpha, power in itertools.product(
            [0.01, 0.05, 0.10, 0.20, 0.40],
            [0.02, 0.05, 0.10, 0.25, 0.50],
            [0.01, 0.05, 0.10],
            [0.70, 0.80, 0.90, 0.95],
        ):
            h = proportion_effectsize(baseline * (1 + lift), baseline)
            expected = zt_ind_solve_power(
                effect_size=h, alpha=alpha / 2, power=power, alternative="larger"
            )
            n = calculate_sample_size(baseline, lift, alpha, power)
            assert n == math.ceil(expected), (baseline, lift, alpha, power)

    def test_higher_baseline_needs_smaller_sample(self):
        n_low_baseline = calculate_sample_size(0.05, 0.10)
        n_high_baseline = calculate_sample_size(0.20, 0.10)
        assert n_high_baseline < n_low_baseline

    def test_larger_effect_needs_smaller_sample(self):
        n_small_effect = calculate_sample_size(0.05, 0.05)
        n_large_effect = calculate_sample_size(0.05, 0.20)
        assert n_large_effect < n_small_effect

    def test_higher_power_needs_larger_sample(self):
        n_80_power = calculate_sample_size(0.05, 0.10, power=0.80)
        n_90_power = calculate_sample_size(0.05, 0.10, power=0.90)
        assert n_90_power > n_80_power

    def test_returns_integer(self):
        assert isinstance(calculate_sample_size(0.05, 0.10), int)

    @pytest.mark.parametrize(
        "baseline, lift, alpha, power",
        [
            (0.0, 0.10, 0.05, 0.80),  # baseline at 0%
            (1.0, 0.10, 0.05, 0.80),  # baseline at 100%
            (0.05, 0.0, 0.05, 0.80),  # no lift
            (0.05, -0.10, 0.05, 0.80),  # negative lift
            (0.50, 1.0, 0.05, 0.80),  # treatment rate reaches 100%
            (0.05, 0.10, 0.0, 0.80),  # alpha at 0
            (0.05, 0.10, 0.05, 1.0),  # power at 100%
        ],
    )
    def test_rejects_impossible_inputs(self, baseline, lift, alpha, power):
        with pytest.raises(ValueError):
            calculate_sample_size(baseline, lift, alpha, power)


class TestCalculatePower:
    def test_sample_size_is_smallest_that_reaches_target(self):
        """One user fewer than calculate_sample_size falls short of 80% power."""
        n = calculate_sample_size(0.05, 0.10, power=0.80)
        assert calculate_power(0.05, 0.10, n) >= 0.80
        assert calculate_power(0.05, 0.10, n - 1) < 0.80

    def test_matches_statsmodels(self):
        """Compared one-sided at alpha / 2, as in the sample size test."""
        h = proportion_effectsize(0.055, 0.05)
        for n in [1_000, 10_000, 50_000]:
            expected = zt_ind_solve_power(
                effect_size=h, nobs1=n, alpha=0.025, alternative="larger"
            )
            assert calculate_power(0.05, 0.10, n) == pytest.approx(expected)

    def test_power_increases_with_sample_size(self):
        assert calculate_power(0.05, 0.10, 10_000) > calculate_power(0.05, 0.10, 1_000)

    def test_power_increases_with_effect_size(self):
        assert calculate_power(0.05, 0.40, 5_000) > calculate_power(0.05, 0.10, 5_000)

    def test_rejects_non_positive_sample_size(self):
        with pytest.raises(ValueError):
            calculate_power(0.05, 0.10, 0)


class TestCohensH:
    @pytest.mark.parametrize(
        "baseline, lift", [(0.05, 0.10), (0.01, 0.50), (0.30, 0.20), (0.49, 1.0)]
    )
    def test_matches_statsmodels(self, baseline, lift):
        expected = proportion_effectsize(baseline * (1 + lift), baseline)
        assert cohens_h(baseline, lift) == pytest.approx(expected, rel=1e-12)

    def test_same_point_change_is_larger_on_a_low_baseline(self):
        """Half a percentage point: 1% to 1.5%, and 50% to 50.5%."""
        assert cohens_h(0.01, 0.50) > 4 * cohens_h(0.50, 0.01)
