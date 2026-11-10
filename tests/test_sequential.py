"""Tests for core.sequential.

Reference values come from Jennison and Turnbull, Group Sequential Methods
with Applications to Clinical Trials (2000): Table 1.1 for the false positive
rate of repeated looks (first published by Armitage, McPherson and Rowe,
1969) and Table 2.3 for the O'Brien-Fleming constants. Both tables give three
decimals, so the tolerance is half a unit in the last place.

Simulated rates are compared with exact values to within four Monte Carlo
standard errors, with fixed seeds.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from core.sequential import (
    _crossing_probability,
    obrien_fleming_boundaries,
    peeking_false_positive_rate,
    simulate_peeking,
    summarize_peeking,
)

TABLE_TOLERANCE = 5e-4


def monte_carlo_tolerance(rate: float, n_experiments: int) -> float:
    """Four standard errors of a rate estimated from n_experiments runs."""
    return 4 * math.sqrt(rate * (1 - rate) / n_experiments)


class TestPeekingFalsePositiveRate:
    @pytest.mark.parametrize(
        "n_looks, expected",
        [
            (1, 0.050),
            (2, 0.083),
            (3, 0.107),
            (4, 0.126),
            (5, 0.142),
            (10, 0.193),
            (20, 0.248),
            (50, 0.320),
        ],
    )
    def test_matches_published_table(self, n_looks, expected):
        rate = peeking_false_positive_rate(n_looks)
        assert rate == pytest.approx(expected, abs=TABLE_TOLERANCE)

    def test_single_look_gives_alpha(self):
        for alpha in [0.01, 0.05, 0.10]:
            assert peeking_false_positive_rate(1, alpha) == pytest.approx(
                alpha, abs=1e-6
            )

    def test_more_looks_give_more_false_positives(self):
        rates = [peeking_false_positive_rate(k) for k in range(1, 11)]
        assert all(later > earlier for earlier, later in zip(rates, rates[1:]))

    @pytest.mark.parametrize("n_looks, alpha", [(0, 0.05), (5, 0.0), (5, 1.0)])
    def test_rejects_impossible_inputs(self, n_looks, alpha):
        with pytest.raises(ValueError):
            peeking_false_positive_rate(n_looks, alpha)


class TestOBrienFlemingBoundaries:
    @pytest.mark.parametrize(
        "n_looks, expected_constant",
        [
            (2, 1.977),
            (3, 2.004),
            (4, 2.024),
            (5, 2.040),
            (10, 2.087),
            (15, 2.110),
            (20, 2.126),
        ],
    )
    def test_final_boundary_matches_published_table(self, n_looks, expected_constant):
        final_boundary = obrien_fleming_boundaries(n_looks)[-1]
        assert final_boundary == pytest.approx(expected_constant, abs=TABLE_TOLERANCE)

    def test_boundary_shape(self):
        """Look k of K requires |z| > c * sqrt(K / k)."""
        boundaries = obrien_fleming_boundaries(5)
        constant = boundaries[-1]
        expected = constant * np.sqrt(5 / np.arange(1, 6))
        np.testing.assert_allclose(boundaries, expected)

    def test_single_look_is_the_usual_threshold(self):
        boundaries = obrien_fleming_boundaries(1)
        assert boundaries == pytest.approx([norm.ppf(0.975)])

    @pytest.mark.parametrize("n_looks", [2, 5, 10, 20])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_false_positive_rate_is_exactly_alpha(self, n_looks, alpha):
        boundaries = obrien_fleming_boundaries(n_looks, alpha)
        assert _crossing_probability(boundaries) == pytest.approx(alpha, abs=1e-6)

    @pytest.mark.parametrize("n_looks, alpha", [(0, 0.05), (5, 0.0), (5, 1.0)])
    def test_rejects_impossible_inputs(self, n_looks, alpha):
        with pytest.raises(ValueError):
            obrien_fleming_boundaries(n_looks, alpha)


class TestSimulatePeeking:
    def test_one_row_per_experiment_and_column_per_look(self):
        z = simulate_peeking(0.05, 1_000, n_looks=4, n_experiments=500, seed=0)
        assert z.shape == (500, 4)
        assert list(z.columns) == [250, 500, 750, 1_000]

    def test_uneven_split_ends_at_full_sample(self):
        z = simulate_peeking(0.05, 1_000, n_looks=3, n_experiments=10, seed=0)
        assert list(z.columns) == [333, 666, 1_000]

    def test_same_seed_gives_same_experiments(self):
        first = simulate_peeking(0.05, 1_000, 5, seed=3)
        second = simulate_peeking(0.05, 1_000, 5, seed=3)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_give_different_experiments(self):
        first = simulate_peeking(0.05, 1_000, 5, seed=3)
        second = simulate_peeking(0.05, 1_000, 5, seed=4)
        assert not first.equals(second)

    def test_final_look_behaves_like_one_full_test(self):
        """The last column is an ordinary z-test on the full sample."""
        z = simulate_peeking(0.05, 31_218, n_looks=10, n_experiments=100_000, seed=0)
        assert z.iloc[:, -1].mean() == pytest.approx(0, abs=0.02)
        assert z.iloc[:, -1].std() == pytest.approx(1, abs=0.01)

    def test_looks_are_correlated_like_a_random_walk(self):
        """Later looks reuse earlier data: corr(z_j, z_k) = sqrt(j / k)."""
        z = simulate_peeking(0.05, 31_218, n_looks=4, n_experiments=100_000, seed=0)
        assert z.iloc[:, 0].corr(z.iloc[:, 3]) == pytest.approx(0.5, abs=0.01)
        assert z.iloc[:, 1].corr(z.iloc[:, 3]) == pytest.approx(0.707, abs=0.01)

    @pytest.mark.parametrize(
        "baseline, n_per_variant, n_looks, n_experiments",
        [
            (0.0, 1_000, 5, 100),  # baseline at 0%
            (1.0, 1_000, 5, 100),  # baseline at 100%
            (0.05, 1_000, 0, 100),  # no looks
            (0.05, 1_000, 5, 0),  # no experiments
            (0.05, 4, 5, 100),  # fewer users than looks
        ],
    )
    def test_rejects_impossible_inputs(
        self, baseline, n_per_variant, n_looks, n_experiments
    ):
        with pytest.raises(ValueError):
            simulate_peeking(baseline, n_per_variant, n_looks, n_experiments)


class TestSummarizePeeking:
    N_EXPERIMENTS = 100_000

    @pytest.mark.parametrize("n_looks", [5, 10, 20])
    def test_reproduces_peeking_table(self, n_looks):
        """The README table: 14% at 5 looks, 19% at 10, 25% at 20."""
        z = simulate_peeking(0.05, 31_218, n_looks, self.N_EXPERIMENTS, seed=0)
        expected = peeking_false_positive_rate(n_looks)
        any_look = summarize_peeking(z)["any_look"]
        assert any_look == pytest.approx(
            expected, abs=monte_carlo_tolerance(expected, self.N_EXPERIMENTS)
        )

    @pytest.mark.parametrize("n_looks", [5, 10, 20])
    def test_correction_restores_alpha(self, n_looks):
        z = simulate_peeking(0.05, 31_218, n_looks, self.N_EXPERIMENTS, seed=0)
        summary = summarize_peeking(z)
        tolerance = monte_carlo_tolerance(0.05, self.N_EXPERIMENTS)
        assert summary["obrien_fleming"] == pytest.approx(0.05, abs=tolerance)
        assert summary["final_look_only"] == pytest.approx(0.05, abs=tolerance)

    def test_correction_at_other_alpha(self):
        z = simulate_peeking(0.05, 31_218, 10, self.N_EXPERIMENTS, seed=0)
        corrected = summarize_peeking(z, alpha=0.01)["obrien_fleming"]
        assert corrected == pytest.approx(
            0.01, abs=monte_carlo_tolerance(0.01, self.N_EXPERIMENTS)
        )

    def test_single_look_rates_agree(self):
        """With one look there is nothing to correct for."""
        z = simulate_peeking(0.05, 31_218, 1, n_experiments=1_000, seed=0)
        summary = summarize_peeking(z)
        assert summary["any_look"] == summary["final_look_only"]
        assert summary["obrien_fleming"] == summary["final_look_only"]

    def test_rates_are_floats(self):
        z = simulate_peeking(0.05, 1_000, 5, n_experiments=100, seed=0)
        summary = summarize_peeking(z)
        assert set(summary) == {"final_look_only", "any_look", "obrien_fleming"}
        assert all(isinstance(rate, float) for rate in summary.values())
