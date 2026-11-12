"""Tests for core.simulator.

Simulated rates are compared with their expected values to within four Monte
Carlo standard errors. Every test uses a fixed seed, so the results are the
same on every run.
"""

import math

import pandas as pd
import pytest

from core.power_analysis import calculate_power, calculate_sample_size
from core.simulator import simulate_experiments, summarize_simulation


def monte_carlo_tolerance(rate: float, n_experiments: int) -> float:
    """Four standard errors of a rate estimated from n_experiments runs."""
    return 4 * math.sqrt(rate * (1 - rate) / n_experiments)


class TestSimulateExperiments:
    def test_one_row_per_experiment(self):
        results = simulate_experiments(0.05, 0.10, 1_000, n_experiments=500, seed=0)
        assert len(results) == 500
        assert list(results.columns) == [
            "conversions_a",
            "conversions_b",
            "observed_lift",
            "z",
            "p_value",
            "ci_low",
            "ci_high",
            "significant",
            "ci_covers_truth",
        ]

    def test_same_seed_gives_same_experiments(self):
        first = simulate_experiments(0.05, 0.10, 1_000, seed=7)
        second = simulate_experiments(0.05, 0.10, 1_000, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_give_different_experiments(self):
        first = simulate_experiments(0.05, 0.10, 1_000, seed=7)
        second = simulate_experiments(0.05, 0.10, 1_000, seed=8)
        assert not first["conversions_a"].equals(second["conversions_a"])

    def test_observed_lift_centers_on_true_lift(self):
        results = simulate_experiments(0.05, 0.10, 31_218, seed=0)
        assert results["observed_lift"].mean() == pytest.approx(0.10, abs=0.002)

    def test_significant_means_p_below_alpha(self):
        results = simulate_experiments(0.05, 0.10, 31_218, alpha=0.01, seed=0)
        assert results["significant"].equals(results["p_value"] < 0.01)

    @pytest.mark.filterwarnings("error")
    def test_control_without_conversions_gives_nan_lift(self):
        """At a 1% rate, most groups of 20 users have no conversions."""
        results = simulate_experiments(0.01, 0.50, 20, n_experiments=1_000, seed=0)
        no_control_conversions = results["conversions_a"] == 0
        assert no_control_conversions.any()
        assert results.loc[no_control_conversions, "observed_lift"].isna().all()
        assert results.loc[~no_control_conversions, "observed_lift"].notna().all()

    @pytest.mark.parametrize(
        "baseline, lift, n_per_variant, n_experiments, alpha",
        [
            (0.0, 0.10, 1_000, 100, 0.05),  # baseline at 0%
            (1.0, 0.10, 1_000, 100, 0.05),  # baseline at 100%
            (0.50, 1.0, 1_000, 100, 0.05),  # treatment rate reaches 100%
            (0.05, -1.0, 1_000, 100, 0.05),  # treatment rate falls to 0%
            (0.05, 0.10, 0, 100, 0.05),  # no users
            (0.05, 0.10, 1_000, 0, 0.05),  # no experiments
            (0.05, 0.10, 1_000, 100, 0.0),  # alpha at 0
        ],
    )
    def test_rejects_impossible_inputs(
        self, baseline, lift, n_per_variant, n_experiments, alpha
    ):
        with pytest.raises(ValueError):
            simulate_experiments(baseline, lift, n_per_variant, n_experiments, alpha)


class TestSummarizeSimulation:
    def test_power_at_planned_sample_size(self):
        """At the Stage 1 sample size, 80% of experiments detect the lift."""
        n = calculate_sample_size(0.05, 0.10)
        results = simulate_experiments(0.05, 0.10, n, seed=0)
        win_rate = summarize_simulation(results)["treatment_win_rate"]
        assert win_rate == pytest.approx(0.80, abs=monte_carlo_tolerance(0.80, 10_000))

    @pytest.mark.parametrize(
        "baseline, lift, n_per_variant",
        [
            (0.05, 0.10, 1_000),  # 7% power
            (0.05, 0.10, 5_000),  # 20% power
            (0.10, 0.20, 3_000),  # 70% power
            (0.20, 0.05, 40_000),  # 94% power
        ],
    )
    def test_win_rate_matches_calculated_power(self, baseline, lift, n_per_variant):
        n_experiments = 100_000
        results = simulate_experiments(
            baseline, lift, n_per_variant, n_experiments, seed=0
        )
        expected = calculate_power(baseline, lift, n_per_variant)
        win_rate = summarize_simulation(results)["treatment_win_rate"]
        assert win_rate == pytest.approx(
            expected, abs=monte_carlo_tolerance(expected, n_experiments)
        )

    def test_significant_rate_includes_wrong_direction(self):
        """At low power, some significant results say treatment is worse.

        Those count towards significant_rate, which is why it overshoots the
        calculated power while treatment_win_rate does not.
        """
        n_experiments = 100_000
        results = simulate_experiments(0.05, 0.10, 1_000, n_experiments, seed=0)
        summary = summarize_simulation(results)
        expected = calculate_power(0.05, 0.10, 1_000)
        tolerance = monte_carlo_tolerance(expected, n_experiments)
        assert summary["significant_rate"] > expected + tolerance

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_false_positive_rate_matches_alpha(self, alpha):
        """With no real difference, a share alpha of experiments is significant."""
        results = simulate_experiments(0.05, 0.0, 31_218, alpha=alpha, seed=0)
        false_positive_rate = summarize_simulation(results)["significant_rate"]
        assert false_positive_rate == pytest.approx(
            alpha, abs=monte_carlo_tolerance(alpha, 10_000)
        )

    @pytest.mark.parametrize("lift", [0.0, 0.10])
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_interval_coverage_matches_confidence_level(self, lift, alpha):
        results = simulate_experiments(0.05, lift, 31_218, alpha=alpha, seed=0)
        coverage = summarize_simulation(results)["ci_coverage"]
        assert coverage == pytest.approx(
            1 - alpha, abs=monte_carlo_tolerance(1 - alpha, 10_000)
        )

    def test_harmful_change_is_rarely_declared_a_winner(self):
        """A 10% drop is usually detected, but almost never as a win."""
        results = simulate_experiments(0.05, -0.10, 31_218, seed=0)
        summary = summarize_simulation(results)
        assert summary["significant_rate"] > 0.75
        assert summary["treatment_win_rate"] < 0.001

    def test_detected_lift_is_close_to_truth_at_planned_power(self):
        results = simulate_experiments(0.05, 0.10, 31_218, seed=0)
        detected_lift = summarize_simulation(results)["mean_detected_lift"]
        assert 0.10 < detected_lift < 0.12

    def test_detected_lift_overstates_effect_at_low_power(self):
        """At 7% power only lifts of about 40% or more reach significance."""
        results = simulate_experiments(0.05, 0.10, 1_000, seed=0)
        detected_lift = summarize_simulation(results)["mean_detected_lift"]
        assert detected_lift > 0.40

    def test_detected_lift_is_nan_without_detections(self):
        results = simulate_experiments(0.05, -0.10, 31_218, n_experiments=100, seed=0)
        assert math.isnan(summarize_simulation(results)["mean_detected_lift"])

    def test_rates_are_floats(self):
        results = simulate_experiments(0.05, 0.10, 1_000, n_experiments=100, seed=0)
        summary = summarize_simulation(results)
        assert set(summary) == {
            "treatment_win_rate",
            "significant_rate",
            "ci_coverage",
            "mean_detected_lift",
        }
        assert all(isinstance(rate, float) for rate in summary.values())
