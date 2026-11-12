"""Tests for ui.charts: outcome labels and the counts behind each chart."""

import numpy as np
import pandas as pd
import pytest

from core.sequential import obrien_fleming_boundaries, simulate_peeking
from core.simulator import simulate_experiments
from ui import charts


def bar_total(figure) -> int:
    """Total count across the histogram bars of a figure."""
    return int(sum(trace.y.sum() for trace in figure.data if trace.type == "bar"))


class TestOutcomeLabels:
    def test_real_lift_separates_detections_from_wrong_direction(self):
        results = pd.DataFrame(
            {"significant": [True, True, False, False], "z": [2.5, -2.5, 1.0, -1.0]}
        )
        assert list(charts.outcome_labels(results, true_lift=0.10)) == [
            charts.DETECTED,
            charts.WRONG_DIRECTION,
            charts.NOT_SIGNIFICANT,
            charts.NOT_SIGNIFICANT,
        ]

    def test_no_lift_makes_every_significant_result_a_false_positive(self):
        results = pd.DataFrame(
            {"significant": [True, True, False], "z": [2.5, -2.5, 0]}
        )
        assert list(charts.outcome_labels(results, true_lift=0)) == [
            charts.FALSE_POSITIVE,
            charts.FALSE_POSITIVE,
            charts.NOT_SIGNIFICANT,
        ]


class TestHistograms:
    def test_lift_histogram_counts_every_experiment(self):
        """Lifts outside the plotted range go into the outermost bars."""
        results = simulate_experiments(0.05, 0.10, 1_000, seed=0)
        figure = charts.observed_lift_histogram(results, 0.10)
        assert bar_total(figure) == results["observed_lift"].notna().sum()

    def test_lift_histogram_without_defined_lifts(self):
        """With one user per variant, most controls have no conversions."""
        results = simulate_experiments(0.05, 0.10, 1, n_experiments=10, seed=0)
        results["observed_lift"] = np.nan
        figure = charts.observed_lift_histogram(results, 0.10)
        assert bar_total(figure) == 0

    @pytest.mark.parametrize("alpha, n_bars", [(0.01, 100), (0.05, 20), (0.10, 20)])
    def test_p_value_bars_align_with_alpha(self, alpha, n_bars):
        results = simulate_experiments(0.05, 0.10, 31_218, alpha=alpha, seed=0)
        figure = charts.p_value_histogram(results, 0.10, alpha)
        bars = [trace for trace in figure.data if trace.type == "bar"]
        assert all(len(bar.x) == n_bars for bar in bars)
        assert bar_total(figure) == len(results)

        not_significant = next(b for b in bars if b.name == charts.NOT_SIGNIFICANT)
        below_alpha = not_significant.x < alpha
        assert not_significant.y[below_alpha].sum() == 0


class TestIntervalChart:
    def test_axis_covers_every_interval(self):
        results = simulate_experiments(0.05, 0.10, 1_000, seed=0)
        figure = charts.interval_chart(results, 0.005, n_shown=50)
        low, high = figure.layout.xaxis.range
        shown = results.head(50)
        assert low < shown["ci_low"].min() * 100
        assert high > shown["ci_high"].max() * 100


class TestZPathsChart:
    def test_red_paths_are_the_ones_that_cross(self):
        z_by_look = simulate_peeking(0.05, 31_218, 10, seed=0)
        figure = charts.z_paths_chart(
            z_by_look, 1.96, obrien_fleming_boundaries(10), n_shown=50
        )
        crossed = next(t for t in figure.data if t.name.startswith("Crossed"))
        n_red_paths = np.isnan(crossed.y).sum()  # one gap after each path
        expected = (z_by_look.head(50).abs() > 1.96).any(axis=1).sum()
        assert n_red_paths == expected
