"""
The three sections of the app: sample size, simulator and peeking.

Each function reads the inputs specific to its section, runs the calculations
in core/, and shows simulated results next to the values the formulas
predict. Simulations are cached, so changing an input in another section does
not rerun them.
"""

import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from core.power_analysis import calculate_power, calculate_sample_size
from core.sequential import (
    obrien_fleming_boundaries,
    peeking_false_positive_rate,
    simulate_peeking,
    summarize_peeking,
)
from core.simulator import simulate_experiments, summarize_simulation
from ui import charts

EXPERIMENT_OPTIONS = [1_000, 10_000, 50_000]

# Boundaries for 20 looks take about 50 ms to compute; many more would slow
# the app noticeably.
MAX_LOOKS = 20

# Below this many expected conversions per variant at the first look, the
# z-test is noticeably discrete and simulated peeking rates fall below the
# exact value.
MIN_CONVERSIONS_PER_LOOK = 10

CACHE_SIZE = 32


def show_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float,
    power: float,
) -> int:
    """Show the required sample size and how power grows with it.

    Returns:
        Users needed per variant, which the other sections use as a default.
    """
    n_required = calculate_sample_size(
        baseline_rate, minimum_detectable_effect, alpha, power
    )
    treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

    st.header("Sample size")
    st.write(
        f"Users needed to detect a change from {baseline_rate:.2%} to "
        f"{treatment_rate:.2%} conversion with a two-sided test at "
        f"α = {alpha:g} and {power:.0%} power."
    )
    per_variant, total = st.columns(2)
    per_variant.metric("Users per variant", f"{n_required:,}")
    total.metric("Users in total", f"{2 * n_required:,}")

    n_values = np.unique(np.linspace(max(1, n_required / 20), 3 * n_required, 100))
    power_values = [
        calculate_power(baseline_rate, minimum_detectable_effect, n, alpha)
        for n in n_values
    ]
    st.plotly_chart(charts.power_curve(n_values, power_values, n_required, power))
    return n_required


def show_simulator(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float,
    n_required: int,
) -> None:
    """Simulate many experiments and compare the results with the formulas."""
    st.header("Simulator")
    st.write(
        "Runs the experiment thousands of times on simulated users with a "
        "known true lift and analyzes each run with a two-proportion z-test."
    )
    users_column, lift_column, count_column = st.columns(3)
    n_per_variant = users_column.number_input(
        "Users per variant",
        min_value=1,
        value=n_required,
        step=1_000,
        help="Starts at the calculated sample size.",
    )
    true_lift = (
        lift_column.slider(
            "True lift (%)",
            min_value=0,
            max_value=100,
            value=round(minimum_detectable_effect * 100),
            help="Starts at the minimum detectable effect. Set it to 0 for an "
            "A/A test, where every significant result is a false positive.",
        )
        / 100
    )
    n_experiments = count_column.selectbox(
        "Simulated experiments",
        EXPERIMENT_OPTIONS,
        index=1,
        format_func="{:,}".format,
        key="n_experiments",
    )
    seed = _redraw_button("simulator")

    try:
        results = _simulate_experiments(
            baseline_rate, true_lift, int(n_per_variant), n_experiments, alpha, seed
        )
    except ValueError as error:
        st.error(str(error))
        return
    summary = summarize_simulation(results)

    if true_lift > 0:
        expected_power = calculate_power(
            baseline_rate, true_lift, int(n_per_variant), alpha
        )
        rows = [
            (
                "Detected the lift (power)",
                summary["treatment_win_rate"],
                expected_power,
            )
        ]
    else:
        rows = [("False positives", summary["significant_rate"], alpha)]
    rows.append(
        ("Intervals containing the true difference", summary["ci_coverage"], 1 - alpha)
    )
    st.dataframe(_comparison_table(rows, n_experiments), hide_index=True)

    undefined_lifts = int(results["observed_lift"].isna().sum())
    if undefined_lifts:
        st.caption(
            f"{undefined_lifts:,} experiments had no conversions in control, so "
            "their relative lift is undefined and they are left out of the "
            "lift histogram."
        )

    lift_column, p_value_column = st.columns(2)
    lift_column.plotly_chart(charts.observed_lift_histogram(results, true_lift))
    p_value_column.plotly_chart(charts.p_value_histogram(results, alpha))
    st.plotly_chart(charts.interval_chart(results, baseline_rate * true_lift))


def show_peeking(baseline_rate: float, alpha: float, n_required: int) -> None:
    """Simulate A/A tests checked repeatedly, with and without a correction."""
    st.header("Peeking")
    st.write(
        "Simulates A/A tests, where both groups are identical, and checks "
        "each one at evenly spaced points. Every significant result is a "
        "false positive."
    )
    looks_column, users_column = st.columns(2)
    n_looks = looks_column.slider(
        "Looks at the data", min_value=1, max_value=MAX_LOOKS, value=10, key="looks"
    )
    n_per_variant = users_column.number_input(
        "Users per variant at the final look",
        min_value=MAX_LOOKS,
        value=max(n_required, MAX_LOOKS),
        step=1_000,
        help="Starts at the calculated sample size.",
    )
    seed = _redraw_button("peeking")

    z_by_look, summary = _simulate_peeking(
        baseline_rate, int(n_per_variant), n_looks, alpha, seed
    )
    rows = [
        ("Analyze once, at the final look", summary["final_look_only"], alpha),
        (
            f"Check every look, stop at the first p < {alpha:g}",
            summary["any_look"],
            _peeking_rates(alpha)[n_looks - 1],
        ),
        (
            "Check every look with O'Brien-Fleming boundaries",
            summary["obrien_fleming"],
            alpha,
        ),
    ]
    st.dataframe(_comparison_table(rows, len(z_by_look)), hide_index=True)

    first_look_conversions = baseline_rate * z_by_look.columns[0]
    if first_look_conversions < MIN_CONVERSIONS_PER_LOOK:
        st.caption(
            f"Each variant expects only {first_look_conversions:.1f} conversions "
            "by the first look. With so few, the z-test is only approximate and "
            "the simulated rates can sit a little below the expected ones."
        )

    threshold = norm.ppf(1 - alpha / 2)
    boundaries = _boundaries(n_looks, alpha)
    st.plotly_chart(charts.z_paths_chart(z_by_look, threshold, boundaries))
    looks = np.arange(1, MAX_LOOKS + 1)
    st.plotly_chart(charts.peeking_curve(looks, _peeking_rates(alpha), alpha, n_looks))

    st.subheader("Thresholds at each look")
    st.dataframe(
        pd.DataFrame(
            {
                "Look": np.arange(1, n_looks + 1),
                "Users per variant": z_by_look.columns,
                "O'Brien-Fleming: |z| above": np.round(boundaries, 3),
                "O'Brien-Fleming: p below": [
                    f"{p:.2g}" for p in 2 * norm.sf(boundaries)
                ],
                "Uncorrected: p below": f"{alpha:g}",
            }
        ),
        hide_index=True,
    )


def _comparison_table(
    rows: list[tuple[str, float, float]], n_experiments: int
) -> pd.DataFrame:
    """Simulated rates with their Monte Carlo margins, next to expected rates."""
    return pd.DataFrame(
        {
            "": [label for label, _, _ in rows],
            "Simulated": [
                f"{rate:.1%} ± {_monte_carlo_margin(rate, n_experiments):.1%}"
                for _, rate, _ in rows
            ],
            "Expected": [f"{expected:.1%}" for _, _, expected in rows],
        }
    )


def _monte_carlo_margin(rate: float, n_experiments: int) -> float:
    """95% margin of a rate estimated from n_experiments simulated runs."""
    return 1.96 * math.sqrt(rate * (1 - rate) / n_experiments)


def _redraw_button(section: str) -> int:
    """Show a button that redraws the random data, and return the seed to use."""
    seed_key = f"{section}_seed"
    st.session_state.setdefault(seed_key, 0)
    if st.button("Draw new random data", key=f"{section}_redraw"):
        st.session_state[seed_key] += 1
    return st.session_state[seed_key]


@st.cache_data(max_entries=CACHE_SIZE, show_spinner=False)
def _simulate_experiments(
    baseline_rate: float,
    true_lift: float,
    n_per_variant: int,
    n_experiments: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    return simulate_experiments(
        baseline_rate, true_lift, n_per_variant, n_experiments, alpha, seed
    )


@st.cache_data(max_entries=CACHE_SIZE, show_spinner=False)
def _simulate_peeking(
    baseline_rate: float, n_per_variant: int, n_looks: int, alpha: float, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    z_by_look = simulate_peeking(baseline_rate, n_per_variant, n_looks, seed=seed)
    return z_by_look, summarize_peeking(z_by_look, alpha)


@st.cache_data(max_entries=CACHE_SIZE, show_spinner=False)
def _boundaries(n_looks: int, alpha: float) -> np.ndarray:
    return obrien_fleming_boundaries(n_looks, alpha)


@st.cache_data(max_entries=CACHE_SIZE, show_spinner=False)
def _peeking_rates(alpha: float) -> np.ndarray:
    """Exact false positive rate of stopping at the first p < alpha, by looks."""
    return np.array(
        [peeking_false_positive_rate(k, alpha) for k in range(1, MAX_LOOKS + 1)]
    )
