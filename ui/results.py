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
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from core.power_analysis import calculate_power, calculate_sample_size, cohens_h
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

# Below this many expected control conversions, the p-values of the z-test
# fall on a coarse grid and some bars of the A/A p-value histogram are 20% or
# more off flat, beyond what simulation noise explains.
MIN_CONVERSIONS_FOR_SMOOTH_P_VALUES = 400

# Experiments drawn individually in the interval and z-path charts.
N_SHOWN = 50

CACHE_SIZE = 32

# Paragraphs are capped at this width in pixels, about 85 characters.
TEXT_WIDTH = 640


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

    _section_intro(
        "How many users does this test need?",
        f"To detect a change in conversion from {baseline_rate:.2%} to "
        f"{treatment_rate:.2%}, a {minimum_detectable_effect:.0%} relative lift, "
        f"with a two-sided test at α = {alpha:g} and {power:.0%} power:",
    )
    with st.container(horizontal=True, gap="xlarge"):
        st.metric("Users per variant", f"{n_required:,}", width="content")
        st.metric("Users in total", f"{2 * n_required:,}", width="content")

    caption = (
        f"Chance of detecting a real {minimum_detectable_effect:.0%} lift at each "
        "sample size."
    )
    if n_required >= 2:
        half = n_required // 2
        half_power = calculate_power(
            baseline_rate, minimum_detectable_effect, half, alpha
        )
        caption += (
            f" With half as many users ({half:,} per variant), power falls to "
            f"{half_power:.0%}."
        )
    n_values = np.unique(np.linspace(max(1, n_required / 20), 3 * n_required, 100))
    power_values = [
        calculate_power(baseline_rate, minimum_detectable_effect, n, alpha)
        for n in n_values
    ]
    _chart(
        f"Power reaches {power:.0%} at {n_required:,} users per variant",
        caption,
        charts.power_curve(n_values, power_values, n_required, power),
    )

    with st.expander("How the sample size is calculated"):
        h = cohens_h(baseline_rate, minimum_detectable_effect)
        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = norm.ppf(power)
        st.markdown(
            "The two rates are first converted to Cohen's h, an effect size for "
            "proportions that puts a change at any baseline on the same scale:",
            width=TEXT_WIDTH,
        )
        st.latex(
            rf"h = 2\arcsin\sqrt{{{treatment_rate:.4f}}} - "
            rf"2\arcsin\sqrt{{{baseline_rate:.4f}}} = {h:.4f}"
        )
        n_exact = 2 * ((z_alpha + z_power) / h) ** 2
        st.markdown(
            "The users needed per variant for a two-sided z-test are then",
            width=TEXT_WIDTH,
        )
        # Braces stop LaTeX from adding a space after the thousands separator.
        n_text = f"{n_exact:,.1f}".replace(",", "{,}")
        st.latex(
            rf"n = 2\left(\frac{{z_{{1-\alpha/2}} + z_{{1-\beta}}}}{{h}}\right)^2"
            rf"= 2\left(\frac{{{z_alpha:.3f} + {z_power:.3f}}}{{{h:.4f}}}\right)^2"
            rf"= {n_text}"
        )
        st.markdown(
            f"which rounds up to {n_required:,} users per variant.", width=TEXT_WIDTH
        )
    return n_required


def show_simulator(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float,
    n_required: int,
) -> None:
    """Simulate many experiments and compare the results with the formulas."""
    _section_intro(
        "What results should I expect?",
        "The simulator runs the experiment thousands of times on simulated users "
        "with a true lift you choose, and analyzes each run with a two-proportion "
        "z-test. If the formulas are right, the share of runs that detect the "
        "lift matches the planned power.",
    )
    users_column, lift_column, count_column = st.columns(3, gap="medium")
    n_per_variant = int(
        users_column.number_input(
            "Users per variant",
            min_value=1,
            value=n_required,
            step=1_000,
            help="Starts at the calculated sample size.",
        )
    )
    true_lift = (
        lift_column.slider(
            "True lift",
            min_value=0,
            max_value=100,
            value=round(minimum_detectable_effect * 100),
            format="%d%%",
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
            baseline_rate, true_lift, n_per_variant, n_experiments, alpha, seed
        )
    except ValueError as error:
        st.error(str(error))
        return
    summary = summarize_simulation(results)

    if true_lift > 0:
        expected_power = calculate_power(baseline_rate, true_lift, n_per_variant, alpha)
        rows = [
            ("Detected the lift (power)", summary["treatment_win_rate"], expected_power)
        ]
    else:
        rows = [("False positives", summary["significant_rate"], alpha)]
    rows.append(
        ("Intervals containing the true difference", summary["ci_coverage"], 1 - alpha)
    )
    _comparison_table(rows, n_experiments)

    undefined_lifts = int(results["observed_lift"].isna().sum())
    if undefined_lifts:
        st.caption(
            f"{undefined_lifts:,} experiments had no conversions in control, so "
            "their relative lift is undefined and they are left out of the lift "
            "histogram.",
            width=TEXT_WIDTH,
        )

    lift_column, p_value_column = st.columns(
        2, gap="large", vertical_alignment="bottom"
    )
    with lift_column:
        _chart(
            *_lift_chart_text(summary, true_lift, n_experiments),
            charts.observed_lift_histogram(results, true_lift),
        )
    with p_value_column:
        _chart(
            *_p_value_chart_text(
                summary, true_lift, alpha, baseline_rate * n_per_variant
            ),
            charts.p_value_histogram(results, true_lift, alpha),
        )

    shown_covered = int(results["ci_covers_truth"].head(N_SHOWN).sum())
    n_shown = min(N_SHOWN, n_experiments)
    _chart(
        f"{shown_covered} of the first {n_shown} intervals contain the true "
        "difference",
        f"{1 - alpha:.0%} confidence intervals for treatment minus control. Across "
        f"all {n_experiments:,} experiments, {summary['ci_coverage']:.1%} contain "
        f"it, against {1 - alpha:.0%} planned.",
        charts.interval_chart(results, baseline_rate * true_lift, N_SHOWN),
    )


def show_peeking(baseline_rate: float, alpha: float, n_required: int) -> None:
    """Simulate A/A tests checked repeatedly, with and without a correction."""
    _section_intro(
        "What happens if I check early?",
        "Every experiment here is an A/A test: both groups get the same "
        "experience, so any significant result is a false positive. Each test is "
        "checked at evenly spaced points as users arrive, and stopped the first "
        f"time p < {alpha:g}.",
    )
    looks_column, users_column = st.columns(2, gap="medium")
    n_looks = looks_column.slider(
        "Looks at the data", min_value=1, max_value=MAX_LOOKS, value=10, key="looks"
    )
    n_per_variant = int(
        users_column.number_input(
            "Users per variant at the final look",
            min_value=MAX_LOOKS,
            value=max(n_required, MAX_LOOKS),
            step=1_000,
            help="Starts at the calculated sample size.",
        )
    )
    seed = _redraw_button("peeking")

    z_by_look, summary = _simulate_peeking(
        baseline_rate, n_per_variant, n_looks, alpha, seed
    )
    n_experiments = len(z_by_look)
    peeking_rates = _peeking_rates(alpha)
    rows = [
        ("Analyze once, at the final look", summary["final_look_only"], alpha),
        (
            f"Check every look, stop at the first p < {alpha:g}",
            summary["any_look"],
            peeking_rates[n_looks - 1],
        ),
        (
            "Check every look with O'Brien–Fleming boundaries",
            summary["obrien_fleming"],
            alpha,
        ),
    ]
    _comparison_table(rows, n_experiments)

    first_look_conversions = baseline_rate * z_by_look.columns[0]
    if first_look_conversions < MIN_CONVERSIONS_PER_LOOK:
        st.caption(
            f"Each variant expects only {first_look_conversions:.1f} conversions "
            "by the first look. With so few, the z-test is only approximate and the "
            "simulated rates can sit a little below the expected ones.",
            width=TEXT_WIDTH,
        )

    threshold = norm.ppf(1 - alpha / 2)
    boundaries = _boundaries(n_looks, alpha)
    crossed = round(summary["any_look"] * n_experiments)
    if n_looks == 1:
        boundary_text = "With a single look, both rules are the same test."
    else:
        boundary_text = (
            f"The O'Brien–Fleming boundary starts at {boundaries[0]:.2f} and only "
            f"comes down to {boundaries[-1]:.2f} at the final look, so early noise "
            "rarely crosses it."
        )
    shown = z_by_look.head(N_SHOWN).to_numpy()
    shown_crossed = int((np.abs(shown) > threshold).any(axis=1).sum())
    _chart(
        f"{crossed:,} of {n_experiments:,} identical-group tests crossed "
        f"p < {alpha:g} at some look",
        f"z-statistic of the first {len(shown)} tests at each look. The "
        f"{shown_crossed} in red cross the dashed line at least once. "
        f"{boundary_text}",
        charts.z_paths_chart(z_by_look, threshold, boundaries, N_SHOWN),
    )

    curve_column, table_column = st.columns([3, 2], gap="large")
    with curve_column:
        rate = peeking_rates[n_looks - 1]
        if n_looks == 1:
            title = f"One look keeps the false positive rate at {alpha:.0%}"
        else:
            title = (
                f"{n_looks} looks raise the false positive rate from {alpha:.0%} "
                f"to {rate:.1%}"
            )
        _chart(
            title,
            "Exact rates by number of looks, computed by numerical integration. "
            "With O'Brien–Fleming boundaries the rate stays at α however often you "
            "look.",
            charts.peeking_curve(
                np.arange(1, MAX_LOOKS + 1), peeking_rates, alpha, n_looks
            ),
        )
    with table_column:
        _chart_heading(
            "Thresholds at each look",
            "A test can stop at a look only if its p-value is below the "
            "O'Brien–Fleming threshold for that look.",
        )
        st.table(
            pd.DataFrame(
                {
                    "Look": np.arange(1, n_looks + 1),
                    "Users per variant": [f"{n:,}" for n in z_by_look.columns],
                    "|z| above": [f"{b:.2f}" for b in boundaries],
                    "p below": [_format_p(p) for p in 2 * norm.sf(boundaries)],
                }
            ),
            border="horizontal",
            hide_index=True,
            alt="O'Brien–Fleming threshold for each look",
        )

    with st.expander("How the boundaries are calculated"):
        st.markdown(
            "Look k of K counts as significant only if the z-statistic is beyond",
            width=TEXT_WIDTH,
        )
        st.latex(r"|z_k| > c\,\sqrt{K / k}")
        st.markdown(
            "so the bar is high when little data has come in and close to the "
            "usual threshold at the end. The constant c is found numerically so "
            "that the chance of crossing at any look is exactly α when there is no "
            "real difference. The same integration gives the uncorrected rates, "
            "which match the published values of Armitage, McPherson and Rowe "
            "(1969).",
            width=TEXT_WIDTH,
        )


def _section_intro(question: str, text: str) -> None:
    """A section heading phrased as the question it answers, and a lede."""
    st.header(question, anchor=False)
    st.markdown(text, width=TEXT_WIDTH)


def _chart_heading(title: str, caption: str) -> None:
    """A chart title that states the finding, and a caption that explains it."""
    st.subheader(title, anchor=False)
    st.caption(caption, width=TEXT_WIDTH)


def _chart(title: str, caption: str, figure: go.Figure) -> None:
    """A chart under its heading, with the title as its accessible name.

    The figure keeps the app's own Plotly template and has no toolbar.
    """
    _chart_heading(title, caption)
    st.plotly_chart(figure, theme=None, config=charts.PLOTLY_CONFIG, alt=title)


def _lift_chart_text(
    summary: dict[str, float], true_lift: float, n_experiments: int
) -> tuple[str, str]:
    """Title and caption for the observed lift histogram."""
    if true_lift == 0:
        false_positives = round(summary["significant_rate"] * n_experiments)
        return (
            f"{false_positives:,} of {n_experiments:,} A/A experiments were "
            "significant by chance",
            "Observed lift in each simulated experiment. With no real difference, "
            "every significant result is a false positive.",
        )
    detected = round(summary["treatment_win_rate"] * n_experiments)
    caption = "Observed lift in each simulated experiment."
    detected_lift = summary["mean_detected_lift"]
    if not math.isnan(detected_lift):
        caption += (
            f" The experiments that detected it measured {detected_lift:.1%} on "
            f"average, against a true {true_lift:.0%}."
        )
        if detected_lift > 1.25 * true_lift:
            caption += " With low power, only lucky overestimates reach significance."
    return (
        f"{detected:,} of {n_experiments:,} experiments detected the "
        f"{true_lift:.0%} lift",
        caption,
    )


def _p_value_chart_text(
    summary: dict[str, float],
    true_lift: float,
    alpha: float,
    control_conversions: float,
) -> tuple[str, str]:
    """Title and caption for the p-value histogram."""
    if true_lift == 0:
        title = "With no real lift, p-values are spread evenly"
        caption = (
            f"So about {alpha:.0%} of them land below α = {alpha:g} by chance, "
            "whatever the sample size."
        )
    else:
        title = "With a real lift, p-values pile up near zero"
        caption = (
            f"{summary['significant_rate']:.1%} of them are below α = {alpha:g}. "
            "The larger the sample, the further they bunch toward zero."
        )
    if control_conversions < MIN_CONVERSIONS_FOR_SMOOTH_P_VALUES:
        caption += (
            f" With only about {control_conversions:,.0f} conversions per variant, "
            "the test can produce only certain p-values, so some bars are taller "
            "than others."
        )
    return title, caption


def _comparison_table(rows: list[tuple[str, float, float]], n_experiments: int) -> None:
    """Simulated rates with their Monte Carlo margins, next to expected rates."""
    st.table(
        pd.DataFrame(
            {
                "": [label for label, _, _ in rows],
                # Non-breaking spaces keep each value on one line.
                "Simulated": [
                    f"{rate:.1%}\u00a0±\u00a0"
                    f"{_monte_carlo_margin(rate, n_experiments):.1%}"
                    for _, rate, _ in rows
                ],
                "Expected": [f"{expected:.1%}" for _, _, expected in rows],
            }
        ),
        border="horizontal",
        hide_index=True,
        width="content",
        alt="Simulated rates next to the rates the formulas predict",
    )


def _monte_carlo_margin(rate: float, n_experiments: int) -> float:
    """95% margin of a rate estimated from n_experiments simulated runs."""
    return 1.96 * math.sqrt(rate * (1 - rate) / n_experiments)


def _format_p(p: float) -> str:
    """A p-value threshold to three decimals, or "< 0.001" below that."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _redraw_button(section: str) -> int:
    """Show a button that redraws the random data, and return the seed to use."""
    seed_key = f"{section}_seed"
    st.session_state.setdefault(seed_key, 0)
    if st.button(
        "Draw new random data", key=f"{section}_redraw", icon=":material/refresh:"
    ):
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
