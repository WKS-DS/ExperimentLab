"""
Plotly charts for ExperimentLab.

Each function takes results already computed in core/ and returns a figure,
so the charts can be built and checked without running Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SIGNIFICANT = "Significant"
NOT_SIGNIFICANT = "Not significant"
RESULT_ORDER = {"Result": [SIGNIFICANT, NOT_SIGNIFICANT]}

# Paths that never cross a threshold are drawn in grey so the ones that do
# stand out.
QUIET_COLOR = "lightgrey"

# Both halves of the O'Brien-Fleming boundary share one color.
BOUNDARY_COLOR = "crimson"

# Early O'Brien-Fleming boundaries can be far above any simulated z, so the
# z-path chart is cut off here to keep the paths readable.
Z_AXIS_LIMIT = 5


def power_curve(
    n_values: np.ndarray,
    power_values: np.ndarray,
    n_required: int,
    target_power: float,
) -> go.Figure:
    """Power against users per variant, marking the required sample size.

    Args:
        n_values: Users per variant along the x-axis.
        power_values: Power at each value of n_values.
        n_required: Sample size that reaches the target power.
        target_power: Power the sample size was calculated for.
    """
    figure = go.Figure(go.Scatter(x=n_values, y=power_values, mode="lines"))
    figure.add_hline(y=target_power, line_dash="dot")
    figure.add_vline(
        x=n_required, line_dash="dot", annotation_text=f"{n_required:,} per variant"
    )
    figure.update_layout(
        title="Power by sample size",
        xaxis_title="Users per variant",
        yaxis_title="Power",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
    )
    return figure


def observed_lift_histogram(results: pd.DataFrame, true_lift: float) -> go.Figure:
    """Distribution of measured lifts, split by whether the test was significant.

    Args:
        results: Output of core.simulator.simulate_experiments.
        true_lift: True relative lift used in the simulation.
    """
    data = pd.DataFrame(
        {
            "Observed lift (%)": results["observed_lift"] * 100,
            "Result": _result_labels(results["significant"]),
        }
    )
    figure = px.histogram(
        data,
        x="Observed lift (%)",
        color="Result",
        nbins=60,
        category_orders=RESULT_ORDER,
        title="Observed lift in each experiment",
    )
    figure.add_vline(x=true_lift * 100, line_dash="dash", annotation_text="True lift")
    figure.update_layout(yaxis_title="Experiments")
    return figure


def p_value_histogram(results: pd.DataFrame, alpha: float) -> go.Figure:
    """Distribution of p-values, in bins of 0.05.

    Args:
        results: Output of core.simulator.simulate_experiments.
        alpha: Significance level, marked on the chart.
    """
    data = pd.DataFrame(
        {
            "p-value": results["p_value"],
            "Result": _result_labels(results["significant"]),
        }
    )
    figure = px.histogram(
        data,
        x="p-value",
        color="Result",
        category_orders=RESULT_ORDER,
        title="p-values",
    )
    figure.update_traces(xbins={"start": 0, "end": 1, "size": 0.05})
    figure.add_vline(x=alpha, line_dash="dash", annotation_text=f"α = {alpha:g}")
    figure.update_layout(yaxis_title="Experiments")
    return figure


def interval_chart(
    results: pd.DataFrame, true_difference: float, n_shown: int = 50
) -> go.Figure:
    """Confidence intervals of the first experiments against the true difference.

    Args:
        results: Output of core.simulator.simulate_experiments.
        true_difference: True treatment rate minus control rate.
        n_shown: Number of experiments to draw.
    """
    shown = results.head(n_shown)
    center = (shown["ci_low"] + shown["ci_high"]) / 2 * 100
    half_width = (shown["ci_high"] - shown["ci_low"]) / 2 * 100
    experiment = pd.Series(np.arange(1, len(shown) + 1), index=shown.index)

    figure = go.Figure()
    for covers, name in [
        (True, "Contains the true difference"),
        (False, "Misses the true difference"),
    ]:
        mask = shown["ci_covers_truth"] == covers
        figure.add_trace(
            go.Scatter(
                x=center[mask],
                y=experiment[mask],
                mode="markers",
                error_x={"type": "data", "array": half_width[mask]},
                name=name,
            )
        )
    figure.add_vline(
        x=true_difference * 100, line_dash="dash", annotation_text="True difference"
    )
    figure.update_layout(
        title=f"Confidence intervals for the first {len(shown)} experiments",
        xaxis_title="Treatment minus control (percentage points)",
        yaxis_title="Experiment",
        yaxis_autorange="reversed",
    )
    return figure


def z_paths_chart(
    z_by_look: pd.DataFrame,
    threshold: float,
    boundaries: np.ndarray,
    n_shown: int = 50,
) -> go.Figure:
    """z-statistic of A/A tests at each look, against both stopping rules.

    Args:
        z_by_look: Output of core.sequential.simulate_peeking.
        threshold: Usual two-sided threshold, e.g. 1.96 for alpha = 0.05.
        boundaries: O'Brien-Fleming threshold at each look.
        n_shown: Number of experiments to draw.
    """
    users = z_by_look.columns.to_numpy()
    shown = z_by_look.head(n_shown)
    crossed = (shown.abs() > threshold).any(axis=1).to_numpy()

    figure = go.Figure()
    for mask, name, color in [
        (crossed, "Significant at some look", None),
        (~crossed, "Never significant", QUIET_COLOR),
    ]:
        x, y = _join_paths(users, shown.to_numpy()[mask])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=name,
                line={"width": 1, "color": color},
                marker={"size": 4},
            )
        )
    for sign in (1, -1):
        figure.add_hline(y=sign * threshold, line_dash="dot")
        figure.add_trace(
            go.Scatter(
                x=users,
                y=sign * boundaries,
                mode="lines+markers",
                name="O'Brien-Fleming boundary",
                line={"dash": "dash", "color": BOUNDARY_COLOR},
                showlegend=sign == 1,
            )
        )
    figure.update_layout(
        title=f"z-statistic of the first {len(shown)} A/A tests at each look",
        xaxis_title="Users per variant",
        yaxis_title="z",
        yaxis_range=[-Z_AXIS_LIMIT, Z_AXIS_LIMIT],
    )
    return figure


def peeking_curve(
    looks: np.ndarray, naive_rates: np.ndarray, alpha: float, current_looks: int
) -> go.Figure:
    """False positive rate against the number of looks, with and without correction.

    Args:
        looks: Numbers of looks along the x-axis.
        naive_rates: False positive rate when stopping at the first p < alpha.
        alpha: Significance level, which the corrected rate equals.
        current_looks: Number of looks selected, marked on the chart.
    """
    figure = go.Figure(
        [
            go.Scatter(
                x=looks, y=naive_rates, mode="lines+markers", name="Stop at first p < α"
            ),
            go.Scatter(
                x=looks,
                y=np.full(len(looks), alpha),
                mode="lines",
                name="O'Brien-Fleming boundaries",
            ),
        ]
    )
    figure.add_vline(x=current_looks, line_dash="dot")
    figure.update_layout(
        title="False positive rate by number of looks",
        xaxis_title="Looks at the data",
        yaxis_title="False positive rate",
        yaxis_tickformat=".0%",
        yaxis_rangemode="tozero",
    )
    return figure


def _result_labels(significant: pd.Series) -> np.ndarray:
    """Label each experiment as significant or not."""
    return np.where(significant, SIGNIFICANT, NOT_SIGNIFICANT)


def _join_paths(x: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten many paths into one line trace, with gaps between paths.

    One trace with gaps draws much faster than one trace per path.
    """
    n_paths = len(paths)
    gap = np.full((n_paths, 1), np.nan)
    joined_x = np.hstack([np.tile(x, (n_paths, 1)), gap]).ravel()
    joined_y = np.hstack([paths, gap]).ravel()
    return joined_x, joined_y
