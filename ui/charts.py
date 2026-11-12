"""
Plotly charts for ExperimentLab.

Each function takes results already computed in core/ and returns a figure,
so the charts can be built and checked without running Streamlit.

Color means the same thing in every chart: blue for a correct detection or
the corrected procedure, red for an error (a false positive, a result in the
wrong direction, or an interval that misses the truth), and grey where
nothing happened.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Palette, matching .streamlit/config.toml. The blue and red pass a
# colorblind-safety check against each other on white.
INK = "#1A2230"
SLATE = "#5A6472"
AXIS = "#C4CBD5"
GRID = "#ECEFF3"
SURFACE = "#FFFFFF"
CORRECT = "#2A78D6"
ERROR = "#E34948"
CONTEXT = "#B9C0CA"
CONTEXT_LIGHT = "#D3D8DF"
FONT = '"Libre Franklin", "Helvetica Neue", Arial, sans-serif'

DETECTED = "Detected the lift"
WRONG_DIRECTION = "Significant in the wrong direction"
FALSE_POSITIVE = "False positive"
NOT_SIGNIFICANT = "Not significant"
OUTCOME_COLORS = {
    DETECTED: CORRECT,
    WRONG_DIRECTION: ERROR,
    FALSE_POSITIVE: ERROR,
    NOT_SIGNIFICANT: CONTEXT,
}

# Early O'Brien-Fleming boundaries can be far above any simulated z, so the
# z-path chart is cut off here to keep the paths readable.
Z_AXIS_LIMIT = 5

# Hide Plotly's zoom and export toolbar; hover still works.
PLOTLY_CONFIG = {"displayModeBar": False}

_AXIS_TITLE_FONT = {"size": 13, "color": SLATE}
TEMPLATE = go.layout.Template(
    layout={
        "font": {"family": FONT, "size": 12, "color": SLATE},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "margin": {"l": 8, "r": 16, "t": 40, "b": 8},
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": AXIS,
            "font": {"family": FONT, "size": 13, "color": INK},
        },
        "legend": {
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1,
            "yanchor": "bottom",
            "font": {"size": 13, "color": INK},
            "traceorder": "normal",
            "itemclick": False,
            "itemdoubleclick": False,
        },
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": AXIS,
            "ticks": "outside",
            "tickcolor": AXIS,
            "ticklen": 4,
            "title": {"font": _AXIS_TITLE_FONT, "standoff": 12},
            "automargin": True,
        },
        "yaxis": {
            "showgrid": True,
            "gridcolor": GRID,
            "zeroline": False,
            "showline": False,
            "title": {"font": _AXIS_TITLE_FONT, "standoff": 12},
            "automargin": True,
        },
    }
)


# Streamlit's front end overrides background colors that come only from a
# template, so every figure also sets them directly.
_BASE_LAYOUT = {"template": TEMPLATE, "paper_bgcolor": SURFACE, "plot_bgcolor": SURFACE}


def outcome_labels(results: pd.DataFrame, true_lift: float) -> np.ndarray:
    """Label each simulated experiment by what its test concluded.

    With a real lift, a significant result is a detection if treatment is
    ahead and an error if it is behind. With no lift, every significant
    result is a false positive.

    Args:
        results: Output of core.simulator.simulate_experiments.
        true_lift: True relative lift used in the simulation.
    """
    significant = results["significant"].to_numpy()
    if true_lift == 0:
        return np.where(significant, FALSE_POSITIVE, NOT_SIGNIFICANT)
    treatment_ahead = results["z"].to_numpy() > 0
    return np.select(
        [significant & treatment_ahead, significant],
        [DETECTED, WRONG_DIRECTION],
        default=NOT_SIGNIFICANT,
    )


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
    figure = go.Figure(
        go.Scatter(
            x=n_values,
            y=power_values,
            mode="lines",
            line={"color": CORRECT, "width": 2},
            hovertemplate="%{x:,.0f} users per variant<br>%{y:.1%} power"
            "<extra></extra>",
        )
    )
    guide = {"color": SLATE, "width": 1, "dash": "dot"}
    figure.add_shape(
        type="line", x0=0, x1=n_required, y0=target_power, y1=target_power, line=guide
    )
    figure.add_shape(
        type="line", x0=n_required, x1=n_required, y0=0, y1=target_power, line=guide
    )
    figure.add_trace(_highlight_marker([n_required], [target_power], CORRECT))
    figure.add_annotation(
        x=n_required,
        y=target_power,
        text=f"{n_required:,} users, {target_power:.0%} power",
        xanchor="left",
        yanchor="top",
        xshift=10,
        yshift=-6,
        showarrow=False,
        font={"size": 13, "color": INK},
    )
    figure.update_layout(
        **_BASE_LAYOUT,
        height=360,
        showlegend=False,
        margin={"t": 16},
        xaxis={"title": "Users per variant", "tickformat": "~s", "rangemode": "tozero"},
        yaxis={"title": "Power", "tickformat": ".0%", "range": [0, 1.04]},
    )
    return figure


def observed_lift_histogram(
    results: pd.DataFrame, true_lift: float, n_bins: int = 40
) -> go.Figure:
    """Distribution of measured lifts, colored by what each test concluded.

    The x-axis covers the middle 99% of lifts; the rare experiments beyond it
    are counted in the outermost bars.

    Args:
        results: Output of core.simulator.simulate_experiments.
        true_lift: True relative lift used in the simulation.
        n_bins: Number of bars.
    """
    lift = results["observed_lift"].to_numpy() * 100
    defined = ~np.isnan(lift)
    figure = go.Figure()
    if defined.any():
        low, high = np.percentile(lift[defined], [0.5, 99.5])
        edges = np.histogram_bin_edges(lift[defined], bins=n_bins, range=(low, high))
        figure.add_traces(
            _stacked_bars(
                np.clip(lift[defined], edges[0], edges[-1]),
                outcome_labels(results, true_lift)[defined],
                edges,
                value_format=".1f",
                value_suffix="%",
            )
        )
    _vertical_reference(figure, true_lift * 100, f"True lift {true_lift:.0%}")
    figure.update_layout(
        **_BASE_LAYOUT,
        height=340,
        barmode="stack",
        bargap=0,
        xaxis={"title": "Observed lift", "ticksuffix": "%"},
        yaxis={"title": "Experiments", "tickformat": ","},
    )
    return figure


def p_value_histogram(
    results: pd.DataFrame, true_lift: float, alpha: float
) -> go.Figure:
    """Distribution of p-values, colored by what each test concluded.

    Bars are 0.05 wide, or as wide as alpha when it is smaller, so the
    significant results fill the leftmost bars exactly.

    Args:
        results: Output of core.simulator.simulate_experiments.
        true_lift: True relative lift used in the simulation.
        alpha: Significance level, marked on the chart.
    """
    p_values = results["p_value"].to_numpy()
    defined = ~np.isnan(p_values)
    bin_width = min(alpha, 0.05)
    edges = np.linspace(0, 1, round(1 / bin_width) + 1)
    figure = go.Figure(
        _stacked_bars(
            p_values[defined],
            outcome_labels(results, true_lift)[defined],
            edges,
            value_format=".2f",
            value_suffix="",
        )
    )
    _vertical_reference(figure, alpha, f"α = {alpha:g}", dash="dash")
    figure.update_layout(
        **_BASE_LAYOUT,
        height=340,
        barmode="stack",
        bargap=0,
        xaxis={"title": "p-value", "range": [0, 1]},
        yaxis={"title": "Experiments", "tickformat": ","},
    )
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
    low = shown["ci_low"].to_numpy() * 100
    high = shown["ci_high"].to_numpy() * 100
    center = (low + high) / 2
    row = np.arange(1, len(shown) + 1)
    covers = shown["ci_covers_truth"].to_numpy()

    figure = go.Figure()
    if true_difference != 0:
        _vertical_reference(
            figure,
            0,
            "No difference",
            color=AXIS,
            label_color=SLATE,
            label_at_bottom=True,
        )
    for mask, name, color in [
        (covers, "Contains the true difference", CONTEXT),
        (~covers, "Misses the true difference", ERROR),
    ]:
        x, y = _join_segments(low[mask], high[mask], row[mask])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": color, "width": 2},
                name=name,
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=center[mask],
                y=row[mask],
                mode="markers",
                marker={"color": color, "size": 7},
                legendgroup=name,
                showlegend=False,
                customdata=np.column_stack([low[mask], high[mask]]),
                hovertemplate="Experiment %{y}<br>%{customdata[0]:.2f} to "
                "%{customdata[1]:.2f} points<extra></extra>",
            )
        )
    if true_difference == 0:
        truth_label = "True difference 0"
    else:
        truth_label = f"True difference {true_difference * 100:+.2f} points"
    _vertical_reference(figure, true_difference * 100, truth_label)
    # Line ends get no automatic padding, so leave room past the widest interval.
    span = [min(low.min(), 0), max(high.max(), 0)]
    padding = 0.04 * (span[1] - span[0])
    figure.update_layout(
        **_BASE_LAYOUT,
        height=520,
        xaxis={
            "title": "Treatment minus control (percentage points)",
            "range": [span[0] - padding, span[1] + padding],
        },
        yaxis={
            "autorange": "reversed",
            "showgrid": False,
            "showticklabels": False,
            "title": None,
        },
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
    shown = z_by_look.head(n_shown).to_numpy()
    crossed = (np.abs(shown) > threshold).any(axis=1)

    figure = go.Figure()
    for mask, name, color, width in [
        (~crossed, "Never crossed", CONTEXT_LIGHT, 1),
        (crossed, f"Crossed ±{threshold:.2f} at some look", ERROR, 1.5),
    ]:
        x, y = _join_paths(users, shown[mask])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": color, "width": width},
                name=name,
                hovertemplate="z = %{y:.2f} at %{x:,} users<extra></extra>",
            )
        )
    for sign in (1, -1):
        figure.add_trace(
            go.Scatter(
                x=users[[0, -1]],
                y=[sign * threshold] * 2,
                mode="lines",
                line={"color": INK, "width": 1, "dash": "dash"},
                name=f"Usual threshold ±{threshold:.2f}",
                legendgroup="threshold",
                showlegend=sign == 1,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=users,
                y=sign * boundaries,
                mode="lines+markers",
                line={"color": CORRECT, "width": 2},
                marker={"size": 7, "line": {"color": SURFACE, "width": 1.5}},
                name="O'Brien–Fleming boundary",
                legendgroup="boundary",
                showlegend=sign == 1,
                hovertemplate="Boundary %{y:.2f} at %{x:,} users<extra></extra>",
            )
        )
    figure.update_layout(
        **_BASE_LAYOUT,
        height=460,
        xaxis={"title": "Users per variant", "tickformat": "~s"},
        yaxis={
            "title": "z-statistic",
            "range": [-Z_AXIS_LIMIT, Z_AXIS_LIMIT],
            "tickvals": [-4, -2, 0, 2, 4],
        },
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
        current_looks: Number of looks selected, highlighted on the chart.
    """
    corrected_rates = np.full(len(looks), alpha)
    figure = go.Figure()
    for rates, name, color in [
        (naive_rates, f"Stop at the first p < {alpha:g}", ERROR),
        (corrected_rates, "O'Brien–Fleming boundaries", CORRECT),
    ]:
        figure.add_trace(
            go.Scatter(
                x=looks,
                y=rates,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 6, "line": {"color": SURFACE, "width": 1.5}},
                name=name,
                hovertemplate="%{x} looks: %{y:.1%}<extra></extra>",
            )
        )
        current_rate = rates[current_looks - 1]
        figure.add_trace(_highlight_marker([current_looks], [current_rate], color))
        figure.add_annotation(
            x=current_looks,
            y=current_rate,
            text=f"{current_rate:.1%}",
            xanchor="right",
            yanchor="bottom",
            xshift=-6,
            yshift=6,
            showarrow=False,
            font={"size": 13, "color": INK},
        )
    figure.update_layout(
        **_BASE_LAYOUT,
        height=380,
        xaxis={
            "title": "Looks at the data",
            "tickvals": [1, *range(5, int(looks[-1]) + 1, 5)],
            "range": [0.5, looks[-1] + 0.5],
        },
        yaxis={
            "title": "False positive rate",
            "tickformat": ".0%",
            "range": [0, max(naive_rates) * 1.15],
        },
    )
    return figure


def _stacked_bars(
    values: np.ndarray,
    outcomes: np.ndarray,
    edges: np.ndarray,
    value_format: str,
    value_suffix: str,
) -> list[go.Bar]:
    """Histogram bars for each outcome, to be stacked.

    The counts are computed here, so the browser receives one number per bar
    rather than every simulated experiment.
    """
    centers = (edges[:-1] + edges[1:]) / 2
    bin_range = np.column_stack([edges[:-1], edges[1:]])
    bars = []
    for outcome, color in OUTCOME_COLORS.items():
        in_group = outcomes == outcome
        if not in_group.any():
            continue
        counts, _ = np.histogram(values[in_group], bins=edges)
        bars.append(
            go.Bar(
                x=centers,
                y=counts,
                width=np.diff(edges),
                name=outcome,
                marker={"color": color, "line": {"color": SURFACE, "width": 1}},
                customdata=bin_range,
                hovertemplate=f"%{{customdata[0]:{value_format}}}{value_suffix} to "
                f"%{{customdata[1]:{value_format}}}{value_suffix}<br>"
                "%{y:,} experiments<extra>%{fullData.name}</extra>",
            )
        )
    return bars


def _vertical_reference(
    figure: go.Figure,
    x: float,
    label: str,
    dash: str = "solid",
    color: str = INK,
    label_color: str = INK,
    label_at_bottom: bool = False,
) -> None:
    """Draw a labelled vertical line across the plot.

    The label sits at the top of the line, or at the bottom so that two
    nearby lines can both be labelled without overlapping.
    """
    figure.add_vline(x=x, line={"color": color, "width": 1.5, "dash": dash})
    figure.add_annotation(
        x=x,
        y=0 if label_at_bottom else 1,
        yref="paper",
        text=label,
        xanchor="left",
        yanchor="bottom" if label_at_bottom else "top",
        xshift=5,
        showarrow=False,
        bgcolor=SURFACE,
        font={"size": 12, "color": label_color},
    )


def _highlight_marker(x: list[float], y: list[float], color: str) -> go.Scatter:
    """A large point with a white ring, for the value a chart is about."""
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={"color": color, "size": 12, "line": {"color": SURFACE, "width": 2}},
        showlegend=False,
        hoverinfo="skip",
    )


def _join_paths(x: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten many paths into one line trace, with gaps between paths.

    One trace with gaps draws much faster than one trace per path.
    """
    n_paths = len(paths)
    gap = np.full((n_paths, 1), np.nan)
    joined_x = np.hstack([np.tile(x, (n_paths, 1)), gap]).ravel()
    joined_y = np.hstack([paths, gap]).ravel()
    return joined_x, joined_y


def _join_segments(
    starts: np.ndarray, ends: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten horizontal segments into one line trace, with gaps between them."""
    gap = np.full(len(rows), np.nan)
    x = np.column_stack([starts, ends, gap]).ravel()
    y = np.column_stack([rows, rows, gap]).ravel()
    return x, y
