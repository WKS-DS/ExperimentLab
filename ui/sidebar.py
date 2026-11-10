"""Sidebar controls for the experiment design shared by every section."""

import streamlit as st

ALPHA_OPTIONS = [0.01, 0.05, 0.10]
POWER_OPTIONS = [0.70, 0.80, 0.90, 0.95]


def experiment_inputs() -> tuple[float, float, float, float]:
    """Render the experiment design controls in the sidebar.

    Stops the app with a message when the lift would push the treatment rate
    to 100% or more, since no sample size exists for that design.

    Returns:
        (baseline_rate, minimum_detectable_effect, alpha, power), all as
        fractions, e.g. 0.05 for a 5% baseline.
    """
    st.sidebar.header("Experiment design")
    baseline_rate = (
        st.sidebar.slider(
            "Baseline conversion rate (%)",
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="baseline",
            help="Share of control users who convert today.",
        )
        / 100
    )
    minimum_detectable_effect = (
        st.sidebar.slider(
            "Minimum detectable effect (%)",
            min_value=1,
            max_value=100,
            value=10,
            key="mde",
            help="Smallest relative lift worth detecting. A 10% lift on a 5% "
            "baseline is 5.5%.",
        )
        / 100
    )
    alpha = st.sidebar.selectbox(
        "Significance level (α)",
        ALPHA_OPTIONS,
        index=1,
        key="alpha",
        help="False positive rate when there is no real difference.",
    )
    power = st.sidebar.selectbox(
        "Power",
        POWER_OPTIONS,
        index=1,
        format_func="{:.0%}".format,
        key="power",
        help="Chance of detecting the lift if it is real.",
    )

    treatment_rate = baseline_rate * (1 + minimum_detectable_effect)
    if treatment_rate >= 1:
        st.error(
            f"A {minimum_detectable_effect:.0%} lift on a {baseline_rate:.1%} "
            f"baseline gives a treatment rate of {treatment_rate:.0%}. Lower the "
            "baseline or the minimum detectable effect."
        )
        st.stop()
    return baseline_rate, minimum_detectable_effect, alpha, power
