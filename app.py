"""
ExperimentLab: design, simulate and analyze A/B tests.

Run with: streamlit run app.py
"""

import streamlit as st

from ui.results import show_peeking, show_sample_size, show_simulator
from ui.sidebar import experiment_inputs

st.set_page_config(page_title="ExperimentLab", layout="wide")
st.title("ExperimentLab")
st.caption("Design, simulate, and analyze A/B tests before you run them.")

baseline_rate, minimum_detectable_effect, alpha, power = experiment_inputs()

sample_size_tab, simulator_tab, peeking_tab = st.tabs(
    ["Sample size", "Simulator", "Peeking"]
)
with sample_size_tab:
    n_required = show_sample_size(
        baseline_rate, minimum_detectable_effect, alpha, power
    )
with simulator_tab:
    show_simulator(baseline_rate, minimum_detectable_effect, alpha, n_required)
with peeking_tab:
    show_peeking(baseline_rate, alpha, n_required)
