"""
ExperimentLab - A/B Testing Simulator

Run with: streamlit run app.py
"""

import streamlit as st

# Page config
st.set_page_config(page_title="ExperimentLab", page_icon="🧪", layout="wide")

st.title("🧪 ExperimentLab")
st.markdown("*Design, simulate, and analyze A/B tests before you run them.*")

# Sidebar inputs
st.sidebar.header("Experiment Parameters")

baseline_rate = (
    st.sidebar.slider(
        "Baseline Conversion Rate (%)",
        min_value=1,
        max_value=50,
        value=5,
        help="Current conversion rate before the experiment",
    )
    / 100
)

minimum_detectable_effect = (
    st.sidebar.slider(
        "Minimum Detectable Effect (%)",
        min_value=1,
        max_value=100,
        value=10,
        help="Smallest relative improvement you want to detect",
    )
    / 100
)

alpha = st.sidebar.selectbox(
    "Significance Level (α)",
    options=[0.01, 0.05, 0.10],
    index=1,
    help="Probability of false positive (Type I error)",
)

power = st.sidebar.selectbox(
    "Statistical Power (1-β)",
    options=[0.70, 0.80, 0.90, 0.95],
    index=1,
    help="Probability of detecting a true effect",
)

# Main content
st.header("📊 Sample Size Calculator")

# TODO: Import and use calculate_sample_size from core.power_analysis
st.info("Implement `core/power_analysis.py` to see sample size calculation here.")

st.markdown("---")

st.header("🎲 Experiment Simulator")
st.info("Implement `core/simulator.py` to run Monte Carlo simulations.")

st.markdown("---")

st.header("⚠️ Peeking Problem Demo")
st.info("Implement `core/sequential.py` to visualize early stopping dangers.")
