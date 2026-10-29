# ExperimentLab

Design, simulate, and analyze A/B tests before you run them.

ExperimentLab is an interactive A/B testing tool built in Python. It covers
the three questions that decide whether an experiment is worth running and
whether its result can be trusted:

1. **How many users do I need?** Sample size and power for a
   conversion-rate test.
2. **What results should I expect?** Monte Carlo simulation of thousands of
   experiments with a known true effect.
3. **What happens if I check early?** A demonstration of how peeking at
   results inflates the false positive rate, and how sequential testing
   keeps it under control.

It works with conversion-rate metrics (whether each user converted) and
compares two variants.

**Status:** In development. See the [roadmap](#roadmap).

---

## Sample size calculator

Inputs are the baseline conversion rate, the minimum detectable effect (as a
relative lift), the significance level α, and the power 1 − β. The output is
the number of users needed in each variant.

The two conversion rates are converted to Cohen's h, an effect size for
proportions that stabilizes the variance, so the same formula works at any
baseline rate:

```math
h = 2\arcsin\sqrt{p_2} - 2\arcsin\sqrt{p_1}
```

The per-variant sample size for a two-sided, two-sample z-test is then

```math
n = 2\left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{h}\right)^2
```

rounded up to the next whole user.

**Example.** A 5% baseline and a 10% relative lift (5.0% → 5.5%), at
α = 0.05 and 80% power, needs **31,218 users per variant**. Evan Miller's
[sample size calculator](https://www.evanmiller.org/ab-testing/sample-size.html),
which uses a slightly different variance approximation, gives 31,234. The
test suite checks results against that reference.

---

## Experiment simulator

The simulator draws synthetic control and treatment data for a chosen true
effect, runs a two-proportion test on each simulated experiment, and
summarizes the results:

- empirical power (the share of experiments that reach significance)
- the distribution of observed lifts and p-values
- confidence intervals for the difference in conversion rates

Running the simulation at the sample size from the calculator should
reproduce the planned power, which makes the simulator a check on the
calculator as well as a teaching tool.

---

## The peeking problem

A p-value is only valid if you look at it once, at the planned sample size.
Checking a running experiment repeatedly and stopping the first time
p < 0.05 gives chance many opportunities to cross the line.

In an A/A test both groups are identical, so every significant result is a
false positive. Checking one at equally spaced points and stopping at the
first p < 0.05:

| Looks at the data | False positive rate |
|------------------:|--------------------:|
| 1                 | 5%                  |
| 5                 | 14%                 |
| 10                | 19%                 |
| 20                | 25%                 |

ExperimentLab simulates this, then applies a sequential testing correction
that raises the bar for significance at each interim look, and shows the
false positive rate returning to 5%.

---

## Getting started

Requires Python 3.10 or later.

```bash
git clone https://github.com/WKS-DS/ExperimentLab.git
cd ExperimentLab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pytest tests/ -v
```

---

## Repository layout

```
ExperimentLab/
├── app.py                  # Streamlit entry point
├── core/                   # Statistics only, no UI code
│   ├── power_analysis.py   # Sample size and power
│   ├── statistics.py       # Two-proportion z-test and confidence interval
│   ├── simulator.py        # Monte Carlo experiment simulation
│   └── sequential.py       # Repeated looks and sequential boundaries
├── ui/                     # Streamlit components
│   ├── sidebar.py          # Experiment inputs
│   ├── results.py          # Result tables and summaries
│   └── charts.py           # Plotly charts
└── tests/                  # pytest suite for core/
```

`core/` has no Streamlit imports, so every statistical function can be
tested and reused on its own.

---

## Built with

NumPy for simulation, SciPy and statsmodels for the statistics, pandas for
results, Plotly for charts, Streamlit for the app, and pytest for tests.

---

## Roadmap

- [ ] **Stage 1, statistics core:** sample size, power, two-proportion z-test,
      confidence interval
- [ ] **Stage 2, simulator:** Monte Carlo experiments with a known true effect
- [ ] **Stage 3, peeking demo:** repeated looks and a sequential correction
- [ ] **Stage 4, working app:** all three tools connected in Streamlit
- [ ] **Stage 5, interface polish:** layout, charts and explanations
- [ ] **Stage 6, ship:** deploy to Streamlit Community Cloud
