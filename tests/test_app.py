"""Tests for the Streamlit app, run headless with Streamlit's AppTest.

These check that inputs reach the right core functions, that the expected
values shown next to each simulation match core/, and that no combination of
inputs crashes the app. The statistics themselves are tested in the other
test files.
"""

import pytest
from streamlit.testing.v1 import AppTest

from core.power_analysis import calculate_power, calculate_sample_size
from core.sequential import peeking_false_positive_rate

APP_PATH = "../app.py"
TIMEOUT_SECONDS = 60


def run_app(**inputs: float) -> AppTest:
    """Run the app, then rerun it with widget values set by key or label."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT_SECONDS)
    app.run()
    if inputs:
        for name, value in inputs.items():
            find_widget(app, name).set_value(value)
        app.run()
    return app


def find_widget(app: AppTest, name: str):
    """Find a widget by its key, or by its label if it has no key."""
    widgets = [*app.slider, *app.selectbox, *app.number_input]
    matches = [w for w in widgets if name in (w.key, w.label)]
    assert len(matches) == 1, f"expected one widget named {name!r}"
    return matches[0]


def table(app: AppTest, index: int) -> dict[str, tuple[str, str]]:
    """Read a comparison table as {row label: (simulated, expected)}."""
    data = app.dataframe[index].value
    return {row[""]: (row["Simulated"], row["Expected"]) for _, row in data.iterrows()}


def assert_runs_cleanly(app: AppTest) -> None:
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


class TestDesignInputs:
    def test_default_design(self):
        app = run_app()
        assert_runs_cleanly(app)
        assert app.metric[0].value == "31,218"
        assert app.metric[1].value == "62,436"

    @pytest.mark.parametrize(
        "baseline, mde, alpha, power",
        [
            (0.5, 1, 0.05, 0.80),  # tiny lift on a low baseline: 31 million users
            (0.5, 100, 0.05, 0.80),
            (49.5, 1, 0.05, 0.80),
            (49.5, 100, 0.05, 0.80),  # treatment converts 99%: 9 users
            (5.0, 10, 0.01, 0.95),
            (5.0, 10, 0.10, 0.70),
        ],
    )
    def test_every_section_matches_core(self, baseline, mde, alpha, power):
        app = run_app(baseline=baseline, mde=mde, alpha=alpha, power=power)
        assert_runs_cleanly(app)

        n = calculate_sample_size(baseline / 100, mde / 100, alpha, power)
        assert app.metric[0].value == f"{n:,}"

        expected_power = calculate_power(baseline / 100, mde / 100, n, alpha)
        simulator = table(app, 0)
        assert simulator["Detected the lift (power)"][1] == f"{expected_power:.1%}"
        assert simulator["Intervals containing the true difference"][1] == (
            f"{1 - alpha:.1%}"
        )

        peeking = table(app, 1)
        expected_peeking = peeking_false_positive_rate(10, alpha)
        assert peeking[f"Check every look, stop at the first p < {alpha:g}"][1] == (
            f"{expected_peeking:.1%}"
        )

    def test_impossible_design_shows_message(self):
        """A 100% lift on a 50% baseline would mean 100% conversion."""
        app = run_app(baseline=50.0, mde=100)
        assert not app.exception
        assert len(app.error) == 1
        assert "treatment rate of 100%" in app.error[0].value
        assert not app.metric

    def test_other_sections_follow_the_calculated_sample_size(self):
        app = run_app(mde=20)
        n = calculate_sample_size(0.05, 0.20)
        assert find_widget(app, "Users per variant").value == n
        assert find_widget(app, "Users per variant at the final look").value == n


class TestSimulator:
    def test_aa_test_reports_false_positives(self):
        app = run_app(**{"True lift (%)": 0})
        assert_runs_cleanly(app)
        simulator = table(app, 0)
        assert "Detected the lift (power)" not in simulator
        assert simulator["False positives"][1] == "5.0%"

    def test_power_at_a_chosen_sample_size(self):
        app = run_app(**{"Users per variant": 10_000})
        expected = calculate_power(0.05, 0.10, 10_000)
        assert table(app, 0)["Detected the lift (power)"][1] == f"{expected:.1%}"

    def test_impossible_true_lift_shows_message(self):
        """The design is valid, but a 100% true lift on 50% is not."""
        app = run_app(baseline=50.0, mde=50)
        find_widget(app, "True lift (%)").set_value(100)
        app.run()
        assert not app.exception
        assert len(app.error) == 1
        assert "treatment rate of 100%" in app.error[0].value
        assert len(app.dataframe) == 2  # the peeking section still runs

    def test_redraw_button_uses_a_new_seed(self):
        app = run_app()
        app.button(key="simulator_redraw").click()
        app.run()
        assert_runs_cleanly(app)
        assert app.session_state["simulator_seed"] == 1
        assert app.session_state["peeking_seed"] == 0


class TestPeeking:
    @pytest.mark.parametrize("n_looks", [1, 20])
    def test_extreme_numbers_of_looks(self, n_looks):
        app = run_app(looks=n_looks)
        assert_runs_cleanly(app)
        expected = peeking_false_positive_rate(n_looks)
        peeking = table(app, 1)
        assert peeking["Check every look, stop at the first p < 0.05"][1] == (
            f"{expected:.1%}"
        )
        thresholds = app.dataframe[2].value
        assert len(thresholds) == n_looks
        assert thresholds["Users per variant"].iloc[-1] == 31_218

    def test_few_conversions_per_look_shows_note(self):
        """With 20 users per look at a 5% baseline, each look expects 1."""
        app = run_app(**{"Users per variant at the final look": 200})
        assert_runs_cleanly(app)
        assert any("only approximate" in caption.value for caption in app.caption)
