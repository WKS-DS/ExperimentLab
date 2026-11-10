"""
Monte Carlo simulation of A/B tests on conversion rates.

The power calculator predicts how often an experiment will detect a lift by
using a formula. The simulator checks that prediction by counting: it runs the
same experiment thousands of times on synthetic users whose true conversion
rates are known, and analyzes each run the way a real experiment would be
analyzed. When the counts match the formula, the sample size calculation and
the significance test agree with each other.

All experiments are drawn and analyzed together as numpy arrays, so 10,000 of
them take a few milliseconds.
"""

import numpy as np
import pandas as pd

from core.statistics import difference_confidence_interval, two_proportion_ztest


def simulate_experiments(
    baseline_rate: float,
    true_lift: float,
    n_per_variant: int,
    n_experiments: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run many simulated A/B tests with a known true lift.

    In each experiment, n_per_variant control users convert at baseline_rate
    and n_per_variant treatment users convert at baseline_rate * (1 +
    true_lift), so the conversions in each group are a binomial draw. Each
    experiment is then analyzed with a two-proportion z-test and a confidence
    interval for the difference in conversion rates.

    Args:
        baseline_rate: True control conversion rate, e.g. 0.05 for 5%.
        true_lift: True relative lift of treatment over control, e.g. 0.10
            for 10%. Use 0 for an A/A test, where every significant result is
            a false positive, or a negative value for a change that hurts.
        n_per_variant: Users in each variant of each experiment.
        n_experiments: Number of experiments to simulate.
        alpha: Significance level for the test and the interval.
        seed: Seed for the random number generator. The same seed gives the
            same experiments.

    Returns:
        One row per experiment, with columns

        - conversions_a, conversions_b: conversions in control and treatment.
        - observed_lift: measured relative lift, rate_b / rate_a - 1. NaN when
          control has no conversions.
        - z, p_value: the two-proportion z-test. z is positive when treatment
          converts better.
        - ci_low, ci_high: confidence interval for rate_b - rate_a.
        - significant: True when p_value < alpha.
        - ci_covers_truth: True when the interval contains the true
          difference in rates.

    Raises:
        ValueError: If baseline_rate or alpha is outside (0, 1), the lift
            pushes the treatment rate outside (0, 1), or n_per_variant or
            n_experiments is not positive.

    Example:
        >>> results = simulate_experiments(0.05, 0.10, 31_218, seed=0)
        >>> len(results)
        10000
    """
    for name, value in [("baseline_rate", baseline_rate), ("alpha", alpha)]:
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    for name, value in [
        ("n_per_variant", n_per_variant),
        ("n_experiments", n_experiments),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    treatment_rate = baseline_rate * (1 + true_lift)
    if not 0 < treatment_rate < 1:
        raise ValueError(
            f"A {true_lift:.0%} lift on a {baseline_rate:.0%} baseline gives a "
            f"treatment rate of {treatment_rate:.0%}; it must be above 0% and "
            "below 100%"
        )

    rng = np.random.default_rng(seed)
    conversions_a = rng.binomial(n_per_variant, baseline_rate, size=n_experiments)
    conversions_b = rng.binomial(n_per_variant, treatment_rate, size=n_experiments)

    n = n_per_variant
    z, p_value = two_proportion_ztest(conversions_a, n, conversions_b, n)
    ci_low, ci_high = difference_confidence_interval(
        conversions_a, n, conversions_b, n, alpha
    )

    # The groups are the same size, so the ratio of rates is the ratio of counts.
    with np.errstate(divide="ignore", invalid="ignore"):
        observed_lift = conversions_b / conversions_a - 1
    observed_lift[conversions_a == 0] = np.nan

    true_difference = treatment_rate - baseline_rate
    return pd.DataFrame(
        {
            "conversions_a": conversions_a,
            "conversions_b": conversions_b,
            "observed_lift": observed_lift,
            "z": z,
            "p_value": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": p_value < alpha,
            "ci_covers_truth": (ci_low <= true_difference)
            & (true_difference <= ci_high),
        }
    )


def summarize_simulation(results: pd.DataFrame) -> dict[str, float]:
    """Summarize simulated experiments as the rates they are meant to check.

    A two-sided test can also reach significance in the wrong direction and
    call a better treatment worse. Those results count towards
    significant_rate but not treatment_win_rate. That is why
    treatment_win_rate, not significant_rate, matches calculate_power. The
    gap between them is noticeable only when power is low, at about 0.7
    points for a 5% baseline, a 10% lift and 1,000 users per variant.

    Args:
        results: Output of simulate_experiments.

    Returns:
        A dict with

        - treatment_win_rate: share of experiments that are significant with
          treatment ahead. With a real lift this is the empirical power.
        - significant_rate: share that are significant in either direction.
          With no real difference this is the false positive rate, which
          should be close to alpha.
        - ci_coverage: share of intervals that contain the true difference,
          which should be close to the confidence level, 1 - alpha.

    Example:
        >>> results = simulate_experiments(0.05, 0.10, 31_218, seed=0)
        >>> summary = summarize_simulation(results)
        >>> f"{summary['treatment_win_rate']:.0%}"
        '80%'
    """
    significant = results["significant"]
    return {
        "treatment_win_rate": float((significant & (results["z"] > 0)).mean()),
        "significant_rate": float(significant.mean()),
        "ci_coverage": float(results["ci_covers_truth"].mean()),
    }
