"""
Power analysis and sample size calculations for A/B tests on conversion rates.

Both functions describe a two-sided, two-sample z-test with equal group sizes.
The difference between the two conversion rates is measured with Cohen's h,
which puts proportions on a scale where the variance no longer depends on the
rate, so one formula works at any baseline.
"""

import math

from scipy.stats import norm


def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate the users needed per variant to detect a given lift.

    Power is the probability that the test reaches significance when the true
    lift equals the minimum detectable effect. The required sample size is

        n = 2 * ((z_{1-alpha/2} + z_{1-beta}) / h)^2

    rounded up, so the test reaches at least the requested power.

    Args:
        baseline_rate: Control conversion rate, e.g. 0.05 for 5%.
        minimum_detectable_effect: Smallest relative lift worth detecting,
            e.g. 0.10 for a 10% lift (5.0% to 5.5%).
        alpha: Significance level, the false positive rate when there is no
            real difference.
        power: Target probability of detecting the lift if it is real.

    Returns:
        Users needed in each variant.

    Raises:
        ValueError: If a rate or probability is outside (0, 1), the lift is
            not positive, or the lift pushes the treatment rate to 100% or more.

    Example:
        >>> calculate_sample_size(0.05, 0.10)
        31218
    """
    _check_probability("alpha", alpha)
    _check_probability("power", power)
    h = _cohens_h(baseline_rate, minimum_detectable_effect)

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return math.ceil(2 * ((z_alpha + z_power) / h) ** 2)


def calculate_power(
    baseline_rate: float,
    minimum_detectable_effect: float,
    n_per_variant: int,
    alpha: float = 0.05,
) -> float:
    """Calculate the power of a test with a fixed number of users per variant.

    This is the inverse of calculate_sample_size:

        power = Phi(h * sqrt(n / 2) - z_{1-alpha/2})

    where Phi is the standard normal CDF.

    Args:
        baseline_rate: Control conversion rate, e.g. 0.05 for 5%.
        minimum_detectable_effect: True relative lift, e.g. 0.10 for 10%.
        n_per_variant: Users in each variant.
        alpha: Significance level.

    Returns:
        Probability that the test reaches significance, between 0 and 1.

    Raises:
        ValueError: If an input is out of range (see calculate_sample_size),
            or n_per_variant is not positive.

    Example:
        >>> round(calculate_power(0.05, 0.10, 31_218), 3)
        0.8
    """
    _check_probability("alpha", alpha)
    if n_per_variant <= 0:
        raise ValueError(f"n_per_variant must be positive, got {n_per_variant}")
    h = _cohens_h(baseline_rate, minimum_detectable_effect)

    z_alpha = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(h * math.sqrt(n_per_variant / 2) - z_alpha))


def _cohens_h(baseline_rate: float, minimum_detectable_effect: float) -> float:
    """Return Cohen's h between the baseline rate and the lifted treatment rate."""
    _check_probability("baseline_rate", baseline_rate)
    if minimum_detectable_effect <= 0:
        raise ValueError(
            "minimum_detectable_effect must be positive, "
            f"got {minimum_detectable_effect}"
        )
    treatment_rate = baseline_rate * (1 + minimum_detectable_effect)
    if treatment_rate >= 1:
        raise ValueError(
            f"A {minimum_detectable_effect:.0%} lift on a {baseline_rate:.0%} "
            "baseline gives a treatment rate of 100% or more"
        )
    treatment_angle = math.asin(math.sqrt(treatment_rate))
    baseline_angle = math.asin(math.sqrt(baseline_rate))
    return 2 * (treatment_angle - baseline_angle)


def _check_probability(name: str, value: float) -> None:
    """Raise ValueError unless 0 < value < 1."""
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
