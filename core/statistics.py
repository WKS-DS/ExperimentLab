"""
Significance test and confidence interval for comparing two conversion rates.

Both functions take plain numbers for a single experiment, or numpy arrays
with one value per experiment. The simulator relies on this to analyze
thousands of experiments in one call instead of looping over them.
"""

import numpy as np
from scipy.stats import norm

# A single number, or an array with one value per experiment
Numeric = float | np.ndarray


def two_proportion_ztest(
    conversions_a: Numeric,
    n_a: Numeric,
    conversions_b: Numeric,
    n_b: Numeric,
) -> tuple[Numeric, Numeric]:
    """Test whether two conversion rates differ.

    Under the null hypothesis both variants share one conversion rate, so the
    standard error uses the pooled rate of both groups. For a 2x2 table this
    is equivalent to a chi-square test without continuity correction: z
    squared equals the chi-square statistic.

    Args:
        conversions_a: Conversions in variant A (control).
        n_a: Users in variant A.
        conversions_b: Conversions in variant B (treatment).
        n_b: Users in variant B.

    Returns:
        (z, p_value), where z is positive when B converts better than A and
        p_value is two-sided. When both groups convert 0% or both convert
        100%, the rates are identical and the result is z = 0, p = 1.

    Example:
        >>> z, p_value = two_proportion_ztest(200, 4_000, 250, 4_000)
        >>> round(float(z), 3), round(float(p_value), 4)
        (2.426, 0.0153)
    """
    difference = conversions_b / n_b - conversions_a / n_a
    pooled_rate = (conversions_a + conversions_b) / (n_a + n_b)
    standard_error = np.sqrt(pooled_rate * (1 - pooled_rate) * (1 / n_a + 1 / n_b))

    # A zero standard error gives 0 / 0, which means no difference.
    with np.errstate(invalid="ignore"):
        z = np.nan_to_num(difference / standard_error)
    p_value = 2 * norm.sf(np.abs(z))
    return z, p_value


def difference_confidence_interval(
    conversions_a: Numeric,
    n_a: Numeric,
    conversions_b: Numeric,
    n_b: Numeric,
    alpha: float = 0.05,
) -> tuple[Numeric, Numeric]:
    """Confidence interval for the difference in conversion rates, B minus A.

    This is the Wald interval. Unlike the z-test, it estimates the size of the
    difference instead of testing for none, so each group keeps its own
    variance. It is accurate at A/B test sample sizes but too narrow when a
    group has only a handful of conversions.

    Args:
        conversions_a: Conversions in variant A (control).
        n_a: Users in variant A.
        conversions_b: Conversions in variant B (treatment).
        n_b: Users in variant B.
        alpha: One minus the confidence level; 0.05 gives a 95% interval.

    Returns:
        (low, high) bounds for rate_b - rate_a.

    Example:
        >>> low, high = difference_confidence_interval(200, 4_000, 250, 4_000)
        >>> round(float(low), 4), round(float(high), 4)
        (0.0024, 0.0226)
    """
    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b
    standard_error = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)

    difference = rate_b - rate_a
    margin = norm.ppf(1 - alpha / 2) * standard_error
    return difference - margin, difference + margin
