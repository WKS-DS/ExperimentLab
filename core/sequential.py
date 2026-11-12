"""
Repeated looks at a running experiment, and a correction that allows them.

A p-value below 0.05 means a 5% false positive rate only if the test is run
once, at the planned sample size. Checking a running experiment several times
and stopping at the first significant result gives chance several attempts,
and the false positive rate grows with every look.

This module measures that inflation in two ways: exactly, by numerical
integration, and empirically, by simulating A/A tests. It also provides
O'Brien-Fleming boundaries, a threshold for each look that brings the overall
false positive rate back to alpha.

Looks are equally spaced. With K looks at an experiment of n users per
variant, look k happens after k * n / K users.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from core.statistics import two_proportion_ztest

# Spacing of the integration grid. Crossing probabilities are accurate to
# about 1e-6, far below Monte Carlo error.
_GRID_SPACING = 0.1


def peeking_false_positive_rate(n_looks: int, alpha: float = 0.05) -> float:
    """False positive rate when an A/A test is checked repeatedly.

    The experiment is checked at n_looks equally spaced points and stopped
    the first time p < alpha. With no true difference, every stop is a false
    positive. One look gives alpha; more looks give more chances to cross the
    threshold by chance.

    Args:
        n_looks: Number of times the experiment is checked, including the
            final analysis.
        alpha: Significance level used at every look.

    Returns:
        Probability of a significant result at one or more looks.

    Raises:
        ValueError: If n_looks is not positive or alpha is outside (0, 1).

    Example:
        >>> round(peeking_false_positive_rate(5), 3)
        0.142
    """
    _check_looks_and_alpha(n_looks, alpha)
    threshold = norm.ppf(1 - alpha / 2)
    return _crossing_probability(np.full(n_looks, threshold))


def obrien_fleming_boundaries(n_looks: int, alpha: float = 0.05) -> np.ndarray:
    """Significance thresholds for each look that keep false positives at alpha.

    The O'Brien-Fleming boundary at look k of K requires |z| > c * sqrt(K / k).
    The bar is very high at early looks, when little data has come in, and
    falls to c at the final look, only slightly above the usual threshold. The
    experiment can still stop early when the evidence is overwhelming, and the
    final analysis loses little power. The constant c is chosen so that an A/A
    test crosses at least one boundary with probability exactly alpha.

    Args:
        n_looks: Number of equally spaced looks, including the final analysis.
        alpha: Overall false positive rate across all looks.

    Returns:
        One z threshold per look, in order. A result is significant at look k
        when |z| exceeds the k-th threshold.

    Raises:
        ValueError: If n_looks is not positive or alpha is outside (0, 1).

    Example:
        >>> [round(float(b), 2) for b in obrien_fleming_boundaries(5)]
        [4.56, 3.23, 2.63, 2.28, 2.04]
    """
    _check_looks_and_alpha(n_looks, alpha)
    looks = np.arange(1, n_looks + 1)
    shape = np.sqrt(n_looks / looks)
    single_look = norm.ppf(1 - alpha / 2)
    if n_looks == 1:
        return shape * single_look

    # Splitting alpha evenly across looks (Bonferroni) is too strict, and the
    # single-look threshold is too lenient, so c lies between them.
    constant = brentq(
        lambda c: _crossing_probability(c * shape) - alpha,
        single_look,
        norm.ppf(1 - alpha / (2 * n_looks)),
        xtol=1e-8,
    )
    return shape * constant


def simulate_peeking(
    baseline_rate: float,
    n_per_variant: int,
    n_looks: int,
    n_experiments: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate A/A tests and compute the z-statistic at every look.

    Both variants convert at baseline_rate, so there is no true difference.
    Users arrive in n_looks equal batches, and at each look the two-proportion
    z-test is run on all users so far. Each look reuses the data from earlier
    looks, which is why repeated looks are correlated and why the false
    positive rate grows more slowly than the number of looks.

    Args:
        baseline_rate: Conversion rate of both variants, e.g. 0.05 for 5%.
        n_per_variant: Users in each variant by the final look.
        n_looks: Number of equally spaced looks, including the final one.
        n_experiments: Number of A/A tests to simulate.
        seed: Seed for the random number generator. The same seed gives the
            same experiments.

    Returns:
        z-statistics with one row per experiment and one column per look. The
        column labels are the users per variant at each look.

    Raises:
        ValueError: If baseline_rate is outside (0, 1), n_looks or
            n_experiments is not positive, or n_per_variant is smaller than
            n_looks.

    Example:
        >>> z = simulate_peeking(0.05, 30_000, n_looks=3, seed=0)
        >>> z.shape, list(z.columns)
        ((10000, 3), [10000, 20000, 30000])
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be between 0 and 1, got {baseline_rate}")
    if n_looks <= 0 or n_experiments <= 0:
        raise ValueError(
            "n_looks and n_experiments must be positive, "
            f"got {n_looks} and {n_experiments}"
        )
    if n_per_variant < n_looks:
        raise ValueError(
            f"n_per_variant ({n_per_variant}) must be at least n_looks ({n_looks}) "
            "so that every look has new users"
        )

    users_at_look = np.arange(1, n_looks + 1) * n_per_variant // n_looks
    new_users = np.diff(users_at_look, prepend=0)

    rng = np.random.default_rng(seed)
    size = (n_experiments, n_looks)
    conversions_a = rng.binomial(new_users, baseline_rate, size=size).cumsum(axis=1)
    conversions_b = rng.binomial(new_users, baseline_rate, size=size).cumsum(axis=1)

    z, _ = two_proportion_ztest(
        conversions_a, users_at_look, conversions_b, users_at_look
    )
    return pd.DataFrame(z, columns=pd.Index(users_at_look, name="users_per_variant"))


def summarize_peeking(z_by_look: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    """False positive rates of simulated A/A tests under three ways of looking.

    Args:
        z_by_look: Output of simulate_peeking.
        alpha: Significance level.

    Returns:
        A dict with

        - final_look_only: share significant at the final look, the single
          planned analysis. Close to alpha.
        - any_look: share significant at one or more looks, the false positive
          rate of stopping at the first p < alpha. Close to
          peeking_false_positive_rate.
        - obrien_fleming: share that cross an O'Brien-Fleming boundary at one
          or more looks. Close to alpha.

    Example:
        Simulated rates are within a point of the exact ones. The draws for a
        given seed can change between NumPy versions, so the example compares
        rates instead of printing them.

        >>> z = simulate_peeking(0.05, 31_218, n_looks=10, seed=0)
        >>> summary = summarize_peeking(z)
        >>> abs(summary["any_look"] - peeking_false_positive_rate(10)) < 0.01
        True
        >>> abs(summary["obrien_fleming"] - 0.05) < 0.01
        True
    """
    abs_z = np.abs(z_by_look.to_numpy())
    threshold = norm.ppf(1 - alpha / 2)
    boundaries = obrien_fleming_boundaries(abs_z.shape[1], alpha)
    return {
        "final_look_only": float((abs_z[:, -1] > threshold).mean()),
        "any_look": float((abs_z > threshold).any(axis=1).mean()),
        "obrien_fleming": float((abs_z > boundaries).any(axis=1).mean()),
    }


def _crossing_probability(boundaries: np.ndarray) -> float:
    """Probability that an A/A test has |z| above the boundary at some look.

    With equally spaced looks, the running sum of standard normal increments
    S_k satisfies z_k = S_k / sqrt(k), so the boundary at look k is the
    interval |S_k| < boundaries[k] * sqrt(k). The function follows the density
    of S_k over experiments that have not stopped yet: at each look it adds
    the next increment by convolving with the normal density, then drops the
    part outside the boundary. What remains after the last look is the
    probability of never stopping. The integrals use Simpson's rule.
    """
    limits = np.asarray(boundaries) * np.sqrt(np.arange(1, len(boundaries) + 1))

    s, weights = _simpson_grid(limits[0])
    density = _normal_pdf(s)
    for limit in limits[1:]:
        s_next, weights_next = _simpson_grid(limit)
        density = _normal_pdf(s_next[:, None] - s[None, :]) @ (weights * density)
        s, weights = s_next, weights_next
    return float(1 - weights @ density)


def _simpson_grid(limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Points and Simpson's rule weights for integrating over (-limit, limit)."""
    n_points = 2 * int(np.ceil(limit / _GRID_SPACING)) + 1
    points = np.linspace(-limit, limit, n_points)
    weights = np.ones(n_points)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    return points, weights * (points[1] - points[0]) / 3


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal density, without scipy's per-call overhead."""
    return np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)


def _check_looks_and_alpha(n_looks: int, alpha: float) -> None:
    """Raise ValueError unless n_looks is positive and 0 < alpha < 1."""
    if n_looks <= 0:
        raise ValueError(f"n_looks must be positive, got {n_looks}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
