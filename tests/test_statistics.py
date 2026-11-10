"""Tests for core.statistics."""

import numpy as np
import pytest
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from core.statistics import difference_confidence_interval, two_proportion_ztest

# Worked example: control converts 200 of 4,000 users (5.0%) and treatment
# converts 250 of 4,000 (6.25%).
CONV_A, N_A, CONV_B, N_B = 200, 4_000, 250, 4_000


class TestTwoProportionZtest:
    def test_known_value(self):
        z, p_value = two_proportion_ztest(CONV_A, N_A, CONV_B, N_B)
        assert z == pytest.approx(2.4262, abs=1e-4)
        assert p_value == pytest.approx(0.0153, abs=1e-4)

    def test_matches_statsmodels(self):
        expected_z, expected_p = proportions_ztest([CONV_B, CONV_A], [N_B, N_A])
        z, p_value = two_proportion_ztest(CONV_A, N_A, CONV_B, N_B)
        assert z == pytest.approx(expected_z)
        assert p_value == pytest.approx(expected_p)

    def test_z_squared_equals_chi_square(self):
        table = [[CONV_A, N_A - CONV_A], [CONV_B, N_B - CONV_B]]
        chi_square = chi2_contingency(table, correction=False)[0]
        z, _ = two_proportion_ztest(CONV_A, N_A, CONV_B, N_B)
        assert z**2 == pytest.approx(chi_square)

    def test_negative_when_treatment_is_worse(self):
        z, _ = two_proportion_ztest(CONV_B, N_B, CONV_A, N_A)
        assert z < 0

    @pytest.mark.parametrize("conversions", [0, 37, 100])
    def test_identical_groups(self, conversions):
        """Includes 0% and 100% conversion, where the standard error is zero."""
        z, p_value = two_proportion_ztest(conversions, 100, conversions, 100)
        assert z == 0
        assert p_value == 1

    def test_accepts_arrays(self):
        """Many experiments at once give the same results as one at a time."""
        conv_a = np.array([200, 210, 0])
        conv_b = np.array([250, 190, 0])
        z, p_value = two_proportion_ztest(conv_a, N_A, conv_b, N_B)
        for i in range(len(conv_a)):
            one_z, one_p = two_proportion_ztest(conv_a[i], N_A, conv_b[i], N_B)
            assert z[i] == pytest.approx(one_z)
            assert p_value[i] == pytest.approx(one_p)


class TestDifferenceConfidenceInterval:
    def test_known_value(self):
        low, high = difference_confidence_interval(CONV_A, N_A, CONV_B, N_B)
        assert low == pytest.approx(0.00241, abs=1e-5)
        assert high == pytest.approx(0.02259, abs=1e-5)

    def test_matches_statsmodels(self):
        expected = confint_proportions_2indep(
            CONV_B, N_B, CONV_A, N_A, compare="diff", method="wald"
        )
        interval = difference_confidence_interval(CONV_A, N_A, CONV_B, N_B)
        assert interval == pytest.approx(expected)

    def test_centered_on_observed_difference(self):
        low, high = difference_confidence_interval(CONV_A, N_A, CONV_B, N_B)
        assert (low + high) / 2 == pytest.approx(CONV_B / N_B - CONV_A / N_A)

    def test_higher_confidence_gives_wider_interval(self):
        low_95, high_95 = difference_confidence_interval(CONV_A, N_A, CONV_B, N_B)
        low_99, high_99 = difference_confidence_interval(
            CONV_A, N_A, CONV_B, N_B, alpha=0.01
        )
        assert high_99 - low_99 > high_95 - low_95

    def test_accepts_arrays(self):
        conv_a = np.array([200, 210, 190])
        conv_b = np.array([250, 190, 230])
        low, high = difference_confidence_interval(conv_a, N_A, conv_b, N_B)
        for i in range(len(conv_a)):
            one_low, one_high = difference_confidence_interval(
                conv_a[i], N_A, conv_b[i], N_B
            )
            assert low[i] == pytest.approx(one_low)
            assert high[i] == pytest.approx(one_high)
