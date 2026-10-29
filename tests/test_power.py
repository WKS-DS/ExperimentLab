"""
Tests for power analysis functions.

Run with: pytest tests/test_power.py -v
"""

import pytest

from core.power_analysis import calculate_power, calculate_sample_size


class TestCalculateSampleSize:
    """Tests for calculate_sample_size function."""

    def test_basic_calculation(self):
        """Test sample size for 5% baseline, 10% MDE, 80% power."""
        # Expected value verified against online calculator
        # https://www.evanmiller.org/ab-testing/sample-size.html
        n = calculate_sample_size(
            baseline_rate=0.05,
            minimum_detectable_effect=0.10,  # 10% relative lift
            alpha=0.05,
            power=0.80,
        )

        # Should be approximately 31,000 per variant
        # Allow some tolerance for different calculation methods
        assert 28000 < n < 35000, f"Expected ~31000, got {n}"

    def test_higher_baseline_needs_smaller_sample(self):
        """Higher baseline rates need smaller samples for same relative MDE."""
        n_low_baseline = calculate_sample_size(0.05, 0.10)
        n_high_baseline = calculate_sample_size(0.20, 0.10)

        assert n_high_baseline < n_low_baseline

    def test_larger_effect_needs_smaller_sample(self):
        """Larger effects are easier to detect, need smaller samples."""
        n_small_effect = calculate_sample_size(0.05, 0.05)
        n_large_effect = calculate_sample_size(0.05, 0.20)

        assert n_large_effect < n_small_effect

    def test_higher_power_needs_larger_sample(self):
        """Higher power requires larger sample size."""
        n_80_power = calculate_sample_size(0.05, 0.10, power=0.80)
        n_90_power = calculate_sample_size(0.05, 0.10, power=0.90)

        assert n_90_power > n_80_power

    def test_returns_integer(self):
        """Sample size should be a whole number."""
        n = calculate_sample_size(0.05, 0.10)
        assert isinstance(n, int)


class TestCalculatePower:
    """Tests for calculate_power function."""

    def test_power_increases_with_sample_size(self):
        """More samples = more power."""
        power_small = calculate_power(0.05, 0.055, n_per_variant=1000)
        power_large = calculate_power(0.05, 0.055, n_per_variant=10000)

        assert power_large > power_small

    def test_power_increases_with_effect_size(self):
        """Larger effects are easier to detect."""
        power_small_effect = calculate_power(0.05, 0.055, n_per_variant=5000)
        power_large_effect = calculate_power(0.05, 0.070, n_per_variant=5000)

        assert power_large_effect > power_small_effect
