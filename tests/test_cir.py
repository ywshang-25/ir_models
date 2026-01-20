"""
Unit tests for the CIR (Cox-Ingersoll-Ross) interest rate model.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from lib.cir import CIRModel, DiscretizationScheme


class TestCIRModelInitialization:
    """Tests for model initialization and parameter validation."""

    def test_valid_initialization(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        assert model.a == 0.5
        assert model.b == 0.05
        assert model.sigma == 0.1
        assert model.r0 == 0.03

    def test_negative_mean_reversion_raises_error(self):
        with pytest.raises(ValueError, match="Mean reversion speed 'a' must be positive"):
            CIRModel(a=-0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_zero_mean_reversion_raises_error(self):
        with pytest.raises(ValueError, match="Mean reversion speed 'a' must be positive"):
            CIRModel(a=0, b=0.05, sigma=0.1, r0=0.03)

    def test_negative_long_term_mean_raises_error(self):
        with pytest.raises(ValueError, match="Long-term mean 'b' must be positive"):
            CIRModel(a=0.5, b=-0.05, sigma=0.1, r0=0.03)

    def test_negative_volatility_raises_error(self):
        with pytest.raises(ValueError, match="Volatility 'sigma' must be positive"):
            CIRModel(a=0.5, b=0.05, sigma=-0.1, r0=0.03)

    def test_negative_initial_rate_raises_error(self):
        with pytest.raises(ValueError, match="Initial rate 'r0' must be positive"):
            CIRModel(a=0.5, b=0.05, sigma=0.1, r0=-0.03)

    def test_zero_initial_rate_raises_error(self):
        with pytest.raises(ValueError, match="Initial rate 'r0' must be positive"):
            CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0)

    def test_repr(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        repr_str = repr(model)
        assert "CIRModel" in repr_str
        assert "a=0.5" in repr_str
        assert "Feller" in repr_str


class TestCIRModelProperties:
    """Tests for model properties."""

    def test_feller_condition_satisfied(self):
        # 2ab = 2*0.5*0.05 = 0.05, σ² = 0.01, so 0.05 >= 0.01
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        assert model.feller_condition_satisfied is True

    def test_feller_condition_not_satisfied(self):
        # 2ab = 2*0.1*0.01 = 0.002, σ² = 0.04, so 0.002 < 0.04
        model = CIRModel(a=0.1, b=0.01, sigma=0.2, r0=0.03)
        assert model.feller_condition_satisfied is False

    def test_feller_ratio(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        expected = 2 * 0.5 * 0.05 / 0.1**2
        assert np.isclose(model.feller_ratio, expected)

    def test_long_term_mean(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        assert model.long_term_mean == 0.05

    def test_long_term_variance(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        expected = 0.05 * 0.1**2 / (2 * 0.5)
        assert np.isclose(model.long_term_variance, expected)

    def test_long_term_std(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        expected = np.sqrt(0.05 * 0.1**2 / (2 * 0.5))
        assert np.isclose(model.long_term_std, expected)


class TestCIRExpectedValues:
    """Tests for analytical expected value and variance formulas."""

    @pytest.fixture
    def model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_expected_rate_at_zero(self, model):
        assert np.isclose(model.expected_rate(0), model.r0)

    def test_expected_rate_converges_to_mean(self, model):
        # As t -> infinity, E[r(t)] -> b
        assert np.isclose(model.expected_rate(100), model.b, atol=1e-10)

    def test_variance_rate_at_zero(self, model):
        assert np.isclose(model.variance_rate(0), 0)

    def test_variance_rate_converges_to_stationary(self, model):
        # As t -> infinity, Var[r(t)] -> bσ²/(2a)
        assert np.isclose(model.variance_rate(100), model.long_term_variance, rtol=0.01)


class TestCIRShortRateSimulation:
    """Tests for short rate simulation."""

    @pytest.fixture
    def model(self):
        # Use parameters satisfying Feller condition
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_simulation_output_shape(self, model):
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=10)
        assert t.shape == (101,)
        assert r.shape == (10, 101)

    def test_simulation_initial_value(self, model):
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=10)
        assert np.all(r[:, 0] == model.r0)

    def test_simulation_time_grid(self, model):
        t, r = model.simulate_short_rate(T=2.0, n_steps=200, n_paths=1)
        assert t[0] == 0.0
        assert np.isclose(t[-1], 2.0)
        assert len(t) == 201

    def test_simulation_reproducibility(self, model):
        t1, r1 = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5, seed=42)
        t2, r2 = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_simulation_different_seeds(self, model):
        t1, r1 = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5, seed=42)
        t2, r2 = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5, seed=123)
        assert not np.allclose(r1, r2)

    @pytest.mark.parametrize("scheme", [
        DiscretizationScheme.EULER,
        DiscretizationScheme.MILSTEIN,
        DiscretizationScheme.FULL_TRUNCATION,
        DiscretizationScheme.REFLECTION
    ])
    def test_all_schemes_run(self, model, scheme):
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=10, scheme=scheme)
        assert r.shape == (10, 101)

    def test_full_truncation_non_negative(self, model):
        """Full truncation scheme should produce non-negative rates."""
        t, r = model.simulate_short_rate(
            T=10.0, n_steps=1000, n_paths=100,
            scheme=DiscretizationScheme.FULL_TRUNCATION, seed=42
        )
        # With Feller condition satisfied, rates should stay positive
        assert np.all(r >= -1e-10)  # Small tolerance for numerical errors

    def test_reflection_non_negative(self, model):
        """Reflection scheme should produce non-negative rates."""
        t, r = model.simulate_short_rate(
            T=10.0, n_steps=1000, n_paths=100,
            scheme=DiscretizationScheme.REFLECTION, seed=42
        )
        assert np.all(r >= 0)


class TestCIRMeanReversion:
    """Tests for mean reversion properties."""

    def test_mean_reversion_from_above(self):
        """Starting above long-term mean should revert down."""
        model = CIRModel(a=1.0, b=0.05, sigma=0.1, r0=0.10)
        t, r = model.simulate_short_rate(T=10.0, n_steps=1000, n_paths=500, seed=42)
        mean_path = np.mean(r, axis=0)
        # Mean should decrease from r0 towards b
        assert mean_path[-1] < model.r0
        assert np.isclose(mean_path[-1], model.b, atol=0.01)

    def test_mean_reversion_from_below(self):
        """Starting below long-term mean should revert up."""
        model = CIRModel(a=1.0, b=0.05, sigma=0.1, r0=0.01)
        t, r = model.simulate_short_rate(T=10.0, n_steps=1000, n_paths=500, seed=42)
        mean_path = np.mean(r, axis=0)
        # Mean should increase from r0 towards b
        assert mean_path[-1] > model.r0
        assert np.isclose(mean_path[-1], model.b, atol=0.01)

    def test_simulated_mean_matches_expected(self):
        """Simulated mean should match analytical expected value."""
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        T = 5.0
        t, r = model.simulate_short_rate(T=T, n_steps=500, n_paths=5000, seed=42)

        simulated_mean = np.mean(r[:, -1])
        expected_mean = model.expected_rate(T)

        assert np.isclose(simulated_mean, expected_mean, rtol=0.05)

    def test_simulated_variance_matches_expected(self):
        """Simulated variance should match analytical expected variance."""
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        T = 5.0
        t, r = model.simulate_short_rate(T=T, n_steps=500, n_paths=5000, seed=42)

        simulated_var = np.var(r[:, -1])
        expected_var = model.variance_rate(T)

        assert np.isclose(simulated_var, expected_var, rtol=0.1)


class TestCIRMonteCarloBondPricing:
    """Tests for Monte Carlo bond pricing."""

    @pytest.fixture
    def model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_bond_price_returns_tuple(self, model):
        result = model.monte_carlo_bond_price(T=1.0, n_paths=1000, n_steps=50, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bond_price_in_valid_range(self, model):
        price, std_error = model.monte_carlo_bond_price(T=1.0, n_paths=1000, n_steps=50, seed=42)
        assert 0 < price < 1
        assert std_error > 0

    def test_bond_price_decreases_with_maturity(self, model):
        price_1y, _ = model.monte_carlo_bond_price(T=1.0, n_paths=5000, n_steps=50, seed=42)
        price_5y, _ = model.monte_carlo_bond_price(T=5.0, n_paths=5000, n_steps=250, seed=42)
        assert price_1y > price_5y

    def test_bond_price_std_error_decreases_with_paths(self, model):
        _, se_100 = model.monte_carlo_bond_price(T=1.0, n_paths=100, n_steps=50, seed=42)
        _, se_10000 = model.monte_carlo_bond_price(T=1.0, n_paths=10000, n_steps=50, seed=42)
        assert se_10000 < se_100

    def test_bond_price_reproducibility(self, model):
        price1, _ = model.monte_carlo_bond_price(T=1.0, n_paths=1000, n_steps=50, seed=42)
        price2, _ = model.monte_carlo_bond_price(T=1.0, n_paths=1000, n_steps=50, seed=42)
        assert price1 == price2


class TestCIRMonteCarloYieldCurve:
    """Tests for Monte Carlo yield curve estimation."""

    @pytest.fixture
    def model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_yield_curve_shape(self, model):
        maturities = np.array([0.5, 1, 2, 5])
        yields, std_errors = model.monte_carlo_yield_curve(
            maturities, n_paths=1000, n_steps_per_year=50, seed=42
        )
        assert yields.shape == (4,)
        assert std_errors.shape == (4,)

    def test_yields_positive(self, model):
        maturities = np.array([0.5, 1, 2, 5])
        yields, _ = model.monte_carlo_yield_curve(
            maturities, n_paths=2000, n_steps_per_year=50, seed=42
        )
        assert np.all(yields > 0)

    def test_std_errors_positive(self, model):
        maturities = np.array([0.5, 1, 2])
        _, std_errors = model.monte_carlo_yield_curve(
            maturities, n_paths=1000, n_steps_per_year=50, seed=42
        )
        assert np.all(std_errors > 0)


class TestCIRVsVasicekComparison:
    """Tests comparing CIR behavior to Vasicek-like behavior."""

    def test_low_volatility_similar_mean(self):
        """With low volatility, CIR should behave similarly to deterministic mean reversion."""
        model = CIRModel(a=1.0, b=0.05, sigma=0.001, r0=0.03)
        t, r = model.simulate_short_rate(T=5.0, n_steps=500, n_paths=100, seed=42)

        # All paths should be very close to expected path
        for i, ti in enumerate(t):
            expected = model.expected_rate(ti)
            assert np.allclose(r[:, i], expected, atol=0.001)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
