"""
Unit tests for the Vasicek interest rate model.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from lib.vasicek import VasicekModel


class TestVasicekModelInitialization:
    """Tests for model initialization and parameter validation."""

    def test_valid_initialization(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        assert model.a == 0.5
        assert model.b == 0.05
        assert model.sigma == 0.02
        assert model.r0 == 0.03

    def test_negative_mean_reversion_raises_error(self):
        with pytest.raises(ValueError, match="Mean reversion speed 'a' must be positive"):
            VasicekModel(a=-0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_zero_mean_reversion_raises_error(self):
        with pytest.raises(ValueError, match="Mean reversion speed 'a' must be positive"):
            VasicekModel(a=0, b=0.05, sigma=0.02, r0=0.03)

    def test_negative_volatility_raises_error(self):
        with pytest.raises(ValueError, match="Volatility 'sigma' must be positive"):
            VasicekModel(a=0.5, b=0.05, sigma=-0.02, r0=0.03)

    def test_zero_volatility_raises_error(self):
        with pytest.raises(ValueError, match="Volatility 'sigma' must be positive"):
            VasicekModel(a=0.5, b=0.05, sigma=0, r0=0.03)

    def test_repr(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        repr_str = repr(model)
        assert "VasicekModel" in repr_str
        assert "a=0.5" in repr_str
        assert "b=0.05" in repr_str


class TestVasicekModelProperties:
    """Tests for model properties."""

    def test_long_term_mean(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        assert model.long_term_mean == 0.05

    def test_long_term_variance(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        expected_variance = 0.02**2 / (2 * 0.5)
        assert np.isclose(model.long_term_variance, expected_variance)

    def test_long_term_std(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        expected_std = np.sqrt(0.02**2 / (2 * 0.5))
        assert np.isclose(model.long_term_std, expected_std)


class TestShortRateSimulation:
    """Tests for short rate simulation."""

    @pytest.fixture
    def model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

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

    def test_mean_reversion_property(self, model):
        """Test that simulated rates converge to long-term mean."""
        t, r = model.simulate_short_rate(T=50.0, n_steps=5000, n_paths=1000, seed=42)
        terminal_mean = np.mean(r[:, -1])
        assert np.isclose(terminal_mean, model.b, atol=0.005)

    def test_stationary_variance(self, model):
        """Test that variance converges to theoretical stationary variance."""
        t, r = model.simulate_short_rate(T=50.0, n_steps=5000, n_paths=1000, seed=42)
        terminal_var = np.var(r[:, -1])
        assert np.isclose(terminal_var, model.long_term_variance, rtol=0.1)


class TestInstantaneousForwardRate:
    """Tests for instantaneous forward rate calculation."""

    @pytest.fixture
    def model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_forward_rate_at_t_equals_T(self, model):
        """Forward rate at t=T should equal the short rate."""
        r_t = 0.04
        f = model.instantaneous_forward_rate(np.array([r_t]), t=0, T=np.array([0]))
        assert np.isclose(f, r_t)

    def test_forward_rate_shape_single_path(self, model):
        r_t = np.array([0.04])
        T = np.array([1, 2, 3, 5, 10])
        f = model.instantaneous_forward_rate(r_t, t=0, T=T)
        assert f.shape == (5,)

    def test_forward_rate_shape_multiple_paths(self, model):
        r_t = np.array([0.03, 0.04, 0.05])
        T = np.array([1, 2, 3, 5, 10])
        f = model.instantaneous_forward_rate(r_t, t=0, T=T)
        assert f.shape == (3, 5)

    def test_forward_rate_converges_to_long_term(self, model):
        """Forward rate should converge to long-term mean minus adjustment."""
        r_t = np.array([0.04])
        T = np.array([100])  # Far in the future
        f = model.instantaneous_forward_rate(r_t, t=0, T=T)
        # As T -> infinity, f(0,T) -> b - sigma^2/(2a^2)
        expected_limit = model.b - model.sigma**2 / (2 * model.a**2)
        assert np.isclose(f, expected_limit, atol=1e-6)

    def test_negative_maturity_raises_error(self, model):
        with pytest.raises(ValueError, match="Maturity T must be >= current time t"):
            model.instantaneous_forward_rate(np.array([0.04]), t=5, T=np.array([3]))


class TestForwardRateCurveSimulation:
    """Tests for forward rate curve simulation."""

    @pytest.fixture
    def model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_forward_curve_output_shape(self, model):
        maturities = np.array([0.25, 0.5, 1, 2, 5])
        t, r, f = model.simulate_forward_rate_curve(
            T_horizon=2.0,
            n_time_steps=20,
            maturities=maturities,
            n_paths=10,
            seed=42
        )
        assert t.shape == (21,)
        assert r.shape == (10, 21)
        assert f.shape == (10, 21, 5)

    def test_forward_curve_initial_matches_forward_rate(self, model):
        """Initial forward curve should match instantaneous forward rate."""
        maturities = np.array([0.5, 1, 2, 5])
        t, r, f = model.simulate_forward_rate_curve(
            T_horizon=1.0,
            n_time_steps=10,
            maturities=maturities,
            n_paths=1,
            seed=42
        )
        expected_f0 = model.instantaneous_forward_rate(np.array([model.r0]), 0, maturities)
        np.testing.assert_array_almost_equal(f[0, 0, :], expected_f0)


class TestZeroCouponBondPrice:
    """Tests for zero-coupon bond pricing."""

    @pytest.fixture
    def model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_bond_price_at_maturity(self, model):
        """Bond price at maturity should be 1."""
        P = model.zero_coupon_bond_price(r_t=0.04, t=5, T=5)
        assert P == 1.0

    def test_bond_price_less_than_one(self, model):
        """Bond price should be less than 1 for T > t."""
        P = model.zero_coupon_bond_price(r_t=0.04, t=0, T=5)
        assert 0 < P < 1

    def test_bond_price_decreases_with_rate(self, model):
        """Higher short rate should give lower bond price."""
        P_low = model.zero_coupon_bond_price(r_t=0.02, t=0, T=5)
        P_high = model.zero_coupon_bond_price(r_t=0.08, t=0, T=5)
        assert P_low > P_high

    def test_bond_price_decreases_with_maturity(self, model):
        """Longer maturity should generally give lower bond price."""
        P_short = model.zero_coupon_bond_price(r_t=0.04, t=0, T=1)
        P_long = model.zero_coupon_bond_price(r_t=0.04, t=0, T=10)
        assert P_short > P_long

    def test_negative_maturity_raises_error(self, model):
        with pytest.raises(ValueError, match="Maturity T must be >= current time t"):
            model.zero_coupon_bond_price(r_t=0.04, t=5, T=3)


class TestYieldCurve:
    """Tests for yield curve calculation."""

    @pytest.fixture
    def model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_yield_curve_shape(self, model):
        maturities = np.array([0.5, 1, 2, 5, 10])
        yields = model.yield_curve(r_t=0.04, t=0, maturities=maturities)
        assert yields.shape == (5,)

    def test_yield_curve_positive(self, model):
        """Yields should be positive for typical parameters."""
        maturities = np.array([0.5, 1, 2, 5, 10])
        yields = model.yield_curve(r_t=0.04, t=0, maturities=maturities)
        assert np.all(yields > 0)

    def test_yield_at_zero_maturity(self, model):
        """Yield at t=T should equal short rate."""
        maturities = np.array([0])
        yields = model.yield_curve(r_t=0.04, t=0, maturities=maturities)
        assert np.isclose(yields[0], 0.04)

    def test_yield_consistency_with_bond_price(self, model):
        """Yield should be consistent with bond price: Y = -ln(P)/tau."""
        r_t = 0.04
        T = 5
        maturities = np.array([T])
        yields = model.yield_curve(r_t=r_t, t=0, maturities=maturities)
        P = model.zero_coupon_bond_price(r_t=r_t, t=0, T=T)
        expected_yield = -np.log(P) / T
        assert np.isclose(yields[0], expected_yield)


class TestVasicekModelIntegration:
    """Integration tests combining multiple model features."""

    def test_forward_rate_integrates_to_yield(self):
        """Numerical integration of forward rate should approximate yield."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        r_t = 0.04
        T = 5

        # Numerical integration of forward rate
        n_points = 1000
        tenors = np.linspace(0.001, T, n_points)
        forward_rates = model.instantaneous_forward_rate(np.array([r_t]), 0, tenors)
        integral = np.trapezoid(forward_rates, tenors)
        yield_from_integral = integral / T

        # Analytical yield
        yields = model.yield_curve(r_t, 0, np.array([T]))

        assert np.isclose(yield_from_integral, yields[0], rtol=0.01)

    def test_simulation_bond_price_consistency(self):
        """Simulated paths should give consistent bond prices on average."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

        # Analytical bond price
        P_analytical = model.zero_coupon_bond_price(model.r0, 0, 1)

        # Monte Carlo bond price using simulated paths
        n_paths = 10000
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=n_paths, seed=42)
        dt = 1.0 / 100
        # Approximate integral of r(t) for each path
        integral_r = np.trapezoid(r, dx=dt, axis=1)
        P_mc = np.mean(np.exp(-integral_r))

        assert np.isclose(P_mc, P_analytical, rtol=0.02)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
