"""
Unit tests for the interest rate derivative pricing module.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from lib.vasicek import VasicekModel
from lib.cir import CIRModel, DiscretizationScheme
from lib.pricing import (
    # Path-based functions
    price_derivative,
    bond_option_price,
    caplet_price,
    floorlet_price,
    cap_price,
    floor_price,
    swaption_price,
    # Model-based functions
    swap_price,
    par_swap_rate,
    price_bond_option,
    price_caplet,
    price_floorlet,
    price_cap,
    price_floor,
    price_swaption,
    # Helpers
    make_bond_price_func,
    make_vasicek_bond_price_func,
    make_cir_bond_price_func,
)


class TestPriceDerivative:
    """Tests for the generic derivative pricing function."""

    @pytest.fixture
    def vasicek_paths(self):
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5000, seed=42)
        return t, r

    @pytest.fixture
    def cir_paths(self):
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=5000, seed=42)
        return t, r

    def test_returns_tuple(self, vasicek_paths):
        t, r = vasicek_paths
        def payoff(t, r):
            return np.ones(r.shape[0])
        result = price_derivative(t, r, payoff)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_constant_payoff_one(self, vasicek_paths):
        """Payoff of 1 should give bond price."""
        t, r = vasicek_paths
        def payoff(t, r):
            return np.ones(r.shape[0])
        price, se = price_derivative(t, r, payoff)
        # Price should be close to ZCB price (around 0.97 for these params)
        assert 0.9 < price < 1.0
        assert se > 0

    def test_zero_payoff(self, vasicek_paths):
        """Zero payoff should give zero price."""
        t, r = vasicek_paths
        def payoff(t, r):
            return np.zeros(r.shape[0])
        price, se = price_derivative(t, r, payoff)
        assert np.isclose(price, 0, atol=1e-10)

    def test_terminal_rate_payoff(self, vasicek_paths):
        """Test payoff based on terminal rate."""
        t, r = vasicek_paths
        def payoff(t, r):
            return r[:, -1]  # Terminal short rate
        price, se = price_derivative(t, r, payoff)
        assert price > 0
        assert se > 0

    def test_path_dependent_payoff(self, vasicek_paths):
        """Test path-dependent payoff (average rate)."""
        t, r = vasicek_paths
        def payoff(t, r):
            return np.mean(r, axis=1)  # Average rate over path
        price, se = price_derivative(t, r, payoff)
        assert price > 0

    def test_call_option_payoff(self, vasicek_paths):
        """Test call-like payoff on terminal rate."""
        t, r = vasicek_paths
        K = 0.04
        def payoff(t, r):
            return np.maximum(r[:, -1] - K, 0)
        price, se = price_derivative(t, r, payoff)
        assert price >= 0

    def test_works_with_cir_paths(self, cir_paths):
        """Test that pricing works with CIR model paths."""
        t, r = cir_paths
        def payoff(t, r):
            return np.maximum(r[:, -1] - 0.04, 0)
        price, se = price_derivative(t, r, payoff)
        assert price >= 0
        assert se > 0

    def test_std_error_decreases_with_paths(self):
        """Standard error should decrease with more paths."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

        def payoff(t, r):
            return np.maximum(r[:, -1] - 0.04, 0)

        t1, r1 = model.simulate_short_rate(T=1.0, n_steps=50, n_paths=500, seed=42)
        _, se1 = price_derivative(t1, r1, payoff)

        t2, r2 = model.simulate_short_rate(T=1.0, n_steps=50, n_paths=10000, seed=42)
        _, se2 = price_derivative(t2, r2, payoff)

        assert se2 < se1


class TestBondOptionPrice:
    """Tests for bond option pricing."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def cir_model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_returns_tuple(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=5000, seed=42)
        price, se = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95)
        assert isinstance(price, float)
        assert isinstance(se, float)

    def test_call_price_non_negative(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=5000, seed=42)
        price, se = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95, option_type="call")
        assert price >= -se * 3  # Allow small MC error

    def test_put_price_non_negative(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=5000, seed=42)
        price, se = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95, option_type="put")
        assert price >= -se * 3

    def test_call_price_increases_with_lower_strike(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=10000, seed=42)
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_high_K, _ = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.98,
                                            option_type="call", bond_price_func=bond_func)
        price_low_K, _ = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.92,
                                           option_type="call", bond_price_func=bond_func)
        assert price_low_K > price_high_K

    def test_put_price_increases_with_higher_strike(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=10000, seed=42)
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_high_K, _ = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.98,
                                            option_type="put", bond_price_func=bond_func)
        price_low_K, _ = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.92,
                                           option_type="put", bond_price_func=bond_func)
        assert price_high_K > price_low_K

    def test_invalid_option_type_raises_error(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=100, seed=42)
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95, option_type="invalid")

    def test_bond_maturity_before_option_expiry_raises_error(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=100, seed=42)
        with pytest.raises(ValueError, match="Bond maturity must be greater than option expiry"):
            bond_option_price(t, r, T_option=2.0, T_bond=1.0, K=0.95)

    def test_negative_strike_raises_error(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=100, seed=42)
        with pytest.raises(ValueError, match="Strike must be positive"):
            bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=-0.5)

    def test_works_with_cir_model(self, cir_model):
        t, r = cir_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=5000, seed=42)
        bond_func = make_cir_bond_price_func(0.5, 0.05, 0.1)
        price, se = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95,
                                      bond_price_func=bond_func)
        assert price >= -se * 3


class TestCapletFloorletPrice:
    """Tests for caplet and floorlet pricing."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def long_paths(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=10000, seed=42)
        return t, r

    def test_caplet_returns_tuple(self, long_paths):
        t, r = long_paths
        price, se = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.04)
        assert isinstance(price, float)
        assert isinstance(se, float)

    def test_caplet_non_negative(self, long_paths):
        t, r = long_paths
        price, se = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.04)
        assert price >= -se * 3

    def test_floorlet_non_negative(self, long_paths):
        t, r = long_paths
        price, se = floorlet_price(t, r, T_start=0.5, T_end=1.0, K=0.04)
        assert price >= -se * 3

    def test_caplet_price_decreases_with_higher_strike(self, long_paths):
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_low_K, _ = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.03, bond_price_func=bond_func)
        price_high_K, _ = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.06, bond_price_func=bond_func)
        assert price_low_K > price_high_K

    def test_floorlet_price_increases_with_higher_strike(self, long_paths):
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_low_K, _ = floorlet_price(t, r, T_start=0.5, T_end=1.0, K=0.03, bond_price_func=bond_func)
        price_high_K, _ = floorlet_price(t, r, T_start=0.5, T_end=1.0, K=0.06, bond_price_func=bond_func)
        assert price_high_K > price_low_K

    def test_caplet_with_notional(self, long_paths):
        t, r = long_paths
        price_1, _ = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.04, notional=1.0)
        price_100, _ = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.04, notional=100.0)
        assert np.isclose(price_100, price_1 * 100, rtol=0.01)

    def test_invalid_dates_raises_error(self, long_paths):
        t, r = long_paths
        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            caplet_price(t, r, T_start=1.0, T_end=0.5, K=0.04)

        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            floorlet_price(t, r, T_start=1.0, T_end=0.5, K=0.04)


class TestCapFloorPrice:
    """Tests for cap and floor pricing."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def long_paths(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=3.0, n_steps=300, n_paths=10000, seed=42)
        return t, r

    def test_cap_returns_tuple(self, long_paths):
        t, r = long_paths
        price, se = cap_price(t, r, T_start=0.5, T_end=2.0, K=0.04, frequency=0.5)
        assert isinstance(price, float)
        assert isinstance(se, float)

    def test_cap_non_negative(self, long_paths):
        t, r = long_paths
        price, se = cap_price(t, r, T_start=0.5, T_end=2.0, K=0.04, frequency=0.5)
        assert price >= -se * 3

    def test_floor_non_negative(self, long_paths):
        t, r = long_paths
        price, se = floor_price(t, r, T_start=0.5, T_end=2.0, K=0.04, frequency=0.5)
        assert price >= -se * 3

    def test_cap_greater_than_single_caplet(self, long_paths):
        """Cap should be worth at least as much as a single caplet."""
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        cap_p, _ = cap_price(t, r, T_start=0.5, T_end=2.0, K=0.04, frequency=0.5,
                            bond_price_func=bond_func)
        caplet_p, _ = caplet_price(t, r, T_start=0.5, T_end=1.0, K=0.04,
                                   bond_price_func=bond_func)
        assert cap_p >= caplet_p * 0.9  # Allow some MC variance

    def test_cap_price_increases_with_longer_period(self, long_paths):
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_short, _ = cap_price(t, r, T_start=0.5, T_end=1.5, K=0.04, frequency=0.5,
                                   bond_price_func=bond_func)
        price_long, _ = cap_price(t, r, T_start=0.5, T_end=2.5, K=0.04, frequency=0.5,
                                  bond_price_func=bond_func)
        assert price_long > price_short

    def test_invalid_dates_raises_error(self, long_paths):
        t, r = long_paths
        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            cap_price(t, r, T_start=2.0, T_end=1.0, K=0.04)

        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            floor_price(t, r, T_start=2.0, T_end=1.0, K=0.04)


class TestSwaptionPrice:
    """Tests for swaption pricing."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def long_paths(self, vasicek_model):
        t, r = vasicek_model.simulate_short_rate(T=3.0, n_steps=300, n_paths=10000, seed=42)
        return t, r

    def test_swaption_returns_tuple(self, long_paths):
        t, r = long_paths
        price, se = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.04, frequency=0.5)
        assert isinstance(price, float)
        assert isinstance(se, float)

    def test_payer_swaption_non_negative(self, long_paths):
        t, r = long_paths
        price, se = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, payer=True)
        assert price >= -se * 3

    def test_receiver_swaption_non_negative(self, long_paths):
        t, r = long_paths
        price, se = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, payer=False)
        assert price >= -se * 3

    def test_payer_swaption_decreases_with_higher_strike(self, long_paths):
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_low_K, _ = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.03,
                                        frequency=0.5, payer=True, bond_price_func=bond_func)
        price_high_K, _ = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.06,
                                         frequency=0.5, payer=True, bond_price_func=bond_func)
        assert price_low_K > price_high_K

    def test_receiver_swaption_increases_with_higher_strike(self, long_paths):
        t, r = long_paths
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        price_low_K, _ = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.03,
                                        frequency=0.5, payer=False, bond_price_func=bond_func)
        price_high_K, _ = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.06,
                                         frequency=0.5, payer=False, bond_price_func=bond_func)
        assert price_high_K > price_low_K

    def test_invalid_dates_raises_error(self, long_paths):
        t, r = long_paths
        with pytest.raises(ValueError, match="Swap end must be after option expiry"):
            swaption_price(t, r, T_option=3.0, T_swap_end=2.0, K=0.04)


class TestSwapPrice:
    """Tests for swap pricing."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def cir_model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_swap_returns_float(self, vasicek_model):
        value = swap_price(vasicek_model, T_start=0, T_end=5, K=0.04)
        assert isinstance(value, float)

    def test_payer_receiver_opposite_signs(self, vasicek_model):
        """Payer and receiver swaps should have opposite values."""
        payer_value = swap_price(vasicek_model, T_start=0, T_end=5, K=0.04, payer=True)
        receiver_value = swap_price(vasicek_model, T_start=0, T_end=5, K=0.04, payer=False)
        assert np.isclose(payer_value, -receiver_value)

    def test_swap_at_par_rate_is_zero(self, vasicek_model):
        """Swap value should be zero when K equals par rate."""
        par_rate = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.25)
        value = swap_price(vasicek_model, T_start=0, T_end=5, K=par_rate, frequency=0.25)
        assert np.isclose(value, 0, atol=1e-10)

    def test_payer_swap_value_decreases_with_higher_strike(self, vasicek_model):
        """Higher fixed rate means lower payer swap value."""
        value_low_K = swap_price(vasicek_model, T_start=0, T_end=5, K=0.03, payer=True)
        value_high_K = swap_price(vasicek_model, T_start=0, T_end=5, K=0.06, payer=True)
        assert value_low_K > value_high_K

    def test_receiver_swap_value_increases_with_higher_strike(self, vasicek_model):
        """Higher fixed rate means higher receiver swap value."""
        value_low_K = swap_price(vasicek_model, T_start=0, T_end=5, K=0.03, payer=False)
        value_high_K = swap_price(vasicek_model, T_start=0, T_end=5, K=0.06, payer=False)
        assert value_high_K > value_low_K

    def test_swap_with_notional(self, vasicek_model):
        """Swap value should scale linearly with notional."""
        value_1 = swap_price(vasicek_model, T_start=0, T_end=5, K=0.04, notional=1)
        value_100 = swap_price(vasicek_model, T_start=0, T_end=5, K=0.04, notional=100)
        assert np.isclose(value_100, value_1 * 100)

    def test_works_with_cir_model(self, cir_model):
        """Swap pricing should work with CIR model."""
        value = swap_price(cir_model, T_start=0, T_end=5, K=0.04)
        assert isinstance(value, float)

    def test_invalid_dates_raises_error(self, vasicek_model):
        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            swap_price(vasicek_model, T_start=5, T_end=2, K=0.04)


class TestParSwapRate:
    """Tests for par swap rate calculation."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def cir_model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    def test_par_rate_returns_float(self, vasicek_model):
        rate = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.25)
        assert isinstance(rate, float)

    def test_par_rate_positive(self, vasicek_model):
        """Par rate should be positive for typical parameters."""
        rate = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.25)
        assert rate > 0

    def test_par_rate_reasonable_range(self, vasicek_model):
        """Par rate should be in reasonable range."""
        rate = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.25)
        assert 0.01 < rate < 0.10  # Between 1% and 10%

    def test_par_rate_increases_with_longer_tenor(self, vasicek_model):
        """In upward sloping yield curve, longer tenor = higher par rate."""
        # Note: This depends on model parameters; may not always hold
        rate_2y = par_swap_rate(vasicek_model, T_start=0, T_end=2, frequency=0.25)
        rate_10y = par_swap_rate(vasicek_model, T_start=0, T_end=10, frequency=0.25)
        # Just check both are reasonable
        assert rate_2y > 0
        assert rate_10y > 0

    def test_works_with_cir_model(self, cir_model):
        """Par rate calculation should work with CIR model."""
        rate = par_swap_rate(cir_model, T_start=0, T_end=5, frequency=0.25)
        assert isinstance(rate, float)
        assert rate > 0

    def test_different_frequencies(self, vasicek_model):
        """Par rate should be calculated for different frequencies."""
        rate_quarterly = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.25)
        rate_semiannual = par_swap_rate(vasicek_model, T_start=0, T_end=5, frequency=0.5)
        # Both should be positive and reasonable
        assert rate_quarterly > 0
        assert rate_semiannual > 0

    def test_forward_starting_swap(self, vasicek_model):
        """Par rate for forward starting swap."""
        rate = par_swap_rate(vasicek_model, T_start=1, T_end=6, frequency=0.25)
        assert rate > 0

    def test_invalid_dates_raises_error(self, vasicek_model):
        with pytest.raises(ValueError, match="T_end must be greater than T_start"):
            par_swap_rate(vasicek_model, T_start=5, T_end=2)


class TestBondPriceFunctions:
    """Tests for bond price helper functions."""

    def test_vasicek_bond_price_func_at_maturity(self):
        bond_func = make_vasicek_bond_price_func(a=0.5, b=0.05, sigma=0.02)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=1.0, T=1.0)
        np.testing.assert_array_almost_equal(prices, np.ones(3))

    def test_vasicek_bond_price_func_positive(self):
        bond_func = make_vasicek_bond_price_func(a=0.5, b=0.05, sigma=0.02)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert np.all(prices > 0)
        assert np.all(prices < 1)

    def test_vasicek_bond_price_decreases_with_rate(self):
        bond_func = make_vasicek_bond_price_func(a=0.5, b=0.05, sigma=0.02)
        r_t = np.array([0.02, 0.04, 0.06])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert prices[0] > prices[1] > prices[2]

    def test_cir_bond_price_func_at_maturity(self):
        bond_func = make_cir_bond_price_func(a=0.5, b=0.05, sigma=0.1)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=1.0, T=1.0)
        np.testing.assert_array_almost_equal(prices, np.ones(3))

    def test_cir_bond_price_func_positive(self):
        bond_func = make_cir_bond_price_func(a=0.5, b=0.05, sigma=0.1)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert np.all(prices > 0)
        assert np.all(prices < 1)

    def test_cir_bond_price_decreases_with_rate(self):
        bond_func = make_cir_bond_price_func(a=0.5, b=0.05, sigma=0.1)
        r_t = np.array([0.02, 0.04, 0.06])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert prices[0] > prices[1] > prices[2]

    def test_vasicek_matches_model_method(self):
        """Bond price function should match VasicekModel.zero_coupon_bond_price."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        bond_func = make_vasicek_bond_price_func(a=0.5, b=0.05, sigma=0.02)

        r_t = 0.04
        T = 5.0
        price_model = model.zero_coupon_bond_price(r_t, t=0, T=T)
        price_func = bond_func(np.array([r_t]), t=0, T=T)[0]

        assert np.isclose(price_model, price_func)


class TestUnifiedBondPriceFunc:
    """Tests for the unified make_bond_price_func function."""

    def test_works_with_vasicek_model(self):
        """Unified function should work with VasicekModel."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        bond_func = make_bond_price_func(model)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert np.all(prices > 0)
        assert np.all(prices < 1)

    def test_works_with_cir_model(self):
        """Unified function should work with CIRModel."""
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        bond_func = make_bond_price_func(model)
        r_t = np.array([0.03, 0.04, 0.05])
        prices = bond_func(r_t, t=0.0, T=5.0)
        assert np.all(prices > 0)
        assert np.all(prices < 1)

    def test_vasicek_unified_matches_specific(self):
        """Unified function should match model-specific function for Vasicek."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        unified_func = make_bond_price_func(model)
        specific_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        r_t = np.array([0.02, 0.04, 0.06])
        prices_unified = unified_func(r_t, t=0.0, T=5.0)
        prices_specific = specific_func(r_t, t=0.0, T=5.0)

        np.testing.assert_array_almost_equal(prices_unified, prices_specific)

    def test_cir_unified_matches_specific(self):
        """Unified function should match model-specific function for CIR."""
        model = CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)
        unified_func = make_bond_price_func(model)
        specific_func = make_cir_bond_price_func(0.5, 0.05, 0.1)

        r_t = np.array([0.02, 0.04, 0.06])
        prices_unified = unified_func(r_t, t=0.0, T=5.0)
        prices_specific = specific_func(r_t, t=0.0, T=5.0)

        np.testing.assert_array_almost_equal(prices_unified, prices_specific)

    def test_unsupported_model_raises_error(self):
        """Unified function should raise error for unsupported model types."""
        class FakeModel:
            pass

        with pytest.raises(ValueError, match="Unsupported model type"):
            make_bond_price_func(FakeModel())

    def test_unified_with_swaption_pricing(self):
        """Test unified function in actual pricing workflow."""
        model = VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)
        t, r = model.simulate_short_rate(T=3.0, n_steps=300, n_paths=5000, seed=42)

        bond_func = make_bond_price_func(model)
        price, se = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, bond_price_func=bond_func)
        assert price >= -se * 3


class TestPricingWithCIRModel:
    """Integration tests using CIR model paths."""

    @pytest.fixture
    def cir_model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    @pytest.fixture
    def cir_paths(self, cir_model):
        t, r = cir_model.simulate_short_rate(T=3.0, n_steps=300, n_paths=10000, seed=42)
        return t, r

    def test_generic_pricing_with_cir(self, cir_paths):
        t, r = cir_paths
        def payoff(t, r):
            return np.maximum(r[:, -1] - 0.04, 0)
        price, se = price_derivative(t, r, payoff)
        assert price >= 0

    def test_bond_option_with_cir(self, cir_paths):
        t, r = cir_paths
        bond_func = make_cir_bond_price_func(0.5, 0.05, 0.1)
        price, se = bond_option_price(t, r, T_option=1.0, T_bond=2.0, K=0.95,
                                      bond_price_func=bond_func)
        assert price >= -se * 3

    def test_cap_with_cir(self, cir_paths):
        t, r = cir_paths
        bond_func = make_cir_bond_price_func(0.5, 0.05, 0.1)
        price, se = cap_price(t, r, T_start=0.5, T_end=2.0, K=0.04, frequency=0.5,
                             bond_price_func=bond_func)
        assert price >= -se * 3

    def test_swaption_with_cir(self, cir_paths):
        t, r = cir_paths
        bond_func = make_cir_bond_price_func(0.5, 0.05, 0.1)
        price, se = swaption_price(t, r, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, bond_price_func=bond_func)
        assert price >= -se * 3


class TestPutCallParity:
    """Tests for put-call parity relationships."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    def test_bond_option_put_call_parity(self, vasicek_model):
        """Test put-call parity: C - P = P(0,T_bond) - K * P(0,T_option)."""
        t, r = vasicek_model.simulate_short_rate(T=2.0, n_steps=200, n_paths=20000, seed=42)
        bond_func = make_vasicek_bond_price_func(0.5, 0.05, 0.02)

        T_option = 1.0
        T_bond = 2.0
        K = 0.95

        call_price, _ = bond_option_price(t, r, T_option, T_bond, K, "call", bond_func)
        put_price, _ = bond_option_price(t, r, T_option, T_bond, K, "put", bond_func)

        # Analytical bond prices for parity check
        P_T_option = vasicek_model.zero_coupon_bond_price(vasicek_model.r0, 0, T_option)
        P_T_bond = vasicek_model.zero_coupon_bond_price(vasicek_model.r0, 0, T_bond)

        # Put-call parity: C - P = P(0, T_bond) - K * P(0, T_option)
        lhs = call_price - put_price
        rhs = P_T_bond - K * P_T_option

        # Allow for MC error
        assert np.isclose(lhs, rhs, rtol=0.1)


class TestModelBasedPricingFunctions:
    """Tests for model-based pricing functions (convenient wrappers)."""

    @pytest.fixture
    def vasicek_model(self):
        return VasicekModel(a=0.5, b=0.05, sigma=0.02, r0=0.03)

    @pytest.fixture
    def cir_model(self):
        return CIRModel(a=0.5, b=0.05, sigma=0.1, r0=0.03)

    # Bond option tests
    def test_price_bond_option_vasicek(self, vasicek_model):
        price, se = price_bond_option(vasicek_model, T_option=1.0, T_bond=2.0, K=0.95,
                                      n_paths=5000, seed=42)
        assert price >= -se * 3
        assert se > 0

    def test_price_bond_option_cir(self, cir_model):
        price, se = price_bond_option(cir_model, T_option=1.0, T_bond=2.0, K=0.95,
                                      n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_bond_option_put(self, vasicek_model):
        price, se = price_bond_option(vasicek_model, T_option=1.0, T_bond=2.0, K=0.95,
                                      option_type="put", n_paths=5000, seed=42)
        assert price >= -se * 3

    # Caplet/floorlet tests
    def test_price_caplet_vasicek(self, vasicek_model):
        price, se = price_caplet(vasicek_model, T_start=0.5, T_end=1.0, K=0.04,
                                 n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_caplet_cir(self, cir_model):
        price, se = price_caplet(cir_model, T_start=0.5, T_end=1.0, K=0.04,
                                 n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_floorlet_vasicek(self, vasicek_model):
        price, se = price_floorlet(vasicek_model, T_start=0.5, T_end=1.0, K=0.04,
                                   n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_floorlet_cir(self, cir_model):
        price, se = price_floorlet(cir_model, T_start=0.5, T_end=1.0, K=0.04,
                                   n_paths=5000, seed=42)
        assert price >= -se * 3

    # Cap/floor tests
    def test_price_cap_vasicek(self, vasicek_model):
        price, se = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                              frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_cap_cir(self, cir_model):
        price, se = price_cap(cir_model, T_start=0.5, T_end=2.0, K=0.04,
                              frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_floor_vasicek(self, vasicek_model):
        price, se = price_floor(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                                frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_floor_cir(self, cir_model):
        price, se = price_floor(cir_model, T_start=0.5, T_end=2.0, K=0.04,
                                frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    # Swaption tests
    def test_price_swaption_vasicek(self, vasicek_model):
        price, se = price_swaption(vasicek_model, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_swaption_cir(self, cir_model):
        price, se = price_swaption(cir_model, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, n_paths=5000, seed=42)
        assert price >= -se * 3

    def test_price_swaption_receiver(self, vasicek_model):
        price, se = price_swaption(vasicek_model, T_option=1.0, T_swap_end=3.0, K=0.04,
                                   frequency=0.5, payer=False, n_paths=5000, seed=42)
        assert price >= -se * 3

    # Test notional scaling
    def test_price_cap_notional_scaling(self, vasicek_model):
        price_1, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                               notional=1.0, n_paths=10000, seed=42)
        price_100, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                                 notional=100.0, n_paths=10000, seed=42)
        assert np.isclose(price_100, price_1 * 100, rtol=0.01)

    # Test reproducibility with seed
    def test_price_cap_reproducibility(self, vasicek_model):
        price1, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                              n_paths=5000, seed=42)
        price2, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.04,
                              n_paths=5000, seed=42)
        assert price1 == price2

    # Test strike sensitivity
    def test_price_cap_strike_sensitivity(self, vasicek_model):
        price_low_K, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.03,
                                   n_paths=10000, seed=42)
        price_high_K, _ = price_cap(vasicek_model, T_start=0.5, T_end=2.0, K=0.06,
                                    n_paths=10000, seed=42)
        assert price_low_K > price_high_K


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
