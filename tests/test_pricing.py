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
    price_derivative,
    bond_option_price,
    caplet_price,
    floorlet_price,
    cap_price,
    floor_price,
    swaption_price,
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
