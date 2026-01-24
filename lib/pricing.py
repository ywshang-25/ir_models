"""
Interest Rate Derivative Pricing Module

This module provides pricing functions for interest rate derivatives using
Monte Carlo simulation. The functions are model-agnostic and work with
simulated rate paths from any short-rate model (Vasicek, CIR, etc.).

Supported Derivatives:
- Generic derivatives with arbitrary payoff functions
- Bond options (calls/puts on zero-coupon bonds)
- Caplets and floorlets
- Caps and floors
- Swaptions
"""

import numpy as np
from typing import Callable, Optional


def price_derivative(
    t: np.ndarray,
    r: np.ndarray,
    payoff_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[float, float]:
    """
    Price an interest rate derivative with arbitrary payoff using Monte Carlo.

    The payoff function receives the time grid and rate paths, allowing for
    path-dependent payoffs. Discounting is done using the simulated short rate.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Simulated rate paths of shape (n_paths, n_steps + 1)
    payoff_func : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that takes (t, r) and returns payoff values of shape (n_paths,)

    Returns
    -------
    price : float
        Estimated derivative price
    std_error : float
        Standard error of the estimate

    Examples
    --------
    Price a derivative that pays max(r(T) - 0.05, 0):

    >>> t, r = model.simulate_short_rate(T=1.0, n_steps=100, n_paths=10000)
    >>> def payoff(t, r):
    ...     return np.maximum(r[:, -1] - 0.05, 0)
    >>> price, se = price_derivative(t, r, payoff)
    """
    n_paths = r.shape[0]
    T = t[-1]
    dt = T / (len(t) - 1)

    # Compute payoffs
    payoffs = payoff_func(t, r)

    # Discount factor: exp(-∫r(s)ds) using trapezoidal integration
    integral_r = np.trapezoid(r, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    # Discounted payoffs
    discounted_payoffs = payoffs * discount_factors

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    return price, std_error


def _compute_discount_factors(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute discount factors from rate paths.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)

    Returns
    -------
    df : np.ndarray
        Discount factors of shape (n_paths,)
    """
    T = t[-1]
    dt = T / (len(t) - 1)
    integral_r = np.trapezoid(r, dx=dt, axis=1)
    return np.exp(-integral_r)


def _compute_zcb_prices_at_time(
    t: np.ndarray,
    r: np.ndarray,
    t_idx: int,
    maturities: np.ndarray,
    n_inner_paths: int = 1000,
    model_params: Optional[dict] = None
) -> np.ndarray:
    """
    Estimate zero-coupon bond prices at a given time using nested Monte Carlo.

    This is a simplified estimation using the conditional expectation formula.
    For production use, consider using model-specific analytical formulas when available.

    Parameters
    ----------
    t : np.ndarray
        Time grid
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    t_idx : int
        Time index at which to compute bond prices
    maturities : np.ndarray
        Bond maturities (absolute times, not tenors)
    n_inner_paths : int
        Number of inner MC paths for bond price estimation
    model_params : dict, optional
        Model parameters for simulation (a, b, sigma, model_type)

    Returns
    -------
    prices : np.ndarray
        Bond prices of shape (n_paths, n_maturities)
    """
    n_paths = r.shape[0]
    n_maturities = len(maturities)
    prices = np.zeros((n_paths, n_maturities))
    t_current = t[t_idx]

    for j, T_mat in enumerate(maturities):
        tau = T_mat - t_current
        if tau <= 0:
            prices[:, j] = 1.0
        else:
            # Use approximate formula based on expected integrated rate
            # For more accuracy, use model-specific formulas
            prices[:, j] = np.exp(-r[:, t_idx] * tau)

    return prices


def bond_option_price(
    t: np.ndarray,
    r: np.ndarray,
    T_option: float,
    T_bond: float,
    K: float,
    option_type: str = "call",
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price a European option on a zero-coupon bond using Monte Carlo.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_option : float
        Option expiry time
    T_bond : float
        Bond maturity time (must be > T_option)
    K : float
        Strike price
    option_type : str
        "call" or "put"
    bond_price_func : Callable, optional
        Function to compute bond prices: bond_price_func(r_t, t, T) -> prices
        If None, uses exponential approximation P ≈ exp(-r * tau)

    Returns
    -------
    price : float
        Option price
    std_error : float
        Standard error of the estimate

    Notes
    -----
    The call option pays max(P(T_option, T_bond) - K, 0) at T_option.
    """
    if T_bond <= T_option:
        raise ValueError("Bond maturity must be greater than option expiry")
    if K <= 0:
        raise ValueError("Strike must be positive")
    if option_type.lower() not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    def payoff(t_grid: np.ndarray, r_paths: np.ndarray) -> np.ndarray:
        # Find time index closest to option expiry
        idx = np.argmin(np.abs(t_grid - T_option))
        r_T = r_paths[:, idx]
        tau = T_bond - T_option

        # Compute bond price at option expiry
        if bond_price_func is not None:
            P_bond = bond_price_func(r_T, T_option, T_bond)
        else:
            # Simple exponential approximation
            P_bond = np.exp(-r_T * tau)

        if option_type.lower() == "call":
            return np.maximum(P_bond - K, 0)
        else:
            return np.maximum(K - P_bond, 0)

    # Discount from T_option, not from T (end of simulation)
    n_paths = r.shape[0]
    T = t[-1]
    dt = T / (len(t) - 1)

    payoffs = payoff(t, r)

    # Discount to time 0 using integral from 0 to T_option
    idx_option = np.argmin(np.abs(t - T_option))
    r_to_option = r[:, :idx_option + 1]
    integral_r = np.trapezoid(r_to_option, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    discounted_payoffs = payoffs * discount_factors

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    return price, std_error


def caplet_price(
    t: np.ndarray,
    r: np.ndarray,
    T_start: float,
    T_end: float,
    K: float,
    notional: float = 1.0,
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price a caplet (call option on a forward rate).

    A caplet pays notional * tau * max(L(T_start, T_end) - K, 0) at T_end,
    where L is the simply-compounded forward rate and tau = T_end - T_start.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_start : float
        Start of the rate period (caplet fixing date)
    T_end : float
        End of the rate period (caplet payment date)
    K : float
        Cap rate (strike)
    notional : float
        Notional principal
    bond_price_func : Callable, optional
        Function to compute bond prices

    Returns
    -------
    price : float
        Caplet price
    std_error : float
        Standard error
    """
    if T_end <= T_start:
        raise ValueError("T_end must be greater than T_start")

    tau = T_end - T_start

    def payoff(t_grid: np.ndarray, r_paths: np.ndarray) -> np.ndarray:
        # Find time index at T_start
        idx_start = np.argmin(np.abs(t_grid - T_start))
        r_start = r_paths[:, idx_start]

        # Compute forward rate L(T_start; T_start, T_end)
        if bond_price_func is not None:
            P_start = bond_price_func(r_start, T_start, T_start)
            P_end = bond_price_func(r_start, T_start, T_end)
        else:
            P_start = np.ones_like(r_start)
            P_end = np.exp(-r_start * tau)

        L = (P_start / P_end - 1) / tau

        # Caplet payoff paid at T_end
        return notional * tau * np.maximum(L - K, 0)

    # Discount from T_end to time 0
    n_paths = r.shape[0]
    T = t[-1]
    dt = T / (len(t) - 1)

    payoffs = payoff(t, r)

    idx_end = np.argmin(np.abs(t - T_end))
    r_to_end = r[:, :idx_end + 1]
    integral_r = np.trapezoid(r_to_end, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    discounted_payoffs = payoffs * discount_factors

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    return price, std_error


def floorlet_price(
    t: np.ndarray,
    r: np.ndarray,
    T_start: float,
    T_end: float,
    K: float,
    notional: float = 1.0,
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price a floorlet (put option on a forward rate).

    A floorlet pays notional * tau * max(K - L(T_start, T_end), 0) at T_end.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_start : float
        Start of the rate period
    T_end : float
        End of the rate period
    K : float
        Floor rate (strike)
    notional : float
        Notional principal
    bond_price_func : Callable, optional
        Function to compute bond prices

    Returns
    -------
    price : float
        Floorlet price
    std_error : float
        Standard error
    """
    if T_end <= T_start:
        raise ValueError("T_end must be greater than T_start")

    tau = T_end - T_start

    def payoff(t_grid: np.ndarray, r_paths: np.ndarray) -> np.ndarray:
        idx_start = np.argmin(np.abs(t_grid - T_start))
        r_start = r_paths[:, idx_start]

        if bond_price_func is not None:
            P_start = bond_price_func(r_start, T_start, T_start)
            P_end = bond_price_func(r_start, T_start, T_end)
        else:
            P_start = np.ones_like(r_start)
            P_end = np.exp(-r_start * tau)

        L = (P_start / P_end - 1) / tau

        return notional * tau * np.maximum(K - L, 0)

    n_paths = r.shape[0]
    T = t[-1]
    dt = T / (len(t) - 1)

    payoffs = payoff(t, r)

    idx_end = np.argmin(np.abs(t - T_end))
    r_to_end = r[:, :idx_end + 1]
    integral_r = np.trapezoid(r_to_end, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    discounted_payoffs = payoffs * discount_factors

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    return price, std_error


def cap_price(
    t: np.ndarray,
    r: np.ndarray,
    T_start: float,
    T_end: float,
    K: float,
    frequency: float = 0.25,
    notional: float = 1.0,
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price an interest rate cap (portfolio of caplets).

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_start : float
        Start of the cap period
    T_end : float
        End of the cap period
    K : float
        Cap rate (strike)
    frequency : float
        Payment frequency in years (e.g., 0.25 for quarterly)
    notional : float
        Notional principal
    bond_price_func : Callable, optional
        Function to compute bond prices

    Returns
    -------
    price : float
        Cap price
    std_error : float
        Combined standard error (sum of variances)
    """
    if T_end <= T_start:
        raise ValueError("T_end must be greater than T_start")

    # Generate caplet periods
    period_starts = np.arange(T_start, T_end - 1e-10, frequency)

    total_price = 0.0
    total_variance = 0.0
    n_paths = r.shape[0]

    for t_s in period_starts:
        t_e = min(t_s + frequency, T_end)
        if t_e > t_s:
            price_i, se_i = caplet_price(t, r, t_s, t_e, K, notional, bond_price_func)
            total_price += price_i
            total_variance += (se_i * np.sqrt(n_paths)) ** 2

    combined_se = np.sqrt(total_variance) / np.sqrt(n_paths)

    return total_price, combined_se


def floor_price(
    t: np.ndarray,
    r: np.ndarray,
    T_start: float,
    T_end: float,
    K: float,
    frequency: float = 0.25,
    notional: float = 1.0,
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price an interest rate floor (portfolio of floorlets).

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_start : float
        Start of the floor period
    T_end : float
        End of the floor period
    K : float
        Floor rate (strike)
    frequency : float
        Payment frequency in years
    notional : float
        Notional principal
    bond_price_func : Callable, optional
        Function to compute bond prices

    Returns
    -------
    price : float
        Floor price
    std_error : float
        Combined standard error
    """
    if T_end <= T_start:
        raise ValueError("T_end must be greater than T_start")

    period_starts = np.arange(T_start, T_end - 1e-10, frequency)

    total_price = 0.0
    total_variance = 0.0
    n_paths = r.shape[0]

    for t_s in period_starts:
        t_e = min(t_s + frequency, T_end)
        if t_e > t_s:
            price_i, se_i = floorlet_price(t, r, t_s, t_e, K, notional, bond_price_func)
            total_price += price_i
            total_variance += (se_i * np.sqrt(n_paths)) ** 2

    combined_se = np.sqrt(total_variance) / np.sqrt(n_paths)

    return total_price, combined_se


def swaption_price(
    t: np.ndarray,
    r: np.ndarray,
    T_option: float,
    T_swap_end: float,
    K: float,
    frequency: float = 0.25,
    notional: float = 1.0,
    payer: bool = True,
    bond_price_func: Optional[Callable[[np.ndarray, float, float], np.ndarray]] = None
) -> tuple[float, float]:
    """
    Price a European swaption using Monte Carlo simulation.

    A payer swaption gives the right to enter a payer swap (pay fixed, receive floating).
    A receiver swaption gives the right to enter a receiver swap.

    Parameters
    ----------
    t : np.ndarray
        Time grid of shape (n_steps + 1,)
    r : np.ndarray
        Rate paths of shape (n_paths, n_steps + 1)
    T_option : float
        Option expiry (swap start date)
    T_swap_end : float
        End date of the underlying swap
    K : float
        Fixed swap rate (strike)
    frequency : float
        Payment frequency in years
    notional : float
        Notional principal
    payer : bool
        True for payer swaption, False for receiver swaption
    bond_price_func : Callable, optional
        Function to compute bond prices: bond_price_func(r_t, t, T) -> prices

    Returns
    -------
    price : float
        Swaption price
    std_error : float
        Standard error of the estimate
    """
    if T_swap_end <= T_option:
        raise ValueError("Swap end must be after option expiry")

    def payoff(t_grid: np.ndarray, r_paths: np.ndarray) -> np.ndarray:
        # Find time index at option expiry
        idx = np.argmin(np.abs(t_grid - T_option))
        r_T = r_paths[:, idx]
        n_paths = r_paths.shape[0]

        # Payment dates for the swap
        payment_dates = np.arange(T_option + frequency, T_swap_end + 1e-10, frequency)

        # Compute swap value at T_option for each path
        swap_values = np.zeros(n_paths)

        for T_i in payment_dates:
            tau = frequency
            T_prev = T_i - frequency

            # Bond prices at T_option
            if bond_price_func is not None:
                P_i = bond_price_func(r_T, T_option, T_i)
                P_prev = bond_price_func(r_T, T_option, max(T_prev, T_option))
            else:
                P_i = np.exp(-r_T * (T_i - T_option))
                P_prev = np.exp(-r_T * max(T_prev - T_option, 0))

            # Forward rate L(T_option; T_prev, T_i)
            if T_prev >= T_option:
                L = (P_prev / P_i - 1) / tau
            else:
                # First period
                L = (1 / P_i - 1) / (T_i - T_option)

            # Swap cashflow: floating leg - fixed leg
            cashflow = notional * tau * (L - K)
            swap_values += cashflow * P_i

        if payer:
            return np.maximum(swap_values, 0)
        else:
            return np.maximum(-swap_values, 0)

    # Discount from T_option
    n_paths = r.shape[0]
    T = t[-1]
    dt = T / (len(t) - 1)

    payoffs = payoff(t, r)

    idx_option = np.argmin(np.abs(t - T_option))
    r_to_option = r[:, :idx_option + 1]
    integral_r = np.trapezoid(r_to_option, dx=dt, axis=1)
    discount_factors = np.exp(-integral_r)

    discounted_payoffs = payoffs * discount_factors

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)

    return price, std_error


# Convenience functions for creating bond price functions from models

def make_vasicek_bond_price_func(a: float, b: float, sigma: float):
    """
    Create a bond price function for the Vasicek model.

    Parameters
    ----------
    a : float
        Speed of mean reversion
    b : float
        Long-term mean level
    sigma : float
        Volatility

    Returns
    -------
    bond_price_func : Callable
        Function that computes P(t, T) given r(t)
    """
    def bond_price_func(r_t: np.ndarray, t: float, T: float) -> np.ndarray:
        tau = T - t
        if tau <= 0:
            return np.ones_like(r_t)

        B = (1 - np.exp(-a * tau)) / a
        A = np.exp(
            (B - tau) * (a**2 * b - sigma**2 / 2) / a**2
            - sigma**2 * B**2 / (4 * a)
        )
        return A * np.exp(-B * r_t)

    return bond_price_func


def make_cir_bond_price_func(a: float, b: float, sigma: float):
    """
    Create a bond price function for the CIR model.

    Parameters
    ----------
    a : float
        Speed of mean reversion
    b : float
        Long-term mean level
    sigma : float
        Volatility

    Returns
    -------
    bond_price_func : Callable
        Function that computes P(t, T) given r(t)
    """
    def bond_price_func(r_t: np.ndarray, t: float, T: float) -> np.ndarray:
        tau = T - t
        if tau <= 0:
            return np.ones_like(r_t)

        gamma = np.sqrt(a**2 + 2 * sigma**2)

        # CIR analytical bond price formula
        numerator = 2 * gamma * np.exp((a + gamma) * tau / 2)
        denominator = (gamma + a) * (np.exp(gamma * tau) - 1) + 2 * gamma

        A = (numerator / denominator) ** (2 * a * b / sigma**2)
        B = 2 * (np.exp(gamma * tau) - 1) / denominator

        return A * np.exp(-B * r_t)

    return bond_price_func
