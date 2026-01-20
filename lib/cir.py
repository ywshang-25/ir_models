"""
Cox-Ingersoll-Ross (CIR) Interest Rate Model

The CIR model is a one-factor short-rate model where the instantaneous
interest rate follows:

    dr(t) = a(b - r(t))dt + σ√r(t)dW(t)

Where:
    - r(t): instantaneous short rate at time t
    - a: speed of mean reversion
    - b: long-term mean level
    - σ: volatility
    - W(t): standard Wiener process

The square root diffusion term ensures non-negative rates when 2ab >= σ²
(Feller condition).

This implementation uses Monte Carlo simulation for path generation.
"""

import numpy as np
from typing import Optional
from enum import Enum


class DiscretizationScheme(Enum):
    """Discretization schemes for CIR simulation."""
    EULER = "euler"
    MILSTEIN = "milstein"
    FULL_TRUNCATION = "full_truncation"
    REFLECTION = "reflection"


class CIRModel:
    """
    Cox-Ingersoll-Ross short-rate model implementation using Monte Carlo simulation.

    Parameters
    ----------
    a : float
        Speed of mean reversion (kappa)
    b : float
        Long-term mean level (theta)
    sigma : float
        Volatility of the short rate
    r0 : float
        Initial short rate (must be positive)
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        if a <= 0:
            raise ValueError("Mean reversion speed 'a' must be positive")
        if b <= 0:
            raise ValueError("Long-term mean 'b' must be positive")
        if sigma <= 0:
            raise ValueError("Volatility 'sigma' must be positive")
        if r0 <= 0:
            raise ValueError("Initial rate 'r0' must be positive")

        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    @property
    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition (2ab >= σ²) is satisfied."""
        return 2 * self.a * self.b >= self.sigma**2

    @property
    def feller_ratio(self) -> float:
        """Ratio 2ab/σ² - should be >= 1 for Feller condition."""
        return 2 * self.a * self.b / self.sigma**2

    @property
    def long_term_mean(self) -> float:
        """Long-term mean of the short rate."""
        return self.b

    @property
    def long_term_variance(self) -> float:
        """Long-term (stationary) variance of the short rate."""
        return self.b * self.sigma**2 / (2 * self.a)

    @property
    def long_term_std(self) -> float:
        """Long-term (stationary) standard deviation of the short rate."""
        return np.sqrt(self.long_term_variance)

    def simulate_short_rate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate short rate paths using Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed for reproducibility
        scheme : DiscretizationScheme
            Discretization scheme to use:
            - EULER: Basic Euler-Maruyama (may produce negative rates)
            - MILSTEIN: Milstein scheme with higher-order correction
            - FULL_TRUNCATION: Euler with max(r, 0) in diffusion (recommended)
            - REFLECTION: Reflect negative values to positive

        Returns
        -------
        t : np.ndarray
            Time grid of shape (n_steps + 1,)
        r : np.ndarray
            Simulated short rate paths of shape (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)
        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        # Generate all random numbers at once
        Z = np.random.standard_normal((n_paths, n_steps))

        if scheme == DiscretizationScheme.EULER:
            for i in range(n_steps):
                drift = self.a * (self.b - r[:, i]) * dt
                diffusion = self.sigma * np.sqrt(np.maximum(r[:, i], 0)) * sqrt_dt * Z[:, i]
                r[:, i + 1] = r[:, i] + drift + diffusion

        elif scheme == DiscretizationScheme.MILSTEIN:
            # Milstein scheme: adds correction term (σ²/4)(Z² - 1)dt
            for i in range(n_steps):
                r_pos = np.maximum(r[:, i], 0)
                sqrt_r = np.sqrt(r_pos)
                drift = self.a * (self.b - r[:, i]) * dt
                diffusion = self.sigma * sqrt_r * sqrt_dt * Z[:, i]
                milstein_correction = 0.25 * self.sigma**2 * (Z[:, i]**2 - 1) * dt
                r[:, i + 1] = r[:, i] + drift + diffusion + milstein_correction

        elif scheme == DiscretizationScheme.FULL_TRUNCATION:
            # Full truncation: use max(r, 0) in both drift and diffusion
            for i in range(n_steps):
                r_pos = np.maximum(r[:, i], 0)
                drift = self.a * (self.b - r_pos) * dt
                diffusion = self.sigma * np.sqrt(r_pos) * sqrt_dt * Z[:, i]
                r[:, i + 1] = r[:, i] + drift + diffusion

        elif scheme == DiscretizationScheme.REFLECTION:
            # Reflection: reflect negative values
            for i in range(n_steps):
                r_pos = np.maximum(r[:, i], 0)
                drift = self.a * (self.b - r[:, i]) * dt
                diffusion = self.sigma * np.sqrt(r_pos) * sqrt_dt * Z[:, i]
                r[:, i + 1] = np.abs(r[:, i] + drift + diffusion)

        return t, r

    def simulate_forward_rate_curve_mc(
        self,
        T_horizon: float,
        n_time_steps: int,
        maturities: np.ndarray,
        n_paths: int = 1000,
        n_inner_steps: int = 100,
        seed: Optional[int] = None,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate forward rate curves using nested Monte Carlo simulation.

        For each time point t and maturity T, estimates f(t, T) by computing
        the instantaneous forward rate from Monte Carlo bond prices.

        Parameters
        ----------
        T_horizon : float
            Simulation time horizon
        n_time_steps : int
            Number of outer time steps
        maturities : np.ndarray
            Array of tenors for the forward curve
        n_paths : int
            Number of outer simulation paths
        n_inner_steps : int
            Number of inner time steps for bond price estimation
        seed : int, optional
            Random seed
        scheme : DiscretizationScheme
            Discretization scheme

        Returns
        -------
        t : np.ndarray
            Time grid, shape (n_time_steps + 1,)
        r : np.ndarray
            Short rate paths, shape (n_paths, n_time_steps + 1)
        f : np.ndarray
            Forward rate curves (mean across paths), shape (n_time_steps + 1, n_maturities)
        """
        if seed is not None:
            np.random.seed(seed)

        # Simulate outer short rate paths
        t, r = self.simulate_short_rate(T_horizon, n_time_steps, n_paths, seed=None, scheme=scheme)

        n_maturities = len(maturities)
        f = np.zeros((n_time_steps + 1, n_maturities))

        # For each time point, estimate forward rates
        for i, t_i in enumerate(t):
            for j, tenor in enumerate(maturities):
                # Estimate f(t, t+tenor) using finite difference on bond prices
                # f(t, T) ≈ -∂ln(P)/∂T ≈ (ln(P(t,T-δ)) - ln(P(t,T+δ))) / (2δ)
                delta = 0.01  # Small shift for finite difference
                T_minus = tenor - delta
                T_plus = tenor + delta

                if T_minus <= 0:
                    T_minus = 0.001
                    T_plus = 2 * delta + 0.001

                # Monte Carlo bond prices using mean short rate at this time
                r_mean = np.mean(r[:, i])
                P_minus = self._mc_bond_price_from_rate(r_mean, T_minus, n_inner_steps, scheme)
                P_plus = self._mc_bond_price_from_rate(r_mean, T_plus, n_inner_steps, scheme)

                f[i, j] = (np.log(P_minus) - np.log(P_plus)) / (T_plus - T_minus)

        return t, r, f

    def _mc_bond_price_from_rate(
        self,
        r0: float,
        T: float,
        n_steps: int,
        scheme: DiscretizationScheme,
        n_mc_paths: int = 5000
    ) -> float:
        """
        Estimate zero-coupon bond price using Monte Carlo.

        P(0, T) = E[exp(-∫₀ᵀ r(s)ds)]

        Parameters
        ----------
        r0 : float
            Initial short rate
        T : float
            Bond maturity
        n_steps : int
            Number of time steps
        scheme : DiscretizationScheme
            Discretization scheme
        n_mc_paths : int
            Number of Monte Carlo paths

        Returns
        -------
        P : float
            Estimated bond price
        """
        # Create temporary model with given r0
        temp_model = CIRModel(self.a, self.b, self.sigma, max(r0, 1e-10))
        t, r = temp_model.simulate_short_rate(T, n_steps, n_mc_paths, scheme=scheme)

        # Numerical integration of rate paths
        dt = T / n_steps
        integral_r = np.trapezoid(r, dx=dt, axis=1)

        # Bond price as average of discounted values
        return np.mean(np.exp(-integral_r))

    def monte_carlo_bond_price(
        self,
        T: float,
        n_paths: int = 10000,
        n_steps: int = 100,
        seed: Optional[int] = None,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ) -> tuple[float, float]:
        """
        Estimate zero-coupon bond price using Monte Carlo simulation.

        P(0, T) = E[exp(-∫₀ᵀ r(s)ds)]

        Parameters
        ----------
        T : float
            Bond maturity
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps per path
        seed : int, optional
            Random seed
        scheme : DiscretizationScheme
            Discretization scheme

        Returns
        -------
        price : float
            Estimated bond price
        std_error : float
            Standard error of the estimate
        """
        t, r = self.simulate_short_rate(T, n_steps, n_paths, seed, scheme)

        # Numerical integration using trapezoidal rule
        dt = T / n_steps
        integral_r = np.trapezoid(r, dx=dt, axis=1)

        # Discounted values
        discount_factors = np.exp(-integral_r)

        price = np.mean(discount_factors)
        std_error = np.std(discount_factors) / np.sqrt(n_paths)

        return price, std_error

    def monte_carlo_yield_curve(
        self,
        maturities: np.ndarray,
        n_paths: int = 10000,
        n_steps_per_year: int = 100,
        seed: Optional[int] = None,
        scheme: DiscretizationScheme = DiscretizationScheme.FULL_TRUNCATION
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate yield curve using Monte Carlo simulation.

        Parameters
        ----------
        maturities : np.ndarray
            Array of bond maturities
        n_paths : int
            Number of Monte Carlo paths per maturity
        n_steps_per_year : int
            Number of time steps per year
        seed : int, optional
            Random seed
        scheme : DiscretizationScheme
            Discretization scheme

        Returns
        -------
        yields : np.ndarray
            Estimated yields
        std_errors : np.ndarray
            Standard errors of yield estimates
        """
        yields = np.zeros(len(maturities))
        std_errors = np.zeros(len(maturities))

        for i, T in enumerate(maturities):
            if seed is not None:
                np.random.seed(seed + i)

            n_steps = max(int(T * n_steps_per_year), 10)
            price, price_se = self.monte_carlo_bond_price(T, n_paths, n_steps, scheme=scheme)

            yields[i] = -np.log(price) / T
            # Delta method for yield standard error
            std_errors[i] = price_se / (price * T)

        return yields, std_errors

    def expected_rate(self, t: float) -> float:
        """
        Calculate expected short rate at time t.

        E[r(t)] = r₀e^{-at} + b(1 - e^{-at})

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        float
            Expected rate at time t
        """
        exp_factor = np.exp(-self.a * t)
        return self.r0 * exp_factor + self.b * (1 - exp_factor)

    def variance_rate(self, t: float) -> float:
        """
        Calculate variance of short rate at time t.

        Var[r(t)] = r₀(σ²/a)(e^{-at} - e^{-2at}) + (bσ²/2a)(1 - e^{-at})²

        Parameters
        ----------
        t : float
            Time

        Returns
        -------
        float
            Variance of rate at time t
        """
        exp_at = np.exp(-self.a * t)
        exp_2at = np.exp(-2 * self.a * t)
        term1 = self.r0 * (self.sigma**2 / self.a) * (exp_at - exp_2at)
        term2 = (self.b * self.sigma**2 / (2 * self.a)) * (1 - exp_at)**2
        return term1 + term2

    def __repr__(self) -> str:
        feller_status = "satisfied" if self.feller_condition_satisfied else "NOT satisfied"
        return (f"CIRModel(a={self.a}, b={self.b}, sigma={self.sigma}, r0={self.r0}, "
                f"Feller: {feller_status})")
