"""
Vasicek Interest Rate Model

The Vasicek model is a one-factor short-rate model where the instantaneous
interest rate follows an Ornstein-Uhlenbeck process:

    dr(t) = a(b - r(t))dt + σdW(t)

Where:
    - r(t): instantaneous short rate at time t
    - a: speed of mean reversion
    - b: long-term mean level
    - σ: volatility
    - W(t): standard Wiener process
"""

import numpy as np
from typing import Optional


class VasicekModel:
    """
    Vasicek short-rate model implementation.

    Parameters
    ----------
    a : float
        Speed of mean reversion (kappa)
    b : float
        Long-term mean level (theta)
    sigma : float
        Volatility of the short rate
    r0 : float
        Initial short rate
    """

    def __init__(self, a: float, b: float, sigma: float, r0: float):
        if a <= 0:
            raise ValueError("Mean reversion speed 'a' must be positive")
        if sigma <= 0:
            raise ValueError("Volatility 'sigma' must be positive")

        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def simulate_short_rate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate short rate paths using the exact discretization.

        The Vasicek model has an exact solution:
        r(t+dt) = r(t)e^{-a*dt} + b(1 - e^{-a*dt}) + σ√((1-e^{-2a*dt})/(2a)) * Z

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
        t = np.linspace(0, T, n_steps + 1)
        r = np.zeros((n_paths, n_steps + 1))
        r[:, 0] = self.r0

        # Precompute constants for exact discretization
        exp_factor = np.exp(-self.a * dt)
        mean_factor = self.b * (1 - exp_factor)
        std_factor = self.sigma * np.sqrt((1 - np.exp(-2 * self.a * dt)) / (2 * self.a))

        # Generate all random numbers at once
        Z = np.random.standard_normal((n_paths, n_steps))

        for i in range(n_steps):
            r[:, i + 1] = r[:, i] * exp_factor + mean_factor + std_factor * Z[:, i]

        return t, r

    def instantaneous_forward_rate(
        self,
        r_t: np.ndarray,
        t: float,
        T: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the instantaneous forward rate f(t, T) given short rate r(t).

        Under the Vasicek model, the instantaneous forward rate is:
        f(t,T) = r(t)e^{-a(T-t)} + b(1 - e^{-a(T-t)}) - (σ²/2a²)(1 - e^{-a(T-t)})²

        Parameters
        ----------
        r_t : np.ndarray
            Current short rate values, shape (n_paths,) or scalar
        t : float
            Current time
        T : np.ndarray
            Maturity times, shape (n_maturities,)

        Returns
        -------
        f : np.ndarray
            Forward rates, shape (n_paths, n_maturities) or (n_maturities,)
        """
        tau = T - t
        if np.any(tau < 0):
            raise ValueError("Maturity T must be >= current time t")

        exp_factor = np.exp(-self.a * tau)

        # Ensure proper broadcasting
        r_t = np.atleast_1d(r_t)
        if r_t.ndim == 1:
            r_t = r_t[:, np.newaxis]

        f = (r_t * exp_factor +
             self.b * (1 - exp_factor) -
             (self.sigma**2 / (2 * self.a**2)) * (1 - exp_factor)**2)

        return np.squeeze(f)

    def simulate_forward_rate_curve(
        self,
        T_horizon: float,
        n_time_steps: int,
        maturities: np.ndarray,
        n_paths: int = 1,
        seed: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate the evolution of the forward rate curve over time.

        Parameters
        ----------
        T_horizon : float
            Simulation time horizon
        n_time_steps : int
            Number of time steps for simulation
        maturities : np.ndarray
            Array of maturities (tenors) for the forward curve
        n_paths : int
            Number of simulation paths
        seed : int, optional
            Random seed

        Returns
        -------
        t : np.ndarray
            Time grid, shape (n_time_steps + 1,)
        r : np.ndarray
            Short rate paths, shape (n_paths, n_time_steps + 1)
        f : np.ndarray
            Forward rate curves, shape (n_paths, n_time_steps + 1, n_maturities)
        """
        t, r = self.simulate_short_rate(T_horizon, n_time_steps, n_paths, seed)

        n_maturities = len(maturities)
        f = np.zeros((n_paths, n_time_steps + 1, n_maturities))

        for i, t_i in enumerate(t):
            # Forward maturities from current time
            T_forward = t_i + maturities
            f[:, i, :] = self.instantaneous_forward_rate(r[:, i], t_i, T_forward)

        return t, r, f

    def zero_coupon_bond_price(
        self,
        r_t: float,
        t: float,
        T: float
    ) -> float:
        """
        Calculate zero-coupon bond price P(t, T) under the Vasicek model.

        P(t,T) = A(t,T) * exp(-B(t,T) * r(t))

        where:
        B(t,T) = (1 - e^{-a(T-t)}) / a
        A(t,T) = exp((B(t,T) - T + t)(a²b - σ²/2)/a² - σ²B(t,T)²/(4a))

        Parameters
        ----------
        r_t : float
            Current short rate
        t : float
            Current time
        T : float
            Bond maturity

        Returns
        -------
        P : float
            Zero-coupon bond price
        """
        tau = T - t
        if tau < 0:
            raise ValueError("Maturity T must be >= current time t")
        if tau == 0:
            return 1.0

        B = (1 - np.exp(-self.a * tau)) / self.a
        A = np.exp(
            (B - tau) * (self.a**2 * self.b - self.sigma**2 / 2) / self.a**2
            - self.sigma**2 * B**2 / (4 * self.a)
        )

        return A * np.exp(-B * r_t)

    def yield_curve(
        self,
        r_t: float,
        t: float,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the yield curve given current short rate.

        Y(t,T) = -ln(P(t,T)) / (T - t)

        Parameters
        ----------
        r_t : float
            Current short rate
        t : float
            Current time
        maturities : np.ndarray
            Array of maturities

        Returns
        -------
        yields : np.ndarray
            Yield curve values
        """
        yields = np.zeros(len(maturities))
        for i, T in enumerate(maturities):
            tau = T - t
            if tau > 0:
                P = self.zero_coupon_bond_price(r_t, t, T)
                yields[i] = -np.log(P) / tau
            else:
                yields[i] = r_t
        return yields

    @property
    def long_term_mean(self) -> float:
        """Long-term mean of the short rate."""
        return self.b

    @property
    def long_term_variance(self) -> float:
        """Long-term (stationary) variance of the short rate."""
        return self.sigma**2 / (2 * self.a)

    @property
    def long_term_std(self) -> float:
        """Long-term (stationary) standard deviation of the short rate."""
        return np.sqrt(self.long_term_variance)

    def __repr__(self) -> str:
        return (f"VasicekModel(a={self.a}, b={self.b}, "
                f"sigma={self.sigma}, r0={self.r0})")
