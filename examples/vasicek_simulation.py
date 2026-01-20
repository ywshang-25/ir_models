"""
Example: Vasicek Model Simulation

Demonstrates simulation of short rates and instantaneous forward rates
using the Vasicek model.
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from lib.vasicek import VasicekModel


def main():
    # Model parameters (typical values)
    a = 0.5       # Mean reversion speed
    b = 0.05      # Long-term mean (5%)
    sigma = 0.02  # Volatility (2%)
    r0 = 0.03     # Initial short rate (3%)

    model = VasicekModel(a=a, b=b, sigma=sigma, r0=r0)
    print(f"Model: {model}")
    print(f"Long-term mean: {model.long_term_mean:.2%}")
    print(f"Long-term std:  {model.long_term_std:.2%}")

    # Simulation parameters
    T = 10.0          # 10 years
    n_steps = 1000    # Time steps
    n_paths = 100     # Number of paths

    # Simulate short rate paths
    t, r = model.simulate_short_rate(T, n_steps, n_paths, seed=42)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Short rate paths
    ax1 = axes[0, 0]
    for i in range(min(20, n_paths)):
        ax1.plot(t, r[i, :], alpha=0.5, linewidth=0.8)
    ax1.axhline(y=b, color='red', linestyle='--', label=f'Long-term mean ({b:.1%})')
    ax1.axhline(y=b + 2*model.long_term_std, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(y=b - 2*model.long_term_std, color='orange', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Short Rate r(t)')
    ax1.set_title('Vasicek Short Rate Simulation (20 paths)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution at terminal time
    ax2 = axes[0, 1]
    ax2.hist(r[:, -1], bins=30, density=True, alpha=0.7, edgecolor='black')
    # Theoretical stationary distribution
    x_range = np.linspace(r[:, -1].min(), r[:, -1].max(), 100)
    theoretical_std = model.long_term_std
    theoretical_pdf = (1 / (theoretical_std * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((x_range - b) / theoretical_std)**2))
    ax2.plot(x_range, theoretical_pdf, 'r-', linewidth=2, label='Stationary distribution')
    ax2.axvline(x=b, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Short Rate r(T)')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Distribution at T={T} years')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Forward rate curve evolution
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])  # Tenors in years
    t_sim, r_sim, f_sim = model.simulate_forward_rate_curve(
        T_horizon=5.0,
        n_time_steps=50,
        maturities=maturities,
        n_paths=1,
        seed=42
    )

    ax3 = axes[1, 0]
    time_indices = [0, 10, 25, 50]  # Different time points
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))
    for idx, ti in enumerate(time_indices):
        ax3.plot(maturities, f_sim[0, ti, :] * 100,
                 marker='o', color=colors[idx],
                 label=f't = {t_sim[ti]:.1f} years')
    ax3.set_xlabel('Tenor (years)')
    ax3.set_ylabel('Forward Rate (%)')
    ax3.set_title('Evolution of Instantaneous Forward Rate Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Yield curve
    ax4 = axes[1, 1]
    maturities_yield = np.linspace(0.1, 30, 100)
    for r_val, label in [(0.02, 'r=2%'), (0.05, 'r=5%'), (0.08, 'r=8%')]:
        yields = model.yield_curve(r_val, 0, maturities_yield) * 100
        ax4.plot(maturities_yield, yields, label=label, linewidth=2)
    ax4.axhline(y=b * 100, color='black', linestyle='--', alpha=0.5, label='Long-term mean')
    ax4.set_xlabel('Maturity (years)')
    ax4.set_ylabel('Yield (%)')
    ax4.set_title('Yield Curves for Different Initial Short Rates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vasicek_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nSimulation complete. Figure saved to 'vasicek_simulation.png'")


if __name__ == '__main__':
    main()
