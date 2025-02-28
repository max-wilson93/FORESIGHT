"""
Option Greeks Calculation

Purpose:
This module computes the Greeks (Delta, Gamma, Vega, Theta, Rho) for options, which measure the
sensitivity of an option's price to various factors. This is critical for hedging and risk management.

Role in FORESIGHT:
- Quantifies options risk for derivatives trading.
- Integrates with portfolio optimization and hedging strategies.
- Supports real-time risk monitoring.

Key Features:
- Black-Scholes model for Greeks calculation.
- Sensitivity analysis for volatility and time decay.
"""

import numpy as np
from scipy.stats import norm

def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    Calculate the Delta of an option using the Black-Scholes model.

    Args:
        S (float): Spot price of the underlying asset.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free rate.
        sigma (float): Volatility of the underlying asset.
        option_type (str): "call" or "put".

    Returns:
        float: Delta of the option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    return delta

# Example usage
if __name__ == "__main__":
    delta = calculate_delta(S=100, K=110, T=1.0, r=0.05, sigma=0.2, option_type="call")
    print("Option Delta:", delta)