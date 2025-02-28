"""
Portfolio Analysis and Optimization

Purpose:
This module analyzes portfolio performance and computes risk metrics such as Value-at-Risk (VaR)
and Sharpe ratio. It also integrates with optimization algorithms to suggest portfolio adjustments.

Role in FORESIGHT:
- Provides insights into portfolio performance and risk.
- Integrates with the portfolio optimization pipeline.
- Supports the financial decision-making process.

Key Features:
- Portfolio performance metrics (e.g., returns, volatility).
- Risk metrics (e.g., VaR, Sharpe ratio).
- Integration with optimization algorithms.
"""

import numpy as np
import pandas as pd

def calculate_portfolio_returns(portfolio: pd.DataFrame) -> float:
    """
    Calculate the total returns of a portfolio.

    Args:
        portfolio (pd.DataFrame): Portfolio data with weights and returns.

    Returns:
        float: Total portfolio returns.
    """
    return np.dot(portfolio['weights'], portfolio['returns'])

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate the Sharpe ratio of a portfolio.

    Args:
        returns (pd.Series): Portfolio returns.
        risk_free_rate (float): Risk-free rate (default: 0.02).

    Returns:
        float: Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# Example usage
if __name__ == "__main__":
    portfolio = pd.DataFrame({
        "weights": [0.5, 0.3, 0.2],
        "returns": [0.1, 0.2, 0.3]
    })
    total_returns = calculate_portfolio_returns(portfolio)
    sharpe_ratio = calculate_sharpe_ratio(portfolio['returns'])
    print("Total portfolio returns:", total_returns)
    print("Sharpe ratio:", sharpe_ratio)