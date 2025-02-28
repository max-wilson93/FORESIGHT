"""
Feature Engineering for Machine Learning

Purpose:
This module creates features from raw financial data for use in machine learning models.
It includes techniques such as technical indicators, lagged features, and rolling statistics.

Role in FORESIGHT:
- Enhances predictive models by providing meaningful features.
- Integrates with the machine learning pipeline.
- Supports the broader financial forecasting system.

Key Features:
- Technical indicators (e.g., moving averages, RSI).
- Lagged features for time-series data.
- Rolling statistics (e.g., rolling mean, rolling standard deviation).
"""

import pandas as pd

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the market data.

    Args:
        data (pd.DataFrame): Market data.

    Returns:
        pd.DataFrame: Market data with technical indicators.
    """
    # Example: Add moving averages
    data['sma_50'] = data['price'].rolling(window=50).mean()
    data['sma_200'] = data['price'].rolling(window=200).mean()
    # Example: Add Relative Strength Index (RSI)
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        "price": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    })
    data_with_indicators = add_technical_indicators(data)
    print("Data with technical indicators:", data_with_indicators)