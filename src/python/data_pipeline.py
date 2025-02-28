"""
Data Pipeline for Financial Data

Purpose:
This module handles the ingestion, cleaning, and preprocessing of financial data (e.g., market data,
alternative data, sentiment data). It ensures that the data is ready for use in machine learning models
and other components of the FORESIGHT system.

Role in FORESIGHT:
- Centralized data ingestion and preprocessing.
- Ensures data quality and consistency.
- Integrates with other modules (e.g., machine learning, portfolio analysis).

Key Features:
- Data loading from various sources (e.g., CSV, Parquet, APIs).
- Data cleaning and transformation.
- Integration with the broader FORESIGHT pipeline.
"""

import pandas as pd

def load_market_data(file_path: str) -> pd.DataFrame:
    """
    Load market data from a file (e.g., CSV, Parquet).

    Args:
        file_path (str): Path to the market data file.

    Returns:
        pd.DataFrame: Market data as a Pandas DataFrame.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")

def clean_market_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean market data by handling missing values and outliers.

    Args:
        data (pd.DataFrame): Raw market data.

    Returns:
        pd.DataFrame: Cleaned market data.
    """
    # Drop rows with missing values
    data = data.dropna()
    # Remove outliers (e.g., prices outside 3 standard deviations)
    data = data[(data['price'] - data['price'].mean()).abs() <= 3 * data['price'].std()]
    return data

# Example usage
if __name__ == "__main__":
    data = load_market_data("data/market_data.csv")
    cleaned_data = clean_market_data(data)
    print("Cleaned market data:", cleaned_data.head())