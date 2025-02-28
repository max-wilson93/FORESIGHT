"""
Sentiment Analysis Visualization

Purpose:
This module generates visualizations (e.g., charts, reports) to display sentiment trends over time.
It helps users understand how public sentiment about specific stocks or the market evolves.

Role in FORESIGHT:
- Provides actionable insights through visualizations.
- Enhances user understanding of sentiment trends.
- Integrates with the broader sentiment analysis pipeline.

Key Features:
- Time-series plots for sentiment trends.
- Bar charts for sentiment distribution.
- Integration with the FinBERT model.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_trends(sentiment_data: pd.DataFrame):
    """
    Plot sentiment trends over time.

    Args:
        sentiment_data (pd.DataFrame): DataFrame containing sentiment data with a "date" column.
    """
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    sentiment_data.set_index('date', inplace=True)
    sentiment_data.resample('D').mean().plot()
    plt.title("Sentiment Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.show()

# Example usage
if __name__ == "__main__":
    sentiment_data = pd.DataFrame({
        "date": ["2023-10-01", "2023-10-02", "2023-10-03"],
        "sentiment_score": [0.8, 0.5, -0.2]
    })
    plot_sentiment_trends(sentiment_data)