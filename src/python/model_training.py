"""
Model Training for Financial Forecasting

Purpose:
This module trains machine learning models (e.g., LSTM, Transformer) on financial data for
tasks such as price prediction, portfolio optimization, and risk management.

Role in FORESIGHT:
- Trains predictive models for financial forecasting.
- Integrates with the broader machine learning pipeline.
- Supports the financial decision-making process.

Key Features:
- Training of time-series models (e.g., LSTM, Transformer).
- Hyperparameter tuning and model evaluation.
- Integration with the broader FORESIGHT system.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_price_prediction_model(data: pd.DataFrame) -> RandomForestRegressor:
    """
    Train a machine learning model for price prediction.

    Args:
        data (pd.DataFrame): Market data with features.

    Returns:
        RandomForestRegressor: Trained model.
    """
    # Split data into features and target
    X = data.drop(columns=["price"])
    y = data["price"]
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    return model

# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "price": [100, 200, 300, 400, 500]
    })
    model = train_price_prediction_model(data)
    print("Trained model:", model)