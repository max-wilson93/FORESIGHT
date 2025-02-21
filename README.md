


# FORESIGHT: AI-Powered Stock Price Prediction System

**FORESIGHT** (**F**ourier-Informed **O**ptimization for **R**einforcement-Learned **E**ntries and **S**entiment-Integrated **I**nvestment using **G**PU-Accelerated **H**istorical & **T**echnical Analysis) is a high-performance stock price prediction system that integrates **deep learning**, **reinforcement learning**, **Fourier transforms**, and **sentiment analysis**. The system leverages **CUDA optimization** for real-time inference and market predictions, enabling dynamic trading strategies.

---

## **Core Components and Their Roles**

### **1. Signal Processing: Fourier and Wavelet Transforms**
#### **Purpose**
To extract meaningful patterns from noisy stock price data by decomposing time-series data into frequency components.

#### **How It Helps**
- **Fourier Transform**: Identifies periodic trends and dominant cycles in stock price movements. It filters out high-frequency noise, allowing the model to focus on significant trends.
- **Wavelet Transform**: Analyzes time-frequency variations, enabling the detection of sudden market shifts and anomalies that are not captured by traditional methods.

#### **Why It Matters**
By isolating meaningful trends and filtering noise, these transforms improve the quality of input data for downstream machine learning models, leading to more accurate predictions.

---

### **2. Reinforcement Learning: Deep Q-Learning and Actor-Critic**
#### **Purpose**
To develop adaptive trading strategies that maximize profit while minimizing risk.

#### **How It Helps**
- **Deep Q-Learning (DQN)**: Trains an agent to make optimal trading decisions by learning a policy that maximizes expected rewards. The agent dynamically adapts to changing market conditions.
- **Advantage Actor-Critic (A2C)**: Stabilizes the learning process by separating the policy and value functions, enabling more efficient exploration of the trading strategy space.

#### **Why It Matters**
Reinforcement learning allows the system to learn and adapt to dynamic market conditions, optimizing trading strategies in real-time based on reward feedback.

---

### **3. Deep Learning: LSTMs and Temporal Convolutional Networks (TCNs)**
#### **Purpose**
To model long-term dependencies in stock price data for accurate price forecasting.

#### **How It Helps**
- **Long Short-Term Memory (LSTM)**: Captures sequential dependencies in stock price movements, making it effective for modeling time-series data.
- **Temporal Convolutional Networks (TCNs)**: Uses dilated convolutions to efficiently model long-range dependencies, improving computational efficiency compared to traditional LSTMs.

#### **Why It Matters**
Deep learning models excel at capturing complex temporal patterns in stock price data, enabling more accurate predictions than traditional statistical methods.

---

### **4. Sentiment Analysis: Transformer Models**
#### **Purpose**
To incorporate financial sentiment from news and social media into stock price predictions.

#### **How It Helps**
- **Transformer Models**: Analyze textual data (e.g., news articles, social media posts) to extract financial sentiment. The attention mechanism allows the model to focus on relevant words and phrases, capturing contextual relationships in the text.
- **ONNX Runtime**: Enables efficient inference of transformer models, ensuring low-latency sentiment analysis.

#### **Why It Matters**
Sentiment analysis provides additional context for stock price predictions, allowing the model to account for market reactions to news and social media trends.

---

### **5. Backtesting and Evaluation Metrics**
#### **Purpose**
To evaluate the performance of trading strategies and ensure robustness.

#### **How It Helps**
- **Sharpe Ratio**: Measures risk-adjusted returns, ensuring that the system balances profitability and risk.
- **Profit & Loss (P&L)**: Tracks cumulative profits and losses over time, providing a clear picture of the system's performance.
- **Maximum Drawdown**: Measures the largest peak-to-trough decline in portfolio value, helping to assess risk exposure.

#### **Why It Matters**
Rigorous backtesting and evaluation ensure that the system performs well under realistic market conditions, providing confidence in its reliability.

---

## **Project Structure**
```
/stock-predictor
│── /src
│   │── main.cpp                 # Main entry point
│   │── /data
│   │   │── data_loader.h         # Handles stock data retrieval
│   │   │── data_loader.cpp       # Implements data fetching & preprocessing
│   │   │── transform.h           # FFT & wavelet functions
│   │   │── transform.cpp         # Implements FFT and wavelet filtering
│   │── /models
│   │   │── rl_model.h            # Reinforcement learning model
│   │   │── rl_model.cpp          # CUDA-optimized RL implementation
│   │   │── dnn_model.h           # Deep learning model (LSTM/TCN)
│   │   │── dnn_model.cpp         # Uses LibTorch with CUDA
│   │── /sentiment
│   │   │── sentiment.h           # NLP-based sentiment analysis
│   │   │── sentiment.cpp         # Uses ONNX Runtime for Transformers
│   │── /evaluation
│   │   │── backtest.h            # Backtesting framework
│   │   │── backtest.cpp          # Implements trading strategy evaluation
│   │   │── metrics.h             # Accuracy metrics
│   │   │── metrics.cpp           # Implements model performance calculations
│   │── /utils
│   │   │── config.h              # Global config (tickers, API keys, parameters)
│   │   │── logging.h             # Logging utilities
│   │   │── logging.cpp           # Implements log system
│── /include                      # Shared header files
│── /tests                        # Unit tests for modules
│── /models                       # Trained models (saved weights)
│── /data                         # Stored stock price history
│── /scripts                      # Python scripts for dataset preparation
│── Makefile                       # Build system configuration
│── README.md                      # Project documentation
```

---

## **Technologies Used**
- **C++ (C++17/20):** High-performance execution.
- **CUDA:** GPU acceleration for deep learning and reinforcement learning.
- **LibTorch:** PyTorch C++ API for deep learning models.
- **ONNX Runtime:** Efficient inference for Transformer models.
- **FFT & Wavelet Transform:** Signal processing for stock price data.

---

## **How it Works**
```
./stock-predictor AAPL current
```
Here’s how FORESIGHT processes this request step by step:

1. Parse Command-Line Arguments
The program reads the arguments (AAPL as the ticker symbol, current as the mode).
It determines that you want a real-time prediction rather than backtesting.
2. Fetch Latest Stock Market Data
The data_loader module (data_loader.cpp) queries a market data API (e.g., Alpha Vantage, Yahoo Finance, or a firm’s proprietary feed) for AAPL’s latest price, volume, and technical indicators.
If using an internal database, it fetches the most recent records. <br> **Example Output (Raw Data Sample)**:
```
{
  "timestamp": "2025-02-19T14:30:00Z",
  "open": 185.12,
  "high": 186.50,
  "low": 184.90,
  "close": 186.10,
  "volume": 12_345_678
}
```
3. Apply Fourier and Wavelet Transformations
The transform module (transform.cpp) preprocesses the stock data using Fourier and wavelet transforms to:
Remove high-frequency noise.
Identify periodic market trends and anomalies. <br> **Example Fourier Transform Output**:
```
{
  "dominant_frequencies": [0.0005, 0.0021],
  "trend_strength": 0.87
}
```
4. Perform Sentiment Analysis on Market News & Social Media
The sentiment module (sentiment.cpp) fetches financial news and Twitter/Reddit sentiment.
A Transformer model (e.g., FinBERT) processes the text and assigns sentiment scores.
ONNX Runtime accelerates inference for near-instant results.
<br> **Example Sentiment Output**:
```
{
  "news_sentiment": 0.72,  // Positive sentiment
  "social_media_sentiment": -0.15,  // Slightly negative sentiment
  "aggregated_score": 0.65  // Weighted combination
}
```
5. Pass Data to the Deep Learning Model (LSTM / TCN)
The DNN module (dnn_model.cpp) loads a pre-trained LSTM or TCN model (stored in /models/ directory).
The model predicts the next price movement based on historical and transformed data.
CUDA acceleration is used for fast inference.
<br> **Example Prediction Output**:

```
{
  "predicted_price": 187.25,
  "confidence": 0.92
}
```
6. Reinforcement Learning Model Suggests an Action
The RL module (rl_model.cpp) takes the predicted price, sentiment data, and Fourier features to decide whether to:
BUY, SELL, or HOLD AAPL shares.
Deep Q-Learning (DQN) or Actor-Critic (A2C) evaluates past rewards and risk exposure.
<br> **Example RL Output**:
```
{
  "recommended_action": "BUY",
  "expected_reward": 1.45,
  "risk_factor": 0.30
}
```
7. Generate Final Output for the User
The CLI displays the result in a human-readable format:
<br> **Terminal Output**:

```
[FORESIGHT] Stock Prediction for AAPL (Real-Time)
---------------------------------------------------
Current Price:      $186.10
Predicted Price:    $187.25 (Confidence: 92%)
Sentiment Score:    0.65 (Moderately Positive)
RL Action:          BUY
Risk Factor:        30%
---------------------------------------------------
Recommendation: BUY - Expected Reward: 1.45x
```
If logging is enabled, the system appends this result to a CSV or database for tracking.

---

## **Future Work**
- Optimize Transformer inference using **TensorRT**.
- Implement alternative RL strategies like **PPO** and **SAC**.
- Develop a real-time trading execution system.

---

## **License**
MIT License. See `LICENSE` for details.

