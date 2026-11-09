# Why This Project? 📈

## Why Predict Market Direction Instead of Price?

Stock price prediction is extremely difficult because prices are affected by:

- Economic conditions
- Global events
- Investor psychology
- Market uncertainty

Instead of predicting the exact price, this project focuses on a **classification problem**:
1 → Next trading day moves upward 📈
0 → Next trading day moves downward 📉

Predicting direction makes the problem more realistic and allows machine learning models to learn market patterns.

---

# Data Decisions 🗂️

## Why NIFTY 50?

NIFTY 50 represents the performance of the **50 largest companies listed on the National Stock Exchange of India**.

It was selected because:

- It represents the overall Indian equity market
- It has high liquidity and historical availability
- It reflects different sectors of the Indian economy

---

## Why Yahoo Finance?

Yahoo Finance was selected as the primary data source because it provides:

- Historical OHLCV market data
- Global market indices
- Long-term historical coverage
- Easy accessibility through `yfinance`

This allowed consistent data collection for multiple financial instruments.

---

# Feature Selection Decisions 🔍

## Why Include Global Market Indicators?

Modern financial markets are interconnected.

Events in global markets can influence Indian markets through:

- Foreign investments
- Economic sentiment
- Currency movements

Therefore, global indicators were included:

| Feature | Reason |
|---|---|
| S&P 500 | US market performance |
| NASDAQ | Technology sector sentiment |
| Dow Jones | Global market trend |
| VIX | Market uncertainty |
| USD/INR | Currency movement |
| Crude Oil | Inflation and energy impact |

---

## Why Include Technical Indicators?

Raw prices alone do not capture market behaviour.

Technical indicators help represent:

- Momentum
- Trend strength
- Volatility
- Price movement patterns

The project uses:

| Indicator | Purpose |
|---|---|
| SMA | Identify overall trend |
| EMA | Give more importance to recent prices |
| RSI | Measure momentum |
| MACD | Detect trend changes |
| Bollinger Bands | Measure volatility |

---

# Data Splitting Decisions ⏳

## Why Time-Based Train Validation Test Split?

Financial data is sequential.

The model should learn from:
Past Data → Future Prediction

Random splitting was avoided because it can introduce **data leakage**, where future information accidentally influences training.

The dataset was divided chronologically:

- Training: Historical market data
- Validation: Used for model selection
- Testing: Final unseen market period

---

# Model Selection Decisions 🤖

## Why Logistic Regression?

Logistic Regression was selected as a baseline model.

Reasons:

- Simple and interpretable
- Provides a performance benchmark
- Works well for binary classification problems

---

## Why Decision Tree?

Decision Trees were included because they can capture:

- Non-linear relationships
- Feature interactions
- Rule-based patterns

---

## Why XGBoost?

XGBoost was selected because it is one of the strongest algorithms for structured/tabular data.

Advantages:

- Handles complex relationships
- Reduces overfitting using regularization
- Strong performance on classification tasks

---

## Why LightGBM?

LightGBM was selected because:

- Faster training compared to traditional boosting methods
- Efficient with large feature spaces
- Handles complex feature interactions

---

# Model Optimization ⚙️

Hyperparameter tuning was performed to improve model performance.

Methods used:

- Grid Search
- Cross Validation

Examples of tuned parameters:

### Logistic Regression

- Regularization strength (`C`)
- Solver selection

### Tree-based Models

- Learning rate
- Number of estimators
- Tree depth
- Regularization parameters

---

# Evaluation Decisions 📊

## Why Not Use Accuracy Alone?

Financial prediction datasets can be misleading with accuracy alone.

Additional metrics were used:

| Metric | Purpose |
|---|---|
| Precision | How many predicted upward movements were correct |
| Recall | How many actual upward movements were detected |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Measures classification ability across thresholds |

---

## Why Feature Importance Analysis?

Understanding features is important because machine learning models should not only predict but also provide insights.

Feature importance helps:

- Identify influential market signals
- Remove unnecessary features
- Improve future model versions

---

# Project Philosophy 🚀

This project focuses on building a complete machine learning workflow:






The goal is not only to build a prediction model, but to understand **why each decision was made during the ML development process**.
