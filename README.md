# NIFTY50 Market Direction Prediction

> 📈 An end-to-end Machine Learning pipeline for predicting the next-day direction of the NIFTY 50 Index using historical market data, global financial indicators, and technical analysis.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

---

## 🚀 Overview

Predicting financial markets is a challenging time-series classification problem due to market volatility, non-linearity, and continuously changing economic conditions.

This project develops a complete **Machine Learning pipeline** to predict the **next-day direction of the NIFTY 50 Index** by combining domestic market data, global financial indices, currency exchange rates, commodity prices, and engineered technical indicators.

Rather than focusing solely on model performance, this repository emphasizes **reproducible experimentation**, **systematic feature engineering**, and **engineering best practices** throughout the complete ML lifecycle.

---

# End-to-End Workflow

<img width="1536" height="1024" alt="Workflow" src="https://github.com/user-attachments/assets/590316a2-1fd7-4ed6-a204-0efdc0321dbf" />


The project follows a structured machine learning workflow beginning with historical market data collection and progressing through preprocessing, feature engineering, chronological data splitting, model development, hyperparameter tuning, feature refinement, and final evaluation.

This modular workflow ensures reproducibility while minimizing data leakage in a financial time-series environment.

---

# Project Architecture
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/217c43ee-a598-429b-ab05-b5fc799629cb" />

The architecture integrates multiple financial markets into a unified prediction pipeline.

### Data Sources

- NIFTY 50
- India VIX
- S&P 500
- NASDAQ Composite
- Dow Jones Industrial Average
- USD/INR Exchange Rate
- Crude Oil Futures

These datasets are synchronized chronologically before feature engineering and model training.

---

# Key Features

- 📊 End-to-end machine learning workflow
- 📈 Multi-market financial data integration
- ⚙️ Technical indicator feature engineering
- 📅 Chronological train-validation-test splitting
- 🤖 Comparative evaluation across multiple ML algorithms
- 🎯 Hyperparameter tuning and feature refinement
- 📚 Complete project documentation for reproducibility

---

# Machine Learning Pipeline

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/04e1bdde-84d1-4cdf-9caa-1c52f1e9f302" />


The pipeline was designed following a modular architecture where each stage can be independently reproduced, validated, and improved.

Major stages include:

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Time-Series Splitting
5. Model Development
6. Hyperparameter Optimization
7. Feature Selection
8. Performance Evaluation

---

# Data Collection

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/f70032bc-24d2-48f8-823b-9bc7f4bff7dd" />

Historical market data was collected using the **Yahoo Finance API (yfinance)**.

## Dataset Summary

| Property | Value |
|-----------|-------|
| Data Source | Yahoo Finance |
| Market | NIFTY 50 |
| Auxiliary Markets | VIX, S&P 500, NASDAQ, Dow Jones, USD/INR, Crude Oil |
| Frequency | Daily |
| Period | 2008 – 2026 |
| Prediction Task | Binary Classification |

The collected datasets were aligned by trading date before preprocessing and feature generation.

---

# Feature Engineering

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/9f92e72a-5753-4416-9aa1-4f53a8f6ccef" />


Raw market prices alone rarely capture the complete dynamics of financial markets.

To improve predictive capability, several technical indicators were engineered to represent trend, momentum, and volatility.

## Trend Indicators

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)

## Momentum Indicators

- Relative Strength Index (RSI)
- MACD
- Rate of Change (ROC)

## Volatility Indicators

- Bollinger Bands
- # Time-Series Split

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/798105bf-e547-4a61-8698-ec11cb2e7e33" />

</p>

Unlike conventional machine learning tasks, financial markets exhibit strong temporal dependencies. To preserve chronological order and eliminate data leakage, a **time-based split** was adopted instead of random sampling.

## Dataset Split

| Dataset | Period | Purpose |
|----------|--------|---------|
| Training | 2008 – 2022 | Model training |
| Validation | 2023 – 2024 | Hyperparameter tuning |
| Testing | 2025 – 2026 | Final model evaluation |

This strategy closely reflects real-world deployment, where models are trained on historical data and evaluated on future unseen observations.

---

# Model Development

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/67191f2d-db6d-484e-843a-9622f5965dd0" />
  
Model development followed an iterative, experiment-driven process rather than relying on a single model configuration.

### Version 1 — Baseline Models

The initial models were trained using the original market features to establish a performance benchmark.

### Version 2 — Engineered Features

A comprehensive set of technical indicators was introduced to enrich the feature space and capture trend, momentum, and volatility characteristics.

### Version 3 — Feature Refinement

The most informative engineered features were selected through feature importance analysis, followed by hyperparameter tuning to improve generalization performance.

This progressive development strategy enabled systematic evaluation of each enhancement while maintaining reproducibility.

---

# Algorithms Evaluated

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/5472307d-4d5c-4993-9d86-1353318fce1c" />

Multiple supervised learning algorithms were evaluated to compare linear and ensemble-based approaches.

| Algorithm | Purpose |
|------------|---------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Interpretable non-linear model |
| Random Forest | Ensemble learning baseline |
| XGBoost | Gradient boosting ensemble |
| CatBoost | Ordered boosting with categorical support |
| LightGBM | Efficient histogram-based boosting |

Each algorithm was evaluated under identical train, validation, and test partitions to ensure fair comparison.

---

# Feature Refinement

<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/061590eb-14c1-405f-886c-8ae3aa48d270" />


Feature importance analysis was used to identify the most influential predictors within the engineered feature set.

The final Version 3 models were trained using a refined subset of high-impact features, reducing redundancy while improving computational efficiency and interpretability.

---

# Model Performance

<p align="center">
  
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/cd4d3082-7f33-42db-9f13-35872a68aaa2" />

The final evaluation compares the refined Version 3 models on the held-out test dataset.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|:------|---------:|----------:|--------:|---------:|--------:|
| **XGBoost (Tuned)** | **0.5847** | **0.5963** | **0.5246** | **0.5581** | **0.5937** |
| CatBoost (Tuned) | 0.5683 | 0.5926 | 0.4372 | 0.5031 | 0.5928 |
| LightGBM (Tuned) | 0.5492 | 0.5763 | 0.3716 | 0.4518 | 0.5644 |
| Logistic Regression | 0.5000 | 0.5000 | **1.0000** | **0.6667** | 0.4604 |

## Key Findings

- 🎯 **XGBoost** achieved the strongest overall balance across Accuracy, F1 Score, and ROC-AUC.
- 📈 **Logistic Regression** achieved the highest Recall, correctly identifying the majority of positive market movements.
- ⚙️ **CatBoost** demonstrated competitive predictive performance with balanced precision and ROC-AUC.
- 🚀 **LightGBM** provided an efficient gradient boosting alternative with comparable classification performance.

Detailed evaluation artifacts—including ROC curves, confusion matrices, feature importance plots, and additional analysis—are available in the technical report and the `reports/figures/evaluation/` directory.

---

# Project Synthesis
<p align="center">
<img width="1376" height="768" alt="image" src="https://github.com/user-attachments/assets/f0c29ab0-59d5-4915-9f4a-54c1515cadd2" />


This project demonstrates a complete machine learning workflow for financial time-series classification, combining domain knowledge, systematic experimentation, and reproducible engineering practices.

The repository is structured to emphasize transparency in data preparation, feature engineering, model development, and evaluation, making it suitable as both a learning resource and a foundation for future research and deployment.
- Average True Range (ATR)
- Rolling Volatility

These engineered features significantly enriched the information available to the learning algorithms.

---

# 🌐 Interactive Web Application

An enterprise-grade financial terminal dashboard built with **Streamlit** and **Plotly** is included to serve real-time predictions, technical analysis, and model benchmarks.

### Features

1. **🔮 AI Direction Forecast**:
   - Next-day direction prediction: **BULLISH** vs **BEARISH**.
   - Model confidence gauge and probability breakdown.
   - Multi-model consensus (XGBoost, CatBoost, LightGBM, Random Forest, Logistic Regression).
   - **What-If Scenario Simulator**: Adjust global market indices (S&P 500, NASDAQ, Crude Oil, USD/INR, VIX) and RSI in real-time to see how simulated shocks flip market forecasts.

2. **📊 Interactive Technical Charts**:
   - Candlestick & line charts for NIFTY 50 with time range filters (1M, 3M, 6M, 1Y, 3Y, All).
   - Overlays: 10 & 20 SMA/EMA, Bollinger Bands with band fill.
   - Oscillators: RSI (14) with overbought/oversold levels, MACD with signal and histogram, Volume.

3. **🤖 Model Benchmarks & Evaluation Suite**:
   - Out-of-sample test set (2025–2026) performance metrics table.
   - Interactive confusion matrices for each model.
   - Top 15/25 feature importance rankings.

4. **📈 Walk-Forward Backtesting**:
   - Strategy simulation comparing the ML signal strategy against Buy & Hold.
   - Performance metrics: Cumulative Return, Maximum Drawdown, Win Rate.

5. **📁 Dataset Explorer & Feature Catalog**:
   - Filter, inspect, and export processed and engineered market data.

### Running the Web Application

Run the application with one click using the batch script or via terminal:

```bash
# Using the Windows batch script
double-click run_app.bat

# Or from terminal
python -m streamlit run app.py
```

Once started, open your browser at `http://localhost:8501`.
#   s t o c k _ p r e d i c t o r _ c o l l e g e  
 