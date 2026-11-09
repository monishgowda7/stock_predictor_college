# Data Collection 🗂️

This document explains the data collection process used for the **NIFTY50 Market Direction Prediction** project.

The objective was to collect historical market data from multiple financial instruments and create a unified dataset for machine learning-based market direction prediction.

The data collection pipeline followed:

```text
Financial Data Sources
          ↓
Historical Data Download
          ↓
Data Cleaning
          ↓
Data Alignment
          ↓
Return Calculation
          ↓
Final Dataset Creation
```

---

# Data Source 📊

## Why Yahoo Finance?

Yahoo Finance was selected as the primary data source because it provides:

- Historical OHLCV market data
- Multiple global financial instruments
- Long-term historical availability
- Free accessibility through the `yfinance` Python library

The data was collected programmatically to ensure:

- Reproducibility
- Consistency
- Easy future updates

---

# Data Collection Library 🐍

The project uses the `yfinance` Python library to download historical market data.

Example:

```python
import yfinance as yf

data = yf.download(
    ticker,
    start=start_date,
    end=end_date
)
```

---

# Market Instruments Collected 🌍

To capture both domestic and global influences on the Indian stock market, data was collected from multiple financial instruments.

---

## 🇮🇳 NIFTY 50 Index

**Ticker:** `^NSEI`

Collected Features:

- Open
- High
- Low
- Close
- Volume

Purpose:

Represents the overall performance of the Indian equity market and serves as the target market for prediction.

---

## 📉 India VIX

Collected Features:

- Open
- High
- Low
- Close

Purpose:

Measures expected market volatility and investor uncertainty.

---

## 🇺🇸 S&P 500

Collected Features:

- Open
- High
- Low
- Close
- Volume

Purpose:

Represents the overall performance of the US stock market.

---

## 💻 NASDAQ Composite

Collected Features:

- Open
- High
- Low
- Close
- Volume

Purpose:

Captures movements in the technology sector, which often influences global investor sentiment.

---

## 🏛️ Dow Jones Industrial Average

Collected Features:

- Open
- High
- Low
- Close
- Volume

Purpose:

Represents large-cap US companies and provides additional global market context.

---

## 💱 USD/INR Exchange Rate

Collected Features:

- Open
- High
- Low
- Close

Purpose:

Currency movements affect foreign investments, imports, exports, and overall market sentiment.

---

## 🛢️ Crude Oil

Collected Features:

- Open
- High
- Low
- Close
- Volume

Purpose:

Crude oil prices influence inflation, transportation costs, and economic activity.

---

# Historical Data Coverage ⏳

The collected dataset spans:

| Information | Value |
|------------|-------|
| Start Date | 2008-03-04 |
| End Date | 2026-07-21 |

The long historical period was chosen to include:

- Bull markets
- Bear markets
- High-volatility periods
- Economic cycles
- Different market conditions

---

# Data Cleaning 🧹

After downloading the data, several preprocessing steps were performed before model development.

## Missing Value Handling

Different markets operate on different trading calendars.

Missing values may occur because of:

- Market holidays
- Different trading sessions
- Delayed market openings

These values were handled to create a consistent dataset across all financial instruments.

---

## Data Alignment

Since all markets trade in different regions and time zones, the downloaded datasets were aligned using trading dates.

This ensured that every row represented the same trading period across all markets.

---

## Dataset Integration

Individual datasets were merged into a single dataframe containing:

- Indian market data
- Global indices
- Currency data
- Commodity data

This unified dataset served as the foundation for feature engineering and model development.

---

# Daily Return Features 📈

Daily percentage returns were calculated for major financial instruments.

Formula:

```text
Return = (Current Price − Previous Price) / Previous Price
```

Generated return features:

- NIFTY_Return
- SP500_Return
- NASDAQ_Return
- DOW_Return
- USDINR_Return
- CRUDE_Return
- VIX_Return

Returns help normalize price movements and allow comparisons between assets with different price ranges.

---

# Final Dataset 📋

After data collection and preprocessing:

| Property | Value |
|----------|-------|
| Dataset Size | **4347 rows × 41 columns** |

The dataset includes:

- Original market features
- Daily return features

Additional technical indicators were created later during the feature engineering stage.

---

# Preventing Data Leakage 🔒

Financial time-series data requires careful handling to avoid introducing future information into the training process.

To prevent data leakage:

- Data was sorted chronologically.
- Future observations were never used during training.
- A time-based train/validation/test split was applied instead of random splitting.

This ensures that model evaluation closely resembles real-world forecasting.

---

# Data Collection Outcome ✅

The completed dataset provided a strong foundation for:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Development
- Model Evaluation
- Financial Time-Series Prediction

The final data pipeline can be summarized as:

```text
Indian Market Data
          +
Global Market Indices
          +
Volatility Index
          +
Currency Data
          +
Commodity Prices
          ↓
Data Cleaning & Alignment
          ↓
Daily Return Calculation
          ↓
Unified Dataset
          ↓
Machine Learning Pipeline
```
