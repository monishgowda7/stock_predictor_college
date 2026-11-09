# Feature Catalog 📚

This document explains all features used in the NIFTY50 Market Direction Prediction project.

The features are divided into two categories:

1. Original Market Features
2. Engineered Technical Indicators

The objective of feature engineering was to capture different aspects of market behaviour:

- Trend 📈
- Momentum ⚡
- Volatility 🌊
- Market sentiment 🌍

---

# 1. Original Market Features 📊

Original features represent raw market information collected from different financial instruments.

These include:

- OHLC prices
- Trading volume
- Daily returns

---

## NIFTY 50 Features 🇮🇳

| Feature | Description |
|---|---|
| NIFTY_Open | Opening price of NIFTY 50 |
| NIFTY_High | Highest price during the trading day |
| NIFTY_Low | Lowest price during the trading day |
| NIFTY_Close | Closing price of NIFTY 50 |
| NIFTY_Volume | Trading volume |

---

## Volatility Index (VIX) Features 📉

VIX represents market uncertainty and investor fear.

| Feature | Description |
|---|---|
| VIX_Open | Opening value |
| VIX_High | Highest value |
| VIX_Low | Lowest value |
| VIX_Close | Closing value |

---

## Global Market Indicators 🌎

Global markets were included because international movements can influence Indian markets.

### S&P 500

| Feature | Description |
|---|---|
| SP500_Open | Opening price |
| SP500_High | Highest price |
| SP500_Low | Lowest price |
| SP500_Close | Closing price |
| SP500_Volume | Trading volume |

---

### NASDAQ

NASDAQ was included to capture global technology sector movement.

Features:

- NASDAQ_Open
- NASDAQ_High
- NASDAQ_Low
- NASDAQ_Close
- NASDAQ_Volume

---

### Dow Jones

Represents broader US market performance.

Features:

- DOW_Open
- DOW_High
- DOW_Low
- DOW_Close
- DOW_Volume

---

## Commodity and Currency Indicators 🌐

### USD/INR

Currency movement can affect foreign investment and market sentiment.

Features:

- USDINR_Open
- USDINR_High
- USDINR_Low
- USDINR_Close


### Crude Oil

Crude oil is important because energy prices influence inflation and economic conditions.

Features:

- CRUDE_Open
- CRUDE_High
- CRUDE_Low
- CRUDE_Close
- CRUDE_Volume

---

# Daily Returns 📈

Returns represent percentage change instead of absolute prices.

| Feature | Description |
|---|---|
| NIFTY_Return | Daily NIFTY movement |
| SP500_Return | Daily S&P 500 movement |
| NASDAQ_Return | Daily NASDAQ movement |
| DOW_Return | Daily Dow Jones movement |
| USDINR_Return | Daily currency movement |
| CRUDE_Return | Daily crude oil movement |
| VIX_Return | Daily volatility change |

---

# 2. Engineered Technical Indicators ⚙️

Technical indicators were created to provide additional information about market behaviour.

They capture:

- Trend direction
- Momentum strength
- Price position
- Market volatility

---

# Moving Average Features 📈

## Simple Moving Average (SMA)

SMA smooths price fluctuations and identifies general trends.

| Feature | Description |
|---|---|
| NIFTY_SMA_10 | 10-day Simple Moving Average |
| NIFTY_SMA_20 | 20-day Simple Moving Average |

---

## Price Relative to SMA

These features show whether current price is above or below its moving average.

| Feature | Description |
|---|---|
| Price_vs_SMA10 | NIFTY_Close / SMA10 |
| Price_vs_SMA20 | NIFTY_Close / SMA20 |

Values:

- Above 1 → Price above average
- Below 1 → Price below average

---

# Exponential Moving Average (EMA) 📊

EMA gives higher importance to recent prices.

| Feature | Description |
|---|---|
| NIFTY_EMA_10 | 10-day EMA |
| NIFTY_EMA_20 | 20-day EMA |

---

## Price Relative to EMA

| Feature | Description |
|---|---|
| Price_vs_EMA10 | NIFTY_Close / EMA10 |
| Price_vs_EMA20 | NIFTY_Close / EMA20 |

---

# Relative Strength Index (RSI) ⚡

RSI measures market momentum.

| Feature | Description |
|---|---|
| RSI_14 | 14-period Relative Strength Index |

Interpretation:

- High RSI → Strong upward momentum
- Low RSI → Weak momentum

---

# MACD Indicator 📉

MACD captures trend changes and momentum.

| Feature | Description |
|---|---|
| MACD | Difference between short and long EMA |
| MACD_Signal | Signal line |
| MACD_Histogram | Difference between MACD and Signal |

---

# Bollinger Bands 🌊

Bollinger Bands measure price volatility.

| Feature | Description |
|---|---|
| BB_High | Upper band |
| BB_Middle | Moving average |
| BB_Low | Lower band |
| Price_vs_BB | Current price position inside bands |

---

# Average True Range (ATR) 📏

ATR measures market volatility.

| Feature | Description |
|---|---|
| ATR_14 | 14-period Average True Range |
| ATR_Percent | ATR relative to closing price |

---

# Rate of Change (ROC) 🚀

ROC measures momentum by comparing current price with previous prices.

| Feature | Description |
|---|---|
| ROC_10 | 10-period Rate of Change |

---

# Rolling Volatility 📉

Measures recent market uncertainty.

| Feature | Description |
|---|---|
| Rolling_Volatility_20 | 20-day rolling standard deviation of NIFTY returns |

---

# Feature Engineering Philosophy 💡

The goal of feature engineering was not to add as many indicators as possible.

Instead, features were designed to represent different market behaviours:
Price Data
↓
Trend Features
↓
Momentum Features
↓
Volatility Features
↓
Machine Learning Models

The experiments showed that additional features do not always improve performance. Feature selection and model evaluation are necessary to identify useful signals.
