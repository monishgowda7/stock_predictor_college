# Raw Dataset

This directory stores the original raw datasets used in this project before any preprocessing or feature engineering.

Due to GitHub file size limitations and data licensing considerations, the raw datasets are not included in this repository.

## Data Sources

The datasets used in this project were collected from Yahoo Finance using the `yfinance` library.

Included market data:

- NIFTY 50
- India VIX
- S&P 500
- NASDAQ Composite
- Dow Jones Industrial Average
- USD/INR Exchange Rate
- Crude Oil Futures

## Download

Run the notebook below to download the complete raw dataset:

```text
notebooks/01_data_collection.ipynb
```

The notebook automatically downloads, cleans, and prepares all required market data.
