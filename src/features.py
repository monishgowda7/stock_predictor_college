import numpy as np
import pandas as pd

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and engineered features from OHLCV data.
    Requires columns:
    - NIFTY_Close, NIFTY_High, NIFTY_Low, NIFTY_Open, NIFTY_Volume
    - S&P 500, NASDAQ, Dow Jones, USD/INR, Crude Oil, VIX data
    """
    df = df.copy()
    
    # 1. Simple Moving Averages
    df["NIFTY_SMA_10"] = df["NIFTY_Close"].rolling(window=10).mean()
    df["NIFTY_SMA_20"] = df["NIFTY_Close"].rolling(window=20).mean()
    
    df["Price_vs_SMA10"] = df["NIFTY_Close"] / df["NIFTY_SMA_10"]
    df["Price_vs_SMA20"] = df["NIFTY_Close"] / df["NIFTY_SMA_20"]
    
    # 2. Exponential Moving Averages
    df["NIFTY_EMA_10"] = df["NIFTY_Close"].ewm(span=10, adjust=False).mean()
    df["NIFTY_EMA_20"] = df["NIFTY_Close"].ewm(span=20, adjust=False).mean()
    
    df["Price_vs_EMA10"] = df["NIFTY_Close"] / df["NIFTY_EMA_10"]
    df["Price_vs_EMA20"] = df["NIFTY_Close"] / df["NIFTY_EMA_20"]
    
    # 3. RSI 14
    delta = df["NIFTY_Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 4. MACD (12, 26, 9)
    ema_12 = df["NIFTY_Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["NIFTY_Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    
    # 5. Bollinger Bands (20 days, 2 std)
    rolling_mean_20 = df["NIFTY_Close"].rolling(window=20).mean()
    rolling_std_20 = df["NIFTY_Close"].rolling(window=20).std()
    df["BB_Middle"] = rolling_mean_20
    df["BB_High"] = rolling_mean_20 + (2 * rolling_std_20)
    df["BB_Low"] = rolling_mean_20 - (2 * rolling_std_20)
    df["Price_vs_BB"] = (df["NIFTY_Close"] - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"] + 1e-9)
    
    # 6. Average True Range (ATR 14)
    tr1 = df["NIFTY_High"] - df["NIFTY_Low"]
    tr2 = (df["NIFTY_High"] - df["NIFTY_Close"].shift(1)).abs()
    tr3 = (df["NIFTY_Low"] - df["NIFTY_Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()
    df["ATR_Percent"] = df["ATR_14"] / df["NIFTY_Close"]
    
    # 7. Rate of Change (ROC 10)
    df["ROC_10"] = ((df["NIFTY_Close"] - df["NIFTY_Close"].shift(10)) / df["NIFTY_Close"].shift(10)) * 100
    
    # 8. Rolling Volatility (20 days)
    if "NIFTY_Return" not in df.columns:
        df["NIFTY_Return"] = df["NIFTY_Close"].pct_change()
    df["Rolling_Volatility_20"] = df["NIFTY_Return"].rolling(window=20).std()
    
    return df
