"""
indicators.py — Shared technical indicators.
(unchanged from existing project version)
"""
import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, period: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    return series.rolling(period).mean()


def trend_direction(df):
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)
    if ema20.iloc[-1] > ema50.iloc[-1]:
        return "UPTREND"
    if ema20.iloc[-1] < ema50.iloc[-1]:
        return "DOWNTREND"
    return "RANGE"


def body_size(df):
    return (df["close"] - df["open"]).abs()


def candle_range(df):
    return df["high"] - df["low"]


def bullish(df, idx):
    return df["close"].iloc[idx] > df["open"].iloc[idx]


def bearish(df, idx):
    return df["close"].iloc[idx] < df["open"].iloc[idx]
