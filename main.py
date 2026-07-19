"""
main.py — SmartTraderBot (New Project)
Pipeline: داده → Swing Detection → Pattern Detection → گزارش + چارت HTML
اجرا: python3 main.py
"""
from __future__ import annotations
import webbrowser
from pathlib import Path
import pandas as pd
import yfinance as yf
from config import SYMBOL, INTERVAL, PERIOD, CHART_OUTPUT_HTML, CHART_THEME
from pattern_detector import DetectorConfig, detect
from swing_detector import SwingConfig, get_swings
from chart import build_chart, save_chart
from report import print_report

def get_data(symbol=SYMBOL, interval=INTERVAL, period=PERIOD):
    print(f"دریافت داده {symbol} - {interval} ({period}) ...")
    raw = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("داده‌ای دریافت نشد.")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw = raw[["open","high","low","close"]].dropna().reset_index()
    raw.rename(columns={raw.columns[0]: "time"}, inplace=True)
    print(f"تعداد کندل: {len(raw)}")
    return raw

def main():
    print("=" * 72)
    print(" SmartTraderBot — 3-Touch Resistance / Rising-Bottom Strategy")
    print("=" * 72)
    df = get_data()
    swing_cfg = SwingConfig()
    swings = get_swings(df, swing_cfg)
    cfg = DetectorConfig(swing_cfg=swing_cfg)
    signals = detect(df, cfg)
    print_report(signals, swings_count=len(swings), candles_count=len(df))
    fig = build_chart(df, swings, signals, theme=CHART_THEME)
    out_path = Path(CHART_OUTPUT_HTML).resolve()
    save_chart(fig, str(out_path))
    print(f"\n چارت ذخیره شد: {out_path}")
    try:
        webbrowser.open(f"file://{out_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
