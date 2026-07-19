"""
debug_run.py — SmartTraderBot Debug Harness (نسخه پروژه جدید)
================================================================

اجرای پایپ‌لاین با لاگ کامل تا مشخص شود چرا سیگنالی صادر/رد می‌شود.

اجرا:
    python debug_run.py
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
import yfinance as yf

from config import SYMBOL, INTERVAL, PERIOD
from pattern_detector import DetectorConfig, detect
from swing_detector import SwingConfig, get_swings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_data(symbol=SYMBOL, interval=INTERVAL, period=PERIOD) -> pd.DataFrame:
    print(f"\n>>> دریافت داده {symbol} | {interval} | {period} ...")
    raw = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("داده‌ای دریافت نشد.")
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw = raw[["open", "high", "low", "close"]].dropna().reset_index()
    raw.rename(columns={raw.columns[0]: "time"}, inplace=True)
    print(f">>> تعداد کندل: {len(raw)}")
    return raw


def main() -> None:
    df = get_data()

    print("\n" + "=" * 70)
    print(" SWINGS")
    print("=" * 70)
    swing_cfg = SwingConfig()
    swings = get_swings(df, swing_cfg)
    print(f"تعداد swing: {len(swings)}")
    for sw in swings:
        print(f"  idx={sw.index:<5} type={sw.swing_type.value:<5} price={sw.price:<10.2f} strength={sw.strength}")

    print("\n" + "=" * 70)
    print(" PATTERN DETECTION (debug=True)")
    print("=" * 70)
    cfg = DetectorConfig(swing_cfg=swing_cfg, debug=True)
    signals = detect(df, cfg)

    print("\n" + "=" * 70)
    print(" نتیجه نهایی")
    print("=" * 70)
    if not signals:
        print("هیچ سیگنالی صادر نشد. به خطوط STATE/REJECT بالا نگاه کنید.")
    else:
        for i, sig in enumerate(signals, 1):
            print(f"\n>> سیگنال {i}: entry={sig.entry:.2f} sl={sig.sl:.2f} tp1={sig.tp1:.2f} tp2={sig.tp2:.2f} rr={sig.risk_reward:.2f}")


if __name__ == "__main__":
    main()
