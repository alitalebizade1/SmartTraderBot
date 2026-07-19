"""Entry point for the SmartTraderBot strategy pipeline."""
from __future__ import annotations

import webbrowser
from pathlib import Path

import pandas as pd

from backtest import Backtester
from chart import draw_chart, get_data, save_chart
from config import CHART_OUTPUT_HTML
from strategy import _print_report, run


def main() -> None:
    df = get_data()
    signals = run(df)
    _print_report(signals)
    if signals:
        bt = Backtester()
        report = bt.run(df, signals)
        bt.print_report(report)
        bt.export_csv(report, "backtest_results.csv")
    fig = draw_chart(df, signals)
    out_path = Path(CHART_OUTPUT_HTML).resolve()
    save_chart(fig, str(out_path))
    print(f"چارت ساخته شد: {out_path}")
    try:
        webbrowser.open(f"file://{out_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
