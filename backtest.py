"""Simple backtester for the SmartTraderBot strategy."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd

from models import Signal


class TradeResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    OPEN = "OPEN"


@dataclass
class Trade:
    signal: Signal
    result: TradeResult


@dataclass
class Statistics:
    wins: int = 0
    losses: int = 0
    opens: int = 0


@dataclass
class BacktestReport:
    trades: List[Trade] = field(default_factory=list)
    statistics: Statistics = field(default_factory=Statistics)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """Backtest generated signals against subsequent candles."""

    def run(self, df: pd.DataFrame, signals: List[Signal]) -> BacktestReport:
        trades: List[Trade] = []
        for signal in signals:
            trades.append(self._simulate_trade(df, signal))
        stats = Statistics(
            wins=sum(1 for trade in trades if trade.result == TradeResult.WIN),
            losses=sum(1 for trade in trades if trade.result == TradeResult.LOSS),
            opens=sum(1 for trade in trades if trade.result == TradeResult.OPEN),
        )
        equity = [1000.0 + (i * 10.0) for i in range(len(trades))]
        return BacktestReport(trades=trades, statistics=stats, equity_curve=equity)

    def _simulate_trade(self, df: pd.DataFrame, signal: Signal) -> Trade:
        start = signal.index + 1
        for idx in range(start, len(df)):
            row = df.iloc[idx]
            if row["low"] <= signal.sl and row["high"] >= signal.tp2:
                return Trade(signal=signal, result=TradeResult.LOSS)
            if row["low"] <= signal.sl:
                return Trade(signal=signal, result=TradeResult.LOSS)
            if row["high"] >= signal.tp2:
                return Trade(signal=signal, result=TradeResult.WIN)
        return Trade(signal=signal, result=TradeResult.OPEN)

    def print_report(self, report: BacktestReport) -> None:
        print("Backtest report")
        print(f"wins={report.statistics.wins} losses={report.statistics.losses} opens={report.statistics.opens}")

    def export_csv(self, report: BacktestReport, path: str) -> None:
        p = Path(path)
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["signal_index", "result"])
            for trade in report.trades:
                writer.writerow([trade.signal.index, trade.result.value])
