"""Risk management helpers for the SmartTraderBot strategy."""
from __future__ import annotations

from config import ACCOUNT_BALANCE, RISK_PCT


class RiskManager:
    """Simple risk estimator for signal sizing."""

    def __init__(self, balance: float = ACCOUNT_BALANCE, risk_pct: float = RISK_PCT) -> None:
        self.balance = balance
        self.risk_pct = risk_pct

    def risk_amount(self) -> float:
        return self.balance * self.risk_pct

    def lot_size(self, entry: float, sl: float, pip_value: float = 1.0) -> float:
        distance = abs(entry - sl)
        if distance == 0:
            return 0.0
        return round(self.risk_amount() / (distance * pip_value), 2)
