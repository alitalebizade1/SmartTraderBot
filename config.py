"""Configuration constants for the SmartTraderBot strategy pipeline."""
from __future__ import annotations

# Data source (yfinance)
SYMBOL: str = "GC=F"
INTERVAL: str = "15m"
PERIOD: str = "15d"

# Strategy constants
ATR_PERIOD: int = 14
STRONG_CANDLE_ATR: float = 0.6
MIN_STRONG_CANDLES: int = 3
ATR_RES: float = 0.6
MAX_BARS_WAIT: int = 60
MAX_SIGNAL_WAIT: int = 20
MAX_TOUCHES: int = 5
MAX_BARS_AFTER_UPTREND: int = 10
MAX_HL1_DISTANCE: int = 5
MAX_RETEST_DISTANCE: int = 5
MAX_HL2_DISTANCE: int = 5
MAX_PATTERN_BARS: int = 20
ACCOUNT_BALANCE: float = 1000.0
RISK_PCT: float = 0.01
REWARD_RATIO: float = 2.0

# Chart output
CHART_OUTPUT_HTML: str = "chart_output.html"
CHART_THEME: str = "plotly_dark"
DEBUG_VISUALIZATION: bool = True
