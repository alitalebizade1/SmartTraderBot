"""
config.py — Run parameters for SmartTraderBot (new project copy).
Kept intentionally small: only the constants main.py needs.
Advanced per-module tuning lives in each module's own *Config dataclass
(DetectorConfig, SwingConfig, ValidatorConfig).
"""
from __future__ import annotations

# Data source (yfinance)
SYMBOL: str = "GC=F"      # Gold futures — closest free yfinance proxy for XAU/USD
INTERVAL: str = "15m"
PERIOD: str = "60d"       # yfinance limit for 15m candles

# Chart output
CHART_OUTPUT_HTML: str = "chart_output.html"
CHART_THEME: str = "plotly_dark"
