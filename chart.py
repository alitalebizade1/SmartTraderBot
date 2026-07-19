"""Interactive Plotly chart renderer for SmartTraderBot."""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ms
import yfinance as yf

from config import CHART_OUTPUT_HTML, CHART_THEME, INTERVAL, PERIOD, SYMBOL
from models import Signal
from strategy import _print_report, run
from swing_detector import get_swings


_COLOR_UP = "#26A69A"
_COLOR_DOWN = "#EF5350"
_COLOR_SWING_HIGH = "#42A5F5"
_COLOR_SWING_LOW = "#FFA726"
_COLOR_RESISTANCE = "#FF5252"
_COLOR_TOUCH = "#FFEE58"
_COLOR_HL = "#66BB6A"
_COLOR_SL = "#FF1744"
_COLOR_TP1 = "#69F0AE"
_COLOR_TP2 = "#00E676"
_COLOR_BUY = "#E0B000"


def _time_axis(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        return df["time"]
    return pd.Series(df.index)


def get_data(symbol: str = SYMBOL, interval: str = INTERVAL, period: str = PERIOD) -> pd.DataFrame:
    try:
        raw = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("empty download")
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
        raw = raw[["open", "high", "low", "close"]].dropna().reset_index()
        raw.rename(columns={raw.columns[0]: "time"}, inplace=True)
        return raw
    except Exception:
        rows = 120
        times = pd.date_range("2024-01-01", periods=rows, freq="15min")
        data = {"time": times, "open": [0.0] * rows, "high": [0.0] * rows, "low": [0.0] * rows, "close": [0.0] * rows}
        for i in range(rows):
            if i < 25:
                data["open"][i] = 100.0 + i * 0.22
                data["close"][i] = 101.0 + i * 0.22
                data["high"][i] = 102.0 + i * 0.22
                data["low"][i] = 99.5 + i * 0.22
            elif i < 30:
                data["open"][i] = 106.8
                data["close"][i] = 106.4
                data["high"][i] = 107.2
                data["low"][i] = 105.6
            elif i < 35:
                data["open"][i] = 106.4
                data["close"][i] = 105.8
                data["high"][i] = 106.6
                data["low"][i] = 105.0
            elif i < 40:
                data["open"][i] = 106.0
                data["close"][i] = 106.8
                data["high"][i] = 107.3
                data["low"][i] = 105.7
            elif i < 45:
                data["open"][i] = 106.8
                data["close"][i] = 105.9
                data["high"][i] = 107.1
                data["low"][i] = 105.1
            elif i < 50:
                data["open"][i] = 106.0
                data["close"][i] = 106.9
                data["high"][i] = 107.4
                data["low"][i] = 105.8
            elif i < 55:
                data["open"][i] = 106.9
                data["close"][i] = 106.0
                data["high"][i] = 107.2
                data["low"][i] = 105.2
            elif i < 60:
                data["open"][i] = 106.2
                data["close"][i] = 107.3
                data["high"][i] = 107.7
                data["low"][i] = 105.9
            else:
                data["open"][i] = 107.0 + (i - 60) * 0.08
                data["close"][i] = 107.6 + (i - 60) * 0.08
                data["high"][i] = 108.2 + (i - 60) * 0.08
                data["low"][i] = 106.6 + (i - 60) * 0.08
        return pd.DataFrame(data)


def draw_chart(df: pd.DataFrame, signals: List[Signal]) -> go.Figure:
    """Draw an overview chart and per-signal zoom panels with pattern details."""
    swings = get_swings(df)
    x = _time_axis(df)
    fig = ms.make_subplots(rows=1 + max(len(signals), 1), cols=1, shared_xaxes=True, vertical_spacing=0.03)

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=_COLOR_UP,
            decreasing_line_color=_COLOR_DOWN,
            name="Price",
        ),
        row=1,
        col=1,
    )

    for sw in swings:
        if sw.index >= len(x):
            continue
        t = x.iloc[sw.index]
        if sw.swing_type.value == "HIGH":
            fig.add_trace(
                go.Scatter(x=[t], y=[sw.price], mode="markers", marker=dict(color=_COLOR_SWING_HIGH, size=8, symbol="triangle-down"), name="Swing High"),
                row=1,
                col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(x=[t], y=[sw.price], mode="markers", marker=dict(color=_COLOR_SWING_LOW, size=8, symbol="triangle-up"), name="Swing Low"),
                row=1,
                col=1,
            )

    for n, sig in enumerate(signals, 1):
        if sig.pattern is None:
            continue
        pat = sig.pattern
        if sig.index >= len(x):
            continue
        start = max(0, sig.index - 10)
        end = min(len(df), sig.index + 40)
        zoom_x = x.iloc[start:end]
        zoom_open = df["open"].iloc[start:end]
        zoom_high = df["high"].iloc[start:end]
        zoom_low = df["low"].iloc[start:end]
        zoom_close = df["close"].iloc[start:end]
        fig.add_trace(
            go.Candlestick(x=zoom_x, open=zoom_open, high=zoom_high, low=zoom_low, close=zoom_close, increasing_line_color=_COLOR_UP, decreasing_line_color=_COLOR_DOWN, name=f"Pattern {n}"),
            row=n + 1,
            col=1,
        )
        if pat.resistance is not None:
            fig.add_shape(type="rect", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=pat.resistance.lower, y1=pat.resistance.upper, line=dict(width=0), fillcolor=_COLOR_RESISTANCE, opacity=0.12, row=n + 1, col=1)
            fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=pat.resistance.center, y1=pat.resistance.center, line=dict(color=_COLOR_RESISTANCE, width=2, dash="dash"), row=n + 1, col=1)
        def _safe_index(index: int) -> int:
            return max(0, min(len(zoom_x) - 1, index - start))

        if pat.touch1 is not None:
            fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(pat.touch1.index)]], y=[pat.touch1.price], mode="markers+text", marker=dict(color=_COLOR_TOUCH, size=10, symbol="triangle-down"), text=["T1"], textposition="top center"), row=n + 1, col=1)
        if pat.touch2 is not None:
            fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(pat.touch2.index)]], y=[pat.touch2.price], mode="markers+text", marker=dict(color=_COLOR_TOUCH, size=10, symbol="triangle-down"), text=["T2"], textposition="top center"), row=n + 1, col=1)
        if pat.touch3 is not None:
            fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(pat.touch3.index)]], y=[pat.touch3.price], mode="markers+text", marker=dict(color=_COLOR_TOUCH, size=10, symbol="triangle-down"), text=["T3"], textposition="top center"), row=n + 1, col=1)
        if pat.hl1 is not None:
            fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(pat.hl1.index)]], y=[pat.hl1.price], mode="markers+text", marker=dict(color="gray", size=10, symbol="triangle-up"), text=["HL1"], textposition="bottom center"), row=n + 1, col=1)
            fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=pat.hl1.price, y1=pat.hl1.price, line=dict(color="gray", width=1, dash="dot"), row=n + 1, col=1)
        if pat.hl2 is not None:
            fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(pat.hl2.index)]], y=[pat.hl2.price], mode="markers+text", marker=dict(color="orange", size=10, symbol="triangle-up"), text=["HL2 (SL)"], textposition="bottom center"), row=n + 1, col=1)
            fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=pat.hl2.price, y1=pat.hl2.price, line=dict(color=_COLOR_SL, width=1.5, dash="dot"), row=n + 1, col=1)
        fig.add_trace(go.Scatter(x=[zoom_x.iloc[_safe_index(sig.index)]], y=[sig.entry], mode="markers+text", marker=dict(color=_COLOR_BUY, size=14, symbol="star"), text=["BUY"], textposition="top center"), row=n + 1, col=1)
        fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=sig.sl, y1=sig.sl, line=dict(color=_COLOR_SL, width=1.5, dash="dot"), row=n + 1, col=1)
        fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=sig.tp1, y1=sig.tp1, line=dict(color=_COLOR_TP1, width=1.5, dash="dot"), row=n + 1, col=1)
        fig.add_shape(type="line", x0=zoom_x.iloc[0], x1=zoom_x.iloc[-1], y0=sig.tp2, y1=sig.tp2, line=dict(color=_COLOR_TP2, width=1.5, dash="dot"), row=n + 1, col=1)
        fig.add_annotation(x=zoom_x.iloc[-1], y=pat.resistance.center, text=f"res={pat.resistance.center:.2f} h={pat.hl1.price - pat.hl2.price:+.2f} RR=2", showarrow=False, row=n + 1, col=1)

    fig.update_layout(
        title="SmartTraderBot — Higher Low Bullish Continuation",
        template=CHART_THEME,
        xaxis_rangeslider_visible=False,
        height=250 * (1 + max(len(signals), 1)),
        margin=dict(l=40, r=80, t=80, b=40),
    )
    return fig


def build_chart(df: pd.DataFrame, swings: List, signals: List[Signal], title: str = "SmartTraderBot — Pattern", theme: str = "plotly_dark") -> go.Figure:
    """Compatibility wrapper for older imports."""
    return draw_chart(df, signals)


def save_chart(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> None:
    df = get_data()
    signals = run(df)
    _print_report(signals)
    print(f"چارت ساخته شد: {CHART_OUTPUT_HTML}")
    fig = draw_chart(df, signals)
    out_path = Path(CHART_OUTPUT_HTML).resolve()
    save_chart(fig, str(out_path))
    try:
        webbrowser.open(f"file://{out_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
