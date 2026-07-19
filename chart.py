"""Interactive Plotly chart renderer for SmartTraderBot."""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from config import CHART_OUTPUT_HTML, CHART_THEME, DEBUG_VISUALIZATION, INTERVAL, PERIOD, SYMBOL
from models import Signal
from strategy import _print_report, run
from swing_detector import get_swings


_COLOR_UP = "#26A69A"
_COLOR_DOWN = "#EF5350"
_COLOR_SWING_HIGH = "#42A5F5"
_COLOR_SWING_LOW = "#FFA726"
_COLOR_RESISTANCE = "#F59E0B"
_COLOR_HL1 = "#3B82F6"
_COLOR_HL2 = "#16A34A"
_COLOR_SL = "#DC2626"
_COLOR_TP = "#10B981"
_COLOR_BUY = "#22C55E"
_COLOR_CHANNEL = "rgba(59, 130, 246, 0.14)"


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


def _add_legend_entry(fig: go.Figure, legend_names: set[str], name: str, color: str, symbol: str = "circle") -> None:
    if name in legend_names:
        return
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=8, symbol=symbol),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    legend_names.add(name)


def _add_signal_annotation(fig: go.Figure, x_values: pd.Series, signal_index: int, text: str, y_value: float, offset_index: int) -> None:
    offsets = [(0, 20), (20, 20), (-20, 20), (0, -20), (20, -20), (-20, -20)]
    x_value = x_values.iloc[signal_index]
    dx, dy = offsets[offset_index % len(offsets)]
    fig.add_annotation(
        x=x_value,
        y=y_value,
        text=text,
        showarrow=True,
        arrowhead=2,
        arrowcolor="rgba(120,120,120,0.75)",
        ax=dx,
        ay=dy,
        xanchor="center",
        yanchor="middle",
        font=dict(size=11, color="#E5E7EB"),
        bgcolor="rgba(17,24,39,0.75)",
        bordercolor="rgba(120,120,120,0.8)",
        borderwidth=1,
        opacity=0.95,
    )


def draw_chart(df: pd.DataFrame, signals: List[Signal]) -> go.Figure:
    """Render a single-candlestick debugging chart with grouped signal overlays and auto-zoom."""
    x = _time_axis(df)
    fig = go.Figure()
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
            hovertemplate="Time: %{x}<br>Open: %{open:.2f}<br>High: %{high:.2f}<br>Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>",
        )
    )

    legend_names: set[str] = {"Price"}
    swings = get_swings(df)
    for sw in swings:
        if sw.index >= len(x):
            continue
        if sw.swing_type.value == "HIGH":
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[sw.index]],
                    y=[sw.price],
                    mode="markers",
                    marker=dict(color=_COLOR_SWING_HIGH, size=8, symbol="triangle-down", line=dict(color="white", width=1)),
                    name="Swing High",
                    showlegend=False,
                    hovertemplate=f"Swing High<br>Price: {sw.price:.2f}<br>Index: {sw.index}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[sw.index]],
                    y=[sw.price],
                    mode="markers",
                    marker=dict(color=_COLOR_SWING_LOW, size=8, symbol="triangle-up", line=dict(color="white", width=1)),
                    name="Swing Low",
                    showlegend=False,
                    hovertemplate=f"Swing Low<br>Price: {sw.price:.2f}<br>Index: {sw.index}<extra></extra>",
                )
            )

    signal_ranges = []
    for sig in signals:
        if sig.pattern is None:
            continue
        if sig.index >= len(x):
            continue
        pat = sig.pattern
        start_idx = max(0, sig.index - 100)
        end_idx = min(len(df) - 1, sig.index + 50)
        signal_ranges.append((start_idx, end_idx))

    if signals:
        visible_start = max(0, min(start for start, _ in signal_ranges))
        visible_end = min(len(df) - 1, max(end for _, end in signal_ranges))
        view_start = x.iloc[visible_start]
        view_end = x.iloc[visible_end]
    else:
        view_start = x.iloc[0]
        view_end = x.iloc[-1]

    for sig in signals:
        if sig.pattern is None:
            continue
        pat = sig.pattern
        if sig.index >= len(x):
            continue

        channel_start = max(0, min([pat.touch1.index if pat.touch1 else sig.index, pat.hl1.index if pat.hl1 else sig.index, pat.hl2.index if pat.hl2 else sig.index, sig.index]) - 2)
        channel_end = sig.index
        channel_low = float(df["low"].iloc[channel_start: channel_end + 1].min())
        channel_high = float(df["high"].iloc[channel_start: channel_end + 1].max())
        fig.add_shape(
            type="rect",
            x0=x.iloc[channel_start],
            x1=x.iloc[channel_end],
            y0=channel_low,
            y1=channel_high,
            fillcolor=_COLOR_CHANNEL,
            line=dict(width=0),
            opacity=0.9,
            layer="below",
        )
        _add_legend_entry(fig, legend_names, "Bullish Channel", _COLOR_CHANNEL, symbol="square")

        resistance_price = pat.resistance.center if pat.resistance is not None else sig.sl
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[sig.index], x.iloc[min(len(df) - 1, visible_end)]],
                y=[resistance_price, resistance_price],
                mode="lines",
                line=dict(color=_COLOR_RESISTANCE, width=2, dash="dash"),
                name="Resistance",
                showlegend=False,
                hovertemplate=f"Resistance<br>Price: {resistance_price:.2f}<br>Confirmed: {x.iloc[sig.index]}<extra></extra>",
            )
        )
        _add_legend_entry(fig, legend_names, "Resistance", _COLOR_RESISTANCE, symbol="diamond")

        if pat.hl1 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[pat.hl1.index]],
                    y=[pat.hl1.price],
                    mode="markers",
                    marker=dict(color=_COLOR_HL1, size=8, symbol="circle", line=dict(color="white", width=1)),
                    name="HL1",
                    showlegend=False,
                    hovertemplate=f"HL1<br>Price: {pat.hl1.price:.2f}<br>Index: {pat.hl1.index}<extra></extra>",
                )
            )
            _add_legend_entry(fig, legend_names, "HL1", _COLOR_HL1, symbol="circle")
        if pat.hl2 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[pat.hl2.index]],
                    y=[pat.hl2.price],
                    mode="markers",
                    marker=dict(color=_COLOR_HL2, size=12, symbol="circle", line=dict(color="white", width=1)),
                    name="HL2",
                    showlegend=False,
                    hovertemplate=f"HL2<br>Price: {pat.hl2.price:.2f}<br>Index: {pat.hl2.index}<extra></extra>",
                )
            )
            _add_legend_entry(fig, legend_names, "HL2", _COLOR_HL2, symbol="circle")

        fig.add_trace(
            go.Scatter(
                x=[x.iloc[sig.index], x.iloc[min(len(df) - 1, visible_end)]],
                y=[sig.sl, sig.sl],
                mode="lines",
                line=dict(color=_COLOR_SL, width=2, dash="solid"),
                name="SL",
                showlegend=False,
                hovertemplate=f"SL<br>Price: {sig.sl:.2f}<br>Risk: {sig.risk:.2f}<br>Distance: {sig.entry - sig.sl:.2f}<extra></extra>",
            )
        )
        _add_legend_entry(fig, legend_names, "SL", _COLOR_SL, symbol="triangle-down")

        fig.add_trace(
            go.Scatter(
                x=[x.iloc[sig.index], x.iloc[min(len(df) - 1, visible_end)]],
                y=[sig.tp2, sig.tp2],
                mode="lines",
                line=dict(color=_COLOR_TP, width=2, dash="solid"),
                name="TP",
                showlegend=False,
                hovertemplate=f"TP<br>Price: {sig.tp2:.2f}<br>RR: {sig.risk_reward:.2f}<br>Distance: {sig.tp2 - sig.entry:.2f}<extra></extra>",
            )
        )
        _add_legend_entry(fig, legend_names, "TP", _COLOR_TP, symbol="diamond")

        fig.add_trace(
            go.Scatter(
                x=[x.iloc[sig.index]],
                y=[sig.entry],
                mode="markers",
                marker=dict(color=_COLOR_BUY, size=16, symbol="triangle-up", line=dict(color="white", width=2)),
                name="BUY",
                showlegend=False,
                hovertemplate=f"BUY<br>Signal ID: {sig.pattern_id}<br>Time: {x.iloc[sig.index]}<br>Entry: {sig.entry:.2f}<br>SL: {sig.sl:.2f}<br>TP: {sig.tp2:.2f}<br>RR: {sig.risk_reward:.2f}<br>Resistance: {resistance_price:.2f}<br>HL1: {pat.hl1.price if pat.hl1 else 'n/a'}<br>HL2: {pat.hl2.price if pat.hl2 else 'n/a'}<br>ATR: {sig.pattern.atr_at_signal if sig.pattern else 'n/a'}<br>Pattern duration: {sig.index - (pat.touch1.index if pat.touch1 else sig.index)}<extra></extra>",
            )
        )
        _add_legend_entry(fig, legend_names, "BUY", _COLOR_BUY, symbol="triangle-up")

        if pat.hl1 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[pat.hl1.index]],
                    y=[pat.hl1.price],
                    mode="markers+text",
                    text=["HL1"],
                    textposition="top center",
                    marker=dict(color=_COLOR_HL1, size=10, symbol="circle", line=dict(color="white", width=1)),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if pat.hl2 is not None:
            fig.add_trace(
                go.Scatter(
                    x=[x.iloc[pat.hl2.index]],
                    y=[pat.hl2.price],
                    mode="markers+text",
                    text=["HL2"],
                    textposition="bottom center",
                    marker=dict(color=_COLOR_HL2, size=14, symbol="circle", line=dict(color="white", width=1)),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        _add_signal_annotation(fig, x, sig.index, f"Pattern {sig.pattern_id}", resistance_price, len(signals) - 1)

    fig.update_layout(
        title="SmartTraderBot — Local Pattern Debug View",
        template=CHART_THEME,
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=70, b=40),
        height=700,
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0.01),
        xaxis=dict(rangeslider_visible=False, showgrid=True, gridcolor="rgba(255,255,255,0.12)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.12)"),
    )
    if signals:
        fig.update_xaxes(range=[view_start, view_end])
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
