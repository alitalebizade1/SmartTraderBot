"""
chart.py — Interactive Plotly chart renderer for SmartTraderBot.

Renders:
    - OHLC candlesticks
    - Detected swing points (HIGH/LOW markers)
    - For each emitted BUY signal: the full pattern anatomy —
        * channel start (impulse origin, dashed vertical marker)
        * resistance zone (horizontal band across the 3 touches)
        * touch 1 / touch 2 / touch 3 markers
        * HL1 / HL2 markers (rising bottoms)
        * Entry / Stop-Loss / TP1 / TP2 horizontal lines from the
          signal bar onward
        * A "BUY" marker at the signal bar

No trading logic lives here — pure visualisation of already-computed
Pattern / Signal objects.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from models import Signal


_COLOR_UP = "#26A69A"
_COLOR_DOWN = "#EF5350"
_COLOR_SWING_HIGH = "#42A5F5"
_COLOR_SWING_LOW = "#FFA726"
_COLOR_RESISTANCE = "#FF5252"
_COLOR_CHANNEL_START = "#AB47BC"
_COLOR_TOUCH = "#FFEE58"
_COLOR_HL = "#66BB6A"
_COLOR_ENTRY = "#FFFFFF"
_COLOR_SL = "#FF1744"
_COLOR_TP1 = "#69F0AE"
_COLOR_TP2 = "#00E676"
_COLOR_BUY = "#00E676"


def _time_axis(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        return df["time"]
    return pd.Series(df.index)


def build_chart(
    df: pd.DataFrame,
    swings: List,
    signals: List[Signal],
    title: str = "SmartTraderBot — 3-Touch Resistance / Rising-Bottom Pattern",
    theme: str = "plotly_dark",
) -> go.Figure:
    x = _time_axis(df)

    fig = go.Figure()

    # ── Candles ──────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=_COLOR_UP,
        decreasing_line_color=_COLOR_DOWN,
        name="Price",
    ))

    # ── Swings (context) ─────────────────────────────────
    highs_x, highs_y, lows_x, lows_y = [], [], [], []
    for sw in swings:
        t = x.iloc[sw.index] if sw.index < len(x) else None
        if t is None:
            continue
        if sw.swing_type.value == "HIGH":
            highs_x.append(t); highs_y.append(sw.price)
        else:
            lows_x.append(t); lows_y.append(sw.price)

    fig.add_trace(go.Scatter(
        x=highs_x, y=highs_y, mode="markers", name="Swing High",
        marker=dict(color=_COLOR_SWING_HIGH, size=6, symbol="triangle-down"),
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=lows_x, y=lows_y, mode="markers", name="Swing Low",
        marker=dict(color=_COLOR_SWING_LOW, size=6, symbol="triangle-up"),
        opacity=0.55,
    ))

    # ── Per-signal pattern anatomy ────────────────────────
    for n, sig in enumerate(signals, 1):
        pat = sig.pattern
        if pat is None:
            continue

        res = pat.resistance

        # Channel start marker (impulse origin — the purple region anchor)
        if pat.channel_start_bar is not None and pat.channel_start_bar < len(x):
            fig.add_trace(go.Scatter(
                x=[x.iloc[pat.channel_start_bar]],
                y=[pat.channel_start_price],
                mode="markers+text",
                marker=dict(color=_COLOR_CHANNEL_START, size=11, symbol="star"),
                text=[f"شروع کانال #{n}"],
                textposition="bottom center",
                name=f"Channel Start #{n}",
                showlegend=(n == 1),
            ))

        # Resistance zone line spanning touch1 -> signal bar
        if res is not None:
            start_i = pat.touch1.index if pat.touch1 else sig.index
            end_i = sig.index
            fig.add_shape(
                type="line",
                x0=x.iloc[start_i], x1=x.iloc[min(end_i, len(x) - 1)],
                y0=res.center, y1=res.center,
                line=dict(color=_COLOR_RESISTANCE, width=2, dash="dot"),
            )
            fig.add_annotation(
                x=x.iloc[min(end_i, len(x) - 1)], y=res.center,
                text=f"مقاومت #{n}", showarrow=False,
                font=dict(color=_COLOR_RESISTANCE, size=11),
                xanchor="left",
            )

        # Touch markers (1, 2, 3)
        for label, sw in (("Touch 1", pat.touch1), ("Touch 2", pat.touch2), ("Touch 3", pat.touch3)):
            if sw is None or sw.index >= len(x):
                continue
            fig.add_trace(go.Scatter(
                x=[x.iloc[sw.index]], y=[sw.price],
                mode="markers+text",
                marker=dict(color=_COLOR_TOUCH, size=10, symbol="circle"),
                text=[label], textposition="top center",
                name=label if n == 1 else None,
                showlegend=(n == 1),
            ))

        # HL markers (rising bottoms)
        for label, sw in (("HL1", pat.hl1), ("HL2 (SL)", pat.hl2)):
            if sw is None or sw.index >= len(x):
                continue
            fig.add_trace(go.Scatter(
                x=[x.iloc[sw.index]], y=[sw.price],
                mode="markers+text",
                marker=dict(color=_COLOR_HL, size=10, symbol="diamond"),
                text=[label], textposition="bottom center",
                name=label if n == 1 else None,
                showlegend=(n == 1),
            ))

        # Entry / SL / TP1 / TP2 lines from the signal bar to the end of the chart
        sig_i = sig.index
        if sig_i < len(x):
            end_x = x.iloc[-1]
            start_x = x.iloc[sig_i]
            level_defs = [
                ("Entry", sig.entry, _COLOR_ENTRY, "solid"),
                ("Stop Loss (HL2)", sig.sl, _COLOR_SL, "dash"),
                ("TP1 (مقاومت)", sig.tp1, _COLOR_TP1, "dash"),
                ("TP2 (Measured Move)", sig.tp2, _COLOR_TP2, "dash"),
            ]
            for label, level, color, dash in level_defs:
                fig.add_shape(
                    type="line",
                    x0=start_x, x1=end_x, y0=level, y1=level,
                    line=dict(color=color, width=1.5, dash=dash),
                )
                fig.add_annotation(
                    x=end_x, y=level, text=f"{label}: {level:.2f}",
                    showarrow=False, xanchor="left",
                    font=dict(color=color, size=10),
                )

            # BUY marker
            fig.add_trace(go.Scatter(
                x=[start_x], y=[sig.entry],
                mode="markers+text",
                marker=dict(color=_COLOR_BUY, size=16, symbol="triangle-up"),
                text=[f"BUY #{n}\nConf {sig.confidence:.0%}"],
                textposition="middle right",
                name=f"BUY Signal #{n}",
            ))

    fig.update_layout(
        title=title,
        template=theme,
        xaxis_rangeslider_visible=False,
        height=850,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=160, t=80, b=40),
    )

    return fig


def save_chart(fig: go.Figure, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn")
