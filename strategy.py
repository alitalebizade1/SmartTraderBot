"""Strategy pipeline for the Higher Low Bullish Continuation signal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from config import (
    ATR_PERIOD,
    ATR_RES,
    MAX_BARS_AFTER_UPTREND,
    MAX_HL1_DISTANCE,
    MAX_HL2_DISTANCE,
    MAX_PATTERN_BARS,
    MAX_RETEST_DISTANCE,
    MAX_SIGNAL_WAIT,
    MIN_STRONG_CANDLES,
    STRONG_CANDLE_ATR,
)
from indicators import atr
from models import Pattern, Resistance, Signal, SignalType, Swing, SwingType


@dataclass
class CandidateBullishChannel:
    """Temporary parent structure for a single bullish channel."""

    id: str
    start_idx: int
    end_idx: int
    peak_price: float
    last_high: float
    last_low: float
    confirmed: bool = False


@dataclass
class CandidateResistance:
    """Temporary resistance candidate inside the current bullish channel."""

    price: float
    index: int
    time: object
    channel_id: str
    confirmed: bool = False


@dataclass
class CandidateHL1:
    """Temporary HL1 candidate inside the current bullish channel."""

    price: float
    index: int
    time: object
    channel_id: str
    confirmed: bool = False


@dataclass
class CandidateHL2:
    """Temporary HL2 candidate inside the current bullish channel."""

    price: float
    index: int
    time: object
    channel_id: str
    confirmed: bool = False


class StrategyConfig:
    """Configuration object for the strategy state machine."""

    def __init__(self) -> None:
        self.atr_period = ATR_PERIOD
        self.strong_candle_atr = STRONG_CANDLE_ATR
        self.min_strong_candles = MIN_STRONG_CANDLES
        self.atr_res = ATR_RES
        self.max_signal_wait = MAX_SIGNAL_WAIT
        self.max_bars_after_uptrend = MAX_BARS_AFTER_UPTREND
        self.max_hl1_distance = MAX_HL1_DISTANCE
        self.max_retest_distance = MAX_RETEST_DISTANCE
        self.max_hl2_distance = MAX_HL2_DISTANCE
        self.max_pattern_bars = MAX_PATTERN_BARS


def _is_strong_channel_candle(row: pd.Series, prev_row: pd.Series) -> bool:
    return (
        float(row["close"]) > float(row["open"])
        and float(row["close"]) > float(prev_row["close"])
        and float(row["high"]) > float(prev_row["high"])
        and float(row["low"]) > float(prev_row["low"])
    )


def _is_local_rejection(row: pd.Series, prev_row: pd.Series, resistance_price: float, tolerance: float) -> bool:
    return (
        float(row["close"]) < float(prev_row["close"])
        and float(row["high"]) <= resistance_price + tolerance
        and float(row["high"]) >= resistance_price - tolerance
    )


def _local_resistance_price(row: pd.Series, prev_row: Optional[pd.Series], tolerance: float) -> float:
    if prev_row is None:
        return float(row["high"])
    return max(float(prev_row["high"]), float(row["high"]))


def _build_local_swing(index: int, price: float, candle_time: object, swing_type: SwingType) -> Swing:
    return Swing(
        index=index,
        time=candle_time,
        price=float(price),
        swing_type=swing_type,
        strength="NORMAL",
        atr=1.0,
        left_strength=0.5,
        right_strength=0.5,
        valid=True,
    )


def _build_pattern(
    *,
    touch1: Swing,
    touch2: Swing,
    touch3: Swing,
    hl1: Swing,
    hl2: Swing,
    resistance: float,
    resistance_tol: float,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    risk: float,
    reward: float,
    signal_index: int,
    candle_time: object,
    pattern_id: str,
) -> Tuple[Pattern, Signal]:
    res = Resistance(
        center=resistance,
        upper=resistance + resistance_tol,
        lower=resistance - resistance_tol,
        width=resistance_tol * 2,
        touches=[touch1.index, touch2.index, touch3.index],
        strength=100.0,
    )
    pattern = Pattern(
        id=pattern_id,
        state="SIGNAL",
        confirmed=True,
        score=100.0,
        confidence=1.0,
        trend=None,
        resistance=res,
        touch1=touch1,
        touch2=touch2,
        touch3=touch3,
        hl1=hl1,
        hl2=hl2,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk=risk,
        reward=reward,
    )
    signal = Signal(
        symbol="GC=F",
        timeframe="15m",
        entry=entry,
        sl=stop_loss,
        tp1=tp1,
        tp2=tp2,
        risk_reward=2.0,
        signal_type=SignalType.BUY,
        confidence=1.0,
        pattern_id=pattern_id,
        index=signal_index,
        candle_time=candle_time,
        risk=risk,
        reward=reward,
        pattern=pattern,
    )
    return pattern, signal


def _run_state_machine(df: pd.DataFrame, atr: pd.Series) -> List[Signal]:
    """Run a single-pass, event-driven state machine for local bullish continuation patterns."""
    cfg = StrategyConfig()
    signals: List[Signal] = []
    seen_ids = set()

    state = "SEARCH_CHANNEL"
    bullish_run = 0
    channel: Optional[CandidateBullishChannel] = None
    resistance: Optional[CandidateResistance] = None
    hl1: Optional[CandidateHL1] = None
    hl2: Optional[CandidateHL2] = None

    for idx in range(len(df)):
        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else None
        atr_val = float(atr.iloc[idx]) if idx < len(atr) else 1.0
        tolerance = max(atr_val * cfg.atr_res, 0.5)

        if state == "SEARCH_CHANNEL":
            if prev_row is None:
                continue
            if _is_strong_channel_candle(row, prev_row):
                bullish_run += 1
                if bullish_run >= cfg.min_strong_candles:
                    start_idx = idx - bullish_run + 1
                    channel = CandidateBullishChannel(
                        id=f"ch-{start_idx}-{idx}",
                        start_idx=start_idx,
                        end_idx=idx,
                        peak_price=float(df["high"].iloc[start_idx: idx + 1].max()),
                        last_high=float(prev_row["high"]),
                        last_low=float(prev_row["low"]),
                    )
                    state = "WAIT_RESISTANCE"
                    resistance = None
                    hl1 = None
                    hl2 = None
                    bullish_run = 0
            else:
                bullish_run = 0
            continue

        if channel is None:
            state = "SEARCH_CHANNEL"
            continue

        if idx - channel.start_idx > cfg.max_pattern_bars:
            state = "SEARCH_CHANNEL"
            channel = None
            resistance = None
            hl1 = None
            hl2 = None
            continue

        if state == "WAIT_RESISTANCE":
            if prev_row is None:
                continue
            resistance_price = _local_resistance_price(row, prev_row, tolerance)
            if row["close"] < prev_row["close"] and row["high"] <= resistance_price + tolerance:
                resistance = CandidateResistance(
                    price=resistance_price,
                    index=idx,
                    time=df["time"].iloc[idx],
                    channel_id=channel.id,
                )
                state = "WAIT_HL1"
                continue
            if idx - channel.end_idx > cfg.max_bars_after_uptrend:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            continue

        if state == "WAIT_HL1":
            if resistance is None:
                state = "SEARCH_CHANNEL"
                channel = None
                continue
            if idx - resistance.index > cfg.max_hl1_distance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if row["high"] > resistance.price + tolerance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if prev_row is None:
                continue
            if row["low"] < prev_row["low"] and row["low"] < resistance.price:
                hl1 = CandidateHL1(
                    price=float(row["low"]),
                    index=idx,
                    time=df["time"].iloc[idx],
                    channel_id=channel.id,
                )
                state = "WAIT_RETEST"
            continue

        if state == "WAIT_RETEST":
            if resistance is None or hl1 is None:
                state = "SEARCH_CHANNEL"
                channel = None
                continue
            if idx - resistance.index > cfg.max_retest_distance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if row["high"] > resistance.price + tolerance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if _is_local_rejection(row, prev_row, resistance.price, tolerance):
                resistance.confirmed = True
                state = "WAIT_HL2"
            continue

        if state == "WAIT_HL2":
            if resistance is None or hl1 is None:
                state = "SEARCH_CHANNEL"
                channel = None
                continue
            if idx - hl1.index > cfg.max_hl2_distance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if row["high"] > resistance.price + tolerance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if prev_row is None:
                continue
            if row["low"] <= prev_row["low"] and row["low"] > hl1.price:
                hl2 = CandidateHL2(
                    price=float(row["low"]),
                    index=idx,
                    time=df["time"].iloc[idx],
                    channel_id=channel.id,
                )
                if hl2.price <= hl1.price:
                    state = "SEARCH_CHANNEL"
                    channel = None
                    resistance = None
                    hl1 = None
                    hl2 = None
                    continue
                state = "WAIT_SIGNAL"
            continue

        if state == "WAIT_SIGNAL":
            if resistance is None or hl1 is None or hl2 is None:
                state = "SEARCH_CHANNEL"
                channel = None
                continue
            if idx - hl2.index > cfg.max_signal_wait:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            if row["close"] <= row["open"]:
                continue
            if row["low"] <= hl2.price:
                continue
            if row["close"] > resistance.price + tolerance:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            entry = float(row["close"])
            stop_loss = float(hl2.price)
            h = resistance.price - hl1.price
            tp1 = entry + h
            tp2 = entry + h * 2.0
            risk = entry - stop_loss
            reward = risk * 2.0
            if risk <= 0:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            pattern_id = f"{channel.id}-{resistance.index}-{hl1.index}-{hl2.index}-{idx}"
            if pattern_id in seen_ids:
                state = "SEARCH_CHANNEL"
                channel = None
                resistance = None
                hl1 = None
                hl2 = None
                continue
            seen_ids.add(pattern_id)
            touch1 = _build_local_swing(resistance.index, resistance.price, df["time"].iloc[resistance.index], SwingType.HIGH)
            touch2 = _build_local_swing(idx, resistance.price, df["time"].iloc[idx], SwingType.HIGH)
            touch3 = _build_local_swing(idx, float(row["high"]), df["time"].iloc[idx], SwingType.HIGH)
            hl1_swing = _build_local_swing(hl1.index, hl1.price, df["time"].iloc[hl1.index], SwingType.LOW)
            hl2_swing = _build_local_swing(hl2.index, hl2.price, df["time"].iloc[hl2.index], SwingType.LOW)
            pattern, signal = _build_pattern(
                touch1=touch1,
                touch2=touch2,
                touch3=touch3,
                hl1=hl1_swing,
                hl2=hl2_swing,
                resistance=resistance.price,
                resistance_tol=tolerance,
                entry=entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk=risk,
                reward=reward,
                signal_index=idx,
                candle_time=df["time"].iloc[idx],
                pattern_id=pattern_id,
            )
            signals.append(signal)
            return sorted(signals, key=lambda s: s.index)

    return sorted(signals, key=lambda s: s.index)


def run(df: pd.DataFrame) -> List[Signal]:
    """Run the strategy end to end on the supplied OHLC data."""
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("DataFrame must contain open/high/low/close columns")
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a time column")
    atr_series = atr(df, ATR_PERIOD)
    return _run_state_machine(df, atr_series)


def detect(df: pd.DataFrame) -> List[Signal]:
    """Compatibility wrapper for older detector imports."""
    return run(df)


def _print_report(signals: List[Signal]) -> None:
    """Print a concise text summary of the generated signals."""
    print(f"{len(signals)} signals found with details")
    for signal in signals:
        pattern = signal.pattern
        print(
            f"signal idx={signal.index} entry={signal.entry:.2f} sl={signal.sl:.2f} tp1={signal.tp1:.2f} tp2={signal.tp2:.2f} rr={signal.risk_reward:.2f}"
        )
        if pattern is not None:
            print(
                f"  resistance={pattern.resistance.price:.2f} hl1={pattern.hl1.price:.2f} hl2={pattern.hl2.price:.2f}"
            )
