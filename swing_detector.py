"""
swing_detector.py — Institutional-Grade Swing Detection Engine
==============================================================
(unchanged from the existing project version — kept for compatibility)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Final, List, Optional

import numpy as np
import pandas as pd

from models import Swing, SwingType

_log = logging.getLogger(__name__)


class SwingStrength(str, Enum):
    WEAK = "WEAK"
    NORMAL = "NORMAL"
    STRONG = "STRONG"


@dataclass(frozen=True)
class SwingConfig:
    atr_period: int = 14
    atr_multiplier: float = 1.0
    pivot_window_min: int = 2
    pivot_window_max: int = 6
    min_candles_between: int = 3
    min_move_atr: float = 0.5
    merge_threshold_atr: float = 0.4
    strength_strong_atr: float = 2.0
    strength_normal_atr: float = 1.0
    rejection_wick_ratio: float = 0.30
    debug: bool = False


@dataclass
class _RawPivot:
    index: int
    price: float
    pivot_type: SwingType
    atr: float
    left_bars: int
    right_bars: int
    move_size: float
    wick_ratio: float
    valid: bool = True


def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _adaptive_window(atr_val: float, price: float, cfg: SwingConfig) -> int:
    if price <= 0 or atr_val <= 0:
        return cfg.pivot_window_min
    volatility_pct = atr_val / price
    raw = cfg.pivot_window_min + int(
        (cfg.pivot_window_max - cfg.pivot_window_min) * (1.0 - min(volatility_pct * 100, 1.0))
    )
    return max(cfg.pivot_window_min, min(cfg.pivot_window_max, raw))


def _wick_ratio(row: pd.Series, pivot_type: SwingType) -> float:
    total_range = row["high"] - row["low"]
    if total_range <= 0:
        return 0.0
    if pivot_type == SwingType.HIGH:
        wick = row["high"] - max(row["open"], row["close"])
    else:
        wick = min(row["open"], row["close"]) - row["low"]
    return max(0.0, wick / total_range)


def _scan_raw_pivots(df: pd.DataFrame, atr: pd.Series, cfg: SwingConfig) -> List[_RawPivot]:
    pivots: List[_RawPivot] = []
    n = len(df)
    for i in range(cfg.pivot_window_max, n - cfg.pivot_window_max):
        atr_val = float(atr.iloc[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        price = float(df["close"].iloc[i])
        window = _adaptive_window(atr_val, price, cfg)
        hi = float(df["high"].iloc[i]); lo = float(df["low"].iloc[i])
        row = df.iloc[i]

        left_highs = df["high"].iloc[i - window: i]
        right_highs = df["high"].iloc[i + 1: i + window + 1]
        left_lows = df["low"].iloc[i - window: i]
        right_lows = df["low"].iloc[i + 1: i + window + 1]

        if (hi > left_highs.max() and hi >= right_highs.max()
                and hi > float(df["close"].iloc[i - 1]) and hi > float(df["close"].iloc[i + 1])):
            wr = _wick_ratio(row, SwingType.HIGH)
            pivots.append(_RawPivot(index=i, price=hi, pivot_type=SwingType.HIGH, atr=atr_val,
                                     left_bars=window, right_bars=window, move_size=0.0, wick_ratio=wr))

        if (lo < left_lows.min() and lo <= right_lows.min()
                and lo < float(df["close"].iloc[i - 1]) and lo < float(df["close"].iloc[i + 1])):
            wr = _wick_ratio(row, SwingType.LOW)
            pivots.append(_RawPivot(index=i, price=lo, pivot_type=SwingType.LOW, atr=atr_val,
                                     left_bars=window, right_bars=window, move_size=0.0, wick_ratio=wr))

    pivots.sort(key=lambda p: p.index)
    return pivots


def _fill_move_sizes(pivots: List[_RawPivot]) -> None:
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    for p in pivots:
        if p.pivot_type == SwingType.HIGH:
            if last_low is not None:
                p.move_size = abs(p.price - last_low)
            last_high = p.price
        else:
            if last_high is not None:
                p.move_size = abs(p.price - last_high)
            last_low = p.price


def _apply_noise_filter(pivots: List[_RawPivot], cfg: SwingConfig) -> List[_RawPivot]:
    filtered: List[_RawPivot] = []
    last_high_idx = -9999
    last_low_idx = -9999
    for p in pivots:
        if p.move_size < p.atr * cfg.min_move_atr:
            continue
        if p.pivot_type == SwingType.HIGH:
            if (p.index - last_high_idx) < cfg.min_candles_between:
                continue
            last_high_idx = p.index
        else:
            if (p.index - last_low_idx) < cfg.min_candles_between:
                continue
            last_low_idx = p.index
        filtered.append(p)
    return filtered


def _enforce_alternation(pivots: List[_RawPivot]) -> List[_RawPivot]:
    if not pivots:
        return []
    result: List[_RawPivot] = [pivots[0]]
    for current in pivots[1:]:
        last = result[-1]
        if current.pivot_type == last.pivot_type:
            if current.pivot_type == SwingType.HIGH:
                if current.price >= last.price:
                    result[-1] = current
            else:
                if current.price <= last.price:
                    result[-1] = current
        else:
            result.append(current)
    return result


def _merge_close_swings(pivots: List[_RawPivot], cfg: SwingConfig) -> List[_RawPivot]:
    if not pivots:
        return []
    merged: List[_RawPivot] = [pivots[0]]
    for current in pivots[1:]:
        last = merged[-1]
        if current.pivot_type != last.pivot_type:
            merged.append(current)
            continue
        threshold = last.atr * cfg.merge_threshold_atr
        if abs(current.price - last.price) <= threshold:
            if current.move_size >= last.move_size:
                merged[-1] = current
        else:
            merged.append(current)
    return merged


def _score_strength(pivot: _RawPivot, cfg: SwingConfig):
    left_strength = min(pivot.left_bars / cfg.pivot_window_max, 1.0)
    right_strength = min(pivot.right_bars / cfg.pivot_window_max, 1.0)
    move_atr = pivot.move_size / pivot.atr if pivot.atr > 0 else 0.0
    wick_score = min(pivot.wick_ratio / max(cfg.rejection_wick_ratio, 0.01), 1.0)
    move_score = min(move_atr / cfg.strength_strong_atr, 1.0)
    composite = 0.5 * move_score + 0.3 * wick_score + 0.2 * (left_strength + right_strength) / 2.0

    if move_atr >= cfg.strength_strong_atr and composite >= 0.65:
        strength = SwingStrength.STRONG
    elif move_atr >= cfg.strength_normal_atr:
        strength = SwingStrength.NORMAL
    else:
        strength = SwingStrength.WEAK
    return left_strength, right_strength, strength


def _validate(pivot: _RawPivot, cfg: SwingConfig) -> bool:
    if pivot.atr <= 0 or not np.isfinite(pivot.atr):
        return False
    if pivot.price <= 0:
        return False
    if pivot.move_size < pivot.atr * cfg.min_move_atr:
        return False
    return True


def _to_swing(pivot: _RawPivot, df: pd.DataFrame, cfg: SwingConfig) -> Swing:
    left_s, right_s, strength_enum = _score_strength(pivot, cfg)
    time = (
        df.index[pivot.index]
        if hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index)
        else (df["time"].iloc[pivot.index] if "time" in df.columns else pd.Timestamp("1970-01-01"))
    )
    return Swing(
        index=pivot.index, price=pivot.price, time=time, swing_type=pivot.pivot_type,
        strength=strength_enum.value, atr=pivot.atr, left_strength=left_s, right_strength=right_s,
        valid=_validate(pivot, cfg),
    )


def _apply_distance_filter(swings: List[Swing], cfg: SwingConfig) -> List[Swing]:
    last_high_price: Optional[float] = None
    last_low_price: Optional[float] = None
    result: List[Swing] = []
    for s in swings:
        threshold = s.atr * cfg.atr_multiplier
        if s.swing_type == SwingType.HIGH:
            if last_high_price is not None and abs(s.price - last_high_price) < threshold:
                continue
            last_high_price = s.price
        else:
            if last_low_price is not None and abs(s.price - last_low_price) < threshold:
                continue
            last_low_price = s.price
        result.append(s)
    return result


def get_swings(df: pd.DataFrame, cfg: Optional[SwingConfig] = None) -> List[Swing]:
    if cfg is None:
        cfg = SwingConfig()
    if cfg.debug:
        logging.basicConfig(level=logging.DEBUG)

    min_rows: Final[int] = cfg.atr_period + cfg.pivot_window_max * 2 + 2
    if len(df) < min_rows:
        _log.warning("DataFrame too short (%d rows, need %d). Returning [].", len(df), min_rows)
        return []

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    atr = _calc_atr(df, cfg.atr_period)
    raw_pivots = _scan_raw_pivots(df, atr, cfg)
    _fill_move_sizes(raw_pivots)
    raw_pivots = _apply_noise_filter(raw_pivots, cfg)
    raw_pivots = _enforce_alternation(raw_pivots)
    raw_pivots = _merge_close_swings(raw_pivots, cfg)

    swings: List[Swing] = []
    for pivot in raw_pivots:
        if _validate(pivot, cfg):
            swings.append(_to_swing(pivot, df, cfg))

    swings = _apply_distance_filter(swings, cfg)
    swings.sort(key=lambda s: s.index)
    return swings
