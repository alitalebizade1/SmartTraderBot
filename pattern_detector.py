"""
pattern_detector.py — Custom Bullish Price Action Pattern Detector
==================================================================

Detects ONE pattern only:

    Strong Uptrend
        → Resistance Touch 1
        → HL1
        → Resistance Touch 2
        → HL2 (HL2 > HL1)
        → Resistance Touch 3
        → HL3 (HL3 > HL2)          [confirmation of a rising bottom]
        → READY → BUY SIGNAL (before breakout)

State Machine:

    SEARCH_TREND → WAIT_RESISTANCE → WAIT_TOUCH1 → WAIT_HL1
        → WAIT_TOUCH2 → WAIT_HL2 → WAIT_TOUCH3 → READY → SIGNAL → RESET

Signal is emitted immediately after the third touch is confirmed and a
bullish confirmation candle closes above HL2. NOT after breakout.
NOT after BOS. NOT after resistance break.

Risk levels
-----------
* Stop-loss  : exactly on HL2 (the second higher low — no ATR buffer).
* TP1        : the resistance level.
* TP2        : "measured move" — entry + (resistance - channel_start_price),
               i.e. the height of the original impulse leg projected
               upward from the entry price.

Accuracy >> Quantity. Missing a pattern is acceptable.
Detecting a fake pattern is NOT acceptable.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from models import Pattern, Resistance, Signal, SignalType, Swing, SwingType
from pattern_validator import PatternValidator, ValidatorConfig
from swing_detector import SwingConfig, get_swings

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────
class DetectorState(str, Enum):
    SEARCH_TREND = "SEARCH_TREND"
    WAIT_RESISTANCE = "WAIT_RESISTANCE"
    WAIT_TOUCH1 = "WAIT_TOUCH1"
    WAIT_HL1 = "WAIT_HL1"
    WAIT_TOUCH2 = "WAIT_TOUCH2"
    WAIT_HL2 = "WAIT_HL2"
    WAIT_TOUCH3 = "WAIT_TOUCH3"
    READY = "READY"
    SIGNAL = "SIGNAL"
    RESET = "RESET"


# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DetectorConfig:
    # ATR
    atr_period: int = 14

    # Trend — tuned for XAU/USD 15-minute data
    trend_window: int = 80            # was 40 — longer window captures the full impulse
    trend_min_bullish_pct: float = 0.55   # was 0.65 — gold often has 55-60% bullish bars in uptrends
    trend_min_hh_hl_count: int = 2
    trend_impulse_atr: float = 1.0    # was 1.5 — gold 15m impulses often 1-3 ATR

    # Resistance zone — wider tolerance so touches are accepted
    res_zone_atr: float = 1.5         # was 0.8 — zone half-width = ATR * 1.5 (~25-30 pips gold)
    res_min_wick_ratio: float = 0.15  # was 0.25 — accept shorter wicks at resistance
    res_min_candles_after: int = 2    # bars after touch before confirming rejection
    res_min_drop_atr: float = 0.3     # was 0.5 — accept smaller drops after touch

    # Higher Low — relaxed for gold's deeper corrections
    hl_min_diff_atr: float = 0.2      # was 0.3 — each HL must be above the previous
    hl_max_depth_atr: float = 6.0     # was 3.5 — allow deeper corrections (gold often 50-61.8% retrace)
    hl_min_strength: str = "NORMAL"   # minimum swing strength accepted

    # Timeouts (bars) — 150 bars ≈ 37.5 hours, covers multi-session patterns
    max_bars_wait: int = 150          # was 80
    max_signal_wait: int = 40         # was 20

    # Signal / risk
    sl_buffer_atr: float = 0.0        # SL sits exactly on HL2 — no ATR buffer

    # Score threshold — was 85, relaxed to accept more valid patterns
    min_score: float = 65.0

    # Swing detector
    swing_cfg: SwingConfig = field(default_factory=SwingConfig)

    # Debug
    debug: bool = False


# ──────────────────────────────────────────────────────────
# ATR HELPER
# ──────────────────────────────────────────────────────────
def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _atr_at(atr: pd.Series, idx: int) -> float:
    v = float(atr.iloc[idx])
    if np.isnan(v):
        v = float(atr.dropna().mean())
    return v if not np.isnan(v) else 1.0


# ──────────────────────────────────────────────────────────
# TREND ANALYSIS
# ──────────────────────────────────────────────────────────
@dataclass
class TrendAnalysis:
    confirmed: bool
    bullish_pct: float
    hh_count: int
    hl_count: int
    impulse_size: float
    momentum_score: float
    score: float               # 0-20 for pattern scoring
    channel_start_price: float  # lowest low before/at the start of the impulse
    channel_start_bar: int


def _analyse_trend(
    df: pd.DataFrame,
    swings: List[Swing],
    end_idx: int,
    atr_val: float,
    cfg: DetectorConfig,
) -> TrendAnalysis:
    """
    Evaluate trend quality over the window ending at end_idx.
    Returns TrendAnalysis with confirmed=True only if all conditions met.
    """
    start = max(0, end_idx - cfg.trend_window)
    window = df.iloc[start: end_idx + 1]

    # Bullish candle percentage
    bullish = (window["close"] > window["open"]).sum()
    pct = bullish / max(len(window), 1)

    # Count HH and HL in swings within window
    w_swings = [s for s in swings if start <= s.index <= end_idx]
    highs = sorted([s for s in w_swings if s.swing_type == SwingType.HIGH], key=lambda s: s.index)
    lows = sorted([s for s in w_swings if s.swing_type == SwingType.LOW], key=lambda s: s.index)

    hh_count = sum(
        1 for i in range(1, len(highs)) if highs[i].price > highs[i - 1].price
    )
    hl_count = sum(
        1 for i in range(1, len(lows)) if lows[i].price > lows[i - 1].price
    )

    # Impulse: largest upward move from low to high in window
    # Track which low anchors the largest impulse — that low is the
    # "channel start" used later for the measured-move TP2 calculation.
    impulse = 0.0
    channel_start_price = float(window["low"].min())
    channel_start_bar = int(window["low"].idxmin()) if hasattr(window["low"], "idxmin") else start

    for h in highs:
        preceding_lows = [l for l in lows if l.index < h.index]
        if preceding_lows:
            anchor_low = preceding_lows[-1]
            move = h.price - anchor_low.price
            if move > impulse:
                impulse = move
                channel_start_price = anchor_low.price
                channel_start_bar = anchor_low.index

    impulse_atr = impulse / atr_val if atr_val > 0 else 0.0

    # Momentum: close vs open of window
    if len(window) >= 2:
        momentum = (window["close"].iloc[-1] - window["open"].iloc[0]) / atr_val
    else:
        momentum = 0.0
    momentum_score = min(max(momentum / 5.0, 0.0), 1.0)

    confirmed = (
        pct >= cfg.trend_min_bullish_pct
        and (hh_count + hl_count) >= cfg.trend_min_hh_hl_count
        and impulse_atr >= cfg.trend_impulse_atr
    )

    # Score [0, 20]
    score_raw = (
        0.4 * min(pct / cfg.trend_min_bullish_pct, 1.0)
        + 0.3 * min((hh_count + hl_count) / max(cfg.trend_min_hh_hl_count * 2, 1), 1.0)
        + 0.3 * min(impulse_atr / (cfg.trend_impulse_atr * 2), 1.0)
    )
    score = round(score_raw * 20.0, 2)

    return TrendAnalysis(
        confirmed=confirmed,
        bullish_pct=pct,
        hh_count=hh_count,
        hl_count=hl_count,
        impulse_size=impulse,
        momentum_score=momentum_score,
        score=score,
        channel_start_price=channel_start_price,
        channel_start_bar=channel_start_bar,
    )


# ──────────────────────────────────────────────────────────
# RESISTANCE ZONE
# ──────────────────────────────────────────────────────────
def _resistance_broken(
    df: pd.DataFrame,
    level: float,
    tol: float,
    from_idx: int,
    to_idx: int,
    confirm_bars: int = 2,
) -> bool:
    """
    True only when `confirm_bars` consecutive closes exceed resistance + tol.
    A single bar close above the zone is treated as a wick/spike, not a break.
    This prevents false resets when price briefly pokes through the zone.
    """
    if from_idx >= to_idx:
        return False
    closes = df["close"].iloc[from_idx:to_idx].values
    threshold = level + tol
    count = 0
    for c in closes:
        if c > threshold:
            count += 1
            if count >= confirm_bars:
                return True
        else:
            count = 0
    return False


def _touch_quality(
    df: pd.DataFrame,
    touch_idx: int,
    level: float,
    tol: float,
    atr_val: float,
    cfg: DetectorConfig,
) -> Tuple[bool, float]:
    """
    Returns (touch_valid, wick_ratio) for a candidate touch candle.

    A touch is valid when:
    - high is within resistance zone
    - sufficient rejection wick
    - price drops by res_min_drop_atr after touch
    """
    row = df.iloc[touch_idx]
    total_range = row["high"] - row["low"]
    if total_range <= 0:
        return False, 0.0

    wick = row["high"] - max(row["open"], row["close"])
    wick_ratio = wick / total_range
    if wick_ratio < cfg.res_min_wick_ratio:
        return False, wick_ratio

    # Price must drop after touch
    end = min(touch_idx + cfg.res_min_candles_after + 1, len(df))
    subsequent = df["low"].iloc[touch_idx + 1: end]
    if subsequent.empty:
        return False, wick_ratio

    drop = row["high"] - subsequent.min()
    if drop < atr_val * cfg.res_min_drop_atr:
        return False, wick_ratio

    return True, wick_ratio


def _resistance_score(wick1: float, wick2: float, wick3: float) -> float:
    """Score [0, 20] based on rejection quality across all three touches."""
    avg_wick = (wick1 + wick2 + wick3) / 3.0
    return round(min(avg_wick / 0.6, 1.0) * 20.0, 2)


# ──────────────────────────────────────────────────────────
# HIGHER LOW SCORING
# ──────────────────────────────────────────────────────────
_STRENGTH_RANK = {"WEAK": 0, "NORMAL": 1, "STRONG": 2}


def _hl_accepted(swing: Swing, cfg: DetectorConfig) -> bool:
    """True if swing meets minimum strength requirement."""
    return _STRENGTH_RANK.get(swing.strength, 0) >= _STRENGTH_RANK.get(cfg.hl_min_strength, 0)


def _hl_score(hl1: Swing, hl2: Swing, atr_val: float, cfg: DetectorConfig) -> float:
    """Score [0, 20] based on HL quality (progression from HL1 to HL2)."""
    diff_atr = (hl2.price - hl1.price) / atr_val if atr_val > 0 else 0.0
    s1 = _STRENGTH_RANK.get(hl1.strength, 0) / 2.0
    s2 = _STRENGTH_RANK.get(hl2.strength, 0) / 2.0
    diff_score = min(diff_atr / (cfg.hl_min_diff_atr * 3), 1.0)
    raw = 0.4 * diff_score + 0.3 * s1 + 0.3 * s2
    return round(raw * 20.0, 2)


# ──────────────────────────────────────────────────────────
# IMPULSE / MOMENTUM SCORES
# ──────────────────────────────────────────────────────────
def _impulse_score(impulse: float, atr_val: float, cfg: DetectorConfig) -> float:
    """Score [0, 20] for impulse move size."""
    atr_units = impulse / atr_val if atr_val > 0 else 0.0
    return round(min(atr_units / (cfg.trend_impulse_atr * 2), 1.0) * 20.0, 2)


def _momentum_score(momentum: float) -> float:
    """Score [0, 20] for momentum."""
    return round(min(momentum, 1.0) * 20.0, 2)


# ──────────────────────────────────────────────────────────
# PATTERN ID
# ──────────────────────────────────────────────────────────
def _make_pattern_id(touch1_idx: int, touch2_idx: int, touch3_idx: int, hl1_idx: int, hl2_idx: int) -> str:
    raw = f"{touch1_idx}-{touch2_idx}-{touch3_idx}-{hl1_idx}-{hl2_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ──────────────────────────────────────────────────────────
# STATE MACHINE
# ──────────────────────────────────────────────────────────
class _PatternStateMachine:
    """
    Processes one candle at a time. Transitions through:

    SEARCH_TREND → WAIT_RESISTANCE → WAIT_TOUCH1 → WAIT_HL1
        → WAIT_TOUCH2 → WAIT_HL2 → WAIT_TOUCH3 → READY → SIGNAL → RESET
    """

    def __init__(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        swings: List[Swing],
        cfg: DetectorConfig,
    ) -> None:
        self._df = df
        self._atr = atr
        self._swings = swings
        self._cfg = cfg
        self._state = DetectorState.SEARCH_TREND
        self._reset_vars()

    # ── STATE VARIABLES ──────────────────────────────────
    def _reset_vars(self) -> None:
        self._state_entry_bar: int = 0
        self._trend: Optional[TrendAnalysis] = None
        self._trend_end_bar: int = 0
        self._resistance_level: float = 0.0
        self._resistance_tol: float = 0.0
        self._touch1_idx: int = -1
        self._touch1_wick: float = 0.0
        self._touch2_idx: int = -1
        self._touch2_wick: float = 0.0
        self._touch3_idx: int = -1
        self._touch3_wick: float = 0.0
        self._hl1: Optional[Swing] = None
        self._hl2: Optional[Swing] = None

    def _transition(self, new_state: DetectorState, bar: int) -> None:
        if self._cfg.debug:
            _log.debug("STATE %s → %s bar=%d", self._state.value, new_state.value, bar)
        self._state = new_state
        self._state_entry_bar = bar

    def _timed_out(self, bar: int) -> bool:
        return (bar - self._state_entry_bar) > self._cfg.max_bars_wait

    def _signal_timed_out(self, bar: int) -> bool:
        return (bar - self._state_entry_bar) > self._cfg.max_signal_wait

    # ── SWING HELPERS ─────────────────────────────────────
    def _highs_in(self, from_idx: int, to_idx: int) -> List[Swing]:
        return sorted(
            [s for s in self._swings if s.swing_type == SwingType.HIGH and from_idx < s.index <= to_idx],
            key=lambda s: s.index,
        )

    def _lows_in(self, from_idx: int, to_idx: int) -> List[Swing]:
        return sorted(
            [s for s in self._swings if s.swing_type == SwingType.LOW and from_idx < s.index <= to_idx],
            key=lambda s: s.index,
        )

    # ── MAIN LOOP ──────────────────────────────────────────
    def run(self) -> List[Signal]:
        signals: List[Signal] = []
        df = self._df
        n = len(df)
        i = 0

        while i < n:
            av = _atr_at(self._atr, i)

            # ────────────────────────────────────────────
            # STATE: SEARCH_TREND
            # ────────────────────────────────────────────
            if self._state == DetectorState.SEARCH_TREND:
                end = min(i + self._cfg.trend_window, n - 1)
                ta = _analyse_trend(df, self._swings, end, av, self._cfg)
                if ta.confirmed:
                    self._trend = ta
                    self._trend_end_bar = end
                    self._transition(DetectorState.WAIT_RESISTANCE, end)
                    i = end + 1
                else:
                    i += max(self._cfg.trend_window // 4, 1)
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_RESISTANCE
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_RESISTANCE:
                if self._timed_out(i):
                    self._debug_reject("WAIT_RESISTANCE timeout", i)
                    self._reset_to_search(i); i += 1; continue

                # Find the first swing High after trend end that is followed
                # by a swing Low (confirming it acted as resistance).
                # Use that first HIGH as the anchor bar (state_entry_bar) so
                # that subsequent touches are searched after it.
                # Use the HIGHEST HIGH in the local cluster as the resistance LEVEL
                # so the zone captures all future re-tests.
                candidates = self._highs_in(self._trend_end_bar, i)
                if candidates:
                    # Find first high followed by a low (resistance confirmation)
                    first_res_sw = None
                    for sw in candidates:
                        lows_after = self._lows_in(sw.index, sw.index + self._cfg.max_bars_wait)
                        if lows_after:
                            first_res_sw = sw
                            break

                    if first_res_sw is None:
                        # No confirmed resistance yet — keep waiting
                        i += 1
                        continue

                    # Resistance level = highest HIGH within ATR×2 of first confirmed high
                    anchor_atr = _atr_at(self._atr, first_res_sw.index)
                    cluster_tol = anchor_atr * 2.0
                    cluster_highs = [
                        s for s in candidates
                        if abs(s.price - first_res_sw.price) <= cluster_tol
                    ]
                    resistance_price = max(s.price for s in cluster_highs)
                    self._resistance_level = resistance_price
                    self._resistance_tol = anchor_atr * self._cfg.res_zone_atr
                    self._transition(DetectorState.WAIT_TOUCH1, first_res_sw.index)
                    i = first_res_sw.index + 1
                else:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_TOUCH1
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_TOUCH1:
                if self._timed_out(i):
                    self._debug_reject("WAIT_TOUCH1 timeout", i)
                    self._reset_to_search(i); i += 1; continue

                # Resistance broken before touch → reset
                if _resistance_broken(df, self._resistance_level, self._resistance_tol, self._state_entry_bar, i):
                    self._debug_reject("resistance broken before touch1", i)
                    self._reset_to_search(i); i += 1; continue

                # Candle-based touch: any bar whose HIGH reaches the resistance zone.
                # (Swing-based detection misses touches because swing_detector merges close highs.)
                found_touch = False
                scan_start = max(self._state_entry_bar, i - 1)
                for bar_idx in range(scan_start, i + 1):
                    if bar_idx >= len(df):
                        break
                    bar_high = float(df["high"].iloc[bar_idx])
                    if abs(bar_high - self._resistance_level) <= self._resistance_tol:
                        av_t = _atr_at(self._atr, bar_idx)
                        valid, wick = _touch_quality(
                            df, bar_idx, self._resistance_level, self._resistance_tol, av_t, self._cfg
                        )
                        if valid:
                            self._touch1_idx = bar_idx
                            self._touch1_wick = wick
                            self._transition(DetectorState.WAIT_HL1, bar_idx)
                            i = bar_idx + 1
                            found_touch = True
                            break
                if not found_touch:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_HL1
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_HL1:
                if self._timed_out(i):
                    self._debug_reject("WAIT_HL1 timeout", i)
                    self._reset_to_search(i); i += 1; continue
                # Note: price rising back toward resistance here is EXPECTED
                # (it is the beginning of touch2, not a resistance break).
                # Only reset on timeout — no resistance_broken check in this state.

                lows = self._lows_in(self._touch1_idx, i)
                for sw in lows:
                    if not _hl_accepted(sw, self._cfg):
                        continue

                    # Too deep correction → reject
                    depth = self._resistance_level - sw.price
                    if depth > _atr_at(self._atr, sw.index) * self._cfg.hl_max_depth_atr:
                        self._debug_reject(f"HL1 too deep depth={depth:.4f}", i)
                        self._reset_to_search(i)
                        break

                    self._hl1 = sw
                    self._transition(DetectorState.WAIT_TOUCH2, sw.index)
                    i = sw.index + 1
                    break
                else:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_TOUCH2
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_TOUCH2:
                if self._timed_out(i):
                    self._debug_reject("WAIT_TOUCH2 timeout", i)
                    self._reset_to_search(i); i += 1; continue
                # resistance_broken is intentionally NOT checked here:
                # price approaching resistance is the beginning of touch2.
                # A true breakout is identified by touch_quality failing AND
                # by the signal state's final resistance_broken guard.

                found_touch = False
                scan_start = max(self._hl1.index + 1, i - 1)
                for bar_idx in range(scan_start, i + 1):
                    if bar_idx >= len(df):
                        break
                    bar_high = float(df["high"].iloc[bar_idx])
                    if abs(bar_high - self._resistance_level) <= self._resistance_tol:
                        av_t = _atr_at(self._atr, bar_idx)
                        valid, wick = _touch_quality(
                            df, bar_idx, self._resistance_level, self._resistance_tol, av_t, self._cfg
                        )
                        if valid:
                            self._touch2_idx = bar_idx
                            self._touch2_wick = wick
                            self._transition(DetectorState.WAIT_HL2, bar_idx)
                            i = bar_idx + 1
                            found_touch = True
                            break
                if not found_touch:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_HL2
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_HL2:
                if self._timed_out(i):
                    self._debug_reject("WAIT_HL2 timeout", i)
                    self._reset_to_search(i); i += 1; continue
                # Note: price rising back toward resistance here is EXPECTED
                # (it is the beginning of touch3, not a resistance break).
                # Only reset on timeout — no resistance_broken check in this state.

                lows = self._lows_in(self._touch2_idx, i)
                for sw in lows:
                    if not _hl_accepted(sw, self._cfg):
                        continue

                    # HL2 must be strictly above HL1 by ATR threshold
                    min_diff = _atr_at(self._atr, sw.index) * self._cfg.hl_min_diff_atr
                    if sw.price <= self._hl1.price:
                        self._debug_reject(
                            f"HL2 {sw.price:.5f} <= HL1 {self._hl1.price:.5f}", i
                        )
                        self._reset_to_search(i)
                        break
                    if (sw.price - self._hl1.price) < min_diff:
                        self._debug_reject(
                            f"HL2-HL1 diff {sw.price - self._hl1.price:.5f} < {min_diff:.5f}", i
                        )
                        self._reset_to_search(i)
                        break

                    self._hl2 = sw
                    self._transition(DetectorState.WAIT_TOUCH3, sw.index)
                    i = sw.index + 1
                    break
                else:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: WAIT_TOUCH3
            # ────────────────────────────────────────────
            if self._state == DetectorState.WAIT_TOUCH3:
                if self._timed_out(i):
                    self._debug_reject("WAIT_TOUCH3 timeout", i)
                    self._reset_to_search(i); i += 1; continue
                # resistance_broken intentionally NOT checked before touch3 search:
                # price approaching resistance = beginning of touch3.

                found_touch = False
                scan_start = max(self._hl2.index + 1, i - 1)
                for bar_idx in range(scan_start, i + 1):
                    if bar_idx >= len(df):
                        break
                    bar_high = float(df["high"].iloc[bar_idx])
                    if abs(bar_high - self._resistance_level) <= self._resistance_tol:
                        av_t = _atr_at(self._atr, bar_idx)
                        valid, wick = _touch_quality(
                            df, bar_idx, self._resistance_level, self._resistance_tol, av_t, self._cfg
                        )
                        if valid:
                            self._touch3_idx = bar_idx
                            self._touch3_wick = wick
                            self._transition(DetectorState.READY, bar_idx)
                            i = bar_idx + 1
                            found_touch = True
                            break
                if not found_touch:
                    i += 1
                continue

            # ────────────────────────────────────────────
            # STATE: READY → SIGNAL
            # ────────────────────────────────────────────
            if self._state == DetectorState.READY:
                if self._signal_timed_out(i):
                    self._debug_reject("READY signal timeout", i)
                    self._reset_to_search(i); i += 1; continue

                if _resistance_broken(df, self._resistance_level, self._resistance_tol, self._touch3_idx, i):
                    self._debug_reject("resistance broken before signal emission", i)
                    self._reset_to_search(i); i += 1; continue

                candle = df.iloc[i]
                is_bullish = candle["close"] > candle["open"]
                above_hl2 = candle["low"] > self._hl2.price

                if is_bullish and above_hl2:
                    sig = self._build_signal(i, _atr_at(self._atr, i))
                    if sig is not None:
                        signals.append(sig)
                    self._reset_to_search(i)

                i += 1
                continue

            # Safety fallthrough
            i += 1

        return sorted(signals, key=lambda s: s.index)

    # ── SIGNAL BUILDER ────────────────────────────────────
    def _build_signal(self, bar: int, av: float) -> Optional[Signal]:
        """Assemble, score, validate and return Signal or None."""
        assert self._hl1 is not None
        assert self._hl2 is not None
        assert self._trend is not None

        # ── Scores ──
        t_score = self._trend.score
        r_score = _resistance_score(self._touch1_wick, self._touch2_wick, self._touch3_wick)
        hl_score = _hl_score(self._hl1, self._hl2, av, self._cfg)
        i_score = _impulse_score(self._trend.impulse_size, av, self._cfg)
        m_score = _momentum_score(self._trend.momentum_score)

        total = round(t_score + r_score + hl_score + i_score + m_score, 2)
        if total < self._cfg.min_score:
            self._debug_reject(f"score {total:.1f} < {self._cfg.min_score}", bar)
            return None

        # ── Resistance object ──
        res_price = (
            self._df["high"].iloc[self._touch1_idx]
            + self._df["high"].iloc[self._touch2_idx]
            + self._df["high"].iloc[self._touch3_idx]
        ) / 3.0

        resistance = Resistance(
            center=res_price,
            upper=res_price + self._resistance_tol,
            lower=res_price - self._resistance_tol,
            width=self._resistance_tol * 2,
            touches=[self._touch1_idx, self._touch2_idx, self._touch3_idx],
            strength=r_score,
        )

        # ── Pattern object ──
        pat_id = _make_pattern_id(
            self._touch1_idx, self._touch2_idx, self._touch3_idx,
            self._hl1.index, self._hl2.index,
        )

        def _touch_swing(idx: int) -> Swing:
            return Swing(
                index=idx,
                price=self._df["high"].iloc[idx],
                time=(
                    self._df.index[idx]
                    if pd.api.types.is_datetime64_any_dtype(self._df.index)
                    else self._df["time"].iloc[idx]
                ),
                swing_type=SwingType.HIGH,
                strength="NORMAL",
                atr=av,
                left_strength=0.5,
                right_strength=0.5,
                valid=True,
            )

        pattern = Pattern(
            id=pat_id,
            state=DetectorState.SIGNAL.value,
            confirmed=True,
            score=total,
            confidence=min(total / 100.0, 1.0),
            trend=None,
            resistance=resistance,
            touch1=_touch_swing(self._touch1_idx),
            touch2=_touch_swing(self._touch2_idx),
            touch3=_touch_swing(self._touch3_idx),
            hl1=self._hl1,
            hl2=self._hl2,
            entry=0.0,   # filled below
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            risk=0.0,
            reward=0.0,
            trend_confirmed=self._trend.confirmed,
            atr_at_signal=av,
            start_bar=self._trend_end_bar,
            channel_start_price=self._trend.channel_start_price,
            channel_start_bar=self._trend.channel_start_bar,
        )

        # ── Risk levels ──
        entry = float(self._df["close"].iloc[bar])

        # SL sits exactly on HL2 (no ATR buffer), as confirmed by the
        # reference chart annotation "خط تشکیل اولین کف".
        sl = self._hl2.price - av * self._cfg.sl_buffer_atr
        risk = entry - sl
        if risk <= 0:
            self._debug_reject("non-positive risk", bar)
            return None

        # TP1 — the resistance level itself.
        tp1 = res_price

        # TP2 — measured move: project the height of the original impulse
        # (resistance - channel_start_price) upward from the entry price.
        measured_move = res_price - self._trend.channel_start_price
        if measured_move <= 0:
            # Fallback — should not normally happen, but guards against
            # a degenerate channel_start reading.
            measured_move = risk * self._cfg.rr_tp2
        tp2 = entry + measured_move

        pattern.entry = round(entry, 5)
        pattern.stop_loss = round(sl, 5)
        pattern.tp1 = round(tp1, 5)
        pattern.tp2 = round(tp2, 5)
        pattern.risk = round(risk, 5)
        pattern.reward = round(tp2 - entry, 5)

        time_val = (
            self._df.index[bar]
            if pd.api.types.is_datetime64_any_dtype(self._df.index)
            else self._df["time"].iloc[bar]
        )

        rr = (tp2 - entry) / risk if risk > 0 else 0.0

        signal = Signal(
            symbol="",
            timeframe="",
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp1=round(tp1, 5),
            tp2=round(tp2, 5),
            risk_reward=round(rr, 3),
            signal_type=SignalType.BUY,
            confidence=pattern.confidence,
            pattern_id=pat_id,
            index=bar,
            candle_time=time_val,
            risk=round(risk, 5),
            reward=round(tp2 - entry, 5),
            pattern=pattern,
        )

        # ── Validate ──
        validator = PatternValidator(
            ValidatorConfig(
                min_score=self._cfg.min_score,
                min_hl_diff_atr=self._cfg.hl_min_diff_atr,
                debug=self._cfg.debug,
            )
        )
        result = validator.validate(pattern, signal, current_bar=bar)
        if not result.valid:
            self._debug_reject(f"validator: {result.reason}", bar)
            return None

        return signal

    # ── HELPERS ────────────────────────────────────────────
    def _reset_to_search(self, bar: int) -> None:
        self._reset_vars()
        self._transition(DetectorState.SEARCH_TREND, bar)

    def _debug_reject(self, reason: str, bar: int) -> None:
        if self._cfg.debug:
            _log.debug("REJECT bar=%d reason=%s", bar, reason)


# ──────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────
def detect(
    df: pd.DataFrame,
    cfg: Optional[DetectorConfig] = None,
) -> List[Signal]:
    """
    Main entry point.

    Args:
        df: OHLC DataFrame with columns [open, high, low, close].
            Optionally includes a 'time' column.
        cfg: DetectorConfig — all defaults used if None.

    Returns:
        List[Signal] sorted by bar index, validated only.

    Missing a pattern is acceptable. Emitting a fake pattern is NOT acceptable.
    """
    if cfg is None:
        cfg = DetectorConfig()

    if cfg.debug:
        logging.basicConfig(level=logging.DEBUG)

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain: {required}")

    if len(df) < cfg.atr_period + cfg.trend_window + 10:
        _log.warning("DataFrame too short for reliable detection.")
        return []

    atr = _calc_atr(df, cfg.atr_period)
    swings = get_swings(df, cfg.swing_cfg)

    if len(swings) < 7:  # trend anchor + 3 touches + 2 HLs (min viable structure)
        _log.debug("Too few swings (%d) — no patterns possible.", len(swings))
        return []

    machine = _PatternStateMachine(df, atr, swings, cfg)
    return machine.run()
