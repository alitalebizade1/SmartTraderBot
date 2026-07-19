"""
models.py — Domain Models for Institutional Price Action Engine
===============================================================
Single responsibility: strongly typed domain objects only.
No business logic. No calculations. No pandas dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import List, Optional


# ══════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════
class SwingType(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


class SwingStrength(str, Enum):
    WEAK = "WEAK"
    NORMAL = "NORMAL"
    STRONG = "STRONG"


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class TrendState(str, Enum):
    IMPULSIVE = "IMPULSIVE"
    CORRECTIVE = "CORRECTIVE"
    RANGING = "RANGING"
    UNDEFINED = "UNDEFINED"


class PatternState(str, Enum):
    SEARCH_TREND = "SEARCH_TREND"
    WAIT_RESISTANCE = "WAIT_RESISTANCE"
    WAIT_TOUCH1 = "WAIT_TOUCH1"
    WAIT_HL1 = "WAIT_HL1"
    WAIT_TOUCH2 = "WAIT_TOUCH2"
    WAIT_HL2 = "WAIT_HL2"
    WAIT_TOUCH3 = "WAIT_TOUCH3"
    READY = "READY"
    SIGNAL = "SIGNAL"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ResistanceType(str, Enum):
    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    ZONE = "ZONE"


class PatternResult(str, Enum):
    VALID = "VALID"
    REJECTED_TREND = "REJECTED_TREND"
    REJECTED_RES = "REJECTED_RES"
    REJECTED_HL = "REJECTED_HL"
    REJECTED_SCORE = "REJECTED_SCORE"
    REJECTED_RISK = "REJECTED_RISK"
    REJECTED_DUP = "REJECTED_DUPLICATE"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class DebugLevel(str, Enum):
    NONE = "NONE"
    INFO = "INFO"
    VERBOSE = "VERBOSE"


# ══════════════════════════════════════════════════════════
# SWING
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Swing:
    """
    A confirmed market structure swing point (High or Low).
    Non-repainting. Validated before construction.
    """
    index: int
    time: datetime
    price: float
    swing_type: SwingType
    strength: str  # SwingStrength value
    atr: float
    left_strength: float  # [0.0, 1.0]
    right_strength: float  # [0.0, 1.0]
    valid: bool
    score: float = 0.0  # composite quality score [0, 100]


# ══════════════════════════════════════════════════════════
# RESISTANCE
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Resistance:
    """
    Resistance is a price ZONE, not a single level.
    Width is ATR-adaptive.
    """
    center: float
    upper: float
    lower: float
    width: float
    touches: List[int]  # bar indices of confirmed touches
    strength: float  # aggregate touch quality [0, 100]
    score: float = 0.0
    resistance_type: ResistanceType = ResistanceType.ZONE

    # Legacy compatibility — price maps to center
    @property
    def price(self) -> float:
        return self.center


# ══════════════════════════════════════════════════════════
# TREND
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Trend:
    """
    Market structure trend object.
    Tracks HH/HL/LH/LL, BOS and ChoCH events.
    """
    direction: TrendDirection
    state: TrendState
    score: float  # [0, 100]
    confidence: float  # [0.0, 1.0]
    last_hh: Optional[Swing]
    last_hl: Optional[Swing]
    last_lh: Optional[Swing]
    last_ll: Optional[Swing]
    bos: bool  # Break of Structure detected
    choch: bool  # Change of Character detected
    valid: bool


# ══════════════════════════════════════════════════════════
# PATTERN
# ══════════════════════════════════════════════════════════
@dataclass
class Pattern:
    """
    The detected bullish Price Action pattern.
    Carries all structural components and computed targets.
    """
    id: str
    state: str  # PatternState value
    confirmed: bool
    score: float
    confidence: float  # [0.0, 1.0]
    trend: Optional[Trend]
    resistance: Optional[Resistance]
    touch1: Optional[Swing]
    touch2: Optional[Swing]
    touch3: Optional[Swing]
    hl1: Optional[Swing]
    hl2: Optional[Swing]
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    risk: float
    reward: float
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    debug_reason: str = ""

    # Internal fields used across modules
    trend_confirmed: bool = False
    atr_at_signal: float = 0.0
    start_bar: Optional[int] = None

    # Channel start price — lowest low before the uptrend began.
    # Used to compute the "measured move" for TP2.
    channel_start_price: Optional[float] = None
    channel_start_bar: Optional[int] = None


# ══════════════════════════════════════════════════════════
# SIGNAL
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Signal:
    """
    Emitted trading signal.
    Separate from Pattern — carries only execution-relevant fields.
    """
    symbol: str
    timeframe: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    risk_reward: float
    signal_type: SignalType
    confidence: float  # [0.0, 1.0]
    pattern_id: str
    index: int
    candle_time: datetime

    # Derived convenience fields
    risk: float = 0.0
    reward: float = 0.0

    # Full pattern context (touches, HLs, resistance, channel start) —
    # used by the chart renderer to draw the complete pattern annotation.
    pattern: Optional["Pattern"] = None

    # Legacy compatibility
    @property
    def stop_loss(self) -> float:
        return self.sl


# ══════════════════════════════════════════════════════════
# DEBUG
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DebugInfo:
    """
    Rejection or validation trace for a single pattern evaluation.
    """
    reason: str
    state: str  # PatternState value at rejection
    score: float
    rejected_step: str
    message: str
    result: PatternResult = PatternResult.UNKNOWN


# ══════════════════════════════════════════════════════════
# CONFIGURATION MODELS
# ══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SwingConfig:
    """All parameters for the Swing Detection Engine."""
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


@dataclass(frozen=True)
class TrendConfig:
    """All parameters for Trend Analysis."""
    window: int = 40
    min_bullish_pct: float = 0.65
    min_hh_hl_count: int = 2
    impulse_atr: float = 1.5
    momentum_window: int = 10
    debug: bool = False


@dataclass(frozen=True)
class ResistanceConfig:
    """All parameters for Resistance Zone detection."""
    zone_atr: float = 0.8
    min_wick_ratio: float = 0.25
    min_candles_after: int = 2
    min_drop_atr: float = 0.5
    resistance_type: ResistanceType = ResistanceType.ZONE
    debug: bool = False


@dataclass(frozen=True)
class PatternConfig:
    """All parameters for Pattern Detection."""
    atr_period: int = 14
    min_score: float = 85.0
    hl_min_diff_atr: float = 0.3
    hl_max_depth_atr: float = 3.5
    hl_min_strength: str = SwingStrength.NORMAL.value
    sl_buffer_atr: float = 0.0  # SL sits exactly on HL2 — no buffer
    rr_tp2: float = 2.0
    max_bars_wait: int = 80
    max_signal_wait: int = 20
    max_pattern_age_bars: int = 120
    swing_cfg: SwingConfig = field(default_factory=SwingConfig)
    trend_cfg: TrendConfig = field(default_factory=TrendConfig)
    resistance_cfg: ResistanceConfig = field(default_factory=ResistanceConfig)
    debug: bool = False
    debug_level: DebugLevel = DebugLevel.NONE
