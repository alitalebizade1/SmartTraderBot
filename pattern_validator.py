"""
pattern_validator.py — Pattern Validation Engine
=================================================
(unchanged from existing project version)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set

from models import Pattern, Signal

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatorConfig:
    min_score: float = 85.0
    min_hl_diff_atr: float = 0.3
    min_resistance_touches: int = 2
    min_risk_atr: float = 0.2
    max_pattern_age_bars: int = 120
    debug: bool = False


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


class PatternValidator:
    def __init__(self, cfg: Optional[ValidatorConfig] = None) -> None:
        self._cfg = cfg or ValidatorConfig()
        self._seen_ids: Set[str] = set()
        if self._cfg.debug:
            logging.basicConfig(level=logging.DEBUG)

    def validate(self, pattern: Pattern, signal: Optional[Signal] = None, current_bar: int = 0) -> ValidationResult:
        checks = [
            self._check_state(pattern),
            self._check_resistance(pattern),
            self._check_higher_lows(pattern),
            self._check_score(pattern),
            self._check_expiry(pattern, current_bar),
        ]
        if signal is not None:
            checks += [
                self._check_risk(signal, pattern),
                self._check_duplicate(signal),
            ]
        for result in checks:
            if not result.valid:
                self._debug("REJECTED: %s", result.reason)
                return result
        if signal is not None:
            self._seen_ids.add(signal.pattern_id)
        return ValidationResult(valid=True, reason="OK")

    def reset(self) -> None:
        self._seen_ids.clear()

    def _check_state(self, pattern: Pattern) -> ValidationResult:
        if pattern.state not in {"READY", "SIGNAL"}:
            return ValidationResult(valid=False, reason=f"pattern state not ready: {pattern.state}")
        return ValidationResult(valid=True)

    def _check_resistance(self, pattern: Pattern) -> ValidationResult:
        res = pattern.resistance
        if res is None:
            return ValidationResult(valid=False, reason="no resistance found")
        if len(res.touches) < self._cfg.min_resistance_touches:
            return ValidationResult(
                valid=False,
                reason=f"resistance touches {len(res.touches)} < {self._cfg.min_resistance_touches}",
            )
        return ValidationResult(valid=True)

    def _check_higher_lows(self, pattern: Pattern) -> ValidationResult:
        hl1 = pattern.hl1
        hl2 = pattern.hl2
        if hl1 is None:
            return ValidationResult(valid=False, reason="HL1 missing")
        if hl2 is None:
            return ValidationResult(valid=False, reason="HL2 missing")
        min_diff = (pattern.atr_at_signal or 1.0) * self._cfg.min_hl_diff_atr
        diff = hl2.price - hl1.price
        if diff <= 0:
            return ValidationResult(valid=False, reason=f"HL2 ({hl2.price:.5f}) not above HL1 ({hl1.price:.5f})")
        if diff < min_diff:
            return ValidationResult(valid=False, reason=f"HL2-HL1 diff {diff:.5f} below ATR threshold {min_diff:.5f}")
        if not hl1.valid:
            return ValidationResult(valid=False, reason="HL1 swing invalid")
        if not hl2.valid:
            return ValidationResult(valid=False, reason="HL2 swing invalid")
        return ValidationResult(valid=True)

    def _check_score(self, pattern: Pattern) -> ValidationResult:
        if pattern.score < self._cfg.min_score:
            return ValidationResult(valid=False, reason=f"pattern score {pattern.score:.1f} < threshold {self._cfg.min_score}")
        return ValidationResult(valid=True)

    def _check_expiry(self, pattern: Pattern, current_bar: int) -> ValidationResult:
        if pattern.start_bar is not None:
            age = current_bar - pattern.start_bar
            if age > self._cfg.max_pattern_age_bars:
                return ValidationResult(valid=False, reason=f"pattern expired: age {age} > {self._cfg.max_pattern_age_bars}")
        return ValidationResult(valid=True)

    def _check_risk(self, signal: Signal, pattern: Pattern) -> ValidationResult:
        risk = signal.entry - signal.sl
        min_risk = (pattern.atr_at_signal or 1.0) * self._cfg.min_risk_atr
        if risk <= 0:
            return ValidationResult(valid=False, reason=f"invalid risk: entry {signal.entry:.5f} <= sl {signal.sl:.5f}")
        if risk < min_risk:
            return ValidationResult(valid=False, reason=f"risk {risk:.5f} below ATR minimum {min_risk:.5f}")
        return ValidationResult(valid=True)

    def _check_duplicate(self, signal: Signal) -> ValidationResult:
        if signal.pattern_id in self._seen_ids:
            return ValidationResult(valid=False, reason=f"duplicate signal pattern_id={signal.pattern_id}")
        return ValidationResult(valid=True)

    def _debug(self, msg: str, *args: object) -> None:
        if self._cfg.debug:
            _log.debug(msg, *args)
