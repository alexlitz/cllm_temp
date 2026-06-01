"""Declarative FFN guarantees for structural numeric representations.

This module emits ``FFNRule`` data only. It does not inspect DraftVM state,
perform runtime substitution, or apply Python-side smoke corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .ir import FFNRule


ConditionSpec = Tuple[str, float]


@dataclass(frozen=True)
class OneHotBandGuarantee:
    """Expected one-hot shape that can be enforced by declarative FFN rules.

    When the activation/read conditions meet ``condition_threshold``, emitted
    rules set the selected lane to ``active_value`` and every other lane to
    ``inactive_value`` by adding a lane-local residual correction:

        delta = target_value - current_lane_value

    The lane read is expressed through ``FFNRule.gated_write`` gate terms, so
    the correction stays in the compiler IR instead of becoming Python runtime
    logic.
    """

    band_base: str
    expected_index: int
    activation_conditions: Sequence[ConditionSpec]
    read_conditions: Sequence[ConditionSpec] = field(default_factory=tuple)
    width: int = 16
    condition_threshold: Optional[float] = None
    min_margin: float = 1.0
    inactive_value: float = 0.0
    active_value: Optional[float] = None
    max_abs_weight: float = 16.0
    name: Optional[str] = None
    scope: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.band_base, str) or not self.band_base:
            raise ValueError("band_base must be a non-empty string")
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise ValueError("width must be an integer")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if (
            not isinstance(self.expected_index, int)
            or isinstance(self.expected_index, bool)
        ):
            raise ValueError("expected_index must be an integer")
        if not 0 <= self.expected_index < self.width:
            raise ValueError("expected_index must be within band width")

        activation_conditions = _normalize_conditions(
            self.activation_conditions,
            label="activation_conditions",
        )
        read_conditions = _normalize_conditions(
            self.read_conditions,
            label="read_conditions",
        )
        if not activation_conditions and not read_conditions:
            raise ValueError(
                "at least one activation or read condition is required"
            )
        object.__setattr__(
            self, "activation_conditions", activation_conditions
        )
        object.__setattr__(self, "read_conditions", read_conditions)

        _require_finite("min_margin", self.min_margin)
        _require_finite("inactive_value", self.inactive_value)
        if self.min_margin < 0.0:
            raise ValueError("min_margin must be non-negative")
        if self.active_value is not None:
            _require_finite("active_value", self.active_value)
        _require_finite("max_abs_weight", self.max_abs_weight)
        if self.max_abs_weight <= 0.0:
            raise ValueError("max_abs_weight must be positive")

        active_value = self.resolved_active_value
        if active_value - self.inactive_value < self.min_margin:
            raise ValueError(
                "active_value - inactive_value must satisfy min_margin"
            )

        threshold = self.resolved_condition_threshold
        _require_finite("condition_threshold", threshold)

        bounded_values = [
            ("condition_threshold", threshold),
            ("active_value", active_value),
            ("inactive_value", self.inactive_value),
            ("lane_read_weight", -1.0),
            ("lane_write_weight", 1.0),
        ]
        bounded_values.extend(
            (f"activation_conditions[{idx}].weight", weight)
            for idx, (_, weight) in enumerate(activation_conditions)
        )
        bounded_values.extend(
            (f"read_conditions[{idx}].weight", weight)
            for idx, (_, weight) in enumerate(read_conditions)
        )
        for label, value in bounded_values:
            if abs(value) > self.max_abs_weight:
                raise ValueError(
                    f"{label}={value!r} exceeds max_abs_weight "
                    f"{self.max_abs_weight!r}"
                )

    @property
    def label(self) -> str:
        return self.name or self.band_base

    @property
    def conditions(self) -> Tuple[ConditionSpec, ...]:
        return tuple(self.activation_conditions) + tuple(self.read_conditions)

    @property
    def resolved_active_value(self) -> float:
        if self.active_value is not None:
            return float(self.active_value)
        return float(self.inactive_value + self.min_margin)

    @property
    def resolved_condition_threshold(self) -> float:
        if self.condition_threshold is not None:
            return float(self.condition_threshold)
        return _default_all_conditions_threshold(self.conditions)

    def lane_name(self, index: int) -> str:
        if not 0 <= index < self.width:
            raise ValueError("lane index must be within band width")
        return f"{self.band_base}+{index}"

    def target_value(self, index: int) -> float:
        return (
            self.resolved_active_value
            if index == self.expected_index
            else float(self.inactive_value)
        )

    def to_ffn_rules(self) -> Tuple[FFNRule, ...]:
        """Emit one bounded FFN correction rule per lane."""

        conditions = self.conditions
        threshold = self.resolved_condition_threshold
        rules = []
        for lane in range(self.width):
            lane_name = self.lane_name(lane)
            rules.append(FFNRule.gated_write(
                name=f"{self.label}.lane_{lane}",
                conditions=conditions,
                threshold=threshold,
                gate_terms=((lane_name, -1.0),),
                gate_bias=self.target_value(lane),
                writes=((lane_name, 1.0),),
                scope=self.scope,
            ))
        return tuple(rules)


@dataclass(frozen=True)
class ScalarValueGuarantee:
    """Expected scalar value that can be restored by declarative FFN rules.

    This is the preferred exactness primitive for scalar nibble slots.  When
    the activation/read conditions prove the value, the emitted rule applies:

        value_dim += expected_value - value_dim

    so the symbolic result is exact and any neural drift is corrected at the
    declared boundary without consulting Python runtime state.
    """

    value_dim: str
    expected_value: float
    activation_conditions: Sequence[ConditionSpec]
    read_conditions: Sequence[ConditionSpec] = field(default_factory=tuple)
    condition_threshold: Optional[float] = None
    max_abs_weight: float = 16.0
    name: Optional[str] = None
    scope: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.value_dim, str) or not self.value_dim:
            raise ValueError("value_dim must be a non-empty string")
        _require_finite("expected_value", self.expected_value)
        _require_finite("max_abs_weight", self.max_abs_weight)
        if self.max_abs_weight <= 0.0:
            raise ValueError("max_abs_weight must be positive")

        activation_conditions = _normalize_conditions(
            self.activation_conditions,
            label="activation_conditions",
        )
        read_conditions = _normalize_conditions(
            self.read_conditions,
            label="read_conditions",
        )
        if not activation_conditions and not read_conditions:
            raise ValueError(
                "at least one activation or read condition is required"
            )
        object.__setattr__(
            self, "activation_conditions", activation_conditions
        )
        object.__setattr__(self, "read_conditions", read_conditions)

        threshold = self.resolved_condition_threshold
        _require_finite("condition_threshold", threshold)
        bounded_values = [
            ("condition_threshold", threshold),
            ("expected_value", self.expected_value),
            ("value_read_weight", -1.0),
            ("value_write_weight", 1.0),
        ]
        bounded_values.extend(
            (f"activation_conditions[{idx}].weight", weight)
            for idx, (_, weight) in enumerate(activation_conditions)
        )
        bounded_values.extend(
            (f"read_conditions[{idx}].weight", weight)
            for idx, (_, weight) in enumerate(read_conditions)
        )
        for label, value in bounded_values:
            if abs(value) > self.max_abs_weight:
                raise ValueError(
                    f"{label}={value!r} exceeds max_abs_weight "
                    f"{self.max_abs_weight!r}"
                )

    @property
    def label(self) -> str:
        return self.name or self.value_dim

    @property
    def conditions(self) -> Tuple[ConditionSpec, ...]:
        return tuple(self.activation_conditions) + tuple(self.read_conditions)

    @property
    def resolved_condition_threshold(self) -> float:
        if self.condition_threshold is not None:
            return float(self.condition_threshold)
        return _default_all_conditions_threshold(self.conditions)

    def to_ffn_rules(self) -> Tuple[FFNRule, ...]:
        return (
            FFNRule.gated_write(
                name=self.label,
                conditions=self.conditions,
                threshold=self.resolved_condition_threshold,
                gate_terms=((self.value_dim, -1.0),),
                gate_bias=float(self.expected_value),
                writes=((self.value_dim, 1.0),),
                scope=self.scope,
            ),
        )


def scalar_value_guarantee_rules(
    *,
    value_dim: str,
    expected_value: float,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    condition_threshold: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Build declarative FFN rules for an exact scalar value."""

    return ScalarValueGuarantee(
        value_dim=value_dim,
        expected_value=expected_value,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        max_abs_weight=max_abs_weight,
        name=name,
        scope=scope,
    ).to_ffn_rules()


def scalar_nibble_guarantee_rules(
    *,
    value_dim: str,
    expected_nibble: int,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    condition_threshold: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Build a scalar exactness rule for a nibble value in ``0..15``."""

    if (
        not isinstance(expected_nibble, int)
        or isinstance(expected_nibble, bool)
    ):
        raise ValueError("expected_nibble must be an integer")
    if not 0 <= expected_nibble <= 0xF:
        raise ValueError("expected_nibble must be in nibble range")
    return scalar_value_guarantee_rules(
        value_dim=value_dim,
        expected_value=float(expected_nibble),
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        max_abs_weight=max_abs_weight,
        name=name,
        scope=scope,
    )


def scalar_byte_guarantee_rules(
    *,
    low_value_dim: str,
    high_value_dim: str,
    expected_byte: int,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    condition_threshold: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Build scalar exactness rules for low/high nibble value slots."""

    if not isinstance(expected_byte, int) or isinstance(expected_byte, bool):
        raise ValueError("expected_byte must be an integer")
    if not 0 <= expected_byte <= 0xFF:
        raise ValueError("expected_byte must be in byte range")
    low_name = f"{name}.lo" if name else None
    high_name = f"{name}.hi" if name else None
    return scalar_nibble_guarantee_rules(
        value_dim=low_value_dim,
        expected_nibble=expected_byte & 0x0F,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        max_abs_weight=max_abs_weight,
        name=low_name,
        scope=scope,
    ) + scalar_nibble_guarantee_rules(
        value_dim=high_value_dim,
        expected_nibble=(expected_byte >> 4) & 0x0F,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        max_abs_weight=max_abs_weight,
        name=high_name,
        scope=scope,
    )


def one_hot_band_guarantee_rules(
    *,
    band_base: str,
    expected_index: int,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    width: int = 16,
    condition_threshold: Optional[float] = None,
    min_margin: float = 1.0,
    inactive_value: float = 0.0,
    active_value: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Build declarative FFN rules for a one-hot structural guarantee."""

    return OneHotBandGuarantee(
        band_base=band_base,
        expected_index=expected_index,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        width=width,
        condition_threshold=condition_threshold,
        min_margin=min_margin,
        inactive_value=inactive_value,
        active_value=active_value,
        max_abs_weight=max_abs_weight,
        name=name,
        scope=scope,
    ).to_ffn_rules()


def expected_nibble_guarantee_rules(
    *,
    band_base: str,
    expected_nibble: int,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    condition_threshold: Optional[float] = None,
    min_margin: float = 1.0,
    inactive_value: float = 0.0,
    active_value: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Convenience wrapper for the common 16-lane nibble-band case."""

    return one_hot_band_guarantee_rules(
        band_base=band_base,
        expected_index=expected_nibble,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        width=16,
        condition_threshold=condition_threshold,
        min_margin=min_margin,
        inactive_value=inactive_value,
        active_value=active_value,
        max_abs_weight=max_abs_weight,
        name=name,
        scope=scope,
    )


def expected_byte_guarantee_rules(
    *,
    expected_byte: int,
    activation_conditions: Sequence[ConditionSpec],
    read_conditions: Sequence[ConditionSpec] = (),
    low_band_base: str = "OUTPUT_LO",
    high_band_base: str = "OUTPUT_HI",
    condition_threshold: Optional[float] = None,
    min_margin: float = 1.0,
    inactive_value: float = 0.0,
    active_value: Optional[float] = None,
    max_abs_weight: float = 16.0,
    name: Optional[str] = None,
    scope: Optional[str] = None,
) -> Tuple[FFNRule, ...]:
    """Emit one-hot guarantees for legacy byte low/high nibble bands."""

    if not isinstance(expected_byte, int) or isinstance(expected_byte, bool):
        raise ValueError("expected_byte must be an integer")
    if not 0 <= expected_byte <= 0xFF:
        raise ValueError("expected_byte must be in byte range")

    low_name = f"{name}.lo" if name else None
    high_name = f"{name}.hi" if name else None
    low = expected_nibble_guarantee_rules(
        band_base=low_band_base,
        expected_nibble=expected_byte & 0x0F,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        min_margin=min_margin,
        inactive_value=inactive_value,
        active_value=active_value,
        max_abs_weight=max_abs_weight,
        name=low_name,
        scope=scope,
    )
    high = expected_nibble_guarantee_rules(
        band_base=high_band_base,
        expected_nibble=(expected_byte >> 4) & 0x0F,
        activation_conditions=activation_conditions,
        read_conditions=read_conditions,
        condition_threshold=condition_threshold,
        min_margin=min_margin,
        inactive_value=inactive_value,
        active_value=active_value,
        max_abs_weight=max_abs_weight,
        name=high_name,
        scope=scope,
    )
    return low + high


def _normalize_conditions(
    conditions: Sequence[ConditionSpec],
    *,
    label: str,
) -> Tuple[ConditionSpec, ...]:
    normalized = []
    for idx, condition in enumerate(conditions):
        if len(condition) != 2:
            raise ValueError(f"{label}[{idx}] must be a (dim, weight) pair")
        dim, weight = condition
        if not isinstance(dim, str) or not dim:
            raise ValueError(f"{label}[{idx}].dim must be a non-empty string")
        weight = float(weight)
        _require_finite(f"{label}[{idx}].weight", weight)
        normalized.append((dim, weight))
    return tuple(normalized)


def _default_all_conditions_threshold(
    conditions: Sequence[ConditionSpec],
) -> float:
    positive_weights = [weight for _, weight in conditions if weight > 0.0]
    if not positive_weights:
        raise ValueError(
            "condition_threshold is required when conditions have no "
            "positive weights"
        )
    return float(sum(positive_weights) - 0.5 * min(positive_weights))


def _require_finite(label: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


__all__ = [
    "ConditionSpec",
    "OneHotBandGuarantee",
    "ScalarValueGuarantee",
    "expected_byte_guarantee_rules",
    "expected_nibble_guarantee_rules",
    "one_hot_band_guarantee_rules",
    "scalar_byte_guarantee_rules",
    "scalar_nibble_guarantee_rules",
    "scalar_value_guarantee_rules",
]
