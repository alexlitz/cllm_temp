"""Structural numeric contracts for declarative one-hot bands.

These checks are intentionally value-level rather than weight-level: compiler
ops can declare that a produced nibble band should be one-hot with a minimum
winner margin, then run this pass over symbolic outputs or residual slices to
get structured violations and optional projection hooks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:  # Torch is a project dependency, but keep import-time failures local.
    import torch
except ImportError:  # pragma: no cover - exercised only in stripped envs.
    torch = None


ACTIVE_MARGIN_LOW = "active_margin_low"
INACTIVE_TOO_HIGH = "inactive_too_high"
MISSING_VALUE = "missing_value"
INVALID_VALUE = "invalid_value"
SCALAR_VALUE_DRIFT = "scalar_value_drift"
WIDTH_MISMATCH = "width_mismatch"

PROJECTED = "projected"
AMBIGUOUS_WINNER = "ambiguous_winner"
UNEXPECTED_WINNER = "unexpected_winner"
UNREADABLE_BAND = "unreadable_band"

_MISSING = object()


@dataclass(frozen=True)
class OneHotBandContract:
    """Expected one-hot shape for a named structural band.

    ``min_active_margin`` is checked as:

        value[expected_index] - max(value[other_index]) >= min_active_margin

    ``max_inactive_value`` is checked independently on every non-selected
    lane. ``tolerance`` relaxes both comparisons by a small amount.
    """

    band_base: str
    width: int
    expected_index: int
    min_active_margin: float
    max_inactive_value: float
    tolerance: float = 0.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if not 0 <= self.expected_index < self.width:
            raise ValueError("expected_index must be within band width")
        if self.min_active_margin < 0.0:
            raise ValueError("min_active_margin must be non-negative")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        for attr in ("min_active_margin", "max_inactive_value", "tolerance"):
            value = float(getattr(self, attr))
            if not math.isfinite(value):
                raise ValueError(f"{attr} must be finite")

    @property
    def label(self) -> str:
        return self.name or self.band_base

    def dim_name(self, index: int) -> str:
        return f"{self.band_base}+{index}"


@dataclass(frozen=True)
class ScalarValueContract:
    """Expected scalar value for a named structural dimension.

    The canonical mapping key is ``f"{dim_base}+{dim_index}"``. The verifier
    also accepts the same bracket/tuple lane keys used by band contracts.
    """

    dim_base: str
    expected_value: float
    tolerance: float = 0.0
    dim_index: int = 0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.dim_index < 0:
            raise ValueError("dim_index must be non-negative")
        for attr in ("expected_value", "tolerance"):
            value = float(getattr(self, attr))
            if not math.isfinite(value):
                raise ValueError(f"{attr} must be finite")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    @property
    def label(self) -> str:
        return self.name or self.dim_name

    @property
    def band_base(self) -> str:
        return self.dim_base

    @property
    def dim_name(self) -> str:
        return f"{self.dim_base}+{self.dim_index}"


@dataclass(frozen=True)
class BandViolation:
    """A structured contract failure for one band lane or band shape."""

    contract_name: str
    band_base: str
    kind: str
    index: int | None
    observed: float | None
    limit: float | None
    details: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        index = "*" if self.index is None else str(self.index)
        observed = "missing" if self.observed is None else f"{self.observed:.6g}"
        limit = "n/a" if self.limit is None else f"{self.limit:.6g}"
        return (
            f"{self.contract_name}[{index}] {self.kind}: "
            f"observed={observed} limit={limit}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "band_base": self.band_base,
            "kind": self.kind,
            "index": self.index,
            "observed": self.observed,
            "limit": self.limit,
            "details": dict(self.details),
        }


@dataclass
class BandVerificationReport:
    """Verification result for a single ``OneHotBandContract``."""

    contract: OneHotBandContract
    values: tuple[float | None, ...]
    violations: list[BandViolation] = field(default_factory=list)
    winner_index: int | None = None
    winner_value: float | None = None
    runner_up_value: float | None = None
    winner_margin: float | None = None

    @property
    def ok(self) -> bool:
        return not self.violations

    def has_violations(self) -> bool:
        return bool(self.violations)

    def violations_by_kind(self) -> dict[str, list[BandViolation]]:
        by_kind: dict[str, list[BandViolation]] = {}
        for violation in self.violations:
            by_kind.setdefault(violation.kind, []).append(violation)
        return by_kind

    def format(self) -> str:
        lines = [
            f"=== Band contract report: {self.contract.label} ===",
            f"Band: {self.contract.band_base}",
            f"Width: {self.contract.width}",
            f"Expected index: {self.contract.expected_index}",
            f"Violations: {len(self.violations)}",
        ]
        if self.winner_index is not None:
            lines.append(
                f"Winner: index={self.winner_index} "
                f"value={self.winner_value:.6g} margin={self.winner_margin:.6g}"
            )
        for violation in self.violations:
            lines.append(f"  {violation.format()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": _contract_as_dict(self.contract),
            "ok": self.ok,
            "values": list(self.values),
            "winner_index": self.winner_index,
            "winner_value": self.winner_value,
            "runner_up_value": self.runner_up_value,
            "winner_margin": self.winner_margin,
            "violations": [
                violation.as_dict()
                for violation in self.violations
            ],
        }


@dataclass
class ScalarValueVerificationReport:
    """Verification result for a single ``ScalarValueContract``."""

    contract: ScalarValueContract
    value: float | None
    violations: list[BandViolation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def observed_delta(self) -> float | None:
        if self.value is None:
            return None
        return self.value - self.contract.expected_value

    def has_violations(self) -> bool:
        return bool(self.violations)

    def violations_by_kind(self) -> dict[str, list[BandViolation]]:
        by_kind: dict[str, list[BandViolation]] = {}
        for violation in self.violations:
            by_kind.setdefault(violation.kind, []).append(violation)
        return by_kind

    def format(self) -> str:
        observed = "missing" if self.value is None else f"{self.value:.6g}"
        delta = (
            "n/a"
            if self.observed_delta is None
            else f"{self.observed_delta:+.6g}"
        )
        lines = [
            f"=== Scalar value contract report: {self.contract.label} ===",
            f"Dim: {self.contract.dim_name}",
            f"Expected value: {self.contract.expected_value:.6g}",
            f"Observed value: {observed}",
            f"Delta: {delta}",
            f"Tolerance: {self.contract.tolerance:.6g}",
            f"Violations: {len(self.violations)}",
        ]
        for violation in self.violations:
            lines.append(f"  {violation.format()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": _contract_as_dict(self.contract),
            "ok": self.ok,
            "value": self.value,
            "observed_delta": self.observed_delta,
            "violations": [
                violation.as_dict()
                for violation in self.violations
            ],
        }


@dataclass
class BandContractPassReport:
    """Aggregate result for checking several band contracts."""

    reports: list[BandVerificationReport] = field(default_factory=list)

    @property
    def violations(self) -> list[BandViolation]:
        return [
            violation
            for report in self.reports
            for violation in report.violations
        ]

    @property
    def ok(self) -> bool:
        return not self.violations

    def has_violations(self) -> bool:
        return bool(self.violations)

    def format(self) -> str:
        lines = [
            "=== Band contract pass report ===",
            f"Contracts checked: {len(self.reports)}",
            f"Violations: {len(self.violations)}",
        ]
        for report in self.reports:
            status = "OK" if report.ok else "DRIFT"
            lines.append(
                f"  [{status}] {report.contract.label}: "
                f"violations={len(report.violations)}"
            )
            for violation in report.violations:
                lines.append(f"     {violation.format()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "contracts_checked": len(self.reports),
            "reports": [report.as_dict() for report in self.reports],
            "violations": [
                violation.as_dict()
                for violation in self.violations
            ],
        }


@dataclass
class ScalarValueContractPassReport:
    """Aggregate result for checking several scalar-value contracts."""

    reports: list[ScalarValueVerificationReport] = field(default_factory=list)

    @property
    def violations(self) -> list[BandViolation]:
        return [
            violation
            for report in self.reports
            for violation in report.violations
        ]

    @property
    def ok(self) -> bool:
        return not self.violations

    def has_violations(self) -> bool:
        return bool(self.violations)

    def format(self) -> str:
        lines = [
            "=== Scalar value contract pass report ===",
            f"Contracts checked: {len(self.reports)}",
            f"Violations: {len(self.violations)}",
        ]
        for report in self.reports:
            status = "OK" if report.ok else "DRIFT"
            observed = (
                "missing"
                if report.value is None
                else f"{report.value:.6g}"
            )
            lines.append(
                f"  [{status}] {report.contract.label}: "
                f"expected={report.contract.expected_value:.6g} "
                f"observed={observed} violations={len(report.violations)}"
            )
            for violation in report.violations:
                lines.append(f"     {violation.format()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "contracts_checked": len(self.reports),
            "reports": [report.as_dict() for report in self.reports],
            "violations": [
                violation.as_dict()
                for violation in self.violations
            ],
        }


@dataclass
class BandProjectionResult:
    """Outcome from projecting a band to an exact one-hot representation."""

    contract: OneHotBandContract
    status: str
    winner_index: int | None
    projected_values: tuple[float, ...] | None
    verification: BandVerificationReport
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == PROJECTED

    def has_projection(self) -> bool:
        return self.projected_values is not None

    def as_symbolic_writes(self) -> dict[str, float]:
        if self.projected_values is None:
            return {}
        return {
            self.contract.dim_name(index): value
            for index, value in enumerate(self.projected_values)
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": _contract_as_dict(self.contract),
            "status": self.status,
            "winner_index": self.winner_index,
            "projected_values": (
                None
                if self.projected_values is None
                else list(self.projected_values)
            ),
            "symbolic_writes": self.as_symbolic_writes(),
            "reason": self.reason,
            "verification": self.verification.as_dict(),
        }


@dataclass
class DeclaredNibbleBandReport:
    """Verification result for an expected byte's output nibble bands.

    This is the diagnostic bridge between symbolic declarations and neural
    residuals: symbolic execution declares an expected byte token, and the
    neural side must expose one-hot ``OUTPUT_LO`` and ``OUTPUT_HI`` bands
    with explicit margins.
    """

    expected_byte: int
    contract_report: BandContractPassReport
    projections: list[BandProjectionResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.contract_report.ok

    @property
    def violations(self) -> list[BandViolation]:
        return self.contract_report.violations

    @property
    def expected_low_nibble(self) -> int:
        return self.expected_byte & 0xF

    @property
    def expected_high_nibble(self) -> int:
        return (self.expected_byte >> 4) & 0xF

    def has_violations(self) -> bool:
        return self.contract_report.has_violations()

    def format(self) -> str:
        lines = [
            "=== Declared output nibble band report ===",
            f"Expected byte: 0x{self.expected_byte:02x}",
            f"Expected low/high: {self.expected_low_nibble}/{self.expected_high_nibble}",
            self.contract_report.format(),
        ]
        if self.projections:
            lines.append("Projection diagnostics:")
            for projection in self.projections:
                reason = (
                    ""
                    if projection.reason is None
                    else f" reason={projection.reason}"
                )
                lines.append(
                    f"  [{projection.status}] {projection.contract.label}: "
                    f"winner={projection.winner_index}{reason}"
                )
        return "\n".join(lines)

    def format_inline(self, *, max_violations: int = 4) -> str:
        status = "OK" if self.ok else "DRIFT"
        pieces = [
            f"band_contracts={status}",
            f"expected_byte=0x{self.expected_byte:02x}",
        ]
        for report in self.contract_report.reports:
            margin = (
                "n/a"
                if report.winner_margin is None
                else f"{report.winner_margin:+.3g}"
            )
            pieces.append(
                f"{report.contract.label}:expected={report.contract.expected_index} "
                f"winner={report.winner_index} margin={margin}"
            )
        if self.violations:
            rendered = [
                _format_inline_violation(violation)
                for violation in self.violations[:max_violations]
            ]
            if len(self.violations) > max_violations:
                rendered.append(f"+{len(self.violations) - max_violations} more")
            pieces.append("violations=" + ";".join(rendered))
        if self.projections:
            projection_bits = [
                (
                    f"{projection.contract.label}:{projection.status}"
                    f" winner={projection.winner_index}"
                )
                for projection in self.projections
            ]
            pieces.append("projection_diag=" + ";".join(projection_bits))
        return " ".join(pieces)

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_byte": self.expected_byte,
            "expected_low_nibble": self.expected_low_nibble,
            "expected_high_nibble": self.expected_high_nibble,
            "ok": self.ok,
            "contract_report": self.contract_report.as_dict(),
            "violations": [
                violation.as_dict()
                for violation in self.violations
            ],
            "projections": [
                projection.as_dict()
                for projection in self.projections
            ],
        }


def verify_one_hot_band(
    output: Any,
    contract: OneHotBandContract,
) -> BandVerificationReport:
    """Verify one vector/tensor/mapping output against a band contract."""

    extracted = _extract_band_values(output, contract)
    violations = _read_violations(contract, extracted)
    ranking = _rank_values(extracted.values)
    winner_index = ranking[0][0] if ranking else None
    winner_value = ranking[0][1] if ranking else None
    runner_up_value = ranking[1][1] if len(ranking) > 1 else None
    if winner_value is None:
        winner_margin = None
    elif runner_up_value is None:
        winner_margin = math.inf
    else:
        winner_margin = winner_value - runner_up_value

    if contract.expected_index not in extracted.missing_indices and (
        contract.expected_index not in extracted.invalid_indices
    ):
        active_value = extracted.values[contract.expected_index]
        if active_value is not None:
            inactive_values = [
                (idx, value)
                for idx, value in enumerate(extracted.values)
                if idx != contract.expected_index and value is not None
            ]
            if inactive_values:
                best_inactive_idx, best_inactive_value = max(
                    inactive_values,
                    key=lambda item: item[1],
                )
                active_margin = active_value - best_inactive_value
            else:
                best_inactive_idx = None
                best_inactive_value = -math.inf
                active_margin = math.inf
            if active_margin + contract.tolerance < contract.min_active_margin:
                violations.append(BandViolation(
                    contract_name=contract.label,
                    band_base=contract.band_base,
                    kind=ACTIVE_MARGIN_LOW,
                    index=contract.expected_index,
                    observed=active_margin,
                    limit=contract.min_active_margin,
                    details={
                        "active_value": active_value,
                        "best_inactive_index": float(best_inactive_idx or 0),
                        "best_inactive_value": best_inactive_value,
                    },
                ))

    for idx, value in enumerate(extracted.values):
        if idx == contract.expected_index or value is None:
            continue
        if value > contract.max_inactive_value + contract.tolerance:
            violations.append(BandViolation(
                contract_name=contract.label,
                band_base=contract.band_base,
                kind=INACTIVE_TOO_HIGH,
                index=idx,
                observed=value,
                limit=contract.max_inactive_value,
            ))

    return BandVerificationReport(
        contract=contract,
        values=extracted.values,
        violations=violations,
        winner_index=winner_index,
        winner_value=winner_value,
        runner_up_value=runner_up_value,
        winner_margin=winner_margin,
    )


def verify_band_contracts(
    output: Any,
    contracts: Iterable[OneHotBandContract],
) -> BandContractPassReport:
    """Run the structural numeric contract pass for several bands."""

    return BandContractPassReport(
        reports=[
            verify_one_hot_band(output, contract)
            for contract in contracts
        ]
    )


def verify_scalar_value(
    output: Any,
    contract: ScalarValueContract,
) -> ScalarValueVerificationReport:
    """Verify one scalar dimension against an expected nibble/byte value."""

    value, status = _extract_scalar_value(output, contract)
    violations: list[BandViolation] = []
    if status == MISSING_VALUE:
        violations.append(BandViolation(
            contract_name=contract.label,
            band_base=contract.dim_base,
            kind=MISSING_VALUE,
            index=contract.dim_index,
            observed=None,
            limit=None,
        ))
    elif status == INVALID_VALUE:
        violations.append(BandViolation(
            contract_name=contract.label,
            band_base=contract.dim_base,
            kind=INVALID_VALUE,
            index=contract.dim_index,
            observed=None,
            limit=None,
        ))
    elif value is not None:
        delta = abs(value - contract.expected_value)
        if delta > contract.tolerance:
            violations.append(BandViolation(
                contract_name=contract.label,
                band_base=contract.dim_base,
                kind=SCALAR_VALUE_DRIFT,
                index=contract.dim_index,
                observed=value,
                limit=contract.expected_value,
                details={
                    "expected_value": contract.expected_value,
                    "absolute_delta": delta,
                    "tolerance": contract.tolerance,
                },
            ))

    return ScalarValueVerificationReport(
        contract=contract,
        value=value,
        violations=violations,
    )


def verify_scalar_value_contracts(
    output: Any,
    contracts: Iterable[ScalarValueContract],
) -> ScalarValueContractPassReport:
    """Run the scalar-value numeric contract pass for several dimensions."""

    return ScalarValueContractPassReport(
        reports=[
            verify_scalar_value(output, contract)
            for contract in contracts
        ]
    )


def declared_output_byte_scalar_contracts(
    expected_byte: int,
    *,
    low_dim: str = "OUTPUT_LO_SCALAR",
    high_dim: str = "OUTPUT_HI_SCALAR",
    tolerance: float = 1e-6,
) -> tuple[ScalarValueContract, ScalarValueContract]:
    """Build low/high scalar contracts for a symbolically expected byte."""

    expected_byte = int(expected_byte) & 0xFF
    return (
        ScalarValueContract(
            name=f"{low_dim}[expected_low_scalar]",
            dim_base=low_dim,
            expected_value=float(expected_byte & 0xF),
            tolerance=tolerance,
        ),
        ScalarValueContract(
            name=f"{high_dim}[expected_high_scalar]",
            dim_base=high_dim,
            expected_value=float((expected_byte >> 4) & 0xF),
            tolerance=tolerance,
        ),
    )


def declared_output_byte_contracts(
    expected_byte: int,
    *,
    low_band: str = "OUTPUT_LO",
    high_band: str = "OUTPUT_HI",
    min_active_margin: float = 0.5,
    max_inactive_value: float = 0.2,
    tolerance: float = 1e-6,
) -> tuple[OneHotBandContract, OneHotBandContract]:
    """Build low/high nibble contracts for a symbolically expected byte."""

    expected_byte = int(expected_byte) & 0xFF
    return (
        OneHotBandContract(
            name=f"{low_band}[expected_low]",
            band_base=low_band,
            width=16,
            expected_index=expected_byte & 0xF,
            min_active_margin=min_active_margin,
            max_inactive_value=max_inactive_value,
            tolerance=tolerance,
        ),
        OneHotBandContract(
            name=f"{high_band}[expected_high]",
            band_base=high_band,
            width=16,
            expected_index=(expected_byte >> 4) & 0xF,
            min_active_margin=min_active_margin,
            max_inactive_value=max_inactive_value,
            tolerance=tolerance,
        ),
    )


def verify_declared_output_nibble_bands(
    output: Any,
    expected_byte: int,
    *,
    low_band: str = "OUTPUT_LO",
    high_band: str = "OUTPUT_HI",
    min_active_margin: float = 0.5,
    max_inactive_value: float = 0.2,
    tolerance: float = 1e-6,
    include_projection: bool = False,
    require_expected_winner: bool = True,
) -> DeclaredNibbleBandReport:
    """Verify neural output nibble bands against a symbolic byte declaration.

    ``include_projection`` is diagnostic-only. It reports what exact one-hot
    correction would be selected if the band has a clear winner; callers must
    not feed the projection back into strict neural execution.
    """

    contracts = declared_output_byte_contracts(
        expected_byte,
        low_band=low_band,
        high_band=high_band,
        min_active_margin=min_active_margin,
        max_inactive_value=max_inactive_value,
        tolerance=tolerance,
    )
    contract_report = verify_band_contracts(output, contracts)
    projections = (
        [
            project_one_hot_band(
                output,
                contract,
                require_expected_winner=require_expected_winner,
            )
            for contract in contracts
        ]
        if include_projection
        else []
    )
    return DeclaredNibbleBandReport(
        expected_byte=int(expected_byte) & 0xFF,
        contract_report=contract_report,
        projections=projections,
    )


def project_one_hot_band(
    output: Any,
    contract: OneHotBandContract,
    *,
    active_value: float | None = None,
    inactive_value: float | None = None,
    require_expected_winner: bool = True,
) -> BandProjectionResult:
    """Project a readable band to exact one-hot values if the winner is clear.

    By default the clear winner must be the declared ``expected_index``. Set
    ``require_expected_winner=False`` for a pure discretization hook that
    accepts whichever lane won by margin.
    """

    verification = verify_one_hot_band(output, contract)
    blocking_kinds = {MISSING_VALUE, INVALID_VALUE, WIDTH_MISMATCH}
    if any(violation.kind in blocking_kinds for violation in verification.violations):
        return BandProjectionResult(
            contract=contract,
            status=UNREADABLE_BAND,
            winner_index=verification.winner_index,
            projected_values=None,
            verification=verification,
            reason="band has missing, invalid, or width-mismatched values",
        )

    if verification.winner_index is None or verification.winner_margin is None:
        return BandProjectionResult(
            contract=contract,
            status=UNREADABLE_BAND,
            winner_index=None,
            projected_values=None,
            verification=verification,
            reason="band has no readable winner",
        )

    if verification.winner_margin + contract.tolerance < contract.min_active_margin:
        return BandProjectionResult(
            contract=contract,
            status=AMBIGUOUS_WINNER,
            winner_index=verification.winner_index,
            projected_values=None,
            verification=verification,
            reason="winner margin is below contract minimum",
        )

    if require_expected_winner and verification.winner_index != contract.expected_index:
        return BandProjectionResult(
            contract=contract,
            status=UNEXPECTED_WINNER,
            winner_index=verification.winner_index,
            projected_values=None,
            verification=verification,
            reason="winner does not match expected index",
        )

    if inactive_value is None:
        inactive_value = min(0.0, contract.max_inactive_value)
    if active_value is None:
        active_value = max(1.0, inactive_value + contract.min_active_margin)

    winner = verification.winner_index
    projected = tuple(
        active_value if idx == winner else inactive_value
        for idx in range(contract.width)
    )
    return BandProjectionResult(
        contract=contract,
        status=PROJECTED,
        winner_index=winner,
        projected_values=projected,
        verification=verification,
    )


@dataclass(frozen=True)
class _ExtractedBand:
    values: tuple[float | None, ...]
    missing_indices: tuple[int, ...] = ()
    invalid_indices: tuple[int, ...] = ()
    observed_width: int | None = None


def _read_violations(
    contract: OneHotBandContract,
    extracted: _ExtractedBand,
) -> list[BandViolation]:
    violations: list[BandViolation] = []
    if (
        extracted.observed_width is not None
        and extracted.observed_width != contract.width
    ):
        violations.append(BandViolation(
            contract_name=contract.label,
            band_base=contract.band_base,
            kind=WIDTH_MISMATCH,
            index=None,
            observed=float(extracted.observed_width),
            limit=float(contract.width),
        ))
    for idx in extracted.missing_indices:
        violations.append(BandViolation(
            contract_name=contract.label,
            band_base=contract.band_base,
            kind=MISSING_VALUE,
            index=idx,
            observed=None,
            limit=None,
        ))
    for idx in extracted.invalid_indices:
        violations.append(BandViolation(
            contract_name=contract.label,
            band_base=contract.band_base,
            kind=INVALID_VALUE,
            index=idx,
            observed=None,
            limit=None,
        ))
    return violations


def _extract_band_values(output: Any, contract: OneHotBandContract) -> _ExtractedBand:
    if isinstance(output, Mapping):
        return _extract_from_mapping(output, contract)
    return _extract_from_sequence(_flatten_sequence(output), contract)


def _extract_from_mapping(
    output: Mapping[Any, Any],
    contract: OneHotBandContract,
) -> _ExtractedBand:
    if contract.band_base in output:
        band = _flatten_sequence(output[contract.band_base])
        if band is not None:
            return _extract_from_sequence(band, contract)

    values: list[float | None] = []
    missing: list[int] = []
    invalid: list[int] = []
    for idx in range(contract.width):
        raw = _lookup_band_lane(output, contract, idx)
        if raw is _MISSING:
            values.append(None)
            missing.append(idx)
            continue
        value = _coerce_scalar(raw)
        values.append(value)
        if value is None:
            invalid.append(idx)
    return _ExtractedBand(
        values=tuple(values),
        missing_indices=tuple(missing),
        invalid_indices=tuple(invalid),
        observed_width=contract.width,
    )


def _lookup_band_lane(
    output: Mapping[Any, Any],
    contract: OneHotBandContract,
    index: int,
) -> Any:
    for key in (
        contract.dim_name(index),
        f"{contract.band_base}[{index}]",
        (contract.band_base, index),
    ):
        if key in output:
            return output[key]
    return _MISSING


def _extract_scalar_value(
    output: Any,
    contract: ScalarValueContract,
) -> tuple[float | None, str | None]:
    if isinstance(output, Mapping):
        raw = _lookup_scalar_dim(output, contract)
        if raw is _MISSING:
            return None, MISSING_VALUE
        value = _coerce_scalar(raw)
        if value is None:
            return None, INVALID_VALUE
        return value, None

    values = _flatten_sequence(output)
    if values is None or contract.dim_index >= len(values):
        return None, MISSING_VALUE
    value = _coerce_scalar(values[contract.dim_index])
    if value is None:
        return None, INVALID_VALUE
    return value, None


def _lookup_scalar_dim(
    output: Mapping[Any, Any],
    contract: ScalarValueContract,
) -> Any:
    for key in (
        contract.dim_name,
        f"{contract.dim_base}[{contract.dim_index}]",
        (contract.dim_base, contract.dim_index),
    ):
        if key in output:
            return output[key]
    if contract.dim_base in output:
        raw = output[contract.dim_base]
        direct = _coerce_scalar(raw)
        if direct is not None:
            return raw
        values = _flatten_sequence(raw)
        if values is not None and contract.dim_index < len(values):
            return values[contract.dim_index]
        if values is None:
            return raw
    return _MISSING


def _extract_from_sequence(
    values: Sequence[Any] | None,
    contract: OneHotBandContract,
) -> _ExtractedBand:
    if values is None:
        return _ExtractedBand(
            values=tuple(None for _ in range(contract.width)),
            missing_indices=tuple(range(contract.width)),
            observed_width=None,
        )

    observed_width = len(values)
    coerced: list[float | None] = []
    missing: list[int] = []
    invalid: list[int] = []
    for idx in range(contract.width):
        if idx >= observed_width:
            coerced.append(None)
            missing.append(idx)
            continue
        value = _coerce_scalar(values[idx])
        coerced.append(value)
        if value is None:
            invalid.append(idx)
    return _ExtractedBand(
        values=tuple(coerced),
        missing_indices=tuple(missing),
        invalid_indices=tuple(invalid),
        observed_width=observed_width,
    )


def _flatten_sequence(value: Any) -> Sequence[Any] | None:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, Sequence):
        return value
    if hasattr(value, "reshape") and hasattr(value, "tolist"):
        try:
            return value.reshape(-1).tolist()
        except Exception:
            pass
    if hasattr(value, "flatten") and hasattr(value, "tolist"):
        try:
            return value.flatten().tolist()
        except Exception:
            pass
    return None


def _coerce_scalar(value: Any) -> float | None:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif hasattr(value, "item") and not isinstance(value, (int, float)):
        try:
            value = value.item()
        except Exception:
            return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _rank_values(
    values: Sequence[float | None],
) -> list[tuple[int, float]]:
    ranked = [
        (idx, value)
        for idx, value in enumerate(values)
        if value is not None
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _contract_as_dict(
    contract: OneHotBandContract | ScalarValueContract,
) -> dict[str, Any]:
    if isinstance(contract, ScalarValueContract):
        return {
            "dim_base": contract.dim_base,
            "dim_index": contract.dim_index,
            "dim_name": contract.dim_name,
            "expected_value": contract.expected_value,
            "tolerance": contract.tolerance,
            "name": contract.name,
            "label": contract.label,
        }
    return {
        "band_base": contract.band_base,
        "width": contract.width,
        "expected_index": contract.expected_index,
        "min_active_margin": contract.min_active_margin,
        "max_inactive_value": contract.max_inactive_value,
        "tolerance": contract.tolerance,
        "name": contract.name,
        "label": contract.label,
    }


def _format_inline_violation(violation: BandViolation) -> str:
    index = "*" if violation.index is None else str(violation.index)
    observed = (
        "missing"
        if violation.observed is None
        else f"{violation.observed:.3g}"
    )
    limit = "n/a" if violation.limit is None else f"{violation.limit:.3g}"
    return (
        f"{violation.band_base}[{index}]/{violation.kind}"
        f":obs={observed}:limit={limit}"
    )
