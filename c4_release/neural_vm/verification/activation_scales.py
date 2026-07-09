"""Per-dim runtime ACTIVATION-SCALE calibration datum (task #395).

The CONTROL-family derivation (docs/DERIVE_CONTROL_2026_07_09.md) proved that a
gate's BLOCKER magnitudes derive to a spec-structural safety factor, but the
gate's POSITIVE weights + threshold do NOT derive from the ISA identity alone:
they encode the model's per-dim RUNTIME ACTIVATION SCALES — the discriminator
dims (``MARK_PC`` / ``OP_BZ`` / ``CMP+k`` / ``HAS_SE`` / ...) do not all activate
at ``1.0`` in the residual stream that the gate reads.

Concretely, measured at the BZ-firing row (docs/DERIVE_ACTSCALE_2026_07_09.md):

    MARK_PC = 1.0   OP_BZ = 5.0   CMP+4 = 1.28   CMP+5 = 1.0   HAS_SE = 1.0

and the hand gate weights are

    MARK_PC = 1.0   OP_BZ = 0.2   CMP+4 = 1.0    CMP+5 = 1.0

so ``hand_weight(dim) == 1.0 / activation_scale(dim)`` (``OP_BZ`` weight ``0.2``
is exactly ``1/5``, normalizing its ``5x`` activation to a unit AND term). The
threshold ``3.5`` is ``n_pos - 0.5`` over the NORMALIZED (unit-scale) positive
contributions. That is the missing spec datum: a per-``(dim, position_class)``
activation scale, from which the whole POSITIVE side of the gate derives.

This module is the calibration datum's LOADER (analogous to
``backbone_bounds.BackboneBounds``). It is populated at BUILD time by
``tools/calibrate_activation_scales.py`` (a small representative-program
calibration forward that hooks each gate's block input and records the
characteristic residual activation of each discriminator dim), and consumed by
``ops.shared.derive_gate`` when ``C4_DERIVE_GATE_SCALES=1``.

The datum falls back to a CANONICAL analytic table (``_CANONICAL_SCALES``) baked
from the calibration measurement, so a fresh checkout with no ``.agent-logs``
JSON still derives the CONTROL gates correctly (the file is an OPTIONAL refresh,
not a build prerequisite — the golden build must never depend on a runtime
artifact).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# The canonical baked activation-scale table.
#
# Keyed by (dim_base_name, position_class). ``position_class`` mirrors the
# taxonomy in ``tools/observe_backbone_contributions.POSITION_CLASSES`` plus the
# gate-relevant ``mark==PC``. A value of ``s`` means "this dim's characteristic
# residual activation at rows of that class is ``s``" (so a unit AND term uses
# weight ``1/s``).
#
# These are the MEASURED scales (docs/DERIVE_ACTSCALE_2026_07_09.md §measurement)
# — NOT hand-tuned gate constants. They are the fallback when the .agent-logs
# calibration JSON is absent, so the derived gate is reproducible on a fresh
# checkout. The scales are STRUCTURAL properties of the compiled model (they come
# from the upstream amplitude the opcode/marker one-hots are baked at), so they
# are stable across programs — the calibration corpus only confirms them.
# ---------------------------------------------------------------------------

# The one-hot markers are baked at unit scale; the opcode one-hots are amplified
# upstream (the decode band writes them at a higher amplitude), so their residual
# scale is the datum. ``CMP+4/5`` (the AX-zero branch flags) sit ~1.0..1.3.
_CANONICAL_SCALES: Dict[str, float] = {
    # markers: clean unit one-hots at their own marker row
    "MARK_PC": 1.0,
    "MARK_AX": 1.0,
    "MARK_SP": 1.0,
    "MARK_BP": 1.0,
    "MARK_STACK0": 1.0,
    "MARK_MEM": 1.0,
    "MARK_SE": 1.0,
    "IS_BYTE": 1.0,
    # branch / step context flags
    "HAS_SE": 1.0,
    "CMP+4": 1.0,
    "CMP+5": 1.0,
    # opcode one-hots: amplified upstream — the measured residual scale
    "OP_BZ": 5.0,
    "OP_BNZ": 5.0,
    "OP_JMP": 5.0,
    "OP_JSR": 5.0,
}

# Default scale for any dim not in the table (treat as a clean unit one-hot).
_DEFAULT_SCALE = 1.0


@dataclass(frozen=True)
class ActivationScales:
    """Per-``(dim, position_class)`` runtime activation-scale table.

    ``scale(dim, position_class)`` returns the characteristic residual
    activation magnitude of ``dim`` at rows of ``position_class`` — the datum a
    ``derive_gate`` uses to set each positive AND weight to ``1/scale``.
    """

    version: int
    corpus_size: int
    model_commit: str
    # scales[dim][position_class] = float ; may be sparse.
    scales: Dict[str, Dict[str, float]]

    @classmethod
    def canonical(cls) -> "ActivationScales":
        """The baked analytic table (no calibration JSON needed).

        Position-class-agnostic: every dim maps its canonical scale under the
        wildcard class ``"*"`` (the gate derivation queries by dim, falling back
        to ``"*"``)."""
        return cls(
            version=0,
            corpus_size=0,
            model_commit="canonical",
            scales={d: {"*": s} for d, s in _CANONICAL_SCALES.items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActivationScales":
        with open(path) as f:
            data = json.load(f)
        return cls(
            version=int(data.get("version", 1)),
            corpus_size=int(data.get("corpus_size", 0)),
            model_commit=str(data.get("model_commit", "")),
            scales=data.get("scales", {}),
        )

    def scale(self, dim: str, position_class: str = "*") -> float:
        """Return the activation scale of ``dim`` at ``position_class``.

        Resolution order: exact ``(dim, position_class)`` -> ``(dim, "*")`` ->
        canonical baked scale -> ``_DEFAULT_SCALE``. A returned scale is always
        strictly positive (a non-positive / missing measurement falls back to
        the canonical / default, since a zero scale would make ``1/scale``
        explode).
        """
        by_dim = self.scales.get(dim)
        if by_dim:
            v = by_dim.get(position_class)
            if v is None:
                v = by_dim.get("*")
            if v is not None and float(v) > 0.0:
                return float(v)
        # fall back to canonical baked scale
        canon = _CANONICAL_SCALES.get(dim)
        if canon is not None and canon > 0.0:
            return float(canon)
        return _DEFAULT_SCALE

    def has(self, dim: str) -> bool:
        return dim in self.scales or dim in _CANONICAL_SCALES


def _default_calibration_path() -> Path:
    # __file__ -> .../c4_release/neural_vm/verification/activation_scales.py
    c4_root = Path(__file__).resolve().parent.parent.parent
    return c4_root / ".agent-logs" / "activation-scales" / "v1.json"


_CACHED: Optional[ActivationScales] = None


def load_activation_scales() -> ActivationScales:
    """Load the calibration JSON if present, else the canonical baked table.

    Cached process-wide (the datum is a build-time constant). The canonical
    fallback means the golden build never depends on a runtime artifact: a fresh
    checkout with no ``.agent-logs`` still derives every CONTROL gate correctly.
    Override the path via ``C4_ACTSCALE_JSON``.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    env_path = os.environ.get("C4_ACTSCALE_JSON")
    path = Path(env_path) if env_path else _default_calibration_path()
    if path.exists():
        try:
            _CACHED = ActivationScales.load(path)
            return _CACHED
        except (OSError, ValueError, KeyError):
            pass
    _CACHED = ActivationScales.canonical()
    return _CACHED
