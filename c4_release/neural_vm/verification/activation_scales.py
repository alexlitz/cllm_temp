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
    # opcode one-hots at the POST-L9 branch gate: MEASURED amplified residual
    # scale ~5.0 (the decode band writes the branch opcode one-hot at a higher
    # amplitude at the L22 post-L9 block the BZ/BNZ gate reads — the hand
    # OP_BZ/OP_BNZ weight 0.2 == 1/5.0 is exactly this). See
    # docs/DERIVE_ACTSCALE_2026_07_09.md §measurement.
    "OP_BZ": 5.0,
    "OP_BNZ": 5.0,
    # OP_JMP / OP_JSR are NOT amplified at the L6 all-step / first-step JMP gate
    # blocks — the MEASUREMENT (probe_measure_scales) shows OP_JMP == 0 at those
    # blocks in the corpus, i.e. the live JMP PC path is ELSEWHERE and those three
    # pc_mux override bands are DEAD reserved bands (docs/DERIVE_CONTROL §1). Their
    # gate-local scale is left at the default 1.0 (the ``"*"`` wildcard) so the
    # deadness check keeps them dead (hand threshold 4.5/5.0/5.5 > raw max-fire
    # ~2.0 at unit scale). The per-block amplified opcode scale is in the
    # PER-CLASS table below (an opcode reads 5.2 at ``mark==AX`` but the L6 JMP
    # override band's OP_JMP reads ~0 at its all-step block — the SAME dim, a
    # DIFFERENT scale per block, which is exactly why the scale is class-keyed).
    "MARK_SE_ONLY": 1.0,
    "PSH_AT_SP": 1.0,
}

# ---------------------------------------------------------------------------
# PER-(position_class) scale overrides (task #452 gate rollout).
#
# The flat table above is the ``"*"`` wildcard (each dim's default scale). A gate
# reading a dim at a SPECIFIC block/row-class can see a DIFFERENT amplitude — the
# per-block-keying the ACTSCALE doc §6 flagged. The decisive case: an opcode
# one-hot reads ~0 at the L6 all-step JMP override block (dead reserved band, so
# ``OP_JMP`` stays 1.0 under ``"*"`` and the L6 deadness holds) but reads ~5.2 at
# the ``mark==AX`` marker row where the L16 branch/frame + L10 CMP-combine
# correctors fire. MEASURED (docs/GATE_ROLLOUT_2026_07_09.md): ``OP_LEA`` /
# ``OP_ENT`` == 5.23 (mode over 100+ AX rows). So the AX-marker corrector gates'
# hand ``0.2`` opcode weight == ``1/5.2`` == ``1/scale(OP_*, "mark==AX")`` — the
# SAME reciprocal the BZ gate proved, now at the AX class. Keyed by class so the
# L6 ``"*"`` deadness and the L16 ``mark==AX`` liveness coexist on one datum.
# ---------------------------------------------------------------------------
_AMPLIFIED_OPCODE_SCALE = 5.2
# Every opcode one-hot the AX-/SP-marker correctors read. The opcode flag is
# BROADCAST in-step to every marker row (the L5 decode band + the Wave-A
# step-end relay carry it), so it reads the SAME ~5.2 amplified plateau at the
# AX and SP marker rows (MEASURED at mark==AX; the SP row shares the broadcast).
_AMPLIFIED_OPCODES = (
    "OP_LEA", "OP_ADD", "OP_SUB", "OP_ADJ", "OP_ENT", "OP_LEV",
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_PSH", "OP_SI", "OP_SC", "OP_LI", "OP_IMM", "OP_JMP", "OP_JSR",
)
_CANONICAL_CLASS_SCALES: Dict[str, Dict[str, float]] = {
    "mark==AX": {d: _AMPLIFIED_OPCODE_SCALE for d in _AMPLIFIED_OPCODES},
    "mark==SP": {d: _AMPLIFIED_OPCODE_SCALE for d in _AMPLIFIED_OPCODES},
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

        Each dim maps its canonical scale under the wildcard class ``"*"`` (the
        gate-derivation default) PLUS any per-``position_class`` overrides
        (``_CANONICAL_CLASS_SCALES``) — e.g. an opcode one-hot reads ``5.2`` at
        ``mark==AX`` but stays ``1.0`` under ``"*"`` (dead at the L6 all-step JMP
        block). The class-keyed entry wins for a class-specific query; ``"*"``
        is the fallback (see :meth:`scale`)."""
        scales: Dict[str, Dict[str, float]] = {
            d: {"*": s} for d, s in _CANONICAL_SCALES.items()
        }
        for cls_name, per_dim in _CANONICAL_CLASS_SCALES.items():
            for d, s in per_dim.items():
                scales.setdefault(d, {})[cls_name] = s
        return cls(
            version=0,
            corpus_size=0,
            model_commit="canonical",
            scales=scales,
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
        # fall back to canonical baked scale — a per-``position_class`` override
        # (``_CANONICAL_CLASS_SCALES``, e.g. an opcode at ``mark==AX`` == 5.2)
        # takes precedence over the flat ``"*"`` scale for a class-specific query.
        canon_cls = _CANONICAL_CLASS_SCALES.get(position_class, {})
        cv = canon_cls.get(dim)
        if cv is not None and cv > 0.0:
            return float(cv)
        canon = _CANONICAL_SCALES.get(dim)
        if canon is not None and canon > 0.0:
            return float(canon)
        return _DEFAULT_SCALE

    def has(self, dim: str) -> bool:
        return (
            dim in self.scales
            or dim in _CANONICAL_SCALES
            or any(dim in per for per in _CANONICAL_CLASS_SCALES.values())
        )


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
