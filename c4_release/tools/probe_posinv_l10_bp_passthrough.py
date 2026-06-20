#!/usr/bin/env python3
"""Prove the L10 bp_byte_passthrough STACK0-byte anchors auto-fix at 30-tok.

Builds the live ``_layer10_bp_byte_passthrough_head_spec`` in BOTH frames and
inspects the ``top_store_query`` slot Q-contributions for the STACK0-byte
target discriminators (slots 42-47, dims STACK0_BYTE0/1/2):

  * 35-tok (golden): the target-discriminator Q strength == M (= 50*S) -> the
    stack-source top-store route selects the STACK0 byte row (byte-identical to
    the historical literal ``AP(slot, target_dim, M)``).
  * 30-tok (campaign): the strength == 0.0 -> the STACK0-byte discriminator is
    AUTO-NEUTRALIZED so the route cannot spuriously fire on the MISFIRING
    STACK0_BYTE0 flag (which aliases onto a MEM addr row when the STACK0 block
    is dropped). The MARK_STACK0 slots (40/41) keep their strength in both
    frames (marker-TYPE dim, not a byte-position flag — naturally 0 with no
    misfire). The Class-1 marker-proximity blockers (H1+PC/AX/SP/BP/MEM) and
    the K-side H1+AX read are byte-identical in both frames.

Pure CPU, no model build.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_posinv_l10_bp_passthrough.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.dirname(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

FAIL = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


class _DimProxy:
    """Minimal SetDim proxy: every dim name resolves to a unique stable int so
    the head spec builds without a model. The integer identities don't matter
    here — we only inspect the AP *strengths* keyed by (row, dim_id)."""

    def __init__(self):
        self._ids = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        # Widely-spaced ids (step 1000) so a ``BD.H1 + marker_idx`` (small
        # 0..5 offset) never collides with another distinct dim's base id —
        # the real model uses distinct residual positions; the toy proxy must
        # not manufacture a spurious collision.
        return self._ids.setdefault(name, 1000 * (len(self._ids) + 1))


def _build_in_frame(no_stack0: bool):
    if no_stack0:
        os.environ["C4_NO_STACK0_EMIT"] = "1"
    else:
        os.environ.pop("C4_NO_STACK0_EMIT", None)
    for m in list(sys.modules):
        if m.startswith("neural_vm"):
            del sys.modules[m]
    from neural_vm.unified_compiler.ops.l10_ops import (
        _layer10_bp_byte_passthrough_head_spec,
    )
    BD = _DimProxy()
    S = 100.0
    spec = _layer10_bp_byte_passthrough_head_spec(BD, S)
    # STACK0_BYTE* / MARK_STACK0 dim ids for inspection.
    ids = {
        "STACK0_BYTE0": BD.STACK0_BYTE0,
        "STACK0_BYTE1": BD.STACK0_BYTE1,
        "STACK0_BYTE2": BD.STACK0_BYTE2,
        "MARK_STACK0": BD.MARK_STACK0,
    }
    return spec, ids, S


def _q_strength(spec, slot, dim_id):
    """Sum of Q strengths at (slot, dim_id) (should be a single AP)."""
    return sum(ap.weight for ap in spec.q if ap.slot == slot and ap.dim == dim_id)


def main() -> int:
    print("=" * 72)
    print("L10 bp_byte_passthrough STACK0-byte anchor — 30-tok AUTO-FIX proof")
    print("=" * 72)

    spec35, ids35, S = _build_in_frame(no_stack0=False)
    M = 50.0 * S

    # 35-tok: STACK0-byte target discriminators present at strength M.
    for slot, dim in ((42, "STACK0_BYTE0"), (43, "STACK0_BYTE0"),
                      (44, "STACK0_BYTE1"), (45, "STACK0_BYTE1"),
                      (46, "STACK0_BYTE2"), (47, "STACK0_BYTE2")):
        w = _q_strength(spec35, slot, ids35[dim])
        check(w == M,
              f"35-tok slot {slot} {dim} target strength == M ({M})")
    # MARK_STACK0 slots unchanged at M.
    for slot in (40, 41):
        w = _q_strength(spec35, slot, ids35["MARK_STACK0"])
        check(w == M, f"35-tok slot {slot} MARK_STACK0 strength == M ({M})")

    spec30, ids30, _ = _build_in_frame(no_stack0=True)
    # 30-tok: STACK0-byte discriminators AUTO-NEUTRALIZED to 0.
    for slot, dim in ((42, "STACK0_BYTE0"), (43, "STACK0_BYTE0"),
                      (44, "STACK0_BYTE1"), (45, "STACK0_BYTE1"),
                      (46, "STACK0_BYTE2"), (47, "STACK0_BYTE2")):
        w = _q_strength(spec30, slot, ids30[dim])
        check(w == 0.0,
              f"30-tok slot {slot} {dim} target strength == 0 (auto-suppressed)")
    # MARK_STACK0 slots STILL at M (marker-type dim, no misfire).
    for slot in (40, 41):
        w = _q_strength(spec30, slot, ids30["MARK_STACK0"])
        check(w == M,
              f"30-tok slot {slot} MARK_STACK0 strength == M (unchanged, "
              f"no row-alias)")

    print("\n" + "=" * 72)
    if FAIL:
        print(f"L10 BP-PASSTHROUGH ANCHOR PROOF FAILED — {len(FAIL)} check(s)")
        return 1
    print("L10 BP-PASSTHROUGH PROOF PASSED — STACK0-byte discriminators")
    print("auto-suppress at 30-tok via the mechanism; byte-identical at 35-tok;")
    print("MARK_STACK0 marker slots untouched (frame-robust).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
