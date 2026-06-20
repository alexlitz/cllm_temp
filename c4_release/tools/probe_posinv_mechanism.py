#!/usr/bin/env python3
"""Unit-prove the positional-invariant mechanism in isolation (no model build).

Demonstrates the two make-or-break properties WITHOUT a GPU or a model bake:

  (1) NO-OP at STEP_TOKENS=35: every helper returns exactly the literal it
      replaces, so the lowered weights stay byte-identical to golden.
  (2) AUTO-SHIFT at STEP_TOKENS=30 with ZERO hand-tuning: the same call site,
      with no env branch, returns the suppressed threshold for an anchor whose
      target byte was dropped, and the live one otherwise.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_posinv_mechanism.py
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

from neural_vm.unified_compiler.positional_invariant import (  # noqa: E402
    marker_bank_index,
    frame_byte_is_emitted,
    invariant_threshold,
    STEP_TOKENS_FULL,
    STEP_TOKENS_DROPPED,
)

FAIL = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main() -> int:
    print("=" * 70)
    print("POSITIONAL-INVARIANT MECHANISM — isolation proof")
    print("=" * 70)

    print("\n[A] marker_bank_index is frame-invariant (Class-1: MEM_I = 4)")
    for st in (STEP_TOKENS_FULL, STEP_TOKENS_DROPPED):
        check(marker_bank_index("MEM", step_tokens=st) == 4,
              f"MEM bank slot == 4 at STEP_TOKENS={st} (was literal MEM_I=4)")
    check(
        marker_bank_index("MEM", step_tokens=35)
        == marker_bank_index("MEM", step_tokens=30),
        "MEM_I identical across frames (auto-resolved, no hand-coded literal)",
    )
    check(
        [marker_bank_index(m, step_tokens=30)
         for m in ("PC", "AX", "SP", "BP", "MEM", "SE")] == [0, 1, 2, 3, 4, 5],
        "full bank order 0..5 reproduces the l0_ops PC_I..SE_I literals",
    )

    print("\n[B] frame_byte_is_emitted — the STACK0_BYTE0 d=6-from-BP anchor")
    # STACK0 byte 0 is d=6 from BP (BP byte3 d=4, STACK0 marker d=5, bytes d=6..9)
    check(frame_byte_is_emitted("BP", 6, step_tokens=35) is True,
          "d=6-from-BP (STACK0 byte0) IS emitted at 35-tok")
    check(frame_byte_is_emitted("BP", 6, step_tokens=30) is False,
          "d=6-from-BP (STACK0 byte0) is DROPPED at 30-tok -> auto-suppress")
    # A non-STACK0 anchor (MEM-relative) is emitted in both frames.
    check(frame_byte_is_emitted("MEM", 1, step_tokens=30) is True,
          "d=1-from-MEM (MEM addr byte0) stays emitted at 30-tok (Class-1)")

    print("\n[C] invariant_threshold replaces the hand-coded env branch")
    LIVE, SUPPRESS = 1.5, 1.0e9
    # 35-tok: byte-identical no-op -> returns LIVE exactly.
    t35 = invariant_threshold(LIVE, SUPPRESS, "BP", 6, step_tokens=35)
    check(t35 == LIVE,
          f"35-tok threshold == {LIVE} (byte-identical to golden literal)")
    # 30-tok: AUTO-neutralizes with NO env branch at the call site.
    t30 = invariant_threshold(LIVE, SUPPRESS, "BP", 6, step_tokens=30)
    check(t30 == SUPPRESS,
          f"30-tok threshold == {SUPPRESS} (auto-suppressed, zero hand-tuning)")
    check(
        t30 == (1.0e9 if STEP_TOKENS_DROPPED == 30 else 1.5),
        "matches the hand-coded '1e9 if no_stack0_emit_enabled() else 1.5' "
        "outcome EXACTLY — mechanism reproduces the manual fix automatically",
    )

    print("\n" + "=" * 70)
    if FAIL:
        print(f"MECHANISM PROOF FAILED — {len(FAIL)} check(s) failed")
        for m in FAIL:
            print("   -", m)
        return 1
    print("MECHANISM PROOF PASSED — no-op at 35-tok, auto-shift at 30-tok, "
          "zero hand-tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
