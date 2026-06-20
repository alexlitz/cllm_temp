#!/usr/bin/env python3
"""Prove the L1 STACK0_BYTE0 anchor auto-fixes in the 30-tok frame.

Builds the live ``_threshold_ffn_rules`` list (the actual rule the lowering
consumes) in BOTH frames and inspects the STACK0_BYTE0 unit's threshold:

  * 35-tok (golden): threshold == 1.5  -> the flag fires on STACK0 byte 0
    (byte-identical to the historical literal).
  * 30-tok (campaign): threshold == 1e9 -> the flag is auto-NEUTRALIZED so it
    cannot mis-fire onto MEM addr byte 0 (the shifted d=6-from-BP row),
    reproducing the prior hand-coded ``no_stack0_emit_enabled()`` fix with NO
    per-op branch.

Then it confirms the prior HAND-CODED logic and the NEW MECHANISM agree in both
frames, so the prototype is a behavioural drop-in. Pure CPU, no model build.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_posinv_l1_anchor.py
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


def _stack0_threshold_in_frame(no_stack0: bool) -> float:
    """Fresh-import the L1 op module in the requested frame and return the
    STACK0_BYTE0 rule's threshold as the live lowering would see it."""
    if no_stack0:
        os.environ["C4_NO_STACK0_EMIT"] = "1"
    else:
        os.environ.pop("C4_NO_STACK0_EMIT", None)
    for m in list(sys.modules):
        if m.startswith("neural_vm"):
            del sys.modules[m]
    from neural_vm.unified_compiler.ops.l1_ops import _threshold_ffn_rules
    rules = _threshold_ffn_rules(S=100.0)
    stack0 = next(r for r in rules if r.name == "stack0_byte0_flag")
    return float(stack0.threshold)


def main() -> int:
    print("=" * 70)
    print("L1 STACK0_BYTE0 anchor — 30-tok AUTO-FIX proof (real rule list)")
    print("=" * 70)

    t35 = _stack0_threshold_in_frame(no_stack0=False)
    t30 = _stack0_threshold_in_frame(no_stack0=True)

    print(f"\n  STACK0_BYTE0 rule threshold  35-tok = {t35}")
    print(f"  STACK0_BYTE0 rule threshold  30-tok = {t30}")

    check(t35 == 1.5,
          "35-tok threshold == 1.5  (byte-identical to golden literal)")
    check(t30 == 1.0e9,
          "30-tok threshold == 1e9  (AUTO-suppressed, zero hand-tuning)")
    check(t35 != t30,
          "mechanism COMPUTED the frame-dependent shift (35 fires / 30 dark)")

    # The mechanism must reproduce EXACTLY what the deleted hand-coded
    # ``1e9 if no_stack0_emit_enabled() else 1.5`` produced.
    for no_stack0, expected in ((False, 1.5), (True, 1.0e9)):
        got = _stack0_threshold_in_frame(no_stack0=no_stack0)
        check(got == expected,
              f"frame no_stack0={no_stack0}: mechanism == old hand-fix "
              f"({expected})")

    print("\n" + "=" * 70)
    if FAIL:
        print(f"L1 ANCHOR PROOF FAILED — {len(FAIL)} check(s)")
        return 1
    print("L1 ANCHOR PROOF PASSED — anchor auto-suppresses in the 30-tok frame")
    print("via the mechanism, byte-identical to golden in the 35-tok frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
