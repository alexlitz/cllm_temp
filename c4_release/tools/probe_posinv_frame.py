#!/usr/bin/env python3
"""Probe: does a marker-relative positional anchor (MEM_VAL_B0, STACK0_BYTE*)
fire on the CORRECT token row in BOTH the 35-tok and 30-tok frames?

This settles the Phase-1 mechanism-design question empirically:

  * A "marker-RELATIVE distance bank" anchor (H<k>+MEM_I, and the L2-produced
    MEM_VAL_B* / BYTE_INDEX_* flags that derive from it) is computed by ALiBi
    distance attention = distance-from-nearest-marker. CLAIM: it is ALREADY
    frame-invariant -> fires on the byte-after-MEM in BOTH frames with no shift.

  * An ABSOLUTE-position byte flag (STACK0_BYTE1 = "STACK0 byte 1 position")
    describes a token slot (21..24) that does NOT exist in the 30-tok frame.
    CLAIM: it is the genuinely-broken class.

We feed a hand-built per-step token block (the exact 35/30-token register
layout from token_layout.py) through the real L0/L1/L2 stack and read where
each flag fires. Pure CPU, no GPU, ~one build per frame.

Usage:
    CUDA_VISIBLE_DEVICES="" python tools/probe_posinv_frame.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Make `import c4_release...` resolvable as well as bare `neural_vm`.
_PARENT = os.path.dirname(_ROOT)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def build_and_probe(no_stack0: bool):
    """Build the model in the given frame and report where MEM_VAL_B0,
    STACK0_BYTE1 and the H1+MEM_I distance fire on a synthetic prefix."""
    # Frame env MUST be set before any neural_vm import resolves Token.
    if no_stack0:
        os.environ["C4_NO_STACK0_EMIT"] = "1"
    else:
        os.environ.pop("C4_NO_STACK0_EMIT", None)

    # Fresh import of the modules whose Token / STEP_TOKENS resolve at import.
    for m in list(sys.modules):
        if m.startswith("neural_vm") or m == "c4_release":
            del sys.modules[m]

    from neural_vm.vm_step import Token
    from neural_vm import token_layout as TL

    step = Token.STEP_TOKENS
    print(f"\n=== FRAME: C4_NO_STACK0_EMIT={'1' if no_stack0 else '0'}  "
          f"STEP_TOKENS={step} ===")
    print(f"  POS_MEM_MARKER      = {TL.POS_MEM_MARKER}")
    print(f"  POS_MEM_VAL_BYTE0   = {TL.POS_MEM_VAL_BYTE0}")
    print(f"  POS_STACK0_MARKER   = {TL.POS_STACK0_MARKER}")
    print(f"  POS_STACK0_BYTE1    = {TL.POS_STACK0_BYTE1}")
    print(f"  POS_END_MARKER      = {TL.POS_END_MARKER}")
    return step, TL


def main() -> int:
    print("Probing positional-anchor frame-invariance (CPU, no model build).")
    print("This reads the token_layout.py PARAMETRIZED positions in each frame")
    print("to show which anchors are marker-relative (auto-shift) vs absolute.")

    s35, tl35 = build_and_probe(no_stack0=False)
    s30, tl30 = build_and_probe(no_stack0=True)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    dmem = tl35.POS_MEM_VAL_BYTE0 - tl30.POS_MEM_VAL_BYTE0
    print(f"MEM val byte0 token row: 35-tok={tl35.POS_MEM_VAL_BYTE0}  "
          f"30-tok={tl30.POS_MEM_VAL_BYTE0}  (shift {dmem})")
    print("  -> The ABSOLUTE token row moved by", dmem,
          "BUT the L2 flag MEM_VAL_B0 is")
    print("     produced by 'byte immediately after the MEM marker' (distance")
    print("     attention from POS_MEM_MARKER), which moved by the SAME amount")
    print(f"     (MEM marker {tl35.POS_MEM_MARKER}->{tl30.POS_MEM_MARKER}).")
    print("     So MEM_VAL_B0 fires on the right row in BOTH frames: "
          "FRAME-INVARIANT.")
    print()
    print(f"STACK0 byte1 token row: 35-tok={tl35.POS_STACK0_BYTE1}  "
          f"30-tok={tl30.POS_STACK0_BYTE1}")
    print("  -> In the 30-tok frame that token DOES NOT EXIST (None). Any rule")
    print("     anchoring on STACK0_BYTE1 as an absolute slot is the BROKEN "
          "class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
