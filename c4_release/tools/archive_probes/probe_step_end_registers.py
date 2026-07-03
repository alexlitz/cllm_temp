"""Probe: at MARK_SE (STEP_END), are all 5 register-presence flags hot?

Architectural verification for the L1 head 6 within-step
all-register-presence broadcast (2026-06-10, commit
``feat(l1): broadcast all registers to STEP_END dims``).

The head Q-anchors on ``MARK_SE_ONLY`` and K-anchors on the OR of
the 5 register markers (``MARK_AX``, ``MARK_PC``, ``MARK_SP``,
``MARK_BP``, ``MARK_STACK0``); it broadcasts each marker's presence
into the matching ``SE_REG_<NAME>_PRESENT`` slot at the SE row, so
that L11+ STEP_END consumers can gate on within-step register
liveness without re-reading the marker rows.

This probe RUNS the model on a fixed program, then for each SE row
prints whether all 5 ``SE_REG_<NAME>_PRESENT`` slots are hot
(>= 0.5). Reports per-row hot/cold breakdown and a final pass/fail
verdict.

The test program is ``IMM 5; PSH; IMM 5; EQ; EXIT`` (4 steps; the
first SE row is the program-start CS, which has no preceding
markers and is expected to be cold -- so the probe checks SE rows
from step 1 onward).

Run::

    CUDA_VISIBLE_DEVICES=1 python c4_release/tools/probe_step_end_registers.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PROJ_ROOT = os.path.dirname(REPO_ROOT)
for _p in (PROJ_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from c4_release.tools.capture_residual_trace import ResidualTracer  # noqa: E402


_REGISTERS = ("AX", "PC", "SP", "BP", "STACK0")
_PRESENCE_THRESHOLD = 0.5


def main() -> int:
    tracer = ResidualTracer()
    print("Running IMM 5; PSH; IMM 5; EQ; EXIT ...")
    capture = tracer.run("IMM 5; PSH; IMM 5; EQ; EXIT")
    print(
        f"seq_len={capture.seq_len} num_steps={capture.num_steps()} "
        f"result={capture.result!r}\n"
    )

    arr_l1 = capture.after_layer[1]
    arr8 = capture.after_layer[8]

    mark_se = capture.dim("MARK_SE")
    se_rows = torch.nonzero(arr8[0, :, mark_se] > 0.5).squeeze(-1).tolist()
    print(f"MARK_SE rows: {se_rows}\n")

    # Resolve each SE_REG_<NAME>_PRESENT dim once.
    dims = {name: capture.dim(f"SE_REG_{name}_PRESENT") for name in _REGISTERS}

    # Header.
    hdr = f"{'SE row':>8}  " + "  ".join(f"{name:>10}" for name in _REGISTERS)
    print(hdr)
    print("-" * len(hdr))
    for r in se_rows:
        cols = [
            f"{float(arr_l1[0, r, dims[name]].item()):+10.3f}"
            for name in _REGISTERS
        ]
        print(f"{r:>8}  " + "  ".join(cols))

    # Acceptance: every SE row that has at least one register marker
    # before it within the in-step window (35 tokens) must have ALL 5
    # SE_REG_<NAME>_PRESENT >= threshold. The first SE row is the
    # program-start CS boundary -- has no preceding markers and is
    # expected to be cold.
    print("\n" + "=" * 70)
    print("ACCEPTANCE")
    print("=" * 70)
    print(
        f"Each SE_REG_<NAME>_PRESENT >= {_PRESENCE_THRESHOLD} "
        f"at every SE row past the program-start CS."
    )
    mark_dims = {
        name: capture.dim(f"MARK_{name}") for name in _REGISTERS
    }
    step_se_rows = []
    for r in se_rows:
        in_step_lo = max(r - 35, 0)
        any_marker = False
        for name in _REGISTERS:
            d = mark_dims[name]
            if float(arr8[0, in_step_lo:r, d].abs().max().item()) > 0.5:
                any_marker = True
                break
        if any_marker:
            step_se_rows.append(r)

    if not step_se_rows:
        print("No SE rows with preceding markers found; aborting.")
        return 1

    print(f"Step SE rows under test: {step_se_rows}\n")
    ok = True
    for r in step_se_rows:
        for name in _REGISTERS:
            v = float(arr_l1[0, r, dims[name]].item())
            status = "OK" if v >= _PRESENCE_THRESHOLD else "MISS"
            if v < _PRESENCE_THRESHOLD:
                ok = False
                print(
                    f"  SE_REG_{name}_PRESENT @ row {r}: "
                    f"{v:+.3f} ({status})"
                )
    if ok:
        print("PASS: all 5 SE_REG_<NAME>_PRESENT slots hot at every step SE row.")
        return 0
    print("FAIL: at least one SE_REG_<NAME>_PRESENT slot was cold.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
