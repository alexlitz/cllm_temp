"""Probe: does the L9 step_end_operand_relay actually transmit AX -> SE?

PHASE 1 audit (Wave B migration). The L9 relay
(``make_layer9_step_end_operand_relay_op``) is registered and bakes
unconditionally, mirroring raw ALU_LO/HI / AX_CARRY_LO/HI / CMP /
CMP_GROUP / OP_<cmp> from the same step's MARK_AX row into the SE_-tagged
dims at MARK_SE_ONLY. The L9 CMP rules (commit 62b64449) read those SE_
mirrors. The 2026-06-12 ordering-engine docstring claims the relay
"never transmits" (SE_CMP_GROUP=0). This probe measures it directly.

Reads a clean-operand LT step (5 < 7) and dumps, per block, the SE_-tagged
mirror dims at the MARK_SE row vs the raw bands at MARK_AX. If the relay
works, SE_ALU_LO+5 (operand A low nibble = 5), SE_AX_CARRY_LO+7 (operand
B = 7), SE_CMP_GROUP+0, SE_OP_LT+0 should all be hot at the SE row from
the L9 attn block onward.

Run (CPU to dodge the shared-GPU OOM)::

    CUDA_VISIBLE_DEVICES="" python c4_release/tools/probe_l9_se_relay.py
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


def hot(slice_, thr=0.3):
    return [(i, round(float(v), 3)) for i, v in enumerate(slice_) if abs(float(v)) > thr]


def main() -> int:
    tracer = ResidualTracer()
    dimp = tracer.dim_positions
    n_blocks = len(tracer.model.blocks)
    print(f"n_blocks={n_blocks}")

    # LT step: operand A=5, B=7. 5<7 -> LT true.
    prog = "IMM 7; PSH; IMM 5; LT; EXIT"
    cap = tracer.run(prog)
    print(f"prog={prog!r} seq_len={cap.seq_len} result={cap.result!r}\n")

    mark_ax = dimp["MARK_AX"]
    mark_se = dimp["MARK_SE_ONLY"] if "MARK_SE_ONLY" in dimp else dimp["MARK_SE"]

    # Use the LAST captured block to locate rows (markers are stable).
    last_blk = max(cap.after_layer)
    arr = cap.after_layer[last_blk]
    ax_rows = torch.nonzero(arr[0, :, mark_ax] > 0.5).squeeze(-1).tolist()
    se_rows = torch.nonzero(arr[0, :, mark_se] > 0.5).squeeze(-1).tolist()
    print(f"MARK_AX rows: {ax_rows}")
    print(f"MARK_SE_ONLY rows: {se_rows}\n")

    # Pair AX->SE within one step (delta ~29).
    pairs = []
    for a in ax_rows:
        for s in se_rows:
            if 20 <= (s - a) <= 38:
                pairs.append((a, s))
    print(f"in-step (AX, SE) pairs: {pairs}\n")
    if not pairs:
        print("no in-step pair; abort")
        return 1
    # The LT step is the 4th VM step (IMM,PSH,IMM,LT) -> pick the last pair
    # (closest to EXIT) so we read the LT operands, not an IMM step.
    ax_row, se_row = pairs[-1]
    print(f"reading AX@{ax_row} / SE@{se_row} (the LT step row)\n")

    raw_dims = {
        "MARK_AX": (mark_ax, 1), "MARK_SE_ONLY": (mark_se, 1),
        "ALU_LO": (dimp["ALU_LO"], 16), "ALU_HI": (dimp["ALU_HI"], 16),
        "AX_CARRY_LO": (dimp["AX_CARRY_LO"], 16),
        "CMP_GROUP": (dimp.get("CMP_GROUP", -1), 1),
        "CMP": (dimp.get("CMP", -1), 4),
        "OP_LT": (dimp.get("OP_LT", -1), 1),
    }
    se_mirror_dims = {
        "SE_ALU_LO": (dimp.get("SE_ALU_LO", -1), 16),
        "SE_ALU_HI": (dimp.get("SE_ALU_HI", -1), 16),
        "SE_AX_CARRY_LO": (dimp.get("SE_AX_CARRY_LO", -1), 16),
        "SE_CMP": (dimp.get("SE_CMP", -1), 4),
        "SE_CMP_GROUP": (dimp.get("SE_CMP_GROUP", -1), 1),
        "SE_OP_LT": (dimp.get("SE_OP_LT", -1), 1),
    }

    blocks_to_probe = sorted(cap.after_layer)
    print("=== RAW bands at MARK_AX row (operand state) ===")
    for blk in blocks_to_probe:
        a = cap.after_layer[blk]
        line = []
        for name, (d, w) in raw_dims.items():
            if d < 0:
                continue
            h = hot(a[0, ax_row, d:d + w])
            if h:
                line.append(f"{name}={h}")
        if line:
            print(f"  blk{blk}: " + "  ".join(line))

    print("\n=== SE_-tagged mirror dims at MARK_SE row (relay output) ===")
    for blk in blocks_to_probe:
        a = cap.after_layer[blk]
        line = []
        for name, (d, w) in se_mirror_dims.items():
            if d < 0:
                continue
            h = hot(a[0, se_row, d:d + w])
            if h:
                line.append(f"{name}={h}")
        tag = ""
        print(f"  blk{blk}: " + ("  ".join(line) if line else "(all SE_ dims cold)") + tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
