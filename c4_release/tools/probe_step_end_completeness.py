"""Probe: at the STEP_END token position, is the residual state complete?

Architectural hypothesis (user directive, 2026-06-10):
    Heavy compute (ALU result, CMP combine, branch decisions) should fire
    at the STEP_END position of the 35-token VM step window, not at
    MARK_AX. By STEP_END the relayed OP_<NAME>, AX byte values, STACK0
    byte values, and ALU/CMP staging are all visible in the residual.

This probe RUNS the model on a fixed program, then for layers L8/L9/L10
prints a side-by-side comparison of the key dims at:

    * MARK_AX row  (row offset 5 within a step)
    * MARK_SE / STEP_END row  (row offset 34 within a step)

Reports the per-dim hot/cold status; lets us answer the architectural
question with measured numbers, not guesswork.

The test program is ``IMM 5; PSH; IMM 5; EQ; EXIT``. Due to a pre-existing
unrelated runner stop the EQ step does not fully execute today, but the
PSH and IMM steps DO emit clean MARK_AX and MARK_SE rows, which is
sufficient to characterise the relay structure.

Run::

    CUDA_VISIBLE_DEVICES=1 python c4_release/tools/probe_step_end_completeness.py
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


def _hot_indices(slice_, threshold: float = 0.1):
    return [
        (i, float(v.item()))
        for i, v in enumerate(slice_)
        if abs(v.item()) > threshold
    ]


def main() -> int:
    tracer = ResidualTracer()
    print("Running IMM 5; PSH; IMM 5; EQ; EXIT ...")
    capture = tracer.run("IMM 5; PSH; IMM 5; EQ; EXIT")
    print(f"seq_len={capture.seq_len} num_steps={capture.num_steps()} "
          f"result={capture.result!r}\n")

    # Locate MARK_AX and MARK_SE rows by scanning the residual.
    arr8 = capture.after_layer[8]
    mark_ax = capture.dim("MARK_AX")
    mark_se = capture.dim("MARK_SE")
    ax_rows = torch.nonzero(arr8[0, :, mark_ax] > 0.5).squeeze(-1).tolist()
    se_rows = torch.nonzero(arr8[0, :, mark_se] > 0.5).squeeze(-1).tolist()
    print(f"MARK_AX rows (step boundaries): {ax_rows}")
    print(f"MARK_SE rows (STEP_END boundaries): {se_rows}\n")

    # We compare at one pair of rows from the same step. The PSH step has
    # AX=5 carried-forward, so AX_CARRY_LO+5 should be hot at MARK_AX if
    # the carry-forward attention works correctly.
    if not ax_rows or not se_rows:
        print("Not enough boundaries captured; aborting.")
        return 1
    ax_row = ax_rows[1] if len(ax_rows) >= 2 else ax_rows[0]
    # SE for the SAME step (one row after the matching AX section: roughly
    # ax_row + 29 in the 35-token window).
    se_row = next((s for s in se_rows if s > ax_row), se_rows[-1])
    print(f"Comparing MARK_AX@{ax_row} vs MARK_SE@{se_row}\n")

    # --- DIM PROBES ---------------------------------------------------------
    probes = [
        # (label, layer, dim_spec, width, expected_at_AX, expected_at_SE)
        ("MARK_AX",        8, "MARK_AX",      1, "hot",  "cold"),
        ("MARK_SE",        8, "MARK_SE",      1, "cold", "hot"),
        ("MARK_SE_ONLY",   8, "MARK_SE_ONLY", 1, "cold", "hot"),
        ("HAS_SE",         8, "HAS_SE",       1, "?",    "hot"),
        ("OP_IMM",         8, "OP_IMM",       1, "hot when step=IMM", "cold"),
        ("OP_PSH",         8, "OP_PSH",       1, "hot when step=PSH", "cold"),
        ("OP_EQ",          8, "OP_EQ",        1, "hot when step=EQ",  "cold"),
        ("AX_CARRY_LO (slot)", 8, "AX_CARRY_LO", 16, "nibble hot",     "cold"),
        ("AX_CARRY_HI (slot)", 8, "AX_CARRY_HI", 16, "nibble hot",     "cold"),
        ("ALU_LO (slot)",      9, "ALU_LO",      16, "nibble hot (ALU step)", "cold"),
        ("ALU_HI (slot)",      9, "ALU_HI",      16, "nibble hot (ALU step)", "cold"),
        ("CMP (slot)",         9, "CMP",          4, "flag hot (CMP step)",   "cold"),
        ("STACK0_BYTE0",       8, "STACK0_BYTE0", 1, "cold (fires at byte row)", "cold"),
        ("NEXT_PC",            8, "NEXT_PC",      1, "cold", "hot (scheduler)"),
        # 2026-06-10: L0/L1 within-step register-presence broadcast.
        # SE_REG_AX_PRESENT is written by L1 head 6 (Q@MARK_SE_ONLY,
        # K@MARK_AX, V@MARK_AX, ALiBi slope 0.2). After L1 the SE row
        # carries the within-step MARK_AX presence ~= 1.0; the AX row
        # itself is unwritten (Q gates the broadcast to SE rows only).
        ("SE_REG_AX_PRESENT",  1, "SE_REG_AX_PRESENT", 1, "cold", "hot (within-step relay)"),
    ]

    print(f"{'probe':24}{'layer':>6}  {'@MARK_AX':<32}{'@MARK_SE':<32}{'expected':<30}")
    print("-" * 134)
    for label, layer, dim_spec, width, exp_ax, exp_se in probes:
        try:
            arr = capture.after_layer[layer]
            d = capture.dim(dim_spec)
            slice_ax = arr[0, ax_row, d:d+width]
            slice_se = arr[0, se_row, d:d+width]
            if width == 1:
                v_ax = float(slice_ax[0].item())
                v_se = float(slice_se[0].item())
                desc_ax = f"{v_ax:+.3f}"
                desc_se = f"{v_se:+.3f}"
            else:
                hot_ax = _hot_indices(slice_ax)
                hot_se = _hot_indices(slice_se)
                desc_ax = (str(hot_ax)[:30] or "[]")
                desc_se = (str(hot_se)[:30] or "[]")
            expected = f"AX={exp_ax} / SE={exp_se}"
            print(f"{label:24}{layer:>6}  {desc_ax:<32}{desc_se:<32}{expected:<30}")
        except Exception as exc:
            print(f"  {label}  ERROR: {exc!r}")

    # --- ARCHITECTURAL VERDICT ---------------------------------------------
    print("\n" + "=" * 70)
    print("ARCHITECTURAL VERDICT")
    print("=" * 70)
    op_imm_d = capture.dim("OP_IMM")
    op_psh_d = capture.dim("OP_PSH")
    op_eq_d = capture.dim("OP_EQ")
    ax_lo_d = capture.dim("AX_CARRY_LO")
    se_op_any = max(
        abs(arr8[0, se_row, op_imm_d].item()),
        abs(arr8[0, se_row, op_psh_d].item()),
        abs(arr8[0, se_row, op_eq_d].item()),
    )
    se_ax_any = float(arr8[0, se_row, ax_lo_d:ax_lo_d+16].abs().max().item())
    ax_op_any = max(
        abs(arr8[0, ax_row, op_imm_d].item()),
        abs(arr8[0, ax_row, op_psh_d].item()),
        abs(arr8[0, ax_row, op_eq_d].item()),
    )
    ax_ax_any = float(arr8[0, ax_row, ax_lo_d:ax_lo_d+16].abs().max().item())
    print(f"MARK_AX row {ax_row}:  max|OP_<NAME>| = {ax_op_any:.3f}, "
          f"max|AX_CARRY_LO| = {ax_ax_any:.3f}")
    print(f"MARK_SE row {se_row}:  max|OP_<NAME>| = {se_op_any:.3f}, "
          f"max|AX_CARRY_LO| = {se_ax_any:.3f}")
    if se_op_any < 0.5 and ax_op_any > 0.5:
        print("\nFINDING: OP_<NAME> flags are LIVE at MARK_AX, DEAD at STEP_END.")
        print("Migrating ALU/CMP rules to gate on STEP_END would require a")
        print("new attention head to relay OP_<NAME> from MARK_AX to STEP_END.")
    if se_ax_any < 0.5 and ax_ax_any > 0.5:
        print("FINDING: AX_CARRY_LO is LIVE at MARK_AX, DEAD at STEP_END.")
        print("Same: STEP_END is not currently the AX-carry-forward target.")
    print("\nThe STEP_END row currently carries scheduler dims (NEXT_PC, "
          "MARK_SE, HAS_SE, MARK_SE_ONLY, CONST), not compute substrate.")

    # --- L0/L1 within-step register-presence broadcast (2026-06-10) -----
    print("\n" + "=" * 70)
    print("L0/L1 WITHIN-STEP RELAY HEAD (L1 head 6)")
    print("=" * 70)
    arr_l1 = capture.after_layer[1]
    se_reg_ax_d = capture.dim("SE_REG_AX_PRESENT")
    se_reg_ax_at_se = float(arr_l1[0, se_row, se_reg_ax_d].item())
    se_reg_ax_at_ax = float(arr_l1[0, ax_row, se_reg_ax_d].item())
    print(f"SE_REG_AX_PRESENT@MARK_SE (L1): {se_reg_ax_at_se:+.3f}")
    print(f"SE_REG_AX_PRESENT@MARK_AX (L1): {se_reg_ax_at_ax:+.3f}")
    if se_reg_ax_at_se >= 0.5 and abs(se_reg_ax_at_ax) < 0.5:
        print("VERIFIED: L1 head 6 within-step relay broadcasts MARK_AX")
        print("presence to MARK_SE position (Q-gated to SE rows only).")
    else:
        print("WARNING: L1 head 6 broadcast did NOT land as expected. The")
        print("relay head spec or ALiBi slope may need tuning.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
