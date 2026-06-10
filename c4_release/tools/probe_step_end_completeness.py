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


def _locate_wave_a_relay_layer(model, dim_positions) -> int:
    """Find the model block where Wave A's step_end_operand_relay landed.

    Searches for a block whose attention W_q has
    ``W_q[head_idx*HD, MARK_SE_ONLY] >= 50`` — the signature of the
    Wave A relay's Q-side anchor. Falls back to L11 (the spec-declared
    target) if no match is found, so the probe stays usable in
    pre-Wave-A builds.
    """

    se_only = dim_positions["MARK_SE_ONLY"]
    for li, block in enumerate(model.blocks):
        attn = block.attn
        wq = attn.W_q.to_dense() if attn.W_q.is_sparse_csr else attn.W_q
        HD = wq.shape[0] // attn.num_heads
        for head_idx in range(attn.num_heads):
            row = head_idx * HD
            if row < wq.shape[0] and abs(wq[row, se_only].item()) >= 50.0:
                return li
    return 11


def _enable_wave_a_relay(tracer) -> int:
    """Bake the Wave A relay heads into the model for probe purposes.

    The production op (``make_layer11_step_end_operand_relay_op``) is
    registered with ``enable=False`` so the relay's downstream side
    effects don't regress baseline smoke. To observe the relay's effect
    at MARK_SE, this helper bakes the same head specs into the block
    that the dep graph resolved as the L11 anchor. Returns the block
    index where the relay landed.

    Search heuristic: pick the block whose FFN ``W_down[TEMP+0, 0]``
    matches the L11 ``layer11_mul_partial`` writeback signature
    (~ 10/S = 0.1). The relay must run at that same block to be
    byte-compatible with the production binding.
    """
    import torch
    from c4_release.neural_vm.unified_compiler.ops.l11_ops import (
        _layer11_step_end_operand_relay_head_specs,
    )
    from c4_release.neural_vm.unified_compiler.ops.shared import _as_setdim_proxy
    from c4_release.neural_vm.unified_compiler.primitives import Primitives

    dim_positions = tracer.dim_positions
    temp_d = dim_positions["TEMP"]
    # Identify the L11 anchor block by its MUL partial signature.
    anchor_idx = None
    for li, block in enumerate(tracer.model.blocks):
        wdown = getattr(block.ffn, "W_down", None)
        if not isinstance(wdown, torch.Tensor):
            continue
        if wdown.is_sparse_csr:
            wdown = wdown.to_dense()
        if (
            wdown.shape[0] > temp_d
            and wdown.shape[1] > 0
            and abs(wdown[temp_d, 0].item()) > 0.05
        ):
            anchor_idx = li
            break
    # Pick the first block AT OR AFTER the anchor whose attention is
    # currently empty (no nonzero Q rows). Avoids overwriting an
    # already-baked attention head.
    relay_block_idx = None
    start = anchor_idx if anchor_idx is not None else 11
    for li in range(start, len(tracer.model.blocks)):
        attn = tracer.model.blocks[li].attn
        wq = attn.W_q.to_dense() if attn.W_q.is_sparse_csr else attn.W_q
        if int((wq != 0).any(dim=1).sum().item()) == 0:
            relay_block_idx = li
            break
    if relay_block_idx is None:
        relay_block_idx = 11
    attn = tracer.model.blocks[relay_block_idx].attn
    # CSR sparse-inference conversion may have rewritten W_*; densify
    # back so the spec writes land in standard nn.Parameter tensors.
    from torch import nn
    for wname in ("W_q", "W_k", "W_v", "W_o"):
        w = getattr(attn, wname)
        if w.is_sparse_csr:
            setattr(attn, wname, nn.Parameter(w.to_dense()))
    HD = attn.W_q.shape[0] // attn.num_heads
    proxy = _as_setdim_proxy(dim_positions)
    spec_a, spec_b = _layer11_step_end_operand_relay_head_specs(
        proxy, 100.0, 0, 1,
    )
    Primitives.generate_attention_head(attn, spec_a, HD)
    Primitives.generate_attention_head(attn, spec_b, HD)
    return relay_block_idx


def main() -> int:
    tracer = ResidualTracer()
    relay_block = _enable_wave_a_relay(tracer)
    print(f"Wave A relay baked into block L{relay_block} (probe-only)")
    RELAY_LAYER = _locate_wave_a_relay_layer(tracer.model, tracer.dim_positions)
    print(f"Wave A relay detected at L{RELAY_LAYER}")
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
    # Collect every (ax, se) pair where ``se = ax + ~29`` (in-step
    # MARK_AX -> MARK_SE delta). The architectural verdict takes the
    # max relayed magnitude across pairs so an ALU/CMP step's
    # contribution to ALU_LO / CMP at MARK_SE is captured even if
    # earlier steps don't fire those dims.
    expected_delta = 29
    step_pairs = []
    for a in ax_rows:
        for s in se_rows:
            d = s - a
            if d <= 0:
                continue
            if abs(d - expected_delta) <= 5:
                step_pairs.append((a, s))
    if not step_pairs:
        print("Could not pair AX and SE rows within one step; aborting.")
        return 1
    ax_row, se_row = step_pairs[0]
    print(f"Step pairs (AX, SE) captured: {step_pairs}")
    print(
        f"Reporting first pair: MARK_AX@{ax_row} vs MARK_SE@{se_row} "
        f"(delta={se_row - ax_row})\n"
    )

    # --- DIM PROBES ---------------------------------------------------------
    # Wave A (docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md) lands the
    # ``layer11_step_end_operand_relay`` head at L11, so the relay's
    # outputs at MARK_SE only appear from ``after_layer[11]`` onward.
    # OP_<NAME> / AX_CARRY / STACK0_BYTE probes pull L11; ALU_LO/HI / CMP
    # are written at L9 then relayed at L11, so they also pull L11 to see
    # the relayed copy at MARK_SE. The ``MARK_*`` and ``NEXT_PC`` rows
    # stay at L8 since those are scheduler-side anchors unchanged by the
    # relay.
    probes = [
        # (label, layer, dim_spec, width, expected_at_AX, expected_at_SE)
        ("MARK_AX",        8, "MARK_AX",      1, "hot",  "cold"),
        ("MARK_SE",        8, "MARK_SE",      1, "cold", "hot"),
        ("MARK_SE_ONLY",   8, "MARK_SE_ONLY", 1, "cold", "hot"),
        ("HAS_SE",         8, "HAS_SE",       1, "?",    "hot"),
        ("OP_IMM",        RELAY_LAYER, "OP_IMM",       1, "hot when step=IMM", "hot when step=IMM (Wave A)"),
        ("OP_PSH",        RELAY_LAYER, "OP_PSH",       1, "hot when step=PSH", "hot when step=PSH (Wave A)"),
        ("OP_EQ",         RELAY_LAYER, "OP_EQ",        1, "hot when step=EQ",  "hot when step=EQ (Wave A)"),
        ("AX_CARRY_LO (slot)", RELAY_LAYER, "AX_CARRY_LO", 16, "nibble hot",  "nibble hot (Wave A)"),
        ("AX_CARRY_HI (slot)", RELAY_LAYER, "AX_CARRY_HI", 16, "nibble hot",  "nibble hot (Wave A)"),
        ("ALU_LO (slot)",      RELAY_LAYER, "ALU_LO",      16, "nibble hot (ALU step)", "nibble hot (Wave A)"),
        ("ALU_HI (slot)",      RELAY_LAYER, "ALU_HI",      16, "nibble hot (ALU step)", "nibble hot (Wave A)"),
        ("CMP (slot)",         RELAY_LAYER, "CMP",          4, "flag hot (CMP step)",   "flag hot (Wave A)"),
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
    print("ARCHITECTURAL VERDICT (Wave A: layer11_step_end_operand_relay)")
    print("=" * 70)
    op_imm_d = capture.dim("OP_IMM")
    op_psh_d = capture.dim("OP_PSH")
    op_eq_d = capture.dim("OP_EQ")
    ax_lo_d = capture.dim("AX_CARRY_LO")
    alu_lo_d = capture.dim("ALU_LO")
    cmp_d = capture.dim("CMP")
    stack0_b0_d = capture.dim("STACK0_BYTE0")
    arr_pre = capture.after_layer[8]            # before the Wave A relay
    arr_post = capture.after_layer[RELAY_LAYER]  # after the Wave A relay

    def _max_op(arr, row):
        return max(
            abs(arr[0, row, op_imm_d].item()),
            abs(arr[0, row, op_psh_d].item()),
            abs(arr[0, row, op_eq_d].item()),
        )

    def _max_band(arr, row, base, width):
        return float(arr[0, row, base:base + width].abs().max().item())

    # Take the max relay magnitude across ALL captured step pairs so an
    # ALU/CMP step's contribution to ALU_LO / CMP at its MARK_SE is
    # caught even when an earlier step doesn't fire those dims.
    se_op_pre = max(_max_op(arr_pre, s) for _, s in step_pairs)
    se_ax_pre = max(_max_band(arr_pre, s, ax_lo_d, 16) for _, s in step_pairs)
    se_op_post = max(_max_op(arr_post, s) for _, s in step_pairs)
    se_ax_post = max(_max_band(arr_post, s, ax_lo_d, 16) for _, s in step_pairs)
    se_alu_post = max(_max_band(arr_post, s, alu_lo_d, 16) for _, s in step_pairs)
    se_cmp_post = max(_max_band(arr_post, s, cmp_d, 4) for _, s in step_pairs)
    se_stack0_post = max(_max_band(arr_post, s, stack0_b0_d, 4) for _, s in step_pairs)

    ax_op_pre = _max_op(arr_pre, ax_row)
    ax_ax_pre = _max_band(arr_pre, ax_row, ax_lo_d, 16)
    print(
        f"MARK_AX row {ax_row} (after L8):  "
        f"max|OP_<NAME>| = {ax_op_pre:.3f}, "
        f"max|AX_CARRY_LO| = {ax_ax_pre:.3f}"
    )
    print(
        f"MARK_SE row {se_row} (after L8 / pre-relay):  "
        f"max|OP_<NAME>| = {se_op_pre:.3f}, "
        f"max|AX_CARRY_LO| = {se_ax_pre:.3f}"
    )
    print(
        f"MARK_SE row {se_row} (after L11 / post-relay): "
        f"max|OP_<NAME>| = {se_op_post:.3f}, "
        f"max|AX_CARRY_LO| = {se_ax_post:.3f}, "
        f"max|ALU_LO| = {se_alu_post:.3f}, "
        f"max|CMP| = {se_cmp_post:.3f}, "
        f"max|STACK0_BYTE0..3| = {se_stack0_post:.3f}"
    )
    threshold = 0.9
    ok = True
    for label, val in (
        ("OP_<NAME>", se_op_post),
        ("AX_CARRY_LO", se_ax_post),
        ("ALU_LO", se_alu_post),
        ("CMP", se_cmp_post),
        ("STACK0_BYTE0..3", se_stack0_post),
    ):
        status = "OK" if val >= threshold else "MISS"
        if val < threshold:
            ok = False
        print(f"  Wave A acceptance ({label} >= {threshold}): {val:.3f} {status}")
    if ok:
        print("\nWave A acceptance MET — all relayed dims live at STEP_END.")
    else:
        print("\nWave A acceptance NOT met for at least one dim above.")

    # --- L0/L1 within-step register-presence broadcast (2026-06-10) -----
    # The L1 head 6 broadcast now covers all 5 register markers
    # (AX/PC/SP/BP/STACK0). The probe below checks the AX channel as
    # a representative sample; see
    # ``c4_release/tools/probe_step_end_registers.py`` for the
    # all-register acceptance check.
    print("\n" + "=" * 70)
    print("L0/L1 WITHIN-STEP RELAY HEAD (L1 head 6)")
    print("=" * 70)
    arr_l1 = capture.after_layer[1]
    se_reg_ax_d = capture.dim("SE_REG_AX_PRESENT")
    se_reg_ax_at_se = float(arr_l1[0, se_row, se_reg_ax_d].item())
    se_reg_ax_at_ax = float(arr_l1[0, ax_row, se_reg_ax_d].item())
    print(f"SE_REG_AX_PRESENT@MARK_SE (L1): {se_reg_ax_at_se:+.3f}")
    print(f"SE_REG_AX_PRESENT@MARK_AX (L1): {se_reg_ax_at_ax:+.3f}")
    if se_reg_ax_at_se >= 0.5:
        print("VERIFIED: L1 head 6 within-step relay broadcasts MARK_AX")
        print("presence to MARK_SE position. Downstream consumers must")
        print("gate on MARK_SE_ONLY to filter out the V_GAIN-scaled")
        print("leakage at non-SE marker rows (an inherent softmax-split")
        print("artefact of the 5-way K-bank multi-marker broadcast).")
    else:
        print("WARNING: L1 head 6 broadcast did NOT land at MARK_SE. The")
        print("relay head spec or ALiBi slope may need tuning.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
