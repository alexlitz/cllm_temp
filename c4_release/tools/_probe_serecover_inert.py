#!/usr/bin/env python3
"""Inertness proof for CmpOperandSeRecoverFFN + MulOperandSeRecoverFFN.

MISSION (docs/SERECOVER_DELETE_2026_07_13.md): with the combined
``C4_ALU_OPERAND_SURVIVE`` fix ON (block-15 L9-clear operand spare +
block-17 head-4 CMP Q-veto) operand-A survives clean through BOTH ALU-clear
crushes, so neither SeRecover wrapper ever detects a ``crushed`` row -> its
forward re-write branch is skipped -> ``wrapper.forward(x) == wrapper.inner(x)``
BYTE-FOR-BYTE on the very crush rows the recovers exist to repair.

We build the efficient-mode smoke-gate model (``build_groundtruth_probe``,
spec_k=0 -> ``AutoregressiveVMRunner(trust_neural_alu=True)`` installs the
wraps), register forward-PRE hooks on the wrapper blocks to CAPTURE the exact
tensor each wrapper receives, then on the captured input compute BOTH
``wrapper.forward(x)`` and ``wrapper.inner(x)`` and report the max abs diff over
ALL rows (and, separately, over the ALU AX crush rows). max-diff ~0 == inert ==
safe to delete.

Value-diverse crush fixtures (per docs/WRAPPER_DELETABILITY_MATRIX): CMP with
operand-A in {7, 23, 57} (both-nibble-nonzero crush values) + MUL with
operand-A in {23, 7}.

Run:
  C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_serecover_inert.py   # inert
  C4_ALU_OPERAND_SURVIVE=0 python tools/_probe_serecover_inert.py   # recover fires
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_ALU_OPERAND_SURVIVE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


# Value-diverse crush fixtures. operand-A (the first IMM) is the value the
# recover exists to restore; each is a both-nibble-nonzero crush value.
PROGS = {
    "eq_7_45":   _mk([(Opcode.IMM, 7),  Opcode.PSH, (Opcode.IMM, 45),
                      Opcode.EQ, Opcode.EXIT]),
    "lt_57_29":  _mk([(Opcode.IMM, 57), Opcode.PSH, (Opcode.IMM, 29),
                      Opcode.LT, Opcode.EXIT]),
    "gt_23_65":  _mk([(Opcode.IMM, 23), Opcode.PSH, (Opcode.IMM, 65),
                      Opcode.GT, Opcode.EXIT]),
    "mul_23_65": _mk([(Opcode.IMM, 23), Opcode.PSH, (Opcode.IMM, 65),
                      Opcode.MUL, Opcode.EXIT]),
    "mul_7_9":   _mk([(Opcode.IMM, 7),  Opcode.PSH, (Opcode.IMM, 9),
                      Opcode.MUL, Opcode.EXIT]),
}


def _find_wrap_blocks(model):
    """Return {block_idx: (module, kind)} for every SeRecover wrap."""
    out = {}
    for i, b in enumerate(getattr(model, "blocks", [])):
        ffn = getattr(b, "ffn", None)
        if ffn is None:
            continue
        if getattr(ffn, "_is_cmp_se_recover_wrap", False):
            out[i] = (ffn, "Cmp")
        elif getattr(ffn, "_is_mul_se_recover_wrap", False):
            out[i] = (ffn, "Mul")
    return out


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]

    wraps = _find_wrap_blocks(model)
    flag = os.environ.get("C4_ALU_OPERAND_SURVIVE", "1")
    print(f"=== C4_ALU_OPERAND_SURVIVE={flag} ===")
    if not wraps:
        print("NO SeRecover wraps present in built model -> nothing to prove "
              "(already deleted OR non-efficient build).")
        return
    print("SeRecover wraps found: " + ", ".join(
        f"blk{i}:{k}" for i, (m, k) in sorted(wraps.items())))

    # Capture each wrapper's input via a forward-pre hook.
    captured = {}

    def _mk_hook(idx):
        def _hook(mod, args):
            captured[idx] = args[0].detach()
        return _hook

    handles = [wraps[i][0].register_forward_pre_hook(_mk_hook(i))
               for i in wraps]

    overall_max = {"Cmp": 0.0, "Mul": 0.0}
    crush_rows_seen = {"Cmp": 0, "Mul": 0}

    for pname, bc in PROGS.items():
        captured.clear()
        ctx = probe._final_context(bc, max_steps=20)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            _ = model.forward(toks)  # populates captured[*]
        for idx, (mod, kind) in sorted(wraps.items()):
            if idx not in captured:
                continue
            x = captured[idx]
            with torch.no_grad():
                # inner-only vs full wrapper forward, on the SAME input.
                y_full = mod.forward(x)
                y_inner = mod.inner(x)
            diff = (y_full - y_inner).abs()
            mx = float(diff.max().item())
            overall_max[kind] = max(overall_max[kind], mx)
            # Which rows would the recover have rewritten? A crush row = ALU
            # band max cell < 0 at the wrapper INPUT (the recover's own gate).
            lo_band = x[0, :, alu_lo:alu_lo + 16]
            hi_band = x[0, :, alu_hi:alu_hi + 16]
            ax = x[0, :, ax_base] > 0.5
            crushed = (lo_band.max(dim=-1).values < 0.0) & \
                      (hi_band.max(dim=-1).values < 0.0)
            crush_ax = int((ax & crushed).sum().item())
            crush_rows_seen[kind] += crush_ax
            # Max diff restricted to the AX rows (the recover's write surface).
            ax_idx = torch.nonzero(ax, as_tuple=False).flatten().tolist()
            ax_mx = 0.0
            for r in ax_idx:
                ax_mx = max(ax_mx, float(diff[0, r].abs().max().item()))
            tag = "INERT" if mx < 1e-6 else "!! DIFFERS"
            print(f"  {pname:>10s} blk{idx} {kind}: crush_AX_rows={crush_ax} "
                  f"max|forward-inner|(all)={mx:.3e} (AX-rows)={ax_mx:.3e} "
                  f"[{tag}]")

    for h in handles:
        h.remove()

    print("--- SUMMARY ---")
    for kind in ("Cmp", "Mul"):
        mx = overall_max[kind]
        verdict = "INERT (forward==inner)" if mx < 1e-6 else "NOT INERT"
        print(f"  {kind}OperandSeRecover: max|forward-inner| over ALL "
              f"fixtures = {mx:.3e} | crush-AX-rows-flagged="
              f"{crush_rows_seen[kind]} -> {verdict}")


if __name__ == "__main__":
    main()
