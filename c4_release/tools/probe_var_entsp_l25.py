#!/usr/bin/env python3
"""Pin the L25 (block-35 tail) ENT-step SP byte2/byte3 re-corruption on id 262.

Two things:
 (A) FULL block-35 W_down ablation -> does id 262 ENT-step SP become
     [_, 0xff, 0x00, 0x00] (proving L25 tail is the byte2/3 carrier)?
     also re-check the a375d917 guards (or_basic, sub_borrow_cascade).
 (B) Residual read of the ENT-step SP byte1/byte2/byte3 PREDICTION rows
     (the SP byte0/byte1/byte2 token rows) after block 34 (L25 input) vs
     after block 35 (L25 output): show the OUTPUT_LO/HI lanes L25 writes.
     SP byte2 is predicted at the SP-byte1 token row; SP byte3 at the
     SP-byte2 token row.

spec_k=0, hook-free, dense weights.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from neural_vm.embedding import Opcode as Op  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}


def find_sp_marker(ctx, prompt_len, want_step):
    step = 0; i = prompt_len
    while i < len(ctx):
        nm = MARKERS.get(ctx[i])
        if nm == "STEP_END":
            step += 1; i += 1; continue
        if nm in ("PC", "AX", "SP", "BP"):
            if step == want_step and nm == "SP":
                return i
            i += 5; continue
        i += 1
    return None


def main():
    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    dev = next(m.parameters()).device
    OUT_LO = int(dp["OUTPUT_LO"]); OHTS = int(dp["OUTPUT_HI_THIS_STEP"])
    BI = {k: int(dp[f"BYTE_INDEX_{k}"]) for k in range(4)}
    H1_2 = int(dp["H1"]) + 2; OPENT = int(dp["OP_ENT"])

    idx = 262
    tests = generate_test_programs()
    src, exp, _ = tests[idx]
    bc, data = compile_c(src)
    prompt_len = len(probe._build_context(bc))

    def ent_sp_bytes():
        ctx = probe._final_context(bc, max_steps=9)
        sp = find_sp_marker(ctx, prompt_len, 1)
        return [ctx[sp + 1 + j] & 0xFF for j in range(4)], ctx, sp

    base, ctx, sp = ent_sp_bytes()
    print(f"# baseline ENT-step SP = {[hex(b) for b in base]}  (oracle wants [0xe8,0xff,0x0,0x0])")
    print(f"# step1 SP marker @ {sp}; rows: marker={sp} b0={sp+1} b1={sp+2} b2={sp+3} b3={sp+4}")

    def mk(ops):
        out = []
        for o in ops:
            if isinstance(o, tuple):
                op, imm = o; out.append(op | (imm << 8))
            else:
                out.append(o)
        return out
    GUARDS = {
        "or_basic": (mk([(Op.IMM, 0x0F), Op.PSH, (Op.IMM, 0x30), Op.OR, Op.EXIT]), 0x3F),
        "sub_borrow_cascade": (mk([(Op.IMM, 0), Op.PSH, (Op.IMM, 1), Op.SUB, Op.EXIT]), 0xFFFFFFFF),
    }
    def guards():
        out = {}
        for nm, (prog, want) in GUARDS.items():
            _, code = probe.emitted_result(prog, max_steps=20)
            out[nm] = (hex(code), code == want)
        return out
    print("# baseline guards:", guards())

    # (A) FULL block-35 ablation
    Wd = m.blocks[35].ffn.W_down.data
    saved = Wd.clone()
    Wd.zero_()
    abl_sp, _, _ = ent_sp_bytes()
    abl_guards = guards()
    Wd.copy_(saved)
    print(f"\n[A] FULL block-35 ablation: ENT-step SP = {[hex(b) for b in abl_sp]}")
    print(f"    guards under full ablation: {abl_guards}")

    # (B) residual band reads at the SP byte rows, block 34 vs 35.
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    def band(row, base, lab, lim=1.0):
        return " ".join(f"{lab}+{k}={float(row[base+k]):+.1f}"
                        for k in range(16) if abs(float(row[base+k])) > lim)

    print("\n[B] SP byte-row OUTPUT bands (row = token that PREDICTS the next byte):")
    for blk in (34, 35):
        with torch.no_grad():
            r = m.forward(padded, stop_after_block=blk)[0].float()
        print(f"  ---- after block {blk} ----")
        for off, label in ((1, "SP-b0-row->pred b1"), (2, "SP-b1-row->pred b2"),
                            (3, "SP-b2-row->pred b3")):
            row = r[sp + off]
            gates = f"H1+2={float(row[H1_2]):+.2f} OP_ENT={float(row[OPENT]):+.2f} " + \
                    " ".join(f"BI{k}={float(row[BI[k]]):+.2f}" for k in range(4))
            print(f"   {label:20s} {gates}")
            print(f"      LO: {band(row, OUT_LO, 'LO')}")
            print(f"      HI: {band(row, OHTS, 'HI')}")


if __name__ == "__main__":
    main()
