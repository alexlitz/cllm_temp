#!/usr/bin/env python3
"""func_min / func_max block-INPUT attribution probe (task #428).

TEACHER-FORCED forward over the DraftVM byte-exact CORRECT tape. For each step
it:
  * finds the AX marker offset in that step's 30/35-token slice,
  * decodes the model's argmax PC/AX bytes at the FIXED value offsets and
    compares them to the DraftVM oracle -> reports the FIRST step whose decoded
    (pc, ax) diverges (the faithful-to-production divergence, per the
    interp_oracle_gate soundness contract: teacher-forced == production up to
    the first register VALUE-byte correction),
  * for that failing step + wrong byte, captures the residual at the wrong
    OUTPUT_LO/HI cell at EVERY block INPUT and reports which block flips the
    winning nibble (the block-input attribution), plus a menu of candidate
    discriminators on the AX row (OP_LEA, OP_LT/OP_GT, CMP, MARK_*, FETCH...).

Run:
  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/funcmin_probe_$$ \
    python tools/_probe_funcmin_blockinput.py --ids 675,650
"""
from __future__ import annotations
import os, sys, contextlib, io, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()

# AX marker is at fixed offset 5 in a step slice; value bytes at 6..9.
AX_MARK_OFF = 5
PC_MARK_OFF = 0


def _argmax_nib(arr, pos, base):
    return int(torch.argmax(arr[0, pos, base:base + 16]).item())


def decode_reg_at(arr, dimpos, pos, out_lo_d, hi_d):
    """Decode (lo_nib, hi_nib) -> byte at a residual position (final resid)."""
    lo = _argmax_nib(arr, pos, out_lo_d)
    hi = _argmax_nib(arr, pos, hi_d)
    return (hi << 4) | lo, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675,650")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    nblocks = len(model.blocks)
    out_lo_d = dimpos["OUTPUT_LO"]
    hi_d = dimpos["OUTPUT_HI_THIS_STEP"]
    mark_ax_d = dimpos["MARK_AX"]
    print(f"nblocks={nblocks} STEP={STEP} d_model={model.d_model} "
          f"OUTPUT_LO@{out_lo_d} OUTPUT_HI@{hi_d}")

    # Candidate discriminator dims to dump on the failing AX row.
    disc_dims = {}
    for nm in ["OP_LEA", "OP_LT", "OP_GT", "OP_LI", "OP_SI", "OP_IMM",
               "OP_LEV", "OP_ENT", "OP_PSH", "MARK_AX", "MARK_SE_ONLY",
               "HAS_SE", "IS_BYTE", "CMP_FLAG"]:
        if nm in dimpos:
            disc_dims[nm] = dimpos[nm]
    for nm, off in [("CMP", 7), ("ALU_HI", 15), ("ALU_LO", 0),
                    ("FETCH_LO", 8), ("FETCH_LO", 0)]:
        if nm in dimpos:
            disc_dims[f"{nm}+{off}"] = dimpos[nm] + off

    # Capture every block INPUT residual for the OUTPUT cells (all offsets).
    caps = {}

    def mk_pre(bi):
        def _hook(m, inp):
            caps[bi] = inp[0].detach().clone()
        return _hook

    hooks = [model.blocks[bi].register_forward_pre_hook(mk_pre(bi))
             for bi in range(nblocks)]
    # final output of last block
    final_cap = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: final_cap.__setitem__("out", o.detach().clone()))

    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        ot = oracle_tape_and_steps(bc, b"", max_steps=40)
        tape = list(prompt) + list(ot.draft_tokens)
        tok = torch.tensor([tape], dtype=torch.long)
        caps.clear(); final_cap.clear()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                model.forward(tok)
        fin = final_cap["out"]
        plen = len(prompt)
        nsteps = len(ot.steps)
        print(f"\n===== id{pid} {desc!r} exp={exp} nsteps={nsteps} =====")
        print(f"  opcodes: {[hex(o) for o in ot.opcodes]}")

        fail_step = None
        for step in range(nsteps):
            lo = plen + step * STEP
            hi = lo + STEP
            if hi > fin.shape[1]:
                break
            # AX marker: locate within slice
            seg = fin[0, lo:hi, mark_ax_d]
            ax_off = int(torch.argmax(seg).item())
            ax_pos = lo + ax_off
            # Decode PC/AX at fixed value offsets (bytes 0..3 little-endian)
            pc_bytes = []
            ax_bytes = []
            for j in range(4):
                pc_bytes.append(int(torch.argmax(
                    fin[0, lo + PC_MARK_OFF + 1 + j, out_lo_d:out_lo_d + 16]
                ).item()) | (int(torch.argmax(
                    fin[0, lo + PC_MARK_OFF + 1 + j, hi_d:hi_d + 16]
                ).item()) << 4))
                ax_bytes.append(int(torch.argmax(
                    fin[0, lo + AX_MARK_OFF + 1 + j, out_lo_d:out_lo_d + 16]
                ).item()) | (int(torch.argmax(
                    fin[0, lo + AX_MARK_OFF + 1 + j, hi_d:hi_d + 16]
                ).item()) << 4))
            got_pc = sum(b << (8 * j) for j, b in enumerate(pc_bytes)) & 0xFFFFFFFF
            got_ax = sum(b << (8 * j) for j, b in enumerate(ax_bytes)) & 0xFFFFFFFF
            exp_pc, exp_ax = ot.steps[step]
            ok = (got_pc == exp_pc) and (got_ax == exp_ax)
            flag = "" if ok else "  <<< DIVERGE"
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            print(f"  step{step:2d} op=0x{op:02x} axoff={ax_off:2d} "
                  f"got(pc={got_pc},ax={got_ax}) "
                  f"exp(pc={exp_pc},ax={exp_ax}){flag}")
            if not ok and fail_step is None:
                fail_step = step

        if fail_step is None:
            print("  ** no teacher-forced divergence (PASS under TF) **")
            continue

        # ---- Attribution at the failing step ----
        step = fail_step
        lo = plen + step * STEP
        hi = lo + STEP
        exp_pc, exp_ax = ot.steps[step]
        op = ot.opcodes[step] if step < len(ot.opcodes) else -1
        print(f"\n  --- ATTRIBUTION id{pid} step{step} op=0x{op:02x} "
              f"exp(pc={exp_pc},ax={exp_ax}) ---")
        # For each of PC/AX byte 0..3, find the wrong value-byte positions.
        for reg, mark_off, exp_val in (("PC", PC_MARK_OFF, exp_pc),
                                       ("AX", AX_MARK_OFF, exp_ax)):
            for j in range(4):
                pos = lo + mark_off + 1 + j
                got_byte, glo, ghi = decode_reg_at(fin, dimpos, pos, out_lo_d, hi_d)
                exp_byte = (exp_val >> (8 * j)) & 0xFF
                if got_byte == exp_byte:
                    continue
                print(f"    {reg} byte{j} pos={pos-lo}(abs {pos}): "
                      f"got=0x{got_byte:02x}(lo{glo}/hi{ghi}) want=0x{exp_byte:02x}")
                exp_lo = exp_byte & 0xF
                exp_hi = (exp_byte >> 4) & 0xF
                # Trace the WINNING lo/hi nibble across blocks (block-input).
                for band, cell_base, exp_nib, got_nib in (
                        ("LO", out_lo_d, exp_lo, glo),
                        ("HI", hi_d, exp_hi, ghi)):
                    if exp_nib == got_nib:
                        continue
                    print(f"      band OUTPUT_{band} exp_nib={exp_nib} got_nib={got_nib}"
                          f" -> per-block winning nibble (argmax over 16 cells):")
                    prev_win = None
                    for bi in range(nblocks):
                        r = caps[bi]
                        win = int(torch.argmax(
                            r[0, pos, cell_base:cell_base + 16]).item())
                        wval = float(r[0, pos, cell_base + win].item())
                        if win != prev_win:
                            print(f"        block{bi:2d} IN win_nib={win} "
                                  f"val={wval:.3g}")
                            prev_win = win
                    # final
                    fwin = int(torch.argmax(
                        fin[0, pos, cell_base:cell_base + 16]).item())
                    print(f"        FINAL win_nib={fwin}")
        # Dump discriminators on the AX row.
        seg = fin[0, lo:hi, mark_ax_d]
        ax_pos = lo + int(torch.argmax(seg).item())
        print(f"    AX-row discriminators (final resid @pos {ax_pos-lo}):")
        s = []
        for nm, d in sorted(disc_dims.items()):
            s.append(f"{nm}={float(fin[0, ax_pos, d].item()):.2f}")
        print("      " + "  ".join(s))

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
