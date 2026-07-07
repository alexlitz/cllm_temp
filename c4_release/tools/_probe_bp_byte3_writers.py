#!/usr/bin/env python3
"""Trace the func step-0 (JSR) BP byte-3 OUTPUT nibble across all block INPUTs
and name the winning writer op via build_writer_index. The BP byte-3 value
token sits at slice offset 19 (BP marker 15 + byte 3 + 1). Oracle wants 0x00;
model leaks 0x01 (byte3 lo-nibble = 1).

  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/bpb3_$$ \
    python tools/_probe_bp_byte3_writers.py --id 675 --step 0
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
BP_MARK_OFF = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=675)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--byte", type=int, default=3)
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    nblocks = len(model.blocks)
    out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI_THIS_STEP"]
    print(f"nblocks={nblocks} STEP={STEP} OUTPUT_LO@{out_lo} OUTPUT_HI@{out_hi}")

    caps = {}

    def mk(bi):
        def _h(m, inp):
            caps[bi] = inp[0].detach().clone()
        return _h
    hooks = [model.blocks[bi].register_forward_pre_hook(mk(bi))
             for bi in range(nblocks)]
    fin = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: fin.__setitem__("o", o.detach().clone()))

    pid = args.id
    src, exp, desc = PROGS[pid]
    bc = compile_c(src)[0]
    prompt = build_code_prompt(bc, b"")
    ot = oracle_tape_and_steps(bc, b"", max_steps=40)
    tape = list(prompt) + list(ot.draft_tokens)
    tok = torch.tensor([tape], dtype=torch.long)
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            model.forward(tok)
    plen = len(prompt)
    pos = plen + args.step * STEP + BP_MARK_OFF + 1 + args.byte
    print(f"id{pid} {desc!r}  BP byte{args.byte} at abs pos {pos} "
          f"(step {args.step})")

    for band, base in (("LO", out_lo), ("HI", out_hi)):
        print(f"  -- OUTPUT_{band} 16-cell argmax across block INPUTs --")
        prev = None
        for bi in range(nblocks):
            r = caps[bi]
            cells = r[0, pos, base:base + 16]
            win = int(torch.argmax(cells).item())
            wv = float(cells[win].item())
            # value at nibble 0 and 1 for context
            v0 = float(cells[0].item()); v1 = float(cells[1].item())
            if win != prev:
                print(f"     block{bi:2d} IN win={win} val={wv:.4g} "
                      f"(cell0={v0:.4g} cell1={v1:.4g})")
                prev = win
        fo = fin["o"]
        fwin = int(torch.argmax(fo[0, pos, base:base + 16]).item())
        print(f"     FINAL win={fwin} "
              f"(cell0={float(fo[0,pos,base].item()):.4g} "
              f"cell1={float(fo[0,pos,base+1].item()):.4g})")

    for h in hooks:
        h.remove()

    # ---- name the writers at the block where nibble flips to 1 (HI band) ----
    print("\n  Building writer index for OUTPUT_HI_THIS_STEP+1 and +0 ...")
    from neural_vm.verification.writer_index import build_writer_index
    from neural_vm.verification.decl_verifier import _build_layout_only  # noqa
    try:
        reg = layout.dim_registry
    except Exception:
        reg = None
    ops = []
    for blk in layout.ops_per_layer:
        ops.extend(blk)
    ops.extend(list(getattr(layout, "block_ops", []) or []))
    ops.extend(list(getattr(layout, "model_ops", []) or []))
    if reg is None:
        print("   (no dim_registry on layout; skip writer_index)")
        return
    widx = build_writer_index(ops, reg)
    for cell in [("OUTPUT_HI_THIS_STEP", 1), ("OUTPUT_HI_THIS_STEP", 0),
                 ("OUTPUT_LO", 1)]:
        ws = widx.get(cell, [])
        pos_ws = [w for w in ws if w.max_contribution > 0]
        print(f"   {cell}: {len(ws)} writers, showing pos-contrib "
              f"with 'bp'/'byte3'/'jsr'/'ent'/'clean'/'dump' in name:")
        for w in sorted(ws, key=lambda x: -abs(x.max_contribution))[:40]:
            nm = w.op_name.lower()
            if any(k in nm for k in ("bp", "byte3", "byte_3", "jsr", "ent",
                                     "clean", "dump", "hibyte", "carry",
                                     "passthrough")):
                print(f"      {w.op_name:52s} contrib={w.max_contribution:.4g} "
                      f"rule={getattr(w.rule,'name','?')}")


if __name__ == "__main__":
    main()
