#!/usr/bin/env python3
"""FAST SP-decrement attribution: ONE full forward, read SP-marker row dims.

Teacher-forced context (oracle tape). For each requested step, reads the
SP-marker row at the FINAL block and dumps OUTPUT byte0 + the gate dims that
decide the L25 ``tail_sp_marker_byte0_f8`` rule + the suppress band + the L6
decrement result high-nibble lanes. Goal: see whether the 0xF8 reset is the
f8 tail rule firing (suppress band 0) or an upstream L6/L20 writer.

Run:
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_spdec python tools/_probe_spdec_fast.py 950 3 4 5
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"; os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "STEP_END"}


def find_sp_marker(ctx, pl, step, STEP):
    base = pl + step * STEP
    for i in range(base, min(base + STEP, len(ctx))):
        if MARK.get(ctx[i]) == "SP":
            return i
    return None


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 950
    steps = [int(x) for x in sys.argv[2:]] or [3, 4, 5]
    p = build_groundtruth_probe()
    m = p.model
    dp = m.dim_positions
    STEP = int(Token.STEP_TOKENS)
    tests = generate_test_programs()
    src, exp, name = tests[idx]
    bc, _ = compile_c(src)
    ctx = p._final_context(bc, max_steps=max(steps) + 3)
    pl = len(p._build_context(bc))
    nblocks = len(m.blocks)
    print(f"=== id{idx} {name} STEP={STEP} nblocks={nblocks} flag_deep={os.environ.get('C4_SP_DEEP_FRAME_DEPTH_TRACK')} ===")

    def b0(row):
        lo = int(torch.argmax(row[dp['OUTPUT_LO']:dp['OUTPUT_LO'] + 16]))
        hi = int(torch.argmax(row[dp['OUTPUT_HI_THIS_STEP']:dp['OUTPUT_HI_THIS_STEP'] + 16]))
        return hi * 16 + lo, hi, lo

    flag_names = ("OP_PSH", "OP_ADJ", "OP_ENT", "OP_JSR", "OP_IMM", "OP_LEV",
                  "PSH_AT_SP", "CMP+3", "CMP+4", "MARK_SP", "HAS_SE",
                  "NONFIRST_PSH_SP_SUPPRESS")

    def fd(nm):
        base = nm.split("+")[0]
        off = int(nm.split("+")[1]) if "+" in nm else 0
        return (dp[base] + off) if base in dp else None

    def emb0(row):
        if 'EMBED_LO' not in dp or 'EMBED_HI' not in dp:
            return None
        lo = int(torch.argmax(row[dp['EMBED_LO']:dp['EMBED_LO'] + 16]))
        hi = int(torch.argmax(row[dp['EMBED_HI']:dp['EMBED_HI'] + 16]))
        return hi * 16 + lo

    padded = torch.tensor([ctx], device=p._device)
    last = nblocks - 1
    # Trace OUTPUT byte0 across a few blocks to localize the writer.
    import os as _os
    _tb = _os.environ.get("SPDEC_BLOCKS")
    if _tb:
        trace_blocks = [int(x) for x in _tb.split(",") if 0 <= int(x) < nblocks]
    else:
        trace_blocks = [b for b in (6, 7, 8, 12, 20, 25, last) if 0 <= b < nblocks]
    if last not in trace_blocks:
        trace_blocks.append(last)
    block_outs = {}
    with torch.no_grad():
        for b in trace_blocks:
            r = m.forward(padded, stop_after_block=b)
            if r.is_sparse:
                r = r.to_dense()
            block_outs[b] = r
    resid_full = block_outs[last]
    for st in steps:
        sp_pos = find_sp_marker(ctx, pl, st, STEP)
        if sp_pos is None:
            print(f"  step{st}: no SP marker"); continue
        row = resid_full[0, sp_pos]
        byte0, hi, lo = b0(row)
        # per-block OUTPUT byte0 + EMBED byte0 at each traced block
        bt = []
        for b in trace_blocks:
            rb = block_outs[b][0, sp_pos]
            bb, _, _ = b0(rb)
            eb = emb0(rb)
            bt.append(f"b{b}:OUT={bb:#04x}/EMB={eb:#04x}" if eb is not None else f"b{b}:OUT={bb:#04x}")
        print(f"  step{st} SP@{sp_pos} trace: " + " ".join(bt))
        flags = {}
        for nm in flag_names:
            d = fd(nm)
            if d is not None and abs(float(row[d])) > 0.2:
                flags[nm] = round(float(row[d]), 2)
        # OUTPUT_HI nibble lanes (the deep-frame discriminator)
        hilanes = {f"HI{k:x}": round(float(row[dp['OUTPUT_HI_THIS_STEP'] + k]), 2)
                   for k in range(16) if abs(float(row[dp['OUTPUT_HI_THIS_STEP'] + k])) > 0.3}
        print(f"  step{st} SP@{sp_pos}: FINAL OUT byte0=0x{byte0:02x} (hi={hi:x} lo={lo:x})")
        print(f"     flags={flags}")
        print(f"     OUT_HI lanes={hilanes}")


if __name__ == "__main__":
    main()
