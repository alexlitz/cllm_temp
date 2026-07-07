#!/usr/bin/env python3
"""Probe: dump the operand bands the func-return ADD reads (id578 step-13).

Reads the residual at the INPUT to each composite ALU block on the step-13
AX-marker row and decodes the one-hot operand nibbles (ALU_LO/HI = operand A,
AX_CARRY_LO/HI = operand B) plus the OUTPUT nibbles after the block. Confirms
whether an operand's HIGH nibble is corrupted upstream of the ADD (the RANK-2
func-return ADD byte-0 high-nibble over-count).

Tooling only. CPU. Not on any build path.
"""
import os
import sys

os.environ.setdefault("C4_VM_CACHE_DIR", f"/tmp/funcadd_probe_{os.getpid()}")
os.environ.setdefault("C4_CAMPAIGN", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
    FaithfulForwardCache, _faithful_attn_forward, _COMPOSITE_FFN, STEP_TOKENS,
)
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402


def onehot_nibble(vec, base):
    """Return (argmax_nibble, value_at_argmax) for a 16-wide one-hot."""
    seg = vec[base:base + 16]
    idx = int(torch.argmax(seg).item())
    return idx, float(seg[idx].item())


def main():
    ID = int(sys.argv[1]) if len(sys.argv) > 1 else 578
    STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 13
    progs = generate_test_programs()
    src, exp, name = progs[ID]
    code, data = compile_c(src)
    print(f"=== {name}  exp={exp}=0x{exp & 0xff:02x} ===")

    model, layout = build_production_model("cpu")
    dp = dict(layout.dim_positions)
    fwd = FaithfulForwardCache(model)

    ot = oracle_tape_and_steps(code, bytes(data) if data else b"", max_steps=48)
    prompt = build_code_prompt(code, bytes(data) if data else b"")
    prefix = len(prompt)
    full_ctx = prompt + ot.draft_tokens

    ax_off = 5
    row = prefix + STEP * STEP_TOKENS + ax_off

    print(f"prefix={prefix} STEP={STEP} STEP_TOKENS={STEP_TOKENS} AX_row={row}")
    print(f"dims: ALU_LO={dp['ALU_LO']} ALU_HI={dp['ALU_HI']} "
          f"AX_CARRY_LO={dp['AX_CARRY_LO']} AX_CARRY_HI={dp['AX_CARRY_HI']} "
          f"OUTPUT_LO={dp['OUTPUT_LO']} OUTPUT_HI={dp['OUTPUT_HI']} "
          f"MARK_AX={dp['MARK_AX']}")

    tok = torch.tensor([list(full_ctx)], dtype=torch.long)
    x = model.embed(tok)[0]

    def dump(tag, xx):
        v = xx[row]
        aL = onehot_nibble(v, dp["ALU_LO"])
        aH = onehot_nibble(v, dp["ALU_HI"])
        bL = onehot_nibble(v, dp["AX_CARRY_LO"])
        bH = onehot_nibble(v, dp["AX_CARRY_HI"])
        oL = onehot_nibble(v, dp["OUTPUT_LO"])
        oH = onehot_nibble(v, dp["OUTPUT_HI"])
        mark = float(v[dp["MARK_AX"]].item())
        opA = aH[0] * 16 + aL[0]
        opB = bH[0] * 16 + bL[0]
        out = oH[0] * 16 + oL[0]
        print(f"  [{tag:24s}] MARK_AX={mark:+.2f}  "
              f"opA(ALU)=0x{opA:02x}({aH[0]}|{aL[0]} @{aH[1]:.1f}/{aL[1]:.1f})  "
              f"opB(CARRY)=0x{opB:02x}({bH[0]}|{bL[0]} @{bH[1]:.1f}/{bL[1]:.1f})  "
              f"OUT=0x{out:02x}({oH[0]}|{oL[0]} @{oH[1]:.1f}/{oL[1]:.1f})")

    def dump_full(tag, xx):
        v = xx[row]
        for nm, base in (("ALU_LO", dp["ALU_LO"]), ("ALU_HI", dp["ALU_HI"]),
                         ("AX_CARRY_LO", dp["AX_CARRY_LO"]),
                         ("AX_CARRY_HI", dp["AX_CARRY_HI"])):
            seg = v[base:base + 16]
            hot = [(k, round(float(seg[k]), 2)) for k in range(16)
                   if float(seg[k]) > 0.5]
            clean_sum = sum(k for k in range(16) if float(seg[k]) > 0.5)
            print(f"    {tag} {nm}: hot>0.5={hot} clean_onehot_sum={clean_sum}"
                  f" (=0x{clean_sum & 0xff:x})")

    TRACK_ALU_HI = os.environ.get("TRACK_ALU_HI", "0") == "1"
    def track(tag, xx):
        seg = xx[row][dp["ALU_HI"]:dp["ALU_HI"] + 16]
        hot = [(k, round(float(seg[k]), 2)) for k in range(16)
               if float(seg[k]) > 0.5]
        c4 = float(seg[4])
        if c4 > 0.5 or hot != [(2, 6.0)]:
            print(f"    ALU_HI {tag}: hot={hot} cell4={c4:.2f}")

    prev_c4 = None
    for bi, (heads, nh, hd, sm1, block, is_comp, ff) in enumerate(fwd.blocks):
        x = _faithful_attn_forward(heads, x, nh, hd, sm1)
        if TRACK_ALU_HI:
            c4 = float(x[row][dp["ALU_HI"] + 4])
            if (c4 > 0.5) != (prev_c4 is not None and prev_c4 > 0.5):
                print(f"  ATTN blk{bi} ({type(block.ffn).__name__}): "
                      f"ALU_HI[4] {prev_c4} -> {c4:.2f}")
            prev_c4 = c4
        bname = type(block.ffn).__name__
        if is_comp:
            dump(f"blk{bi} PRE {bname}", x)
            if bname == "AddSub5StageBlock":
                dump_full(f"blk{bi} PRE", x)
            x = block.ffn(x.unsqueeze(0))[0]
            dump(f"blk{bi} POST {bname}", x)
        else:
            W_up, b_up, W_gate, b_gate, W_down = ff
            up = x @ W_up.t() + b_up
            gate = x @ W_gate.t() + b_gate
            hidden = torch.nn.functional.silu(up) * gate
            x = x + hidden @ W_down.t()
        if TRACK_ALU_HI:
            c4 = float(x[row][dp["ALU_HI"] + 4])
            if (c4 > 0.5) != (prev_c4 > 0.5):
                print(f"  FFN  blk{bi} ({bname}): "
                      f"ALU_HI[4] {prev_c4:.2f} -> {c4:.2f}")
            prev_c4 = c4

    logits = x @ fwd.head_w.t() + fwd.head_b
    print("final AX-row argmax token:", int(logits[row].argmax().item()))


if __name__ == "__main__":
    main()
