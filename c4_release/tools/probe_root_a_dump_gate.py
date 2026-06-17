#!/usr/bin/env python3
"""Print the bp_save_dump gate-condition dim values at the BP-byte1 row.

Confirms WHY ``bp_save_dump_repopulate`` fires on the BP-register VALUE byte1
row (where BP=0xfff0 byte1=0xFF lives) and crushes it to 0x00 from a stale
BP_SAVE_PREV band carrying the previous step's old_BP nibbles.

CPU ONLY, single forward up to (not including) the dump block (50).
Usage: CUDA_VISIBLE_DEVICES="" python tools/probe_root_a_dump_gate.py 275 1
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

import sys
import argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from tools.interp_oracle_gate import (
    build_production_model, build_code_prompt, oracle_tape_and_steps,
    STEP_TOKENS,
)
from tools.faithful_interpreter_validate import (
    _attn_block_to_specs, _faithful_attn_forward, _faithful_ffn_forward,
    _COMPOSITE_FFN,
)

GATE_DIMS = [
    "OP_ENT", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
    "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
    "MARK_SE", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1",
]


def main(idx, ent_step, crush_block=50, marker_off=15, byte_k=1):
    src, exp, desc = generate_test_programs()[idx]
    bc, data = compile_c(src)
    prompt = build_code_prompt(bc, data)
    tape = oracle_tape_and_steps(bc, data, max_steps=40)
    full_ctx = list(prompt) + list(tape.draft_tokens)
    prefix = len(prompt)

    model, layout = build_production_model("cpu")
    dp = dict(layout.dim_positions)
    bp_prev = dp.get("BP_SAVE_PREV", None)

    pred_pos = prefix + ent_step * STEP_TOKENS + marker_off + byte_k
    print(f"=== id{idx} {desc} step={ent_step} byte{byte_k} pred_pos={pred_pos} ===")

    token_ids = torch.tensor([full_ctx], dtype=torch.long)
    x = model.embed(token_ids)[0]
    d_model = model.d_model
    for bi, block in enumerate(model.blocks):
        if bi == crush_block:
            break
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        x = _faithful_attn_forward(heads, x, attn.num_heads, attn.head_dim,
                                   getattr(attn, "use_softmax1", True))
        if type(block.ffn).__name__ in _COMPOSITE_FFN:
            x = block.ffn(x.unsqueeze(0))[0]
        else:
            x = _faithful_ffn_forward(block.ffn, x)

    r = x[pred_pos]
    print(f"  --- gate-condition dim values entering block{crush_block} ---")
    for name in GATE_DIMS:
        c = dp.get(name)
        if c is None:
            print(f"    {name:14s} <not in layout>")
            continue
        print(f"    {name:14s} col={c:4d}  val={float(r[c]):+.4g}")

    if bp_prev is not None:
        band = r[bp_prev:bp_prev + 32]
        lo = band[:16]
        hi = band[16:]
        lo_arg = int(torch.argmax(lo).item())
        hi_arg = int(torch.argmax(hi).item())
        print(f"  --- BP_SAVE_PREV band (col {bp_prev}) ---")
        print(f"    LO argmax nib={lo_arg:x} (val {float(lo[lo_arg]):+.3g})  "
              f"HI argmax nib={hi_arg:x} (val {float(hi[hi_arg]):+.3g})  "
              f"=> carried byte={(hi_arg<<4)|lo_arg:#04x}")
        print(f"    LO cells: {[round(float(v),2) for v in lo]}")
        print(f"    HI cells: {[round(float(v),2) for v in hi]}")

    # Reconstruct the dump gate score for byte_k.
    S = 100.0
    ent = float(r[dp["OP_ENT"]])
    memvk = float(r[dp[f"MEM_VAL_B{byte_k}"]])
    blockers = 0.0
    for nm, w in (("MARK_PC", -1000.0), ("MARK_AX", -1000.0), ("MARK_SP", -1000.0),
                  ("MARK_BP", -1000.0), ("MARK_STACK0", -1000.0),
                  ("MARK_MEM", -1000.0), ("MARK_SE", -1000.0)):
        if nm in dp:
            blockers += w * float(r[dp[nm]])
    thr = 14.0
    up = S * (ent * 1.0 + memvk * 8.0 + blockers - thr)
    sig = float(torch.nn.functional.silu(torch.tensor(up)).item())
    print(f"  --- dump gate for val{byte_k} ---")
    print(f"    OP_ENT={ent:+.3g}  MEM_VAL_B{byte_k}={memvk:+.3g}  "
          f"blockers_sum={blockers:+.3g}")
    print(f"    pre-silu up = S*({ent:.2f} + 8*{memvk:.2f} + {blockers:.2f} - {thr}) = {up:.3g}")
    print(f"    silu(up) = {sig:.4g}  (>0 => rule FIRES)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="?", default=275)
    ap.add_argument("step", type=int, nargs="?", default=1)
    ap.add_argument("--crush", type=int, default=50)
    ap.add_argument("--marker", type=int, default=15)
    ap.add_argument("--byte", type=int, default=1)
    a = ap.parse_args()
    main(a.idx, a.step, a.crush, a.marker, a.byte)
