#!/usr/bin/env python3
"""Attribute the block-50 BP-byte1 OUTPUT crush to its owning declarative rule.

Runs the faithful forward up to (but not including) the crushing block, then
ranks the FFN rules of that block by their runtime SwiGLU contribution to the
OUTPUT_LO+0 / OUTPUT_HI+0 cells at the BP-byte1 row.

CPU ONLY, single forward.
Usage: CUDA_VISIBLE_DEVICES="" python tools/probe_root_a_attrib.py 275 1 50
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
from neural_vm.verification.faithful_interpreter import FaithfulInterpreter


def main(idx, ent_step, crush_block, marker_off=15, byte_k=1):
    src, exp, desc = generate_test_programs()[idx]
    bc, data = compile_c(src)
    prompt = build_code_prompt(bc, data)
    tape = oracle_tape_and_steps(bc, data, max_steps=40)
    full_ctx = list(prompt) + list(tape.draft_tokens)
    prefix = len(prompt)

    model, layout = build_production_model("cpu")
    dp = dict(layout.dim_positions)
    out_lo0 = dp["OUTPUT_LO"] + 0
    out_hi0 = dp["OUTPUT_HI"] + 0

    pred_pos = prefix + ent_step * STEP_TOKENS + marker_off + byte_k
    print(f"=== id{idx} {desc} step={ent_step} byte{byte_k} pred_pos={pred_pos} "
          f"crush_block={crush_block} ===")

    # Faithful forward up to (not including) crush_block.
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

    resid_in = x[pred_pos]
    print(f"  resid entering block{crush_block}: "
          f"OUTPUT_LO+0={float(resid_in[out_lo0]):.3g}  "
          f"OUTPUT_HI+0={float(resid_in[out_hi0]):.3g}  "
          f"OUTPUT_LO+15={float(resid_in[dp['OUTPUT_LO']+15]):.3g}  "
          f"OUTPUT_HI+15={float(resid_in[dp['OUTPUT_HI']+15]):.3g}")

    # Build interpreter to attribute the block's FFN ops.
    interp = FaithfulInterpreter(
        dim_positions=dp, ops_per_block=[],
        d_model=model.d_model, num_heads=model.blocks[0].attn.num_heads,
        head_dim=model.blocks[0].attn.head_dim,
    )
    # Collect ALL declarative ops (the rule name tells us the owning family).
    flat = []
    for blk in layout.ops_per_layer:
        flat.extend(blk)
    flat.extend(list(getattr(layout, "block_ops", []) or []))
    flat.extend(list(getattr(layout, "model_ops", []) or []))
    for col_name, col in (("OUTPUT_LO+0", out_lo0), ("OUTPUT_HI+0", out_hi0)):
        ranked = interp.attribute_runtime_contribution(flat, col, resid_in)
        print(f"\n  --- top contributors to {col_name} (col {col}) ---")
        for op_name, rule_name, contrib in ranked[:12]:
            print(f"    {contrib:+14.3g}  {op_name}  ::  {rule_name}")
        if not ranked:
            print("    (no declarative FFN rule writes this col at this row)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="?", default=275)
    ap.add_argument("step", type=int, nargs="?", default=1)
    ap.add_argument("crush_block", type=int, nargs="?", default=50)
    ap.add_argument("--marker", type=int, default=15)
    ap.add_argument("--byte", type=int, default=1)
    a = ap.parse_args()
    main(a.idx, a.step, a.crush_block, a.marker, a.byte)
