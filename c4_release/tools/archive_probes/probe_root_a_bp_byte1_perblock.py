#!/usr/bin/env python3
"""Root A per-block residual probe: BP-byte1 OUTPUT crush (CPU spec_k=0).

At the ENT frame-establishing step BP=0x0000fff0, so BP byte1 = 0xFF. The
register VALUE byte 1 of BP is predicted at draft position
``prefix + step*35 + 15 + 1`` (BP marker offset 15, byte index 1). Its decode
reads the OUTPUT_LO / OUTPUT_HI nibble cells (the same OUTPUT family AX byte0
decodes from). Root A: a LATE block (L25 tail) crushes those OUTPUT cells back
to nibble 0 (0xFF -> 0x00), desyncing the frame.

This probe runs the faithful forward BLOCK-BY-BLOCK and prints, after each
block, the argmax OUTPUT_LO / OUTPUT_HI nibble at the BP-byte1 row — so we SEE
which block flips 0xf -> 0x0 and confirm the fix lets 0xff survive.

CPU ONLY. NO autoregressive decode (which times out). Single teacher-forced
forward.

Usage:
  CUDA_VISIBLE_DEVICES="" python tools/probe_root_a_bp_byte1_perblock.py 275 1
  (args: <corpus_id> <ent_step>; default 275 1)
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

# BP marker offset inside a 35-token step; byte 1 is at marker_off + 1 + 1.
BP_MARKER_OFF = 15
# STACK0 marker offset (if/bool surface).
STACK0_MARKER_OFF = 20


def build_full_ctx(bc, data, max_steps=40):
    prompt = build_code_prompt(bc, data)
    tape = oracle_tape_and_steps(bc, data, max_steps=max_steps)
    full = list(prompt) + list(tape.draft_tokens)
    return full, len(prompt), tape


def nibble_argmax(resid_pos, base):
    """Argmax nibble (0..15) over the 16 cells starting at column `base`."""
    cells = resid_pos[base:base + 16]
    return int(torch.argmax(cells).item()), [round(float(v), 1) for v in cells]


def probe(idx, ent_step, marker_off=BP_MARKER_OFF, byte_k=1, label="BP"):
    src, exp, desc = generate_test_programs()[idx]
    bc, data = compile_c(src)
    full_ctx, prefix, tape = build_full_ctx(bc, data)

    model, layout = build_production_model("cpu")
    dp = dict(layout.dim_positions)
    out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI"]

    # Predicting position for register VALUE byte `byte_k`.
    pred_pos = prefix + ent_step * STEP_TOKENS + marker_off + byte_k
    print(f"=== id{idx} {desc} ===")
    print(f"  step={ent_step} {label}-byte{byte_k} pred_pos={pred_pos} "
          f"(prefix={prefix})  expected nibble for 0xff = lo:0xf hi:0xf")
    print(f"  OUTPUT_LO base={out_lo} OUTPUT_HI base={out_hi}")

    token_ids = torch.tensor([full_ctx], dtype=torch.long)
    x = model.embed(token_ids)[0]
    d_model = model.d_model

    prev = None
    for bi, block in enumerate(model.blocks):
        attn = block.attn
        heads = _attn_block_to_specs(attn, d_model)
        x = _faithful_attn_forward(
            heads, x, attn.num_heads, attn.head_dim,
            getattr(attn, "use_softmax1", True),
        )
        if type(block.ffn).__name__ in _COMPOSITE_FFN:
            x = block.ffn(x.unsqueeze(0))[0]
        else:
            x = _faithful_ffn_forward(block.ffn, x)
        rp = x[pred_pos]
        lo_n, _ = nibble_argmax(rp, out_lo)
        hi_n, _ = nibble_argmax(rp, out_hi)
        byte_val = (hi_n << 4) | lo_n
        flip = ""
        if prev is not None and prev != byte_val:
            flip = f"   <<< CHANGED {prev:#04x} -> {byte_val:#04x}"
        prev = byte_val
        # only print blocks of interest (>= 22) plus any change
        if bi >= 22 or flip:
            print(f"  block{bi:2d} {type(block.ffn).__name__:24s} "
                  f"OUTPUT lo={lo_n:x} hi={hi_n:x} => byte={byte_val:#04x}{flip}")
    print(f"  FINAL BP-byte{byte_k} OUTPUT-decoded = {prev:#04x} "
          f"(want 0xff){'  OK' if prev == 0xff else '  CRUSHED' if prev == 0x00 else ''}")
    return prev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="?", default=275)
    ap.add_argument("step", type=int, nargs="?", default=1)
    ap.add_argument("--marker", type=int, default=BP_MARKER_OFF)
    ap.add_argument("--byte", type=int, default=1)
    ap.add_argument("--label", type=str, default="BP")
    a = ap.parse_args()
    probe(a.idx, a.step, a.marker, a.byte, a.label)
