#!/usr/bin/env python3
"""For the 3 sweep target programs (if_gt 350, expr_add_mul 800, var_mul 275),
dump the STRUCTURAL gate dims (ADDR_B0_HI+14, ADDR_B0_LO+0/8, HAS_SE, MARK_*,
OP_SHR) AND the OUTPUT band at the STACK0 / AX marker rows at the input to the
block-41 tail bank, to confirm the de-coupling signal cleanly separates a
genuine e8/e0 frame from an if/bool empty / non-overflow row.  GPU-free.

This tells us whether ADDR_B0_HI+14 (the e8-frame high nibble) is the right
load-bearing silu requirement for tail_stack0_store_loaded (#2), the e0
materializer (#3), and whether OP_SHR cleanly gates tail_shr_marker (#4) once
its OUTPUT_HI silu term is moved to gate_terms.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import torch
from src.compiler import compile_c
from interp_oracle_gate import (
    build_gate_context, build_code_prompt, oracle_tape_and_steps,
    _faithful_attn_forward, STEP_TOKENS,
)

PROGS = {
    "if_gt(35>43)": ("int main(){if(35>43)return 7;return 9;}", None),
    "expr_add_mul(18+12*3)": ("int main(){return 18+12*3;}", None),
    "var_mul(a*b)": ("int main(){int a=23,b=47;return a*b;}", None),
}


def step_block(fwd, x, bi):
    heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[bi]
    x = _faithful_attn_forward(heads, x, nh, hd, sm1)
    if is_comp:
        x = block.ffn(x.unsqueeze(0))[0]
    else:
        W_up, b_up, W_gate, b_gate, W_down = ff
        up = x @ W_up.t() + b_up
        gate = x @ W_gate.t() + b_gate
        hidden = torch.nn.functional.silu(up) * gate
        x = x + hidden @ W_down.t()
    return x


def band_argmax(vec, dp, base, width=16):
    b = dp[base]
    seg = vec[b:b + width]
    return int(torch.argmax(seg).item()), float(seg.max().item()), float(seg.sum().item())


def main():
    ctx = build_gate_context()
    dp = ctx.dim_positions
    fwd = ctx.fwd
    TAIL_BI = 41
    structural = ["ADDR_B0_HI+14", "ADDR_B0_LO+0", "ADDR_B0_LO+8", "HAS_SE",
                  "MARK_STACK0", "MARK_AX", "OP_SHR", "OUTPUT_HI_THIS_STEP+0"]
    for label, (src, _) in PROGS.items():
        bc, data = compile_c(src)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        ot = oracle_tape_and_steps(bc, data, max_steps=20)
        full = prompt + ot.draft_tokens
        tok = torch.tensor([list(full)], dtype=torch.long)
        x = fwd.model.embed(tok)[0]
        for bi in range(TAIL_BI):
            x = step_block(fwd, x, bi)
        heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[TAIL_BI]
        x_in = _faithful_attn_forward(heads, x, nh, hd, sm1)
        print(f"\n=== {label}  (steps={len(ot.steps)}) ===")
        # Scan all rows, report STACK0 + AX marker rows with their structural dims.
        for r in range(prefix, x_in.shape[0]):
            is_s0 = x_in[r, dp["MARK_STACK0"]] > 0.5
            is_ax = x_in[r, dp["MARK_AX"]] > 0.5
            if not (is_s0 or is_ax):
                continue
            tag = "STACK0" if is_s0 else "AX"
            step = (r - prefix) // STEP_TOKENS
            vals = {k: float(x_in[r, dp[k.split('+')[0]] + (int(k.split('+')[1]) if '+' in k else 0)]) for k in structural}
            olo = band_argmax(x_in[r], dp, "OUTPUT_LO")
            ohi = band_argmax(x_in[r], dp, "OUTPUT_HI")
            dimstr = " ".join(f"{k}={vals[k]:+.1f}" for k in structural)
            print(f"  step{step:2} row{r-prefix:3} [{tag}] OUT_LO(amax={olo[0]} max={olo[1]:.0f} sum={olo[2]:.0f}) "
                  f"OUT_HI(amax={ohi[0]} max={ohi[1]:.0f} sum={ohi[2]:.0f})")
            print(f"           {dimstr}")


if __name__ == "__main__":
    main()
