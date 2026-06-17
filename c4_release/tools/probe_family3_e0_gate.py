#!/usr/bin/env python3
"""Validate the family-#3 e0-materializer structural gate: does a GENUINE var
local-frame STACK0 row (var_simple id250, ENT frame at 0xffe8/0xffe0) have
ADDR_B0_HI+14 > 4, while the if/bool empty-stack STACK0 leak row has
ADDR_B0_HI+14 <= 0?  If yes, promoting ADDR_B0_HI+14 to a hard requirement
cleanly excludes the if/bool leak without touching genuine e0 frames.  GPU-free.
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
    "var_simple(x=990)": "int main() { int x; x = 990; return x; }",
    "var_mul(a*b)": "int main() { int a; int b; a = 23; b = 47; return a * b; }",
    "if_gt(35>43)": "int main() { if (35 > 43) return 1; return 0; }",
}

# l16 e0-materializer is at block 34 (L20 region). Probe its INPUT.
E0_BI = 34


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


def main():
    ctx = build_gate_context()
    dp = ctx.dim_positions
    fwd = ctx.fwd
    # find the e0-materializer block: it's in the L20 layer16_lev_routing region.
    # Probe several candidate blocks; report ADDR_B0_HI+14 at STACK0 rows.
    dims = ["ADDR_B0_HI+14", "ADDR_B0_LO+0", "ADDR_B0_LO+8", "HAS_SE",
            "MARK_STACK0", "MEM_STORE"]
    def d(name):
        b = name.split('+')[0]
        off = int(name.split('+')[1]) if '+' in name else 0
        return dp[b] + off
    for label, src in PROGS.items():
        bc, data = compile_c(src)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        ot = oracle_tape_and_steps(bc, data, max_steps=20)
        full = prompt + ot.draft_tokens
        tok = torch.tensor([list(full)], dtype=torch.long)
        x = fwd.model.embed(tok)[0]
        for bi in range(E0_BI):
            x = step_block(fwd, x, bi)
        heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[E0_BI]
        x_in = _faithful_attn_forward(heads, x, nh, hd, sm1)
        print(f"\n=== {label}  opcodes={ot.opcodes} ===")
        for r in range(prefix, x_in.shape[0]):
            if x_in[r, dp["MARK_STACK0"]] <= 0.5:
                continue
            step = (r - prefix) // STEP_TOKENS
            addr_hi14 = float(x_in[r, d("ADDR_B0_HI+14")])
            tag = "e0/e8-FRAME" if addr_hi14 > 4 else "non-frame"
            vals = " ".join(f"{nm}={float(x_in[r, d(nm)]):+.2f}" for nm in dims)
            print(f"  step{step:2} row{r-prefix:3} [{tag}] {vals}")


if __name__ == "__main__":
    main()
