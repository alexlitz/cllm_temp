#!/usr/bin/env python3
"""Capture the structural + OUTPUT dims at the SHR-by-8 AX marker row (the
firing position of tail_shr_marker_byte0_01) for the test_shr_8bit smoke
program 0x100>>8=1, at the input to the block-41 tail bank.  This pins the
genuine-SHR signature so the family-#4 OUTPUT->gate_terms move stays
byte-identical on the real SHR row.  GPU-free.
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
    TAIL_BI = 41
    # 0x100>>8=1: IMM 0x100; PSH; IMM 8; SHR; EXIT
    src = "int main(){return 256>>8;}"
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
    W_up, b_up, W_gate, b_gate, W_down = ff
    UNIT = 1790
    print("opcodes per step:", ot.opcodes)
    dims = ["MARK_AX", "H1+1", "TEMP+7", "OP_SHR", "OP_IMM", "OP_LEA",
            "OUTPUT_HI_THIS_STEP+0", "OUTPUT_HI_THIS_STEP+2", "OUTPUT_LO+10", "IS_BYTE"]
    def d(name):
        b = name.split('+')[0]
        off = int(name.split('+')[1]) if '+' in name else 0
        return dp[b] + off
    for r in range(prefix, x_in.shape[0]):
        if x_in[r, dp["MARK_AX"]] <= 0.5:
            continue
        step = (r - prefix) // STEP_TOKENS
        opc = ot.opcodes[step] if step < len(ot.opcodes) else -1
        up = float(x_in[r] @ W_up[UNIT] + b_up[UNIT])
        g = float(x_in[r] @ W_gate[UNIT] + b_gate[UNIT])
        h = float(torch.nn.functional.silu(torch.tensor(up)) * g)
        vals = " ".join(f"{nm}={float(x_in[r, d(nm)]):+.1f}" for nm in dims)
        print(f"step{step:2} row{r-prefix:3} opc={opc} unit1790: up={up:+.1f} gate={g:+.2f} hidden={h:+.1f}")
        print(f"     {vals}")


if __name__ == "__main__":
    main()
