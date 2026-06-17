#!/usr/bin/env python3
"""CPU probe: trace OUTPUT_LO/HI band + LM-head argmax at the MUL-step AX-marker
row block-by-block AFTER the MUL block, to find which downstream block crushes
the correct MUL result (36) and lets a stray 0x01 win.  GPU-free.

ROOT CAUSE (2026-06-17, CPU-only via interp_oracle_gate FaithfulForwardCache):
The single-byte expr-chain leak (expr_add_mul / expr_mul_div) is NOT the
operand-gather, NOT the MUL block, and NOT the AX-byte-1 dump root. The MUL
block emits the CORRECT product (post-MUL OUTPUT byte0 = 36, big margins). It is
then destroyed downstream on chained (stack-depth>=2) ALU steps:
  * block 33 (PureFFN, 14-head attn = L20 layer16_lev_routing region): its
    ATTENTION adds ~+5345 to BOTH OUTPUT_LO and OUTPUT_HI bands at the depth-2
    MUL AX row (Δ=0 at a depth-1 row -> PASS), flooding OUTPUT_HI band-sum to
    ~5768.
  * block 41 (PureFFN = _tail_bit32_result_correction family) unit 1790: an
    overflow guard gated `silu(up)`, up = -500500 + 100*Σ(OUTPUT_HI). The flood
    trips it (up=+32428), and W_down[1790,OUTPUT_LO+0]=-100 crushes OUTPUT_LO+0
    by -3.2M -> stray 0x01 wins the AX byte-0 dump.
This is the documented STACK0 OUTPUT-band-crush / framing-desync mega-root
(project_var_fulltrace_stack0_frame_desync + project_ff_tail_emitter_mega_root);
the tail bank is width-sensitive (project_l10_tail_bank_width_sensitive) so it
needs a coordinated L20-attn + L25-tail build, NOT a single-rule tweak."""
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


def onehot_val(vec, base, width=16):
    seg = vec[base:base + width]
    return int(torch.argmax(seg).item()), float(seg.max().item())


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
    return x, type(block.ffn).__name__


def main():
    ctx = build_gate_context()
    dp = ctx.dim_positions
    fwd = ctx.fwd
    # MUL block
    mul_bi = next(bi for bi in range(len(fwd.blocks))
                  if type(fwd.blocks[bi][4].ffn).__name__ == "FlattenedALUMul")

    src = "int main(){return 18+12*3;}"  # add_mul_2, MUL=36 expected
    bc, data = compile_c(src)
    prompt = build_code_prompt(bc, data)
    prefix = len(prompt)
    ot = oracle_tape_and_steps(bc, data, max_steps=20)
    mul_step = next(s for s, op in enumerate(ot.opcodes) if op == 27)
    full = prompt + ot.draft_tokens

    # forward to AX row of MUL step
    tok = torch.tensor([list(full)], dtype=torch.long)
    x = fwd.model.embed(tok)[0]
    # apply MUL block's attention to find AX row
    for bi in range(mul_bi):
        x, _ = step_block(fwd, x, bi)
    heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[mul_bi]
    x_attn = _faithful_attn_forward(heads, x, nh, hd, sm1)
    base = mul_step * STEP_TOKENS
    ax_row = next(prefix + base + k for k in range(STEP_TOKENS)
                  if x_attn[prefix + base + k, dp["MARK_AX"]] > 0.5)
    print(f"MUL block={mul_bi}  mul_step={mul_step}  ax_row={ax_row}")

    # run MUL FFN
    x = block.ffn(x_attn.unsqueeze(0))[0]
    def report(tag):
        v = x[ax_row]
        lo = onehot_val(v, dp["OUTPUT_LO"])
        hi = onehot_val(v, dp["OUTPUT_HI"])
        byte0 = lo[0] | (hi[0] << 4)
        lg = v @ fwd.head_w.t() + fwd.head_b
        t = int(torch.argmax(lg).item())
        # top-3 tokens
        top = torch.topk(lg, 4)
        tops = [(int(i), round(float(s), 1)) for i, s in zip(top.indices, top.values)]
        print(f"  {tag:30} OUTPUT byte0={byte0:3} (lo-nib={lo[0]:2} m={lo[1]:+8.2f} | "
              f"hi-nib={hi[0]:2} m={hi[1]:+8.2f})  argmax_tok={t}  top4={tops}")
    report(f"post-MUL(blk{mul_bi})")

    for bi in range(mul_bi + 1, len(fwd.blocks)):
        x, ffn_name = step_block(fwd, x, bi)
        report(f"post-blk{bi}({ffn_name[:14]})")


if __name__ == "__main__":
    main()
