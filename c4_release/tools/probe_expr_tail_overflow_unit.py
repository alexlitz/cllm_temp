#!/usr/bin/env python3
"""Find block-41 unit 1790's gate condition: which residual dims drive its
W_up / W_gate, and how those dims differ between the depth-2 FAIL (add_mul_2)
and depth-1 PASS (mul_div_6) MUL AX rows.  Also identify the owning op by
matching the W_down OUTPUT_LO+0=-100 signature across the op source.  GPU-free.
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

TARGET_BI = 41
UNIT = 1790


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


def get_input(fwd, dp, src):
    bc, data = compile_c(src)
    prompt = build_code_prompt(bc, data); prefix = len(prompt)
    ot = oracle_tape_and_steps(bc, data, max_steps=20)
    mul_step = next(s for s, op in enumerate(ot.opcodes) if op == 27)
    full = prompt + ot.draft_tokens
    tok = torch.tensor([list(full)], dtype=torch.long)
    x = fwd.model.embed(tok)[0]
    mul_bi = next(bi for bi in range(len(fwd.blocks))
                  if type(fwd.blocks[bi][4].ffn).__name__ == "FlattenedALUMul")
    for bi in range(mul_bi):
        x = step_block(fwd, x, bi)
    heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[mul_bi]
    x_attn = _faithful_attn_forward(heads, x, nh, hd, sm1)
    base = mul_step * STEP_TOKENS
    ax_row = next(prefix + base + k for k in range(STEP_TOKENS)
                  if x_attn[prefix + base + k, dp["MARK_AX"]] > 0.5)
    x = block.ffn(x_attn.unsqueeze(0))[0]
    for bi in range(mul_bi + 1, TARGET_BI):
        x = step_block(fwd, x, bi)
    heads, nh, hd, sm1, block, is_comp, ff = fwd.blocks[TARGET_BI]
    x_in = _faithful_attn_forward(heads, x, nh, hd, sm1)
    return x_in[ax_row], ff


def main():
    ctx = build_gate_context()
    dp = ctx.dim_positions; fwd = ctx.fwd
    inv = {}
    for name, idx in dp.items():
        inv.setdefault(idx, name)
    name_of = lambda d: inv.get(d, f"dim{d}")

    vf, ff = get_input(fwd, dp, "int main(){return 18+12*3;}")  # FAIL depth2
    vp, _ = get_input(fwd, dp, "int main(){return 5*6/3;}")     # PASS depth1
    W_up, b_up, W_gate, b_gate, W_down = ff

    wu = W_up[UNIT]; wg = W_gate[UNIT]
    up_f = float(vf @ wu + b_up[UNIT]); up_p = float(vp @ wu + b_up[UNIT])
    g_f = float(vf @ wg + b_gate[UNIT]); g_p = float(vp @ wg + b_gate[UNIT])
    sil = torch.nn.functional.silu
    h_f = float(sil(torch.tensor(up_f)) * g_f)
    h_p = float(sil(torch.tensor(up_p)) * g_p)
    print(f"unit {UNIT} @ block {TARGET_BI}")
    print(f"  FAIL(depth2): up={up_f:+.3f} gate={g_f:+.3f} hidden={h_f:+.3f}")
    print(f"  PASS(depth1): up={up_p:+.3f} gate={g_p:+.3f} hidden={h_p:+.3f}")
    print(f"  b_up={float(b_up[UNIT]):+.3f}  b_gate={float(b_gate[UNIT]):+.3f}")

    # gate = b_gate + sum(dim w * resid). Show per-dim contribution to gate and
    # up, and the FAIL-vs-PASS delta in the resid that moves it.
    for tag, wvec in (("UP", wu), ("GATE", wg)):
        nz = torch.nonzero(wvec).flatten().tolist()
        print(f"\n  {tag} nonzero input dims ({len(nz)}):")
        rows = []
        for d in nz:
            w = float(wvec[d])
            cf = w * float(vf[d]); cp = w * float(vp[d])
            rows.append((abs(cf - cp), d, w, float(vf[d]), float(vp[d]), cf, cp))
        rows.sort(reverse=True)
        for delta, d, w, rf, rp, cf, cp in rows[:14]:
            print(f"    {name_of(d):26}(d{d:4}) w={w:+7.2f}  "
                  f"resid F={rf:+8.3f} P={rp:+8.3f}  contrib F={cf:+9.2f} P={cp:+9.2f}  Δ={cf-cp:+9.2f}")

    # W_down OUTPUT signature
    olo = dp["OUTPUT_LO"]
    print(f"\n  W_down[{UNIT}] -> OUTPUT_LO+0 = {float(W_down[olo, UNIT]):+.2f}")


if __name__ == "__main__":
    main()
