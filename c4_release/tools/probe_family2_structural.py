#!/usr/bin/env python3
"""Find the STRUCTURAL signal that separates a genuine memory-load STACK0 row
(test_si_li_roundtrip, where tail_stack0_store_loaded_byte_* MUST fire to
preserve 42) from the if/bool empty-stack STACK0 leak row (where it must go
inert).  Dumps the family's base_conditions dims + ADDR + the per-rule silu sum
at the input to the block-41 tail bank.  GPU-free.
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
    "si_li(0x200<-42)": "int main(){return 42;}",  # placeholder; use raw bytecode below
    "if_gt(35>43)": "int main(){if(35>43)return 7;return 9;}",
    "var_mul(a*b)": "int main(){int a=23;int b=47;return a*b;}",
}


def make_si_li():
    # IMM 0x200; PSH; IMM 42; SI; IMM 0x200; LI; EXIT
    from src.opcodes import Opcode
    from tests.conftest import _make_bytecode  # may not exist; fall back
    raise RuntimeError


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

    progs = []
    # SI/LI roundtrip raw bytecode (mirrors test_si_li_roundtrip):
    # IMM 0x200; PSH; IMM 42; SI; IMM 0x200; LI; EXIT
    from neural_vm.embedding import Opcode
    def asm(ops):
        bc = []
        for op in ops:
            if isinstance(op, tuple):
                opcode, imm = op
                bc.append(int(opcode) | (imm << 8))
            else:
                bc.append(int(op))
        return bc
    si_bc = asm([(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 42), Opcode.SI,
                 (Opcode.IMM, 0x200), Opcode.LI, Opcode.EXIT])
    progs.append(("si_li_roundtrip", si_bc, b""))
    for label, src in [("if_gt", PROGS["if_gt(35>43)"])]:
        bc, data = compile_c(src)
        progs.append((label, bc, data))

    base_dims = ["HAS_SE", "CMP+3", "MEM_STORE", "IS_BYTE", "MARK_STACK0",
                 "ADDR_B0_HI+14", "ADDR_B0_LO+0", "ADDR_B0_LO+8",
                 "H1+0", "H1+1"]
    def d(name):
        b = name.split('+')[0]
        off = int(name.split('+')[1]) if '+' in name else 0
        return dp[b] + off

    for label, bc, data in progs:
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
        print(f"\n=== {label}  opcodes={ot.opcodes} ===")
        for r in range(prefix, x_in.shape[0]):
            if x_in[r, dp["MARK_STACK0"]] <= 0.5:
                continue
            step = (r - prefix) // STEP_TOKENS
            # family-#2 silu sum (base + OUTPUT match at the argmax nibble)
            olo_amax = int(torch.argmax(x_in[r, dp["OUTPUT_LO"]:dp["OUTPUT_LO"]+16]).item())
            ohi_amax = int(torch.argmax(x_in[r, dp["OUTPUT_HI"]:dp["OUTPUT_HI"]+16]).item())
            silu = (1.0*float(x_in[r, d("HAS_SE")]) + 0.5*float(x_in[r, d("CMP+3")])
                    + 1.0*float(x_in[r, d("MEM_STORE")]) - 100.0*float(x_in[r, d("IS_BYTE")])
                    + 1.0*float(x_in[r, dp["OUTPUT_LO"]+olo_amax])
                    + 1.0*float(x_in[r, dp["OUTPUT_HI"]+ohi_amax]) - 25.0)
            vals = " ".join(f"{nm}={float(x_in[r, d(nm)]):+.2f}" for nm in base_dims)
            fires = "FIRES" if silu > 0 else "inert"
            print(f"  step{step:2} row{r-prefix:3} family2_silu_sum={silu:+.1f} [{fires}] "
                  f"OUT_LO_amax={olo_amax} OUT_HI_amax={ohi_amax}")
            print(f"           {vals}")


if __name__ == "__main__":
    main()
