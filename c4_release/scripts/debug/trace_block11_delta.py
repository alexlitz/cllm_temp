#!/usr/bin/env python3
"""Trace what block 11 (= original L10 = ALUAndOrXor) writes to OUTPUT/CARRY
at REG_AX_mark for an IMM-only program."""
import sys
import torch
import torch.nn.functional as F

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token


def _make_bc(prog):
    bc = []
    for item in prog:
        if isinstance(item, tuple):
            op, imm = item
            bc.append((imm << 8) | op)
        else:
            bc.append(item)
    return bc


def main():
    val = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    runner._memory = {}
    runner._mem_history = {}
    runner._mem_access_order = []
    model = runner.model
    model.eval()
    device = next(model.parameters()).device

    prog = [(Opcode.IMM, val), Opcode.EXIT]
    bytecode = _make_bc(prog)
    context = runner._build_context(bytecode, b"", [], b"")
    prefix_len = len(context)

    with torch.no_grad():
        for i in range(Token.STEP_TOKENS):
            tok = model.generate_next(context, use_incremental=False)
            context.append(tok)

    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    pos = prefix_len + 5  # REG_AX_mark

    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for li in range(11):
            x = model.blocks[li](x)
        x_before = x.clone()
        # Apply block 11 step by step
        attn_out = model.blocks[11].attn(x)
        ffn_out = model.blocks[11].ffn(attn_out)
        delta_attn = attn_out[0, pos] - x_before[0, pos]
        delta_ffn = ffn_out[0, pos] - attn_out[0, pos]
        delta = ffn_out[0, pos] - x_before[0, pos]
        x_after = ffn_out
        print(f"Block 11 type: ffn={type(model.blocks[11].ffn).__name__}", flush=True)
        # Print attn vs ffn deltas at key dims
        print(f"\n  At pos {pos} (REG_AX_mark):")
        for d, name in [(BD.OUTPUT_LO + 5, "OLO+5"), (BD.OUTPUT_LO + 0, "OLO+0"),
                        (BD.OUTPUT_HI + 0, "OHI+0"), (BD.CARRY + 1, "CARRY+1"),
                        (BD.CARRY + 2, "CARRY+2")]:
            print(f"    {name:>10s}: before={x_before[0, pos, d].item():+.3f}  after_attn={attn_out[0, pos, d].item():+.3f}  (Δ={delta_attn[d].item():+.3f})  after_ffn={ffn_out[0, pos, d].item():+.3f}  (Δ={delta_ffn[d].item():+.3f})")

        # Show top 20 by |delta|
        topk = delta.abs().topk(40)
        print(f"\nTop 40 dims by |delta| at pos {pos} (REG_AX_mark):")
        for i in topk.indices:
            d = i.item()
            dname = ""
            for attr in dir(BD):
                if attr.startswith("_"):
                    continue
                val_attr = getattr(BD, attr)
                if isinstance(val_attr, int) and val_attr == d:
                    dname = attr
                    break
            if not dname:
                for base_attr, count in [("OUTPUT_LO", 16), ("OUTPUT_HI", 16), ("EMBED_LO", 16),
                                          ("EMBED_HI", 16), ("ALU_LO", 16), ("ALU_HI", 16),
                                          ("ADDR_KEY", 48), ("CARRY", 16), ("TEMP", 16),
                                          ("H0", 7), ("H1", 7), ("H2", 7), ("H3", 7),
                                          ("H4", 7), ("AX_CARRY_LO", 16), ("AX_CARRY_HI", 16)]:
                    if hasattr(BD, base_attr):
                        base_val = getattr(BD, base_attr)
                        if base_val <= d < base_val + count:
                            dname = f"{base_attr}+{d - base_val}"
                            break
            print(f"  d{d:>3d} ({dname:>20s}): before={x_before[0, pos, d].item():+.3f}  after={x_after[0, pos, d].item():+.3f}  delta={delta[d].item():+.3f}")


if __name__ == "__main__":
    main()
