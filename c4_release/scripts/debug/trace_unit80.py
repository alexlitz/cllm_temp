#!/usr/bin/env python3
"""Decompose unit 80's W_up dot product at REG_AX_mark to understand
why it's firing despite MARK_AX suppression."""
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
    pos = prefix_len + 5

    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for li in range(13):
            x = model.blocks[li](x)
        block13 = model.blocks[13]
        x_attn = block13.attn(x)
        x_pos = x_attn[0, pos]
        ffn = block13.ffn
        W_up = ffn.W_up
        if W_up.is_sparse:
            W_up = W_up.to_dense()

        u = 80
        wu = W_up[u]
        # Compute contributions from every dim
        contribs = wu * x_pos
        # Sort by abs
        topk = contribs.abs().topk(30)
        print(f"Unit 80 W_up dot product decomposition at REG_AX_mark (pos {pos}):")
        print(f"  b_up = {ffn.b_up[u].item():.2f}")
        total = 0.0
        for i in topk.indices:
            d = i.item()
            c = contribs[d].item()
            total += c
            # Find name
            dname = ""
            for attr in dir(BD):
                if attr.startswith("_"):
                    continue
                val_attr = getattr(BD, attr)
                if isinstance(val_attr, int) and val_attr == d:
                    dname = attr
                    break
            # Try ranges
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
            print(f"  d{d:>3d} ({dname:>20s}): W_up={wu[d].item():+.2f}  residual={x_pos[d].item():+.4f}  -> {c:+.2f}")
        # And the rest
        all_contrib_sum = contribs.sum().item()
        top_contrib_sum = sum(contribs[i].item() for i in topk.indices)
        print(f"\nSum of top-30 contribs: {top_contrib_sum:.2f}")
        print(f"Total W_up @ x sum (all dims): {all_contrib_sum:.2f}")
        print(f"Plus b_up: {all_contrib_sum + ffn.b_up[u].item():.2f}  (should match `up` value)")


if __name__ == "__main__":
    main()
