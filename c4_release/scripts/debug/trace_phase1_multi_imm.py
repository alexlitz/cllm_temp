#!/usr/bin/env python3
"""Trace Phase 1 IMM emission - inspect what flags are active at each position
that drive the IMM write at REG_AX_mark.

In particular: at pos 25 (REG_AX_mark), check after each layer what opcodes /
flags are 'high' that might cause downstream FFNs (L13 SHIFT, L14 cleanup,
L15 nibble copy) to misfire.
"""
import sys
import torch

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

    # Position 25 = REG_AX_mark. Drives AX_b0 emission.
    pos = prefix_len + 5
    print(f"Inspecting residual at pos {pos} (REG_AX_mark, drives AX_b0)", flush=True)

    # All BD names that we want to inspect
    flag_names = [
        "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
        "MARK_MEM_ADDR", "MARK_MEM_VAL",
        "IS_MARK", "IS_BYTE",
        "OP_IMM", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
        "OP_SHL", "OP_SHR",
        "OP_AND", "OP_OR", "OP_XOR",
        "OP_BZ", "OP_BNZ", "OP_JMP",
        "OP_LEA", "OP_LEV", "OP_ENT", "OP_JSR", "OP_PSH", "OP_LI", "OP_LC",
        "OP_EXIT",
        "HAS_SE", "NEXT_SE",
        "MEM_STORE",
        "CONST",
    ]

    # Resolve BD attributes
    bd_dims = {}
    for n in flag_names:
        if hasattr(BD, n):
            bd_dims[n] = getattr(BD, n)

    # Run model layer-by-layer with hooks
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        layer_outs = [("embed", x)]
        for li, block in enumerate(model.blocks):
            x = block(x)
            layer_outs.append((f"L{li}", x))

    # Print flags at pos at each layer (only non-zero ones)
    print(f"\n=== Flag values at pos {pos} per layer ===", flush=True)
    for lname, xl in layer_outs:
        v = xl[0, pos]
        # Only show flags that are non-trivial
        active = []
        for n, d in bd_dims.items():
            val_v = v[d].item()
            if abs(val_v) > 0.05:
                active.append((n, val_v))
        # Also EMBED_LO/HI, OUTPUT_LO/HI top
        elo = v[BD.EMBED_LO:BD.EMBED_LO+16]
        ehi = v[BD.EMBED_HI:BD.EMBED_HI+16]
        olo = v[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
        ohi = v[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
        active_str = ", ".join([f"{n}={v_:.1f}" for n, v_ in active])
        olo_top = olo.abs().topk(3)
        ohi_top = ohi.abs().topk(3)
        olo_s = ",".join([f"{i.item()}:{olo[i].item():+.1f}" for i in olo_top.indices])
        ohi_s = ",".join([f"{i.item()}:{ohi[i].item():+.1f}" for i in ohi_top.indices])
        print(f"\n  [{lname}]", flush=True)
        print(f"    active flags: {active_str}", flush=True)
        print(f"    ELO_argmax={elo.argmax().item()}({elo.max():.2f}) EHI_argmax={ehi.argmax().item()}({ehi.max():.2f})", flush=True)
        print(f"    OLO_top={olo_s}  OHI_top={ohi_s}", flush=True)


if __name__ == "__main__":
    main()
