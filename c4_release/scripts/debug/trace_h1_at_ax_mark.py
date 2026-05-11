#!/usr/bin/env python3
"""Trace H1[1] activation at REG_AX_marker to understand carry-prop firing."""
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

    # Inspect H1[*] flags at all step-0 positions
    print(f"\n=== H1 (threshold-from-marker) at step 0 positions ===", flush=True)
    print(f"H1 base = {BD.H1}, IS_MARK={BD.IS_MARK}, IS_BYTE={BD.IS_BYTE}", flush=True)
    print(f"MARK_AX = {BD.MARK_AX}, BD.H1+1 = {BD.H1+1}", flush=True)
    print(f"BYTE_INDEX_0={BD.BYTE_INDEX_0}, BYTE_INDEX_1={BD.BYTE_INDEX_1}, BYTE_INDEX_2={BD.BYTE_INDEX_2}, BYTE_INDEX_3={BD.BYTE_INDEX_3}", flush=True)

    # Run model up through L11 (BitwiseByteProp doesn't exist yet, but after L11 = ALUAndOrXor=block11)
    # Inspect at every position, what's H1[0..6], BYTE_INDEX_0..3, IS_MARK, IS_BYTE, MARK_AX, MARK_PC
    with torch.no_grad():
        x = model.embed(token_ids, active_opcode=model._active_opcode)
        for li in range(12):  # blocks 0..11 inclusive
            x = model.blocks[li](x)
        # After L11 (= original L10 ALUAndOrXor)
        print(f"\nAfter L11 (= original L10):", flush=True)
        pos_names = ["RPC.mk", "PC.0", "PC.1", "PC.2", "PC.3",
                     "RAX.mk", "AX.0", "AX.1", "AX.2", "AX.3",
                     "RSP.mk", "SP.0", "SP.1", "SP.2", "SP.3",
                     "RBP.mk", "BP.0", "BP.1", "BP.2", "BP.3",
                     "ST0.mk", "ST.0", "ST.1", "ST.2", "ST.3",
                     "MEM.mk", "MA.0", "MA.1", "MA.2", "MA.3",
                     "MV.0", "MV.1", "MV.2", "MV.3", "SE"]
        for k in range(min(Token.STEP_TOKENS, 35)):
            pos = prefix_len + k
            v = x[0, pos]
            h1_vals = [v[BD.H1 + i].item() for i in range(7)]
            byteidx = [v[BD.BYTE_INDEX_0 + i].item() for i in range(4)]
            is_mark = v[BD.IS_MARK].item()
            is_byte = v[BD.IS_BYTE].item()
            mark_ax = v[BD.MARK_AX].item()
            mark_pc = v[BD.MARK_PC].item()
            tok = context[pos]
            print(f"  [{k:02d}] tok={tok:3d} ({pos_names[k]:6s}) IS_MARK={is_mark:.1f} IS_BYTE={is_byte:.1f} "
                  f"MARK_AX={mark_ax:+.1f} MARK_PC={mark_pc:+.1f}  H1={[f'{v_:+.1f}' for v_ in h1_vals]}  "
                  f"BYTE_I={[f'{v_:.1f}' for v_ in byteidx]}", flush=True)


if __name__ == "__main__":
    main()
