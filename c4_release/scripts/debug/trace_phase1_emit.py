#!/usr/bin/env python3
"""Trace what the model emits autoregressively for a 1-step IMM EXIT."""
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

    # Print step 0 with expected values
    pos_names = ["RPC_mk", "PC.0", "PC.1", "PC.2", "PC.3", "RAX_mk", "AX.0", "AX.1", "AX.2", "AX.3",
                 "RSP_mk", "SP.0", "SP.1", "SP.2", "SP.3", "RBP_mk", "BP.0", "BP.1", "BP.2", "BP.3",
                 "ST0_mk", "ST.0", "ST.1", "ST.2", "ST.3", "MEM_mk", "MA.0", "MA.1", "MA.2", "MA.3",
                 "MV.0", "MV.1", "MV.2", "MV.3", "SE"]
    expected = {
        1: 10, 5: 258,
        6: val, 7: 0, 8: 0, 9: 0,
    }
    print(f"\nIMM {val} EXIT step 0 emissions:")
    for k in range(Token.STEP_TOKENS):
        tok = context[prefix_len + k]
        exp = expected.get(k, None)
        mark = f" [expected {exp}, {'OK' if tok == exp else 'WRONG'}]" if exp is not None else ""
        print(f"  pos{k:02d} ({pos_names[k]:6s}) = {tok}{mark}")

    # Compute AX
    ax = 0
    for j in range(4):
        ax |= (context[prefix_len + 6 + j] & 0xFF) << (j * 8)
    print(f"\nFinal AX: 0x{ax:08x} = {ax} (expected: {val})")

    # Now inspect the residual at REG_AX_mark and AX_byte0 driving positions
    token_ids = torch.tensor([context], dtype=torch.long, device=device)
    OUT_LO = BD.OUTPUT_LO
    OUT_HI = BD.OUTPUT_HI
    EMBED_LO = BD.EMBED_LO
    EMBED_HI = BD.EMBED_HI

    drive_positions = {
        "REG_AX_mark (drives AX_b0)": prefix_len + 5,
        "AX_b0 (drives AX_b1)":       prefix_len + 6,
        "AX_b1 (drives AX_b2)":       prefix_len + 7,
        "AX_b2 (drives AX_b3)":       prefix_len + 8,
    }
    with torch.no_grad():
        logits = model.forward(token_ids)[0]  # [S, vocab]
        for label, pos in drive_positions.items():
            log_p = logits[pos]
            top = log_p.topk(8)
            print(f"\n  [{label}] pos={pos}: argmax={log_p.argmax().item()}")
            for k in range(8):
                print(f"    rank{k}: tok={top.indices[k].item()} logit={top.values[k].item():.2f}")


if __name__ == "__main__":
    main()
