#!/usr/bin/env python3
"""Check what operations are working/broken."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

print("Loading model...", flush=True)
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model loaded.\n")

def run_steps(bytecode, num_steps):
    """Run neural VM for num_steps and return final tokens."""
    context = build_context(bytecode)
    draft = DraftVM(bytecode)

    for _ in range(num_steps):
        draft.step()

    current_ctx = context[:]
    with torch.no_grad():
        for step in range(num_steps):
            for i in range(39):
                token_ids = torch.tensor([current_ctx], dtype=torch.long)
                logits = model(token_ids)
                predicted = logits[0, -1, :].argmax().item()
                current_ctx.append(predicted)

    # Extract registers from last step
    last_step = current_ctx[-39:]
    ax = last_step[6] | (last_step[7] << 8) | (last_step[8] << 16) | (last_step[9] << 24)
    sp = last_step[11] | (last_step[12] << 8) | (last_step[13] << 16) | (last_step[14] << 24)
    bp = last_step[16] | (last_step[17] << 8) | (last_step[18] << 16) | (last_step[19] << 24)

    return ax, sp, bp, draft

def test(name, bytecode, steps, expected_ax=None, expected_sp=None, expected_bp=None):
    ax, sp, bp, draft = run_steps(bytecode, steps)

    errors = []
    if expected_ax is not None and ax != expected_ax:
        errors.append(f"AX: got 0x{ax:08X}, expected 0x{expected_ax:08X}, draft 0x{draft.ax:08X}")
    if expected_sp is not None and sp != expected_sp:
        errors.append(f"SP: got 0x{sp:08X}, expected 0x{expected_sp:08X}, draft 0x{draft.sp:08X}")
    if expected_bp is not None and bp != expected_bp:
        errors.append(f"BP: got 0x{bp:08X}, expected 0x{expected_bp:08X}, draft 0x{draft.bp:08X}")

    if errors:
        print(f"{name}: FAIL")
        for e in errors:
            print(f"  {e}")
        return False
    print(f"{name}: PASS")
    return True

results = {"pass": 0, "fail": 0}

def run(name, bytecode, steps, **kwargs):
    if test(name, bytecode, steps, **kwargs):
        results["pass"] += 1
    else:
        results["fail"] += 1

print("=== Single-step operations ===")
run("NOP", [Opcode.NOP], 1, expected_ax=0)
run("IMM 42", [Opcode.IMM | (42 << 8)], 1, expected_ax=42)
run("LEA 8", [Opcode.LEA | (8 << 8)], 1, expected_ax=0x10008)

print("\n=== Two-step operations ===")
run("NOP+LEA 256", [Opcode.NOP, Opcode.LEA | (256 << 8)], 2, expected_ax=0x10100)
run("NOP+LEA 65535", [Opcode.NOP, Opcode.LEA | (65535 << 8)], 2, expected_ax=0x1FFFF)
run("IMM 10, PSH", [Opcode.IMM | (10 << 8), Opcode.PSH], 2, expected_ax=10, expected_sp=0x20000-4)

print("\n=== Multi-step arithmetic ===")
run("ADD 5+3", [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD], 4, expected_ax=8)
run("SUB 10-3", [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.SUB], 4, expected_ax=7)

print("\n=== Control flow ===")
run("JMP 2", [Opcode.JMP | (2 << 8), Opcode.NOP], 2, expected_ax=0)  # JMP skips NOP, lands on NOP
run("ENT 8", [Opcode.ENT | (8 << 8)], 1, expected_sp=0x20000-8, expected_bp=0x20000)

print(f"\n=== Summary: {results['pass']}/{results['pass']+results['fail']} passed ===")
