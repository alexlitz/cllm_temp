#!/usr/bin/env python3
"""Diagnose why transformer predictions don't match DraftVM in strict mode."""
import sys
sys.path.insert(0, '.')

import torch
from src.compiler import compile_c
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.speculative import DraftVM

# Simple test program
code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print("=" * 70)
print("DIAGNOSING STRICT MODE FAILURE")
print("=" * 70)

print(f"\nProgram: {code}")
print(f"Bytecode ({len(bytecode)} instructions):")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    print(f"  [{i}] {instr:08x} = op={op:02x} ({op}), imm={imm:06x} ({imm})")

# Build context
print("\nBuilding context...")
context = [Token.CODE_START]
for instr in bytecode:
    op = instr & 0xFF
    imm = instr >> 8
    context.append(op)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
context.append(Token.CODE_END)
context.append(Token.DATA_START)
context.append(Token.DATA_END)

print(f"Context length: {len(context)}")
print("Context tokens:")
for i, tok in enumerate(context):
    if tok == Token.CODE_START:
        print(f"  [{i:3d}] {tok:3d} = CODE_START")
    elif tok == Token.CODE_END:
        print(f"  [{i:3d}] {tok:3d} = CODE_END")
    elif tok == Token.DATA_START:
        print(f"  [{i:3d}] {tok:3d} = DATA_START")
    elif tok == Token.DATA_END:
        print(f"  [{i:3d}] {tok:3d} = DATA_END")
    elif 0 <= i - 1 < len(bytecode) * 5:
        idx_in_code = i - 1
        instr_idx = idx_in_code // 5
        byte_in_instr = idx_in_code % 5
        if byte_in_instr == 0:
            print(f"  [{i:3d}] {tok:3d} = Instruction {instr_idx} opcode")
        else:
            print(f"  [{i:3d}] {tok:3d} = Instruction {instr_idx} imm byte {byte_in_instr-1}")

# Execute DraftVM
print("\n" + "=" * 70)
print("DRAFTVM EXECUTION")
print("=" * 70)

vm = DraftVM(bytecode)
print(f"\nInitial state:")
print(f"  idx={vm.idx}, pc={vm.pc}, ax={vm.ax}, sp={vm.sp:08x}, bp={vm.bp:08x}")

print(f"\nExecuting first instruction...")
vm.step()
print(f"After step 1:")
print(f"  idx={vm.idx}, pc={vm.pc}, ax={vm.ax}, sp={vm.sp:08x}, bp={vm.bp:08x}")

draft = vm.draft_tokens()
print(f"\nDraft tokens (35 total):")
print(f"  REG_PC: {draft[0]} (token), {draft[1:5]} (bytes) = {draft[1] | (draft[2]<<8) | (draft[3]<<16) | (draft[4]<<24)}")
print(f"  REG_AX: {draft[5]} (token), {draft[6:10]} (bytes) = {draft[6] | (draft[7]<<8) | (draft[8]<<16) | (draft[9]<<24)}")
print(f"  REG_SP: {draft[10]} (token), {draft[11:15]} (bytes) = {draft[11] | (draft[12]<<8) | (draft[13]<<16) | (draft[14]<<24):08x}")

# Execute Transformer
print("\n" + "=" * 70)
print("TRANSFORMER PREDICTIONS")
print("=" * 70)

print("\nCreating model...")
model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

print("\nPredicting tokens one by one...")
pred_tokens = []
for i in range(5):
    full_ctx = context + pred_tokens
    token_ids = torch.tensor([full_ctx], dtype=torch.long)
    logits = model.forward(token_ids)
    pred = logits[0, -1, :].argmax(-1).item()

    match = "✓" if pred == draft[i] else "✗"

    if i == 0:
        print(f"  Token {i} (REG_PC marker): pred={pred:3d}, expected={draft[i]:3d} {match}")
    elif i == 1:
        print(f"  Token {i} (PC byte 0):     pred={pred:3d}, expected={draft[i]:3d} {match}")
        print(f"    Transformer predicts PC byte 0 = {pred}")
        print(f"    DraftVM expects PC byte 0 = {draft[i]}")
        print(f"    Difference: {abs(pred - draft[i])} bytes")
    else:
        print(f"  Token {i} (PC byte {i-1}):     pred={pred:3d}, expected={draft[i]:3d} {match}")

    pred_tokens.append(pred)

    if pred != draft[i]:
        print(f"\n  MISMATCH at token {i}!")
        print(f"  Predicted value as PC: {pred | (pred_tokens[2]<<8 if len(pred_tokens)>2 else 0) | (pred_tokens[3]<<16 if len(pred_tokens)>3 else 0) | (pred_tokens[4]<<24 if len(pred_tokens)>4 else 0) if i==1 else 'N/A'}")
        print(f"  Expected PC: {draft[1] | (draft[2]<<8) | (draft[3]<<16) | (draft[4]<<24)}")
        break

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print(f"\nFirst instruction: op={bytecode[0] & 0xFF} (opcode {bytecode[0] & 0xFF})")
print(f"  Opcode 3 = JSR (Jump to Subroutine)")
print(f"  Immediate: {bytecode[0] >> 8} = {(bytecode[0] >> 8):06x}")

print(f"\nDraftVM behavior:")
print(f"  After JSR: PC={vm.pc}")

print(f"\nTransformer prediction:")
print(f"  Predicts PC byte 0 = {pred_tokens[1] if len(pred_tokens) > 1 else 'N/A'}")

print(f"\nExpected vs Actual:")
print(f"  DraftVM PC: {draft[1]} (byte 0)")
print(f"  Transformer PC: {pred_tokens[1] if len(pred_tokens) > 1 else 'N/A'} (byte 0)")
print(f"  Difference: {abs(pred_tokens[1] - draft[1]) if len(pred_tokens) > 1 else 'N/A'}")
