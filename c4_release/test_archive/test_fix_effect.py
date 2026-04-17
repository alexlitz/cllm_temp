#!/usr/bin/env python3
"""Test if CLEAN_EMBED_HI change affects OUTPUT_HI[12] for LEA 0."""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

print("Checking relevant dimensions:")
print(f"  CLEAN_EMBED_HI = {BD.CLEAN_EMBED_HI} (new location after fix)")
print(f"  OUTPUT_HI = {BD.OUTPUT_HI}")
print(f"  TEMP = {BD.TEMP}")
print(f"  MUL_ACCUM = {BD.MUL_ACCUM}")
print()

# Check if there's any overlap that could cause issues
print("Dimension ranges:")
print(f"  CLEAN_EMBED_HI: {BD.CLEAN_EMBED_HI}-{BD.CLEAN_EMBED_HI + 15}")
print(f"  MUL_ACCUM: {BD.MUL_ACCUM}-{BD.MUL_ACCUM + 15}")
print(f"  Overlap: {max(BD.CLEAN_EMBED_HI, BD.MUL_ACCUM)}-{min(BD.CLEAN_EMBED_HI + 16, BD.MUL_ACCUM + 16)}")
print()

# The issue at dim 202 (OUTPUT_HI[12]) should NOT be related to CLEAN_EMBED_HI
print("Testing dim 202 (OUTPUT_HI[12]) vs CLEAN_EMBED_HI range:")
print(f"  Dim 202 in CLEAN_EMBED_HI range? {BD.CLEAN_EMBED_HI <= 202 < BD.CLEAN_EMBED_HI + 16}")
print()

# Check how CLEAN_EMBED_HI[12] interacts with MUL_ACCUM
print("CLEAN_EMBED_HI[12] overlap:")
clean_hi_12 = BD.CLEAN_EMBED_HI + 12
print(f"  CLEAN_EMBED_HI[12] = dim {clean_hi_12}")
print(f"  In MUL_ACCUM range? {BD.MUL_ACCUM <= clean_hi_12 < BD.MUL_ACCUM + 16}")
if BD.MUL_ACCUM <= clean_hi_12 < BD.MUL_ACCUM + 16:
    print(f"  = MUL_ACCUM[{clean_hi_12 - BD.MUL_ACCUM}]")
