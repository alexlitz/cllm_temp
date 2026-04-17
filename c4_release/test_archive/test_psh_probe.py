#!/usr/bin/env python3
"""Probe hidden states during PSH step to debug AX=12 issue."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)
model = runner.model

program = [
    Opcode.IMM | (5 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.ADD,
    Opcode.EXIT,
]

# Run step 0 (IMM 5) by generating all 35 tokens
bytecode = program
context = runner._build_context(bytecode, b"", [], "")
runner._bytecode = bytecode
runner._stdin_buffer = []
runner._stdin_pos = 0
runner._tool_handler = None
runner._tool_call_id = 0
runner._last_sp = 0
runner._memory = {}
runner._heap_base = 0x10000
runner._heap_ptr = 0x10000
runner._alloc_sizes = {}

print(f"Context after build: len={len(context)}")

# Generate step 0 (IMM 5)
for _ in range(Token.STEP_TOKENS):
    t = model.generate_next(context)
    context.append(t)
print(f"Step 0 last token: {context[-1]} (expect {Token.STEP_END}={Token.STEP_END})")
sp0 = runner._extract_register(context, Token.REG_SP)
ax0 = runner._extract_register(context, Token.REG_AX)
pc0 = runner._extract_register(context, Token.REG_PC)
s00 = runner._extract_register(context, Token.STACK0)
print(f"Step 0: PC={pc0} AX={ax0} SP={sp0} STACK0={s00}")
runner._last_sp = sp0 if sp0 else 0

# Generate step 1 (PSH) token by token, probing hidden states
print("\n=== Step 1 (PSH) - token by token ===")
for tok_i in range(Token.STEP_TOKENS):
    t = model.generate_next(context)
    context.append(t)

    # Identify what register/position we just generated
    if tok_i == 0:
        print(f"  Token {tok_i}: REG_PC marker = {t} (expect {Token.REG_PC}={Token.REG_PC})")
    elif tok_i == 5:
        print(f"  Token {tok_i}: REG_AX marker = {t} (expect {Token.REG_AX}={Token.REG_AX})")
    elif tok_i == 10:
        print(f"  Token {tok_i}: REG_SP marker = {t} (expect {Token.REG_SP}={Token.REG_SP})")
    elif tok_i == 15:
        print(f"  Token {tok_i}: REG_BP marker = {t} (expect {Token.REG_BP}={Token.REG_BP})")
    elif tok_i == 20:
        print(f"  Token {tok_i}: STACK0 marker = {t} (expect {Token.STACK0}={Token.STACK0})")
    elif tok_i == 25:
        print(f"  Token {tok_i}: MEM marker = {t} (expect {Token.MEM}={Token.MEM})")
    elif tok_i == 34:
        print(f"  Token {tok_i}: STEP_END = {t} (expect {Token.STEP_END}={Token.STEP_END})")
    elif tok_i in (1,2,3,4):
        print(f"  Token {tok_i}: PC byte {tok_i-1} = {t}")
    elif tok_i in (6,7,8,9):
        print(f"  Token {tok_i}: AX byte {tok_i-6} = {t}")
    elif tok_i in (11,12,13,14):
        print(f"  Token {tok_i}: SP byte {tok_i-11} = {t}")
    elif tok_i in (21,22,23,24):
        print(f"  Token {tok_i}: STACK0 byte {tok_i-21} = {t}")

    # Probe hidden states right before SP byte 0 and STACK0 byte 0
    if tok_i in (10, 20):  # REG_SP marker, STACK0 marker
        # Full forward pass to get hidden states at current position
        token_ids = torch.tensor([context], dtype=torch.long)
        x = model.embed(token_ids)
        model._inject_bytecode_addresses(token_ids, x)
        for layer_idx, block in enumerate(model.blocks):
            x = block(x)
            if layer_idx == 6:  # After L6 (routing layer)
                pos = len(context) - 1  # current position
                state = x[0, pos, :].detach()
                marker = "SP" if tok_i == 10 else "STACK0"
                print(f"  --- L6 hidden state at {marker} marker ---")

                # Check CMP dims
                for c in range(8):
                    val = state[BD.CMP + c].item()
                    if abs(val) > 0.01:
                        print(f"    CMP[{c}] = {val:.4f}")

                # Check EMBED_LO/HI
                for k in range(16):
                    val = state[BD.EMBED_LO + k].item()
                    if abs(val) > 0.01:
                        print(f"    EMBED_LO[{k}] = {val:.4f}")
                for k in range(16):
                    val = state[BD.EMBED_HI + k].item()
                    if abs(val) > 0.01:
                        print(f"    EMBED_HI[{k}] = {val:.4f}")

                # Check ALU_LO/HI (relay from head 2)
                for k in range(16):
                    val = state[BD.ALU_LO + k].item()
                    if abs(val) > 0.01:
                        print(f"    ALU_LO[{k}] = {val:.4f}")
                for k in range(16):
                    val = state[BD.ALU_HI + k].item()
                    if abs(val) > 0.01:
                        print(f"    ALU_HI[{k}] = {val:.4f}")

                # Check OUTPUT_LO/HI
                for k in range(16):
                    val = state[BD.OUTPUT_LO + k].item()
                    if abs(val) > 0.01:
                        print(f"    OUTPUT_LO[{k}] = {val:.4f}")
                for k in range(16):
                    val = state[BD.OUTPUT_HI + k].item()
                    if abs(val) > 0.01:
                        print(f"    OUTPUT_HI[{k}] = {val:.4f}")

                # Check OP_PSH dim
                print(f"    OP_PSH = {state[BD.OP_PSH].item():.4f}")
                print(f"    MARK_SP = {state[BD.MARK_SP].item():.4f}")
                print(f"    MARK_STACK0 = {state[BD.MARK_STACK0].item():.4f}")
                print(f"    MARK_AX = {state[BD.MARK_AX].item():.4f}")

print(f"\nStep 1: PC={runner._extract_register(context, Token.REG_PC)} "
      f"AX={runner._extract_register(context, Token.REG_AX)} "
      f"SP={runner._extract_register(context, Token.REG_SP)} "
      f"STACK0={runner._extract_register(context, Token.STACK0)}")

# Apply handler
handler = runner._syscall_handlers.get(Opcode.PSH)
if handler:
    handler(context, [])
    print(f"After PSH handler: SP={runner._extract_register(context, Token.REG_SP)} "
          f"STACK0={runner._extract_register(context, Token.STACK0)}")
runner._last_sp = runner._extract_register(context, Token.REG_SP) or 0
