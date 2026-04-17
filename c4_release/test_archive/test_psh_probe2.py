#!/usr/bin/env python3
"""Deeper probe: check AX_CARRY at AX marker and ALU at STACK0 before/after L6."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)
model = runner.model

program = [Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.ADD, Opcode.EXIT]
context = runner._build_context(program, b"", [], "")
runner._bytecode = program
runner._stdin_buffer = []
runner._stdin_pos = 0
runner._tool_handler = None
runner._tool_call_id = 0
runner._last_sp = 0
runner._memory = {}
runner._heap_base = 0x10000
runner._heap_ptr = 0x10000
runner._alloc_sizes = {}

# Generate step 0 (IMM 5)
for _ in range(Token.STEP_TOKENS):
    t = model.generate_next(context)
    context.append(t)
print(f"Step 0: AX={runner._extract_register(context, Token.REG_AX)}")

# Generate step 1 (PSH) up to STACK0 marker
for _ in range(21):  # 0-20: REG_PC(5) + REG_AX(5) + REG_SP(5) + REG_BP(5) + STACK0 marker
    t = model.generate_next(context)
    context.append(t)

# Now context ends with STACK0 marker. Do a full forward pass to probe.
print(f"\n=== Probing at STACK0 marker (just after it was generated) ===")
token_ids = torch.tensor([context], dtype=torch.long)
x = model.embed(token_ids)
model._inject_bytecode_addresses(token_ids, x)

pos_stack0 = len(context) - 1  # STACK0 marker
# Find AX marker position in step 1
pos_ax = None
for i in range(len(context) - 1, max(0, len(context) - 40), -1):
    if context[i] == Token.REG_AX:
        pos_ax = i
        break

print(f"STACK0 marker at context pos {pos_stack0}, AX marker at pos {pos_ax}")

for layer_idx, block in enumerate(model.blocks):
    x_before = x.clone()
    # Run attention only
    attn = block.attn
    attn_out = attn(x_before)
    x_after_attn = x_before + attn_out  # x after attention (before FFN)

    if layer_idx == 6:
        # Check AX_CARRY at AX marker BEFORE L6 attention
        ax_state = x_before[0, pos_ax, :].detach()
        print(f"\n--- Before L6 attn: AX_CARRY at AX marker (pos {pos_ax}) ---")
        for k in range(16):
            v = ax_state[BD.AX_CARRY_LO + k].item()
            if abs(v) > 0.01:
                print(f"  AX_CARRY_LO[{k}] = {v:.4f}")
        for k in range(16):
            v = ax_state[BD.AX_CARRY_HI + k].item()
            if abs(v) > 0.01:
                print(f"  AX_CARRY_HI[{k}] = {v:.4f}")

        # Check ALU at STACK0 marker BEFORE L6 attention
        s0_before = x_before[0, pos_stack0, :].detach()
        print(f"\n--- Before L6 attn: ALU at STACK0 marker (pos {pos_stack0}) ---")
        for k in range(16):
            v = s0_before[BD.ALU_LO + k].item()
            if abs(v) > 0.01:
                print(f"  ALU_LO[{k}] = {v:.4f}")
        any_alu_before = any(abs(s0_before[BD.ALU_LO + k].item()) > 0.01 for k in range(16))
        if not any_alu_before:
            print("  (all ALU_LO < 0.01)")
        for k in range(16):
            v = s0_before[BD.ALU_HI + k].item()
            if abs(v) > 0.01:
                print(f"  ALU_HI[{k}] = {v:.4f}")

        # Check ALU at STACK0 marker AFTER L6 attention (before FFN)
        s0_after_attn = x_after_attn[0, pos_stack0, :].detach()
        print(f"\n--- After L6 attn: ALU at STACK0 marker ---")
        for k in range(16):
            v = s0_after_attn[BD.ALU_LO + k].item()
            if abs(v) > 0.01:
                print(f"  ALU_LO[{k}] = {v:.4f}")
        for k in range(16):
            v = s0_after_attn[BD.ALU_HI + k].item()
            if abs(v) > 0.01:
                print(f"  ALU_HI[{k}] = {v:.4f}")

        # Check EMBED at STACK0 marker (identity carry source)
        print(f"\n--- Before L6 attn: EMBED at STACK0 marker ---")
        for k in range(16):
            v = s0_before[BD.EMBED_LO + k].item()
            if abs(v) > 0.01:
                print(f"  EMBED_LO[{k}] = {v:.4f}")
        for k in range(16):
            v = s0_before[BD.EMBED_HI + k].item()
            if abs(v) > 0.01:
                print(f"  EMBED_HI[{k}] = {v:.4f}")

    # Full layer
    x = block(x_before)

# After all layers, check OUTPUT at STACK0 marker
s0_final = x[0, pos_stack0, :].detach()
print(f"\n--- After all layers: OUTPUT at STACK0 marker ---")
for k in range(16):
    v = s0_final[BD.OUTPUT_LO + k].item()
    if abs(v) > 0.01:
        print(f"  OUTPUT_LO[{k}] = {v:.4f}")
for k in range(16):
    v = s0_final[BD.OUTPUT_HI + k].item()
    if abs(v) > 0.01:
        print(f"  OUTPUT_HI[{k}] = {v:.4f}")

# What does the head produce?
logits = model.head(x)
logits_s0 = logits[0, pos_stack0, :].detach()
top5 = torch.topk(logits_s0[:256], 5)
print(f"\n--- Head output (top5 bytes) at STACK0 marker ---")
for v, i in zip(top5.values.tolist(), top5.indices.tolist()):
    print(f"  byte {i} (0x{i:02X}): logit {v:.4f}")
