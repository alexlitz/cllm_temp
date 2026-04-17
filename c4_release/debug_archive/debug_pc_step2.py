#!/usr/bin/env python3
"""Debug PC computation for step 2 (after JSR)."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Simple JSR test: JSR 26, EXIT, NOP, IMM 42, EXIT
bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26
    Opcode.EXIT,              # Instr 1: Never reached
    Opcode.NOP,               # Instr 2: Padding
    Opcode.IMM | (42 << 8),   # Instr 3: Target at PC=26
    Opcode.EXIT,              # Instr 4: Exit
]

runner = AutoregressiveVMRunner()
model = runner.model

# Build initial context
context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context length: {len(context)}")

# Generate step 1 (JSR)
print("\n=== Step 1 (JSR) ===")
step1_start = len(context)
step1_tokens = []
for i in range(35):
    token = model.generate_next(context)
    context.append(token)
    step1_tokens.append(token)
    if token == Token.STEP_END:
        break

# Find PC bytes in step 1
pc_marker_idx = None
for i in range(step1_start, len(context)):
    if context[i] == Token.REG_PC:
        pc_marker_idx = i
        break

if pc_marker_idx:
    pc_bytes = context[pc_marker_idx+1:pc_marker_idx+5]
    pc_value = sum(b << (i*8) for i, b in enumerate(pc_bytes))
    print(f"Step 1 PC: marker at {pc_marker_idx}, bytes {list(pc_bytes)} = {pc_value}")
else:
    print("No PC marker found in step 1!")
    exit()

# Analyze step 2 initial state
print("\n=== Step 2 Analysis ===")
step2_start = len(context)

# Get embeddings for step 1's PC byte 0
pc_byte0_idx = pc_marker_idx + 1
pc_byte0_token = context[pc_byte0_idx]
print(f"Step 1 PC byte 0: token index {pc_byte0_idx}, value {pc_byte0_token}")
print(f"  Expected EMBED_LO[{pc_byte0_token % 16}] = 1, EMBED_HI[{pc_byte0_token // 16}] = 1")

# Run one token generation to see the state
with torch.no_grad():
    device = next(model.parameters()).device

    # Forward pass up to the last token (which will inform the next generation)
    # The model computes internal state based on all context

    # Let's generate the first token of step 2 and inspect state
    print("\nGenerating step 2 token 0 (STEP_END)...")
    token = model.generate_next(context)
    context.append(token)
    name = {262: "STEP_END"}.get(token, str(token))
    print(f"  Generated: {token} ({name})")

    print("\nGenerating step 2 token 1 (PC marker)...")
    token = model.generate_next(context)
    context.append(token)
    name = {257: "PC"}.get(token, str(token))
    print(f"  Generated: {token} ({name})")

    # Now generate PC byte 0 and inspect the logits
    print("\nGenerating step 2 token 2 (PC byte 0)...")

    # Get hidden state at this position
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    hidden, _ = model.forward_layers(input_ids)

    # Get the last position's state
    state = hidden[0, -1, :]  # [D]

    # Check dimensions at the PC marker (which is at position -1 in context)
    print(f"\n  At PC marker (last position in context):")
    print(f"    HAS_SE = {state[BD.HAS_SE].item():.3f}")
    print(f"    MARK_PC = {state[BD.MARK_PC].item():.3f}")

    # Check EMBED_LO/HI (should have previous PC from carry-forward)
    embed_lo = state[BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = state[BD.EMBED_HI:BD.EMBED_HI+16]
    lo_max = embed_lo.argmax().item()
    hi_max = embed_hi.argmax().item()
    print(f"    EMBED_LO max: [{lo_max}] = {embed_lo[lo_max].item():.3f}")
    print(f"    EMBED_HI max: [{hi_max}] = {embed_hi[hi_max].item():.3f}")
    print(f"    EMBED value = {lo_max + 16*hi_max}")

    # Check OUTPUT_LO/HI (should have PC + 8 from L3 FFN)
    out_lo = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    out_hi = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    out_lo_max = out_lo.argmax().item()
    out_hi_max = out_hi.argmax().item()
    print(f"    OUTPUT_LO max: [{out_lo_max}] = {out_lo[out_lo_max].item():.3f}")
    print(f"    OUTPUT_HI max: [{out_hi_max}] = {out_hi[out_hi_max].item():.3f}")
    print(f"    OUTPUT value = {out_lo_max + 16*out_hi_max}")

    # Generate the actual token
    token = model.generate_next(context)
    context.append(token)
    print(f"\n  Generated PC byte 0: {token}")
    print(f"  Expected: 2 (34 % 256 = 34, byte 0 = 34)")

# Continue generating step 2
print("\n=== Completing Step 2 ===")
for i in range(30):
    token = model.generate_next(context)
    context.append(token)
    if i < 10:
        name = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "STACK0", 262: "STEP_END", 263: "HALT"}.get(token, str(token))
        print(f"  Token {i+4}: {token} ({name})")
    if token == Token.HALT or token == Token.STEP_END:
        break

# Parse step 2 results
ax_idx = None
pc2_idx = None
for j in range(step2_start, len(context)):
    if context[j] == Token.REG_AX:
        ax_idx = j
    if context[j] == Token.REG_PC and j > step2_start + 2:
        pc2_idx = j

if pc2_idx:
    pc2_bytes = context[pc2_idx+1:pc2_idx+5]
    pc2_value = sum(b << (i*8) for i, b in enumerate(pc2_bytes))
    print(f"\nStep 2 PC: bytes {list(pc2_bytes)} = {pc2_value}")
    print(f"Expected: 34 (26 + 8)")

if ax_idx:
    ax_bytes = context[ax_idx+1:ax_idx+5]
    ax_value = sum(b << (i*8) for i, b in enumerate(ax_bytes))
    print(f"Step 2 AX: bytes {list(ax_bytes)} = {ax_value}")
