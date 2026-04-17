#!/usr/bin/env python3
"""Debug PC carry-forward and increment for step 2."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

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
init_len = len(context)

# Generate step 1 (JSR)
print("=== Step 1 generation ===")
step1_start = len(context)
for i in range(26):
    token = model.generate_next(context)
    context.append(token)
    if token >= 256:
        name = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 268: "STACK0", 262: "STEP_END"}.get(token, str(token))
        print(f"  Token at idx {len(context)-1}: {token} ({name})")

# Find PC marker and PC byte 0 in step 1
pc_marker_idx = None
for i in range(step1_start, len(context)):
    if context[i] == Token.REG_PC:
        pc_marker_idx = i
        break

pc_byte0_idx = pc_marker_idx + 1
pc_byte0_token = context[pc_byte0_idx]
print(f"\nStep 1 PC marker at idx {pc_marker_idx}")
print(f"Step 1 PC byte 0 at idx {pc_byte0_idx}, value = {pc_byte0_token}")

# Now let's trace the hidden states during step 2 generation
print("\n=== Step 2 generation with trace ===")

device = next(model.parameters()).device

# Generate first token of step 2 (should be STEP_END if previous step completed)
# Actually we already generated 26 tokens so step 1 should be done
# Let's manually inspect what we need

# Prepare context for step 2
step2_start = len(context)

# Generate PC marker token for step 2
token = model.generate_next(context)
context.append(token)
print(f"Step 2 token 0: {token} ({['PC', 'AX', 'SP', 'BP', '', '', '', '', '', '', '', 'STACK0', 'STEP_END'].get(token-257, str(token)) if token >= 257 else str(token)})")

if token == Token.REG_PC:
    print("  Good - PC marker")

    # Now before generating PC byte 0, let's trace the hidden state
    with torch.no_grad():
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        hidden, _ = model._forward_layers(input_ids, None)

        # Get state at the last position (PC marker position)
        state = hidden[0, -1, :]

        print(f"\n  === At PC marker position (step 2) ===")
        print(f"  HAS_SE = {state[BD.HAS_SE].item():.2f}")
        print(f"  MARK_PC = {state[BD.MARK_PC].item():.2f}")

        # Check L1H0/L1H1 at previous step's PC byte 0 position
        prev_pc_byte0_pos = pc_byte0_idx
        prev_state = hidden[0, prev_pc_byte0_pos, :]
        print(f"\n  === At step 1 PC byte 0 position (idx {prev_pc_byte0_pos}) ===")
        print(f"  EMBED_LO max: [{prev_state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}] = {prev_state[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.2f}")
        print(f"  EMBED_HI max: [{prev_state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}] = {prev_state[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.2f}")
        print(f"  L1H0[PC] = {prev_state[BD.L1H0 + 0].item():.2f}")
        print(f"  L1H1[PC] = {prev_state[BD.L1H1 + 0].item():.2f}")

        # Check what EMBED values are at current PC marker position
        print(f"\n  === EMBED at current PC marker ===")
        embed_lo_max = state[BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
        embed_hi_max = state[BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
        print(f"  EMBED_LO max: [{embed_lo_max}] = {state[BD.EMBED_LO:BD.EMBED_LO+16].max().item():.2f}")
        print(f"  EMBED_HI max: [{embed_hi_max}] = {state[BD.EMBED_HI:BD.EMBED_HI+16].max().item():.2f}")
        print(f"  EMBED value decoded: {embed_lo_max + 16*embed_hi_max}")
        print(f"  Expected: 26 (from step 1 PC)")

        # Check OUTPUT values at current PC marker position
        print(f"\n  === OUTPUT at current PC marker ===")
        out_lo_max = state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
        out_hi_max = state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
        print(f"  OUTPUT_LO max: [{out_lo_max}] = {state[BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item():.2f}")
        print(f"  OUTPUT_HI max: [{out_hi_max}] = {state[BD.OUTPUT_HI:BD.OUTPUT_HI+16].max().item():.2f}")
        print(f"  OUTPUT value decoded: {out_lo_max + 16*out_hi_max}")
        print(f"  Expected: 34 (26 + 8)")

# Continue generating step 2
print("\n=== Continue step 2 ===")
for i in range(25):
    token = model.generate_next(context)
    context.append(token)
    if i < 15:
        name = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 268: "STACK0", 262: "STEP_END", 263: "HALT"}.get(token, str(token))
        print(f"  Token {i+2}: {token} ({name})")
    if token in [Token.STEP_END, Token.HALT]:
        break
