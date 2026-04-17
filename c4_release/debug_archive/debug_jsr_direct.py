#!/usr/bin/env python3
"""Direct JSR diagnostic - inspect hidden states manually."""

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token, _SetDim as BD
import sys

print("=" * 80)
print("JSR DIRECT DIAGNOSTIC")
print("=" * 80)

# Simple JSR test
bytecode = [
    Opcode.JSR | (25 << 8),  # 0: JSR to byte 25
    Opcode.EXIT,
]

print("\nBytecode: JSR 25, EXIT")
print("Expected: PC should jump to 25\n")

runner = AutoregressiveVMRunner()
model = runner.model

# Build context
context = runner._build_context(bytecode, b"", [], "")
prefix_len = len(context)

print(f"Prefix length: {prefix_len} tokens")
print(f"First few context tokens: {context[:10]}")

# Find the CODE token for JSR instruction
# The bytecode is encoded in the context
print(f"\nLooking for JSR opcode in context...")

# Check if JSR opcode is in embeddings
jsr_opcode = Opcode.JSR
print(f"JSR opcode value: {jsr_opcode} (0x{jsr_opcode:02x})")

# Build first step context and run forward pass
print("\n" + "=" * 80)
print("RUNNING FORWARD PASS ON INITIAL CONTEXT")
print("=" * 80)

# Convert context to tensor
with torch.no_grad():
    token_ids = torch.tensor([context], dtype=torch.long)
    print(f"Input shape: {token_ids.shape}")
    
    # Get embeddings
    x = model.embed(token_ids)
    print(f"Embeddings shape: {x.shape}")
    
    # Check if first instruction has JSR opcode set
    # Instruction bytes are in the prefix
    print(f"\nChecking opcode encoding in embeddings...")
    
    # The first instruction should be at a specific position in prefix
    # Let's check several positions for OP_JSR flag
    if hasattr(BD, 'OP_JSR'):
        op_jsr_dim = BD.OP_JSR
        print(f"OP_JSR dimension: {op_jsr_dim}")
        
        # Check first 20 positions for OP_JSR flag
        for pos in range(min(20, x.shape[1])):
            op_jsr_val = x[0, pos, op_jsr_dim].item()
            if op_jsr_val > 0.5:
                print(f"  Position {pos}: OP_JSR = {op_jsr_val:.2f} ✓")
    
    # Run through layers
    print(f"\nRunning through transformer layers...")
    
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i == 5:  # After L5 - should have TEMP[0]
            print(f"\nAfter Layer 5 (L5):")
            # Find PC marker in the output
            # The first step's PC marker should be at the end
            # Let's check last position and a few before it
            for offset in [0, 1, 2, 3, 4, 5]:
                pos = x.shape[1] - 1 - offset
                if pos >= 0:
                    temp0 = x[0, pos, BD.TEMP + 0].item()
                    mark_pc = x[0, pos, BD.MARK_PC].item() if hasattr(BD, 'MARK_PC') else 0
                    if temp0 > 0.1 or mark_pc > 0.5:
                        print(f"  Pos {pos} (offset -{offset}): TEMP[0]={temp0:.4f}, MARK_PC={mark_pc:.4f}")
        
        if i == 6:  # After L6 - should have OUTPUT
            print(f"\nAfter Layer 6 (L6):")
            for offset in [0, 1, 2, 3, 4, 5]:
                pos = x.shape[1] - 1 - offset
                if pos >= 0:
                    # Check OUTPUT_LO to see if it's 25 or 10
                    output_bits = [x[0, pos, BD.OUTPUT_LO + k].item() for k in range(16)]
                    output_val = sum(k for k in range(16) if output_bits[k] > 0.5)
                    
                    mark_pc = x[0, pos, BD.MARK_PC].item() if hasattr(BD, 'MARK_PC') else 0
                    if mark_pc > 0.5 or output_val > 0:
                        print(f"  Pos {pos}: OUTPUT_LO={output_val}, MARK_PC={mark_pc:.4f}")

print("\n" + "=" * 80)

# Now generate the first step token by token
print("\nGENERATING FIRST STEP (token by token)...")
print("=" * 80)

for step_token in range(35):
    next_token = model.generate_next(context)
    context.append(next_token)
    
    if step_token == 0:
        print(f"Token 1: {next_token} (should be PC marker = {Token.REG_PC})")
    elif step_token < 5:
        print(f"Token {step_token+1}: {next_token} (PC byte {step_token-1})")

# Decode final PC
pc_bytes = context[-35+1:-35+5]
pc = sum((b & 0xFF) << (i * 8) for i, b in enumerate(pc_bytes))

print(f"\nFinal PC: 0x{pc:08x} ({pc})")
if pc == 25:
    print("✅ JSR WORKED!")
    sys.exit(0)
else:
    print(f"❌ JSR FAILED - PC should be 25, got {pc}")
    sys.exit(1)
