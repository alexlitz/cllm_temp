#!/usr/bin/env python3
"""Scan for PC values across all positions in step 0."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Use runner's context builder
context = runner._build_context(bytecode, b'', [])
print(f"Initial context: {len(context)} tokens\n")

# Generate step 0
print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"Step 0 complete at position {len(context)-1}\n")
        break

# Scan all positions for PC
se_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(16):
        x = model.blocks[i](x, kv_cache=None)

print("Scanning for PC values (OUTPUT_LO[10] > 0.5)...")
print("Position | Token | OUTPUT_LO[10] | PC bytes")
print("-" * 60)

for pos in range(len(context)):
    output_lo_10 = x[0, pos, BD.OUTPUT_LO + 10].item()

    if abs(output_lo_10) > 0.5:
        # Decode PC
        pc_bytes = []
        for i in range(4):
            vals = x[0, pos, BD.OUTPUT_LO + i*16 : BD.OUTPUT_LO + (i+1)*16]
            byte_val = vals.argmax(-1).item()
            pc_bytes.append(byte_val)
        pc = sum(b << (i*8) for i, b in enumerate(pc_bytes))

        tok_name = context[pos]
        if tok_name < 256:
            tok_str = f"0x{tok_name:02x}"
        elif tok_name == Token.STEP_END:
            tok_str = "STEP_END"
        elif tok_name == Token.REG_PC:
            tok_str = "REG_PC"
        else:
            tok_str = str(tok_name)

        print(f"{pos:8} | {tok_str:10} | {output_lo_10:13.3f} | 0x{pc:08x}")

# Also check for PC_LO values
print("\nScanning for PC_LO values...")
print("Position | Token | PC_LO[10] | Value")
print("-" * 50)

for pos in range(len(context)):
    pc_lo_10 = x[0, pos, BD.PC_LO + 10].item()

    if abs(pc_lo_10) > 0.5:
        tok_name = context[pos]
        if tok_name < 256:
            tok_str = f"0x{tok_name:02x}"
        elif tok_name == Token.STEP_END:
            tok_str = "STEP_END"
        elif tok_name == Token.REG_PC:
            tok_str = "REG_PC"
        else:
            tok_str = str(tok_name)

        print(f"{pos:8} | {tok_str:10} | {pc_lo_10:9.3f} | {pc_lo_10:.3f}")
