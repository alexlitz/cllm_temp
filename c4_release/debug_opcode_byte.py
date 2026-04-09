#!/usr/bin/env python3
"""Debug OPCODE_BYTE_LO/HI population at PC marker."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("=" * 80)
print("OPCODE_BYTE POPULATION ANALYSIS")
print("=" * 80)

# Generate step 0
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find PC marker
pc_marker_pos = None
for i in range(20, len(context)):
    if context[i] == Token.REG_PC:
        pc_marker_pos = i
        break

print(f"REG_PC marker at position {pc_marker_pos}")
print(f"IMM opcode = {Opcode.IMM} = 0x{Opcode.IMM:02x} = nibbles [lo=1, hi=0]")
print()

token_ids = torch.tensor([context[:pc_marker_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Check OPCODE_BYTE at PC marker through layers
    print("OPCODE_BYTE_LO/HI at PC marker through layers:")
    print()

    for layer_idx in range(6):  # Only check up to Layer 5
        if layer_idx > 0:
            x = model.blocks[layer_idx-1](x, kv_cache=None)

        print(f"After Layer {layer_idx-1 if layer_idx > 0 else 'Embedding'}:")

        # Check OPCODE_BYTE_LO nibbles
        ob_lo_vals = []
        for i in range(16):
            val = x[0, pc_marker_pos, BD.OPCODE_BYTE_LO + i].item()
            if abs(val) > 0.1:
                ob_lo_vals.append((i, val))

        if ob_lo_vals:
            print(f"  OPCODE_BYTE_LO: {ob_lo_vals}")
        else:
            print(f"  OPCODE_BYTE_LO: all near zero")

        # Check OPCODE_BYTE_HI nibbles
        ob_hi_vals = []
        for i in range(16):
            val = x[0, pc_marker_pos, BD.OPCODE_BYTE_HI + i].item()
            if abs(val) > 0.1:
                ob_hi_vals.append((i, val))

        if ob_hi_vals:
            print(f"  OPCODE_BYTE_HI: {ob_hi_vals}")
        else:
            print(f"  OPCODE_BYTE_HI: all near zero")

        # Check OP_IMM
        op_imm = x[0, pc_marker_pos, BD.OP_IMM].item()
        print(f"  OP_IMM: {op_imm:.3f}")
        print()

print("=" * 80)
print("EXPECTED BEHAVIOR")
print("=" * 80)
print("""
For IMM opcode (byte 1 = 0x01 = nibbles [lo=1, hi=0]):

1. Layer 5 attention should fetch opcode byte from CODE section
2. Layer 5 attention should write to OPCODE_BYTE_LO[1]=1.0, OPCODE_BYTE_HI[0]=1.0
3. Layer 5 FFN should decode: if OPCODE_BYTE_LO[1] AND OPCODE_BYTE_HI[0] then OP_IMM=5.0

If OPCODE_BYTE_LO/HI are not populated, the opcode fetch mechanism is broken.
If they are populated but OP_IMM is still 0, the decode FFN is broken.
""")
