#!/usr/bin/env python3
"""Debug why AX gets SP's value instead of immediate."""
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

# Generate step 0
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find marker positions
reg_bp_pos = None
reg_sp_pos = None
reg_ax_pos = None

for pos in range(20, len(context)):
    if context[pos] == 258:  # REG_BP
        reg_bp_pos = pos
    elif context[pos] == 259:  # REG_SP
        reg_sp_pos = pos
    elif context[pos] == 260:  # REG_AX
        reg_ax_pos = pos

print("REGISTER MARKER POSITIONS:")
print(f"  REG_BP: {reg_bp_pos}")
print(f"  REG_SP: {reg_sp_pos}")
print(f"  REG_AX: {reg_ax_pos}")

print("\nREGISTER VALUES:")
for name, pos in [("BP", reg_bp_pos), ("SP", reg_sp_pos), ("AX", reg_ax_pos)]:
    if pos:
        bytes_val = [context[pos + 1 + i] for i in range(4)]
        value = sum(b << (i*8) for i, b in enumerate(bytes_val))
        print(f"  {name}: {bytes_val} = 0x{value:08x} ({value})")

print("\n" + "=" * 80)
print("LAYER-BY-LAYER ANALYSIS AT REG_AX MARKER")
print("=" * 80)

token_ids = torch.tensor([context[:reg_ax_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    for layer_idx in range(16):
        x = model.blocks[layer_idx](x, kv_cache=None)

        if layer_idx in [5, 6, 14, 15]:  # Key layers for routing
            print(f"\nAfter Layer {layer_idx} at REG_AX (pos {reg_ax_pos}):")

            # Check FETCH (immediate value from Layer 5)
            if layer_idx >= 5:
                fetch_lo_vals = []
                for i in range(16):
                    val = x[0, reg_ax_pos, BD.FETCH_LO + i].item()
                    if abs(val) > 0.1:
                        fetch_lo_vals.append((i, val))

                if fetch_lo_vals:
                    print(f"  FETCH_LO: {fetch_lo_vals[:5]}")
                    max_idx, max_val = max(fetch_lo_vals, key=lambda x: abs(x[1]))
                    print(f"  FETCH_LO max: 0x{max_idx:02x} = {max_val:.3f}")
                else:
                    print(f"  FETCH_LO: all near zero")

            # Check EMBED (contains SP bytes from relay)
            embed_lo_vals = []
            for i in range(16):
                val = x[0, reg_ax_pos, BD.EMBED_LO + i].item()
                if abs(val) > 0.1:
                    embed_lo_vals.append((i, val))

            if embed_lo_vals:
                print(f"  EMBED_LO: {embed_lo_vals[:5]}")
                # Decode EMBED as byte
                embed_byte = max(embed_lo_vals, key=lambda x: x[1])[0]
                print(f"  EMBED_LO implies byte: 0x{embed_byte:02x}")

            # Check if opcode is detected
            if layer_idx >= 2:
                op_imm = x[0, reg_ax_pos, BD.OP_IMM].item()
                print(f"  OP_IMM: {op_imm:.3f}")

            # Check OUTPUT (final output before head)
            if layer_idx >= 6:
                output_lo_vals = []
                for i in range(16):
                    val = x[0, reg_ax_pos, BD.OUTPUT_LO + i].item()
                    if abs(val) > 0.1:
                        output_lo_vals.append((i, val))

                if output_lo_vals:
                    print(f"  OUTPUT_LO: {output_lo_vals[:5]}")
                    # Decode OUTPUT as byte
                    output_byte = max(output_lo_vals, key=lambda x: abs(x[1]))[0]
                    print(f"  OUTPUT_LO implies byte: 0x{output_byte:02x}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

print("""
BP gets 0x2a (correct immediate value)
SP gets 0x00010000 (correct initial stack pointer)
AX gets 0x00010000 (WRONG - same as SP!)

The issue is likely in Layer 6 FFN routing logic:
- IMM opcode should route FETCH → OUTPUT
- But it's routing SP → OUTPUT instead

Possible causes:
1. OP_IMM flag is not set (opcode not detected)
2. Layer 6 FFN routing logic has a bug
3. SP is being incorrectly copied to AX markers
""")
