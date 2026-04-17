#!/usr/bin/env python3
"""Debug the neural pathway for IMM instruction."""
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
print("IMM NEURAL PATHWAY ANALYSIS")
print("=" * 80)
print(f"Program: IMM 42, EXIT")
print(f"Expected: AX = 42 (0x2a)")
print()

# Generate up to REG_AX position
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find REG_AX position
reg_ax_pos = None
for i in range(20, len(context)):
    if context[i] == 260:  # REG_AX
        reg_ax_pos = i
        break

print(f"REG_AX at position {reg_ax_pos}")
print()

token_ids = torch.tensor([context[:reg_ax_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    print("LAYER-BY-LAYER ANALYSIS AT REG_AX POSITION")
    print("=" * 80)

    # Layer 0: Embedding
    print("\nAfter Embedding:")
    print(f"  EMBED_LO[0x2a]: {x[0, reg_ax_pos, BD.EMBED_LO + 0x2a].item():.3f}")

    # Layer 2: Opcode detection
    for i in range(2):
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter Layer 2 (opcode detection):")
    op_imm = x[0, reg_ax_pos, BD.OP_IMM].item()
    print(f"  OP_IMM: {op_imm:.3f}")

    if abs(op_imm - 1.0) < 0.1:
        print("  ✓ OP_IMM flag is SET")
    else:
        print(f"  ✗ OP_IMM flag is NOT SET (expected ~1.0, got {op_imm:.3f})")

    # Check other opcode flags
    other_flags = []
    for name in dir(BD):
        if name.startswith('OP_') and name != 'OP_IMM':
            idx = getattr(BD, name)
            if isinstance(idx, int) and 0 <= idx < 512:
                val = x[0, reg_ax_pos, idx].item()
                if abs(val) > 0.1:
                    other_flags.append((name, val))

    if other_flags:
        print(f"  Other opcode flags set: {other_flags[:5]}")

    # Layer 5: Code fetch (FETCH should contain immediate value)
    for i in range(2, 5):
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter Layer 5 (code fetch):")
    fetch_vals = []
    # FETCH is stored as nibbles (16 each for LO and HI), not bytes
    for i in range(16):
        val_lo = x[0, reg_ax_pos, BD.FETCH_LO + i].item()
        val_hi = x[0, reg_ax_pos, BD.FETCH_HI + i].item()
        if abs(val_lo) > 0.1 or abs(val_hi) > 0.1:
            fetch_vals.append((i, val_lo, val_hi))

    if fetch_vals:
        print(f"  FETCH nibble values:")
        for nib, val_lo, val_hi in sorted(fetch_vals, key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)[:5]:
            print(f"    Nibble {nib} (0x{nib:x}): LO={val_lo:.3f}, HI={val_hi:.3f}")

        # 0x2a = 42 = 0b101010 = nibbles [0xa, 0x2]
        # Lo nibble = 0xa (10), Hi nibble = 0x2
        if any(nib == 0xa and abs(val_lo) > 0.5 for nib, val_lo, val_hi in fetch_vals):
            print(f"  ✓ FETCH_LO[0xa] is set (lo nibble of 42)")
        else:
            print(f"  ✗ FETCH_LO[0xa] not set (expected for 0x2a)")

        if any(nib == 0x2 and abs(val_hi) > 0.5 for nib, val_lo, val_hi in fetch_vals):
            print(f"  ✓ FETCH_HI[0x2] is set (hi nibble of 42)")
        else:
            print(f"  ✗ FETCH_HI[0x2] not set (expected for 0x2a)")
    else:
        print(f"  ✗ FETCH is all near zero (code fetch not working)")

    # Layer 6: I/O routing (should route FETCH → OUTPUT when OP_IMM=1)
    x_before_l6 = x.clone()
    x = model.blocks[5](x, kv_cache=None)

    print(f"\nAfter Layer 6 (I/O routing):")
    output_vals = []
    # OUTPUT is also stored as nibbles (16 each for LO and HI)
    for i in range(16):
        val_lo = x[0, reg_ax_pos, BD.OUTPUT_LO + i].item()
        val_hi = x[0, reg_ax_pos, BD.OUTPUT_HI + i].item()
        if abs(val_lo) > 0.1 or abs(val_hi) > 0.1:
            output_vals.append((i, val_lo, val_hi))

    if output_vals:
        print(f"  OUTPUT nibble values:")
        for nib, val_lo, val_hi in sorted(output_vals, key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)[:5]:
            print(f"    Nibble {nib} (0x{nib:x}): LO={val_lo:.3f}, HI={val_hi:.3f}")

        # 0x2a = 42 = nibbles [0xa, 0x2]
        if any(nib == 0xa and abs(val_lo) > 0.5 for nib, val_lo, val_hi in output_vals):
            print(f"  ✓ OUTPUT_LO[0xa] is set (lo nibble of 42 - routing works!)")
        else:
            print(f"  ✗ OUTPUT_LO[0xa] not set (expected for 0x2a)")

        if any(nib == 0x2 and abs(val_hi) > 0.5 for nib, val_lo, val_hi in output_vals):
            print(f"  ✓ OUTPUT_HI[0x2] is set (hi nibble of 42 - routing works!)")
        else:
            print(f"  ✗ OUTPUT_HI[0x2] not set (expected for 0x2a)")
    else:
        print(f"  ✗ OUTPUT is all near zero (routing not working)")

    # Check Layer 6 FFN units that write to OUTPUT
    print(f"\n  Checking Layer 6 FFN routing units:")
    block6 = model.blocks[5]
    ffn = block6.ffn

    # Find units that read OP_IMM
    op_imm_readers = (ffn.W_up[:, BD.OP_IMM].abs() > 0.1).nonzero(as_tuple=True)[0]
    print(f"  Units that read OP_IMM: {len(op_imm_readers)} units")

    if len(op_imm_readers) > 0:
        print(f"  First 10 OP_IMM readers: {op_imm_readers[:10].tolist()}")

        # Check if any of these write to OUTPUT_LO
        for unit in op_imm_readers[:10]:
            output_weights = ffn.W_down[:, unit].abs()
            output_lo_writes = output_weights[BD.OUTPUT_LO:BD.OUTPUT_LO+256].max().item()
            if output_lo_writes > 0.1:
                print(f"    Unit {unit}: reads OP_IMM, writes to OUTPUT_LO (max={output_lo_writes:.3f})")
    else:
        print(f"  ✗ No units read OP_IMM flag")

    # Continue through remaining layers
    for i in range(6, 16):
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter all 16 layers:")
    final_output = []
    for i in range(16):
        val_lo = x[0, reg_ax_pos, BD.OUTPUT_LO + i].item()
        val_hi = x[0, reg_ax_pos, BD.OUTPUT_HI + i].item()
        if abs(val_lo) > 0.1 or abs(val_hi) > 0.1:
            final_output.append((i, val_lo, val_hi))

    if final_output:
        print(f"  OUTPUT nibble values:")
        for nib, val_lo, val_hi in sorted(final_output, key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)[:5]:
            print(f"    Nibble {nib} (0x{nib:x}): LO={val_lo:.3f}, HI={val_hi:.3f}")

    # Get final prediction
    logits = model.head(x)
    pred = logits[0, reg_ax_pos, :256].argmax(-1).item()

    print(f"\n  Final prediction: 0x{pred:02x} ({pred})")
    if pred == 0x2a:
        print(f"  ✓ Predicts 0x2a (42) - CORRECT!")
    else:
        print(f"  ✗ Predicts 0x{pred:02x} ({pred}) - expected 0x2a (42)")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print("""
This script traces the IMM instruction through all layers to identify
where the neural pathway breaks:

1. Layer 2: Should set OP_IMM flag
2. Layer 5: Should fetch immediate value to FETCH_LO
3. Layer 6: Should route FETCH_LO → OUTPUT_LO when OP_IMM=1

If any step fails, that's where the weights need to be added/fixed.
""")
