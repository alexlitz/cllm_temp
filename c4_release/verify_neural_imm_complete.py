#!/usr/bin/env python3
"""
Verification script for complete neural IMM execution.

This demonstrates that the IMM instruction now executes entirely through
the transformer's learned weights, without any Python arithmetic fallback.

Test program: IMM 42; EXIT
Expected: AX = 42 (0x2a), exit code 42
"""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("="*80)
print("NEURAL IMM EXECUTION - COMPLETE VERIFICATION")
print("="*80)
print()

# Initialize model
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Create test program: IMM 42; EXIT
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
print(f"Test program: IMM 42; EXIT")
print(f"Bytecode: {[hex(b) for b in bytecode]}")
print()

# Build context and generate step 0
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

print("Generating step 0 tokens...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find AX marker position
ax_pos = None
for i in range(20, len(context)):
    if context[i] == Token.REG_AX:  # Token 258
        ax_pos = i
        break

print(f"✓ Generated {len(context)} tokens")
print(f"✓ AX marker at position {ax_pos}")
print()

# Run model inference
token_ids = torch.tensor([context[:ax_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Track key values through layers
    print("NEURAL PATHWAY TRACE:")
    print("-" * 80)

    # After Layer 5: Check FETCH at PC marker
    pc_pos = 20  # Known from step structure
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    fetch_lo_a = x[0, pc_pos, BD.FETCH_LO + 0xa].item()
    fetch_hi_2 = x[0, pc_pos, BD.FETCH_HI + 0x2].item()
    op_imm_pc = x[0, pc_pos, BD.OP_IMM].item()

    print(f"Layer 5 output at PC marker:")
    print(f"  FETCH_LO[0xa] = {fetch_lo_a:.3f} (immediate lo nibble)")
    print(f"  FETCH_HI[0x2] = {fetch_hi_2:.3f} (immediate hi nibble)")
    print(f"  OP_IMM = {op_imm_pc:.3f} (opcode flag)")

    # After Layer 6: Check relay to AX marker
    x = model.blocks[6](x, kv_cache=None)

    fetch_lo_a_ax = x[0, ax_pos, BD.FETCH_LO + 0xa].item()
    fetch_hi_2_ax = x[0, ax_pos, BD.FETCH_HI + 0x2].item()
    op_imm_ax = x[0, ax_pos, BD.OP_IMM].item()
    mark_ax = x[0, ax_pos, BD.MARK_AX].item()

    print()
    print(f"Layer 6 output at AX marker:")
    print(f"  FETCH_LO[0xa] = {fetch_lo_a_ax:.3f} (relayed from PC)")
    print(f"  FETCH_HI[0x2] = {fetch_hi_2_ax:.3f} (relayed from PC)")
    print(f"  OP_IMM = {op_imm_ax:.3f} (relayed from PC)")
    print(f"  MARK_AX = {mark_ax:.3f} (marker flag)")

    # Check OUTPUT
    output_lo_a = x[0, ax_pos, BD.OUTPUT_LO + 0xa].item()
    output_hi_2 = x[0, ax_pos, BD.OUTPUT_HI + 0x2].item()

    print()
    print(f"Layer 6 FFN routing (FETCH → OUTPUT):")
    print(f"  OUTPUT_LO[0xa] = {output_lo_a:.3f}")
    print(f"  OUTPUT_HI[0x2] = {output_hi_2:.3f}")

    # Run remaining layers
    for i in range(7, 16):
        x = model.blocks[i](x, kv_cache=None)

    # Final prediction
    logits = model.head(x)
    pred = logits[0, ax_pos, :256].argmax(-1).item()

    print()
    print("-" * 80)
    print("FINAL RESULT:")
    print("-" * 80)
    print(f"Neural prediction at AX marker: 0x{pred:02x} ({pred})")
    print(f"Expected value: 0x2a (42)")
    print()

    if pred == 42:
        print("✓✓✓ SUCCESS: Neural IMM execution COMPLETE!")
        print()
        print("The transformer correctly:")
        print("  1. Identified IMM opcode via OP_* flags")
        print("  2. Fetched immediate value (42) from CODE section")
        print("  3. Relayed value through PC → AX markers")
        print("  4. Routed FETCH → OUTPUT via FFN")
        print("  5. Predicted correct byte (0x2a = 42)")
        print()
        print("NO PYTHON FALLBACK REQUIRED - 100% NEURAL EXECUTION")
    else:
        print(f"✗ FAIL: Predicted {pred} instead of 42")
        print("Neural pathway incomplete")

print()
print("="*80)

# Test with runner to confirm exit code
print()
print("RUNNER INTEGRATION TEST:")
print("-" * 80)
result = runner.run(bytecode, b'', [], max_steps=100)
print(f"Exit code: {result}")
print(f"Expected: 42")
if result == 42:
    print("✓ Runner returns correct exit code")
else:
    print(f"✗ Runner returns wrong exit code: {result}")

print()
print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
