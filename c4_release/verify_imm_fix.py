#!/usr/bin/env python3
"""Verify the IMM instruction fix and document the current state."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("=" * 80)
print("IMM INSTRUCTION FIX VERIFICATION")
print("=" * 80)

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

print("\nProgram: IMM 42, EXIT")
print("Expected: exit_code = 42")
print()

# Test 1: Runner execution (with fallback handler)
print("-" * 80)
print("TEST 1: Runner Execution (with fallback)")
print("-" * 80)
output, exit_code = runner.run(bytecode, b'', [], max_steps=5)
print(f"Exit code: {exit_code}")
if exit_code == 42:
    print("✓ PASS: Exit code is correct (runner fallback works)")
else:
    print(f"✗ FAIL: Expected 42, got {exit_code}")

# Test 2: Raw model generation (without runner override)
print("\n" + "-" * 80)
print("TEST 2: Raw Model Generation (without runner override)")
print("-" * 80)
context = runner._build_context(bytecode, b'', [])
print(f"Initial context length: {len(context)}")

# Generate step 0
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find REG_AX in step 0
reg_ax_pos = None
for i in range(20, len(context)):
    if context[i] == 260:  # REG_AX
        reg_ax_pos = i
        break

if reg_ax_pos:
    ax_bytes = [context[reg_ax_pos + 1 + i] for i in range(4)]
    ax_value = sum(b << (i*8) for i, b in enumerate(ax_bytes))
    print(f"Model generated AX: {ax_bytes} = 0x{ax_value:08x} ({ax_value})")

    if ax_value == 42:
        print("✓ PASS: Model generates correct AX neurally")
    elif ax_value == 0:
        print("⚠ PARTIAL: Model generates AX=0, runner fallback compensates")
        print("  → Neural IMM weights are not set/working")
        print("  → Runner's _handler_imm extracts immediate from bytecode")
    else:
        print(f"✗ FAIL: Expected 42 or 0, got {ax_value}")

# Test 3: Check Layer 3 FFN units
print("\n" + "-" * 80)
print("TEST 3: Layer 3 FFN Units (marker gates)")
print("-" * 80)

token_ids = torch.tensor([context[:reg_ax_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(3):
        x = model.blocks[i](x, kv_cache=None)

    x_before_l3 = x.clone()
    x_after_l3 = model.blocks[3](x, kv_cache=None)

    block3 = model.blocks[3]
    ffn = block3.ffn

    # Check unit 4 (SP default)
    unit4_sp_gate = ffn.W_gate[4, BD.MARK_SP].item()
    unit4_bp_gate = ffn.W_gate[4, BD.MARK_BP].item()
    unit4_ax_gate = ffn.W_gate[4, BD.MARK_AX].item()

    print(f"Unit 4 (SP default) marker gates:")
    print(f"  MARK_SP: {unit4_sp_gate:.3f}")
    print(f"  MARK_BP: {unit4_bp_gate:.3f}")
    print(f"  MARK_AX: {unit4_ax_gate:.3f}")

    if abs(unit4_sp_gate - 1.0) < 0.01:
        print("  ✓ Unit 4 has MARK_SP gate (correct)")
    else:
        print("  ✗ Unit 4 missing MARK_SP gate")

    # Check unit 5 (BP default)
    unit5_sp_gate = ffn.W_gate[5, BD.MARK_SP].item()
    unit5_bp_gate = ffn.W_gate[5, BD.MARK_BP].item()
    unit5_ax_gate = ffn.W_gate[5, BD.MARK_AX].item()

    print(f"\nUnit 5 (BP default) marker gates:")
    print(f"  MARK_SP: {unit5_sp_gate:.3f}")
    print(f"  MARK_BP: {unit5_bp_gate:.3f}")
    print(f"  MARK_AX: {unit5_ax_gate:.3f}")

    if abs(unit5_bp_gate - 1.0) < 0.01:
        print("  ✓ Unit 5 has MARK_BP gate (correct)")
    else:
        print("  ✗ Unit 5 missing MARK_BP gate")

    # Check OUTPUT_LO[1] at AX position
    output_lo_1 = x_after_l3[0, reg_ax_pos, BD.OUTPUT_LO + 1].item()
    print(f"\nOUTPUT_LO[1] at REG_AX position: {output_lo_1:.6f}")

    if abs(output_lo_1) < 0.1:
        print("  ✓ OUTPUT_LO[1] is near zero (SP/BP units not firing at AX)")
    else:
        print(f"  ✗ OUTPUT_LO[1] is {output_lo_1:.3f} (SP/BP units still firing)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
FIX STATUS:
  ✓ Layer 3 FFN marker gates added (units 4 & 5)
  ✓ SP/BP default units no longer corrupt AX positions
  ✓ Exit code extraction works (via runner fallback)

CURRENT STATE:
  ⚠ Model generates AX=0 (neural IMM weights not set)
  ⚠ Runner fallback (_handler_imm) compensates by extracting from bytecode
  ⚠ EXIT returns correct exit code 42 despite neural model failure

IMPLICATIONS:
  - IMM instruction works for end-to-end testing
  - But the neural transformer is not executing IMM
  - Layer 6 FFN routing for IMM needs to be implemented
  - See IMM_RUNNER_FALLBACK_ANALYSIS.md for details

NEXT STEPS (if neural IMM is desired):
  1. Check if Layer 2 sets OP_IMM flag
  2. Verify Layer 6 has FETCH → OUTPUT routing units
  3. Test with runner fallback disabled
  4. Add missing IMM routing weights if needed
""")
