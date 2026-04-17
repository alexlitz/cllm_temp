"""Debug ADD operation to find why neural weights fail.

Investigation Plan:
1. Trace STACK0 → ALU_LO (Layer 7 attention)
2. Trace AX → AX_CARRY_LO (Layer 3 attention)
3. Check Layer 8 FFN ADD unit activation
4. Check if Layer 10 overrides ALU output with passthrough

Test Case: 10 + 32 = 42
Expected: exit_code = 42
Actual (without handler): exit_code = 10 (first operand only)
"""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
from neural_vm.embedding import Opcode as OpcodeEmbed
import torch
import sys

# Test program
code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("DEBUGGING ADD NEURAL WEIGHTS", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Test: {code}", file=sys.stderr)
print(f"Expected exit_code: 42", file=sys.stderr)
print("", file=sys.stderr)

# Skip bytecode disassembly for now
print(f"Bytecode length: {len(bytecode)} bytes", file=sys.stderr)
print("", file=sys.stderr)

# Create runner and disable ADD handler
runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("✓ Disabled ADD handler (testing pure neural)", file=sys.stderr)

runner.model.cuda()

# Hook to capture activations
BD = _SetDim
activations = {}

def capture_hook(name):
    """Create hook to capture layer output."""
    def hook(module, input, output):
        # output is (batch, seq, d_model)
        activations[name] = output.detach().cpu()
    return hook

# Register hooks on key layers only (to save memory)
# We care about: L0 (baseline), L3 (AX carry), L7 (operand gather), L8 (ALU), L10 (passthrough check)
key_layers = {0, 3, 7, 8, 10, 15}
handles = []
for i, block in enumerate(runner.model.blocks):
    if i in key_layers:
        # Hook the FFN output
        handle = block.ffn.register_forward_hook(capture_hook(f"layer_{i}_ffn"))
        handles.append(handle)
        # Hook the attention output for layers that do gathering
        if i in {3, 7}:
            handle_attn = block.attn.register_forward_hook(capture_hook(f"layer_{i}_attn"))
            handles.append(handle_attn)

print("Running VM with activation capture...", file=sys.stderr)
try:
    _, exit_code = runner.run(bytecode, max_steps=50)
    print(f"Exit code: {exit_code}", file=sys.stderr)
    print("", file=sys.stderr)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    exit_code = None

# Remove hooks
for handle in handles:
    handle.remove()

if exit_code is None:
    print("VM run failed, skipping analysis", file=sys.stderr)
    sys.exit(1)

# Analyze activations at the ADD instruction step
# Find the step where ADD executes
print("=" * 70, file=sys.stderr)
print("ACTIVATION ANALYSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# The ADD instruction should be around step 5-6 based on bytecode
# Let's examine the activations to find it
print("\nSearching for ADD step in execution...", file=sys.stderr)

# Get sequence length from activations
if 'layer_0_ffn' in activations:
    seq_len = activations['layer_0_ffn'].shape[1]
    print(f"Total steps executed: {seq_len}", file=sys.stderr)
else:
    print("ERROR: No activations captured!", file=sys.stderr)
    sys.exit(1)

# Look for the ADD opcode marker in context
# ADD opcode is 25, which should appear in opcode dimensions
for step in range(min(seq_len, 20)):
    print(f"\n--- Step {step} ---", file=sys.stderr)

    # Get activations at this step (use layer 0 FFN output)
    if 'layer_0_ffn' in activations:
            h = activations['layer_0_ffn'][0, step, :]  # (d_model,)

            # Check if OP_ADD is active (simpler than decoding opcode value)
            op_add_active = h[BD.OP_ADD].item()

            # Only analyze steps where OP_ADD might be active
            if op_add_active > 0.1:
                print(f"\n{'=' * 70}", file=sys.stderr)
                print(f"FOUND ADD INSTRUCTION AT STEP {step}", file=sys.stderr)
                print(f"{'=' * 70}", file=sys.stderr)

                # Analyze Layer 3 (AX carry)
                if 'layer_3_attn' in activations:
                    h3 = activations['layer_3_attn'][0, step, :]
                    print(f"\n[Layer 3] AX → AX_CARRY_LO relay:", file=sys.stderr)
                    ax_carry_vals = []
                    for i in range(16):
                        val = h3[BD.AX_CARRY_LO + i].item()
                        if val > 0.1:
                            ax_carry_vals.append((i, val))
                    if ax_carry_vals:
                        print(f"  AX_CARRY_LO active values: {ax_carry_vals[:5]}", file=sys.stderr)
                    else:
                        print(f"  ⚠ WARNING: No AX_CARRY_LO values active!", file=sys.stderr)

                # Analyze Layer 7 (operand gathering)
                if 'layer_7_attn' in activations:
                    h7 = activations['layer_7_attn'][0, step, :]
                    print(f"\n[Layer 7] STACK0 → ALU_LO gathering:", file=sys.stderr)
                    alu_lo_vals = []
                    for i in range(16):
                        val = h7[BD.ALU_LO + i].item()
                        if val > 0.1:
                            alu_lo_vals.append((i, val))
                    if alu_lo_vals:
                        print(f"  ALU_LO active values: {alu_lo_vals[:5]}", file=sys.stderr)
                    else:
                        print(f"  ⚠ WARNING: No ALU_LO values active!", file=sys.stderr)

                # Analyze Layer 8 (ADD computation)
                if 'layer_8_ffn' in activations:
                    h8 = activations['layer_8_ffn'][0, step, :]
                    print(f"\n[Layer 8] ADD FFN computation:", file=sys.stderr)

                    # Check if MARK_AX is active (needed for ADD)
                    mark_ax = h8[BD.MARK_AX].item()
                    print(f"  MARK_AX: {mark_ax:.3f}", file=sys.stderr)

                    # Check OUTPUT_LO (where result should appear)
                    output_lo_vals = []
                    for i in range(16):
                        val = h8[BD.OUTPUT_LO + i].item()
                        if val > 0.1:
                            output_lo_vals.append((i, val))
                    if output_lo_vals:
                        print(f"  OUTPUT_LO active values: {output_lo_vals[:5]}", file=sys.stderr)
                        # Check if result is 42 & 0xF = 10
                        if any(i == 10 for i, _ in output_lo_vals):
                            print(f"  ✓ OUTPUT_LO contains expected result nibble (10)", file=sys.stderr)
                    else:
                        print(f"  ⚠ WARNING: No OUTPUT_LO values active!", file=sys.stderr)

                # Analyze Layer 10 (passthrough check)
                if 'layer_10_ffn' in activations:
                    h10 = activations['layer_10_ffn'][0, step, :]
                    print(f"\n[Layer 10] Passthrough vs ALU output:", file=sys.stderr)

                    # Check if OUTPUT_LO changed from Layer 8
                    output_lo_vals_l10 = []
                    for i in range(16):
                        val = h10[BD.OUTPUT_LO + i].item()
                        if val > 0.1:
                            output_lo_vals_l10.append((i, val))

                    if output_lo_vals_l10:
                        print(f"  OUTPUT_LO (L10): {output_lo_vals_l10[:5]}", file=sys.stderr)

                        # Compare with Layer 8
                        if output_lo_vals != output_lo_vals_l10:
                            print(f"  ⚠ WARNING: OUTPUT changed from Layer 8 to Layer 10!", file=sys.stderr)
                            print(f"    L8:  {output_lo_vals[:3]}", file=sys.stderr)
                            print(f"    L10: {output_lo_vals_l10[:3]}", file=sys.stderr)
                        else:
                            print(f"  ✓ OUTPUT preserved from Layer 8", file=sys.stderr)

                # Analyze final layer (Layer 15)
                if 'layer_15_ffn' in activations:
                    h15 = activations['layer_15_ffn'][0, step, :]
                    print(f"\n[Layer 15] Final output:", file=sys.stderr)

                    output_lo_vals_l15 = []
                    for i in range(16):
                        val = h15[BD.OUTPUT_LO + i].item()
                        if val > 0.1:
                            output_lo_vals_l15.append((i, val))

                    if output_lo_vals_l15:
                        print(f"  OUTPUT_LO (L15): {output_lo_vals_l15[:5]}", file=sys.stderr)

                # Next step analysis (to see what AX becomes)
                if step + 1 < seq_len and 'layer_0_ffn' in activations:
                    h_next = activations['layer_0_ffn'][0, step + 1, :]
                    ax_lo_next = 0
                    for i in range(16):
                        if h_next[BD.AX_LO + i].item() > 0.5:
                            ax_lo_next = i
                            break
                    print(f"\n[Step {step + 1}] AX after ADD:", file=sys.stderr)
                    print(f"  AX_LO: {ax_lo_next} (expected: 10 for 42 & 0xF)", file=sys.stderr)

                    if ax_lo_next == 10:
                        print(f"  ✓ AX_LO has correct value!", file=sys.stderr)
                    else:
                        print(f"  ✗ AX_LO is WRONG (expected 10, got {ax_lo_next})", file=sys.stderr)

                break  # Found ADD, stop searching

print("\n" + "=" * 70, file=sys.stderr)
print("SUMMARY", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Result: {exit_code}", file=sys.stderr)
print(f"Expected: 42", file=sys.stderr)
if exit_code == 42:
    print("✓ TEST PASSED (unexpectedly!)", file=sys.stderr)
elif exit_code == 10:
    print("✗ TEST FAILED - returned first operand only", file=sys.stderr)
else:
    print(f"✗ TEST FAILED - unexpected result", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Cleanup
del runner
torch.cuda.empty_cache()
