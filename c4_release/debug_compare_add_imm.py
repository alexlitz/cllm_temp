"""Compare ADD vs IMM execution with detailed activation analysis."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

def analyze_execution(code, desc, disable_handler=None):
    """Run program and capture key layer activations."""
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"{desc}", file=sys.stderr)
    print(f"Code: {code}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    bytecode, data = compile_c(code)

    # Disassemble bytecode to understand what should execute
    print(f"\nBytecode ({len(bytecode)} bytes):", file=sys.stderr)
    # Simple hex dump
    for i in range(0, min(len(bytecode), 64), 4):
        chunk = bytecode[i:i+4]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        print(f"  {i:3d}: {hex_str}", file=sys.stderr)

    runner = AutoregressiveVMRunner()
    if disable_handler and disable_handler in runner._func_call_handlers:
        del runner._func_call_handlers[disable_handler]
        print(f"\n✓ Disabled {disable_handler} handler", file=sys.stderr)

    runner.model.cuda()

    # Capture activations from key layers
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            # Only capture first time (initial forward pass builds full context)
            if name not in activations:
                activations[name] = output.detach().cpu()
        return hook

    # Hook key layers: L3 (AX carry), L7 (operand gather), L8 (ALU)
    handles = []
    handles.append(runner.model.blocks[3].attn.register_forward_hook(make_hook('L3_attn')))
    handles.append(runner.model.blocks[7].attn.register_forward_hook(make_hook('L7_attn')))
    handles.append(runner.model.blocks[8].ffn.register_forward_hook(make_hook('L8_ffn')))

    print("\nRunning VM...", file=sys.stderr)
    _, exit_code = runner.run(bytecode, max_steps=50)

    # Remove hooks
    for h in handles:
        h.remove()

    print(f"Exit code: {exit_code}", file=sys.stderr)

    # Analyze activations
    if 'L8_ffn' in activations:
        h8 = activations['L8_ffn']
        seq_len = h8.shape[1]
        num_steps = seq_len // 35

        print(f"\nSequence length: {seq_len} tokens ({num_steps} steps)", file=sys.stderr)

        # Check each step for interesting opcode flags and ALU activity
        print(f"\nStep Analysis:", file=sys.stderr)

        for step in range(min(num_steps, 10)):
            step_start = step * 35
            step_end = step_start + 35

            # Get Layer 8 FFN output for this step
            step_h8 = h8[0, step_start:step_end, :]

            # Check opcode flags
            opcodes_active = []
            opcode_map = {
                BD.OP_IMM: 'IMM',
                BD.OP_PSH: 'PSH',
                BD.OP_ADD: 'ADD',
                BD.OP_SUB: 'SUB',
                BD.OP_JMP: 'JMP',
                BD.OP_EXIT: 'EXIT',
            }

            for op_dim, op_name in opcode_map.items():
                if step_h8[:, op_dim].max().item() > 0.01:
                    opcodes_active.append(op_name)

            # Check ALU dimensions
            alu_lo_active = step_h8[:, BD.ALU_LO:BD.ALU_LO+16].max().item()
            ax_carry_lo_active = step_h8[:, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()
            output_lo_active = step_h8[:, BD.OUTPUT_LO:BD.OUTPUT_LO+16].max().item()

            # Check MARK_AX
            mark_ax_active = step_h8[:, BD.MARK_AX].max().item()

            if opcodes_active or alu_lo_active > 0.01 or output_lo_active > 0.01:
                print(f"\n  Step {step}:", file=sys.stderr)
                if opcodes_active:
                    print(f"    Opcodes: {opcodes_active}", file=sys.stderr)
                print(f"    MARK_AX: {mark_ax_active:.3f}", file=sys.stderr)
                print(f"    ALU_LO: {alu_lo_active:.3f}", file=sys.stderr)
                print(f"    AX_CARRY_LO: {ax_carry_lo_active:.3f}", file=sys.stderr)
                print(f"    OUTPUT_LO: {output_lo_active:.3f}", file=sys.stderr)

                # If this looks like an ALU operation, analyze in detail
                if 'ADD' in opcodes_active or mark_ax_active > 0.5:
                    print(f"\n    ** Detailed analysis for step {step} **", file=sys.stderr)

                    # Check Layer 7 attention (operand gathering)
                    if 'L7_attn' in activations:
                        h7 = activations['L7_attn'][0, step_start:step_end, :]
                        alu_lo_vals = []
                        for i in range(16):
                            val = h7[:, BD.ALU_LO + i].max().item()
                            if val > 0.1:
                                alu_lo_vals.append((i, val))
                        if alu_lo_vals:
                            print(f"      L7 ALU_LO: {alu_lo_vals[:3]}", file=sys.stderr)
                        else:
                            print(f"      L7 ALU_LO: NONE (⚠ operand not gathered!)", file=sys.stderr)

                    # Check Layer 3 attention (AX carry)
                    if 'L3_attn' in activations:
                        h3 = activations['L3_attn'][0, step_start:step_end, :]
                        ax_carry_vals = []
                        for i in range(16):
                            val = h3[:, BD.AX_CARRY_LO + i].max().item()
                            if val > 0.1:
                                ax_carry_vals.append((i, val))
                        if ax_carry_vals:
                            print(f"      L3 AX_CARRY_LO: {ax_carry_vals[:3]}", file=sys.stderr)
                        else:
                            print(f"      L3 AX_CARRY_LO: NONE (⚠ AX not carried!)", file=sys.stderr)

                    # Check Layer 8 FFN output
                    output_vals = []
                    for i in range(16):
                        val = step_h8[:, BD.OUTPUT_LO + i].max().item()
                        if val > 0.1:
                            output_vals.append((i, val))
                    if output_vals:
                        print(f"      L8 OUTPUT_LO: {output_vals[:3]}", file=sys.stderr)
                    else:
                        print(f"      L8 OUTPUT_LO: NONE (⚠ no output!)", file=sys.stderr)

    del runner
    torch.cuda.empty_cache()

    return exit_code

# Compare ADD vs IMM
print("\n" + "=" * 70, file=sys.stderr)
print("COMPARING ADD VS IMM EXECUTION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

result_imm = analyze_execution("int main() { return 42; }", "IMM TEST (baseline)")
print("\n\n", file=sys.stderr)
result_add = analyze_execution("int main() { return 10 + 32; }", "ADD TEST (handler disabled)", disable_handler=Opcode.ADD)

print("\n" + "=" * 70, file=sys.stderr)
print("RESULTS", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"IMM result: {result_imm} (expected 42) - {'✓ PASS' if result_imm == 42 else '✗ FAIL'}", file=sys.stderr)
print(f"ADD result: {result_add} (expected 42) - {'✓ PASS' if result_add == 42 else '✗ FAIL'}", file=sys.stderr)

if result_add == 10:
    print("\nADD returns first operand only (10) - confirms data routing issue", file=sys.stderr)
print("=" * 70, file=sys.stderr)
