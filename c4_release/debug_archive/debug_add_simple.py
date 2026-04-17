"""Simpler ADD debugging - just check if OP_ADD flag ever activates."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("Testing ADD neural weights...", file=sys.stderr)
print(f"Test: {code}", file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("✓ Disabled ADD handler", file=sys.stderr)

runner.model.cuda()

# Hook Layer 6 FFN output to see opcode flags after decoding
BD = _SetDim
layer6_output = None

def layer6_hook(module, input, output):
    global layer6_output
    layer6_output = output.detach().cpu()

handle = runner.model.blocks[6].ffn.register_forward_hook(layer6_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=50)

handle.remove()

print(f"\nExit code: {exit_code}", file=sys.stderr)
print(f"Expected: 42", file=sys.stderr)

if layer6_output is not None:
    print(f"\nChecking OP_ADD activation in Layer 6 FFN output...", file=sys.stderr)
    print(f"Sequence length: {layer6_output.shape[1]} tokens", file=sys.stderr)

    # Each VM step is 35 tokens
    num_steps = layer6_output.shape[1] // 35
    print(f"Number of steps: {num_steps}", file=sys.stderr)

    # Check each step for all opcodes
    for step in range(min(num_steps, 20)):
        step_start = step * 35
        step_end = step_start + 35

        # Get the step's tokens
        step_tokens = layer6_output[0, step_start:step_end, :]

        # Check all opcode flags to see what opcodes are present
        opcodes_found = {}
        for opcode_dim in range(BD.OPCODE_BASE, BD.OPCODE_BASE + 40):
            opcode_vals = step_tokens[:, opcode_dim]
            max_val = opcode_vals.max().item()
            if max_val > 0.01:  # Lower threshold
                opcode_name = f"OP_{opcode_dim - BD.OPCODE_BASE}"
                # Map back to actual opcode names
                if opcode_dim == BD.OP_IMM:
                    opcode_name = "IMM"
                elif opcode_dim == BD.OP_PSH:
                    opcode_name = "PSH"
                elif opcode_dim == BD.OP_ADD:
                    opcode_name = "ADD"
                elif opcode_dim == BD.OP_EXIT:
                    opcode_name = "EXIT"
                elif opcode_dim == BD.OP_JMP:
                    opcode_name = "JMP"
                elif opcode_dim == BD.OP_BZ:
                    opcode_name = "BZ"

                opcodes_found[opcode_name] = max_val

        if opcodes_found:
            print(f"\n  Step {step}: Opcodes = {opcodes_found}", file=sys.stderr)

print("\nDone.", file=sys.stderr)

del runner
torch.cuda.empty_cache()
