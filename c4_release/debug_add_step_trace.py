"""Trace Layer 1 → Layer 3 during the ADD operation specifically."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("TRACING L1 → L3 DURING ADD OPERATION", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 1 and Layer 3 outputs for ALL forward passes
l1_outputs = []
l3_outputs = []
step_count = [0]  # Track which step we're in

def count_steps(module, input, output):
    # Count STEP_END tokens
    if len(input) > 0 and input[0].shape[1] > 0:
        last_tok = input[0][0, -1].item()
        if last_tok == Token.STEP_END:
            step_count[0] += 1

def l1_hook(module, input, output):
    l1_outputs.append((step_count[0], output.detach().cpu()))

def l3_hook(module, input, output):
    l3_outputs.append((step_count[0], output.detach().cpu()))

# Hook embed to count steps
h_embed = runner.model.embed.register_forward_hook(count_steps)
# Hook Layer 1 FFN output (after residual)
h1 = runner.model.blocks[1].register_forward_hook(l1_hook)
# Hook Layer 3 attention output
h3 = runner.model.blocks[3].attn.register_forward_hook(l3_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=10)

h_embed.remove()
h1.remove()
h3.remove()

print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Steps completed: {step_count[0]}", file=sys.stderr)
print(f"Layer 1 passes: {len(l1_outputs)}", file=sys.stderr)
print(f"Layer 3 passes: {len(l3_outputs)}\n", file=sys.stderr)

# Find the ADD operation
# ADD is opcode 0x0D (13), at bytecode[3] based on compiler output
# Step sequence: ENT, IMM(10), IMM(32), ADD, LEV, EXIT
# So ADD should be step 4 (counting from 1)

ADD_STEP = 4

print(f"Analyzing step {ADD_STEP} (ADD operation):", file=sys.stderr)

# Find the Layer 1 output from that step
# Each step generates 35 tokens, so we need passes around step_num == ADD_STEP
for step_num, l1_out in l1_outputs:
    if step_num == ADD_STEP - 1:  # Just before ADD step completes
        seq_len = l1_out.shape[1]
        print(f"\nLayer 1 output (step {step_num}, seq_len={seq_len}):", file=sys.stderr)

        # Check L1H1/L1H0 for AX
        AX_I = 1
        l1h1_ax = l1_out[0, :, BD.L1H1 + AX_I]
        l1h0_ax = l1_out[0, :, BD.L1H0 + AX_I]

        # Find byte 0 positions (L1H1=1, L1H0=0)
        byte0_positions = []
        for pos in range(seq_len):
            if l1h1_ax[pos] > 0.5 and l1h0_ax[pos] < 0.5:
                byte0_positions.append(pos)

        print(f"  Byte 0 positions (L1H1=1, L1H0=0): {len(byte0_positions)} found", file=sys.stderr)
        if len(byte0_positions) > 0:
            print(f"    Last 5: {byte0_positions[-5:]}", file=sys.stderr)
        else:
            print("    ❌ No byte 0 positions! L1 threshold heads not working for ADD", file=sys.stderr)

# Find the Layer 3 output from that step
for step_num, l3_out in l3_outputs:
    if step_num == ADD_STEP - 1:
        seq_len = l3_out.shape[1]
        print(f"\nLayer 3 attention output (step {step_num}, seq_len={seq_len}):", file=sys.stderr)

        # Check AX_CARRY_LO/HI
        ax_carry_lo = l3_out[0, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
        ax_carry_hi = l3_out[0, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

        ax_carry_lo_max = ax_carry_lo.max().item()
        ax_carry_hi_max = ax_carry_hi.max().item()

        print(f"  Max AX_CARRY_LO: {ax_carry_lo_max:.3f}", file=sys.stderr)
        print(f"  Max AX_CARRY_HI: {ax_carry_hi_max:.3f}", file=sys.stderr)

        if ax_carry_lo_max < 0.1:
            print("\n  ❌ AX_CARRY_LO not populated!", file=sys.stderr)
            print("     Layer 3 is not using L1H1/L1H0 to copy AX", file=sys.stderr)

            # Check if L1H1/L1H0 are present in the input to Layer 3
            # Layer 3 input = Layer 1 output (after going through Layer 2)
        else:
            print(f"\n  ✓ AX_CARRY_LO populated correctly!", file=sys.stderr)

            # Show where it's active
            for pos in range(max(0, seq_len - 10), seq_len):
                if ax_carry_lo[:, :].sum() > 0.1:
                    max_val = ax_carry_lo[pos, :].max().item()
                    if max_val > 0.1:
                        print(f"    Position {pos}: max={max_val:.2f}", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
