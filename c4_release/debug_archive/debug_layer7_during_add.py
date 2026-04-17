"""Debug Layer 7 attention during ADD step to see why AX_CARRY isn't accessible."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("LAYER 7 ATTENTION DURING ADD STEP", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Track which step we're in
step_count = [0]
token_count = [0]

# Capture Layer 3 block output and Layer 7 attention
l3_outputs = []
l7_attn_input = []
l7_attn_output = []

def count_tokens(module, input, output):
    token_count[0] += 1
    if len(input) > 0 and input[0].shape[1] > 0:
        last_tok = input[0][0, -1].item()
        if last_tok == Token.STEP_END:
            step_count[0] += 1

def l3_hook(module, input, output):
    # Capture Layer 3 BLOCK output (after attention + FFN + residual)
    l3_outputs.append((step_count[0], token_count[0], output.detach().cpu()))

def l7_attn_pre_hook(module, input):
    # Capture input TO Layer 7 attention
    if len(input) > 0:
        l7_attn_input.append((step_count[0], token_count[0], input[0].detach().cpu()))

def l7_attn_hook(module, input, output):
    # Capture Layer 7 attention output
    l7_attn_output.append((step_count[0], token_count[0], output.detach().cpu()))

h_embed = runner.model.embed.register_forward_hook(count_tokens)
h_l3 = runner.model.blocks[3].register_forward_hook(l3_hook)
h_l7_pre = runner.model.blocks[7].attn.register_forward_pre_hook(l7_attn_pre_hook)
h_l7 = runner.model.blocks[7].attn.register_forward_hook(l7_attn_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=10)

h_embed.remove()
h_l3.remove()
h_l7_pre.remove()
h_l7.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# ADD should execute at step 4 (after ENT, IMM(10), IMM(32))
# Step 4 generates 35 tokens, so tokens ~105-139
ADD_STEP = 4
ADD_TOKEN_START = (ADD_STEP - 1) * 35
ADD_TOKEN_END = ADD_STEP * 35

print(f"ADD step: {ADD_STEP} (tokens ~{ADD_TOKEN_START}-{ADD_TOKEN_END})", file=sys.stderr)
print("", file=sys.stderr)

# Find Layer 3 outputs during ADD step
print("Layer 3 block outputs during ADD step:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

add_step_l3 = [(s, t, out) for s, t, out in l3_outputs if ADD_TOKEN_START <= t < ADD_TOKEN_END]
print(f"Found {len(add_step_l3)} Layer 3 passes during ADD step", file=sys.stderr)

# Check if any have AX_CARRY set
for i, (step, tok, out) in enumerate(add_step_l3[:5]):  # First 5
    seq_len = out.shape[1]
    # Check last position
    ax_carry_lo = out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    max_lo = ax_carry_lo.max().item()
    print(f"  Token {tok} (step {step}, seq_len={seq_len}): AX_CARRY_LO max={max_lo:.3f}", file=sys.stderr)

print("", file=sys.stderr)

# Find Layer 7 attention inputs during ADD step
print("Layer 7 attention inputs during ADD step:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

add_step_l7_in = [(s, t, inp) for s, t, inp in l7_attn_input if ADD_TOKEN_START <= t < ADD_TOKEN_END]
print(f"Found {len(add_step_l7_in)} Layer 7 attention passes during ADD step", file=sys.stderr)

# Check if AX_CARRY is present in the input
for i, (step, tok, inp) in enumerate(add_step_l7_in[:5]):
    seq_len = inp.shape[1]
    # Check ALL positions for AX_CARRY
    ax_carry_all = inp[0, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    max_per_pos = ax_carry_all.max(dim=1)[0]
    positions_with_carry = (max_per_pos > 0.1).sum().item()

    max_overall = ax_carry_all.max().item()
    print(f"  Token {tok} (step {step}, seq_len={seq_len}):", file=sys.stderr)
    print(f"    AX_CARRY_LO max (any position): {max_overall:.3f}", file=sys.stderr)
    print(f"    Positions with AX_CARRY > 0.1: {positions_with_carry}/{seq_len}", file=sys.stderr)

    if positions_with_carry > 0:
        # Show where AX_CARRY is set
        for pos in range(seq_len):
            if max_per_pos[pos] > 0.1:
                print(f"      Position {pos}: max={max_per_pos[pos]:.3f}", file=sys.stderr)
                if positions_with_carry > 10:  # Limit output
                    print(f"      ... ({positions_with_carry} total)", file=sys.stderr)
                    break

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check if Layer 7 input has AX_CARRY available
has_carry_in_l7 = False
for step, tok, inp in add_step_l7_in:
    if inp[0, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max() > 0.1:
        has_carry_in_l7 = True
        break

if has_carry_in_l7:
    print("✓ AX_CARRY IS present in Layer 7 input during ADD", file=sys.stderr)
    print("  Issue: Layer 7 attention not reading it correctly", file=sys.stderr)
    print("  OR: Layer 7 attention weights not configured properly", file=sys.stderr)
else:
    print("❌ AX_CARRY NOT present in Layer 7 input during ADD", file=sys.stderr)
    print("  Issue: Values lost between Layer 3 and Layer 7", file=sys.stderr)
    print("  Need to check intermediate layers (4, 5, 6)", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
