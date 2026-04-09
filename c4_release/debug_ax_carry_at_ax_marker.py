"""Check if AX_CARRY is present AT THE AX MARKER during ADD step."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING AX_CARRY AT AX MARKER POSITION DURING ADD", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 8 FFN input (= Layer 7 block output after residual)
l7_block_outputs = []
token_count = [0]

def count_tokens(module, input, output):
    token_count[0] += 1

def l7_block_hook(module, input, output):
    l7_block_outputs.append((token_count[0], output.detach().cpu()))

h_embed = runner.model.embed.register_forward_hook(count_tokens)
h_l7 = runner.model.blocks[7].register_forward_hook(l7_block_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=10)

h_embed.remove()
h_l7.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# ADD step is around token 105-139 (step 4)
# Within each step, AX marker is at position 5 (in 35-token format)

print("Checking Layer 7 block output (input to Layer 8 FFN):", file=sys.stderr)
print("-" * 70, file=sys.stderr)

# Find outputs during ADD step
ADD_STEP_START = 105
ADD_STEP_END = 140

add_step_outputs = [(tok, out) for tok, out in l7_block_outputs
                    if ADD_STEP_START <= tok < ADD_STEP_END]

print(f"Found {len(add_step_outputs)} Layer 7 outputs during ADD step\n", file=sys.stderr)

# Check a few key positions
for i, (tok, out) in enumerate(add_step_outputs[:5]):
    seq_len = out.shape[1]

    # Find AX marker position in this sequence
    # AX marker should be at offset 5 within each 35-token step
    # But during generation, we're building up the sequence token by token

    # Check LAST position (the one just generated) - is it the AX marker?
    # Also check for MARK_AX flag

    last_pos = seq_len - 1
    mark_ax = out[0, last_pos, BD.MARK_AX].item()

    # Check if this position has AX_CARRY
    ax_carry_lo = out[0, last_pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = out[0, last_pos, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    max_carry_lo = ax_carry_lo.max().item()
    max_carry_hi = ax_carry_hi.max().item()

    print(f"Token {tok} (seq_len={seq_len}, pos={last_pos}):", file=sys.stderr)
    print(f"  MARK_AX: {mark_ax:.2f}", file=sys.stderr)
    print(f"  AX_CARRY_LO max: {max_carry_lo:.3f}", file=sys.stderr)
    print(f"  AX_CARRY_HI max: {max_carry_hi:.3f}", file=sys.stderr)

    if mark_ax > 0.5:
        print(f"  → This IS an AX marker position", file=sys.stderr)
        if max_carry_lo < 0.1:
            print(f"  ❌ But AX_CARRY is NOT present!", file=sys.stderr)
        else:
            print(f"  ✓ AX_CARRY is present", file=sys.stderr)
    print("", file=sys.stderr)

print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

# Check if ANY position during ADD has both MARK_AX and AX_CARRY
found_good_pos = False
for tok, out in add_step_outputs:
    seq_len = out.shape[1]
    for pos in range(seq_len):
        mark_ax = out[0, pos, BD.MARK_AX].item()
        ax_carry_max = out[0, pos, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()

        if mark_ax > 0.5 and ax_carry_max > 0.1:
            print(f"✓ Found AX marker with AX_CARRY at token {tok}, position {pos}", file=sys.stderr)
            print(f"  MARK_AX={mark_ax:.2f}, AX_CARRY max={ax_carry_max:.3f}", file=sys.stderr)
            found_good_pos = True
            break
    if found_good_pos:
        break

if not found_good_pos:
    print("❌ NO position found with both MARK_AX and AX_CARRY during ADD!", file=sys.stderr)
    print("   Layer 8 FFN cannot compute ADD without AX_CARRY at AX marker", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
