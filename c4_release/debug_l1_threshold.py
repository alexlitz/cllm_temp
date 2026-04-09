"""Check if Layer 1 threshold heads (L1H1, L1H0) are setting byte index flags correctly."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("DEBUGGING LAYER 1 THRESHOLD HEADS", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Test: {code}", file=sys.stderr)
print("Expected: L1H1/L1H0 should mark byte 0 positions for carry-forward", file=sys.stderr)
print("", file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 1 FFN output
l1_ffn_out = None

def l1_ffn_hook(module, input, output):
    global l1_ffn_out
    if l1_ffn_out is None:
        l1_ffn_out = output.detach().cpu()

h1 = runner.model.blocks[1].ffn.register_forward_hook(l1_ffn_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=50)

h1.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Analyze L1 FFN output
if l1_ffn_out is not None:
    seq_len = l1_ffn_out.shape[1]
    print(f"Sequence length: {seq_len} tokens", file=sys.stderr)
    print("", file=sys.stderr)

    # L1H1 and L1H0 are threshold heads that identify byte positions
    # L1H1[i] + L1H0[i] form a 2-bit code for byte index (0-3)
    # Byte 0: L1H1=1, L1H0=0
    # Byte 1: L1H1=0, L1H0=1
    # Byte 2: L1H1=1, L1H0=1
    # Byte 3: L1H1=0, L1H0=0

    print("Checking L1H1/L1H0 flags (for AX byte index):", file=sys.stderr)
    print("Expected pattern for byte 0: L1H1[1]=1, L1H0[1]=0", file=sys.stderr)
    print("", file=sys.stderr)

    # PC, AX, SP, BP indices
    AX_I = 1

    # Scan for positions where L1H1[AX] and L1H0[AX] are set
    print("Positions with L1H1/L1H0 byte flags:", file=sys.stderr)

    found_byte0 = []
    for pos in range(min(seq_len, 200)):
        l1h1_ax = l1_ffn_out[0, pos, BD.L1H1 + AX_I].item()
        l1h0_ax = l1_ffn_out[0, pos, BD.L1H0 + AX_I].item()

        # Byte 0 pattern: L1H1=1, L1H0=0
        if l1h1_ax > 0.5 and l1h0_ax < 0.5:
            found_byte0.append(pos)
            if len(found_byte0) <= 10:
                print(f"  Position {pos}: byte 0 (L1H1[AX]={l1h1_ax:.2f}, L1H0[AX]={l1h0_ax:.2f})", file=sys.stderr)

    print("", file=sys.stderr)
    print(f"Total byte 0 positions found: {len(found_byte0)}", file=sys.stderr)

    if len(found_byte0) == 0:
        print("", file=sys.stderr)
        print("⚠ WARNING: No byte 0 positions found for AX!", file=sys.stderr)
        print("This means Layer 3 cannot identify which positions to attend to.", file=sys.stderr)
        print("", file=sys.stderr)

        # Check if L1H1/L1H0 are set at all
        l1h1_ax_max = l1_ffn_out[0, :, BD.L1H1 + AX_I].max().item()
        l1h0_ax_max = l1_ffn_out[0, :, BD.L1H0 + AX_I].max().item()

        print(f"Max L1H1[AX] value: {l1h1_ax_max:.3f}", file=sys.stderr)
        print(f"Max L1H0[AX] value: {l1h0_ax_max:.3f}", file=sys.stderr)

        if l1h1_ax_max < 0.1 and l1h0_ax_max < 0.1:
            print("", file=sys.stderr)
            print("❌ ISSUE: L1H1/L1H0[AX] are not being set at all!", file=sys.stderr)
            print("Layer 1 threshold heads are not working correctly.", file=sys.stderr)
    else:
        # Check if byte 0 positions are actually at AX value tokens
        print("", file=sys.stderr)
        print("Checking if byte 0 positions correspond to AX value tokens:", file=sys.stderr)

        # In the 35-token format, AX appears at positions 5-9 (REG_AX + 4 bytes)
        # Byte 0 should be at position 6 (5 + 1)
        # Then every 35 tokens: 41, 76, 111, etc.

        expected_ax_byte0_positions = [6 + i * 35 for i in range(10)]

        print(f"  Expected AX byte 0 positions (in 35-token format): {expected_ax_byte0_positions[:5]}", file=sys.stderr)
        print(f"  Actually found byte 0 at: {found_byte0[:5]}", file=sys.stderr)

        matches = [pos for pos in found_byte0 if pos in expected_ax_byte0_positions]
        if matches:
            print(f"  ✓ Found {len(matches)} matching positions", file=sys.stderr)
        else:
            print(f"  ⚠ WARNING: No matches! Byte 0 flags at wrong positions", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("SUMMARY", file=sys.stderr)
print("=" * 70, file=sys.stderr)

if l1_ffn_out is not None and len(found_byte0) == 0:
    print("❌ Layer 1 threshold heads (L1H1/L1H0) not setting byte 0 flags", file=sys.stderr)
    print("", file=sys.stderr)
    print("Impact:", file=sys.stderr)
    print("  - Layer 3 attention cannot identify AX byte 0 positions", file=sys.stderr)
    print("  - AX_CARRY_LO/HI will not be populated", file=sys.stderr)
    print("  - ADD operation missing second operand", file=sys.stderr)
    print("  - Returns first operand only", file=sys.stderr)
elif l1_ffn_out is not None:
    print(f"✓ Layer 1 setting byte 0 flags at {len(found_byte0)} positions", file=sys.stderr)
    print("  (Need to verify Layer 3 attention is using these correctly)", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
