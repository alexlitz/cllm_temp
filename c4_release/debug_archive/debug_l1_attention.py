"""Check Layer 1 ATTENTION output (not FFN) for threshold heads."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("DEBUGGING LAYER 1 ATTENTION (THRESHOLD HEADS)", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print(f"Test: {code}", file=sys.stderr)
print("Expected: L1 attention should set L1H1[AX]=1, L1H0[AX]=0 at byte 0", file=sys.stderr)
print("", file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 1 ATTENTION output (not FFN)
l1_attn_out = None

def l1_attn_hook(module, input, output):
    global l1_attn_out
    if l1_attn_out is None:
        l1_attn_out = output.detach().cpu()

h1 = runner.model.blocks[1].attn.register_forward_hook(l1_attn_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=50)

h1.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Analyze L1 attention output
if l1_attn_out is not None:
    seq_len = l1_attn_out.shape[1]
    print(f"Sequence length: {seq_len} tokens", file=sys.stderr)
    print("", file=sys.stderr)

    # AX marker index in MARKS array
    AX_I = 1

    print("Threshold heads configuration:", file=sys.stderr)
    print("  L1H0 (dim {}) = threshold 0.5 (very close to marker)".format(BD.L1H0), file=sys.stderr)
    print("  L1H1 (dim {}) = threshold 1.5 (byte 0 position)".format(BD.L1H1), file=sys.stderr)
    print("  L1H2 (dim {}) = threshold 2.5 (byte 1 position)".format(BD.L1H2), file=sys.stderr)
    print("", file=sys.stderr)

    print("Scanning for L1H1/L1H0 flags at AX positions:", file=sys.stderr)

    # Scan for positions where L1H1[AX] and L1H0[AX] are set
    found_byte0 = []
    for pos in range(min(seq_len, 200)):
        l1h1_ax = l1_attn_out[0, pos, BD.L1H1 + AX_I].item()
        l1h0_ax = l1_attn_out[0, pos, BD.L1H0 + AX_I].item()

        # Byte 0 pattern: L1H1=1, L1H0=0
        if l1h1_ax > 0.5 and l1h0_ax < 0.5:
            found_byte0.append(pos)
            if len(found_byte0) <= 10:
                print(f"  Position {pos}: L1H1[AX]={l1h1_ax:.2f}, L1H0[AX]={l1h0_ax:.2f} ← byte 0", file=sys.stderr)

    print("", file=sys.stderr)
    print(f"Total byte 0 positions found: {len(found_byte0)}", file=sys.stderr)

    if len(found_byte0) == 0:
        print("", file=sys.stderr)
        print("⚠ WARNING: No byte 0 positions found!", file=sys.stderr)

        # Check max values
        l1h1_ax_max = l1_attn_out[0, :, BD.L1H1 + AX_I].max().item()
        l1h0_ax_max = l1_attn_out[0, :, BD.L1H0 + AX_I].max().item()

        print(f"Max L1H1[AX]: {l1h1_ax_max:.3f}", file=sys.stderr)
        print(f"Max L1H0[AX]: {l1h0_ax_max:.3f}", file=sys.stderr)

        if l1h1_ax_max < 0.1:
            print("", file=sys.stderr)
            print("❌ L1H1[AX] not being set by Layer 1 attention!", file=sys.stderr)
            print("This means threshold head mechanism is broken.", file=sys.stderr)
    else:
        print(f"✓ Found {len(found_byte0)} byte 0 positions", file=sys.stderr)

        # Now check if Layer 3 is using these correctly
        print("", file=sys.stderr)
        print("Next: Check if Layer 3 attention sees these flags and populates AX_CARRY", file=sys.stderr)

print("", file=sys.stderr)
print("=" * 70, file=sys.stderr)
print("DIAGNOSIS", file=sys.stderr)
print("=" * 70, file=sys.stderr)

if l1_attn_out is not None:
    if len(found_byte0) == 0:
        print("❌ Layer 1 attention NOT setting L1H1/L1H0 threshold heads", file=sys.stderr)
        print("   Threshold mechanism is broken at Layer 1 attention level", file=sys.stderr)
    else:
        print("✓ Layer 1 attention IS setting L1H1/L1H0 correctly", file=sys.stderr)
        print("  Issue must be downstream (Layer 3 not using them?)", file=sys.stderr)

print("=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
