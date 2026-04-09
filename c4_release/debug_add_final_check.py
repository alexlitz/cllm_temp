"""Check Layer 3 AX_CARRY during ADD - final pass only."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("CHECKING LAYER 3 AX_CARRY DURING ADD", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Only capture the LAST forward pass of each layer
l1_last = [None]
l3_last = [None]

def l1_hook(module, input, output):
    l1_last[0] = output.detach().cpu()

def l3_hook(module, input, output):
    l3_last[0] = output.detach().cpu()

h1 = runner.model.blocks[1].register_forward_hook(l1_hook)
h3 = runner.model.blocks[3].attn.register_forward_hook(l3_hook)

print("Running VM (3 steps to reach ADD)...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=5)

h1.remove()
h3.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Analyze final Layer 1 output
if l1_last[0] is not None:
    l1_out = l1_last[0]
    seq_len = l1_out.shape[1]
    print(f"Final Layer 1 output (seq_len={seq_len}):", file=sys.stderr)

    AX_I = 1
    l1h1_ax = l1_out[0, :, BD.L1H1 + AX_I]
    l1h0_ax = l1_out[0, :, BD.L1H0 + AX_I]

    # Find byte 0 positions
    byte0_pos = []
    for pos in range(seq_len):
        if l1h1_ax[pos] > 0.5 and l1h0_ax[pos] < 0.5:
            byte0_pos.append(pos)

    print(f"  Byte 0 positions (L1H1=1, L1H0=0): {len(byte0_pos)}", file=sys.stderr)
    if len(byte0_pos) > 0:
        print(f"    Last 10: {byte0_pos[-10:]}", file=sys.stderr)

# Analyze final Layer 3 output
if l3_last[0] is not None:
    l3_out = l3_last[0]
    seq_len = l3_out.shape[1]
    print(f"\nFinal Layer 3 attention output (seq_len={seq_len}):", file=sys.stderr)

    ax_carry_lo = l3_out[0, :, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l3_out[0, :, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    max_lo = ax_carry_lo.max().item()
    max_hi = ax_carry_hi.max().item()

    print(f"  Max AX_CARRY_LO: {max_lo:.3f}", file=sys.stderr)
    print(f"  Max AX_CARRY_HI: {max_hi:.3f}", file=sys.stderr)

    if max_lo < 0.1:
        print("\n❌ AX_CARRY_LO NOT populated by Layer 3!", file=sys.stderr)
        print("   This confirms the carry-forward mechanism is broken.", file=sys.stderr)

        # Check if the attention configuration is correct
        attn3 = runner.model.blocks[3].attn
        head1_base = 64  # head 1

        # Check key weights for head 1
        wk_l1h1 = attn3.W_k[head1_base, BD.L1H1 + AX_I].item()
        wk_l1h0 = attn3.W_k[head1_base, BD.L1H0 + AX_I].item()

        print(f"\nLayer 3 Head 1 key weights:", file=sys.stderr)
        print(f"  W_k[{head1_base}, L1H1+AX={BD.L1H1 + AX_I}] = {wk_l1h1:.2f}", file=sys.stderr)
        print(f"  W_k[{head1_base}, L1H0+AX={BD.L1H0 + AX_I}] = {wk_l1h0:.2f}", file=sys.stderr)

        if abs(wk_l1h1) < 0.1:
            print("\n❌ Layer 3 head 1 W_k[L1H1] not set!", file=sys.stderr)
            print("   Weight configuration is broken.", file=sys.stderr)
        else:
            print("\n✓ Weights configured correctly", file=sys.stderr)
            print("  Issue must be in attention computation or residual stream.", file=sys.stderr)
    else:
        print("\n✓ AX_CARRY populated correctly!", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
