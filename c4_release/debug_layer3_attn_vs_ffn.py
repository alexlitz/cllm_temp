"""Check Layer 3 attention output vs FFN output."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("LAYER 3: ATTENTION vs FFN OUTPUT", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]

runner.model.cuda()

# Capture Layer 3 attention output and FFN output
l3_attn_out = [None]
l3_ffn_out = [None]
l3_block_out = [None]

def attn_hook(module, input, output):
    l3_attn_out[0] = output.detach().cpu()

def ffn_hook(module, input, output):
    l3_ffn_out[0] = output.detach().cpu()

def block_hook(module, input, output):
    l3_block_out[0] = output.detach().cpu()

h_attn = runner.model.blocks[3].attn.register_forward_hook(attn_hook)
h_ffn = runner.model.blocks[3].ffn.register_forward_hook(ffn_hook)
h_block = runner.model.blocks[3].register_forward_hook(block_hook)

print("Running VM...", file=sys.stderr)
_, exit_code = runner.run(bytecode, max_steps=10)

h_attn.remove()
h_ffn.remove()
h_block.remove()

print(f"Exit code: {exit_code}\n", file=sys.stderr)

# Check AX_CARRY at each stage
print("Layer 3 AX_CARRY progression:", file=sys.stderr)
print("-" * 70, file=sys.stderr)

if l3_attn_out[0] is not None:
    attn_out = l3_attn_out[0]
    ax_carry_lo = attn_out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = attn_out[0, -1, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  After ATTENTION: AX_CARRY_LO max={ax_carry_lo.max():.3f}, AX_CARRY_HI max={ax_carry_hi.max():.3f}", file=sys.stderr)

if l3_ffn_out[0] is not None:
    ffn_out = l3_ffn_out[0]
    ax_carry_lo = ffn_out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = ffn_out[0, -1, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  After FFN:       AX_CARRY_LO max={ax_carry_lo.max():.3f}, AX_CARRY_HI max={ax_carry_hi.max():.3f}", file=sys.stderr)

if l3_block_out[0] is not None:
    block_out = l3_block_out[0]
    ax_carry_lo = block_out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = block_out[0, -1, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]
    print(f"  After BLOCK:     AX_CARRY_LO max={ax_carry_lo.max():.3f}, AX_CARRY_HI max={ax_carry_hi.max():.3f}", file=sys.stderr)

print("-" * 70, file=sys.stderr)
print("\nDiagnosis:", file=sys.stderr)

if l3_attn_out[0] is not None and l3_ffn_out[0] is not None:
    attn_max = l3_attn_out[0][0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()
    ffn_max = l3_ffn_out[0][0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16].max().item()

    if attn_max > 0.5 and ffn_max < 0.1:
        print("  ❌ Layer 3 FFN is OVERWRITING AX_CARRY!", file=sys.stderr)
        print("     Attention sets it, but FFN zeros it out.", file=sys.stderr)
        print("\n  Root cause: Layer 3 FFN W_down writes to AX_CARRY dimensions", file=sys.stderr)

        # Check if W_down has non-zero weights for AX_CARRY
        ffn3 = runner.model.blocks[3].ffn
        w_down_ax_carry = ffn3.W_down[BD.AX_CARRY_LO:BD.AX_CARRY_HI+16, :]
        nonzero_count = (w_down_ax_carry.abs() > 1e-6).sum().item()
        total_weights = w_down_ax_carry.numel()

        print(f"\n  Layer 3 FFN W_down[AX_CARRY_LO:AX_CARRY_HI, :]:", file=sys.stderr)
        print(f"    Non-zero weights: {nonzero_count} / {total_weights}", file=sys.stderr)

        if nonzero_count > 0:
            print(f"    ⚠ FFN is configured to write to AX_CARRY dimensions!", file=sys.stderr)
            print(f"    This violates the dimension reservation contract.", file=sys.stderr)
        else:
            print(f"    ✓ W_down doesn't write to AX_CARRY", file=sys.stderr)
            print(f"    Issue must be elsewhere (bias term?).", file=sys.stderr)
    elif attn_max > 0.5 and ffn_max > 0.5:
        print("  ✓ FFN preserves AX_CARRY correctly", file=sys.stderr)
    elif attn_max < 0.1:
        print("  ❌ Attention itself is not setting AX_CARRY!", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
