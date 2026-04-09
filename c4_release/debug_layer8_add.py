"""Check Layer 8 ADD computation with full trace."""

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode, Token, _SetDim
import torch
import sys

BD = _SetDim

code = "int main() { return 10 + 32; }"
bytecode, data = compile_c(code)

print("=" * 70, file=sys.stderr)
print("LAYER 8 ADD COMPUTATION TRACE", file=sys.stderr)
print("=" * 70, file=sys.stderr)

runner = AutoregressiveVMRunner()
if Opcode.ADD in runner._func_call_handlers:
    del runner._func_call_handlers[Opcode.ADD]
    print("✓ ADD handler removed - using pure neural weights\n", file=sys.stderr)

runner.model.cuda()

# Capture Layer 7, 8 outputs
l7_last = [None]
l8_last = [None]

def l7_hook(module, input, output):
    l7_last[0] = output.detach().cpu()

def l8_hook(module, input, output):
    l8_last[0] = output.detach().cpu()

h7 = runner.model.blocks[7].register_forward_hook(l7_hook)
h8 = runner.model.blocks[8].ffn.register_forward_hook(l8_hook)

print("Running VM...", file=sys.stderr)
result, exit_code = runner.run(bytecode, max_steps=10)

h7.remove()
h8.remove()

print(f"\nResult: {result}", file=sys.stderr)
print(f"Exit code: {exit_code}", file=sys.stderr)
print(f"Expected: 42 (10 + 32)\n", file=sys.stderr)

# Check Layer 7 output (gathers operands to ALU_LO/HI)
if l7_last[0] is not None:
    l7_out = l7_last[0]
    seq_len = l7_out.shape[1]
    print(f"Layer 7 output (gather operands):", file=sys.stderr)

    # Get last position (should have ALU dimensions set)
    alu_lo = l7_out[0, -1, BD.ALU_LO:BD.ALU_LO+16]
    alu_hi = l7_out[0, -1, BD.ALU_HI:BD.ALU_HI+16]
    ax_carry_lo = l7_out[0, -1, BD.AX_CARRY_LO:BD.AX_CARRY_LO+16]
    ax_carry_hi = l7_out[0, -1, BD.AX_CARRY_HI:BD.AX_CARRY_HI+16]

    # Decode nibbles
    def decode_nibbles(tensor):
        val = 0
        for i, v in enumerate(tensor):
            if v > 0.5:
                val |= (i << (4 * 0))  # Just the low nibble for simplicity
        return val

    alu_lo_val = 0
    for i in range(16):
        if alu_lo[i] > 0.5:
            alu_lo_val = i
            break

    ax_carry_lo_val = 0
    for i in range(16):
        if ax_carry_lo[i] > 0.5:
            ax_carry_lo_val = i
            break

    print(f"  ALU_LO (first operand): nibble {alu_lo_val} (max={alu_lo.max():.2f})", file=sys.stderr)
    print(f"  AX_CARRY_LO (second op): nibble {ax_carry_lo_val} (max={ax_carry_lo.max():.2f})", file=sys.stderr)
    print(f"  Expected: ALU_LO=10 (0xA), AX_CARRY_LO=0 (from 32=0x20)", file=sys.stderr)

# Check Layer 8 FFN output (computes ADD)
if l8_last[0] is not None:
    l8_out = l8_last[0]
    print(f"\nLayer 8 FFN output (ADD computation):", file=sys.stderr)

    # Get AX result at last position
    ax_out_lo = l8_out[0, -1, BD.AX_OUT_LO:BD.AX_OUT_LO+16]
    ax_out_hi = l8_out[0, -1, BD.AX_OUT_HI:BD.AX_OUT_HI+16]

    ax_lo_val = 0
    for i in range(16):
        if ax_out_lo[i] > 0.5:
            ax_lo_val = i
            break

    ax_hi_val = 0
    for i in range(16):
        if ax_out_hi[i] > 0.5:
            ax_hi_val = i
            break

    result_byte0 = ax_lo_val
    result_byte1 = ax_hi_val

    print(f"  AX_OUT_LO: nibble {ax_lo_val}", file=sys.stderr)
    print(f"  AX_OUT_HI: nibble {ax_hi_val}", file=sys.stderr)
    print(f"  Result (first 2 nibbles): 0x{ax_hi_val:X}{ax_lo_val:X}", file=sys.stderr)
    print(f"  Expected: 0x2A (42 decimal)", file=sys.stderr)

    if ax_lo_val == 0xA and ax_hi_val == 0x2:
        print(f"\n✓ ADD WORKING CORRECTLY!", file=sys.stderr)
    else:
        print(f"\n❌ ADD result incorrect", file=sys.stderr)

print("\n" + "=" * 70, file=sys.stderr)

del runner
torch.cuda.empty_cache()
