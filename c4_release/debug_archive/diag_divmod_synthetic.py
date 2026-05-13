"""Standalone synthetic test of FlattenedDivMod given perfect inputs.

Bypasses the L7 operand-gather problem entirely: hand-crafts a BD tensor
with operand A in ALU_LO, operand B in AX_CARRY_LO, OP_DIV/MOD set, and
MARK_AX=1, then runs FlattenedDivMod and checks OUTPUT_LO/HI nibbles.

Confirms whether the Phase 7 DIV/MOD pipeline itself is correct,
independent of the Phase 2/3 L7 operand-gather bug that test_add_basic
exposes.
"""
import os, sys
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from c4_release.neural_vm.efficient_alu_divmod_split import FlattenedDivMod
from c4_release.neural_vm.unified_compiler.full_vm_compiler import compile_full_vm

print(">>> Compiling full VM to obtain dim_positions...")
model, layout = compile_full_vm()
dim = layout.dim_positions
print(f"  d_model={layout.d_model}, n_layers={layout.n_layers}")


class _DimProxy:
    def __init__(self, dim):
        for name, pos in dim.items():
            setattr(self, name, pos)


BD = _DimProxy(dim)

print("\n>>> Constructing FlattenedDivMod composite directly...")
S = 100.0
div = FlattenedDivMod(S, BD)
div.install_bdtoge()
div.install_longdiv()
div.install_getobd()
print(f"  stages installed: {list(div.stages.keys())}")

device = next(model.parameters()).device
div = div.to(device)
D = layout.d_model
seq = 4


def encode_nibble_lo(val): return val & 0xF
def encode_nibble_hi(val): return (val >> 4) & 0xF


def make_bd_input(opA, opB, op_div=False, op_mod=False, mark_ax=True):
    x = torch.zeros(1, seq, D, device=device)
    pos = 2
    x[0, pos, BD.ALU_LO + encode_nibble_lo(opA)] = 1.0
    x[0, pos, BD.ALU_HI + encode_nibble_hi(opA)] = 1.0
    x[0, pos, BD.AX_CARRY_LO + encode_nibble_lo(opB)] = 1.0
    x[0, pos, BD.AX_CARRY_HI + encode_nibble_hi(opB)] = 1.0
    if op_div:
        x[0, pos, BD.OP_DIV] = 1.0
    if op_mod:
        x[0, pos, BD.OP_MOD] = 1.0
    if mark_ax:
        x[0, pos, BD.MARK_AX] = 1.0
    return x, pos


def decode_output(x_out, pos):
    lo = x_out[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO + 16]
    hi = x_out[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI + 16]
    lo_a = int(lo.argmax().item())
    hi_a = int(hi.argmax().item())
    val = (hi_a << 4) | lo_a
    return val, (lo_a, float(lo[lo_a].item())), (hi_a, float(hi[hi_a].item()))


tests = [
    ("DIV 10/2", 10, 2, True, False, 5),
    ("DIV 84/2", 84, 2, True, False, 42),
    ("MOD 43%10", 43, 10, False, True, 3),
    ("DIV 0/2 (L7-fail)", 0, 2, True, False, 0),
]

for name, a, b, do_div, do_mod, expect in tests:
    with torch.no_grad():
        x_in, pos = make_bd_input(a, b, op_div=do_div, op_mod=do_mod)
        x_out = div(x_in)
    val, lo, hi = decode_output(x_out, pos)
    status = "PASS" if val == expect else "FAIL"
    print(f"  {status}: {name} -> {val} (expect {expect}); "
          f"LO[arg={lo[0]} m={lo[1]:.2f}] HI[arg={hi[0]} m={hi[1]:.2f}]")

print("\n>>> Done.")
