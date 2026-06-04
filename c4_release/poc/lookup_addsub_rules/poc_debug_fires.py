"""Debug how many rules fire for a single (a, b) ADD."""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

import torch  # noqa: E402

from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


S = 100.0

# Build a single byte rule (a=0xFF, b=0x01, op=add) -- should fire once with carry.
rules = []
a, b = 0xFF, 0x01
a_lo, a_hi, b_lo, b_hi = a & 0xF, (a >> 4) & 0xF, b & 0xF, (b >> 4) & 0xF
total = a + b
result = total & 0xFF
r_lo, r_hi = result & 0xF, (result >> 4) & 0xF
write_amp = 2.0 / S
rules.append(multi_way_and_rule(
    name=f"only_a{a:02x}_b{b:02x}",
    conditions=(
        ("MARK_AX", 40.0),
        (f"ALU_LO+{a_lo}", 15.0),
        (f"ALU_HI+{a_hi}", 15.0),
        (f"AX_CARRY_LO+{b_lo}", 15.0),
        (f"AX_CARRY_HI+{b_hi}", 15.0),
    ),
    threshold=90.0,
    gate="OP_ADD",
    writes=(
        (f"OUTPUT_LO+{r_lo}", write_amp),
        (f"OUTPUT_HI+{r_hi}", write_amp),
        ("CARRY+1", write_amp),
    ),
))

ffn = PureFFN(dim=512, hidden_dim=len(rules))
names = Primitives.ffn_rule_dim_names(rules)
dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
end = Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit=0, S=S)
ffn.eval()

x = torch.zeros(1, 1, 512)
x[0, 0, _SetDim.MARK_AX] = 1.0
x[0, 0, _SetDim.CONST] = 1.0
x[0, 0, _SetDim.OP_ADD] = 1.0
x[0, 0, _SetDim.ALU_LO + a_lo] = 1.0
x[0, 0, _SetDim.ALU_HI + a_hi] = 1.0
x[0, 0, _SetDim.AX_CARRY_LO + b_lo] = 1.0
x[0, 0, _SetDim.AX_CARRY_HI + b_hi] = 1.0

with torch.no_grad():
    y = ffn(x)
print("Single-rule fire:")
print(f"  OUTPUT_LO[r_lo={r_lo}] = {y[0, 0, _SetDim.OUTPUT_LO + r_lo].item():.4f}")
print(f"  OUTPUT_HI[r_hi={r_hi}] = {y[0, 0, _SetDim.OUTPUT_HI + r_hi].item():.4f}")
print(f"  CARRY+1 = {y[0, 0, _SetDim.CARRY + 1].item():.4f}")
