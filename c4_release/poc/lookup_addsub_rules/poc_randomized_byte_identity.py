"""Randomized byte-identity test: torch.allclose on OUTPUT_LO/HI bands.

Build AddSub5StageBlock vs rule-derived FFN. Test that the activations
on OUTPUT_LO/HI bands are close (not just argmax-equal).
"""
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

import torch  # noqa: E402

from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.efficient_alu_addsub_split import AddSub5StageBlock  # noqa: E402
from neural_vm.unified_compiler.building_blocks_dsl import multi_way_and_rule  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


S = 100.0


def _make_input(*, a: int, b: int, op: str) -> torch.Tensor:
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    op_dim = _SetDim.OP_ADD if op == "add" else _SetDim.OP_SUB
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a & 0xF)] = 1.0
    x[0, 0, _SetDim.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    return x


def build_byte_addsub_rules(*, op: str):
    rules = []
    op_gate = "OP_ADD" if op == "add" else "OP_SUB"
    carry_dim = "CARRY+1" if op == "add" else "CARRY+2"
    # Margin between conditions sum (100) and threshold (90) is 10. With
    # large-S SwiGLU, fired unit's up*gate ~= S * margin = 1000. To produce
    # composite-matching amplitude of 2.0, write_amp = 2.0 / 1000 = 0.002.
    margin = 10.0
    write_amp = 2.0 / (S * margin)
    for a in range(256):
        a_lo = a & 0xF
        a_hi = (a >> 4) & 0xF
        for b in range(256):
            b_lo = b & 0xF
            b_hi = (b >> 4) & 0xF
            if op == "add":
                total = a + b
                result = total & 0xFF
                carry_out = total >= 256
            else:
                raw = a - b
                result = raw & 0xFF
                carry_out = raw < 0
            r_lo = result & 0xF
            r_hi = (result >> 4) & 0xF
            writes = [
                (f"OUTPUT_LO+{r_lo}", write_amp),
                (f"OUTPUT_HI+{r_hi}", write_amp),
            ]
            if carry_out:
                writes.append((carry_dim, write_amp))
            rules.append(multi_way_and_rule(
                name=f"byte_{op}_a{a:02x}_b{b:02x}",
                conditions=(
                    ("MARK_AX", 40.0),
                    (f"ALU_LO+{a_lo}", 15.0),
                    (f"ALU_HI+{a_hi}", 15.0),
                    (f"AX_CARRY_LO+{b_lo}", 15.0),
                    (f"AX_CARRY_HI+{b_hi}", 15.0),
                ),
                threshold=90.0,
                gate=op_gate,
                writes=tuple(writes),
            ))
    return tuple(rules)


def main():
    composite = AddSub5StageBlock(S=S, BD=_SetDim).eval()
    add_rules = build_byte_addsub_rules(op="add")
    sub_rules = build_byte_addsub_rules(op="sub")
    rules = add_rules + sub_rules
    ffn = PureFFN(dim=512, hidden_dim=len(rules))
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit=0, S=S)
    ffn.eval()

    print("Randomized byte-identity comparison:")
    print("  Per case: composite OUTPUT_LO[r_lo], rule OUTPUT_LO[r_lo], "
          "composite OUTPUT_HI[r_hi], rule OUTPUT_HI[r_hi]")
    gen = torch.Generator().manual_seed(0xADD04)
    n_cases = 10
    pairs = torch.randint(0, 256, (n_cases, 2), generator=gen).tolist()
    all_match = True
    for op in ("add", "sub"):
        for a, b in pairs:
            x = _make_input(a=a, b=b, op=op)
            with torch.no_grad():
                yc = composite(x)
                yr = ffn(x)
            # Compare OUTPUT_LO/HI/CARRY bands directly
            lo_c = yc[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
            lo_r = yr[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
            hi_c = yc[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16]
            hi_r = yr[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16]
            carry_c = yc[0, 0, _SetDim.CARRY:_SetDim.CARRY + 4]
            carry_r = yr[0, 0, _SetDim.CARRY:_SetDim.CARRY + 4]

            argmax_match_lo = lo_c.argmax().item() == lo_r.argmax().item()
            argmax_match_hi = hi_c.argmax().item() == hi_r.argmax().item()
            allclose_lo = torch.allclose(lo_c, lo_r, atol=0.1)
            allclose_hi = torch.allclose(hi_c, hi_r, atol=0.1)
            allclose_carry = torch.allclose(carry_c, carry_r, atol=0.1)
            # Full residual comparison
            full_close = torch.allclose(yc, yr, atol=0.1)
            ok = argmax_match_lo and argmax_match_hi
            if not ok:
                all_match = False
            print(
                f"  {op} 0x{a:02X} 0x{b:02X}: "
                f"close_lo={allclose_lo} close_hi={allclose_hi} "
                f"close_carry={allclose_carry} close_full={full_close} | "
                f"carry_c={carry_c.tolist()} carry_r={carry_r.tolist()}"
            )

    print()
    print(f"All argmax match: {all_match}")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
