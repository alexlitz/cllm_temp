"""POC: build rule-derived FFN for L8 single-byte ADD/SUB, check
byte-identity vs AddSub5StageBlock.

This script:
  1. Builds AddSub5StageBlock(S, BD).
  2. Constructs a candidate rule set covering:
       - ADD: 256 lo-nibble rules + 256 hi-nibble rules + carry-out
       - SUB: 256 lo-nibble rules + 256 hi-nibble rules + borrow-out
       - Internal LO->HI nibble carry handling (a + b carry from low)
  3. Lowers via Primitives.lower_ffn_rules into a single PureFFN.
  4. Runs both on randomized 8-bit ADD/SUB inputs and decodes
     OUTPUT_LO/HI + CARRY[1]/[2].

Note: A single PureFFN cannot self-cascade carries within one forward
pass (W_up only reads input). The 5-stage block handles LO->HI carry
via the carry-lookahead stage (stage 2). To match in a single pass we
build a wider rule set that conditions hi-nibble rules on both b_hi
and the pre-computed lo-nibble carry.

But the lo-nibble carry depends on the input lo nibbles -- which we
DO have in the input residual. So we can encode the hi-nibble rule
as a 4-way AND over (MARK, OP, a_hi, b_hi) PLUS a lo-carry condition
encoded as a 256-way disjunction over the lo nibble pairs that
produce a carry. That blows up the rule count to ~256*256 for the
hi nibble.

A more compact form: emit 256 hi-nibble rules per (a_lo, b_lo)
configuration that produces a carry -- this is the same lookup-table
approach. Total: 16*16 * (16*16) = 65,536 rules for the
carry-inclusive hi nibble, which is too many.

The right move is to use TWO sequential FFN passes (= the existing
post-op architecture's _expand_wrapper_blocks splits the AddSub
post-op into 5 successive blocks already). Each rule-derived FFN
would replace one of the 5 stages.

This POC tries the single-pass option first to see if the rule count
is actually tractable, since the existing 5-stage block has ~512-1024
hidden units per stage already.
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
    """Build a single-position residual at MARK_AX with operand a/b + op."""
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


def _decode(y: torch.Tensor) -> tuple[int, int, int, int]:
    lo = int(y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item())
    hi = int(y[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16].argmax().item())
    carry = float(y[0, 0, _SetDim.CARRY + 1].item())
    borrow = float(y[0, 0, _SetDim.CARRY + 2].item())
    return lo, hi, carry, borrow


def build_byte_addsub_rules(*, op: str):
    """Build per-byte ADD or SUB lookup rules covering both nibbles + carry.

    Strategy: enumerate all 65,536 (a, b) byte pairs. For each:
      - Emit ONE rule conditioned on (MARK, a_lo, a_hi, b_lo, b_hi)
        that writes OUTPUT_LO[sum_lo] + OUTPUT_HI[sum_hi]
        + CARRY[1 or 2] (if op produces carry/borrow).
    This is the "full byte lookup" approach -- one rule per byte pair.
    """
    rules = []
    op_gate = "OP_ADD" if op == "add" else "OP_SUB"
    carry_dim = "CARRY+1" if op == "add" else "CARRY+2"
    write_amp = 2.0 / S
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
                carry_out = raw < 0  # borrow
            r_lo = result & 0xF
            r_hi = (result >> 4) & 0xF
            writes = [
                (f"OUTPUT_LO+{r_lo}", write_amp),
                (f"OUTPUT_HI+{r_hi}", write_amp),
            ]
            if carry_out:
                writes.append((carry_dim, write_amp))
            # 5-way AND: marker(40) + a_lo(15) + a_hi(15) + b_lo(15) + b_hi(15) > 90.
            # All 5 present: 40 + 15*4 = 100 > 90.
            # Any 4: max(40 + 15*3) = 85 < 90.
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
    print("Building AddSub5StageBlock...")
    composite = AddSub5StageBlock(S=S, BD=_SetDim)
    composite.eval()

    print("Building rule-derived FFN (this may take a moment)...")
    add_rules = build_byte_addsub_rules(op="add")
    sub_rules = build_byte_addsub_rules(op="sub")
    rules = add_rules + sub_rules
    print(f"  rule count: {len(rules)}")

    # Lower into a fresh PureFFN.
    print("Lowering rules into PureFFN...")
    ffn = PureFFN(dim=512, hidden_dim=len(rules))
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(ffn, rules, dim_positions, start_unit=0, S=S)
    assert end == len(rules), f"lower wrote {end} units, expected {len(rules)}"
    ffn.eval()

    print("Running byte-identity comparison...")
    test_cases = [
        ("add", 0x12, 0x34),
        ("add", 0xFF, 0x01),  # carry
        ("add", 0xAB, 0xCD),
        ("add", 0x80, 0x80),  # exact 256
        ("add", 0x00, 0x00),
        ("sub", 0x50, 0x30),
        ("sub", 0x10, 0x20),  # borrow
        ("sub", 0x00, 0x01),  # underflow
        ("sub", 0xFF, 0xFF),
        ("sub", 0xAB, 0xCD),  # borrow
    ]
    mismatches = []
    for op, a, b in test_cases:
        x = _make_input(a=a, b=b, op=op)
        with torch.no_grad():
            y_comp = composite(x)
            y_rule = ffn(x)
        comp = _decode(y_comp)
        rule = _decode(y_rule)
        expected_byte = (a + b) & 0xFF if op == "add" else (a - b) & 0xFF
        # Decode bytes
        comp_byte = comp[0] | (comp[1] << 4)
        rule_byte = rule[0] | (rule[1] << 4)
        print(
            f"  {op} 0x{a:02X} 0x{b:02X}: expected=0x{expected_byte:02X} "
            f"composite=0x{comp_byte:02X} carry/borrow={comp[2]:.3f}/{comp[3]:.3f}  "
            f"rule=0x{rule_byte:02X} carry/borrow={rule[2]:.3f}/{rule[3]:.3f}"
        )
        if comp_byte != expected_byte or rule_byte != expected_byte:
            mismatches.append((op, a, b, expected_byte, comp_byte, rule_byte))
        if op == "add" and (comp[2] > 0.005) != (rule[2] > 0.005):
            mismatches.append((op, a, b, "carry_mismatch", comp[2], rule[2]))
        if op == "sub" and (comp[3] > 0.005) != (rule[3] > 0.005):
            mismatches.append((op, a, b, "borrow_mismatch", comp[3], rule[3]))

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} mismatches:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print("\nPASS: all test cases byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
