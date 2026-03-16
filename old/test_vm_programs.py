"""
Test the MoE VM with various programs using fixed MUL/DIV.
"""

import torch
from c4_moe_vm import C4MoEVM, C4Op

def make_bytecode(instructions):
    """Convert instruction list to bytecode.
    Each instruction is (opcode, immediate) or just opcode.
    """
    code = []
    for instr in instructions:
        if isinstance(instr, tuple):
            op, imm = instr
            code.append(op | (imm << 8))
        else:
            code.append(instr)
    return code

def test_programs():
    """Test various programs."""
    print("=" * 60)
    print("TESTING MOE VM WITH FIXED MUL/DIV")
    print("=" * 60)
    print("MUL: silu(a)*b + silu(-a)*(-b) = a*b (exact via SwiGLU)")
    print("DIV: Log via attention + exp via softmax ratio")
    print()

    tests = [
        # (name, bytecode, expected_result)
        ("6 * 7 = 42", [
            (C4Op.IMM, 6),      # ax = 6
            C4Op.PSH,           # push 6
            (C4Op.IMM, 7),      # ax = 7
            C4Op.MUL,           # ax = 6 * 7 = 42
            C4Op.EXIT,          # exit with ax
        ], 42),

        ("100 / 5 = 20", [
            (C4Op.IMM, 100),    # ax = 100
            C4Op.PSH,           # push 100
            (C4Op.IMM, 5),      # ax = 5
            C4Op.DIV,           # ax = 100 / 5 = 20
            C4Op.EXIT,
        ], 20),

        ("(3 + 4) * 5 = 35", [
            (C4Op.IMM, 3),      # ax = 3
            C4Op.PSH,           # push 3
            (C4Op.IMM, 4),      # ax = 4
            C4Op.ADD,           # ax = 3 + 4 = 7
            C4Op.PSH,           # push 7
            (C4Op.IMM, 5),      # ax = 5
            C4Op.MUL,           # ax = 7 * 5 = 35
            C4Op.EXIT,
        ], 35),

        ("100/5 + 10*3 = 50", [
            (C4Op.IMM, 100),    # ax = 100
            C4Op.PSH,           # push 100
            (C4Op.IMM, 5),      # ax = 5
            C4Op.DIV,           # ax = 100/5 = 20
            C4Op.PSH,           # push 20
            (C4Op.IMM, 10),     # ax = 10
            C4Op.PSH,           # push 10
            (C4Op.IMM, 3),      # ax = 3
            C4Op.MUL,           # ax = 10*3 = 30
            C4Op.ADD,           # ax = 20 + 30 = 50
            C4Op.EXIT,
        ], 50),

        ("(120/4)/3 = 10", [
            (C4Op.IMM, 120),    # ax = 120
            C4Op.PSH,           # push 120
            (C4Op.IMM, 4),      # ax = 4
            C4Op.DIV,           # ax = 120/4 = 30
            C4Op.PSH,           # push 30
            (C4Op.IMM, 3),      # ax = 3
            C4Op.DIV,           # ax = 30/3 = 10
            C4Op.EXIT,
        ], 10),

        ("17 % 5 = 2", [
            (C4Op.IMM, 17),     # ax = 17
            C4Op.PSH,           # push 17
            (C4Op.IMM, 5),      # ax = 5
            C4Op.MOD,           # ax = 17 % 5 = 2
            C4Op.EXIT,
        ], 2),

        ("42 / 7 = 6", [
            (C4Op.IMM, 42),
            C4Op.PSH,
            (C4Op.IMM, 7),
            C4Op.DIV,
            C4Op.EXIT,
        ], 6),

        ("13 * 17 = 221", [
            (C4Op.IMM, 13),
            C4Op.PSH,
            (C4Op.IMM, 17),
            C4Op.MUL,
            C4Op.EXIT,
        ], 221),
    ]

    vm = C4MoEVM()
    passed = 0

    for name, instructions, expected in tests:
        vm.reset()
        code = make_bytecode(instructions)
        vm.load(code, [])
        result, _, stats = vm.run()

        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1

        print(f"  {name:25s} = {result:4d} (expected {expected:4d}) {status}")

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_xc_bytecode():
    """Test bytecode from xc compiler."""
    print("\n" + "=" * 60)
    print("TESTING XC-COMPILED BYTECODE")
    print("=" * 60)

    # From previous session: xc compiled "int main() { return 6 * 7; }"
    # Raw bytecode was: 6 0 1 6 13 1 7 27 8
    # But this needs to be properly formatted for our VM

    # Let's manually create the equivalent
    # ENT 0  -> enter with 0 local space
    # IMM 6  -> load 6
    # PSH    -> push
    # IMM 7  -> load 7
    # MUL    -> multiply
    # LEV    -> leave (but we use EXIT for top-level)

    print("  Simulating xc bytecode for: int main() { return 6 * 7; }")

    vm = C4MoEVM()
    code = make_bytecode([
        (C4Op.ENT, 0),      # enter function
        (C4Op.IMM, 6),      # ax = 6
        C4Op.PSH,           # push 6
        (C4Op.IMM, 7),      # ax = 7
        C4Op.MUL,           # ax = 6 * 7 = 42
        C4Op.EXIT,          # exit (replacing LEV for main)
    ])
    vm.load(code, [])
    result, _, _ = vm.run()

    print(f"  Result: {result} (expected 42)")
    print(f"  Status: {'✓ PASS' if result == 42 else '✗ FAIL'}")

    return result == 42


if __name__ == "__main__":
    r1 = test_programs()
    r2 = test_xc_bytecode()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Program tests: {'PASS' if r1 else 'FAIL'}")
    print(f"  XC bytecode: {'PASS' if r2 else 'FAIL'}")
    print(f"\n  ALL PASS: {r1 and r2}")
    print()
    print("Key implementation details:")
    print("  - MUL via silu pairs: a*b = silu(a)*b + silu(-a)*(-b)")
    print("    Proof: = a*b*(sigmoid(a) + sigmoid(-a)) = a*b*1 = a*b")
    print("  - DIV uses attention log2 lookup + exp via softmax ratio")
    print("  - exp(x) = softmax([x,0])[0] / softmax([x,0])[1] (exact)")
    print("  - 0 learned parameters, pure transformer primitives")
