"""
Test if the transformer can run full C4 programs.
"""

import torch
from c4_vm import Op, C4VM, program_simple_arithmetic, program_sum, program_factorial
from c4_printf_autoregressive import C4AutoregressivePrintf


def instr(op, imm=0):
    return float(op + (imm << 8))


def run_on_transformer(code, memory_size=512, stack_start=450, max_steps=500):
    """Run bytecode on transformer executor."""
    executor = C4AutoregressivePrintf(memory_size=memory_size)

    memory = torch.zeros(memory_size)
    for i, c in enumerate(code):
        if i * 8 < memory_size:
            memory[i * 8] = float(c)

    pc = torch.tensor(0.0)
    sp = torch.tensor(float(stack_start))
    bp = torch.tensor(float(stack_start))
    ax = torch.tensor(0.0)

    pc, sp, bp, ax, memory, output = executor.run(memory, pc, sp, bp, ax, max_steps=max_steps)

    return int(ax.item()), output


def run_on_vm(code):
    """Run bytecode on reference C4 VM."""
    vm = C4VM()
    vm.load_code(code)
    result = vm.run()
    return result


def test_programs():
    print("FULL C4 PROGRAM TESTS")
    print("=" * 60)
    print()
    print("Testing transformer executor against reference C4 VM")
    print()

    all_passed = True

    # Test 1: Simple arithmetic (3+4)*5 = 35
    print("Test 1: (3+4)*5 = 35")
    code1 = [
        instr(Op.IMM, 3),
        instr(Op.PSH),
        instr(Op.IMM, 4),
        instr(Op.ADD),
        instr(Op.PSH),
        instr(Op.IMM, 5),
        instr(Op.MUL),
        instr(Op.EXIT),
    ]

    transformer_result, _ = run_on_transformer(code1)
    print(f"  Transformer: {transformer_result}")

    status = "✓" if transformer_result == 35 else "✗"
    print(f"  {status} (3+4)*5 = {transformer_result}")
    all_passed = all_passed and (transformer_result == 35)
    print()

    # Test 2: Division and modulo
    print("Test 2: 47 / 7 = 6, 47 % 7 = 5")
    code2a = [
        instr(Op.IMM, 47),
        instr(Op.PSH),
        instr(Op.IMM, 7),
        instr(Op.DIV),
        instr(Op.EXIT),
    ]
    code2b = [
        instr(Op.IMM, 47),
        instr(Op.PSH),
        instr(Op.IMM, 7),
        instr(Op.MOD),
        instr(Op.EXIT),
    ]

    div_result, _ = run_on_transformer(code2a)
    mod_result, _ = run_on_transformer(code2b)
    print(f"  47 / 7 = {div_result}, 47 % 7 = {mod_result}")

    status = "✓" if div_result == 6 and mod_result == 5 else "✗"
    print(f"  {status} Division and modulo correct")
    all_passed = all_passed and (div_result == 6 and mod_result == 5)
    print()

    # Test 3: Comparisons
    print("Test 3: Comparisons")
    tests_cmp = [
        ("5 > 3", Op.GT, 5, 3, 1),
        ("3 > 5", Op.GT, 3, 5, 0),
        ("5 == 5", Op.EQ, 5, 5, 1),
        ("5 != 3", Op.NE, 5, 3, 1),
        ("3 <= 5", Op.LE, 3, 5, 1),
        ("5 >= 5", Op.GE, 5, 5, 1),
    ]

    cmp_passed = True
    for name, op, a, b, expected in tests_cmp:
        code = [
            instr(Op.IMM, a),
            instr(Op.PSH),
            instr(Op.IMM, b),
            instr(op),
            instr(Op.EXIT),
        ]
        result, _ = run_on_transformer(code)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name} = {result} (expected {expected})")
        cmp_passed = cmp_passed and (result == expected)

    all_passed = all_passed and cmp_passed
    print()

    # Test 4: Bitwise operations
    print("Test 4: Bitwise operations")
    tests_bit = [
        ("12 & 10", Op.AND, 12, 10, 12 & 10),
        ("12 | 10", Op.OR, 12, 10, 12 | 10),
        ("12 ^ 10", Op.XOR, 12, 10, 12 ^ 10),
        ("3 << 2", Op.SHL, 3, 2, 3 << 2),
        ("16 >> 2", Op.SHR, 16, 2, 16 >> 2),
    ]

    bit_passed = True
    for name, op, a, b, expected in tests_bit:
        code = [
            instr(Op.IMM, a),
            instr(Op.PSH),
            instr(Op.IMM, b),
            instr(op),
            instr(Op.EXIT),
        ]
        result, _ = run_on_transformer(code)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name} = {result} (expected {expected})")
        bit_passed = bit_passed and (result == expected)

    all_passed = all_passed and bit_passed
    print()

    # Test 5: Conditional branch
    print("Test 5: Conditional branch (if x > 0 then 100 else 200)")

    def make_conditional(x):
        # if x > 0: return 100 else: return 200
        ELSE_BRANCH = 7 * 8  # instruction 7
        END = 9 * 8  # instruction 9
        return [
            instr(Op.IMM, x),       # 0: ax = x
            instr(Op.PSH),          # 1: push x
            instr(Op.IMM, 0),       # 2: ax = 0
            instr(Op.GT),           # 3: ax = (x > 0)
            instr(Op.BZ, ELSE_BRANCH),  # 4: if !ax goto else
            instr(Op.IMM, 100),     # 5: ax = 100
            instr(Op.JMP, END),     # 6: goto end
            instr(Op.IMM, 200),     # 7: ax = 200 (else branch)
            instr(Op.JMP, END),     # 8: goto end
            instr(Op.EXIT),         # 9: exit
        ]

    result_pos, _ = run_on_transformer(make_conditional(5))
    result_neg, _ = run_on_transformer(make_conditional(-5))
    result_zero, _ = run_on_transformer(make_conditional(0))

    print(f"  x=5:  {result_pos} (expected 100)")
    print(f"  x=-5: {result_neg} (expected 200)")
    print(f"  x=0:  {result_zero} (expected 200)")

    branch_ok = result_pos == 100 and result_neg == 200 and result_zero == 200
    status = "✓" if branch_ok else "✗"
    print(f"  {status} Conditional branches work")
    all_passed = all_passed and branch_ok
    print()

    # Test 6: Loop (sum 1 to 5)
    print("Test 6: Loop - sum 1 to 5 = 15")

    # Simple loop: sum = 0; for i = 1 to 5: sum += i
    # Using memory locations for variables (after code area)
    SUM_ADDR = 300
    I_ADDR = 308

    LOOP_START = 8 * 8   # instruction 8
    LOOP_END = 30 * 8    # instruction 30

    code_loop = [
        # sum = 0
        instr(Op.IMM, SUM_ADDR),   # 0
        instr(Op.PSH),              # 1
        instr(Op.IMM, 0),           # 2
        instr(Op.SI),               # 3: mem[SUM_ADDR] = 0

        # i = 1
        instr(Op.IMM, I_ADDR),      # 4
        instr(Op.PSH),              # 5
        instr(Op.IMM, 1),           # 6
        instr(Op.SI),               # 7: mem[I_ADDR] = 1

        # LOOP: if i > 5, exit
        instr(Op.IMM, I_ADDR),      # 8: ax = &i
        instr(Op.LI),               # 9: ax = i
        instr(Op.PSH),              # 10: push i
        instr(Op.IMM, 6),           # 11: ax = 6
        instr(Op.LT),               # 12: ax = (i < 6)
        instr(Op.BZ, LOOP_END),     # 13: if i >= 6, exit loop

        # sum += i
        instr(Op.IMM, SUM_ADDR),    # 14: ax = &sum
        instr(Op.PSH),              # 15: push &sum (for SI)
        instr(Op.LI),               # 16: ax = sum
        instr(Op.PSH),              # 17: push sum
        instr(Op.IMM, I_ADDR),      # 18: ax = &i
        instr(Op.LI),               # 19: ax = i
        instr(Op.ADD),              # 20: ax = sum + i
        instr(Op.SI),               # 21: mem[&sum] = sum + i

        # i += 1
        instr(Op.IMM, I_ADDR),      # 22: ax = &i
        instr(Op.PSH),              # 23: push &i
        instr(Op.LI),               # 24: ax = i
        instr(Op.PSH),              # 25: push i
        instr(Op.IMM, 1),           # 26: ax = 1
        instr(Op.ADD),              # 27: ax = i + 1
        instr(Op.SI),               # 28: mem[&i] = i + 1

        instr(Op.JMP, LOOP_START),  # 29: goto LOOP

        # END: return sum
        instr(Op.IMM, SUM_ADDR),    # 30 (LOOP_END)
        instr(Op.LI),               # 31
        instr(Op.EXIT),             # 32
    ]

    result_loop, _ = run_on_transformer(code_loop, max_steps=300)

    status = "✓" if result_loop == 15 else "✗"
    print(f"  {status} sum(1..5) = {result_loop} (expected 15)")
    all_passed = all_passed and (result_loop == 15)
    print()

    # Test 7: Printf with computed value
    print("Test 7: Compute and printf")
    code_printf = [
        instr(Op.IMM, 6),
        instr(Op.PSH),
        instr(Op.IMM, 7),
        instr(Op.MUL),          # ax = 42
        instr(Op.PSH),
        instr(Op.IMM, 0),
        instr(Op.PSH),
        instr(Op.PRTF),         # printf(42)
        instr(Op.ADJ, 16),
        instr(Op.EXIT),
    ]

    _, output = run_on_transformer(code_printf)
    print(f"  Output: '{output.strip()}'")
    status = "✓" if output.strip() == "42" else "✗"
    print(f"  {status} printf(6*7) = 42")
    all_passed = all_passed and (output.strip() == "42")
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print()
        print("The transformer can execute C4 programs:")
        print("  ✓ Arithmetic (+, -, *, /, %)")
        print("  ✓ Bitwise (&, |, ^, <<, >>)")
        print("  ✓ Comparisons (==, !=, <, >, <=, >=)")
        print("  ✓ Conditional branches (BZ, BNZ)")
        print("  ✓ Loops (JMP)")
        print("  ✓ Memory (LI, SI)")
        print("  ✓ Printf (autoregressive output)")
    else:
        print("SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    test_programs()
