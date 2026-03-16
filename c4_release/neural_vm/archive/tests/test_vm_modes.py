"""
Comprehensive tests for both VM modes:
1. Token-based (bytecode in prompt, attention fetch)
2. Weight-based (bytecode baked into FFN weights)

Both should produce identical results for the same programs.
"""

import sys
import os

# Handle pytest import for decorator compatibility
try:
    import pytest
    parametrize = pytest.mark.parametrize
except ImportError:
    pytest = None
    # Dummy decorator when pytest not available
    def parametrize(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_vm.token_sequence_vm import TokenSequenceVM
from neural_vm.raw_bytecode_vm import RawBytecodeVM
from neural_vm.baked_program import compile_packed_to_baked, BakedVM


# =============================================================================
# TEST PROGRAMS
# =============================================================================

# Format: (name, bytecode, expected_result)
ARITHMETIC_TESTS = [
    ("6 * 7", [1|(6<<8), 13, 1|(7<<8), 27, 38], 42),
    ("100 - 58", [1|(100<<8), 13, 1|(58<<8), 26, 38], 42),
    ("10 + 32", [1|(10<<8), 13, 1|(32<<8), 25, 38], 42),
    ("84 / 2", [1|(84<<8), 13, 1|(2<<8), 28, 38], 42),
    ("50 % 8", [1|(50<<8), 13, 1|(8<<8), 29, 38], 2),
    ("3 * 3 * 3", [1|(3<<8), 13, 1|(3<<8), 27, 13, 1|(3<<8), 27, 38], 27),
    ("(2 + 3) * 4", [1|(2<<8), 13, 1|(3<<8), 25, 13, 1|(4<<8), 27, 38], 20),
]

COMPARISON_TESTS = [
    ("5 == 5", [1|(5<<8), 13, 1|(5<<8), 17, 38], 1),
    ("5 == 6", [1|(5<<8), 13, 1|(6<<8), 17, 38], 0),
    ("5 != 6", [1|(5<<8), 13, 1|(6<<8), 18, 38], 1),
    ("5 != 5", [1|(5<<8), 13, 1|(5<<8), 18, 38], 0),
    ("3 < 5", [1|(3<<8), 13, 1|(5<<8), 19, 38], 1),
    ("5 < 3", [1|(5<<8), 13, 1|(3<<8), 19, 38], 0),
    ("5 > 3", [1|(5<<8), 13, 1|(3<<8), 20, 38], 1),
    ("3 > 5", [1|(3<<8), 13, 1|(5<<8), 20, 38], 0),
    ("3 <= 5", [1|(3<<8), 13, 1|(5<<8), 21, 38], 1),
    ("5 <= 5", [1|(5<<8), 13, 1|(5<<8), 21, 38], 1),
    ("5 >= 3", [1|(5<<8), 13, 1|(3<<8), 22, 38], 1),
    ("5 >= 5", [1|(5<<8), 13, 1|(5<<8), 22, 38], 1),
]

BITWISE_TESTS = [
    ("7 & 3", [1|(7<<8), 13, 1|(3<<8), 16, 38], 3),
    ("4 | 2", [1|(4<<8), 13, 1|(2<<8), 14, 38], 6),
    ("7 ^ 5", [1|(7<<8), 13, 1|(5<<8), 15, 38], 2),
    ("3 << 2", [1|(3<<8), 13, 1|(2<<8), 23, 38], 12),
    ("16 >> 2", [1|(16<<8), 13, 1|(2<<8), 24, 38], 4),
    ("255 & 15", [1|(255<<8), 13, 1|(15<<8), 16, 38], 15),
    ("1 << 4", [1|(1<<8), 13, 1|(4<<8), 23, 38], 16),
]

# Branch tests with correct address calculations (5 bytes per instruction)
BRANCH_TESTS = [
    # if(0) 99 else 42: when ax=0, BZ jumps to else branch
    ("if(0) 99 else 42", [
        1|(0<<8),      # 0: IMM 0
        4|(20<<8),     # 5: BZ 20 (jump to addr 20 if zero)
        1|(99<<8),     # 10: IMM 99 (then branch - skipped)
        2|(25<<8),     # 15: JMP 25 (skip else)
        1|(42<<8),     # 20: IMM 42 (else branch - taken)
        38,            # 25: EXIT
    ], 42),
    # if(1) 42 else 99: when ax=1, BZ does NOT jump
    ("if(1) 42 else 99", [
        1|(1<<8),      # 0: IMM 1
        4|(20<<8),     # 5: BZ 20 (not taken, ax != 0)
        1|(42<<8),     # 10: IMM 42 (then branch - taken)
        2|(25<<8),     # 15: JMP 25 (skip else)
        1|(99<<8),     # 20: IMM 99 (else branch - skipped)
        38,            # 25: EXIT
    ], 42),
    # Nested: if(5 > 3) return 100
    ("if(5>3) 100", [
        1|(5<<8),      # 0: IMM 5
        13,            # 5: PSH
        1|(3<<8),      # 10: IMM 3
        20,            # 15: GT (ax = 5 > 3 = 1)
        4|(30<<8),     # 20: BZ 30 (not taken)
        1|(100<<8),    # 25: IMM 100
        38,            # 30: EXIT
    ], 100),
]

ALL_TESTS = ARITHMETIC_TESTS + COMPARISON_TESTS + BITWISE_TESTS + BRANCH_TESTS


# =============================================================================
# TOKEN SEQUENCE VM TESTS
# =============================================================================

class TestTokenSequenceVM:
    """Test bytecode-in-prompt mode with attention-based fetch."""

    @parametrize("name,bytecode,expected", ARITHMETIC_TESTS)
    def test_arithmetic(self, name, bytecode, expected):
        vm = TokenSequenceVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", COMPARISON_TESTS)
    def test_comparison(self, name, bytecode, expected):
        vm = TokenSequenceVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BITWISE_TESTS)
    def test_bitwise(self, name, bytecode, expected):
        vm = TokenSequenceVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BRANCH_TESTS)
    def test_branch(self, name, bytecode, expected):
        vm = TokenSequenceVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"


# =============================================================================
# RAW BYTECODE VM TESTS
# =============================================================================

class TestRawBytecodeVM:
    """Test raw-bytes-in-prompt mode."""

    @parametrize("name,bytecode,expected", ARITHMETIC_TESTS)
    def test_arithmetic(self, name, bytecode, expected):
        vm = RawBytecodeVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", COMPARISON_TESTS)
    def test_comparison(self, name, bytecode, expected):
        vm = RawBytecodeVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BITWISE_TESTS)
    def test_bitwise(self, name, bytecode, expected):
        vm = RawBytecodeVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BRANCH_TESTS)
    def test_branch(self, name, bytecode, expected):
        vm = RawBytecodeVM()
        vm.load(bytecode)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"


# =============================================================================
# BAKED PROGRAM VM TESTS
# =============================================================================

class TestBakedVM:
    """Test bytecode-in-weights mode with FFN-based fetch."""

    @parametrize("name,bytecode,expected", ARITHMETIC_TESTS)
    def test_arithmetic(self, name, bytecode, expected):
        program = compile_packed_to_baked(bytecode)
        vm = BakedVM(program)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", COMPARISON_TESTS)
    def test_comparison(self, name, bytecode, expected):
        program = compile_packed_to_baked(bytecode)
        vm = BakedVM(program)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BITWISE_TESTS)
    def test_bitwise(self, name, bytecode, expected):
        program = compile_packed_to_baked(bytecode)
        vm = BakedVM(program)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"

    @parametrize("name,bytecode,expected", BRANCH_TESTS)
    def test_branch(self, name, bytecode, expected):
        program = compile_packed_to_baked(bytecode)
        vm = BakedVM(program)
        result = vm.run()
        assert result == expected, f"{name}: got {result}, expected {expected}"


# =============================================================================
# EQUIVALENCE TESTS
# =============================================================================

class TestEquivalence:
    """Verify all VM modes produce identical results."""

    @parametrize("name,bytecode,expected", ALL_TESTS)
    def test_all_modes_equivalent(self, name, bytecode, expected):
        # Token sequence VM
        vm1 = TokenSequenceVM()
        vm1.load(bytecode)
        result1 = vm1.run()

        # Raw bytecode VM
        vm2 = RawBytecodeVM()
        vm2.load(bytecode)
        result2 = vm2.run()

        # Baked program VM
        program = compile_packed_to_baked(bytecode)
        vm3 = BakedVM(program)
        result3 = vm3.run()

        assert result1 == result2 == result3 == expected, \
            f"{name}: TokenSeq={result1}, RawByte={result2}, Baked={result3}, expected={expected}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Quick test without pytest
    print("=" * 70)
    print("COMPREHENSIVE VM MODE TESTS")
    print("=" * 70)

    modes = [
        ("TokenSequenceVM", lambda bc: (TokenSequenceVM(), lambda vm, bc: vm.load(bc))),
        ("RawBytecodeVM", lambda bc: (RawBytecodeVM(), lambda vm, bc: vm.load(bc))),
        ("BakedVM", lambda bc: (BakedVM(compile_packed_to_baked(bc)), lambda vm, bc: None)),
    ]

    for mode_name, make_vm in modes:
        print(f"\n{mode_name}:")
        passed = 0
        failed = 0

        for name, bytecode, expected in ALL_TESTS:
            if mode_name == "BakedVM":
                program = compile_packed_to_baked(bytecode)
                vm = BakedVM(program)
            else:
                vm = make_vm(bytecode)[0]
                vm.load(bytecode)

            result = vm.run()
            if result == expected:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {name} = {result} (expected {expected})")

        print(f"  {passed}/{passed+failed} tests passed")

    # Equivalence check
    print("\n" + "=" * 70)
    print("EQUIVALENCE CHECK")
    print("=" * 70)

    all_equiv = True
    for name, bytecode, expected in ALL_TESTS:
        vm1 = TokenSequenceVM()
        vm1.load(bytecode)
        r1 = vm1.run()

        vm2 = RawBytecodeVM()
        vm2.load(bytecode)
        r2 = vm2.run()

        program = compile_packed_to_baked(bytecode)
        vm3 = BakedVM(program)
        r3 = vm3.run()

        if not (r1 == r2 == r3 == expected):
            all_equiv = False
            print(f"  MISMATCH: {name}")
            print(f"    TokenSeq={r1}, RawByte={r2}, Baked={r3}, expected={expected}")

    if all_equiv:
        print("  All modes produce identical results!")

    print("\n" + "=" * 70)
    total = len(ALL_TESTS) * 3  # 3 modes
    print(f"TOTAL: {total} test executions")
    print("=" * 70)
