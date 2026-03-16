"""
Bootstrap test: Run C4 compiler inside the transformer VM.

Pipeline:
1. Use native xc to compile a simple C program to bytecode
2. Run that bytecode in our transformer VM
3. Verify result
"""

import subprocess
import os
import torch
from c4_moe_vm import C4MoEVM, C4Op


def get_bytecode_from_xc(source_code):
    """Use xc_dump to compile source and extract bytecode."""
    # Write source to temp file
    with open("/tmp/test_prog.c", "w") as f:
        f.write(source_code)

    # Check if xc_dump exists
    if not os.path.exists("/tmp/xc_dump"):
        print("  ERROR: /tmp/xc_dump not found")
        return None, None

    # Run xc_dump to get bytecode
    result = subprocess.run(
        ["/tmp/xc_dump", "/tmp/test_prog.c"],
        capture_output=True, text=True, timeout=5
    )

    return result.stdout, result.returncode


def parse_xc_bytecode(output):
    """Parse xc_dump output to extract bytecode.

    xc_dump outputs:
    === BYTECODE ===
    6
    0
    1
    6
    ...
    === END ===
    MAIN:0

    Bytecode is a stream where some ops have immediates following.
    We need to pack them for our VM (op | imm << 8).
    """
    # Opcodes that take an immediate argument
    OPS_WITH_IMM = {0, 1, 2, 3, 4, 5, 6, 7}  # LEA, IMM, JMP, JSR, JZ/BZ, JNZ/BNZ, ENT, ADJ

    lines = output.strip().split('\n')

    # Find bytecode section
    in_bytecode = False
    raw_bytecode = []
    main_offset = 0

    for line in lines:
        line = line.strip()
        if line == "=== BYTECODE ===":
            in_bytecode = True
            continue
        if line == "=== END ===":
            in_bytecode = False
            continue
        if line.startswith("MAIN:"):
            main_offset = int(line.replace("MAIN:", ""))
            continue
        if in_bytecode and line:
            try:
                raw_bytecode.append(int(line))
            except:
                pass

    # Pack bytecode: combine op with following immediate
    packed = []
    i = 0
    while i < len(raw_bytecode):
        op = raw_bytecode[i]
        if op in OPS_WITH_IMM and i + 1 < len(raw_bytecode):
            imm = raw_bytecode[i + 1]
            packed.append(op | (imm << 8))
            i += 2
        else:
            packed.append(op)
            i += 1

    return packed, main_offset


def test_basic():
    """Test basic programs through the pipeline."""
    print("=" * 60)
    print("TESTING: C4 -> Bytecode -> Transformer VM")
    print("=" * 60)

    tests = [
        ("return 42", "int main() { return 42; }", 42),
        ("6 * 7", "int main() { return 6 * 7; }", 42),
        ("100 / 5", "int main() { return 100 / 5; }", 20),
        ("3 + 4", "int main() { return 3 + 4; }", 7),
        ("(3+4)*5", "int main() { return (3 + 4) * 5; }", 35),
        ("100/5+10*3", "int main() { return 100/5 + 10*3; }", 50),
        ("17 % 5", "int main() { return 17 % 5; }", 2),
        ("10 - 3", "int main() { return 10 - 3; }", 7),
        ("2 << 3", "int main() { return 2 << 3; }", 16),
        ("16 >> 2", "int main() { return 16 >> 2; }", 4),
    ]

    vm = C4MoEVM()
    passed = 0

    for name, source, expected in tests:
        print(f"\nTest: {name}")
        print(f"  Source: {source}")

        output_str, ret = get_bytecode_from_xc(source)
        if output_str is None:
            print("  SKIP - couldn't compile")
            continue

        # Parse to bytecode
        bytecode, main_offset = parse_xc_bytecode(output_str)
        if not bytecode:
            print("  SKIP - no bytecode extracted")
            continue

        print(f"  Bytecode: {bytecode}")
        print(f"  Main offset: {main_offset}")

        # Replace FIRST LEV with EXIT for main return (LEV = 8, EXIT = 38)
        # Since we start at main, the first LEV is the return from main
        for i in range(len(bytecode)):
            if (bytecode[i] & 0xFF) == 8:  # LEV
                bytecode[i] = 38  # EXIT
                break

        print(f"  Patched bytecode: {bytecode}")

        # Run in VM
        vm.reset()
        vm.load(bytecode, [])
        vm.pc = torch.tensor(float(main_offset * 8))

        result, vm_output, stats = vm.run(max_steps=10000)

        status = "✓ PASS" if result == expected else f"✗ FAIL (got {result})"
        print(f"  Result: {result} (expected {expected}) {status}")

        if result == expected:
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_manual_bytecode():
    """Test with manually constructed bytecode."""
    print("\n" + "=" * 60)
    print("TESTING: Manual Bytecode in Transformer VM")
    print("=" * 60)

    vm = C4MoEVM()

    # Simple test: return 6 * 7
    # ENT 0, IMM 6, PSH, IMM 7, MUL, EXIT
    bytecode = [
        C4Op.ENT | (0 << 8),   # ENT 0
        C4Op.IMM | (6 << 8),   # IMM 6
        C4Op.PSH,              # PSH
        C4Op.IMM | (7 << 8),   # IMM 7
        C4Op.MUL,              # MUL
        C4Op.EXIT,             # EXIT
    ]

    vm.reset()
    vm.load(bytecode, [])
    result, _, _ = vm.run()

    print(f"  6 * 7 = {result} (expected 42)")
    print(f"  Status: {'✓ PASS' if result == 42 else '✗ FAIL'}")

    # Test division
    bytecode = [
        C4Op.ENT | (0 << 8),
        C4Op.IMM | (100 << 8),
        C4Op.PSH,
        C4Op.IMM | (5 << 8),
        C4Op.DIV,
        C4Op.EXIT,
    ]

    vm.reset()
    vm.load(bytecode, [])
    result, _, _ = vm.run()

    print(f"  100 / 5 = {result} (expected 20)")
    print(f"  Status: {'✓ PASS' if result == 20 else '✗ FAIL'}")

    return True


if __name__ == "__main__":
    test_manual_bytecode()
    test_basic()
