"""Test multiple configurations with the full test suite."""

import torch
from neural_vm.embedding import Opcode
from neural_vm.batch_runner_v2 import UltraBatchRunner

def run_test_suite_with_config(use_softmax1, pos_encoding):
    """Run a subset of the test suite with a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: use_softmax1={use_softmax1}, pos_encoding='{pos_encoding}'")
    print(f"{'='*70}")

    # Test programs covering different opcodes
    test_cases = [
        # IMM tests
        ([Opcode.IMM | (0 << 8), Opcode.EXIT], 0, "IMM 0"),
        ([Opcode.IMM | (42 << 8), Opcode.EXIT], 42, "IMM 42"),
        ([Opcode.IMM | (255 << 8), Opcode.EXIT], 255, "IMM 255"),

        # ADD tests
        ([Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.ADD, Opcode.EXIT], 0, "ADD 0+0"),
        ([Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (1 << 8), Opcode.ADD, Opcode.EXIT], 2, "ADD 1+1"),
        ([Opcode.IMM | (100 << 8), Opcode.PSH, Opcode.IMM | (50 << 8), Opcode.ADD, Opcode.EXIT], 150, "ADD 100+50"),

        # SUB tests
        ([Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.SUB, Opcode.EXIT], 7, "SUB 10-3"),
        ([Opcode.IMM | (100 << 8), Opcode.PSH, Opcode.IMM | (100 << 8), Opcode.SUB, Opcode.EXIT], 0, "SUB 100-100"),

        # MUL tests
        ([Opcode.IMM | (0 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT], 0, "MUL 0*0"),
        ([Opcode.IMM | (2 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.MUL, Opcode.EXIT], 6, "MUL 2*3"),
        ([Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.MUL, Opcode.EXIT], 25, "MUL 5*5"),
        ([Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (10 << 8), Opcode.MUL, Opcode.EXIT], 100, "MUL 10*10"),

        # DIV tests
        ([Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.DIV, Opcode.EXIT], 3, "DIV 10/3"),
        ([Opcode.IMM | (100 << 8), Opcode.PSH, Opcode.IMM | (10 << 8), Opcode.DIV, Opcode.EXIT], 10, "DIV 100/10"),
        ([Opcode.IMM | (42 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.DIV, Opcode.EXIT], 0, "DIV 42/0"),

        # MOD tests
        ([Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.MOD, Opcode.EXIT], 1, "MOD 10%3"),
        ([Opcode.IMM | (255 << 8), Opcode.PSH, Opcode.IMM | (16 << 8), Opcode.MOD, Opcode.EXIT], 15, "MOD 255%16"),

        # Bitwise tests
        ([Opcode.IMM | (255 << 8), Opcode.PSH, Opcode.IMM | (15 << 8), Opcode.AND, Opcode.EXIT], 15, "AND 255&15"),
        ([Opcode.IMM | (240 << 8), Opcode.PSH, Opcode.IMM | (15 << 8), Opcode.OR, Opcode.EXIT], 255, "OR 240|15"),
        ([Opcode.IMM | (255 << 8), Opcode.PSH, Opcode.IMM | (255 << 8), Opcode.XOR, Opcode.EXIT], 0, "XOR 255^255"),

        # Comparison tests
        ([Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.EQ, Opcode.EXIT], 1, "EQ 5==5"),
        ([Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.NE, Opcode.EXIT], 1, "NE 5!=3"),
        ([Opcode.IMM | (3 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.LT, Opcode.EXIT], 1, "LT 3<5"),
        ([Opcode.IMM | (5 << 8), Opcode.PSH, Opcode.IMM | (3 << 8), Opcode.GT, Opcode.EXIT], 1, "GT 5>3"),
    ]

    try:
        # Create runner with this configuration
        runner = UltraBatchRunner(
            batch_size=len(test_cases),
            device='cpu',  # Use CPU for consistency across configs
            use_softmax1=use_softmax1,
            pos_encoding=pos_encoding
        )

        # Extract bytecodes and expected results
        bytecodes = [bc for bc, _, _ in test_cases]
        expected = [exp for _, exp, _ in test_cases]
        names = [name for _, _, name in test_cases]

        # Run all tests
        results = runner.run_batch(bytecodes)

        # Check results
        passed = 0
        failed = 0
        for i, (result, exp, name) in enumerate(zip(results, expected, names)):
            if result == exp:
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {name}: got {result}, expected {exp}")

        print(f"\nResults: {passed}/{len(test_cases)} passed")

        if failed == 0:
            print(f"✅ All tests passed!")
            return True
        else:
            print(f"❌ {failed} tests failed")
            return False

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*70)
    print("Testing All Configurations with Representative Test Suite")
    print("="*70)

    configs = [
        (True, 'alibi'),   # Default
        (False, 'alibi'),  # F.softmax with ZFOD
        (True, 'rope'),    # RoPE with recency
        (False, 'rope'),   # Both adaptations
    ]

    results = {}
    for use_softmax1, pos_encoding in configs:
        config_name = f"softmax1={use_softmax1}, pos_encoding='{pos_encoding}'"
        success = run_test_suite_with_config(use_softmax1, pos_encoding)
        results[config_name] = success

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    all_passed = all(results.values())

    for config_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {config_name}")

    print("\n" + "="*70)
    if all_passed:
        print("🎉 SUCCESS: All configurations pass the test suite!")
        print("The adaptations (ZFOD for F.softmax, recency for RoPE) work correctly!")
    else:
        print("⚠️  Some configurations failed")
    print("="*70)
