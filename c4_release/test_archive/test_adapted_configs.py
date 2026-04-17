"""Test that adapted configurations work with hand-crafted weights."""

import torch
from neural_vm.embedding import Opcode
from neural_vm.batch_runner_v2 import UltraBatchRunner

def test_config(use_softmax1, pos_encoding):
    """Test a simple program with specific configuration."""
    print(f"\nTesting: use_softmax1={use_softmax1}, pos_encoding='{pos_encoding}'")

    # Simple test programs
    test_programs = [
        # IMM 42; EXIT
        [Opcode.IMM | (42 << 8), Opcode.EXIT],
        # IMM 10; PSH; IMM 5; ADD; EXIT
        [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (5 << 8), Opcode.ADD, Opcode.EXIT],
        # IMM 6; PSH; IMM 7; MUL; EXIT
        [Opcode.IMM | (6 << 8), Opcode.PSH, Opcode.IMM | (7 << 8), Opcode.MUL, Opcode.EXIT],
    ]

    expected_results = [42, 15, 42]

    try:
        # Create runner with this configuration
        runner = UltraBatchRunner(
            batch_size=len(test_programs),
            device='cpu',  # Use CPU for consistency
            use_softmax1=use_softmax1,
            pos_encoding=pos_encoding
        )

        # Run programs
        results = runner.run_batch(test_programs)

        # Check results
        all_correct = True
        for i, (result, expected) in enumerate(zip(results, expected_results)):
            if result == expected:
                print(f"  ✓ Program {i}: {result} == {expected}")
            else:
                print(f"  ✗ Program {i}: {result} != {expected}")
                all_correct = False

        if all_correct:
            print(f"  ✅ Configuration works correctly!")
            return True
        else:
            print(f"  ❌ Configuration produces incorrect results")
            return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*70)
    print("Testing Adapted Configurations with Hand-Crafted Weights")
    print("="*70)

    configs = [
        (True, 'alibi'),   # Default - should work
        (False, 'alibi'),  # F.softmax with ZFOD adaptation
        (True, 'rope'),    # RoPE with recency bias adaptation
        (False, 'rope'),   # Both adaptations
        (True, 'none'),    # No pos encoding (just softmax1 ZFOD)
        (False, 'none'),   # Just null key adaptation
    ]

    results = {}
    for use_softmax1, pos_encoding in configs:
        config_name = f"softmax1={use_softmax1}, pos_encoding='{pos_encoding}'"
        success = test_config(use_softmax1, pos_encoding)
        results[config_name] = success

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)

    working = [name for name, success in results.items() if success]
    not_working = [name for name, success in results.items() if not success]

    if working:
        print("\n✅ WORKING CONFIGURATIONS:")
        for config in working:
            print(f"  - {config}")

    if not_working:
        print("\n❌ NOT WORKING CONFIGURATIONS:")
        for config in not_working:
            print(f"  - {config}")

    print("\n" + "="*70)
    if len(working) == len(configs):
        print("🎉 SUCCESS: All configurations work with hand-crafted weights!")
    elif len(working) > 1:
        print(f"⚠️  PARTIAL: {len(working)}/{len(configs)} configurations work")
    else:
        print(f"❌ FAILURE: Only default configuration works")
    print("="*70)
