"""
Test single-operation execution with compiled weights.

This is the minimal end-to-end test - can we execute ONE operation correctly?
"""

from neural_vm.single_op_executor import SingleOperationExecutor
from neural_vm.embedding import Opcode

def test_add_operation():
    """Test ADD operation end-to-end."""
    
    print("="*70)
    print("SINGLE OPERATION EXECUTION TEST: ADD")
    print("="*70)
    print()
    
    print("Initializing executor...")
    executor = SingleOperationExecutor()
    print("  ✅ VM created")
    print("  ✅ Weights loaded")
    print("  ✅ Embedding ready")
    print()
    
    print("Running ADD tests...")
    print("-" * 70)
    
    test_cases = [
        (2, 3, 5),
        (10, 20, 30),
        (100, 50, 150),
        (7, 8, 15),
        (0, 0, 0),
        (1, 1, 2),
        (255, 1, 256),
    ]
    
    results = executor.test_operation(
        opcode=Opcode.ADD,
        op_name="ADD",
        test_cases=test_cases
    )
    
    # Print results
    for a, b, expected, result, status in results['cases']:
        if status == 'PASS':
            print(f"  ✅ {a} + {b} = {result} (expected {expected})")
        elif status == 'FAIL':
            print(f"  ❌ {a} + {b} = {result} (expected {expected})")
        else:
            print(f"  ⚠️  {a} + {b}: {status}")
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Passed: {results['passed']}/{len(test_cases)}")
    print(f"  Failed: {results['failed']}/{len(test_cases)}")
    print(f"  Errors: {results['errors']}/{len(test_cases)}")
    print()
    
    if results['passed'] == len(test_cases):
        print("✅ ALL TESTS PASSED! Single-operation execution works!")
    else:
        print("❌ Some tests failed. Debugging needed.")
    
    print("="*70)
    
    return results['passed'] == len(test_cases)

if __name__ == "__main__":
    success = test_add_operation()
    exit(0 if success else 1)
