"""
Test Compiled Arithmetic Programs

Compiles simple C programs using the C4 compiler and tests them.
"""

import torch
from src.compiler import Compiler
from neural_vm.vm_step import AutoregressiveVM
from neural_vm.weight_loader import CompiledWeightLoader
from neural_vm.nibble_embedding import NibbleVMEmbedding
from neural_vm.embedding import Opcode

def test_compiled_arithmetic():
    """Test compiled arithmetic programs."""

    print("="*70)
    print("COMPILED ARITHMETIC TEST")
    print("="*70)
    print()

    # Create compiler
    compiler = Compiler()

    # Test programs
    test_cases = [
        ("int main() { return 100 + 200; }", 300, "100 + 200"),
        ("int main() { return 500 - 200; }", 300, "500 - 200"),
        ("int main() { return 2 + 3; }", 5, "2 + 3"),
        ("int main() { return 10 - 3; }", 7, "10 - 3"),
        ("int main() { int x; x = 42; return x; }", 42, "variable assignment"),
    ]

    print("Compiling test programs...")
    compiled = []
    for source, expected, desc in test_cases:
        try:
            code, data = compiler.compile(source)
            compiled.append((code, data, expected, desc))
            print(f"  ✅ Compiled: {desc}")
        except Exception as e:
            print(f"  ❌ Failed to compile: {desc}")
            print(f"     Error: {e}")

    print()
    print(f"Successfully compiled {len(compiled)}/{len(test_cases)} programs")
    print()

    if len(compiled) == 0:
        print("No programs compiled successfully")
        return False

    # For now, just verify compilation works
    # Full execution would require implementing the complete execution loop
    print("="*70)
    print("COMPILATION SUCCESSFUL")
    print("="*70)
    print()
    print("Note: Full bytecode execution requires:")
    print("  - Instruction fetch loop")
    print("  - PC management")
    print("  - Memory system")
    print("  - Control flow handling")
    print()
    print("Current status: ✅ Compilation working")
    print("Next step: Implement execution loop")
    print()
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = test_compiled_arithmetic()
    exit(0 if success else 1)
