"""
Test C → C4 bytecode → Transformer execution.
"""

import torch
from c4_vm import Op
from c4_compiler import compile_c
from c4_printf_autoregressive import C4AutoregressivePrintf


def run_c_program(source: str, max_steps=200) -> tuple:
    """Compile and run C program, return (result, output)."""
    # Compile
    bytecode = compile_c(source)

    # Load into memory
    executor = C4AutoregressivePrintf(memory_size=512)
    memory = torch.zeros(512)
    for i, instr in enumerate(bytecode):
        memory[i * 8] = float(instr)

    # Run
    pc, sp, bp, ax, memory, output = executor.run(
        memory,
        pc=torch.tensor(0.0),
        sp=torch.tensor(460.0),
        bp=torch.tensor(460.0),
        ax=torch.tensor(0.0),
        max_steps=max_steps
    )

    return int(ax.item()), output


def test():
    print("C → TRANSFORMER PIPELINE")
    print("=" * 60)
    print()
    print("C source → compile → bytecode → transformer → result")
    print()

    tests = [
        ("3 + 4", "int main() { return 3 + 4; }", 7),
        ("5 * 6", "int main() { return 5 * 6; }", 30),
        ("10 - 3", "int main() { return 10 - 3; }", 7),
        ("20 / 4", "int main() { return 20 / 4; }", 5),
        ("17 % 5", "int main() { return 17 % 5; }", 2),
        ("(2+3)*4", "int main() { return (2 + 3) * 4; }", 20),
    ]

    for name, source, expected in tests:
        result, _ = run_c_program(source)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {name} = {result} (expected {expected})")

    print()

    # Variables
    print("Variables:")
    result, _ = run_c_program("""
        int main() {
            int x;
            int y;
            x = 10;
            y = 5;
            return x + y;
        }
    """)
    status = "✓" if result == 15 else "✗"
    print(f"  {status} x=10, y=5, x+y = {result}")

    print()

    # If/else
    print("If/else:")
    result1, _ = run_c_program("""
        int main() {
            int x;
            x = 10;
            if (x > 5) { return 1; }
            return 0;
        }
    """)
    result2, _ = run_c_program("""
        int main() {
            int x;
            x = 3;
            if (x > 5) { return 1; }
            return 0;
        }
    """)
    status1 = "✓" if result1 == 1 else "✗"
    status2 = "✓" if result2 == 0 else "✗"
    print(f"  {status1} if (10 > 5) = {result1}")
    print(f"  {status2} if (3 > 5) = {result2}")

    print()

    # While loop
    print("While loop:")
    result, _ = run_c_program("""
        int main() {
            int sum;
            int i;
            sum = 0;
            i = 1;
            while (i <= 5) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
    """, max_steps=500)
    status = "✓" if result == 15 else "✗"
    print(f"  {status} sum(1..5) = {result}")

    print()

    # Printf
    print("Printf:")
    result, output = run_c_program("""
        int main() {
            printf(42);
            return 0;
        }
    """)
    print(f"  Output: '{output.strip()}'")
    status = "✓" if "42" in output else "✗"
    print(f"  {status} printf(42)")

    print()
    print("=" * 60)
    print("C PROGRAMS RUN ON TRANSFORMER!")


if __name__ == "__main__":
    test()
