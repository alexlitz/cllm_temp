#!/usr/bin/env python3
"""Debug SP byte positions to understand hop-count matching."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from src.compiler import compile_c
from pathlib import Path
import tempfile

code = """
int main() {
    int x;
    x = 42;
    return x;
}
"""

print("=" * 70)
print("SP Byte Position Debug")
print("=" * 70)

# Compile
with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
    f.write(code)
    f.flush()
    temp_path = f.name

bytecode, data = compile_c(Path(temp_path).read_text())
Path(temp_path).unlink()

# Create runner
runner = AutoregressiveVMRunner()

# Patch to inspect context after first PSH
original_step = runner._run_one_step
step_count = [0]

def debug_step(context):
    result = original_step(context)

    if step_count[0] == 1:  # After first PSH
        print(f"\n[STEP 1] Context length: {len(context)}")
        print("Looking for SP section...\n")

        # Find SP marker
        for i, tok in enumerate(context):
            if tok == Token.REG_SP:
                print(f"SP marker at position {i}")
                print(f"  SP marker token: {tok}")

                # Print next 4 tokens (SP bytes)
                for d in range(1, 5):
                    if i + d < len(context):
                        byte_tok = context[i + d]
                        print(f"  d={d} (pos {i+d}): byte=0x{byte_tok:02x}")

                # Check what address we expect
                # After ENT, SP should be some value
                # Then after PSH, SP -= 8
                print("\nExpected: SP bytes should represent (initial_SP - 8)")
                break

    step_count[0] += 1
    return result

runner._run_one_step = debug_step

print("\nRunning program...\n")
result = runner.run(bytecode, data, [], max_steps=5)
print(f"\nResult: {result}")
