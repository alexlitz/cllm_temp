#!/usr/bin/env python3
"""Debug token generation."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

# Compile simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

print(f"Bytecode: {bytecode}\n")

# Patch runner to capture tokens
class DebugRunner(AutoregressiveVMRunner):
    def run(self, bytecode, data=b"", argv=None, stdin="", max_steps=100000, tool_handler=None):
        # Call parent
        self._token_count = 0
        self._step_count = 0
        self._halt_found = False

        # Capture original generate_next
        original_generate = self.model.generate_next

        def debug_generate(context):
            token = original_generate(context)
            self._token_count += 1

            if token == Token.STEP_END:
                self._step_count += 1
                if self._step_count <= 5:  # Print first 5 steps
                    print(f"Step {self._step_count}: {self._token_count} tokens generated")

            if token == Token.HALT:
                self._halt_found = True
                print(f"HALT at token {self._token_count}, step {self._step_count}")

            return token

        self.model.generate_next = debug_generate

        try:
            result = super().run(bytecode, data, argv, stdin, max_steps, tool_handler)
        finally:
            self.model.generate_next = original_generate

        print(f"\nTotal tokens: {self._token_count}")
        print(f"Total steps: {self._step_count}")
        print(f"HALT found: {self._halt_found}")

        return result

# Create runner
runner = DebugRunner()
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

# Run
print("Running...\n")
output, exit_code = runner.run(bytecode, data, max_steps=20)

print(f"\nOutput: '{output}'")
print(f"Exit code: {exit_code}")
