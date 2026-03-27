#!/usr/bin/env python3
"""Check if shadow state corrections are being applied."""

from neural_vm.run_vm import AutoregressiveVMRunner, Token
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

# Token name mapping
TOKEN_NAMES = {
    Token.REG_PC: "PC",
    Token.REG_AX: "AX",
    Token.REG_SP: "SP",
    Token.REG_BP: "BP",
    Token.MEM: "MEM",
    Token.STACK0: "STACK0",
}

# Compile simple program
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

# Patch runner to see if corrections are applied
class DebugRunner(AutoregressiveVMRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_count = 0

    def _override_register_in_last_step(self, context, marker_token, value):
        self.correction_count += 1
        reg_name = TOKEN_NAMES.get(marker_token, f"Token({marker_token})")
        print(f"  CORRECTION #{self.correction_count}: {reg_name} = 0x{value:08x} ({value})")
        return super()._override_register_in_last_step(context, marker_token, value)

# Create runner
print("Creating runner with pure_attention_memory=False (default)")
runner = DebugRunner(pure_attention_memory=False)
set_vm_weights(runner.model)
runner.model.compact(block_size=32)
runner.model.compact_moe()

print(f"pure_attention_memory: {runner.pure_attention_memory}\n")

print("Running...\n")
output, exit_code = runner.run(bytecode, data, max_steps=5)

print(f"\nTotal corrections applied: {runner.correction_count}")
print(f"Output: '{output}'")
print(f"Exit code: {exit_code}")
print(f"Expected: 42")
