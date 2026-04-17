"""Test IMM 42 with strict validation (no cache)."""
from neural_vm.embedding import Opcode
from neural_vm.batch_runner_v2 import run_batch_ultra

# Test IMM 42
bytecode = [[Opcode.IMM | (42 << 8), Opcode.EXIT]]
result = run_batch_ultra(bytecode, batch_size=1, device='cpu')
print(f"Result: {result}")
print(f"Expected: [42]")
assert result == [42], f"Expected [42], got {result}"
print("PASS!")
