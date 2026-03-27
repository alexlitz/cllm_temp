"""Debug what DraftVM produces for IMM 42."""
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

bytecode = [Opcode.IMM | (42 << 8)]

vm = DraftVM(bytecode)
print(f"Before step:")
print(f"  PC: {vm.pc}")
print(f"  AX: {vm.ax}")

vm.step()
print(f"\nAfter step:")
print(f"  PC: {vm.pc}")
print(f"  AX: {vm.ax}")
print(f"  Halted: {vm.halted}")

draft = vm.draft_tokens()
print(f"\nDraft tokens (first 10):")
for i in range(10):
    print(f"  Token {i}: {draft[i]}")

print(f"\nExpected:")
print(f"  Token 0 (REG_AX): 257")
print(f"  Token 1 (AX byte 0): 42")
print(f"  Token 2 (AX byte 1): 0")
print(f"  Token 3 (AX byte 2): 0")
print(f"  Token 4 (AX byte 3): 0")
