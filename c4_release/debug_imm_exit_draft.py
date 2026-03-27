"""Debug what DraftVM produces for IMM 42; EXIT."""
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

vm = DraftVM(bytecode)
print(f"Bytecode: IMM 42, EXIT")
print(f"Before step:")
print(f"  PC: {vm.pc}")
print(f"  AX: {vm.ax}")

vm.step()
print(f"\nAfter IMM step:")
print(f"  PC: {vm.pc}")
print(f"  AX: {vm.ax}")
print(f"  Halted: {vm.halted}")

draft1 = vm.draft_tokens()
print(f"\nDraft tokens after IMM (first 10):")
for i in range(10):
    print(f"  Token {i}: {draft1[i]}")

# Second step
vm.step()
print(f"\nAfter EXIT step:")
print(f"  PC: {vm.pc}")
print(f"  AX: {vm.ax}")
print(f"  Halted: {vm.halted}")

draft2 = vm.draft_tokens()
print(f"\nDraft tokens after EXIT (first 10):")
for i in range(10):
    print(f"  Token {i}: {draft2[i]}")
