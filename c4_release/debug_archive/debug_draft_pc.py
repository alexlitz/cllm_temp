"""Check DraftVM PC value."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

bytecode = [Opcode.LEA | (8 << 8), Opcode.EXIT]

draft = DraftVM(bytecode)

print(f"Before step:")
print(f"  PC: {draft.pc} (0x{draft.pc:08x})")
print(f"  IDX: {draft.idx}")
print()

draft.step()

print(f"After LEA step:")
print(f"  PC: {draft.pc} (0x{draft.pc:08x})")
print(f"  AX: {draft.ax} (0x{draft.ax:08x})")
print(f"  BP: {draft.bp} (0x{draft.bp:08x})")

expected = draft.draft_tokens()
print()
print(f"Expected output tokens:")
for i, tok in enumerate(expected[:10]):
    labels = ["PC_MARKER", "PC_b0", "PC_b1", "PC_b2", "PC_b3", "AX_MARKER", "AX_b0", "AX_b1", "AX_b2", "AX_b3"]
    if i < len(labels):
        print(f"  {i}: {tok:3d} ({labels[i]})")
