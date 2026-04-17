"""Test DraftVM PC calculation."""
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH, idx_to_pc

bytecode = [Opcode.IMM | (42 << 8)]
print(f"Bytecode: {[hex(b) for b in bytecode]}")
print(f"PC_OFFSET = {PC_OFFSET}")
print(f"INSTR_WIDTH = {INSTR_WIDTH}")
print(f"idx_to_pc(0) = {idx_to_pc(0)}")
print(f"idx_to_pc(1) = {idx_to_pc(1)}")

draft_vm = DraftVM(bytecode)
print(f"\nBefore step:")
print(f"  idx={draft_vm.idx}, pc={draft_vm.pc}, ax={draft_vm.ax}")

draft_vm.step()
print(f"\nAfter step:")
print(f"  idx={draft_vm.idx}, pc={draft_vm.pc}, ax={draft_vm.ax}")

draft_tokens = draft_vm.draft_tokens()
print(f"\nDraft tokens (first 10): {draft_tokens[:10]}")
print(f"PC bytes: [{draft_tokens[1]}, {draft_tokens[2]}, {draft_tokens[3]}, {draft_tokens[4]}]")
print(f"Expected PC after IMM: {idx_to_pc(1)} = {hex(idx_to_pc(1))}")
