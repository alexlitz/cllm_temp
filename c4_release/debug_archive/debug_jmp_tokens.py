"""Debug JMP tokens."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

bytecode = [Opcode.JMP | (8 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"JMP 8 draft tokens (first 10): {draft_tokens[:10]}")
print()

bytecode = [Opcode.JMP | (16 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"JMP 16 draft tokens (first 10): {draft_tokens[:10]}")
