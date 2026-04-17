"""
Test without compaction to see if that's the issue.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Build model WITHOUT compaction
print("Building model (no compaction)...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Build bytecode
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

print(f"\nGenerating step 0...")
for i in range(10):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_PC:
        print(f"  [{i}] REG_PC")
    elif tok == Token.REG_AX:
        print(f"  [{i}] REG_AX")
        # Print next 4 bytes
        for j in range(1, 5):
            if i + j < 10:
                next_tok = model.generate_next(context)
                context.append(tok)
                print(f"  [{i+j}] byte_{next_tok:02x}")
        break
    elif tok < 256:
        print(f"  [{i}] byte_{tok:02x}")
