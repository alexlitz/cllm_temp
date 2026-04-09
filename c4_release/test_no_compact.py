"""
Test IMM+EXIT without model compaction.
"""

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

print("Building model WITHOUT compaction...")
model = AutoregressiveVM()
set_vm_weights(model)
# DON'T compact
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

print("Generating step 1...")
for i in range(20):
    tok = model.generate_next(context)
    context.append(tok)
    if i == 6:  # First AX byte
        print(f"First AX byte: 0x{tok:02x} (expected 0x2a)")
    if tok == Token.HALT:
        print(f"  HALT found!")
        break

# Extract exit code
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_AX:
        ax_bytes = context[i+1:i+5]
        exit_code = ax_bytes[0] if len(ax_bytes) >= 1 else None
        print(f"Exit code: 0x{exit_code:02x} (expected 0x2a)")
        break
