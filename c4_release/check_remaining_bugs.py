"""Check for remaining bugs in neural predictions."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

print("Checking raw neural predictions (without runner overrides):\n")
print("=" * 70)

bugs = []

# Test IMM - should set AX to immediate value
print("\n1. IMM (immediate load)")
bytecode = [Opcode.IMM | (42 << 8)]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
ax_b0_expected = draft_tokens[6]  # Token 6 is AX byte 0
ax_b0_predicted = logits[0, ctx_len + 5, :].argmax().item()

if ax_b0_expected != ax_b0_predicted:
    print(f"   BUG: AX byte 0 = {ax_b0_predicted}, expected {ax_b0_expected}")
    bugs.append("IMM: AX byte 0 wrong")
else:
    print(f"   ✓ AX byte 0 = {ax_b0_predicted}")

# Test JMP - should jump to target PC
print("\n2. JMP (jump)")
bytecode = [Opcode.JMP | (16 << 8)]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
pc_b0_expected = draft_tokens[1]  # Token 1 is PC byte 0
pc_b0_predicted = logits[0, ctx_len, :].argmax().item()

if pc_b0_expected != pc_b0_predicted:
    print(f"   BUG: PC byte 0 = {pc_b0_predicted}, expected {pc_b0_expected}")
    bugs.append("JMP: PC byte 0 wrong")
else:
    print(f"   ✓ PC byte 0 = {pc_b0_predicted}")

# Test EXIT - should emit HALT token
print("\n3. EXIT (halt)")
bytecode = [Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

ctx_tensor = torch.tensor([context + draft_tokens], dtype=torch.long)
with torch.no_grad():
    logits = model.forward(ctx_tensor)

ctx_len = len(context)
end_expected = draft_tokens[34]  # Token 34 is END
end_predicted = logits[0, ctx_len + 33, :].argmax().item()

if end_expected != end_predicted:
    print(f"   BUG: END token = {end_predicted}, expected {end_expected}")
    bugs.append("EXIT: END token wrong")
else:
    print(f"   ✓ END token = {end_predicted}")

print("\n" + "=" * 70)
print("\nSUMMARY:")
if bugs:
    print(f"Found {len(bugs)} raw neural prediction bugs:")
    for bug in bugs:
        print(f"  - {bug}")
    print("\nNOTE: These bugs are worked around by runner overrides in AutoregressiveVMRunner.")
    print("The test suite passes because it uses runner corrections.")
else:
    print("✓ No bugs found in raw neural predictions!")
