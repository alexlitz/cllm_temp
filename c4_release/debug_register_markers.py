"""Debug register marker predictions for different opcodes."""
import sys
for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

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
model.compact(block_size=32)
model.compact_moe()
model.eval()

print("Register Marker Predictions for Different Opcodes")
print("=" * 70)
print()

# Test different opcodes
test_cases = [
    ("IMM 0", [Opcode.IMM | (0 << 8)]),
    ("IMM 42", [Opcode.IMM | (42 << 8)]),
    ("JMP 16", [Opcode.JMP | (16 << 8)]),
    ("NOP", [Opcode.NOP]),
]

for name, bytecode in test_cases:
    context = build_context(bytecode)

    draft_vm = DraftVM(bytecode)
    draft_vm.step()
    draft_tokens = draft_vm.draft_tokens()

    ctx_tensor = torch.tensor([context], dtype=torch.long)
    with torch.no_grad():
        logits = model.forward(ctx_tensor)

    predicted = torch.argmax(logits[0, -1]).item()
    expected = draft_tokens[0]

    match = "✓" if predicted == expected else "✗"

    # Identify register
    reg_names = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP", 260: "REG_BP", 268: "STACK0"}
    pred_name = reg_names.get(predicted, f"Token_{predicted}")
    exp_name = reg_names.get(expected, f"Token_{expected}")

    print(f"{name:10s}: predicted={pred_name:10s} ({predicted}), expected={exp_name:10s} ({expected}) {match}")

print()
print("Analysis:")
print("  - All opcodes should predict their first output register")
print("  - IMM: should predict REG_AX (258)")
print("  - JMP: should predict REG_PC (257)")
print("  - NOP: should predict REG_AX (258)")
