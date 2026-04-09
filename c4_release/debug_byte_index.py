#\!/usr/bin/env python3
"""Debug BYTE_INDEX flags at various positions."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim, Token
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

bytecode = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode):
    context = [Token.CODE_START]
    for instr in bytecode:
        op, imm = instr & 0xFF, instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.extend([Token.CODE_END, Token.DATA_START, Token.DATA_END])
    return context

context = build_context(bytecode)

model = AutoregressiveVM(vocab_size=512, n_layers=16, n_heads=8, ffn_hidden=4096)
set_vm_weights(model)
model.eval()
BD = _SetDim

vm = DraftVM(bytecode)
all_tokens = context[:]
for i in range(3):
    vm.step()
    all_tokens.extend(vm.draft_tokens())
vm.step()
step3_tokens = vm.draft_tokens()

print("Step 3 tokens structure:")
print(f"  [0] PC marker: {step3_tokens[0]} (Token.REG_PC={Token.REG_PC})")
print(f"  [1-4] PC bytes: {step3_tokens[1:5]}")
print(f"  [5] AX marker: {step3_tokens[5]} (Token.REG_AX={Token.REG_AX})")
print(f"  [6-9] AX bytes: {step3_tokens[6:10]}")
print(f"  [10] SP marker: {step3_tokens[10]} (Token.REG_SP={Token.REG_SP})")
print(f"  [11-14] SP bytes: {step3_tokens[11:15]}")
print()

activations = {}
def hook(name):
    def fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations[name] = out.detach().clone()
    return fn

for i, block in enumerate(model.blocks):
    block.ffn.register_forward_hook(hook(f'L{i}_ffn'))

# Test at each SP byte position
print("=== BYTE_INDEX values at each SP position ===")
for byte_pos in range(5):  # marker + 4 bytes
    test_ctx = all_tokens + step3_tokens[:10 + byte_pos + 1]  # Include up to this position
    token_ids = torch.tensor([test_ctx], dtype=torch.long)

    with torch.no_grad():
        logits = model(token_ids)

    h = activations['L1_ffn'][0, -1, :]
    bi0 = h[BD.BYTE_INDEX_0].item()
    bi1 = h[BD.BYTE_INDEX_1].item()
    bi2 = h[BD.BYTE_INDEX_2].item()
    bi3 = h[BD.BYTE_INDEX_3].item()
    is_byte = h[BD.IS_BYTE].item()
    is_mark = h[BD.IS_MARK].item()

    token_val = test_ctx[-1]
    pos_name = "SP marker" if byte_pos == 0 else f"SP byte {byte_pos - 1}"

    print(f"{pos_name} (token={token_val}):")
    print(f"  IS_BYTE={is_byte:.2f}, IS_MARK={is_mark:.2f}")
    print(f"  BI_0={bi0:.2f}, BI_1={bi1:.2f}, BI_2={bi2:.2f}, BI_3={bi3:.2f}")

    # Also check after L10
    h10 = activations['L10_ffn'][0, -1, :]
    bi0_10 = h10[BD.BYTE_INDEX_0].item()
    bi1_10 = h10[BD.BYTE_INDEX_1].item()
    bi2_10 = h10[BD.BYTE_INDEX_2].item()
    bi3_10 = h10[BD.BYTE_INDEX_3].item()
    print(f"  After L10: BI_0={bi0_10:.2f}, BI_1={bi1_10:.2f}, BI_2={bi2_10:.2f}, BI_3={bi3_10:.2f}")
    print()
