"""Check H1 values at context positions."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
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

# Test IMM 5
bytecode = [Opcode.IMM | (5 << 8), Opcode.EXIT]
draft_vm = DraftVM(bytecode)
context = build_context(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Load model
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Hook L9
l9_out = {}
def hook(m, i, o):
    l9_out['x'] = o.clone()
model.blocks[9].register_forward_hook(hook)

context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

x = l9_out['x'][0, :, :]
print("H1 values at context positions:")
print("=" * 70)

print("\nContext structure:")
for i, tok in enumerate(context):
    h1_vals = x[i, BD.H1:BD.H1+8]
    h1_ax = h1_vals[1].item()

    # Decode token
    if tok == Token.CODE_START:
        desc = "CODE_START"
    elif tok == Token.CODE_END:
        desc = "CODE_END"
    elif tok == Token.DATA_START:
        desc = "DATA_START"
    elif tok == Token.DATA_END:
        desc = "DATA_END"
    elif tok < 256:
        desc = f"byte {tok}"
    else:
        desc = f"special {tok}"

    print(f"  pos {i:2d}: tok={tok:3d} ({desc:15s})  H1[AX_IDX]={h1_ax:.3f}")

print()
print("Key finding:")
print("-" * 70)
print(f"Position 2 (immediate byte 0 = 5):")
print(f"  H1[AX_IDX] = {x[2, BD.H1 + 1].item():.6f}")
print()
print("If H1[AX_IDX] != 0 at context positions, the anti-leakage won't work!")
