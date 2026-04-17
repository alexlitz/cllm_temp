import torch, sys
sys.path.insert(0, ".")
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

print("=== BUG 1: JMP 16 ===")
bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()
print(f"PC should be 16=0x10, bytes=[16,0,0,0]")
print(f"Expected tokens: {tokens[:5]}")
with torch.no_grad():
    ctx = context + tokens[:2]  # up to PC byte 1
    ids = torch.tensor([ctx], dtype=torch.long)
    x = model.embed(ids)
    for i in range(16): x = model.blocks[i](x)
    pos = len(ctx) - 1
    print(f"At PC byte 1 position ({pos}):")
    print(f"  OUTPUT_LO: {[(k,x[0,pos,BD.OUTPUT_LO+k].item()) for k in range(16) if abs(x[0,pos,BD.OUTPUT_LO+k].item())>0.5]}")
    logits = model(ids)
    top3 = torch.topk(logits[0,-1,:], 3)
    print(f"  Top 3: {[(v.item(),i.item()) for v,i in zip(top3.values, top3.indices)]}")

print()
print("=== BUG 2: LEA 8 ===")
bytecode = [Opcode.LEA | (8 << 8)]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()
print(f"AX = BP + 8 = 0x10000 + 8 = 0x10008")
print(f"AX bytes = [8,0,1,0], so AX marker predicts 8, byte0 predicts 0")
print(f"But test says token 1 expected 10, got 8")
print(f"Expected tokens: {tokens[:7]}")
