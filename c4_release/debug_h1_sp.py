"""Debug when H1[SP] gets set."""
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

model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + draft_tokens[:13]  # Up to SP_b1

print(f"Tokens: {current_context[-5:]}")
print(f"  REG_AX at pos {len(current_context)-5}: {current_context[-5]}")
print(f"  AX_b2 at pos {len(current_context)-4}: {current_context[-4]}")
print(f"  AX_b3 at pos {len(current_context)-3}: {current_context[-3]}")
print(f"  REG_SP at pos {len(current_context)-2}: {current_context[-2]}")
print(f"  SP_b0 at pos {len(current_context)-1}: {current_context[-1]}")

ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    sp_b0_pos = len(current_context) - 1

    print(f"\n=== Tracing H1[SP] at SP_b0 position (pos {sp_b0_pos}) ===")
    for layer_idx in range(3):
        x = model.blocks[layer_idx](x)
        h1_sp = x[0, sp_b0_pos, BD.H1 + 2].item()
        print(f"After Layer {layer_idx}: H1[SP] = {h1_sp:.3f}")
