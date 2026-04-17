"""Debug DivMod activation at PC marker."""
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

bytecode = [Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + [draft_tokens[0]]
ctx_tensor = torch.tensor([current_context], dtype=torch.long)
pc_marker_pos = len(current_context) - 1

# Capture FFN output
ffn_out = None
def hook(module, input, output):
    global ffn_out
    ffn_out = output.detach().clone()

model.blocks[10].ffn.register_forward_hook(hook)

with torch.no_grad():
    _ = model.forward(ctx_tensor)

print(f"=== VALUES AT PC MARKER (position {pc_marker_pos}) ===")
x = ffn_out[0, pc_marker_pos, :]

print(f"MARK_AX = {x[BD.MARK_AX].item():.3f}")
print(f"OP_DIV = {x[BD.OP_DIV].item():.3f}")
print(f"OP_MOD = {x[BD.OP_MOD].item():.3f}")
print(f"ALU_LO[0] = {x[BD.ALU_LO + 0].item():.3f}")
print(f"ALU_HI[0] = {x[BD.ALU_HI + 0].item():.3f}")
print(f"AX_CARRY_LO[0] = {x[BD.AX_CARRY_LO + 0].item():.3f}")
print(f"AX_CARRY_HI[0] = {x[BD.AX_CARRY_HI + 0].item():.3f}")

# Check first DivMod unit (a=0, b=1, q=0)
divmod = model.blocks[10].post_ops[0]
up_input = x @ divmod.W_up.t() + divmod.b_up
print(f"\n=== DIVMOD UNIT 1 (a=0, b=1, q=0) ===")
print(f"up_input[0] = {up_input[0].item():.3f}")
print(f"Expected: should be < 0 to block activation")
print(f"  Calculation: MARK_AX*100 + ALU_LO[0]*100 + ALU_HI[0]*100 + AX_CARRY_LO[1]*100 + AX_CARRY_HI[0]*100 + OP_DIV*100 - 550")
print(f"  = {x[BD.MARK_AX].item():.1f}*100 + ... - 550")
