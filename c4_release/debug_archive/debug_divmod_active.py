"""Find which DivMod units are activating at PC marker."""
import torch
import torch.nn.functional as F
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

x = ffn_out[0, pc_marker_pos, :]

# Manually compute DivMod forward pass
divmod = model.blocks[10].post_ops[0]
up = x @ divmod.W_up.t() + divmod.b_up
gate = x @ divmod.W_gate.t() + divmod.b_gate
hidden = F.silu(up) * gate

print(f"=== TOP 10 ACTIVE DIVMOD UNITS ===")
top_indices = torch.topk(hidden, k=10).indices
for i, idx in enumerate(top_indices):
    # Decode unit index to (a, b, op)
    if idx < 256 * 256:  # DIV units
        a = idx // 256
        b = idx % 256
        op = "DIV"
        result = a // b if b > 0 else 0
    else:  # MOD units
        idx_mod = idx - 256 * 256
        a = idx_mod // 256
        b = idx_mod % 256
        op = "MOD"
        result = a % b if b > 0 else 0

    print(f"Unit {idx.item():6d}: {op}({a:3d}, {b:3d}) = {result:3d}, hidden={hidden[idx].item():.3f}, up={up[idx].item():.3f}")

print(f"\n=== VALUES AT PC MARKER ===")
print(f"MARK_AX = {x[BD.MARK_AX].item():.3f}")
print(f"OP_DIV = {x[BD.OP_DIV].item():.3f}")
print(f"OP_MOD = {x[BD.OP_MOD].item():.3f}")
