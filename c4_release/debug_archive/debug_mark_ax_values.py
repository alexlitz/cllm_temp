"""Check MARK_AX values at different positions."""
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

# Capture state before DivMod
ffn_out = None
def hook(module, input, output):
    global ffn_out
    ffn_out = output.detach().clone()

model.blocks[10].ffn.register_forward_hook(hook)

# Run through tokens 0-21
current_context = context[:]
for i in range(22):
    ctx_tensor = torch.tensor([current_context], dtype=torch.long)
    with torch.no_grad():
        _ = model.forward(ctx_tensor)
    current_context.append(draft_tokens[i])

# Check MARK_AX at each position in the sequence
print("MARK_AX values at each sequence position:")
print("(These are the values that gate DivMod delta)")
print()

S = ffn_out.shape[1]
for pos in range(S):
    mark_ax = ffn_out[0, pos, BD.MARK_AX].item()
    mark_pc = ffn_out[0, pos, BD.MARK_PC].item()

    # Identify position type
    if pos < 10:
        pos_type = f"Context[{pos}]"
    else:
        token_idx = pos - len(context)
        if token_idx == 0:
            pos_type = "REG_PC"
        elif 1 <= token_idx <= 4:
            pos_type = f"PC_b{token_idx-1}"
        elif token_idx == 5:
            pos_type = "REG_AX"
        elif 6 <= token_idx <= 9:
            pos_type = f"AX_b{token_idx-6}"
        else:
            pos_type = f"Token[{token_idx}]"

    if abs(mark_ax) > 0.01 or abs(mark_pc) > 0.01:
        print(f"Pos {pos:2d} ({pos_type:15s}): MARK_AX={mark_ax:6.3f}, MARK_PC={mark_pc:6.3f}")
