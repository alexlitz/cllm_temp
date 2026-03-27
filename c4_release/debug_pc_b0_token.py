"""Debug PC_b0 token and position."""
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
model.eval()

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to PC_b0
current_context = context + [draft_tokens[0], draft_tokens[1]]  # REG_PC + PC_b0

print(f"Context: {current_context}")
print(f"Context length: {len(current_context)}")
print(f"Last 3 tokens: {current_context[-3:]}")
print(f"  Position {len(current_context)-3}: Token {current_context[-3]} (DATA_END)")
print(f"  Position {len(current_context)-2}: Token {current_context[-2]} (REG_PC = {Token.REG_PC})")
print(f"  Position {len(current_context)-1}: Token {current_context[-1]} (PC_b0, should be 10)")

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)

    pc_b0_pos = len(current_context) - 1

    print(f"\n=== EMBEDDING at PC_b0 position (pos {pc_b0_pos}) ===")
    print(f"Token: {current_context[pc_b0_pos]}")
    print(f"IS_BYTE: {x[0, pc_b0_pos, BD.IS_BYTE].item():.3f}")
    print(f"IS_MARK: {x[0, pc_b0_pos, BD.IS_MARK].item():.3f}")
    print(f"MARK_PC: {x[0, pc_b0_pos, BD.MARK_PC].item():.3f}")
    print(f"EMBED_LO[10]: {x[0, pc_b0_pos, BD.EMBED_LO + 10].item():.3f}")
    print(f"EMBED_HI[0]: {x[0, pc_b0_pos, BD.EMBED_HI + 0].item():.3f}")

    # Check Layer 0 threshold heads
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    x_l0 = x + model.blocks[0].attn(x)

    print(f"\n=== AFTER LAYER 0 ATTENTION at PC_b0 position ===")
    print(f"H0[PC]: {x_l0[0, pc_b0_pos, BD.H0 + 0].item():.3f} (should be >0.5)")
    print(f"H1[PC]: {x_l0[0, pc_b0_pos, BD.H1 + 0].item():.3f} (should be >0.5)")
    print(f"H0[AX]: {x_l0[0, pc_b0_pos, BD.H0 + 1].item():.3f}")
    print(f"H1[AX]: {x_l0[0, pc_b0_pos, BD.H1 + 1].item():.3f}")
