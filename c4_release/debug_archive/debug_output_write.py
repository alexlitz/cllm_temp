"""Trace which layer writes OUTPUT at PC_b0 position."""
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

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to PC_b0
current_context = context + [draft_tokens[0], draft_tokens[1]]  # REG_PC + PC_b0

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_b0_pos = len(current_context) - 1

    print(f"Tracing OUTPUT_LO[10] at PC_b0 position (pos {pc_b0_pos}):")
    print(f"Token: {current_context[pc_b0_pos]} (should be 10)\n")

    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

        out_lo_10 = x[0, pc_b0_pos, BD.OUTPUT_LO + 10].item()
        out_hi_0 = x[0, pc_b0_pos, BD.OUTPUT_HI + 0].item()

        if abs(out_lo_10) > 0.1 or abs(out_hi_0) > 0.1:
            print(f"Layer {layer_idx:2d}: OUTPUT_LO[10]={out_lo_10:6.3f}, OUTPUT_HI[0]={out_hi_0:6.3f}")
            if layer_idx > 0 and abs(out_lo_10) > 0.5 and abs(x_prev[0, pc_b0_pos, BD.OUTPUT_LO + 10].item()) < 0.5:
                print(f"  ^^^ OUTPUT_LO[10] first appeared after Layer {layer_idx}")

        x_prev = x.clone()
