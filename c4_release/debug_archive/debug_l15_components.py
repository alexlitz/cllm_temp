"""Debug Layer 15 attention vs FFN."""
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

    # Run through Layer 14
    for layer_idx in range(15):
        x = model.blocks[layer_idx](x)

    print(f"=== BEFORE LAYER 15 at PC_b0 position ===")
    print(f"OUTPUT_LO[10]: {x[0, pc_b0_pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"OUTPUT_HI[0]: {x[0, pc_b0_pos, BD.OUTPUT_HI + 0].item():.3f}")

    # Layer 15 attention
    x_after_attn = x + model.blocks[15].attn(x)

    print(f"\n=== AFTER LAYER 15 ATTENTION ===")
    print(f"OUTPUT_LO[10]: {x_after_attn[0, pc_b0_pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"OUTPUT_HI[0]: {x_after_attn[0, pc_b0_pos, BD.OUTPUT_HI + 0].item():.3f}")

    # Layer 15 FFN
    x_after_ffn = x_after_attn + model.blocks[15].ffn(x_after_attn)

    print(f"\n=== AFTER LAYER 15 FFN ===")
    print(f"OUTPUT_LO[10]: {x_after_ffn[0, pc_b0_pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"OUTPUT_HI[0]: {x_after_ffn[0, pc_b0_pos, BD.OUTPUT_HI + 0].item():.3f}")
