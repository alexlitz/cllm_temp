"""Check H1[AX] at PC byte positions."""
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

    # Run through layers
    for layer_idx in range(11):  # Up to Layer 10
        x = model.blocks[layer_idx](x)

    print(f"=== AFTER LAYER 10 at PC_b0 position (pos {pc_b0_pos}) ===")
    print(f"Token: {current_context[pc_b0_pos]} (PC_b0=10)")
    print(f"H1[PC]: {x[0, pc_b0_pos, BD.H1 + 0].item():.3f} (should be 1.0)")
    print(f"H1[AX]: {x[0, pc_b0_pos, BD.H1 + 1].item():.3f} (should be 0.0)")
    print(f"H1[SP]: {x[0, pc_b0_pos, BD.H1 + 2].item():.3f} (should be 0.0)")

    # Check OUTPUT values
    print(f"\nOUTPUT channels:")
    for k in range(16):
        lo_val = x[0, pc_b0_pos, BD.OUTPUT_LO + k].item()
        if abs(lo_val) > 0.1:
            print(f"  OUTPUT_LO[{k}] = {lo_val:.3f}")
    for k in range(16):
        hi_val = x[0, pc_b0_pos, BD.OUTPUT_HI + k].item()
        if abs(hi_val) > 0.1:
            print(f"  OUTPUT_HI[{k}] = {hi_val:.3f}")
