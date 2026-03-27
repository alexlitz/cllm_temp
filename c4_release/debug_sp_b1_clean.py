"""Debug SP_b1 position cleanly."""
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

# Build context up to SP_b1
current_context = context + draft_tokens[:13]  # REG_PC ... SP_b0, SP_b1

sp_b1_pos = len(current_context) - 1
sp_marker_pos = sp_b1_pos - 2  # SP marker is 2 positions before SP_b1

print(f"Context length: {len(current_context)}")
print(f"SP marker at position {sp_marker_pos}: token {current_context[sp_marker_pos]} (should be {Token.REG_SP})")
print(f"SP_b0 at position {sp_b1_pos-1}: token {current_context[sp_b1_pos-1]}")
print(f"SP_b1 at position {sp_b1_pos}: token {current_context[sp_b1_pos]}")

ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    print(f"\n=== AT SP_b1 (pos {sp_b1_pos}) AFTER EMBEDDING ===")
    print(f"H1[SP]: {x[0, sp_b1_pos, BD.H1 + 2].item():.3f}")
    print(f"BYTE_INDEX_1: {x[0, sp_b1_pos, BD.BYTE_INDEX_1].item():.3f}")
    print(f"HAS_SE: {x[0, sp_b1_pos, BD.HAS_SE].item():.3f}")

    # Run through layers
    for layer_idx in range(4):
        x = model.blocks[layer_idx](x)
        if layer_idx in [0, 1, 3]:
            h1_sp = x[0, sp_b1_pos, BD.H1 + 2].item()
            byte_idx_1 = x[0, sp_b1_pos, BD.BYTE_INDEX_1].item()
            print(f"\nAfter Layer {layer_idx}:")
            print(f"  H1[SP]: {h1_sp:.3f}")
            print(f"  BYTE_INDEX_1: {byte_idx_1:.3f}")

    print(f"\n=== AFTER LAYER 3 FFN ===")
    print(f"OUTPUT_LO values:")
    for k in range(16):
        val = x[0, sp_b1_pos, BD.OUTPUT_LO + k].item()
        if abs(val) > 0.01:
            print(f"  OUTPUT_LO[{k}] = {val:.3f}")
