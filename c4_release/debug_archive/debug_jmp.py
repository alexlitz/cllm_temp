"""Debug JMP 16 prediction."""
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

# Get draft tokens
draft_vm = DraftVM(bytecode)
print(f"DraftVM state before step:")
print(f"  PC = {draft_vm.pc}")
draft_vm.step()
print(f"DraftVM state after JMP 16:")
print(f"  PC = {draft_vm.pc}")

draft_tokens = draft_vm.draft_tokens()

print(f"\nDraft tokens for JMP 16:")
print(f"  REG_PC: {draft_tokens[0]}")
print(f"  PC_b0: {draft_tokens[1]} (expected after JMP to address 16)")
print(f"  PC_b1: {draft_tokens[2]}")
print(f"  PC_b2: {draft_tokens[3]}")
print(f"  PC_b3: {draft_tokens[4]}")

# Predict PC_b0
current_context = context + [draft_tokens[0]]  # Just REG_PC marker

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through all layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

    pc_marker_pos = len(current_context) - 1

    print(f"\n=== AFTER ALL LAYERS at PC marker ===")
    print(f"Opcode flags:")
    print(f"  OP_JMP: {x[0, pc_marker_pos, BD.OP_JMP].item():.3f}")
    print(f"  OP_IMM: {x[0, pc_marker_pos, BD.OP_IMM].item():.3f}")

    print(f"\nOUTPUT channels:")
    lo_vals = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])

    for k in range(16):
        if abs(lo_vals[k]) > 0.1 or abs(hi_vals[k]) > 0.1:
            print(f"  k={k:2d}: OUTPUT_LO={lo_vals[k]:6.3f}, OUTPUT_HI={hi_vals[k]:6.3f}")

    pc_predicted = lo_max | (hi_max << 4)
    print(f"\nDecoded PC: {pc_predicted}")
    print(f"Expected PC: {draft_tokens[1]} (from DraftVM)")

    # Get prediction
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()
    print(f"\nTransformer prediction for PC_b0: {predicted}")
