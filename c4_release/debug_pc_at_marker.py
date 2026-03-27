"""Debug PC output at the REG_PC marker position."""
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

# Setup
model = AutoregressiveVM()
set_vm_weights(model)
model.compact(block_size=32)
model.compact_moe()
model.eval()

bytecode = [Opcode.NOP]
context = build_context(bytecode)

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Context length: {len(context)}")
print(f"Draft tokens[0]: {draft_tokens[0]} (REG_PC marker)")
print(f"Draft tokens[1]: {draft_tokens[1]} (PC byte 0)")

# Teacher forcing: append draft to context
full_context = context + draft_tokens
ctx_tensor = torch.tensor([full_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through layers
    for i in range(4):
        x = model.blocks[i](x)

    # Check position 9 (REG_PC marker position)
    pc_marker_pos = len(context)  # Position 9
    print(f"\n=== AT POSITION {pc_marker_pos} (REG_PC marker) ===")
    print(f"Token: {full_context[pc_marker_pos]} (should be {Token.REG_PC})")
    print(f"HAS_SE: {x[0, pc_marker_pos, BD.HAS_SE].item():.3f}")
    print(f"MARK_PC: {x[0, pc_marker_pos, BD.MARK_PC].item():.3f}")
    print(f"EMBED_LO[0]: {x[0, pc_marker_pos, BD.EMBED_LO + 0].item():.3f}")
    print(f"EMBED_LO[2]: {x[0, pc_marker_pos, BD.EMBED_LO + 2].item():.3f}")
    print(f"OUTPUT_LO[0]: {x[0, pc_marker_pos, BD.OUTPUT_LO + 0].item():.3f}")
    print(f"OUTPUT_LO[10]: {x[0, pc_marker_pos, BD.OUTPUT_LO + 10].item():.3f}")

    # Get logits
    logits = model.forward(ctx_tensor)

    # Check what's predicted at position 9 (should predict PC byte 0 = 10)
    predicted = logits[0, pc_marker_pos, :].argmax().item()
    expected = draft_tokens[1]  # PC byte 0
    print(f"\nPrediction at REG_PC marker position:")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {predicted}")
    print(f"  Match: {predicted == expected}")
