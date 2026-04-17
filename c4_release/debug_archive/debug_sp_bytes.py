"""Debug SP byte predictions."""
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

print(f"Draft tokens:")
print(f"  SP marker: {draft_tokens[10]} (Token.REG_SP={Token.REG_SP})")
print(f"  SP_b0: {draft_tokens[11]}")
print(f"  SP_b1: {draft_tokens[12]}")
print(f"  SP_b2: {draft_tokens[13]} (expected: 1 for 0x010000)")
print(f"  SP_b3: {draft_tokens[14]} (expected: 0)")

# Build context up to SP_b1
current_context = context + draft_tokens[:13]  # Up to SP_b1

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through all layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

    sp_b1_pos = len(current_context) - 1

    print(f"\n=== AFTER ALL LAYERS at SP_b1 position (pos {sp_b1_pos}) ===")
    print(f"Token: {current_context[sp_b1_pos]} (SP_b1=0)")

    # Check OUTPUT channels
    print(f"\nOUTPUT channels for SP_b2 prediction:")
    for k in range(16):
        lo_val = x[0, sp_b1_pos, BD.OUTPUT_LO + k].item()
        hi_val = x[0, sp_b1_pos, BD.OUTPUT_HI + k].item()
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: OUTPUT_LO={lo_val:6.3f}, OUTPUT_HI={hi_val:6.3f}")

    lo_vals = [x[0, sp_b1_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, sp_b1_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    print(f"\nDecoded: lo={lo_max}, hi={hi_max} → byte value = {lo_max | (hi_max << 4)}")
    print(f"Expected: byte value = 1 (for SP byte 2)")

    # Get prediction
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()
    print(f"\nPredicted SP_b2: {predicted} (expected: 1)")
