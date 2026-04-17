"""Debug PC on second step (EXIT after IMM)."""
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

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = build_context(bytecode)

# Step 1
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

print("=== STEP 1 (IMM) ===")
print(f"DraftVM PC after step: {draft_vm.pc}")
print(f"Draft PC byte 0: {draft1[1]}")

# Step 2
draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

print("\n=== STEP 2 (EXIT) ===")
print(f"DraftVM PC after step: {draft_vm.pc}")
print(f"Draft PC byte 0: {draft2[1]}")

# Check transformer prediction for step 2
ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len_step2 = len(context_step1)
pc_marker_pos_step2 = ctx_len_step2  # First token of step 2

with torch.no_grad():
    # Check L3 carry-forward
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through L3 attention (carry-forward)
    for i in range(3):
        x = model.blocks[i](x)

    print("\n=== Before L3 FFN (after carry-forward) ===")
    embed_lo = x[0, pc_marker_pos_step2, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x[0, pc_marker_pos_step2, BD.EMBED_HI:BD.EMBED_HI+16]
    has_se = x[0, pc_marker_pos_step2, BD.HAS_SE]
    print(f"PC marker EMBED: lo={embed_lo.argmax().item()}, hi={embed_hi.argmax().item()} → PC={(embed_lo.argmax().item() | (embed_hi.argmax().item() << 4))}")
    print(f"HAS_SE: {has_se.item():.3f}")

    # Run L3 FFN
    x = model.blocks[3](x)

    print("\n=== After L3 FFN (increment) ===")
    output_lo = x[0, pc_marker_pos_step2, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
    output_hi = x[0, pc_marker_pos_step2, BD.OUTPUT_HI:BD.OUTPUT_HI+16]
    pc_lo = output_lo.argmax().item()
    pc_hi = output_hi.argmax().item()
    print(f"PC marker OUTPUT: lo={pc_lo}, hi={pc_hi} → PC={pc_lo | (pc_hi << 4)}")
    print(f"Expected: PC=16")

    # Check final prediction
    logits = model.forward(ctx_tensor)
    pred_pc_byte0 = logits[0, ctx_len_step2, :].argmax().item()
    print(f"\n=== Final Prediction ===")
    print(f"Predicted PC byte 0: {pred_pc_byte0} (expected 16)")
