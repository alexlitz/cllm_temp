"""Debug PC_b1 prediction."""
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

print(f"Draft tokens for IMM 42:")
print(f"  REG_PC: {draft_tokens[0]}")
print(f"  PC_b0: {draft_tokens[1]} (expected: 10)")
print(f"  PC_b1: {draft_tokens[2]} (expected: 0)")
print(f"  PC_b2: {draft_tokens[3]} (expected: 0)")
print(f"  PC_b3: {draft_tokens[4]} (expected: 0)")

# Predict PC_b1
current_context = context + [draft_tokens[0], draft_tokens[1]]  # REG_PC + PC_b0

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_b0_pos = len(current_context) - 1

    print(f"\n=== AFTER EMBEDDING at PC_b0 position ===")
    print(f"H1 flags:")
    print(f"  H1[PC]: {x[0, pc_b0_pos, BD.H1 + 0].item():.3f}")
    print(f"  H1[AX]: {x[0, pc_b0_pos, BD.H1 + 1].item():.3f}")

    # Run through layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

        if layer_idx in [2, 3, 9, 10]:
            print(f"\n=== AFTER LAYER {layer_idx} at PC_b0 position ===")
            lo_vals = [x[0, pc_b0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
            lo_max = max(range(16), key=lambda k: lo_vals[k])
            print(f"OUTPUT_LO[{lo_max}] = {lo_vals[lo_max]:.3f}")

            hi_vals = [x[0, pc_b0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
            hi_max = max(range(16), key=lambda k: hi_vals[k])
            print(f"OUTPUT_HI[{hi_max}] = {hi_vals[hi_max]:.3f}")

    # Get final prediction
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()
    print(f"\n=== FINAL PREDICTION ===")
    print(f"Predicted PC_b1: {predicted} (expected: 0)")
    print(f"Top 5 logits:")
    top_k = torch.topk(logits[0, -1, :], 5)
    for val, idx in zip(top_k.values, top_k.indices):
        print(f"  {idx.item()}: {val.item():.3f}")
