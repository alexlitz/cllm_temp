"""Debug final prediction for PC_b1."""
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

    # Run through all layers
    for layer_idx in range(16):
        x = model.blocks[layer_idx](x)

    print(f"=== AFTER ALL LAYERS at PC_b0 position (pos {pc_b0_pos}) ===")
    print(f"Token: {current_context[pc_b0_pos]} (PC_b0=10)")

    # Check OUTPUT channels
    print(f"\nOUTPUT channels:")
    lo_vals = []
    hi_vals = []
    for k in range(16):
        lo_val = x[0, pc_b0_pos, BD.OUTPUT_LO + k].item()
        hi_val = x[0, pc_b0_pos, BD.OUTPUT_HI + k].item()
        lo_vals.append(lo_val)
        hi_vals.append(hi_val)
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: OUTPUT_LO={lo_val:6.3f}, OUTPUT_HI={hi_val:6.3f}")

    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    print(f"\nMax: OUTPUT_LO[{lo_max}]={lo_vals[lo_max]:.3f}, OUTPUT_HI[{hi_max}]={hi_vals[hi_max]:.3f}")
    print(f"Decoded value: {lo_max | (hi_max << 4)}")

    # Get final prediction
    logits = model.head(x)
    predicted = logits[0, -1, :].argmax().item()
    print(f"\n=== FINAL PREDICTION ===")
    print(f"Predicted PC_b1: {predicted} (expected: 0)")
    print(f"\nTop 10 logits:")
    top_k = torch.topk(logits[0, -1, :], 10)
    for val, idx in zip(top_k.values, top_k.indices):
        print(f"  Token {idx.item():3d}: {val.item():7.3f}")

    # Check head weights for OUTPUT dimensions
    print(f"\n=== HEAD WEIGHTS for OUTPUT dims ===")
    print(f"Checking head.weight[token, OUTPUT_LO/HI] for tokens 0-15:")
    for tok in range(16):
        max_weight = 0.0
        max_dim = None
        for k in range(16):
            w_lo = model.head.weight[tok, BD.OUTPUT_LO + k].item()
            w_hi = model.head.weight[tok, BD.OUTPUT_HI + k].item()
            if abs(w_lo) > abs(max_weight):
                max_weight = w_lo
                max_dim = f"OUTPUT_LO[{k}]"
            if abs(w_hi) > abs(max_weight):
                max_weight = w_hi
                max_dim = f"OUTPUT_HI[{k}]"
        if abs(max_weight) > 1.0:
            print(f"  Token {tok:3d}: max weight {max_weight:7.3f} at {max_dim}")
