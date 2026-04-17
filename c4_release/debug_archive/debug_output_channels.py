"""Debug OUTPUT channels at byte positions."""
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

bytecode = [Opcode.IMM | (42 << 8)]
context = build_context(bytecode)

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print(f"Context: {context}")
print(f"Draft tokens: {draft_tokens[:10]}")

# Teacher forcing
full_context = context + draft_tokens
ctx_tensor = torch.tensor([full_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through all layers
    for i, block in enumerate(model.blocks):
        x = block(x)

        # Check OUTPUT channels at key positions after each layer
        if i >= 2:  # L3 onwards
            print(f"\n=== AFTER LAYER {i} ===")
            # Position 9: REG_PC marker
            lo_vals = [x[0, 9, BD.OUTPUT_LO + k].item() for k in range(16)]
            lo_max = max(range(16), key=lambda k: lo_vals[k])
            hi_vals = [x[0, 9, BD.OUTPUT_HI + k].item() for k in range(16)]
            hi_max = max(range(16), key=lambda k: hi_vals[k])
            pc_at_marker = lo_max | (hi_max << 4)
            print(f"  Pos 9 (REG_PC marker): OUTPUT encodes {pc_at_marker}")

            # Position 10: PC_b0 = 10 (token from teacher forcing)
            lo_vals = [x[0, 10, BD.OUTPUT_LO + k].item() for k in range(16)]
            lo_max = max(range(16), key=lambda k: lo_vals[k])
            hi_vals = [x[0, 10, BD.OUTPUT_HI + k].item() for k in range(16)]
            hi_max = max(range(16), key=lambda k: hi_vals[k])
            val_at_10 = lo_max | (hi_max << 4)
            print(f"  Pos 10 (PC_b0=10): OUTPUT encodes {val_at_10}")

            # Position 14: REG_AX marker
            lo_vals = [x[0, 14, BD.OUTPUT_LO + k].item() for k in range(16)]
            lo_max = max(range(16), key=lambda k: lo_vals[k])
            hi_vals = [x[0, 14, BD.OUTPUT_HI + k].item() for k in range(16)]
            hi_max = max(range(16), key=lambda k: hi_vals[k])
            ax_at_marker = lo_max | (hi_max << 4)
            print(f"  Pos 14 (REG_AX marker): OUTPUT encodes {ax_at_marker}")
