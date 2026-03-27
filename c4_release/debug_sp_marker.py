"""Debug SP marker OUTPUT."""
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

# Build context up to SP marker
current_context = context + draft_tokens[:11]  # Up to SP marker (REG_SP)

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    sp_marker_pos = len(current_context) - 1

    print(f"=== AFTER EMBEDDING at SP marker (pos {sp_marker_pos}) ===")
    print(f"Token: {current_context[sp_marker_pos]} (REG_SP={Token.REG_SP})")
    print(f"MARK_SP: {x[0, sp_marker_pos, BD.MARK_SP].item():.3f}")
    print(f"HAS_SE: {x[0, sp_marker_pos, BD.HAS_SE].item():.3f}")

    # Run through Layer 3
    for layer_idx in range(4):
        x = model.blocks[layer_idx](x)

    print(f"\n=== AFTER LAYER 3 at SP marker ===")
    print(f"OUTPUT channels:")
    for k in range(16):
        lo_val = x[0, sp_marker_pos, BD.OUTPUT_LO + k].item()
        hi_val = x[0, sp_marker_pos, BD.OUTPUT_HI + k].item()
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: OUTPUT_LO={lo_val:6.3f}, OUTPUT_HI={hi_val:6.3f}")

    print(f"\nExpected: OUTPUT_LO[1] should be positive (for SP byte 2 = 0x01)")
