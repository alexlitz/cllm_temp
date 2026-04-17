"""Debug JMP opcode fetch."""
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

print(f"Context: {context}")
print(f"Code section:")
print(f"  Address 0: opcode byte = {context[1]} (JMP={Opcode.JMP})")
print(f"  Address 1-4: immediate bytes = {context[2:6]}")

# Get draft tokens
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

# Build context up to AX marker
current_context = context + draft_tokens[:6]  # REG_PC + 4 bytes + REG_AX

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    ax_marker_pos = len(current_context) - 1
    pc_marker_pos = len(context)

    print(f"\n=== AFTER EMBEDDING ===")
    print(f"PC marker at pos {pc_marker_pos}")
    print(f"AX marker at pos {ax_marker_pos}")

    # Run through Layer 4 (PC relay)
    for layer_idx in range(5):
        x = model.blocks[layer_idx](x)

    print(f"\n=== AFTER LAYER 4 at AX marker ===")
    print(f"EMBED_LO/HI (PC relayed from PC marker):")
    for k in range(16):
        lo_val = x[0, ax_marker_pos, BD.EMBED_LO + k].item()
        hi_val = x[0, ax_marker_pos, BD.EMBED_HI + k].item()
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: EMBED_LO={lo_val:6.3f}, EMBED_HI={hi_val:6.3f}")

    lo_vals = [x[0, ax_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
    hi_vals = [x[0, ax_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    pc_at_ax = lo_max | (hi_max << 4)
    print(f"\nPC at AX marker: {pc_at_ax}")
    print(f"Should be 2 (PC_OFFSET) to fetch opcode at address 0")

    # Run Layer 5 (opcode fetch)
    x = model.blocks[5](x)

    print(f"\n=== AFTER LAYER 5 at AX marker ===")
    print(f"Opcode flags:")
    print(f"  OP_JMP: {x[0, ax_marker_pos, BD.OP_JMP].item():.3f}")
    print(f"  OP_IMM: {x[0, ax_marker_pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_NOP: {x[0, ax_marker_pos, BD.OP_NOP].item():.3f}")

    print(f"\nFETCH_LO/HI (immediate value):")
    for k in range(16):
        lo_val = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        hi_val = x[0, ax_marker_pos, BD.FETCH_HI + k].item()
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: FETCH_LO={lo_val:6.3f}, FETCH_HI={hi_val:6.3f}")
