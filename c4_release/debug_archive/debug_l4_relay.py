"""Debug Layer 4 PC relay to AX marker."""
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

# Build context up to AX marker with draft tokens
current_context = context + draft_tokens[:6]  # REG_PC + 4 bytes + REG_AX

print(f"Context:")
print(f"  Code section: {context[1:6]} (addresses 0-4)")
print(f"  Draft PC bytes: {draft_tokens[1:5]} (PC after exec = 10)")
print(f"  Context ends at AX marker (token {Token.REG_AX})")

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Check PC marker after L3
    for i in range(4):
        x = model.blocks[i](x)

    pc_marker_pos = len(context)  # Position of REG_PC marker
    ax_marker_pos = len(current_context) - 1  # Position of REG_AX marker

    print(f"\n=== AFTER LAYER 3 AT PC MARKER (pos {pc_marker_pos}) ===")
    print(f"OUTPUT at PC marker (PC value for this step):")
    lo_vals = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    pc_at_marker = lo_max | (hi_max << 4)
    print(f"  PC = {pc_at_marker} (should be 10 for NEXT instruction)")

    # Check what Layer 4 relays to AX marker
    x_after_l4 = x + model.blocks[4].attn(x)

    print(f"\n=== AFTER LAYER 4 ATTENTION AT AX MARKER (pos {ax_marker_pos}) ===")
    print(f"EMBED_LO/HI at AX marker (relayed from PC marker):")
    for k in range(16):
        val = x_after_l4[0, ax_marker_pos, BD.EMBED_LO + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_LO[{k}] = {val:.3f}")
    for k in range(16):
        val = x_after_l4[0, ax_marker_pos, BD.EMBED_HI + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_HI[{k}] = {val:.3f}")

    # Decode the PC value
    lo_vals = [x_after_l4[0, ax_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
    hi_vals = [x_after_l4[0, ax_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    pc_relayed = lo_max | (hi_max << 4)
    print(f"\nRelayed PC value: {pc_relayed}")
    print(f"Should be 2 (PC BEFORE increment) to fetch opcode at address 0")
