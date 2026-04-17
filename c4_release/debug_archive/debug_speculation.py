"""Debug with speculative decoding context."""
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

# Simulate speculative decoding up to AX marker
current_context = context[:]
for i in range(6):  # Add REG_PC + 4 PC bytes + REG_AX
    current_context.append(draft_tokens[i])

print(f"Context with speculation up to AX marker:")
print(f"Length: {len(current_context)}")
print(f"Last tokens: {current_context[-6:]}")
print(f"REG_PC={Token.REG_PC}, REG_AX={Token.REG_AX}")

# Now check what the transformer predicts for AX_b0
ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through Layer 5 (fetch)
    for i in range(6):
        x = model.blocks[i](x)

    ax_marker_pos = len(current_context) - 1

    print(f"\n=== AFTER LAYER 5 AT AX MARKER (pos {ax_marker_pos}) ===")
    print(f"Token at AX marker: {current_context[ax_marker_pos]}")
    print(f"\nOPCODE flags:")
    print(f"  OP_IMM: {x[0, ax_marker_pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_NOP: {x[0, ax_marker_pos, BD.OP_NOP].item():.3f}")
    
    print(f"\nFETCH_LO/HI (immediate value):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.1:
            print(f"  FETCH_LO[{k}] = {val:.3f}")
    
    print(f"\nEMBED_LO/HI (PC value at AX marker):")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.EMBED_LO + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_LO[{k}] = {val:.3f}")

    # Run through all layers
    for i in range(6, 16):
        x = model.blocks[i](x)

    # Check OUTPUT at AX marker
    print(f"\n=== AFTER ALL LAYERS ===")
    lo_vals = [x[0, ax_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, ax_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    val = lo_max | (hi_max << 4)
    print(f"OUTPUT encodes: {val}")
    print(f"  OUTPUT_LO[{lo_max}] = {lo_vals[lo_max]:.3f}")
    print(f"  OUTPUT_HI[{hi_max}] = {hi_vals[hi_max]:.3f}")
    
    # Get prediction
    logits = model.head(x)
    predicted = logits[0, ax_marker_pos, :].argmax().item()
    print(f"\nTransformer prediction for AX_b0: {predicted} (expected: 42)")
