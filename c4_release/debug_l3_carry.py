"""Debug Layer 3 carry-forward."""
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

# Build context with draft tokens
current_context = context + draft_tokens[:6]

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through L3 attention (carry-forward)
    for i in range(3):
        x = model.blocks[i](x)

    pc_marker_pos = len(context)

    print(f"=== AFTER LAYER 3 ATTENTION (BEFORE FFN) AT PC MARKER ===")
    print(f"EMBED_LO/HI at PC marker (carried forward from previous step):")
    for k in range(16):
        val = x[0, pc_marker_pos, BD.EMBED_LO + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_LO[{k}] = {val:.3f}")
    for k in range(16):
        val = x[0, pc_marker_pos, BD.EMBED_HI + k].item()
        if abs(val) > 0.1:
            print(f"  EMBED_HI[{k}] = {val:.3f}")

    # Decode
    lo_vals = [x[0, pc_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    pc_before_ffn = lo_max | (hi_max << 4)
    print(f"\nPC in EMBED before L3 FFN: {pc_before_ffn}")

    # Run L3 FFN
    x = x + model.blocks[3].ffn(x)

    print(f"\n=== AFTER LAYER 3 FFN AT PC MARKER ===")
    print(f"OUTPUT_LO/HI at PC marker (after increment):")
    lo_vals = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    pc_after_ffn = lo_max | (hi_max << 4)
    print(f"PC in OUTPUT after L3 FFN: {pc_after_ffn}")

    print(f"\nSummary:")
    print(f"  EMBED has OLD PC: {pc_before_ffn} (instruction just executed)")
    print(f"  OUTPUT has NEW PC: {pc_after_ffn} (next instruction)")
    print(f"  Layer 4 should relay EMBED (not OUTPUT) for opcode fetch")
