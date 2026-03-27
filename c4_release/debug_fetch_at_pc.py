"""Check if FETCH is available at PC marker after Layer 5."""
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

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

current_context = context + [draft_tokens[0]]  # Just REG_PC marker
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_marker_pos = len(current_context) - 1

    # Run through Layer 5
    for layer_idx in range(6):
        x = model.blocks[layer_idx](x)

    print(f"=== AFTER LAYER 5 at PC marker ===")
    print(f"OP_JMP: {x[0, pc_marker_pos, BD.OP_JMP].item():.3f}")

    fetch_lo = [x[0, pc_marker_pos, BD.FETCH_LO + k].item() for k in range(16)]
    fetch_hi = [x[0, pc_marker_pos, BD.FETCH_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: fetch_lo[k])
    hi_max = max(range(16), key=lambda k: fetch_hi[k])
    fetch_val = lo_max | (hi_max << 4)

    print(f"FETCH_LO: {fetch_lo}")
    print(f"FETCH_HI: {fetch_hi}")
    print(f"FETCH value: {fetch_val} (expected 16)")

    # Also check AX_CARRY (used by L6 FFN for JMP)
    carry_lo = [x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item() for k in range(16)]
    carry_hi = [x[0, pc_marker_pos, BD.AX_CARRY_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: carry_lo[k])
    hi_max = max(range(16), key=lambda k: carry_hi[k])
    carry_val = lo_max | (hi_max << 4)

    print(f"\nAX_CARRY_LO: {carry_lo}")
    print(f"AX_CARRY_HI: {carry_hi}")
    print(f"AX_CARRY value: {carry_val}")
