"""Debug JMP CMP[0] flag."""
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

    print(f"PC marker at position {pc_marker_pos}")
    print(f"HAS_SE at PC marker: {x[0, pc_marker_pos, BD.HAS_SE].item():.3f}")

    for layer_idx in range(7):
        x = model.blocks[layer_idx](x)

    print(f"\n=== AFTER LAYER 6 at PC marker ===")
    print(f"CMP[0] (IS_JMP): {x[0, pc_marker_pos, BD.CMP + 0].item():.3f}")
    print(f"CMP[1] (IS_EXIT): {x[0, pc_marker_pos, BD.CMP + 1].item():.3f}")
    print(f"\nAX_CARRY_LO/HI (JMP target):")
    for k in range(16):
        lo_val = x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item()
        hi_val = x[0, pc_marker_pos, BD.AX_CARRY_HI + k].item()
        if abs(lo_val) > 0.1 or abs(hi_val) > 0.1:
            print(f"  k={k:2d}: AX_CARRY_LO={lo_val:6.3f}, AX_CARRY_HI={hi_val:6.3f}")

    lo_vals = [x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.AX_CARRY_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    target = lo_max | (hi_max << 4)
    print(f"\nJMP target from AX_CARRY: {target}")
    print(f"Expected: 16 (from immediate)")
