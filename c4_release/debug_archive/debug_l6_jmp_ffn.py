"""Debug Layer 6 FFN JMP override logic."""
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

    print(f"=== AFTER LAYER 5 at PC marker (before L6 FFN) ===")
    print(f"OP_JMP: {x[0, pc_marker_pos, BD.OP_JMP].item():.3f}")
    print(f"CMP[0]: {x[0, pc_marker_pos, BD.CMP + 0].item():.3f}")

    carry_lo = [x[0, pc_marker_pos, BD.AX_CARRY_LO + k].item() for k in range(16)]
    carry_hi = [x[0, pc_marker_pos, BD.AX_CARRY_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: carry_lo[k])
    hi_max = max(range(16), key=lambda k: carry_hi[k])
    carry_val = lo_max | (hi_max << 4)
    print(f"AX_CARRY value: {carry_val}")

    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: output_lo[k])
    hi_max = max(range(16), key=lambda k: output_hi[k])
    output_val = lo_max | (hi_max << 4)
    print(f"OUTPUT value (before L6 FFN): {output_val}")

    # Run Layer 6 FFN only
    x = model.blocks[6].ffn(x)

    print(f"\n=== AFTER LAYER 6 FFN at PC marker ===")
    output_lo = [x[0, pc_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, pc_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    print(f"OUTPUT_LO: {[f'{v:.2f}' for v in output_lo]}")
    print(f"OUTPUT_HI: {[f'{v:.2f}' for v in output_hi]}")

    lo_max = max(range(16), key=lambda k: output_lo[k])
    hi_max = max(range(16), key=lambda k: output_hi[k])
    output_val = lo_max | (hi_max << 4)
    print(f"OUTPUT value (after L6 FFN): {output_val}")
    print(f"Expected: 16 (JMP target) or 18 (16 + PC_OFFSET=2)")
