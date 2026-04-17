"""Debug Layer 5 head 2 opcode fetch for first step."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.embedding import Opcode

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
print(f"Position 0: CODE_START = {context[0]}")
print(f"Position 1: JMP opcode = {context[1]}")
print(f"Position 9: REG_PC marker")

# Build context up to PC marker
from neural_vm.speculative import DraftVM
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

    print(f"\n=== AFTER EMBEDDING ==")
    print(f"PC marker at pos {pc_marker_pos}")
    print(f"HAS_SE at PC marker: {x[0, pc_marker_pos, BD.HAS_SE].item():.3f}")

    # Run through Layer 4 (before Layer 5)
    for layer_idx in range(5):
        x = model.blocks[layer_idx](x)

    print(f"\n=== AFTER LAYER 4 (before L5 opcode fetch) at PC marker ===")
    print(f"OPCODE_BYTE_LO: {[x[0, pc_marker_pos, BD.OPCODE_BYTE_LO + k].item() for k in range(16)]}")
    print(f"OPCODE_BYTE_HI: {[x[0, pc_marker_pos, BD.OPCODE_BYTE_HI + k].item() for k in range(16)]}")

    # Run Layer 5 (opcode fetch)
    x = model.blocks[5](x)

    print(f"\n=== AFTER LAYER 5 (opcode fetch and decode) at PC marker ===")
    print(f"OPCODE_BYTE_LO: {[x[0, pc_marker_pos, BD.OPCODE_BYTE_LO + k].item() for k in range(16)]}")
    print(f"OPCODE_BYTE_HI: {[x[0, pc_marker_pos, BD.OPCODE_BYTE_HI + k].item() for k in range(16)]}")

    # Decode which nibbles are set
    lo_vals = [x[0, pc_marker_pos, BD.OPCODE_BYTE_LO + k].item() for k in range(16)]
    hi_vals = [x[0, pc_marker_pos, BD.OPCODE_BYTE_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    opcode_byte = lo_max | (hi_max << 4)

    print(f"\nDecoded opcode byte: {opcode_byte} (expected 2 for JMP)")
    print(f"OP_JMP at PC marker: {x[0, pc_marker_pos, BD.OP_JMP].item():.3f}")
    print(f"OP_IMM at PC marker: {x[0, pc_marker_pos, BD.OP_IMM].item():.3f}")
