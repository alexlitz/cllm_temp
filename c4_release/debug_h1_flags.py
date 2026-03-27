"""Debug H1 flags at byte positions."""
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

# Teacher forcing
full_context = context + draft_tokens
ctx_tensor = torch.tensor([full_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through layers up to L1 (where H1 flags should be set)
    for i in range(2):
        x = model.blocks[i](x)

    print("=== AFTER LAYER 1 ===")
    print("\nPosition 9 (REG_PC marker):")
    print(f"  IS_BYTE: {x[0, 9, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_PC: {x[0, 9, BD.MARK_PC].item():.3f}")
    print(f"  H1[0] (PC_IDX): {x[0, 9, BD.H1 + 0].item():.3f}")
    print(f"  H1[1] (AX_IDX): {x[0, 9, BD.H1 + 1].item():.3f}")

    print("\nPosition 10 (PC_b0=10 token):")
    print(f"  IS_BYTE: {x[0, 10, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_PC: {x[0, 10, BD.MARK_PC].item():.3f}")
    print(f"  H1[0] (PC_IDX): {x[0, 10, BD.H1 + 0].item():.3f}")
    print(f"  H1[1] (AX_IDX): {x[0, 10, BD.H1 + 1].item():.3f}")
    print(f"  BYTE_INDEX_0: {x[0, 10, BD.BYTE_INDEX_0].item():.3f}")

    print("\nPosition 14 (REG_AX marker):")
    print(f"  IS_BYTE: {x[0, 14, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_AX: {x[0, 14, BD.MARK_AX].item():.3f}")
    print(f"  H1[0] (PC_IDX): {x[0, 14, BD.H1 + 0].item():.3f}")
    print(f"  H1[1] (AX_IDX): {x[0, 14, BD.H1 + 1].item():.3f}")

    print("\nPosition 15 (AX_b0 token):")
    print(f"  IS_BYTE: {x[0, 15, BD.IS_BYTE].item():.3f}")
    print(f"  MARK_AX: {x[0, 15, BD.MARK_AX].item():.3f}")
    print(f"  H1[0] (PC_IDX): {x[0, 15, BD.H1 + 0].item():.3f}")
    print(f"  H1[1] (AX_IDX): {x[0, 15, BD.H1 + 1].item():.3f}")
    print(f"  BYTE_INDEX_0: {x[0, 15, BD.BYTE_INDEX_0].item():.3f}")
