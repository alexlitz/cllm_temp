"""Debug all Layer 15 FFN units."""
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

# Build context up to PC_b0
current_context = context + [draft_tokens[0], draft_tokens[1]]  # REG_PC + PC_b0

ctx_tensor = torch.tensor([current_context], dtype=torch.long)
with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    pc_b0_pos = len(current_context) - 1

    # Run through Layer 14
    for layer_idx in range(15):
        x = model.blocks[layer_idx](x)

    ffn = model.blocks[15].ffn

    print(f"=== LAYER 15 FFN at PC_b0 position ===")
    print(f"Token: {current_context[pc_b0_pos]} (PC_b0=10)")
    print(f"H1[PC]: {x[0, pc_b0_pos, BD.H1 + 0].item():.3f}")
    print(f"IS_BYTE: {x[0, pc_b0_pos, BD.IS_BYTE].item():.3f}")

    print(f"\nChecking first 32 units (nibble copy):")
    for unit in [0, 10, 16]:  # unit 0 (EMBED_LO[0]), unit 10 (EMBED_LO[10]), unit 16 (EMBED_HI[0])
        embed_dim = BD.EMBED_LO + unit if unit < 16 else BD.EMBED_HI + (unit - 16)
        embed_val = x[0, pc_b0_pos, embed_dim].item()
        h1_pc = x[0, pc_b0_pos, BD.H1 + 0].item()

        gate_val = ffn.b_gate[unit].item() + embed_val * ffn.W_gate[unit, embed_dim].item() + h1_pc * ffn.W_gate[unit, BD.H1 + 0].item()

        output_dim = BD.OUTPUT_LO + unit if unit < 16 else BD.OUTPUT_HI + (unit - 16)
        w_down = ffn.W_down[output_dim, unit].item()

        print(f"\nUnit {unit:2d}:")
        print(f"  Target dim: {'OUTPUT_LO' if unit < 16 else 'OUTPUT_HI'}[{unit if unit < 16 else unit-16}]")
        print(f"  EMBED value: {embed_val:.3f}")
        print(f"  Gate: {gate_val:.3f}")
        print(f"  W_down: {w_down:.3f}")
