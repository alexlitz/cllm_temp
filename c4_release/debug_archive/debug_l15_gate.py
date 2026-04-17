"""Debug Layer 15 FFN gate values."""
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

    print(f"=== BEFORE LAYER 15 FFN at PC_b0 position (pos {pc_b0_pos}) ===")
    print(f"Token: {current_context[pc_b0_pos]} (PC_b0=10)")
    print(f"H1[PC]: {x[0, pc_b0_pos, BD.H1 + 0].item():.3f}")
    print(f"EMBED_LO[10]: {x[0, pc_b0_pos, BD.EMBED_LO + 10].item():.3f}")
    print(f"EMBED_HI[0]: {x[0, pc_b0_pos, BD.EMBED_HI + 0].item():.3f}")
    print(f"IS_BYTE: {x[0, pc_b0_pos, BD.IS_BYTE].item():.3f}")

    # Check FFN W_gate for unit 10 (EMBED_LO[10])
    ffn = model.blocks[15].ffn
    unit = 10  # Corresponds to EMBED_LO[10]

    print(f"\n=== LAYER 15 FFN WEIGHTS FOR UNIT {unit} (EMBED_LO[10]) ===")
    print(f"W_gate[{unit}, EMBED_LO+10]: {ffn.W_gate[unit, BD.EMBED_LO + 10].item():.3f}")
    print(f"W_gate[{unit}, H1+0]: {ffn.W_gate[unit, BD.H1 + 0].item():.3f}")
    print(f"b_gate[{unit}]: {ffn.b_gate[unit].item():.3f}")

    # Compute gate value manually
    gate_val = (ffn.b_gate[unit].item() +
                x[0, pc_b0_pos, BD.EMBED_LO + 10].item() * ffn.W_gate[unit, BD.EMBED_LO + 10].item() +
                x[0, pc_b0_pos, BD.H1 + 0].item() * ffn.W_gate[unit, BD.H1 + 0].item())
    print(f"\nComputed gate value: {gate_val:.3f}")
    print(f"  = {ffn.b_gate[unit].item():.3f} + {x[0, pc_b0_pos, BD.EMBED_LO + 10].item():.3f}*{ffn.W_gate[unit, BD.EMBED_LO + 10].item():.3f} + {x[0, pc_b0_pos, BD.H1 + 0].item():.3f}*{ffn.W_gate[unit, BD.H1 + 0].item():.3f}")
    print(f"Should be negative to suppress")
