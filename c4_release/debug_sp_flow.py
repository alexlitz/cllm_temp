"""Debug SP byte data flow."""
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
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Autoregressive prediction flow:")
print("Position N → predicts token at position N+1")
print()

# Predict SP_b0 (at SP marker position)
current_context = context + draft_tokens[:10]  # Up to REG_SP marker
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)
    for i in range(16):
        x = model.blocks[i](x)

    sp_marker_pos = len(current_context) - 1
    print(f"=== At SP marker (pos {sp_marker_pos}), predicting SP_b0 ===")
    print(f"Token: {current_context[sp_marker_pos]} (REG_SP)")

    lo_vals = [x[0, sp_marker_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, sp_marker_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    print(f"OUTPUT: lo_max={lo_max}, hi_max={hi_max} → predicts token {lo_max | (hi_max << 4)}")
    print(f"Expected: token {draft_tokens[11]} (SP_b0)")
    print()

# Predict SP_b2 (at SP_b1 position)
current_context = context + draft_tokens[:13]  # Up to SP_b1
ctx_tensor = torch.tensor([current_context], dtype=torch.long)

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._add_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)
    for i in range(16):
        x = model.blocks[i](x)

    sp_b1_pos = len(current_context) - 1
    print(f"=== At SP_b1 (pos {sp_b1_pos}), predicting SP_b2 ===")
    print(f"Token: {current_context[sp_b1_pos]} (SP_b1={draft_tokens[12]})")
    print(f"H1[SP]: {x[0, sp_b1_pos, BD.H1 + 2].item():.3f}")
    print(f"BYTE_INDEX_1: {x[0, sp_b1_pos, BD.BYTE_INDEX_1].item():.3f}")
    print(f"EMBED_LO[0]: {x[0, sp_b1_pos, BD.EMBED_LO + 0].item():.3f}")
    print(f"EMBED_LO[1]: {x[0, sp_b1_pos, BD.EMBED_LO + 1].item():.3f}")

    lo_vals = [x[0, sp_b1_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    hi_vals = [x[0, sp_b1_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    lo_max = max(range(16), key=lambda k: lo_vals[k])
    hi_max = max(range(16), key=lambda k: hi_vals[k])
    print(f"\nOUTPUT: lo_max={lo_max}, hi_max={hi_max} → predicts token {lo_max | (hi_max << 4)}")
    print(f"Expected: token {draft_tokens[13]} (SP_b2)")
