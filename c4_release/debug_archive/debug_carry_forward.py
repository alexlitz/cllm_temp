"""Debug carry-forward attention on step 2."""
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
model.eval()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = build_context(bytecode)

# Step 1
draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

# Step 2
draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len_step1 = len(context)
pc_byte0_pos_step1 = ctx_len_step1 + 1  # PC byte 0 from step 1

pc_marker_pos_step2 = len(context_step1)  # PC marker for step 2

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    print("=== After embedding ===")
    print(f"PC byte 0 from step 1 (position {pc_byte0_pos_step1}, token={ctx_tensor[0, pc_byte0_pos_step1].item()}):")
    embed_lo = x[0, pc_byte0_pos_step1, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x[0, pc_byte0_pos_step1, BD.EMBED_HI:BD.EMBED_HI+16]
    print(f"  EMBED_LO: argmax={embed_lo.argmax().item()} (value={embed_lo.max().item():.3f})")
    print(f"  EMBED_HI: argmax={embed_hi.argmax().item()} (value={embed_hi.max().item():.3f})")

    # Run through L0-L2 to compute threshold heads
    for i in range(3):
        x = model.blocks[i](x)

    print(f"\n=== After L2 (threshold heads computed) ===")
    print(f"PC byte 0 from step 1 (position {pc_byte0_pos_step1}):")
    l1h1 = x[0, pc_byte0_pos_step1, BD.L1H1:BD.L1H1+7]
    l1h0 = x[0, pc_byte0_pos_step1, BD.H0:BD.H0+7]
    print(f"  L1H1[PC_idx=0]: {l1h1[0].item():.3f}")
    print(f"  L1H0[PC_idx=0]: {l1h0[0].item():.3f}")
    print(f"  Pattern (L1H1 AND NOT L1H0): {(l1h1[0] > 0.5 and l1h0[0] < 0.5)}")

    # Run L3 attention (carry-forward)
    # Just attention, not FFN
    attn_out = model.blocks[3].attn(x)
    x_after_attn = x + attn_out

    print(f"\n=== After L3 attention (carry-forward) ===")
    print(f"PC marker position {pc_marker_pos_step2}:")
    embed_lo = x_after_attn[0, pc_marker_pos_step2, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x_after_attn[0, pc_marker_pos_step2, BD.EMBED_HI:BD.EMBED_HI+16]
    print(f"  EMBED_LO: argmax={embed_lo.argmax().item()} (value={embed_lo.max().item():.3f})")
    print(f"  EMBED_HI: argmax={embed_hi.argmax().item()} (value={embed_hi.max().item():.3f})")
    print(f"  Expected: EMBED_LO[8]=high, EMBED_HI[0]=high → PC=8")
