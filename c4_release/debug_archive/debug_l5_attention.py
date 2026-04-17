"""Debug L5 attention pattern to see where it's attending."""
import torch
import torch.nn.functional as F
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

# NOP + JMP 16 - check step 2
bytecode = [Opcode.NOP, Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

draft_vm.step()
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len_step2 = len(context_step1)
ax_marker_pos = ctx_len_step2 + 5

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    # Run through L4
    for i in range(5):
        if i < 4:
            x = model.blocks[i](x)
        else:
            x = x + model.blocks[4].attn(x)
            x = x + model.blocks[4].ffn(x)

    # Manually compute L5 attention
    attn5 = model.blocks[5].attn
    Q = x @ attn5.W_q.T  # [B, S, D]
    K = x @ attn5.W_k.T
    V = x @ attn5.W_v.T

    B, S, D = Q.shape
    H = attn5.num_heads
    HD = D // H

    Q = Q.view(B, S, H, HD).transpose(1, 2)  # [B, H, S, HD]
    K = K.view(B, S, H, HD).transpose(1, 2)
    V = V.view(B, S, H, HD).transpose(1, 2)

    scores = (Q @ K.transpose(-2, -1)) * attn5.scale  # [B, H, S, S]

    # Add ALiBi
    positions = torch.arange(S, device=x.device)
    dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # [S, S]
    alibi = -attn5.alibi_slopes.view(1, H, 1, 1) * dist.abs().float()
    scores = scores + alibi

    # Head 0: fetch immediate
    print(f"=== L5 HEAD 0: Fetch immediate ===")
    print(f"Query position: {ax_marker_pos} (AX marker)")
    print(f"Expected to attend to position 7 (immediate byte, value 16)")

    head0_scores = scores[0, 0, ax_marker_pos, :]  # Scores from AX marker
    print(f"\nTop 5 attention targets:")
    top5 = torch.topk(head0_scores, 5)
    for score, pos in zip(top5.values, top5.indices):
        token_val = ctx_tensor[0, pos].item()
        print(f"  Position {pos:3d}: score={score.item():7.2f}, token={token_val:3d}")

    # Check score at position 7
    print(f"\nScore at position 7 (immediate): {head0_scores[7].item():.2f}")
    print(f"Score at position 10 (last imm byte): {head0_scores[10].item():.2f}")

    # Check ADDR_KEY encoding
    print(f"\n=== ADDR_KEY Check ===")
    print(f"AX marker TEMP (PC+1) expects address 9")
    for pos in [7, 8, 9, 10]:
        addr_key = x[0, pos, BD.ADDR_KEY:BD.ADDR_KEY+48]
        # Decode address from ADDR_KEY (nibble encoding)
        addr_lo = x[0, pos, BD.ADDR_KEY:BD.ADDR_KEY+16].argmax().item()
        addr_mid = x[0, pos, BD.ADDR_KEY+16:BD.ADDR_KEY+32].argmax().item()
        addr = addr_lo | (addr_mid << 4)
        print(f"Position {pos}: ADDR_KEY encodes address={addr}, token={ctx_tensor[0,pos].item()}")
