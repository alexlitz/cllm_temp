#!/usr/bin/env python3
"""Debug LEA imm[1] fetch and relay pipeline."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

def build_context(bytecode, data=b""):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

# Use NOP first, then LEA 256 - so LEA is at step 2 (not first step)
# This ensures HAS_SE is set and standard PC relay works
# LEA 256: AX = BP + 256 = 0x10000 + 0x100 = 0x10100
# imm[0] = 0x00, imm[1] = 0x01
imm_val = 256
bytecode = [Opcode.NOP, Opcode.LEA | (imm_val << 8)]
context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()  # Execute NOP
step1_tokens = draft.draft_tokens()  # Step 1 tokens (after NOP)
draft.step()  # Execute LEA
step2_tokens = draft.draft_tokens()  # Step 2 tokens (after LEA)

print(f"LEA {imm_val}: AX = 0x{draft.ax:08X}")
print(f"  Immediate bytes: imm[0]={imm_val & 0xFF}, imm[1]={(imm_val >> 8) & 0xFF}")
print(f"  PC at step 2 (after LEA) = {draft.pc}")
print(f"  step2_tokens[:10] = {step2_tokens[:10]}")

# Build full context: code section + step 1 tokens (NOP) + step 2 tokens (LEA)
# Now step 2 has HAS_SE since step 1 has STEP_END
full_context = context + step1_tokens

# Check for HAS_SE - is there a STEP_END in context?
has_step_end = Token.STEP_END in full_context
print(f"  HAS_SE (STEP_END in context): {has_step_end}")

# Build context up to AX marker of step 2
ctx = full_context + step2_tokens[:5]  # PC marker + 4 PC bytes
ax_marker_pos = len(ctx)
ctx.append(step2_tokens[5])  # AX marker
print(f"\nAX marker position: {ax_marker_pos}")
print(f"Context length: {len(ctx)}")

token_ids = torch.tensor([ctx], dtype=torch.long)

# Find code positions with ADDR_KEY
print("\nChecking ADDR_KEY values in code section:")
with torch.no_grad():
    x_embed = model.embed(token_ids)
    for i, tok in enumerate(ctx):
        if tok == Token.CODE_START:
            print(f"  pos {i}: CODE_START")
        elif tok == Token.CODE_END:
            print(f"  pos {i}: CODE_END")
            break
        elif tok < 256:
            # Check ADDR_KEY
            addr_key_lo = [x_embed[0, i, BD.ADDR_KEY + k].item() for k in range(16)]
            addr_key_hi = [x_embed[0, i, BD.ADDR_KEY + 16 + k].item() for k in range(16)]
            if max(addr_key_lo) > 0.5:
                ak_lo = max(range(16), key=lambda k: addr_key_lo[k])
                ak_hi = max(range(16), key=lambda k: addr_key_hi[k])
                addr = ak_lo + (ak_hi << 4)
                print(f"  pos {i}: tok={tok}, ADDR_KEY={addr}")

# Find the previous step's PC marker position
prev_pc_marker_pos = None
for i, tok in enumerate(ctx):
    if tok == Token.REG_PC:
        prev_pc_marker_pos = i
# The last REG_PC before current step is the one we want
print(f"Previous step's PC marker position: {prev_pc_marker_pos}")

with torch.no_grad():
    x = model.embed(token_ids)

    # Check what's at the previous PC marker after L3 (before L4 relay)
    for i in range(3):
        x = model.blocks[i](x)

    if prev_pc_marker_pos is not None:
        prev_embed_lo = [x[0, prev_pc_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
        prev_embed_hi = [x[0, prev_pc_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
        prev_lo = max(range(16), key=lambda k: prev_embed_lo[k])
        prev_hi = max(range(16), key=lambda k: prev_embed_hi[k])
        print(f"\nAfter L3 at prev PC marker (pos {prev_pc_marker_pos}):")
        print(f"  EMBED_LO/HI: lo={prev_lo}, hi={prev_hi} → PC = {prev_lo + (prev_hi << 4)}")

        # Check MARK_PC flag
        mark_pc = x[0, prev_pc_marker_pos, BD.MARK_PC].item()
        print(f"  MARK_PC = {mark_pc:.4f}")

    # Run L4 (relay)
    x = model.blocks[3](x)
    x = model.blocks[4](x)

    embed_lo = [x[0, ax_marker_pos, BD.EMBED_LO + k].item() for k in range(16)]
    embed_hi = [x[0, ax_marker_pos, BD.EMBED_HI + k].item() for k in range(16)]
    pc_lo = max(range(16), key=lambda k: embed_lo[k])
    pc_hi = max(range(16), key=lambda k: embed_hi[k])
    pc_val = pc_lo + (pc_hi << 4)
    print(f"\nAfter L4 at AX marker (pos {ax_marker_pos}):")
    print(f"  EMBED_LO/HI (PC): lo={pc_lo}, hi={pc_hi} → PC = {pc_val}")

    # Check PC+1 in TEMP
    temp_lo = [x[0, ax_marker_pos, BD.TEMP + k].item() for k in range(16)]
    temp_hi = [x[0, ax_marker_pos, BD.TEMP + 16 + k].item() for k in range(16)]
    pc1_lo = max(range(16), key=lambda k: temp_lo[k])
    pc1_hi = max(range(16), key=lambda k: temp_hi[k])
    pc1_val = pc1_lo + (pc1_hi << 4)
    print(f"  TEMP (PC+1): lo={pc1_lo}, hi={pc1_hi} → PC+1 = {pc1_val}")

    # Check PC+2 in ADDR_B2
    addr_b2_lo = [x[0, ax_marker_pos, BD.ADDR_B2_LO + k].item() for k in range(16)]
    addr_b2_hi = [x[0, ax_marker_pos, BD.ADDR_B2_HI + k].item() for k in range(16)]
    pc2_lo = max(range(16), key=lambda k: addr_b2_lo[k])
    pc2_hi = max(range(16), key=lambda k: addr_b2_hi[k])
    pc2_val = pc2_lo + (pc2_hi << 4)
    print(f"  ADDR_B2 (PC+2): lo={pc2_lo}, hi={pc2_hi} → PC+2 = {pc2_val}")
    print(f"  Expected PC+2 = {pc_val + 2}")

    # Run L5 (fetch)
    x = model.blocks[5](x)

    # Check imm[1] in ADDR_B0_HI/ADDR_B1_HI (written by L5 head 5)
    # IMM1_LO = ADDR_B0_HI (dims 206-221), IMM1_HI = ADDR_B1_HI (dims 222-237)
    imm1_lo_vals = [x[0, ax_marker_pos, BD.ADDR_B0_HI + k].item() for k in range(16)]
    imm1_hi_vals = [x[0, ax_marker_pos, BD.ADDR_B1_HI + k].item() for k in range(16)]
    imm1_lo = max(range(16), key=lambda k: imm1_lo_vals[k])
    imm1_hi = max(range(16), key=lambda k: imm1_hi_vals[k])
    imm1_val = imm1_lo + (imm1_hi << 4)
    print(f"\nAfter L5 at AX marker:")
    print(f"  IMM1 (ADDR_B0_HI/B1_HI): lo={imm1_lo} ({imm1_lo_vals[imm1_lo]:.2f}), hi={imm1_hi} ({imm1_hi_vals[imm1_hi]:.2f}) → imm[1] = {imm1_val}")
    print(f"  Expected imm[1] = {(imm_val >> 8) & 0xFF}")

    # Also show ADDR_B2 to confirm it still has PC+2
    addr_b2_lo_after = [x[0, ax_marker_pos, BD.ADDR_B2_LO + k].item() for k in range(16)]
    addr_b2_hi_after = [x[0, ax_marker_pos, BD.ADDR_B2_HI + k].item() for k in range(16)]
    pc2_lo_after = max(range(16), key=lambda k: addr_b2_lo_after[k])
    pc2_hi_after = max(range(16), key=lambda k: addr_b2_hi_after[k])
    print(f"  ADDR_B2 (PC+2, unchanged): lo={pc2_lo_after}, hi={pc2_hi_after}")

    # Check FETCH (imm[0]) for comparison
    fetch_lo = [x[0, ax_marker_pos, BD.FETCH_LO + k].item() for k in range(16)]
    fetch_hi = [x[0, ax_marker_pos, BD.FETCH_HI + k].item() for k in range(16)]
    imm0_lo = max(range(16), key=lambda k: fetch_lo[k])
    imm0_hi = max(range(16), key=lambda k: fetch_hi[k])
    imm0_val = imm0_lo + (imm0_hi << 4)
    print(f"  FETCH (imm[0]): lo={imm0_lo}, hi={imm0_hi} → imm[0] = {imm0_val}")
    print(f"  Expected imm[0] = {imm_val & 0xFF}")

# Now check at byte 0 position
print("\n" + "="*50)
print("Checking at AX byte 0 position:")
ctx_byte0 = ctx + [step2_tokens[6]]  # AX byte 0
byte0_pos = len(ctx_byte0) - 1
print(f"Byte 0 position: {byte0_pos}")

token_ids_byte0 = torch.tensor([ctx_byte0], dtype=torch.long)

ax_marker_in_byte0_ctx = byte0_pos - 1  # AX marker is 1 position before byte 0

with torch.no_grad():
    x = model.embed(token_ids_byte0)

    # Check ADDR_B0_HI at AX marker after embedding (before any layers)
    addr_b0_hi_embed = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B0_HI + k].item() for k in range(16)]
    addr_b1_hi_embed = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B1_HI + k].item() for k in range(16)]
    print(f"\nADDR_B0_HI at AX marker after embedding (should be 0):")
    print(f"  {[f'{v:.2f}' for v in addr_b0_hi_embed[:4]]}")
    print(f"ADDR_B1_HI at AX marker after embedding:")
    print(f"  {[f'{v:.2f}' for v in addr_b1_hi_embed[:4]]}")

    # Run through L5 and check IMM1 at AX marker
    for i in range(6):
        x = model.blocks[i](x)

    imm1_lo_at_ax = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B0_HI + k].item() for k in range(16)]
    imm1_hi_at_ax = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B1_HI + k].item() for k in range(16)]
    imm1_lo_idx = max(range(16), key=lambda k: imm1_lo_at_ax[k])
    imm1_hi_idx = max(range(16), key=lambda k: imm1_hi_at_ax[k])
    print(f"\nAfter L5 at AX marker (pos {ax_marker_in_byte0_ctx}):")
    print(f"  IMM1: lo={imm1_lo_idx} ({imm1_lo_at_ax[imm1_lo_idx]:.2f}), hi={imm1_hi_idx} ({imm1_hi_at_ax[imm1_hi_idx]:.2f})")

    # Trace ADDR_B0_HI/ADDR_B1_HI through layers
    print("\nTracing IMM1 through layers:")
    for layer in range(6, 10):
        x = model.blocks[layer](x)
        b0_hi = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B0_HI + k].item() for k in range(4)]
        b1_hi = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B1_HI + k].item() for k in range(4)]
        print(f"  After L{layer}: ADDR_B0_HI[0:4]={[f'{v:.2f}' for v in b0_hi]}, ADDR_B1_HI[0:4]={[f'{v:.2f}' for v in b1_hi]}")

    op_lea_relay = x[0, byte0_pos, BD.OP_LEA_RELAY].item()
    print(f"\nAfter L9 at byte 0 position:")
    print(f"  OP_LEA_RELAY = {op_lea_relay:.4f}")

    # Check L10 head 3 conditions before L10
    print(f"\nL10 head 3 Q conditions at byte 0 (pos {byte0_pos}):")
    print(f"  IS_BYTE = {x[0, byte0_pos, BD.IS_BYTE].item():.2f}")
    print(f"  BYTE_INDEX_0 = {x[0, byte0_pos, BD.BYTE_INDEX_0].item():.2f}")
    print(f"  OP_LEA_RELAY = {x[0, byte0_pos, BD.OP_LEA_RELAY].item():.2f}")
    AX_IDX = 1
    print(f"  H1[AX] = {x[0, byte0_pos, BD.H1 + AX_IDX].item():.2f}")
    print(f"  MARK_AX at pos {ax_marker_in_byte0_ctx} = {x[0, ax_marker_in_byte0_ctx, BD.MARK_AX].item():.2f}")

    # Run L8-L9
    for i in range(8, 10):
        x = model.blocks[i](x)

    # Check Q/K for L10 head 3 before L10 attention
    attn10 = model.blocks[10].attn
    HD = attn10.head_dim
    base = 3 * HD  # head 3
    x_byte0 = x[0, byte0_pos, :]
    x_ax = x[0, ax_marker_in_byte0_ctx, :]
    q_full = x_byte0 @ attn10.W_q.T
    k_full = x_ax @ attn10.W_k.T
    q_head3 = q_full[base:base+HD]
    k_head3 = k_full[base:base+HD]

    print(f"\nL10 head 3 Q/K analysis:")
    print(f"  Q[0] (main gate) = {q_head3[0].item():.2f}")
    print(f"  Q[1] (H1 gate) = {q_head3[1].item():.2f}")
    print(f"  Q[33] (AND gate) = {q_head3[33].item():.2f}")
    print(f"  K[0] (MARK_AX) = {k_head3[0].item():.2f}")
    print(f"  K[33] = {k_head3[33].item():.2f}")
    score = (q_head3 @ k_head3) / (HD ** 0.5)
    print(f"  Attention score byte0 → AX marker = {score.item():.2f}")

    # Check V values at AX marker
    v_full = x_ax @ attn10.W_v.T
    v_head3 = v_full[base:base+HD]
    print(f"\n  V values at AX marker (head 3):")
    print(f"    V[0:5] = {[v_head3[k].item() for k in range(5)]}")
    print(f"    V[16:21] = {[v_head3[16+k].item() for k in range(5)]}")

    # Check ADDR_B0_HI/ADDR_B1_HI at AX marker (after L8-L9)
    imm1_lo_after_l9 = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B0_HI + k].item() for k in range(16)]
    imm1_hi_after_l9 = [x[0, ax_marker_in_byte0_ctx, BD.ADDR_B1_HI + k].item() for k in range(16)]
    imm1_lo_max = max(range(16), key=lambda k: imm1_lo_after_l9[k])
    imm1_hi_max = max(range(16), key=lambda k: imm1_hi_after_l9[k])
    print(f"\n  IMM1 at AX marker (after L9):")
    print(f"    ADDR_B0_HI[{imm1_lo_max}] = {imm1_lo_after_l9[imm1_lo_max]:.2f}")
    print(f"    ADDR_B1_HI[{imm1_hi_max}] = {imm1_hi_after_l9[imm1_hi_max]:.2f}")

    # Run L10
    x = model.blocks[10](x)

    # Check OUTPUT
    output_lo = [x[0, byte0_pos, BD.OUTPUT_LO + k].item() for k in range(16)]
    output_hi = [x[0, byte0_pos, BD.OUTPUT_HI + k].item() for k in range(16)]
    out_lo = max(range(16), key=lambda k: output_lo[k])
    out_hi = max(range(16), key=lambda k: output_hi[k])
    out_val = out_lo + (out_hi << 4)
    print(f"\nAfter L10 at byte 0 position:")
    print(f"  OUTPUT: lo={out_lo}, hi={out_hi} → byte 1 = {out_val}")
    print(f"  Expected byte 1 = {(draft.ax >> 8) & 0xFF}")
