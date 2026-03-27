"""Debug JMP: trace L4/L5 to see why FETCH isn't populated."""
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

# NOP + JMP 16 - trace step 2
bytecode = [Opcode.NOP, Opcode.JMP | (16 << 8)]
context = build_context(bytecode)

draft_vm = DraftVM(bytecode)
draft_vm.step()  # Step 1: NOP
draft1 = draft_vm.draft_tokens()
context_step1 = context + draft1

draft_vm.step()  # Step 2: JMP 16
draft2 = draft_vm.draft_tokens()
context_step2 = context_step1 + draft2

ctx_tensor = torch.tensor([context_step2], dtype=torch.long)
ctx_len_step2 = len(context_step1)
ax_marker_pos = ctx_len_step2 + 5  # AX marker in step 2

with torch.no_grad():
    x = model.embed(ctx_tensor)
    model._inject_code_addr_keys(ctx_tensor, x)
    model._inject_mem_store(ctx_tensor, x)

    print("=== STEP 2: JMP 16 ===")
    print(f"AX marker position: {ax_marker_pos}")

    # Check code layout
    print("\n=== CODE LAYOUT ===")
    print(f"Context tokens from CODE_START:")
    code_start_idx = 0
    code_end_idx = context.index(Token.CODE_END)
    print(f"Positions {code_start_idx}-{code_end_idx}: {context[code_start_idx:code_end_idx+1]}")
    print(f"Instruction layout (5 bytes each with PC_OFFSET=0, INSTR_WIDTH=8):")
    print(f"  Instr 0 (NOP): bytes 0-7, tokens at positions 1-5")
    print(f"  Instr 1 (JMP 16): bytes 8-15, tokens at positions 6-10")
    print(f"Token at position 6 (byte 8, opcode): {context[6]}")
    print(f"Token at position 7 (byte 9, imm[0]): {context[7]}")
    print(f"Expected: opcode=2 (JMP), imm[0]=16")

    # After L4 attention
    for i in range(4):
        x = model.blocks[i](x)
    x = x + model.blocks[4].attn(x)

    print("\n=== AFTER L4 ATTENTION ===")
    embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16]
    embed_hi = x[0, ax_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16]
    pc_from_embed = embed_lo.argmax().item() | (embed_hi.argmax().item() << 4)
    print(f"AX marker EMBED (PC): {pc_from_embed} (should be 8 from step 1)")

    # After L4 FFN
    x = x + model.blocks[4].ffn(x)

    print("\n=== AFTER L4 FFN ===")
    temp_lo = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+16]
    temp_hi = x[0, ax_marker_pos, BD.TEMP+16:BD.TEMP+32]
    pc_plus1 = temp_lo.argmax().item() | (temp_hi.argmax().item() << 4)
    print(f"AX marker TEMP (PC+1): {pc_plus1} (should be 9)")

    # Check what's at position 7 (the immediate byte) before L5
    print("\n=== BEFORE L5: Check immediate byte position ===")
    imm_pos = 7  # Position of immediate byte (value 16)
    clean_embed_lo = x[0, imm_pos, BD.CLEAN_EMBED_LO:BD.CLEAN_EMBED_LO+16]
    clean_embed_hi = x[0, imm_pos, BD.CLEAN_EMBED_HI:BD.CLEAN_EMBED_HI+16]
    val_from_embed = clean_embed_lo.argmax().item() | (clean_embed_hi.argmax().item() << 4)
    print(f"Position {imm_pos} CLEAN_EMBED value: {val_from_embed} (should be 16)")
    print(f"  CLEAN_EMBED_LO: argmax={clean_embed_lo.argmax().item()}, max={clean_embed_lo.max().item():.3f}")
    print(f"  CLEAN_EMBED_HI: argmax={clean_embed_hi.argmax().item()}, max={clean_embed_hi.max().item():.3f}")

    addr_key = x[0, imm_pos, BD.ADDR_KEY:BD.ADDR_KEY+48]
    print(f"Position {imm_pos} ADDR_KEY: max={addr_key.max().item():.3f} at {addr_key.argmax().item()}")

    # After L5 (fetch)
    x = model.blocks[5](x)

    print("\n=== AFTER L5 (fetch) ===")
    fetch_lo = x[0, ax_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16]
    fetch_hi = x[0, ax_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16]
    fetch_val = fetch_lo.argmax().item() | (fetch_hi.argmax().item() << 4)
    print(f"AX marker FETCH (immediate): {fetch_val} (should be 16)")
    print(f"FETCH_LO max: {fetch_lo.max().item():.3f}")
    print(f"FETCH_HI max: {fetch_hi.max().item():.3f}")
