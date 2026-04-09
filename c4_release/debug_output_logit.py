#!/usr/bin/env python3
"""Debug how OUTPUT_LO maps to logits."""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode, data=b''):
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

def main():
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context = build_context(BYTECODE)
    draft = DraftVM(BYTECODE)

    # Calculate correct step starts
    prefix_len = 1 + len(BYTECODE) * (1 + IMMEDIATE_SIZE + PADDING_SIZE) + 1 + 1 + 1
    print(f"Prefix length: {prefix_len}")
    print(f"Step starts: {[prefix_len + i*35 for i in range(4)]}")

    # Run 3 steps
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    print(f"\n=== Step 4 BP bytes ===")
    print(f"Draft BP: {draft.bp:#x} = {draft.bp}")
    bp_bytes = step4_tokens[16:20]
    print(f"BP bytes in draft: {bp_bytes}")

    # Build context up to BP marker (we want to predict BP byte 0)
    context_for_pred = context + step4_tokens[:16]  # Up to and including BP marker
    bp_marker_pos = len(context_for_pred) - 1

    print(f"\nContext length: {len(context_for_pred)}")
    print(f"BP marker position: {bp_marker_pos}")
    print(f"BP marker token: {context_for_pred[bp_marker_pos]}")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    # Check what OUTPUT_LO should represent
    print(f"\n=== OUTPUT_LO interpretation ===")
    print(f"For byte value 0: nibble-lo = 0, nibble-hi = 0")
    print(f"Expected OUTPUT_LO: one-hot at index 0 → [1, 0, 0, ...] × L")
    print(f"Expected OUTPUT_HI: one-hot at index 0 → [1, 0, 0, ...] × L")

    # Run model
    with torch.no_grad():
        logits = model(token_ids)

    # Examine logit computation
    print(f"\n=== Logits for byte tokens ===")
    byte_logits = logits[0, -1, :256]
    top_vals, top_idx = byte_logits.topk(10)
    print(f"Top 10: {[(idx.item(), f'{val.item():.2f}') for idx, val in zip(top_idx, top_vals)]}")

    # Check logit for byte 0 specifically
    print(f"\nLogit for byte 0: {byte_logits[0].item():.2f}")
    print(f"Logit for byte 240 (0xF0): {byte_logits[240].item():.2f}")

    # Check the unembedding mechanism
    print(f"\n=== Unembedding mechanism ===")
    print(f"Byte 0: nibble-lo=0, nibble-hi=0 → EMBED_LO[0]=1, EMBED_HI[0]=1")
    print(f"Byte 240 (0xF0): nibble-lo=0, nibble-hi=15 → EMBED_LO[0]=1, EMBED_HI[15]=1")

    # What does the final hidden state look like?
    x = model.embed(token_ids)
    with torch.no_grad():
        for block in model.blocks:
            x = block(x)

    print(f"\n=== Final hidden state at BP marker ===")
    print(f"OUTPUT_LO[0:4]: {x[0, -1, BD.OUTPUT_LO:BD.OUTPUT_LO+4].tolist()}")
    print(f"OUTPUT_HI[0:4]: {x[0, -1, BD.OUTPUT_HI:BD.OUTPUT_HI+4].tolist()}")
    print(f"OUTPUT_HI[12:16]: {x[0, -1, BD.OUTPUT_HI+12:BD.OUTPUT_HI+16].tolist()}")

    # For byte 0 vs byte 240:
    # Byte 0: wants OUTPUT_LO[0]=high, OUTPUT_HI[0]=high, others low
    # Byte 240: wants OUTPUT_LO[0]=high, OUTPUT_HI[15]=high, others low
    # If OUTPUT_HI[15] > OUTPUT_HI[0], then byte 240 wins

if __name__ == "__main__":
    main()
