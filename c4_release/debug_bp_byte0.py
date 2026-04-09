#!/usr/bin/env python3
"""Debug Step 4 Token 16: BP byte 0 (expected 0, got 240)."""

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

    # Run 3 steps to get to step 4
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    print(f"=== Step 4 draft tokens ===")
    print(f"BP marker: token 15 = {step4_tokens[15]}")
    print(f"BP bytes: {step4_tokens[16:20]}")
    print(f"Expected BP byte 0: {step4_tokens[16]}")
    print(f"Got: 240 (0xF0)")

    # Build context up to BP byte 0 prediction
    target_token = 16  # BP byte 0
    context_for_pred = context + step4_tokens[:target_token]
    query_pos = len(context_for_pred) - 1

    print(f"\n=== Context for BP byte 0 prediction ===")
    print(f"Query position: {query_pos} (token {context_for_pred[-1]})")
    print(f"Predicting token at position {query_pos + 1}")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    # Trace per-layer OUTPUT_LO changes
    print(f"\n=== Per-layer OUTPUT_LO at BP marker (pos {query_pos}) ===")
    with torch.no_grad():
        for layer_idx in range(16):
            x_before = x.clone()
            x = model.blocks[layer_idx](x)
            delta = x[0, query_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+8] - x_before[0, query_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+8]
            if delta.abs().max() > 0.1:
                print(f"L{layer_idx}: delta OUTPUT_LO = {delta.tolist()[:4]}")

    print(f"\n=== Final OUTPUT_LO at BP marker ===")
    output_lo = x[0, query_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+8].tolist()
    print(f"OUTPUT_LO: {output_lo[:4]}")

    # Check what logit values are
    logits = model(token_ids)
    top_logits, top_indices = logits[0, -1, :256].topk(5)
    print(f"\n=== Top 5 logits ===")
    for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
        print(f"  {idx} (0x{idx:02x}): {logit:.2f}")

    # Check step 3 BP values for reference
    print(f"\n=== Step 3 BP section ===")
    step3_start = len(context) - 35  # Start of step 3
    bp_marker_pos = step3_start + 15
    print(f"Step 3 BP marker at pos {bp_marker_pos}")
    print(f"Step 3 BP bytes: {context[bp_marker_pos+1:bp_marker_pos+5]}")

    # Check what MARK_BP and related dims look like at query position
    print(f"\n=== Marker dims at BP marker position ===")
    x_embed = model.embed(token_ids)
    for layer_idx in range(16):
        with torch.no_grad():
            x_embed = model.blocks[layer_idx](x_embed)

    print(f"MARK_BP: {x_embed[0, query_pos, BD.MARK_BP].item():.2f}")
    print(f"MARK_SP: {x_embed[0, query_pos, BD.MARK_SP].item():.2f}")
    print(f"BYTE_INDEX_0: {x_embed[0, query_pos, BD.BYTE_INDEX_0].item():.2f}")

    # Check SP embedding at step 4
    print(f"\n=== SP values ===")
    print(f"Draft SP: {draft.sp:#x}")

if __name__ == "__main__":
    main()
