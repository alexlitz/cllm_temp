#\!/usr/bin/env python3
"""Debug Step 3 Token 21 (STACK0 byte 0) - expected 1, got 0."""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

# Test program: IMM 1, PSH, IMM 0, MUL, EXIT
BYTECODE = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

def build_context(bytecode, data=b''):
    """Build initial context."""
    context = []
    context.append(Token.CODE_START)
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
    print(f"Initial context length: {len(context)}")

    draft = DraftVM(BYTECODE)

    for step in range(3):
        draft.step()
        draft_tokens = draft.draft_tokens()
        context.extend(draft_tokens)
        stack0 = draft._mem_read(draft.sp)
        print(f"After step {step+1}: PC={draft.pc:#x}, AX={draft.ax:#x}, SP={draft.sp:#x}, mem[SP]={stack0:#x}")

    step_start = len(context) - 35
    print(f"\n=== Step 3 Token Analysis ===")
    print(f"Step starts at context position {step_start}")

    # Token 21 is STACK0 byte 0
    target_token = 21
    context_for_pred = context[:step_start + target_token]

    print(f"Context length for predicting token {target_token}: {len(context_for_pred)}")
    expected = context[step_start + target_token]
    print(f"Draft says STACK0 byte 0 should be: {expected}")

    # Get model prediction
    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    with torch.no_grad():
        logits = model(token_ids)
    predicted = logits[0, -1, :].argmax().item()
    print(f"Model predicts: {predicted}")

    # Get top-5 predictions
    top_logits, top_tokens = logits[0, -1, :].topk(5)
    print(f"\nTop 5 predictions:")
    for logit, tok in zip(top_logits.tolist(), top_tokens.tolist()):
        print(f"  Token {tok}: logit={logit:.4f}")

    # Trace through layers
    x = model.embed(token_ids)
    query_pos = len(context_for_pred) - 1

    print(f"\n=== Layer-by-layer trace at position {query_pos} ===")

    print(f"\nInput embedding at STACK0 marker (pos {query_pos}):")
    print(f"  MARK_STACK0: {x[0, query_pos, BD.MARK_STACK0].item():.4f}")
    print(f"  IS_BYTE: {x[0, query_pos, BD.IS_BYTE].item():.4f}")

    for layer_idx in range(16):
        with torch.no_grad():
            x = model.blocks[layer_idx](x)

        out_lo = x[0, query_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+4].tolist()

        if layer_idx in [5, 10, 14, 15]:
            print(f"\nLayer {layer_idx} output at pos {query_pos}:")
            print(f"  OUTPUT_LO[0:4]: {[f'{v:.3f}' for v in out_lo]}")
            temp = x[0, query_pos, BD.TEMP:BD.TEMP+8].tolist()
            print(f"  TEMP[0:8]: {[f'{v:.3f}' for v in temp]}")

    final_logits = model.head(x)
    final_pred = final_logits[0, -1, :].argmax().item()
    print(f"\nFinal prediction: {final_pred} (expected: {expected})")

    # Check MEM sections in context
    print(f"\n=== MEM sections in context ===")
    step2_start = 44 + 35
    step2_mem_start = step2_start + 25
    print(f"Step 2 MEM section (tokens {step2_mem_start}-{step2_mem_start+8}):")
    mem_tokens = context[step2_mem_start:step2_mem_start+9]
    print(f"  Tokens: {mem_tokens}")
    if mem_tokens[0] == Token.MEM:
        addr = mem_tokens[1] | (mem_tokens[2] << 8) | (mem_tokens[3] << 16) | (mem_tokens[4] << 24)
        val = mem_tokens[5] | (mem_tokens[6] << 8) | (mem_tokens[7] << 16) | (mem_tokens[8] << 24)
        print(f"  Address: {addr:#x}")
        print(f"  Value: {val:#x}")

if __name__ == "__main__":
    main()
