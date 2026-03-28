#!/usr/bin/env python3
"""Trace what actually happens during speculative decoding."""
import sys
sys.path.insert(0, '.')

import torch
from src.compiler import compile_c
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.speculative import DraftVM

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

# Build initial context
context = [Token.CODE_START]
for instr in bytecode:
    op = instr & 0xFF
    imm = instr >> 8
    context.append(op)
    for i in range(4):
        context.append((imm >> (i * 8)) & 0xFF)
context.append(Token.CODE_END)
context.append(Token.DATA_START)
context.append(Token.DATA_END)

print("SPECULATIVE DECODING TRACE")
print("=" * 70)
print(f"\nInitial context length: {len(context)}")

# Create model
model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# Create DraftVM
vm = DraftVM(bytecode)

# Simulate 3 steps of speculative execution
for step in range(3):
    print(f"\n{'=' * 70}")
    print(f"STEP {step}")
    print(f"{'=' * 70}")
    
    # DraftVM generates draft tokens
    print(f"\nDraftVM state before step:")
    print(f"  idx={vm.idx}, pc={vm.pc}, ax={vm.ax}")
    
    vm.step()
    draft = vm.draft_tokens()
    
    print(f"\nDraftVM state after step:")
    print(f"  idx={vm.idx}, pc={vm.pc}, ax={vm.ax}")
    print(f"  Draft tokens (first 10): {draft[:10]}")
    
    # Transformer validates
    contexts_with_draft = [context + draft]
    token_ids = torch.tensor(contexts_with_draft, dtype=torch.long)
    logits = model.forward(token_ids)
    
    ctx_len = len(context)
    accepted = 0
    mismatches = []
    
    for i in range(35):  # draft_lens = 35
        pred = logits[0, ctx_len - 1 + i, :].argmax(-1).item()
        if draft[i] == pred:
            accepted += 1
        else:
            mismatches.append((i, draft[i], pred))
            break  # Stop at first mismatch
    
    print(f"\nValidation:")
    print(f"  Accepted: {accepted}/35 tokens")
    
    if mismatches:
        i, expected, predicted = mismatches[0]
        print(f"  First mismatch at token {i}:")
        print(f"    Draft:       {expected}")
        print(f"    Transformer: {predicted}")
    
    # Update context (what batch_runner does)
    print(f"\nContext update:")
    print(f"  Before: len={len(context)}")
    context = context + draft[:accepted]
    print(f"  After:  len={len(context)} (added {accepted} tokens)")
    
    if accepted < 35:
        print(f"\n  ⚠️ Only accepted {accepted}/35 tokens!")
        print(f"  DraftVM is at idx={vm.idx}, but context is incomplete")
        print(f"  Next iteration: DraftVM will generate from idx={vm.idx}")
        print(f"              but context is missing {35-accepted} tokens from this step")
        break
    else:
        print(f"  ✓ Accepted full VM step (35 tokens)")

print(f"\n{'=' * 70}")
print("ANALYSIS")
print(f"{'=' * 70}")

if accepted < 35:
    print("""
The speculation breaks down when accepted < 35:

1. DraftVM executed a full instruction (JSR)
2. DraftVM is now at the next instruction (idx=2, pc=16)
3. But only 1 token was accepted (REG_PC marker)
4. Context is missing the other 34 tokens (PC value, AX, SP, BP, etc.)

Next iteration:
5. DraftVM generates 35 tokens from its CURRENT state (idx=2, pc=16)
6. These tokens represent the state AFTER executing instruction 2
7. But the context is still incomplete from instruction 0
8. The tokens will be completely wrong for the context

This is a fundamental problem: VM steps are ATOMIC (35 tokens),
but speculative decoding treats them as INDIVIDUAL tokens.
    """)
else:
    print("\nFull step accepted - speculation working correctly!")
