#!/usr/bin/env python3
"""Test JSR with multiple targets."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights

model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

test_cases = [
    ("JSR 0", 0),
    ("JSR 8", 8),
    ("JSR 16", 16),
    ("JSR 24", 24),
    ("JSR 32", 32),
    ("JSR 128", 128),
]

print("Testing JSR PC prediction with multiple targets:")
print("=" * 50)

all_pass = True
for name, target in test_cases:
    context = [Token.CODE_START, 3, target, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]

    token_ids = torch.tensor([context], dtype=torch.long)
    logits = model.forward(token_ids)
    pred0 = logits[0, -1, :].argmax(-1).item()

    context2 = context + [pred0]
    token_ids2 = torch.tensor([context2], dtype=torch.long)
    logits2 = model.forward(token_ids2)
    pred1 = logits2[0, -1, :].argmax(-1).item()

    match = "✓" if pred1 == target else "✗"
    all_pass = all_pass and (pred1 == target)
    print(f"{name:10} → PC byte 0 = {pred1:3} (expected {target:3}) {match}")

print("=" * 50)
print(f"Result: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
