#!/usr/bin/env python3
"""Quick test of JSR PC prediction."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights

# Create model
model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

# JSR 16 (jump to address 16)
# Bytecode: [JSR(3), target(16), 0, 0, 0]
context = [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]

token_ids = torch.tensor([context], dtype=torch.long)
logits = model.forward(token_ids)

# Predict first output token
pred0 = logits[0, -1, :].argmax(-1).item()
print(f"First token:  {pred0} (expected {Token.REG_PC}={Token.REG_PC})")

# Predict PC byte 0
context2 = context + [pred0]
token_ids2 = torch.tensor([context2], dtype=torch.long)
logits2 = model.forward(token_ids2)
pred1 = logits2[0, -1, :].argmax(-1).item()

print(f"PC byte 0:    {pred1} (expected 16)")
print(f"Match: {'✓' if pred1 == 16 else '✗'}")
