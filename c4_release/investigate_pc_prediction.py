#!/usr/bin/env python3
"""Investigate WHY transformer predicts wrong PC value."""
import sys
sys.path.insert(0, '.')

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.constants import PC_OFFSET, INSTR_WIDTH

print("INVESTIGATING PC PREDICTION")
print("=" * 70)

# Create model
model = AutoregressiveVM()
model.eval()
set_vm_weights(model)

print(f"\nConfiguration:")
print(f"  PC_OFFSET = {PC_OFFSET}")
print(f"  INSTR_WIDTH = {INSTR_WIDTH}")

# Test different PC values to see what the model predicts
test_cases = [
    # Simple contexts with just code markers
    ("Empty program", [Token.CODE_START, Token.CODE_END, Token.DATA_START, Token.DATA_END]),
    
    # JSR 0 (jump to address 0)
    ("JSR 0", [Token.CODE_START, 3, 0, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]),
    
    # JSR 8 (jump to address 8)
    ("JSR 8", [Token.CODE_START, 3, 8, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]),
    
    # JSR 16 (jump to address 16)
    ("JSR 16", [Token.CODE_START, 3, 16, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]),
    
    # JSR 24 (jump to address 24)
    ("JSR 24", [Token.CODE_START, 3, 24, 0, 0, 0, Token.CODE_END, Token.DATA_START, Token.DATA_END]),
]

print("\nPredictions for different JSR targets:")
print("-" * 70)

for name, context in test_cases:
    token_ids = torch.tensor([context], dtype=torch.long)
    logits = model.forward(token_ids)
    
    # Predict first output token (should be REG_PC)
    pred0 = logits[0, -1, :].argmax(-1).item()
    
    # Predict PC byte 0
    context2 = context + [pred0]
    token_ids2 = torch.tensor([context2], dtype=torch.long)
    logits2 = model.forward(token_ids2)
    pred1 = logits2[0, -1, :].argmax(-1).item()
    
    print(f"\n{name}:")
    print(f"  First token:  {pred0} (expected {Token.REG_PC})")
    print(f"  PC byte 0:    {pred1}")
    
    if name.startswith("JSR"):
        target = int(name.split()[1])
        expected = target
        print(f"  Expected PC:  {expected}")
        print(f"  Difference:   {pred1 - expected}")
        
        # Check if it's predicting "next instruction"
        next_instr_pc = INSTR_WIDTH
        if pred1 == next_instr_pc:
            print(f"  -> Predicting NEXT instruction PC ({next_instr_pc}), not jump target")

print("\n" + "=" * 70)
print("PATTERN ANALYSIS")
print("=" * 70)

print("""
If the model always predicts PC=8 regardless of JSR target:
  -> Model ignores the jump target and just does PC += INSTR_WIDTH

If the model predicts different values for different JSR targets:
  -> Model is reading the jump target but computing wrong result

If the model predicts PC = target - 8:
  -> Possible PC_OFFSET issue (off by one instruction)
""")
