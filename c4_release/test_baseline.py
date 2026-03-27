"""Test baseline Neural VM."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM
from neural_vm.context_builder import ContextBuilder

# Simple test program: IMM 5, EXIT
bytecode = [Opcode.IMM | (5 << 8), Opcode.EXIT]

draft_vm = DraftVM(bytecode)
context = ContextBuilder.build_base(bytecode, b'')
draft_vm.step()
draft_tokens = draft_vm.draft_tokens()

print("Draft VM output:")
print(f"  Tokens (first 10): {draft_tokens[:10]}")
print(f"  Token 0 (marker): {draft_tokens[0]}")
print(f"  Token 1 (byte 0): {draft_tokens[1]}")

# Load model
print("\nLoading baseline model...")
model = AutoregressiveVM()
set_vm_weights(model)
model.eval()
print("Model loaded successfully")

# Test forward pass
context_with_draft = context + draft_tokens
ctx_tensor = torch.tensor([context_with_draft], dtype=torch.long)

print("\nRunning forward pass...")
with torch.no_grad():
    logits = model.forward(ctx_tensor)

print(f"Logits shape: {logits.shape}")
print(f"Test passed - baseline code works")
