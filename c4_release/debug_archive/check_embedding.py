"""Check embedding values for token 0 and STACK0."""
import sys
if 'neural_vm.vm_step' in sys.modules:
    del sys.modules['neural_vm.vm_step']

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD

model = AutoregressiveVM()
set_vm_weights(model)

# Get embedding table
embed_layer = model.embed.token_embed if hasattr(model.embed, 'token_embed') else model.embed.embed
embed = embed_layer.weight.data

print("Embedding Analysis")
print("=" * 70)
print()

# Check token 0 (byte)
print("Token 0 (byte):")
print(f"  IS_BYTE: {embed[0, BD.IS_BYTE]:.3f}")
print(f"  MARK_STACK0: {embed[0, BD.MARK_STACK0]:.3f}")
print()

# Check token 268 (STACK0 marker)
print(f"Token {Token.STACK0} (STACK0 marker):")
print(f"  IS_BYTE: {embed[Token.STACK0, BD.IS_BYTE]:.3f}")
print(f"  MARK_STACK0: {embed[Token.STACK0, BD.MARK_STACK0]:.3f}")
print()

print("Expected:")
print("  Token 0 should have: IS_BYTE=1.0, MARK_STACK0=0.0")
print("  Token 268 should have: IS_BYTE=0.0, MARK_STACK0=1.0")
