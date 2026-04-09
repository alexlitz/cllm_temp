#!/usr/bin/env python3
"""Check HAS_SE during step 0 generation (not after)."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

# Use runner's context builder (no manual STEP_END)
context = runner._build_context(bytecode, b'', [])
print(f"Initial context: {len(context)} tokens")
print(f"Context: {context}")

# Generate one token at a time and check HAS_SE
print("\nGenerating step 0 tokens...")
for i in range(40):
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

    with torch.no_grad():
        x = model.embed(token_ids)

        # Check HAS_SE after embedding (before any layers)
        if i == 0:
            print(f"Token {i}: After embedding, HAS_SE values:")
            for pos in range(len(context)):
                has_se = x[0, pos, BD.HAS_SE].item()
                if has_se > 0.01:
                    print(f"  pos {pos} (token {context[pos]}): HAS_SE={has_se:.3f}")

        # Run through all layers
        for layer_idx in range(16):
            x = model.blocks[layer_idx](x, kv_cache=None)

            # After Layer 1, check if HAS_SE was set
            if layer_idx == 1 and i == 0:
                print(f"\nAfter Layer 1:")
                for pos in range(len(context)):
                    has_se = x[0, pos, BD.HAS_SE].item()
                    if has_se > 0.01:
                        print(f"  pos {pos} (token {context[pos]}): HAS_SE={has_se:.3f}")

        # Get next token
        logits = model.head(x)
        tok = logits[0, -1, :].argmax(-1).item()

    context.append(tok)

    if i == 0:
        print(f"\nFirst generated token: {tok} ({hex(tok) if tok < 256 else tok})")

    if tok == Token.STEP_END:
        print(f"\nStep 0 complete at token {i}")

        # Now check HAS_SE at various positions BEFORE the STEP_END
        token_ids = torch.tensor([context], dtype=torch.long, device='cuda')
        with torch.no_grad():
            x = model.embed(token_ids)
            for layer_idx in range(16):
                x = model.blocks[layer_idx](x, kv_cache=None)

        # Check at position just before STEP_END
        se_pos = len(context) - 1
        before_se = se_pos - 1

        print(f"\nAt position {before_se} (before STEP_END):")
        print(f"  HAS_SE: {x[0, before_se, BD.HAS_SE].item():.6f}")

        print(f"\nAt position {se_pos} (STEP_END itself):")
        print(f"  HAS_SE: {x[0, se_pos, BD.HAS_SE].item():.6f}")

        break
