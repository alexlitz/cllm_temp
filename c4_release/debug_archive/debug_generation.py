"""
Debug the actual autoregressive generation process.
Simulates what rebuild_and_test.py does.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model
print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Build initial context like runner does
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

print(f"Initial context: {len(context)} tokens")
print(f"Bytecode: IMM {value}; EXIT\n")

# Generate first few tokens with diagnostics
for step in range(3):
    print(f"=" * 60)
    print(f"Generation step {step}")
    print(f"=" * 60)

    # Forward pass
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

    with torch.no_grad():
        x = model.embed(token_ids)

        # Check embedding at last position
        last_pos = len(context) - 1
        print(f"Last position ({last_pos}): ", end='')
        tok = context[last_pos]
        if tok == Token.STEP_END:
            print("STEP_END")
        elif tok < 256:
            print(f"byte_{tok:02x}")
        else:
            for name, val in vars(Token).items():
                if isinstance(val, int) and val == tok and name.isupper():
                    print(name)
                    break

        # Layer by layer - only trace step 1 (after REG_PC)
        if step == 1:
            for i, block in enumerate(model.blocks):
                # Trace Layer 6 in detail (attention vs FFN)
                if i == 6:
                    # Attention sublayer
                    x_attn = x + block.attn(x, kv_cache=None)
                    output_hi_2_attn = x_attn[0, last_pos, BD.OUTPUT_HI + 2].item()
                    print(f"After blocks[6] attention: OUTPUT_HI[2] = {output_hi_2_attn:.3f}")

                    # FFN sublayer
                    x = x_attn + block.ffn(x_attn)
                    output_hi_2_ffn = x[0, last_pos, BD.OUTPUT_HI + 2].item()
                    print(f"After blocks[6] FFN: OUTPUT_HI[2] = {output_hi_2_ffn:.3f}")
                else:
                    x = block(x, kv_cache=None)

                # Check all layers for OUTPUT_HI[2]
                output_hi_2 = x[0, last_pos, BD.OUTPUT_HI + 2].item()
                if abs(output_hi_2) > 0.1 and i != 6:
                    print(f"After blocks[{i}] (Layer {i}): OUTPUT_HI[2] = {output_hi_2:.3f}")
        else:
            # Normal processing for other steps
            for i, block in enumerate(model.blocks):
                x = block(x, kv_cache=None)

        # Show final OUTPUT values for step 1
        if step == 1:
            print(f"\nFinal OUTPUT values at position {last_pos}:")
            print("  OUTPUT_LO:", end='')
            for k in range(16):
                lo = x[0, last_pos, BD.OUTPUT_LO + k].item()
                if abs(lo) > 0.1:
                    print(f" [{k}]={lo:.2f}", end='')
            print()
            print("  OUTPUT_HI:", end='')
            for k in range(16):
                hi = x[0, last_pos, BD.OUTPUT_HI + k].item()
                if abs(hi) > 0.1:
                    print(f" [{k}]={hi:.2f}", end='')
            print()

        # Predict next token
        logits = model.head(x)
        next_token = logits[0, -1, :].argmax(-1).item()

        # Show prediction
        print(f"\nPredicted token: ", end='')
        if next_token == Token.REG_PC:
            print("REG_PC (correct)")
        elif next_token == Token.REG_AX:
            print("REG_AX")
        elif next_token == Token.STEP_END:
            print("STEP_END")
        elif next_token < 256:
            print(f"byte_{next_token:02x}")
            # If this is a PC byte, show what it should be
            if step == 1:  # First byte after REG_PC
                expected = 0x0a  # PC should advance by 8 (INSTR_WIDTH) from 0x02 to 0x0a
                if next_token == expected:
                    print(f"  ✓ Correct PC byte")
                else:
                    print(f"  ✗ Wrong! Expected 0x{expected:02x}, got 0x{next_token:02x}")
        else:
            for name, val in vars(Token).items():
                if isinstance(val, int) and val == next_token and name.isupper():
                    print(name)
                    break

        context.append(next_token)

        # Stop after a few steps
        if step >= 10:
            break

    print()

print("\nFirst 20 generated tokens:")
initial_len = len(runner._build_context(bytecode, b'', []))
for i in range(min(20, len(context) - initial_len)):
    tok = context[initial_len + i]
    if tok < 256:
        print(f"  [{i}] byte_{tok:02x}")
    else:
        for name, val in vars(Token).items():
            if isinstance(val, int) and val == tok and name.isupper():
                print(f"  [{i}] {name}")
                break
