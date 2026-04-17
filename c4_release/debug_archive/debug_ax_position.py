"""
Debug what's happening at AX byte position where IMM should write 0x2a.
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

# Build initial context
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate first 6 tokens (REG_PC + 4 PC bytes + REG_AX)
for _ in range(6):
    tok = model.generate_next(context)
    context.append(tok)

# Now context ends at REG_AX marker
# Next token should be first AX byte (byte_2a = 0x2a = 42)

print(f"Context length: {len(context)}")
print(f"Last few tokens:")
for i in range(max(0, len(context) - 6), len(context)):
    tok = context[i]
    if tok == Token.REG_PC:
        print(f"  [{i}] REG_PC")
    elif tok == Token.REG_AX:
        print(f"  [{i}] REG_AX")
    elif tok < 256:
        print(f"  [{i}] byte_{tok:02x}")

# Forward pass
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    pos = len(context) - 1  # REG_AX marker position

    # Forward to Layer 6
    for i in range(6):
        x = model.blocks[i](x, kv_cache=None)

    # Layer 6 attention
    x_after_attn = x + model.blocks[6].attn(x, kv_cache=None)

    print(f"\nAt position {pos} (REG_AX marker), after Layer 6 attention:")
    S = 16.0
    T = 9.5
    mark_pc = x_after_attn[0, pos, BD.MARK_PC].item()
    mark_ax = x_after_attn[0, pos, BD.MARK_AX].item()
    op_imm = x_after_attn[0, pos, BD.OP_IMM].item()

    print(f"  MARK_PC: {mark_pc:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  OP_IMM: {op_imm:.3f}")

    # IMM unit activation
    activation = S * op_imm + S * mark_ax - S * mark_pc - S * T
    print(f"\n  IMM unit activation:")
    print(f"    {S} * {op_imm:.3f} + {S} * {mark_ax:.3f} - {S} * {mark_pc:.3f} - {S} * {T}")
    print(f"    = {activation:.3f}")
    print(f"    Should fire: {activation > 0}")

    if activation > 0:
        # Check FETCH values
        print(f"\n  FETCH values:")
        for k in range(16):
            fetch_lo = x_after_attn[0, pos, BD.FETCH_LO + k].item()
            fetch_hi = x_after_attn[0, pos, BD.FETCH_HI + k].item()
            if abs(fetch_lo) > 0.1 or abs(fetch_hi) > 0.1:
                print(f"    FETCH_LO[{k}]: {fetch_lo:.2f}, FETCH_HI[{k}]: {fetch_hi:.2f}")

    # Now FFN
    ffn_out = model.blocks[6].ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out

    print(f"\n  After Layer 6 FFN:")
    print(f"    OUTPUT_LO[10] (for nibble A): {x_after_ffn[0, pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2] (for nibble 2): {x_after_ffn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Continue to Layer 15
    for i in range(7, 16):
        x_after_ffn = model.blocks[i](x_after_ffn, kv_cache=None)

    print(f"\n  After Layer 15:")
    print(f"    OUTPUT_LO[10] (for nibble A): {x_after_ffn[0, pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2] (for nibble 2): {x_after_ffn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Predict next token
    logits = model.head(x_after_ffn)
    next_token = logits[0, -1, :].argmax(-1).item()

    print(f"\n  Predicted next token: ", end='')
    if next_token < 256:
        print(f"byte_{next_token:02x}")
        if next_token == 0x2a:
            print(f"    ✓ CORRECT!")
        else:
            print(f"    ✗ WRONG! Expected byte_2a (0x2a = 42)")
    else:
        print(f"{next_token} (not a byte)")
