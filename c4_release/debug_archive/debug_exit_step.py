"""
Debug EXIT instruction at step 1.
After IMM, AX=0x2a. EXIT should preserve AX value.
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

# Generate entire step 0 (should end at STEP_END)
print("Generating step 0...")
step_tokens = 0
while step_tokens < 100:
    tok = model.generate_next(context)
    context.append(tok)
    step_tokens += 1
    if tok == Token.STEP_END:
        break

print(f"Step 0 complete, {step_tokens} tokens generated")

# Now generate step 1 up to REG_AX marker
print("\nGenerating step 1 up to REG_AX...")
for _ in range(6):  # REG_PC + 4 bytes + REG_AX
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.REG_AX:
        break

# Show last few tokens
print(f"\nLast 10 tokens in context:")
for i in range(max(0, len(context) - 10), len(context)):
    tok = context[i]
    if tok == Token.REG_PC:
        print(f"  [{i}] REG_PC")
    elif tok == Token.REG_AX:
        print(f"  [{i}] REG_AX")
    elif tok == Token.STEP_END:
        print(f"  [{i}] STEP_END")
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

    x_before_l6 = x.clone()
    x_after_attn = x + model.blocks[6].attn(x, kv_cache=None)

    print(f"\nAt position {pos} (REG_AX marker, step 1):")
    S = 16.0
    T = 5.5

    # Check values
    is_byte = x_after_attn[0, pos, BD.IS_BYTE].item()
    mark_ax = x_after_attn[0, pos, BD.MARK_AX].item()
    mark_pc = x_after_attn[0, pos, BD.MARK_PC].item()
    op_imm = x_after_attn[0, pos, BD.OP_IMM].item()
    op_exit = x_after_attn[0, pos, BD.OP_EXIT].item()

    print(f"  IS_BYTE: {is_byte:.3f}")
    print(f"  MARK_AX: {mark_ax:.3f}")
    print(f"  MARK_PC: {mark_pc:.3f}")
    print(f"  OP_IMM: {op_imm:.3f}")
    print(f"  OP_EXIT: {op_exit:.3f}")

    # Check EXIT unit activation
    activation_exit = S * op_exit + S * mark_ax - S * mark_pc - S * is_byte - S * T
    print(f"\n  EXIT unit activation:")
    print(f"    {S}*{op_exit:.1f} + {S}*{mark_ax:.1f} - {S}*{mark_pc:.1f} - {S}*{is_byte:.1f} - {S}*{T}")
    print(f"    = {activation_exit:.3f}")
    print(f"    Should fire: {activation_exit > 0}")

    # Check AX_CARRY values (what EXIT should copy to OUTPUT)
    print(f"\n  AX_CARRY values (what EXIT copies):")
    for k in range(16):
        ax_lo = x_after_attn[0, pos, BD.AX_CARRY_LO + k].item()
        ax_hi = x_after_attn[0, pos, BD.AX_CARRY_HI + k].item()
        if abs(ax_lo) > 0.1 or abs(ax_hi) > 0.1:
            print(f"    AX_CARRY_LO[{k}]: {ax_lo:.2f}, AX_CARRY_HI[{k}]: {ax_hi:.2f}")

    # FFN
    ffn_out = model.blocks[6].ffn(x_after_attn)
    x_after_ffn = x_after_attn + ffn_out

    print(f"\n  After Layer 6 FFN:")
    print(f"    OUTPUT_LO[10]: {x_after_ffn[0, pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2]: {x_after_ffn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Continue to Layer 15
    for i in range(7, 16):
        x_after_ffn = model.blocks[i](x_after_ffn, kv_cache=None)

    print(f"\n  After Layer 15:")
    print(f"    OUTPUT_LO[10]: {x_after_ffn[0, pos, BD.OUTPUT_LO + 10].item():.3f}")
    print(f"    OUTPUT_HI[2]: {x_after_ffn[0, pos, BD.OUTPUT_HI + 2].item():.3f}")

    # Predict
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
