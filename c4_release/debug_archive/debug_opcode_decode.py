"""
Debug why OP_EXIT isn't being set at step 1.
Trace opcode decoding through Layer 5.
"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

# Build model (force fresh build)
print("Building model from scratch...")
model = AutoregressiveVM()
set_vm_weights(model)
print("Compacting...")
model.compact(block_size=32)
model.compact_moe()
model = model.cuda()
model.eval()

# Build bytecode using packed format like rebuild_and_test.py
value = 42
bytecode = [
    Opcode.IMM | (value << 8),
    Opcode.EXIT,
]

print(f"Bytecode: IMM {value}; EXIT")

runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, b'', [])

# Generate entire step 0
print("\nGenerating step 0...")
step_tokens = 0
generated = []
while step_tokens < 50:  # Reduced to 50 to see what's happening
    tok = model.generate_next(context)
    context.append(tok)
    generated.append(tok)
    step_tokens += 1
    if tok == Token.STEP_END:
        print(f"  STEP_END generated at token {step_tokens}, position {len(context)-1}")
        break

print(f"Step 0: generated {len(generated)} tokens")

# Show what was generated
print(f"\nFirst 40 generated tokens:")
for i, tok in enumerate(generated[:40]):
    if tok == Token.STEP_END:
        print(f"  [{i}] STEP_END")
    elif tok == Token.REG_PC:
        print(f"  [{i}] REG_PC")
    elif tok == Token.REG_AX:
        print(f"  [{i}] REG_AX")
    elif tok < 256:
        print(f"  [{i}] byte_{tok:02x}")

# Check for STEP_END
print(f"\nChecking for STEP_END in context...")
step_end_positions = [i for i, tok in enumerate(context) if tok == Token.STEP_END]
print(f"  STEP_END found at positions: {step_end_positions}")

# Generate step 1 up to REG_PC marker
print("\nGenerating step 1 up to REG_PC...")
for _ in range(1):  # Just REG_PC
    tok = model.generate_next(context)
    context.append(tok)

# Show tokens around STEP_END
if step_end_positions:
    pos = step_end_positions[0]
    print(f"\n  Tokens around STEP_END (position {pos}):")
    for i in range(max(0, pos-3), min(len(context), pos+4)):
        tok = context[i]
        if tok == Token.STEP_END:
            print(f"    [{i}] STEP_END")
        elif tok == Token.REG_PC:
            print(f"    [{i}] REG_PC")
        elif tok < 256:
            print(f"    [{i}] byte_{tok:02x}")

# Forward pass to check opcode decoding
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)
    pos = len(context) - 1  # REG_PC marker position

    print(f"\nAt position {pos} (REG_PC marker, step 1):")

    # Check embedding values
    print(f"\nAfter embedding:")
    print(f"  HAS_SE: {x[0, pos, BD.HAS_SE].item():.3f}")
    print(f"  MARK_PC: {x[0, pos, BD.MARK_PC].item():.3f}")

    # Forward through layers
    for i in range(2):  # Layers 0-1
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter Layer 1 (HAS_SE should be set here):")
    print(f"  HAS_SE: {x[0, pos, BD.HAS_SE].item():.3f}")
    print(f"  MARK_PC: {x[0, pos, BD.MARK_PC].item():.3f}")

    for i in range(2, 5):  # Layers 2-4
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nBefore Layer 5 (opcode decode layer):")
    print(f"  HAS_SE: {x[0, pos, BD.HAS_SE].item():.3f}")
    print(f"  MARK_PC: {x[0, pos, BD.MARK_PC].item():.3f}")
    print(f"  OP_IMM: {x[0, pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_EXIT: {x[0, pos, BD.OP_EXIT].item():.3f}")

    # Check OPCODE_BYTE values (Layer 5 head 1 writes opcode byte here)
    print(f"\n  OPCODE_BYTE values:")
    for k in range(16):
        opcode_lo = x[0, pos, BD.OPCODE_BYTE_LO + k].item()
        opcode_hi = x[0, pos, BD.OPCODE_BYTE_HI + k].item()
        if abs(opcode_lo) > 0.1 or abs(opcode_hi) > 0.1:
            print(f"    OPCODE_BYTE_LO[{k}]: {opcode_lo:.2f}, OPCODE_BYTE_HI[{k}]: {opcode_hi:.2f}")

    # Layer 5 attention
    x_after_l5_attn = x + model.blocks[5].attn(x, kv_cache=None)

    print(f"\nAfter Layer 5 attention:")
    print(f"  OPCODE_BYTE values:")
    for k in range(16):
        opcode_lo = x_after_l5_attn[0, pos, BD.OPCODE_BYTE_LO + k].item()
        opcode_hi = x_after_l5_attn[0, pos, BD.OPCODE_BYTE_HI + k].item()
        if abs(opcode_lo) > 0.1 or abs(opcode_hi) > 0.1:
            print(f"    OPCODE_BYTE_LO[{k}]: {opcode_lo:.2f}, OPCODE_BYTE_HI[{k}]: {opcode_hi:.2f}")

    # Layer 5 FFN (opcode decode)
    x_after_l5 = x_after_l5_attn + model.blocks[5].ffn(x_after_l5_attn)

    print(f"\nAfter Layer 5 FFN (opcode decode):")
    print(f"  OP_IMM: {x_after_l5[0, pos, BD.OP_IMM].item():.3f}")
    print(f"  OP_EXIT: {x_after_l5[0, pos, BD.OP_EXIT].item():.3f}")
    print(f"  OP_NOP: {x_after_l5[0, pos, BD.OP_NOP].item():.3f}")
    print(f"  OP_JMP: {x_after_l5[0, pos, BD.OP_JMP].item():.3f}")

    # Check what the opcode byte decodes to
    # EXIT opcode = 0x02, which is byte_02
    # Nibbles: LO=2, HI=0
    print(f"\n  Expected for EXIT (0x02):")
    print(f"    OPCODE_BYTE_LO[2] should be active")
    print(f"    OPCODE_BYTE_HI[0] should be active")

    # Check if EMBED_LO/HI has the opcode
    print(f"\n  EMBED values at PC:")
    for k in range(16):
        embed_lo = x_after_l5_attn[0, pos, BD.EMBED_LO + k].item()
        embed_hi = x_after_l5_attn[0, pos, BD.EMBED_HI + k].item()
        if abs(embed_lo) > 0.1 or abs(embed_hi) > 0.1:
            print(f"    EMBED_LO[{k}]: {embed_lo:.2f}, EMBED_HI[{k}]: {embed_hi:.2f}")

    # Check CLEAN_EMBED
    print(f"\n  CLEAN_EMBED values at PC:")
    for k in range(16):
        clean_lo = x_after_l5_attn[0, pos, BD.CLEAN_EMBED_LO + k].item()
        if abs(clean_lo) > 0.1:
            print(f"    CLEAN_EMBED_LO[{k}]: {clean_lo:.2f}")
