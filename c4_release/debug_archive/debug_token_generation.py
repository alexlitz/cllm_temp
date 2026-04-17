#!/usr/bin/env python3
"""Trace token-by-token generation to find where 0x01 appears."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("Starting generation from initial context")
print(f"Initial length: {len(context)}\n")

print("TOKEN-BY-TOKEN GENERATION:")
print("=" * 80)

token_map = {
    257: "REG_PC",
    258: "REG_BP",
    259: "REG_SP",
    260: "REG_AX",
    261: "REG_CYCLE",
    262: "STEP_END",
    263: "MEM",
}

reg_ax_found = False
bytes_after_ax = []

for i in range(60):
    token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

    with torch.no_grad():
        logits = model.forward(token_ids)
        # Get prediction at last position
        pred_logits = logits[0, -1, :256]
        predicted = pred_logits.argmax(-1).item()

        # Generate next token
        next_token = model.generate_next(context)

    context.append(next_token)

    # Display the token
    if next_token in token_map:
        token_str = token_map[next_token]
    elif next_token < 256:
        token_str = f"0x{next_token:02x}"
    else:
        token_str = str(next_token)

    print(f"Token {i:2d} (pos {len(context)-1:2d}): {token_str:15s}", end="")

    # If this is right after REG_AX, show prediction vs actual
    if reg_ax_found and len(bytes_after_ax) < 4:
        print(f"  [predicted: 0x{predicted:02x}, actual: 0x{next_token:02x}]", end="")
        if predicted != next_token:
            print(" ← MISMATCH!", end="")
        bytes_after_ax.append(next_token)

    print()

    if next_token == 260:  # REG_AX
        reg_ax_found = True
        print("  └─ REG_AX marker found, tracking next 4 bytes...")

    if next_token == Token.STEP_END:
        print("\nSTEP_END reached")
        break

if bytes_after_ax:
    print("\n" + "=" * 80)
    print("BYTES AFTER REG_AX:")
    for i, byte_val in enumerate(bytes_after_ax):
        print(f"  Byte {i}: 0x{byte_val:02x}")

    ax_value = sum(b << (i*8) for i, b in enumerate(bytes_after_ax[:4]))
    print(f"\nAX value: 0x{ax_value:08x} ({ax_value})")
    print(f"Expected: 0x0000002a (42)")
    print(f"Status: {'PASS ✓' if ax_value == 42 else 'FAIL ✗'}")
