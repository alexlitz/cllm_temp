#!/usr/bin/env python3
"""Check PC value during step 0 generation."""
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
print(f"Initial context: {len(context)} tokens\n")

# Generate step 0
print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        print(f"Step 0 complete at {len(context)-1} tokens\n")
        break

# Check PC at STEP_END position
se_pos = len(context) - 1
token_ids = torch.tensor([context], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Check after Layer 3 (where PC is written)
    for i in range(4):
        x = model.blocks[i](x, kv_cache=None)

    print(f"After Layer 3, at STEP_END position {se_pos}:")
    print(f"  HAS_SE: {x[0, se_pos, BD.HAS_SE].item():.3f}")
    print(f"  MARK_PC: {x[0, se_pos, BD.MARK_PC].item():.3f}")

    # Check OUTPUT_LO/HI for PC
    print(f"\n  OUTPUT_LO values:")
    for i in range(16):
        val = x[0, se_pos, BD.OUTPUT_LO + i].item()
        if abs(val) > 0.1:
            print(f"    OUTPUT_LO[{i}] = {val:.3f}")

    print(f"\n  OUTPUT_HI values:")
    for i in range(16):
        val = x[0, se_pos, BD.OUTPUT_HI + i].item()
        if abs(val) > 0.1:
            print(f"    OUTPUT_HI[{i}] = {val:.3f}")

    # Continue through all layers
    for i in range(4, 16):
        x = model.blocks[i](x, kv_cache=None)

    print(f"\nAfter Layer 15 (final), at STEP_END position:")

    # Decode PC from OUTPUT
    pc_bytes = []
    for i in range(4):
        vals = x[0, se_pos, BD.OUTPUT_LO + i*16 : BD.OUTPUT_LO + (i+1)*16]
        byte_val = vals.argmax(-1).item()
        pc_bytes.append(byte_val)
    pc = sum(b << (i*8) for i, b in enumerate(pc_bytes))

    print(f"  PC bytes: {[hex(b) for b in pc_bytes]}")
    print(f"  PC: 0x{pc:08x}")
    print(f"  Expected: 0x0000000a (10)")

    if pc == 0x0a:
        print("\n✓ PC IS CORRECT!")
    else:
        print(f"\n✗ PC IS WRONG (expected 0x0a, got 0x{pc:08x})")

        # Debug: check what Layer 3 FFN units contributed
        print("\nDEBUG: Checking Layer 3 FFN...")
        x = model.embed(token_ids)
        for i in range(3):
            x = model.blocks[i](x, kv_cache=None)

        # Apply FFN manually to see unit contributions
        ffn = model.blocks[3].ffn
        residual = x
        x_norm = ffn.norm(x)

        up = torch.matmul(x_norm, ffn.W_up.T) + ffn.b_up
        gate = torch.matmul(x_norm, ffn.W_gate.T) + ffn.b_gate
        hidden = torch.nn.functional.silu(up) * gate

        print(f"\nLayer 3 FFN unit activations at position {se_pos}:")
        print(f"  (checking units that write to OUTPUT_LO[10])")

        # Find units that write to OUTPUT_LO[10]
        output_lo_10_writers = (ffn.W_down[BD.OUTPUT_LO + 10, :] != 0).nonzero(as_tuple=True)[0]
        print(f"  Units writing to OUTPUT_LO[10]: {output_lo_10_writers.tolist()}")

        for unit in output_lo_10_writers[:10]:  # Check first 10
            act = hidden[0, se_pos, unit].item()
            weight = ffn.W_down[BD.OUTPUT_LO + 10, unit].item()
            contrib = act * weight
            print(f"    Unit {unit}: activation={act:.3f}, weight={weight:.3f}, contribution={contrib:.3f}")
