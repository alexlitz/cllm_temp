#!/usr/bin/env python3
"""Debug how opcode flags get from code bytes to PC marker."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("=" * 80)
print("OPCODE FLAG RELAY ANALYSIS")
print("=" * 80)

# Find opcode byte position in CODE section
code_start = context.index(Token.CODE_START)
code_end = context.index(Token.CODE_END)
imm_opcode_pos = code_start + 1  # First byte after CODE_START

print(f"CODE section: positions {code_start}-{code_end}")
print(f"IMM opcode at position {imm_opcode_pos}: {context[imm_opcode_pos]} (should be {Opcode.IMM})")
print()

# Generate step 0 to get PC marker
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    if tok == Token.STEP_END:
        break

# Find PC marker in step 0
pc_marker_pos = None
for i in range(20, len(context)):
    if context[i] == Token.REG_PC:
        pc_marker_pos = i
        break

print(f"REG_PC marker at position {pc_marker_pos}")
print()

token_ids = torch.tensor([context[:pc_marker_pos+1]], dtype=torch.long, device='cuda')

with torch.no_grad():
    x = model.embed(token_ids)

    # Check OP_IMM at opcode byte position (in CODE section)
    print(f"At IMM opcode byte (position {imm_opcode_pos}):")
    op_imm_at_code = x[0, imm_opcode_pos, BD.OP_IMM].item()
    print(f"  OP_IMM: {op_imm_at_code:.3f}")
    if abs(op_imm_at_code - 1.0) < 0.01:
        print(f"  ✓ OP_IMM is set in embedding")
    else:
        print(f"  ✗ OP_IMM not set in opcode byte embedding")

    # Check at PC marker after embedding
    print(f"\nAt REG_PC marker (position {pc_marker_pos}) after embedding:")
    op_imm_at_pc = x[0, pc_marker_pos, BD.OP_IMM].item()
    print(f"  OP_IMM: {op_imm_at_pc:.3f}")

    # Run through layers and check when OP_IMM appears at PC marker
    for layer_idx in range(16):
        x_before = x.clone()
        x = model.blocks[layer_idx](x, kv_cache=None)

        op_imm_before = x_before[0, pc_marker_pos, BD.OP_IMM].item()
        op_imm_after = x[0, pc_marker_pos, BD.OP_IMM].item()

        if abs(op_imm_after - op_imm_before) > 0.01:
            print(f"\nLayer {layer_idx}:")
            print(f"  OP_IMM at PC marker: {op_imm_before:.3f} → {op_imm_after:.3f}")
            if abs(op_imm_after - 1.0) < 0.1:
                print(f"  ✓ OP_IMM flag appears at PC marker in Layer {layer_idx}")
                break

    # Final check
    final_op_imm = x[0, pc_marker_pos, BD.OP_IMM].item()
    print(f"\nFinal OP_IMM at PC marker: {final_op_imm:.3f}")

    if abs(final_op_imm - 1.0) < 0.1:
        print("✓ OP_IMM successfully relayed to PC marker")
    else:
        print("✗ OP_IMM NOT relayed to PC marker")
        print("\nThis means the opcode detection mechanism is broken.")
        print("There should be an attention head that copies opcode flags from")
        print("the CODE section bytes to the PC marker position.")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The opcode flags (OP_IMM, OP_EXIT, etc.) are set in the opcode byte embeddings.
But they need to be relayed to the PC marker so that subsequent layers can use
them for routing decisions.

Expected flow:
1. CODE section contains opcode bytes with OP_* flags in embeddings
2. Some layer (probably Layer 1-3) should look back from PC marker to CODE
3. That layer copies the current instruction's opcode flags to PC marker
4. Layer 5 then relays these flags from PC marker to AX marker
5. Layer 6 uses OP_IMM to route FETCH → OUTPUT
""")
