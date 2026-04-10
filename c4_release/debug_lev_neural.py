#!/usr/bin/env python3
"""
Debug LEV neural implementation - trace why L15 heads 4-11 return zeros.
"""

import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

import torch
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.dim_registry import DimRegistry

# Simple function that returns a value
code = '''
int helper() {
    return 42;
}

int main() {
    return helper();
}
'''

print("Compiling code...")
bytecode, data = compile_c(code, link_stdlib=False)
print(f"✓ Compiled: {len(bytecode)} bytes of bytecode\n")

print("Creating runner with n_layers=17...")
runner = AutoregressiveVMRunner(n_layers=17, pure_attention_memory=False)

# Build context
from neural_vm.context_builder import build_vm_context
context = build_vm_context(bytecode, data, argv=[])
print(f"✓ Context built: {len(context)} tokens\n")

# Run until first LEV operation
print("Running until first LEV operation...")
from neural_vm.embedding import Token, Opcode
import torch.nn.functional as F

generation_context = []
max_steps = 100
BD = DimRegistry()

for step in range(max_steps):
    # Generate next token
    input_ids = torch.tensor([context + generation_context], dtype=torch.long)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    with torch.no_grad():
        logits = runner.model.forward(input_ids)[0, -1, :]
        next_token = torch.argmax(logits).item()

    generation_context.append(next_token)

    # Check if this is a LEV opcode
    if next_token == Token.OP_LEV:
        print(f"\n{'='*60}")
        print(f"STEP {step}: Found OP_LEV at token position {len(context) + len(generation_context) - 1}")
        print(f"{'='*60}\n")

        # Run one more forward pass to see what happens at BP/PC markers
        print("Running forward pass to trace LEV execution...\n")

        # Add a few more tokens to get to the markers
        for i in range(10):
            input_ids = torch.tensor([context + generation_context], dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            with torch.no_grad():
                # Run forward and capture L15 activations
                x = runner.model.embed(input_ids)

                # Run through layers up to L15
                for layer_idx in range(15):
                    x = runner.model.blocks[layer_idx](x)

                # L15 - capture attention weights for heads 4-11
                print(f"\n--- Before L15 (step {i}) ---")
                print(f"Input shape: {x.shape}")

                # Run L15
                l15_attn = runner.model.blocks[15].attn
                x_after_l15 = runner.model.blocks[15](x)

                print(f"After L15 shape: {x_after_l15.shape}")

                # Check last few positions (where BP/PC markers should be)
                seq_len = x.shape[1]
                print(f"\nLast 5 positions (seq_len={seq_len}):")
                for pos in range(max(0, seq_len-5), seq_len):
                    # Check if this is a marker position
                    op_lev_val = x[0, pos, BD.OP_LEV].item()
                    mark_bp_val = x[0, pos, BD.MARK_BP].item()
                    mark_pc_val = x[0, pos, BD.MARK_PC].item()

                    if op_lev_val > 0.5 or mark_bp_val > 0.5 or mark_pc_val > 0.5:
                        print(f"  Pos {pos}: OP_LEV={op_lev_val:.2f}, MARK_BP={mark_bp_val:.2f}, MARK_PC={mark_pc_val:.2f}")

                        # Check OUTPUT dims (where L15 should write)
                        output_lo = x_after_l15[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16]
                        output_hi = x_after_l15[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16]

                        # Decode nibbles
                        lo_val = torch.argmax(output_lo).item()
                        hi_val = torch.argmax(output_hi).item()
                        byte_val = lo_val | (hi_val << 4)

                        print(f"    OUTPUT after L15: byte={byte_val:#04x} (lo={lo_val}, hi={hi_val})")

                        # Check ADDR dims (from Phase 1)
                        addr_b0_lo = x[0, pos, BD.ADDR_B0_LO:BD.ADDR_B0_LO+16]
                        addr_b0_hi = x[0, pos, BD.ADDR_B0_HI:BD.ADDR_B0_HI+16]

                        addr_lo = torch.argmax(addr_b0_lo).item()
                        addr_hi = torch.argmax(addr_b0_hi).item()
                        addr_byte0 = addr_lo | (addr_hi << 4)

                        print(f"    ADDR_B0 (BP relay): {addr_byte0:#04x} (lo={addr_lo}, hi={addr_hi})")

                logits = runner.model.forward(input_ids)[0, -1, :]
                next_token = torch.argmax(logits).item()

            generation_context.append(next_token)

            if next_token == Token.STEP_END:
                print(f"\n  → STEP_END at iteration {i}")
                break

        break

    if next_token == Token.STEP_END:
        pass  # Continue to next step

print("\n\nDebug complete.")
