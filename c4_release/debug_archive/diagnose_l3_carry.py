#!/usr/bin/env python3
"""Diagnose L3 PC carry-forward attention."""

import sys
import torch
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

def trace_l3_carry():
    """Trace L3 carry-forward attention mechanism."""
    runner = AutoregressiveVMRunner()
    model = runner.model

    # Remove JMP handler to test neural path
    if Opcode.JMP in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.JMP]

    bytecode = [
        Opcode.JMP | (16 << 8),
        Opcode.IMM | (99 << 8),
        Opcode.IMM | (42 << 8),
        Opcode.EXIT
    ]

    print("=== L3 Carry-Forward Debug ===")
    print()

    gen_count = [0]
    original_forward = model.forward

    def tracing_forward(token_ids, **kwargs):
        gen = gen_count[0]
        gen_count[0] += 1
        ctx_len = token_ids.size(1)

        if gen > 80:
            return original_forward(token_ids, **kwargs)

        x = model.embed(token_ids)

        # Trace L3 attention specifically
        for layer_idx, block in enumerate(model.blocks):
            if layer_idx == 3 and gen >= 35 and gen <= 40:
                # This is around step 2 PC generation
                # Trace the attention mechanism

                # Find step 2's PC marker position (should be newly added)
                pc_pos = None
                for pos in range(ctx_len - 1, -1, -1):
                    if x[0, pos, BD.MARK_PC].item() > 0.5:
                        pc_pos = pos
                        break

                # Find step 1's PC byte 0 position (should be at pos ~37)
                # PC byte 0 is the token right after REG_PC marker
                step1_pc_byte0_pos = None
                for pos in range(ctx_len):
                    # Check for L1H1[0]=1 pattern (PC byte 0 indicator)
                    l1h1_0 = x[0, pos, BD.L1H1].item()
                    l1h0_0 = x[0, pos, BD.L1H0].item()
                    if l1h1_0 > 0.5 and l1h0_0 < 0.5:
                        step1_pc_byte0_pos = pos

                if pc_pos is not None:
                    print(f"gen {gen}: PC marker at pos {pc_pos}")
                    print(f"  Before L3: EMBED_LO={x[0, pc_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}, EMBED_HI={x[0, pc_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}")
                    print(f"  Step 1 PC byte 0 likely at pos {step1_pc_byte0_pos}")
                    if step1_pc_byte0_pos is not None:
                        print(f"  Step 1 PC byte 0 EMBED: lo={x[0, step1_pc_byte0_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}, hi={x[0, step1_pc_byte0_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}")

            x = block(x)

            if layer_idx == 3 and gen >= 35 and gen <= 40:
                if pc_pos is not None:
                    print(f"  After L3: EMBED_LO={x[0, pc_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()}, EMBED_HI={x[0, pc_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()}")
                    # Also check OUTPUT
                    print(f"  After L3: OUTPUT_LO={x[0, pc_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()}, OUTPUT_HI={x[0, pc_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()}")
                    print()

        return model.head(x)

    model.forward = tracing_forward

    try:
        tokens, result = runner.run(bytecode, b'', max_steps=4)
        print(f"\nResult: {result} (expected 42)")
    finally:
        model.forward = original_forward

if __name__ == "__main__":
    trace_l3_carry()
