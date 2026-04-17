#!/usr/bin/env python3
"""Diagnose L3-L6 for step 2 PC."""

import sys
import torch
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

def trace():
    runner = AutoregressiveVMRunner()
    model = runner.model

    # Remove JMP handler
    if Opcode.JMP in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.JMP]

    bytecode = [
        Opcode.JMP | (16 << 8),
        Opcode.IMM | (99 << 8),
        Opcode.IMM | (42 << 8),
        Opcode.EXIT
    ]

    print("=== L3-L6 Trace for Step 2 PC ===\n")

    gen_count = [0]
    original_forward = model.forward

    def tracing_forward(token_ids, **kwargs):
        gen = gen_count[0]
        gen_count[0] += 1
        ctx_len = token_ids.size(1)

        if gen > 80:
            return original_forward(token_ids, **kwargs)

        x = model.embed(token_ids)

        # Only trace gen 36 (step 2's PC byte 0 generation)
        trace_this = (gen == 36)

        for layer_idx, block in enumerate(model.blocks):
            if trace_this and layer_idx in [3, 4, 5, 6]:
                # Find step 2's PC marker (last one)
                pc_pos = None
                for pos in range(ctx_len - 1, -1, -1):
                    if x[0, pos, BD.MARK_PC].item() > 0.5:
                        pc_pos = pos
                        break

                if pc_pos:
                    embed_lo = x[0, pc_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
                    embed_hi = x[0, pc_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
                    output_lo = x[0, pc_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
                    output_hi = x[0, pc_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
                    fetch_lo = x[0, pc_pos, BD.FETCH_LO:BD.FETCH_LO+16].argmax().item()
                    fetch_hi = x[0, pc_pos, BD.FETCH_HI:BD.FETCH_HI+16].argmax().item()
                    op_jmp = x[0, pc_pos, BD.OP_JMP].item()

                    print(f"Before L{layer_idx}: EMBED=0x{embed_hi*16+embed_lo:02x} OUTPUT=0x{output_hi*16+output_lo:02x} FETCH=0x{fetch_hi*16+fetch_lo:02x} OP_JMP={op_jmp:.1f}")

            x = block(x)

            if trace_this and layer_idx in [3, 4, 5, 6]:
                if pc_pos:
                    embed_lo = x[0, pc_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
                    embed_hi = x[0, pc_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
                    output_lo = x[0, pc_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
                    output_hi = x[0, pc_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
                    fetch_lo = x[0, pc_pos, BD.FETCH_LO:BD.FETCH_LO+16].argmax().item()
                    fetch_hi = x[0, pc_pos, BD.FETCH_HI:BD.FETCH_HI+16].argmax().item()
                    op_jmp = x[0, pc_pos, BD.OP_JMP].item()

                    print(f"After  L{layer_idx}: EMBED=0x{embed_hi*16+embed_lo:02x} OUTPUT=0x{output_hi*16+output_lo:02x} FETCH=0x{fetch_hi*16+fetch_lo:02x} OP_JMP={op_jmp:.1f}")

        # Check final output at last position
        if trace_this:
            logits = model.head(x)[0, -1, :]
            top_tokens = logits.topk(5)
            print(f"\nFinal logits top 5:")
            for val, idx in zip(top_tokens.values.tolist(), top_tokens.indices.tolist()):
                name = f"0x{idx:02x}" if idx < 256 else str(idx)
                print(f"  {name}: {val:.2f}")

        return model.head(x)

    model.forward = tracing_forward

    try:
        tokens, result = runner.run(bytecode, b'', max_steps=4)
        print(f"\nResult: {result} (expected 42)")
    finally:
        model.forward = original_forward

if __name__ == "__main__":
    trace()
