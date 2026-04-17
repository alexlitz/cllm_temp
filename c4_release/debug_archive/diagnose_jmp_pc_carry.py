#!/usr/bin/env python3
"""Diagnose JMP PC carry-forward issue."""

import sys
import torch
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

def trace_jmp_pc_carry():
    """Trace JMP PC carry-forward step by step."""
    runner = AutoregressiveVMRunner()
    model = runner.model

    # Remove JMP handler to test neural path
    if Opcode.JMP in runner._func_call_handlers:
        del runner._func_call_handlers[Opcode.JMP]
        print("Removed JMP handler - testing pure neural path")

    # JMP 16 (jump to PC=16 = instruction index 2)
    bytecode = [
        Opcode.JMP | (16 << 8),   # index 0 (PC=0): JMP to 16
        Opcode.IMM | (99 << 8),   # index 1 (PC=8): should be skipped
        Opcode.IMM | (42 << 8),   # index 2 (PC=16): target - load 42
        Opcode.EXIT               # index 3 (PC=24): exit with 42
    ]

    print("=== JMP 16 Neural Path Debug ===")
    print()

    # Run with token-by-token tracing
    gen_count = [0]
    original_forward = model.forward

    def tracing_forward(token_ids, **kwargs):
        gen = gen_count[0]
        gen_count[0] += 1
        ctx_len = token_ids.size(1)

        if gen > 100:
            return original_forward(token_ids, **kwargs)

        # Get embedding and run through layers
        x = model.embed(token_ids)

        for layer_idx, block in enumerate(model.blocks):
            x = block(x)

        # Print what token we're generating
        logits = model.head(x)[0, -1, :]
        next_tok = logits.argmax().item()

        # Get last token info
        last_tok = token_ids[0, -1].item()

        # Find all PC markers and their positions
        pc_positions = []
        for pos in range(ctx_len):
            if x[0, pos, BD.MARK_PC].item() > 0.5:
                output_lo = x[0, pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
                output_hi = x[0, pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
                output_byte = output_hi * 16 + output_lo
                pc_positions.append((pos, output_byte))

        if gen < 80:
            tok_name = f"0x{next_tok:02x}" if next_tok < 256 else {
                Token.REG_PC: "REG_PC",
                Token.REG_AX: "REG_AX",
                Token.REG_SP: "REG_SP",
                Token.REG_BP: "REG_BP",
                Token.STEP_END: "STEP_END",
                Token.HALT: "HALT",
                Token.STACK0: "STACK0",
                Token.MEM: "MEM",
            }.get(next_tok, str(next_tok))

            last_name = f"0x{last_tok:02x}" if last_tok < 256 else {
                Token.REG_PC: "REG_PC",
                Token.REG_AX: "REG_AX",
                Token.REG_SP: "REG_SP",
                Token.REG_BP: "REG_BP",
                Token.STEP_END: "STEP_END",
                Token.HALT: "HALT",
                Token.STACK0: "STACK0",
                Token.MEM: "MEM",
            }.get(last_tok, str(last_tok))

            print(f"gen {gen:2d}: ctx={ctx_len:3d}, last={last_name:8s} -> next={tok_name:8s}  PC_markers: {pc_positions[-3:] if pc_positions else []}")

        return model.head(x)

    model.forward = tracing_forward

    try:
        tokens, result = runner.run(bytecode, b'', max_steps=6)

        print(f"\n=== Result: {result} (expected 42) ===")

        # Print token sequence with interpretation
        print("\n=== Full Token Sequence ===")
        tok_map = {
            Token.REG_PC: "PC", Token.REG_AX: "AX", Token.REG_SP: "SP",
            Token.REG_BP: "BP", Token.STEP_END: "SE", Token.HALT: "HALT",
            Token.STACK0: "STK0", Token.MEM: "MEM"
        }
        seq = []
        for tok in tokens:
            if tok < 256:
                seq.append(f"0x{tok:02x}")
            else:
                seq.append(tok_map.get(tok, str(tok)))
        print(" ".join(seq))

    finally:
        model.forward = original_forward

if __name__ == "__main__":
    trace_jmp_pc_carry()
