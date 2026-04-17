#!/usr/bin/env python3
"""Diagnose JMP L5 FETCH mechanism.

Traces what happens at each layer for JMP instruction:
1. L3: PC carry-forward
2. L4: PC relay to AX marker
3. L5: Opcode/immediate fetch
4. L6: JMP PC override
"""

import sys
import torch
sys.path.insert(0, '.')

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD

def trace_jmp():
    """Run JMP and trace internal state."""
    runner = AutoregressiveVMRunner()
    model = runner.model

    # JMP 16 (jump to PC=16, instruction index 2)
    bytecode = [
        Opcode.JMP | (16 << 8),   # index 0: JMP to 16
        Opcode.IMM | (99 << 8),   # index 1: should be skipped
        Opcode.IMM | (42 << 8),   # index 2: target
        Opcode.EXIT               # index 3: exit
    ]

    print("=== JMP 16 Test ===")
    print(f"Bytecode: JMP 16, IMM 99, IMM 42, EXIT")
    print()

    # Run with hooks to trace internal state
    original_forward = model.forward
    step_count = [0]

    def tracing_forward(token_ids, **kwargs):
        step = step_count[0]
        step_count[0] += 1

        # Get input embeddings
        x = model.embed_tokens(token_ids)
        if hasattr(model, 'position_encoder'):
            x = model.position_encoder(x)

        # Run through layers and capture intermediate states
        for layer_idx, block in enumerate(model.blocks):
            x_before = x.clone()
            x = block(x)

            if layer_idx in [3, 4, 5, 6] and step < 3:
                # Trace PC marker state
                pc_marker_pos = None
                ax_marker_pos = None
                for pos in range(x.size(1)):
                    if x[0, pos, BD.MARK_PC].item() > 0.5:
                        pc_marker_pos = pos
                    if x[0, pos, BD.MARK_AX].item() > 0.5:
                        ax_marker_pos = pos

                if pc_marker_pos is not None or ax_marker_pos is not None:
                    print(f"Step {step}, Layer {layer_idx}:")

                    if pc_marker_pos is not None:
                        embed_lo = x[0, pc_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
                        embed_hi = x[0, pc_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
                        output_lo = x[0, pc_marker_pos, BD.OUTPUT_LO:BD.OUTPUT_LO+16].argmax().item()
                        output_hi = x[0, pc_marker_pos, BD.OUTPUT_HI:BD.OUTPUT_HI+16].argmax().item()
                        fetch_lo = x[0, pc_marker_pos, BD.FETCH_LO:BD.FETCH_LO+16].max().item()
                        fetch_hi = x[0, pc_marker_pos, BD.FETCH_HI:BD.FETCH_HI+16].max().item()
                        op_jmp = x[0, pc_marker_pos, BD.OP_JMP].item()

                        print(f"  PC marker (pos {pc_marker_pos}):")
                        print(f"    EMBED: lo={embed_lo}, hi={embed_hi} -> byte=0x{embed_hi*16+embed_lo:02x}")
                        print(f"    OUTPUT: lo={output_lo}, hi={output_hi} -> byte=0x{output_hi*16+output_lo:02x}")
                        print(f"    FETCH max: lo={fetch_lo:.2f}, hi={fetch_hi:.2f}")
                        print(f"    OP_JMP: {op_jmp:.2f}")

                    if ax_marker_pos is not None and layer_idx >= 4:
                        embed_lo = x[0, ax_marker_pos, BD.EMBED_LO:BD.EMBED_LO+16].argmax().item()
                        embed_hi = x[0, ax_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16].argmax().item()
                        temp_lo = x[0, ax_marker_pos, BD.TEMP:BD.TEMP+16].max().item()
                        opcode_lo = x[0, ax_marker_pos, BD.OPCODE_BYTE_LO:BD.OPCODE_BYTE_LO+16].argmax().item()
                        opcode_hi = x[0, ax_marker_pos, BD.OPCODE_BYTE_HI:BD.OPCODE_BYTE_HI+16].argmax().item()

                        print(f"  AX marker (pos {ax_marker_pos}):")
                        print(f"    EMBED: lo={embed_lo}, hi={embed_hi} -> PC=0x{embed_hi*16+embed_lo:02x}")
                        print(f"    TEMP max: {temp_lo:.2f}")
                        print(f"    OPCODE_BYTE: lo={opcode_lo}, hi={opcode_hi} -> opcode={opcode_hi*16+opcode_lo}")

        # Get final output
        return model._final_forward(x)

    # Patch forward
    model.forward = tracing_forward

    try:
        _, result = runner.run(bytecode, b'', max_steps=10)
        print()
        print(f"=== RESULT: {result} (expected 42) ===")
        if result == 42:
            print("SUCCESS!")
        else:
            print("FAILED - JMP not working correctly")
    finally:
        model.forward = original_forward

if __name__ == "__main__":
    trace_jmp()
