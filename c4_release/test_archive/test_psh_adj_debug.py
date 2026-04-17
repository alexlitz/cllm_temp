#!/usr/bin/env python3
"""Debug PSH+ADD test with step tracing."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, set_vm_weights
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner


runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)

program = [
    Opcode.IMM | (5 << 8),
    Opcode.PSH,
    Opcode.IMM | (3 << 8),
    Opcode.ADD,
    Opcode.EXIT,
]

# Manually run with step tracing
bytecode = program
context = runner._build_context(bytecode, b"", [], "")
runner._bytecode = bytecode
runner._stdin_buffer = []
runner._stdin_pos = 0
runner._tool_handler = None
runner._tool_call_id = 0
runner._last_sp = 0

# Initialize data section shadow memory
runner._memory = {}
runner._heap_base = 0x10000
runner._heap_ptr = 0x10000
runner._alloc_sizes = {}

output = []
step_count = 0

print(f"Initial context length: {len(context)}")
print(f"Context: {context}")

for i in range(20 * Token.STEP_TOKENS):
    next_token = runner.model.generate_next(context)
    context.append(next_token)

    if next_token in (Token.STEP_END, Token.HALT):
        # Extract registers
        pc = runner._extract_register(context, Token.REG_PC)
        ax = runner._extract_register(context, Token.REG_AX)
        sp = runner._extract_register(context, Token.REG_SP)
        bp = runner._extract_register(context, Token.REG_BP)
        s0 = runner._extract_register(context, Token.STACK0)

        if pc is not None:
            instr_idx = (pc - 2) // 5
            if 0 <= instr_idx < len(bytecode):
                op = bytecode[instr_idx] & 0xFF
                opname = {v: k for k, v in vars(Opcode).items()
                          if isinstance(v, int)}.get(op, f"?{op}")
            else:
                opname = f"?idx={instr_idx}"
        else:
            opname = "?nopc"

        end_type = "HALT" if next_token == Token.HALT else "STEP_END"
        print(f"Step {step_count}: [{end_type}] PC={pc} ({opname}) "
              f"AX={ax} SP={sp} BP={bp} STACK0={s0}")

        # Show the raw bytes for this step
        step_start = len(context) - Token.STEP_TOKENS
        if step_start >= 0:
            step_tokens = context[step_start:]
            print(f"  Raw: {step_tokens}")

        # Run handler
        if pc is not None and next_token == Token.STEP_END:
            instr_idx = (pc - 2) // 5
            if 0 <= instr_idx < len(bytecode):
                op = bytecode[instr_idx] & 0xFF
                from neural_vm.run_vm import _BINARY_POP_OPS
                handler = runner._syscall_handlers.get(op)
                if handler:
                    handler(context, output)
                    # Re-extract after handler
                    ax2 = runner._extract_register(context, Token.REG_AX)
                    sp2 = runner._extract_register(context, Token.REG_SP)
                    s02 = runner._extract_register(context, Token.STACK0)
                    print(f"  After handler: AX={ax2} SP={sp2} STACK0={s02}")
                if op in _BINARY_POP_OPS:
                    new_sp = (runner._last_sp + 8) & 0xFFFFFFFF
                    runner._override_register_in_last_step(
                        context, Token.REG_SP, new_sp)
                    print(f"  Binary pop SP correction: {runner._last_sp} + 8 = {new_sp}")
                runner._track_memory_write(context, op)

        # Update SP tracking
        sp_after = runner._extract_register(context, Token.REG_SP)
        if sp_after is not None:
            runner._last_sp = sp_after
            print(f"  SP tracking updated: {sp_after} (0x{sp_after:08X})")

        step_count += 1

    if next_token == Token.HALT:
        break

print(f"\nFinal exit code: {runner._decode_exit_code(context)}")
