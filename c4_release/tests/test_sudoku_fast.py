#!/usr/bin/env python3
"""
Test Sudoku Solver with C-accelerated VM

Uses ctypes to call the C VM core loop for maximum speed.
"""

import sys
import os
import time
import ctypes
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c


def run_with_c_vm(bytecode, data, max_steps=100000000):
    """Run bytecode using the C VM implementation."""
    # Load shared library
    lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'src', 'fast_vm.so'
    )
    lib = ctypes.CDLL(lib_path)

    # Set up function signature
    lib.vm_run.restype = ctypes.c_longlong
    lib.vm_run.argtypes = [
        ctypes.POINTER(ctypes.c_int),          # ops
        ctypes.POINTER(ctypes.c_longlong),      # imms
        ctypes.c_int,                            # code_len
        ctypes.c_longlong,                       # sp
        ctypes.c_longlong,                       # bp
        ctypes.c_longlong,                       # ax
        ctypes.c_longlong,                       # pc
        ctypes.c_longlong,                       # heap_ptr
        ctypes.c_char_p,                         # stdout_buf
        ctypes.c_int,                            # stdout_cap
        ctypes.c_longlong,                       # max_steps
        ctypes.POINTER(ctypes.c_longlong),      # mem_init_keys
        ctypes.POINTER(ctypes.c_longlong),      # mem_init_vals
        ctypes.c_int,                            # mem_init_count
        ctypes.POINTER(ctypes.c_longlong),      # out_sp
        ctypes.POINTER(ctypes.c_longlong),      # out_bp
        ctypes.POINTER(ctypes.c_longlong),      # out_ax
        ctypes.POINTER(ctypes.c_longlong),      # out_pc
        ctypes.POINTER(ctypes.c_longlong),      # out_heap_ptr
        ctypes.POINTER(ctypes.c_longlong),      # out_steps
        ctypes.POINTER(ctypes.c_int),           # out_stdout_pos
    ]

    # Parse bytecode
    code = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append((op, imm))

    code_len = len(code)
    ops_arr = (ctypes.c_int * code_len)(*[c[0] for c in code])
    imms_arr = (ctypes.c_longlong * code_len)(*[c[1] for c in code])

    # Initialize memory with data segment
    mem_keys = []
    mem_vals = []
    if data:
        for i, b in enumerate(data):
            mem_keys.append(0x10000 + i)
            mem_vals.append(b)

    mem_count = len(mem_keys)
    if mem_count > 0:
        keys_arr = (ctypes.c_longlong * mem_count)(*mem_keys)
        vals_arr = (ctypes.c_longlong * mem_count)(*mem_vals)
    else:
        keys_arr = (ctypes.c_longlong * 1)(0)
        vals_arr = (ctypes.c_longlong * 1)(0)

    # Output buffer
    stdout_cap = 4096
    stdout_buf = ctypes.create_string_buffer(stdout_cap)

    # Output state
    out_sp = ctypes.c_longlong(0)
    out_bp = ctypes.c_longlong(0)
    out_ax = ctypes.c_longlong(0)
    out_pc = ctypes.c_longlong(0)
    out_heap = ctypes.c_longlong(0)
    out_steps = ctypes.c_longlong(0)
    out_stdout_pos = ctypes.c_int(0)

    # Run
    result = lib.vm_run(
        ops_arr, imms_arr, code_len,
        0x10000, 0x10000, 0, 0,  # sp, bp, ax, pc
        0x200000,                 # heap_ptr
        stdout_buf, stdout_cap,
        max_steps,
        keys_arr, vals_arr, mem_count,
        ctypes.byref(out_sp), ctypes.byref(out_bp),
        ctypes.byref(out_ax), ctypes.byref(out_pc),
        ctypes.byref(out_heap), ctypes.byref(out_steps),
        ctypes.byref(out_stdout_pos),
    )

    output = stdout_buf.value.decode('ascii', errors='replace')
    steps = out_steps.value

    return result, output, steps


def main():
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'demos', 'sudoku.c'
    )
    with open(src_path) as f:
        source = f.read()

    print("=" * 60)
    print("  Sudoku Solver - C-Accelerated VM")
    print("=" * 60)
    print()
    print("Puzzle: Arto Inkala's 'World's Hardest Sudoku' (2012)")
    print()

    # Compile
    print("Compiling...", end=" ", flush=True)
    start = time.time()
    bytecode, data = compile_c(source)
    compile_time = time.time() - start
    print(f"done ({compile_time*1000:.1f}ms, {len(bytecode)} instructions)")

    # Run C VM
    print("Solving (C VM)...", end=" ", flush=True)
    start = time.time()
    result, output, steps = run_with_c_vm(bytecode, data)
    solve_time = time.time() - start
    steps_per_sec = steps / solve_time if solve_time > 0 else 0
    print(f"done ({solve_time:.3f}s, {steps:,} VM steps, {steps_per_sec/1e6:.1f}M steps/sec)")
    print()

    if result == 1:
        print("SOLVED! Output:")
        print()
        for line in output.strip().split('\n'):
            print(f"  {line}")
        print()

        expected = [
            "8 1 2 7 5 3 6 4 9",
            "9 4 3 6 8 2 1 7 5",
            "6 7 5 4 9 1 2 8 3",
            "1 5 4 2 3 7 8 9 6",
            "3 6 9 8 4 5 7 2 1",
            "2 8 7 1 6 9 5 3 4",
            "5 2 1 9 7 4 3 6 8",
            "4 3 8 5 2 6 9 1 7",
            "7 9 6 3 1 8 4 5 2",
        ]
        actual_lines = output.strip().split('\n')
        all_match = all(e == a for e, a in zip(expected, actual_lines)) and len(actual_lines) == 9
        if all_match:
            print("VERIFIED: Solution matches known answer!")
        else:
            print("VERIFICATION FAILED")
            for i, (exp, act) in enumerate(zip(expected, actual_lines)):
                if exp != act:
                    print(f"  Row {i}: expected '{exp}', got '{act}'")
    else:
        print(f"FAILED: No solution found (result={result})")
        print(f"  Output: {repr(output)}")

    # Also run Python VM for comparison
    print()
    print("--- Python VM comparison ---")
    from test_sudoku import SudokuVM
    from src.io_support import IOExtendedVM

    vm = SudokuVM()
    vm.load(bytecode, data)
    start = time.time()
    py_result = vm.run(max_steps=100000000)
    py_time = time.time() - start
    py_steps = vm.step_count
    py_sps = py_steps / py_time if py_time > 0 else 0
    print(f"Python VM: {py_time:.2f}s, {py_steps:,} steps, {py_sps/1e6:.1f}M steps/sec")
    print(f"C VM speedup: {py_time/solve_time:.1f}x")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
