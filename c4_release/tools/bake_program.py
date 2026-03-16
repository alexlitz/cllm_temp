#!/usr/bin/env python3
"""
Bytecode patching utilities for C4 VM.

Compiles C subroutines for memset/memcmp/argv and patches bytecode to
replace MSET/MCMP opcodes with JSR calls to those subroutines.

Used by the neural bundler (bundler/neural_bundler.py).

ONNX weight-baking functions have been moved to tools/archive/bake_onnx_weights.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Subroutine compilation from C source
# ============================================================

# C4 opcodes used for patching
MSET_OP = 36
MCMP_OP = 37

# Memory layout for argv
ARGV_BASE = 0x1C000
STRING_BASE = 0x1D000

# C source for subroutines compiled by C4
_MEMSET_SRC = """
int __memset(int ptr, int val, int size) {
    int i;
    i = 0;
    while (i < size) {
        *(char *)(ptr + i) = val;
        i = i + 1;
    }
    return ptr;
}
"""

_MEMCMP_SRC = """
int __memcmp(int p1, int p2, int size) {
    int i;
    int a;
    int b;
    i = 0;
    while (i < size) {
        a = *(char *)(p1 + i);
        b = *(char *)(p2 + i);
        if (a != b) { return a - b; }
        i = i + 1;
    }
    return 0;
}
"""

_ARGV_SETUP_SRC = """
int __argv_setup(int argv_base, int string_base) {
    int argc;
    int i;
    int ch;
    int str_ptr;

    argc = getchar();
    argc = argc + getchar() * 256;
    argc = argc + getchar() * 65536;
    argc = argc + getchar() * 16777216;

    str_ptr = string_base;
    i = 0;
    while (i < argc) {
        *(int *)(argv_base + i * 8) = str_ptr;
        ch = getchar();
        while (ch) {
            *(char *)str_ptr = ch;
            str_ptr = str_ptr + 1;
            ch = getchar();
        }
        *(char *)str_ptr = 0;
        str_ptr = str_ptr + 1;
        i = i + 1;
    }
    return argc;
}
"""

_ARGV_WRAPPER_SRC = """
int __argv_wrapper(int argv_base, int string_base, int main_addr) {
    return 0;
}
"""


def _encode_instr(op, imm=0):
    """Encode (op, imm) -> 64-bit bytecode word."""
    return int(op) + ((imm & 0x00FFFFFFFFFFFFFF) << 8)


def _compile_subroutine(c_source, base_addr):
    """Compile a C function and return its bytecode relocated to base_addr.

    The source must define exactly one function. Returns the relocated
    bytecode (list of 64-bit ints) for just the function body (skipping
    the JSR+EXIT prologue the compiler emits at code[0..1]).
    """
    from src.compiler import compile_c

    bytecode, _ = compile_c(c_source)

    # code[0] = JSR main, code[1] = EXIT — skip these
    func_code = bytecode[2:]

    # Relocate: adjust branch/jump targets from original base (2*8=16) to base_addr
    original_base = 2 * 8  # function body starts at offset 16 in compiled output
    offset = base_addr - original_base

    relocated = []
    for instr in func_code:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)

        # Opcodes with absolute address immediates: JMP, JSR, BZ, BNZ
        if op in (2, 3, 4, 5) and imm != 0:
            imm += offset

        relocated.append(_encode_instr(op, imm))

    return relocated


def patch_bytecode(bytecode, argv=False):
    """Patch bytecode: replace MSET/MCMP with JSR to subroutines, optionally add argv support.

    Compiles C subroutines for memset/memcmp (and argv if requested),
    appends them to the bytecode, and replaces MSET/MCMP opcodes with JSR.
    """
    base = len(bytecode) * 8

    # Compile memset subroutine
    memset_addr = base
    memset_code = _compile_subroutine(_MEMSET_SRC, memset_addr)
    next_base = memset_addr + len(memset_code) * 8

    # Compile memcmp subroutine
    memcmp_addr = next_base
    memcmp_code = _compile_subroutine(_MEMCMP_SRC, memcmp_addr)
    next_base = memcmp_addr + len(memcmp_code) * 8

    extra_code = memset_code + memcmp_code

    if argv:
        # Compile argv_setup
        argv_setup_addr = next_base
        argv_setup_code = _compile_subroutine(_ARGV_SETUP_SRC, argv_setup_addr)
        next_base = argv_setup_addr + len(argv_setup_code) * 8

        # Extract original main address from code[0] (JSR main)
        original_main_addr = bytecode[0] >> 8
        if original_main_addr >= (1 << 55):
            original_main_addr -= (1 << 56)

        # Build wrapper: calls argv_setup then main
        # ENT 0; IMM argv_base; PSH; IMM string_base; PSH;
        # JSR argv_setup; ADJ 16; PSH; IMM argv_base; PSH;
        # JSR main; ADJ 16; LEV
        wrapper_addr = next_base
        wrapper_code = [
            _encode_instr(6, 0),                     # ENT 0
            _encode_instr(1, ARGV_BASE),             # IMM ARGV_BASE
            _encode_instr(13),                       # PSH
            _encode_instr(1, STRING_BASE),           # IMM STRING_BASE
            _encode_instr(13),                       # PSH
            _encode_instr(3, argv_setup_addr),       # JSR argv_setup
            _encode_instr(7, 16),                    # ADJ 16
            _encode_instr(13),                       # PSH (argc)
            _encode_instr(1, ARGV_BASE),             # IMM ARGV_BASE
            _encode_instr(13),                       # PSH
            _encode_instr(3, original_main_addr),    # JSR main
            _encode_instr(7, 16),                    # ADJ 16
            _encode_instr(8),                        # LEV
        ]

        extra_code += argv_setup_code + wrapper_code

        # Patch code[0] to JSR wrapper instead of main
        bytecode = list(bytecode)
        bytecode[0] = _encode_instr(3, wrapper_addr)
    else:
        bytecode = list(bytecode)

    # Replace MSET (36) -> JSR memset_addr, MCMP (37) -> JSR memcmp_addr
    for i in range(len(bytecode)):
        op = bytecode[i] & 0xFF
        if op == MSET_OP:
            bytecode[i] = _encode_instr(3, memset_addr)
        elif op == MCMP_OP:
            bytecode[i] = _encode_instr(3, memcmp_addr)

    return bytecode + extra_code
