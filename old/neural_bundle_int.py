#!/usr/bin/env python3
"""
Neural VM Bundler - Embeds ONNX runtime + bytecode

Bundles:
1. The neural arithmetic from onnx_runner_c4.c (SwiGLU identity, nibble add)
2. Hardcoded bytecode
3. A VM that uses neural ops for arithmetic

C4 compatible (no floats, no for loops).
"""

import argparse
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c


def generate_c_array(name: str, data: bytes, items_per_line: int = 16) -> str:
    """Generate C array from binary data."""
    lines = [f"static const unsigned char {name}[] = {{"]
    for i in range(0, len(data), items_per_line):
        chunk = data[i:i+items_per_line]
        hex_vals = ', '.join(f'0x{b:02x}' for b in chunk)
        lines.append(f"    {hex_vals},")
    lines.append("};")
    lines.append(f"static const int {name}_len = {len(data)};")
    return '\n'.join(lines)


ONNX_NEURAL_RUNTIME = '''
/* ============ NEURAL ARITHMETIC (from onnx_runner_c4.c) ============ */
/*
 * SwiGLU identity for exact multiplication: silu(a)*b + silu(-a)*(-b) = a*b
 * Nibble-based addition with carry propagation
 * Fixed-point: 12.12 format (SCALE=4096)
 */

int SCALE;
int HALF_SCALE;
int *exp_tbl;

int init_neural() {
    SCALE = 4096;
    HALF_SCALE = 2048;
    exp_tbl = malloc(64);

    /* exp(-n) * 4096 for n = 0..15 */
    exp_tbl[0] = 4096;
    exp_tbl[1] = 1507;
    exp_tbl[2] = 554;
    exp_tbl[3] = 204;
    exp_tbl[4] = 75;
    exp_tbl[5] = 28;
    exp_tbl[6] = 10;
    exp_tbl[7] = 4;
    exp_tbl[8] = 1;
    exp_tbl[9] = 0;
    exp_tbl[10] = 0;
    exp_tbl[11] = 0;
    exp_tbl[12] = 0;
    exp_tbl[13] = 0;
    exp_tbl[14] = 0;
    exp_tbl[15] = 0;

    return 0;
}

int fp_exp_neg(int x) {
    int idx;
    int frac;
    int e1;
    int e2;
    int diff;

    if (x <= 0) return SCALE;
    if (x >= SCALE * 9) return 0;

    idx = x / SCALE;
    if (idx >= 8) return 0;

    frac = x - idx * SCALE;
    e1 = exp_tbl[idx];
    e2 = exp_tbl[idx + 1];
    diff = e2 - e1;
    return e1 + (diff * frac) / SCALE;
}

int fp_sigmoid(int x) {
    int exp_neg_x;
    int denom;

    if (x >= SCALE * 9) return SCALE;
    if (x <= 0 - SCALE * 9) return 0;

    if (x >= 0) {
        exp_neg_x = fp_exp_neg(x);
        denom = SCALE + exp_neg_x;
        return (SCALE * SCALE + denom / 2) / denom;
    } else {
        exp_neg_x = fp_exp_neg(0 - x);
        if (exp_neg_x == 0) return 0;
        denom = exp_neg_x + SCALE;
        return (exp_neg_x * SCALE + denom / 2) / denom;
    }
}

int fp_silu(int x) {
    int sig;
    sig = fp_sigmoid(x);
    return (x / SCALE) * sig + ((x % SCALE) * sig + SCALE / 2) / SCALE;
}

/* SwiGLU identity: silu(a)*b + silu(-a)*(-b) = a*b */
long long swiglu_multiply(long long a, long long b) {
    int a_fp;
    int silu_a;
    int silu_neg_a;
    int int_part;
    int frac_part;
    int neg_b;

    a_fp = (int)(a * SCALE);
    silu_a = fp_silu(a_fp);
    silu_neg_a = fp_silu(0 - a_fp);
    neg_b = (int)(0 - b);

    int_part = (silu_a / SCALE) * (int)b + (silu_neg_a / SCALE) * neg_b;
    frac_part = ((silu_a % SCALE) * (int)b + (silu_neg_a % SCALE) * neg_b + HALF_SCALE) / SCALE;

    return (long long)(int_part + frac_part);
}

int nibble_add(int a, int b, int carry_in, int *sum, int *carry_out) {
    int total;
    total = a + b + carry_in;
    *sum = total & 15;
    *carry_out = (total / 16) & 1;
    return 0;
}

long long neural_add(long long a, long long b) {
    long long result;
    int carry;
    int i;
    int shift;
    int nibble_a;
    int nibble_b;
    int sum;
    int new_carry;

    result = 0;
    carry = 0;
    i = 0;

    while (i < 16) {
        shift = i * 4;
        nibble_a = (int)((a >> shift) & 15);
        nibble_b = (int)((b >> shift) & 15);

        nibble_add(nibble_a, nibble_b, carry, &sum, &new_carry);
        result = result | ((long long)sum << shift);
        carry = new_carry;

        i = i + 1;
    }

    return result;
}

long long neural_sub(long long a, long long b) {
    return neural_add(a, neural_add(~b, 1));
}

/* Division via repeated subtraction */
long long neural_divide(long long a, long long b) {
    long long result;
    int neg;

    if (b == 0) return 0;

    neg = 0;
    if (a < 0) { a = 0 - a; neg = neg + 1; }
    if (b < 0) { b = 0 - b; neg = neg + 1; }

    result = 0;
    while (a >= b) {
        a = a - b;
        result = result + 1;
    }

    if (neg == 1) return 0 - result;
    return result;
}

/* Modulo via repeated subtraction */
long long neural_mod(long long a, long long b) {
    int neg;

    if (b == 0) return 0;

    neg = 0;
    if (a < 0) { a = 0 - a; neg = 1; }
    if (b < 0) { b = 0 - b; }

    while (a >= b) {
        a = a - b;
    }

    if (neg) return 0 - a;
    return a;
}

'''

VM_CODE = '''
/* ============ NEURAL VM ============ */

unsigned char *memory;
int mem_size;
long long *code;
int code_len;
long long pc;
long long sp;
long long bp;
long long ax;
int halted;

long long mem_ri(long long a) { return *(long long *)(memory + a); }
int mem_wi(long long a, long long v) { *(long long *)(memory + a) = v; return 0; }
int mem_rb(long long a) { return memory[a]; }
int mem_wb(long long a, int v) { memory[a] = (unsigned char)v; return 0; }

/* Opcodes */
int LEA_OP, IMM_OP, JMP_OP, JSR_OP, BZ_OP, BNZ_OP;
int ENT_OP, ADJ_OP, LEV_OP, LI_OP, LC_OP, SI_OP;
int SC_OP, PSH_OP, OR_OP, XOR_OP, AND_OP, EQ_OP;
int NE_OP, LT_OP, GT_OP, LE_OP, GE_OP, SHL_OP;
int SHR_OP, ADD_OP, SUB_OP, MUL_OP, DIV_OP, MOD_OP;
int EXIT_OP, GETC_OP, PUTC_OP;

int init_opcodes() {
    LEA_OP = 0; IMM_OP = 1; JMP_OP = 2; JSR_OP = 3; BZ_OP = 4; BNZ_OP = 5;
    ENT_OP = 6; ADJ_OP = 7; LEV_OP = 8; LI_OP = 9; LC_OP = 10; SI_OP = 11;
    SC_OP = 12; PSH_OP = 13; OR_OP = 14; XOR_OP = 15; AND_OP = 16; EQ_OP = 17;
    NE_OP = 18; LT_OP = 19; GT_OP = 20; LE_OP = 21; GE_OP = 22; SHL_OP = 23;
    SHR_OP = 24; ADD_OP = 25; SUB_OP = 26; MUL_OP = 27; DIV_OP = 28; MOD_OP = 29;
    EXIT_OP = 38; GETC_OP = 64; PUTC_OP = 65;
    return 0;
}

int step() {
    long long op;
    long long imm;
    long long a;

    if (halted) return 1;
    if (pc < 0) { halted = 1; return 1; }
    if (pc >= code_len * 8) { halted = 1; return 1; }

    op = code[pc / 8] & 255;
    imm = code[pc / 8] >> 8;
    if (imm >= (1LL << 55)) imm = imm - (1LL << 56);
    pc = pc + 8;

    if (op == LEA_OP) { ax = bp + imm; }
    else if (op == IMM_OP) { ax = imm; }
    else if (op == JMP_OP) { pc = imm; }
    else if (op == JSR_OP) { sp = sp - 8; mem_wi(sp, pc); pc = imm; }
    else if (op == BZ_OP) { if (ax == 0) pc = imm; }
    else if (op == BNZ_OP) { if (ax != 0) pc = imm; }
    else if (op == ENT_OP) { sp = sp - 8; mem_wi(sp, bp); bp = sp; sp = sp - imm; }
    else if (op == ADJ_OP) { sp = sp + imm; }
    else if (op == LEV_OP) { sp = bp; bp = mem_ri(sp); sp = sp + 8; pc = mem_ri(sp); sp = sp + 8; }
    else if (op == LI_OP) { ax = mem_ri(ax); }
    else if (op == LC_OP) { ax = mem_rb(ax); }
    else if (op == SI_OP) { a = mem_ri(sp); sp = sp + 8; mem_wi(a, ax); }
    else if (op == SC_OP) { a = mem_ri(sp); sp = sp + 8; mem_wb(a, (int)ax); }
    else if (op == PSH_OP) { sp = sp - 8; mem_wi(sp, ax); }

    /* NEURAL ARITHMETIC */
    else if (op == ADD_OP) { a = mem_ri(sp); sp = sp + 8; ax = neural_add(a, ax); }
    else if (op == SUB_OP) { a = mem_ri(sp); sp = sp + 8; ax = neural_sub(a, ax); }
    else if (op == MUL_OP) { a = mem_ri(sp); sp = sp + 8; ax = swiglu_multiply(a, ax); }
    else if (op == DIV_OP) { a = mem_ri(sp); sp = sp + 8; ax = neural_divide(a, ax); }
    else if (op == MOD_OP) { a = mem_ri(sp); sp = sp + 8; ax = neural_mod(a, ax); }

    /* Bitwise - native */
    else if (op == AND_OP) { a = mem_ri(sp); sp = sp + 8; ax = a & ax; }
    else if (op == OR_OP) { a = mem_ri(sp); sp = sp + 8; ax = a | ax; }
    else if (op == XOR_OP) { a = mem_ri(sp); sp = sp + 8; ax = a ^ ax; }
    else if (op == SHL_OP) { a = mem_ri(sp); sp = sp + 8; ax = a << ax; }
    else if (op == SHR_OP) { a = mem_ri(sp); sp = sp + 8; ax = (long long)((unsigned long long)a >> ax); }

    /* Comparison - native */
    else if (op == EQ_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a == ax); }
    else if (op == NE_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a != ax); }
    else if (op == LT_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a < ax); }
    else if (op == GT_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a > ax); }
    else if (op == LE_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a <= ax); }
    else if (op == GE_OP) { a = mem_ri(sp); sp = sp + 8; ax = (a >= ax); }

    /* I/O */
    else if (op == GETC_OP) { ax = getchar(); if (ax < 0) ax = 0; }
    else if (op == PUTC_OP) { putchar((int)mem_ri(sp)); }
    else if (op == EXIT_OP) { halted = 1; return (int)mem_ri(sp); }

    return 0;
}

int run(int argc, char **argv) {
    int cycles;
    int i;
    int j;
    int len;
    long long str_base;
    long long argv_base;

    mem_size = 1024 * 1024;
    memory = malloc(mem_size);
    i = 0;
    while (i < mem_size) { memory[i] = 0; i = i + 1; }

    /* Copy argv strings to memory at 0x80000 */
    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        j = 0;
        while (argv[i][j]) {
            memory[str_base + j] = argv[i][j];
            j = j + 1;
        }
        memory[str_base + j] = 0;  /* null terminator */
        str_base = str_base + j + 1;
        i = i + 1;
    }

    /* Build argv array at 0x7F000 */
    argv_base = 0x7F000;
    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        mem_wi(argv_base + i * 8, str_base);
        j = 0;
        while (memory[str_base + j]) { j = j + 1; }
        str_base = str_base + j + 1;
        i = i + 1;
    }

    /* Stack setup */
    sp = 0x90000 + 0x40000 - 8;
    bp = sp;
    pc = 0;
    ax = 0;
    halted = 0;

    /* Push argv and argc onto stack (left-to-right for this compiler) */
    sp = sp - 8;
    mem_wi(sp, argc);       /* argc - first */
    sp = sp - 8;
    mem_wi(sp, argv_base);  /* argv - second */

    code = (long long *)bundled_bytecode;
    code_len = bundled_bytecode_len / 8;

    if (bundled_data_len > 0) {
        i = 0;
        while (i < bundled_data_len) {
            memory[0x10000 + i] = bundled_data[i];
            i = i + 1;
        }
    }

    cycles = 0;
    while (!halted) {
        step();
        cycles = cycles + 1;
        if (cycles > 100000000) { halted = 1; }
    }

    return (int)ax;
}

int main(int argc, char **argv) {
    init_opcodes();
    init_neural();
    return run(argc, argv);
}
'''


def create_neural_bundle_int(source_path: str, output_path: str) -> dict:
    """Create neural bundle with ONNX runtime + hardcoded bytecode."""
    with open(source_path, 'r') as f:
        source = f.read()

    bytecode_list, data = compile_c(source)

    bytecode_bytes = b''
    for instr in bytecode_list:
        if instr < 0:
            instr = instr & 0xFFFFFFFFFFFFFFFF
        bytecode_bytes += struct.pack('<Q', instr)

    data_bytes = data if data else b''

    # Generate C
    c_code = '''/*
 * Neural VM Bundle
 * ONNX runtime + hardcoded bytecode
 * C4 compatible (no floats, no for loops)
 *
 * Build: gcc -o prog prog.c
 */

int printf(char *fmt, ...);
int putchar(int c);
int getchar();
char *malloc(int size);
int free(char *p);
int exit(int code);

'''

    # Bundled bytecode
    c_code += '/* ============ BUNDLED PROGRAM ============ */\n\n'
    c_code += generate_c_array('bundled_bytecode', bytecode_bytes) + '\n\n'
    if data_bytes:
        c_code += generate_c_array('bundled_data', data_bytes) + '\n\n'
    else:
        c_code += 'unsigned char bundled_data[] = {};\nint bundled_data_len = 0;\n\n'

    # ONNX neural runtime
    c_code += ONNX_NEURAL_RUNTIME

    # VM code
    c_code += VM_CODE

    with open(output_path, 'w') as f:
        f.write(c_code)

    return {
        'bytecode_size': len(bytecode_bytes),
        'data_size': len(data_bytes),
        'source_size': len(c_code),
    }


def main():
    parser = argparse.ArgumentParser(description='Create neural VM bundle')
    parser.add_argument('--program', required=True, help='C source file')
    parser.add_argument('--output', required=True, help='Output C file')

    args = parser.parse_args()

    print(f"Creating neural VM bundle...")
    meta = create_neural_bundle_int(args.program, args.output)

    print(f"Bundle created:")
    print(f"  Bytecode: {meta['bytecode_size']} bytes")
    print(f"  Data: {meta['data_size']} bytes")
    print(f"  C source: {meta['source_size']} bytes")
    print()
    print(f"Compile with gcc: gcc -o prog {args.output}")
    print(f"Compile with c4:  ./c4 {args.output}")


if __name__ == "__main__":
    main()
