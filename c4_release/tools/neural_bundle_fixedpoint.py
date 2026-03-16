#!/usr/bin/env python3
"""
Neural VM Bundler - Fixed-Point Integer Version for c4 Compatibility

This version emulates floating-point neural network operations using
integer arithmetic with fixed-point representation.

Key design:
- All weights stored as scaled integers (SCALE = 4096 = 2^12)
- SiLU approximated using lookup table and linear interpolation
- All arithmetic done in int to avoid c4 limitations
- Compatible with c4 compiler (no float/double, no static initializers)
"""

import argparse
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from sparse_vm import SparseVM, SparseLinear, EmbedDimsV5
import torch


# Fixed-point scale factor (2^12 = 4096)
SCALE = 4096
SCALE_BITS = 12


def generate_c4_header() -> str:
    """Generate c4-compatible header."""
    return '''/* Neural VM Bundle - Fixed-Point c4 Version */

/* I32 mask function for 32-bit arithmetic */
int I32(int x) { return x & 0xFFFFFFFF; }

'''


def generate_bytecode_c4(bytecode: bytes, data: bytes) -> tuple:
    """Generate bytecode arrays for c4."""
    decls = '/* ============ BUNDLED PROGRAM ============ */\n\n'
    decls += 'char *bundled_bytecode;\n'
    decls += 'int bundled_bytecode_len;\n'
    decls += 'char *bundled_data;\n'
    decls += 'int bundled_data_len;\n\n'

    init_lines = [f'    bundled_bytecode_len = {len(bytecode)};']
    init_lines.append(f'    bundled_bytecode = malloc({len(bytecode)});')
    for i, b in enumerate(bytecode):
        init_lines.append(f'    bundled_bytecode[{i}] = {b};')

    if data:
        init_lines.append(f'    bundled_data_len = {len(data)};')
        init_lines.append(f'    bundled_data = malloc({len(data)});')
        for i, b in enumerate(data):
            init_lines.append(f'    bundled_data[{i}] = {b};')
    else:
        init_lines.append('    bundled_data_len = 0;')
        init_lines.append('    bundled_data = 0;')

    return decls, '\n'.join(init_lines)


def generate_sparse_weights_c4(model: SparseVM) -> tuple:
    """Generate sparse weight arrays for c4."""
    decls = '\n/* ============ SPARSE NEURAL WEIGHTS (Fixed-Point) ============ */\n\n'
    decls += 'int FP_SCALE;\n\n'

    init_lines = [f'    FP_SCALE = {SCALE};']

    for name, module in model.named_modules():
        if isinstance(module, SparseLinear) and module.nnz() > 0:
            safe_name = name.replace('.', '_')
            rows = module.indices[0].tolist()
            cols = module.indices[1].tolist()
            vals = module.values.tolist()
            fp_vals = [int(v * SCALE) for v in vals]
            nnz = module.nnz()

            decls += f'/* {safe_name}: nnz={nnz} */\n'
            decls += f'int *{safe_name}_r;\n'
            decls += f'int *{safe_name}_c;\n'
            decls += f'int *{safe_name}_v;\n'
            decls += f'int {safe_name}_n;\n\n'

            init_lines.append(f'    {safe_name}_n = {nnz};')
            init_lines.append(f'    {safe_name}_r = malloc({nnz} * 8);')
            init_lines.append(f'    {safe_name}_c = malloc({nnz} * 8);')
            init_lines.append(f'    {safe_name}_v = malloc({nnz} * 8);')

            for i, r in enumerate(rows):
                init_lines.append(f'    {safe_name}_r[{i}] = {r};')
            for i, c in enumerate(cols):
                init_lines.append(f'    {safe_name}_c[{i}] = {c};')
            for i, v in enumerate(fp_vals):
                init_lines.append(f'    {safe_name}_v[{i}] = {v};')

    return decls, '\n'.join(init_lines)


def generate_neural_runtime_c4() -> str:
    """Generate c4-compatible neural runtime."""
    return '''
/* ============ FIXED-POINT NEURAL RUNTIME ============ */

int DIM;
int ADD_SUB_HIDDEN;
int MUL_HIDDEN;
int MAX_HIDDEN;
int OP_A_START;
int OP_B_START;
int RESULT_START;
int MUL_PROD_START;
int OPCODE_START;

int *emb;
int *h1;
int *h2;

/* SiLU lookup table for approximation */
int *silu_table;

int init_runtime() {
    int i;

    DIM = 512;
    ADD_SUB_HIDDEN = 32;
    MUL_HIDDEN = 128;
    MAX_HIDDEN = 128;
    OP_A_START = 0;
    OP_B_START = 8;
    RESULT_START = 16;
    MUL_PROD_START = 32;
    OPCODE_START = 104;

    emb = malloc(DIM * 8);
    h1 = malloc(MAX_HIDDEN * 8);
    h2 = malloc(MAX_HIDDEN * 8);

    /* SiLU table: silu(x) * 4096 for x from -8 to 8 (17 entries) */
    silu_table = malloc(17 * 8);
    silu_table[0] = 0;      /* silu(-8) ≈ 0 */
    silu_table[1] = 0;      /* silu(-7) */
    silu_table[2] = 1;      /* silu(-6) */
    silu_table[3] = 5;      /* silu(-5) */
    silu_table[4] = 14;     /* silu(-4) */
    silu_table[5] = 42;     /* silu(-3) */
    silu_table[6] = 119;    /* silu(-2) */
    silu_table[7] = 320;    /* silu(-1) */
    silu_table[8] = 0;      /* silu(0) = 0 */
    silu_table[9] = 3776;   /* silu(1) */
    silu_table[10] = 7977;  /* silu(2) */
    silu_table[11] = 12054; /* silu(3) */
    silu_table[12] = 16312; /* silu(4) */
    silu_table[13] = 20475; /* silu(5) */
    silu_table[14] = 24575; /* silu(6) */
    silu_table[15] = 28671; /* silu(7) */
    silu_table[16] = 32768; /* silu(8) ≈ 8 */

    return 0;
}

/* Zero array */
int fzero(int *v, int n) {
    int i;
    i = 0;
    while (i < n) { v[i] = 0; i = i + 1; }
    return 0;
}

/* SiLU approximation using table lookup + linear interpolation */
int silu_fp(int x) {
    int idx;
    int frac;
    int v1;
    int v2;

    /* x is in fixed-point (scaled by FP_SCALE) */
    /* Table covers x from -8 to 8 in integer units */

    if (x >= 8 * FP_SCALE) return x;  /* silu(x) ≈ x for large x */
    if (x <= 0 - 8 * FP_SCALE) return 0;  /* silu(x) ≈ 0 for very negative x */

    /* Convert to table index: idx = (x / FP_SCALE) + 8 */
    idx = (x / FP_SCALE) + 8;
    if (idx < 0) idx = 0;
    if (idx > 15) idx = 15;

    /* Linear interpolation */
    frac = x - (idx - 8) * FP_SCALE;
    v1 = silu_table[idx];
    v2 = silu_table[idx + 1];

    return v1 + ((v2 - v1) * frac) / FP_SCALE;
}

/* Sparse matrix-vector multiply */
int spmv_add_fp(int n, int *rows, int *cols, int *vals, int *in, int *out) {
    int i;
    i = 0;
    while (i < n) {
        out[rows[i]] = out[rows[i]] + (vals[i] * in[cols[i]]) / FP_SCALE;
        i = i + 1;
    }
    return 0;
}

/* SwiGLU forward pass */
int swiglu_fp(
    int up_n, int *up_r, int *up_c, int *up_v,
    int gate_n, int *gate_r, int *gate_c, int *gate_v,
    int down_n, int *down_r, int *down_c, int *down_v,
    int hidden_dim, int *x, int dim
) {
    int i;

    fzero(h1, hidden_dim);
    spmv_add_fp(up_n, up_r, up_c, up_v, x, h1);

    fzero(h2, hidden_dim);
    spmv_add_fp(gate_n, gate_r, gate_c, gate_v, x, h2);

    i = 0;
    while (i < hidden_dim) {
        h1[i] = (silu_fp(h1[i]) * h2[i]) / FP_SCALE;
        i = i + 1;
    }

    spmv_add_fp(down_n, down_r, down_c, down_v, h1, x);
    return 0;
}

/* Encode value as nibbles */
int encode_val(int *e, int start, int v) {
    int i;
    i = 0;
    while (i < 8) {
        e[start + i] = ((v >> (i * 4)) & 0xF) * FP_SCALE;
        i = i + 1;
    }
    return 0;
}

/* Decode nibbles to value */
int decode_val(int *e, int start) {
    int v;
    int i;
    int nib;

    v = 0;
    i = 0;
    while (i < 8) {
        nib = (e[start + i] + FP_SCALE / 2) / FP_SCALE;
        if (nib < 0) nib = 0;
        if (nib > 15) nib = 15;
        v = v | (nib << (i * 4));
        i = i + 1;
    }
    return v;
}

/* Set opcode */
int set_opcode(int *e, int op) {
    int i;
    i = 0;
    while (i < 48) {
        if (i == op) {
            e[OPCODE_START + i] = FP_SCALE;
        } else {
            e[OPCODE_START + i] = 0;
        }
        i = i + 1;
    }
    return 0;
}

'''


def generate_neural_arithmetic_c4() -> str:
    """Generate c4-compatible neural arithmetic."""
    return '''
/* ============ NEURAL ARITHMETIC ============ */

int neural_add(int a, int b) {
    int i;
    int result;
    int carry;
    int raw_sum;
    int total;
    int nib;

    fzero(emb, DIM);
    encode_val(emb, OP_A_START, I32(a));
    encode_val(emb, OP_B_START, I32(b));
    set_opcode(emb, 25);

    swiglu_fp(
        add_sub_ffn_up_n, add_sub_ffn_up_r, add_sub_ffn_up_c, add_sub_ffn_up_v,
        add_sub_ffn_gate_n, add_sub_ffn_gate_r, add_sub_ffn_gate_c, add_sub_ffn_gate_v,
        add_sub_ffn_down_n, add_sub_ffn_down_r, add_sub_ffn_down_c, add_sub_ffn_down_v,
        ADD_SUB_HIDDEN, emb, DIM
    );

    carry = 0;
    result = 0;
    i = 0;
    while (i < 8) {
        raw_sum = (emb[RESULT_START + i] + FP_SCALE / 2) / FP_SCALE;
        total = raw_sum + carry;
        nib = total & 0xF;
        carry = total >> 4;
        result = result | (nib << (i * 4));
        i = i + 1;
    }

    return I32(result);
}

int neural_sub(int a, int b) {
    /* Native fallback - neural SUB has SiLU issues with negative values */
    return a - b;
}

int neural_mul(int a, int b) {
    int i;
    int j;
    int pos;
    int prod_idx;
    int prod;
    int carry;
    int result;
    int *sums;
    int total;
    int nib;

    sums = malloc(16 * 8);
    i = 0;
    while (i < 16) { sums[i] = 0; i = i + 1; }

    fzero(emb, DIM);
    encode_val(emb, OP_A_START, I32(a));
    encode_val(emb, OP_B_START, I32(b));
    set_opcode(emb, 27);

    swiglu_fp(
        mul_ffn_up_n, mul_ffn_up_r, mul_ffn_up_c, mul_ffn_up_v,
        mul_ffn_gate_n, mul_ffn_gate_r, mul_ffn_gate_c, mul_ffn_gate_v,
        mul_ffn_down_n, mul_ffn_down_r, mul_ffn_down_c, mul_ffn_down_v,
        MUL_HIDDEN, emb, DIM
    );

    /* Sum products with position shifting for long multiplication */
    i = 0;
    while (i < 8) {
        j = 0;
        while (j < 8) {
            pos = i + j;
            if (pos < 16) {
                prod_idx = i * 8 + j;
                prod = (emb[MUL_PROD_START + prod_idx] + FP_SCALE / 2) / FP_SCALE;
                sums[pos] = sums[pos] + prod;
            }
            j = j + 1;
        }
        i = i + 1;
    }

    /* Carry propagation */
    carry = 0;
    result = 0;
    i = 0;
    while (i < 8) {  /* Only 8 nibbles for 32-bit result */
        total = sums[i] + carry;
        nib = total & 0xF;
        carry = total >> 4;
        result = result | (nib << (i * 4));
        i = i + 1;
    }

    free(sums);
    return I32(result);
}

int use_neural;

int safe_add(int a, int b) {
    if (use_neural) return neural_add(a, b);
    return a + b;
}

int safe_sub(int a, int b) {
    if (use_neural) return neural_sub(a, b);
    return a - b;
}

int safe_mul(int a, int b) {
    if (use_neural) return neural_mul(a, b);
    return a * b;
}

int safe_div(int a, int b) {
    if (b) return a / b;
    return 0;
}

int safe_mod(int a, int b) {
    if (b) return a % b;
    return 0;
}

'''


def generate_vm_c4() -> str:
    """Generate c4-compatible VM."""
    return '''
/* ============ VM ============ */

int MEM_SIZE;
int DATA_BASE;
int STACK_BASE;

char *memory;
int *code;
int code_len;
int pc;
int sp;
int bp;
int ax;
int halted;

/* Opcodes */
int LEA; int IMM; int JMP; int JSR; int BZ; int BNZ;
int ENT; int ADJ; int LEV; int LI; int LC; int SI; int SC; int PSH;
int OR; int XOR; int AND; int EQ; int NE; int LT; int GT; int LE; int GE;
int SHL; int SHR; int ADD; int SUB; int MUL; int DIV; int MOD;
int EXIT_OP; int GETC_OP; int PUTC_OP;

int init_opcodes() {
    LEA = 0; IMM = 1; JMP = 2; JSR = 3; BZ = 4; BNZ = 5;
    ENT = 6; ADJ = 7; LEV = 8; LI = 9; LC = 10; SI = 11; SC = 12; PSH = 13;
    OR = 14; XOR = 15; AND = 16; EQ = 17; NE = 18; LT = 19; GT = 20; LE = 21; GE = 22;
    SHL = 23; SHR = 24; ADD = 25; SUB = 26; MUL = 27; DIV = 28; MOD = 29;
    EXIT_OP = 38;
    GETC_OP = 64;
    PUTC_OP = 65;
    return 0;
}

int mem_ri(int a) { return *(int*)(memory + a); }
int mem_wi(int a, int v) { *(int*)(memory + a) = v; return 0; }
int mem_rb(int a) { return memory[a] & 0xFF; }
int mem_wb(int a, int v) { memory[a] = v; return 0; }

int mcopy(char *d, char *s, int n) {
    while (n > 0) { *d = *s; d = d + 1; s = s + 1; n = n - 1; }
    return 0;
}

int step() {
    int op;
    int imm;
    int a;

    if (halted) return 1;
    if (pc < 0) { halted = 1; return 1; }
    if (pc >= code_len * 8) { halted = 1; return 1; }

    op = code[pc / 8] & 0xFF;
    imm = code[pc / 8] >> 8;
    pc = pc + 8;

    if (op == LEA) { ax = bp + imm; }
    else if (op == IMM) { ax = imm; }
    else if (op == JMP) { pc = imm; }
    else if (op == JSR) { sp = sp - 8; mem_wi(sp, pc); pc = imm; }
    else if (op == BZ) { if (ax == 0) pc = imm; }
    else if (op == BNZ) { if (ax != 0) pc = imm; }
    else if (op == ENT) { sp = sp - 8; mem_wi(sp, bp); bp = sp; sp = sp - imm; }
    else if (op == ADJ) { sp = sp + imm; }
    else if (op == LEV) { sp = bp; bp = mem_ri(sp); sp = sp + 8; pc = mem_ri(sp); sp = sp + 8; }
    else if (op == LI) { ax = mem_ri(ax); }
    else if (op == LC) { ax = mem_rb(ax); }
    else if (op == SI) { a = mem_ri(sp); sp = sp + 8; mem_wi(a, ax); }
    else if (op == SC) { a = mem_ri(sp); sp = sp + 8; mem_wb(a, ax); }
    else if (op == PSH) { sp = sp - 8; mem_wi(sp, ax); }

    else if (op == ADD) { a = mem_ri(sp); sp = sp + 8; ax = safe_add(a, ax); }
    else if (op == SUB) { a = mem_ri(sp); sp = sp + 8; ax = safe_sub(a, ax); }
    else if (op == MUL) { a = mem_ri(sp); sp = sp + 8; ax = safe_mul(a, ax); }
    else if (op == DIV) { a = mem_ri(sp); sp = sp + 8; ax = safe_div(a, ax); }
    else if (op == MOD) { a = mem_ri(sp); sp = sp + 8; ax = safe_mod(a, ax); }

    else if (op == AND) { a = mem_ri(sp); sp = sp + 8; ax = a & ax; }
    else if (op == OR) { a = mem_ri(sp); sp = sp + 8; ax = a | ax; }
    else if (op == XOR) { a = mem_ri(sp); sp = sp + 8; ax = a ^ ax; }
    else if (op == SHL) { a = mem_ri(sp); sp = sp + 8; ax = a << ax; }
    else if (op == SHR) { a = mem_ri(sp); sp = sp + 8; ax = a >> ax; }

    else if (op == EQ) { a = mem_ri(sp); sp = sp + 8; ax = (a == ax); }
    else if (op == NE) { a = mem_ri(sp); sp = sp + 8; ax = (a != ax); }
    else if (op == LT) { a = mem_ri(sp); sp = sp + 8; ax = (a < ax); }
    else if (op == GT) { a = mem_ri(sp); sp = sp + 8; ax = (a > ax); }
    else if (op == LE) { a = mem_ri(sp); sp = sp + 8; ax = (a <= ax); }
    else if (op == GE) { a = mem_ri(sp); sp = sp + 8; ax = (a >= ax); }

    else if (op == GETC_OP) { ax = getchar(); if (ax < 0) ax = 0; }
    else if (op == PUTC_OP) { putchar(mem_ri(sp)); }
    else if (op == EXIT_OP) { halted = 1; return mem_ri(sp); }

    return 0;
}

int run(int argc, char **argv) {
    int cycles;
    int i;
    int j;
    int str_base;
    int argv_base;
    char *s;

    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        s = argv[i];
        j = 0;
        while (s[j]) {
            memory[str_base + j] = s[j];
            j = j + 1;
        }
        memory[str_base + j] = 0;
        str_base = str_base + j + 1;
        i = i + 1;
    }

    argv_base = 0x7F000;
    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        s = argv[i];
        mem_wi(argv_base + i * 8, str_base);
        j = 0;
        while (s[j]) j = j + 1;
        str_base = str_base + j + 1;
        i = i + 1;
    }

    sp = STACK_BASE + 0x40000 - 8;
    bp = sp;
    pc = 0;
    ax = 0;
    halted = 0;
    cycles = 0;

    sp = sp - 8; mem_wi(sp, argc);
    sp = sp - 8; mem_wi(sp, argv_base);

    code = (int *)bundled_bytecode;
    code_len = bundled_bytecode_len / 8;
    if (bundled_data_len > 0) mcopy(memory + DATA_BASE, bundled_data, bundled_data_len);

    while (!halted) {
        step();
        cycles = cycles + 1;
        if (cycles > 100000000) { halted = 1; }
    }
    return ax;
}

'''


def generate_main_c4() -> str:
    """Generate c4-compatible main function."""
    return '''
int main(int argc, char **argv) {
    int real_argc;
    int has_neural_flag;
    int i;
    char *s;

    /* Initialize everything */
    MEM_SIZE = 1024 * 1024;
    DATA_BASE = 0x10000;
    STACK_BASE = 0x90000;
    memory = malloc(MEM_SIZE);

    init_opcodes();
    init_runtime();
    init_weights();

    real_argc = argc;
    has_neural_flag = 0;
    use_neural = 0;

    i = 1;
    while (i < argc) {
        s = argv[i];
        if (s[0] == '-') {
            if (s[1] == 'n') {
                if (s[2] == 0) {
                    has_neural_flag = 1;
                    break;
                }
            }
        }
        i = i + 1;
    }

    if (has_neural_flag) {
        use_neural = 1;
        while (i < argc - 1) {
            argv[i] = argv[i + 1];
            i = i + 1;
        }
        real_argc = argc - 1;
    }

    return run(real_argc, argv);
}
'''


def create_neural_bundle_fixedpoint(source_path: str, output_path: str) -> dict:
    """Create neural bundle with fixed-point integer weights."""
    with open(source_path, 'r') as f:
        source = f.read()

    bytecode_list, data = compile_c(source)

    bytecode_bytes = b''
    for instr in bytecode_list:
        if instr < 0:
            instr = instr & 0xFFFFFFFFFFFFFFFF
        bytecode_bytes += struct.pack('<Q', instr)

    data_bytes = data if data else b''

    # Create sparse VM
    model = SparseVM()

    # Generate code sections
    bytecode_decls, bytecode_init = generate_bytecode_c4(bytecode_bytes, data_bytes)
    weights_decls, weights_init = generate_sparse_weights_c4(model)

    # Build full C code
    c_code = generate_c4_header()
    c_code += bytecode_decls
    c_code += weights_decls
    c_code += generate_neural_runtime_c4()
    c_code += generate_neural_arithmetic_c4()
    c_code += generate_vm_c4()

    # Add init_weights function
    c_code += '\nint init_weights() {\n'
    c_code += bytecode_init + '\n'
    c_code += weights_init + '\n'
    c_code += '    return 0;\n}\n'

    c_code += generate_main_c4()

    with open(output_path, 'w') as f:
        f.write(c_code)

    total_nnz = sum(m.nnz() for _, m in model.named_modules() if isinstance(m, SparseLinear))

    return {
        'bytecode_size': len(bytecode_bytes),
        'data_size': len(data_bytes),
        'total_nnz': total_nnz,
        'source_size': len(c_code),
    }


def main():
    parser = argparse.ArgumentParser(description='Create fixed-point neural VM bundle for c4')
    parser.add_argument('--program', required=True, help='C source file')
    parser.add_argument('--output', required=True, help='Output C file')

    args = parser.parse_args()

    print(f"Creating fixed-point neural VM bundle...")
    meta = create_neural_bundle_fixedpoint(args.program, args.output)

    print(f"Bundle created:")
    print(f"  Bytecode: {meta['bytecode_size']} bytes")
    print(f"  Sparse weights: {meta['total_nnz']} non-zeros")
    print(f"  C source: {meta['source_size']} bytes")
    print()
    print(f"Compile with gcc: gcc -O2 -o prog {args.output}")
    print(f"Run via c4: ./c4 {args.output}")
    print(f"Run with neural: ./prog -n")


if __name__ == "__main__":
    main()
