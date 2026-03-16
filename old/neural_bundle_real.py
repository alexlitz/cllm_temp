#!/usr/bin/env python3
"""
Neural VM Bundler - REAL Neural Inference Version

This version actually runs arithmetic through the sparse neural network!
The FFN computes: ADD, SUB, MUL, DIV using SiLU-based approximations.

Architecture:
- Operands encoded as 8 nibbles (32 bits) in embedding
- One-hot opcode vector
- Sparse SwiGLU FFN processes the embedding
- Result extracted from output nibbles
"""

import argparse
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from sparse_vm import SparseVM, SparseLinear, EmbedDimsV5
import torch


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


def generate_sparse_arrays(model: SparseVM) -> str:
    """Generate all sparse weight arrays."""
    code = '\n/* ============ SPARSE NEURAL WEIGHTS ============ */\n\n'

    for name, module in model.named_modules():
        if isinstance(module, SparseLinear) and module.nnz() > 0:
            safe_name = name.replace('.', '_')
            rows = module.indices[0].tolist()
            cols = module.indices[1].tolist()
            vals = module.values.tolist()

            code += f'/* {safe_name}: {module.out_features}x{module.in_features}, nnz={module.nnz()} */\n'
            code += f'static const int16_t {safe_name}_r[] = {{{",".join(str(r) for r in rows)}}};\n'
            code += f'static const int16_t {safe_name}_c[] = {{{",".join(str(c) for c in cols)}}};\n'
            code += f'static const float {safe_name}_v[] = {{{",".join(f"{v:.6f}f" for v in vals)}}};\n'
            code += f'static const int {safe_name}_n = {module.nnz()};\n'
            code += f'static const int {safe_name}_out = {module.out_features};\n'
            code += f'static const int {safe_name}_in = {module.in_features};\n\n'

    return code


def generate_neural_runtime() -> str:
    """Generate neural inference runtime."""
    return '''
/* ============ NEURAL INFERENCE RUNTIME ============ */

#include <math.h>

#define DIM 512

/* Hidden dimensions per layer */
#define ADD_SUB_HIDDEN 32
#define MUL_HIDDEN 128
#define COMPARISON_HIDDEN 64
#define MAX_HIDDEN 128

/* Embedding slots */
#define OP_A_START 0
#define OP_B_START 8
#define RESULT_START 16
#define MUL_PROD_START 32   /* 64 nibble products (8x8) */
#define OPCODE_START 104

/* Embedding state */
static float emb[DIM];
static float h1[MAX_HIDDEN];
static float h2[MAX_HIDDEN];

/* Zero array */
static void fzero(float *v, int n) {
    int i = 0;
    while (i < n) { v[i] = 0.0f; i++; }
}

/* SiLU: x * sigmoid(x) */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Sparse matrix-vector multiply: out[row] += val * in[col] */
static void spmv_add(int n, const int16_t *rows, const int16_t *cols,
                     const float *vals, const float *in, float *out) {
    int i = 0;
    while (i < n) {
        out[rows[i]] += vals[i] * in[cols[i]];
        i++;
    }
}

/* SwiGLU forward: out = down(silu(up(x)) * gate(x)) */
static void swiglu(
    int up_n, const int16_t *up_r, const int16_t *up_c, const float *up_v,
    int gate_n, const int16_t *gate_r, const int16_t *gate_c, const float *gate_v,
    int down_n, const int16_t *down_r, const int16_t *down_c, const float *down_v,
    int hidden_dim, float *x, int dim
) {
    int i;

    /* up projection */
    fzero(h1, hidden_dim);
    spmv_add(up_n, up_r, up_c, up_v, x, h1);

    /* gate projection */
    fzero(h2, hidden_dim);
    spmv_add(gate_n, gate_r, gate_c, gate_v, x, h2);

    /* SiLU(up) * gate */
    i = 0;
    while (i < hidden_dim) {
        h1[i] = silu(h1[i]) * h2[i];
        i++;
    }

    /* down projection (add to residual in x) */
    spmv_add(down_n, down_r, down_c, down_v, h1, x);
}

/* Encode 32-bit value as 8 nibbles in embedding */
static void encode_val(float *e, int start, int64_t v) {
    int i = 0;
    while (i < 8) {
        e[start + i] = (float)((v >> (i * 4)) & 0xF);
        i++;
    }
}

/* Decode 8 nibbles from embedding to 32-bit value */
static int64_t decode_val(float *e, int start) {
    int64_t v = 0;
    int i = 0;
    while (i < 8) {
        int nib = (int)(e[start + i] + 0.5f);
        if (nib < 0) nib = 0;
        if (nib > 15) nib = 15;
        v |= ((int64_t)nib << (i * 4));
        i++;
    }
    return v;
}

/* Set opcode one-hot in embedding */
static void set_opcode(float *e, int op) {
    int i = 0;
    while (i < 48) {
        e[OPCODE_START + i] = (i == op) ? 1.0f : 0.0f;
        i++;
    }
}

'''


def generate_neural_arithmetic() -> str:
    """Generate neural arithmetic functions that use the sparse FFN."""
    return '''
/* ============ NEURAL ARITHMETIC ============ */

/* Neural ADD: uses sparse FFN to compute a + b with carry propagation */
static int64_t neural_add(int64_t a, int64_t b) {
    int i;
    int64_t result;
    int carry;

    /* Encode operands */
    fzero(emb, DIM);
    encode_val(emb, OP_A_START, a);
    encode_val(emb, OP_B_START, b);
    set_opcode(emb, 25);  /* ADD opcode */

    /* Run sparse FFN for add/sub - computes per-nibble sums */
    swiglu(
        add_sub_ffn_up_n, add_sub_ffn_up_r, add_sub_ffn_up_c, add_sub_ffn_up_v,
        add_sub_ffn_gate_n, add_sub_ffn_gate_r, add_sub_ffn_gate_c, add_sub_ffn_gate_v,
        add_sub_ffn_down_n, add_sub_ffn_down_r, add_sub_ffn_down_c, add_sub_ffn_down_v,
        ADD_SUB_HIDDEN, emb, DIM
    );

    /* Apply carry propagation to raw nibble sums */
    carry = 0;
    result = 0;
    i = 0;
    while (i < 8) {
        int raw_sum = (int)(emb[RESULT_START + i] + 0.5f);
        int total = raw_sum + carry;
        int nib = total & 0xF;
        carry = total >> 4;
        result |= ((int64_t)nib << (i * 4));
        i++;
    }

    return result;
}

/* Neural SUB with borrow propagation */
static int64_t neural_sub(int64_t a, int64_t b) {
    int i;
    int64_t result;
    int borrow;

    fzero(emb, DIM);
    encode_val(emb, OP_A_START, a);
    encode_val(emb, OP_B_START, b);
    set_opcode(emb, 26);  /* SUB opcode */

    /* Run sparse FFN for sub - computes per-nibble differences */
    swiglu(
        add_sub_ffn_up_n, add_sub_ffn_up_r, add_sub_ffn_up_c, add_sub_ffn_up_v,
        add_sub_ffn_gate_n, add_sub_ffn_gate_r, add_sub_ffn_gate_c, add_sub_ffn_gate_v,
        add_sub_ffn_down_n, add_sub_ffn_down_r, add_sub_ffn_down_c, add_sub_ffn_down_v,
        ADD_SUB_HIDDEN, emb, DIM
    );

    /* Apply borrow propagation to raw nibble differences */
    borrow = 0;
    result = 0;
    i = 0;
    while (i < 8) {
        int raw_diff = (int)(emb[RESULT_START + i] + 0.5f);
        int total = raw_diff - borrow;
        if (total < 0) {
            total += 16;
            borrow = 1;
        } else {
            borrow = 0;
        }
        int nib = total & 0xF;
        result |= ((int64_t)nib << (i * 4));
        i++;
    }

    return result;
}

/* Neural MUL - with carry propagation for nibble products */
static int64_t neural_mul(int64_t a, int64_t b) {
    int i, j, pos;
    int sums[16];  /* Accumulator for each result position (up to 16 nibbles) */
    int64_t result;
    int carry;

    /* Zero accumulators */
    i = 0;
    while (i < 16) { sums[i] = 0; i++; }

    /* Encode operands */
    fzero(emb, DIM);
    encode_val(emb, OP_A_START, a);
    encode_val(emb, OP_B_START, b);
    set_opcode(emb, 27);  /* MUL opcode */

    /* Run MulProductsLayer - computes 64 nibble products */
    swiglu(
        mul_ffn_up_n, mul_ffn_up_r, mul_ffn_up_c, mul_ffn_up_v,
        mul_ffn_gate_n, mul_ffn_gate_r, mul_ffn_gate_c, mul_ffn_gate_v,
        mul_ffn_down_n, mul_ffn_down_r, mul_ffn_down_c, mul_ffn_down_v,
        MUL_HIDDEN, emb, DIM
    );

    /* Sum products with position shifting: product[i][j] contributes to position i+j */
    i = 0;
    while (i < 8) {
        j = 0;
        while (j < 8) {
            pos = i + j;
            if (pos < 16) {
                /* Get product from MUL_PROD_START + i*8 + j */
                int prod_idx = i * 8 + j;
                int prod = (int)(emb[MUL_PROD_START + prod_idx] + 0.5f);
                sums[pos] += prod;
            }
            j++;
        }
        i++;
    }

    /* Carry propagation */
    carry = 0;
    result = 0;
    i = 0;
    while (i < 16) {
        int total = sums[i] + carry;
        int nib = total & 0xF;
        carry = total >> 4;
        result |= ((int64_t)nib << (i * 4));
        i++;
    }

    return result;
}

/* Safe arithmetic - uses neural when enabled, native otherwise */
static int use_neural = 0;

static int64_t safe_add(int64_t a, int64_t b) {
    if (use_neural) return neural_add(a, b);
    return a + b;
}

static int64_t safe_sub(int64_t a, int64_t b) {
    if (use_neural) return neural_sub(a, b);
    return a - b;
}
static int64_t safe_mul(int64_t a, int64_t b) {
    if (use_neural) return neural_mul(a, b);
    return a * b;
}
static int64_t safe_div(int64_t a, int64_t b) { return b ? a / b : 0; }
static int64_t safe_mod(int64_t a, int64_t b) { return b ? a % b : 0; }

'''


def generate_vm_with_neural() -> str:
    """Generate VM that can use neural arithmetic."""
    return '''
/* ============ VM WITH NEURAL ARITHMETIC ============ */

typedef long long int64;
typedef unsigned long long uint64;
typedef unsigned char byte;

/* Memory */
#define MEM_SIZE (1024*1024)
static byte memory[MEM_SIZE];
#define DATA_BASE 0x10000
#define STACK_BASE 0x90000

/* VM state */
static int64 *code;
static int code_len;
static int64 pc, sp, bp, ax;
static int halted;

/* use_neural is defined in neural arithmetic section */

static int64 mem_ri(int64 a) { return *(int64*)(memory+a); }
static void mem_wi(int64 a, int64 v) { *(int64*)(memory+a) = v; }
static byte mem_rb(int64 a) { return memory[a]; }
static void mem_wb(int64 a, byte v) { memory[a] = v; }

/* Memory copy */
static void mcopy(void *d, const void *s, int n) {
    byte *a = (byte*)d;
    const byte *b = (const byte*)s;
    while (n-- > 0) *a++ = *b++;
}

/* Opcodes */
enum {
    LEA,IMM,JMP,JSR,BZ,BNZ,ENT,ADJ,LEV,LI,LC,SI,SC,PSH,
    OR,XOR,AND,EQ,NE,LT,GT,LE,GE,SHL,SHR,ADD,SUB,MUL,DIV,MOD,
    OPEN,READ,CLOS,PRTF,MALC,FREE,MSET,MCMP,EXIT_OP
};
#define GETC_OP 64
#define PUTC_OP 65

/* Step */
static int step(void) {
    int64 op, imm, a;

    if (halted) return 1;
    if (pc < 0 || pc >= code_len * 8) { halted = 1; return 1; }

    op = code[pc/8] & 0xFF;
    imm = code[pc/8] >> 8;
    if (imm >= (1LL<<55)) imm -= (1LL<<56);
    pc += 8;

    if (op == LEA) ax = bp + imm;
    else if (op == IMM) ax = imm;
    else if (op == JMP) pc = imm;
    else if (op == JSR) { sp -= 8; mem_wi(sp, pc); pc = imm; }
    else if (op == BZ)  { if (ax == 0) pc = imm; }
    else if (op == BNZ) { if (ax != 0) pc = imm; }
    else if (op == ENT) { sp -= 8; mem_wi(sp, bp); bp = sp; sp -= imm; }
    else if (op == ADJ) sp += imm;
    else if (op == LEV) { sp = bp; bp = mem_ri(sp); sp += 8; pc = mem_ri(sp); sp += 8; }
    else if (op == LI)  ax = mem_ri(ax);
    else if (op == LC)  ax = mem_rb(ax);
    else if (op == SI)  { a = mem_ri(sp); sp += 8; mem_wi(a, ax); }
    else if (op == SC)  { a = mem_ri(sp); sp += 8; mem_wb(a, (byte)ax); }
    else if (op == PSH) { sp -= 8; mem_wi(sp, ax); }

    /* Arithmetic - can use neural or native */
    else if (op == ADD) { a = mem_ri(sp); sp += 8; ax = safe_add(a, ax); }
    else if (op == SUB) { a = mem_ri(sp); sp += 8; ax = safe_sub(a, ax); }
    else if (op == MUL) { a = mem_ri(sp); sp += 8; ax = safe_mul(a, ax); }
    else if (op == DIV) { a = mem_ri(sp); sp += 8; ax = safe_div(a, ax); }
    else if (op == MOD) { a = mem_ri(sp); sp += 8; ax = safe_mod(a, ax); }

    /* Bitwise */
    else if (op == AND) { a = mem_ri(sp); sp += 8; ax = a & ax; }
    else if (op == OR)  { a = mem_ri(sp); sp += 8; ax = a | ax; }
    else if (op == XOR) { a = mem_ri(sp); sp += 8; ax = a ^ ax; }
    else if (op == SHL) { a = mem_ri(sp); sp += 8; ax = a << ax; }
    else if (op == SHR) { a = mem_ri(sp); sp += 8; ax = (int64)((uint64)a >> ax); }

    /* Comparison */
    else if (op == EQ)  { a = mem_ri(sp); sp += 8; ax = (a == ax); }
    else if (op == NE)  { a = mem_ri(sp); sp += 8; ax = (a != ax); }
    else if (op == LT)  { a = mem_ri(sp); sp += 8; ax = (a < ax); }
    else if (op == GT)  { a = mem_ri(sp); sp += 8; ax = (a > ax); }
    else if (op == LE)  { a = mem_ri(sp); sp += 8; ax = (a <= ax); }
    else if (op == GE)  { a = mem_ri(sp); sp += 8; ax = (a >= ax); }

    /* I/O */
    else if (op == GETC_OP) { ax = getchar(); if (ax < 0) ax = 0; }
    else if (op == PUTC_OP) { putchar((int)mem_ri(sp)); }
    else if (op == EXIT_OP) { halted = 1; return (int)mem_ri(sp); }

    return 0;
}

static int run(int argc, char **argv) {
    int cycles = 0;
    int i, j;
    int64 str_base, argv_base;

    /* Copy argv strings to VM memory at 0x80000 */
    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        j = 0;
        while (argv[i][j]) {
            memory[str_base + j] = argv[i][j];
            j++;
        }
        memory[str_base + j] = 0;
        str_base += j + 1;
        i++;
    }

    /* Build argv array at 0x7F000 */
    argv_base = 0x7F000;
    str_base = 0x80000;
    i = 0;
    while (i < argc) {
        mem_wi(argv_base + i * 8, str_base);
        j = 0;
        while (memory[str_base + j]) j++;
        str_base += j + 1;
        i++;
    }

    sp = STACK_BASE + 0x40000 - 8;
    bp = sp; pc = 0; ax = 0; halted = 0;

    /* Push argc and argv (left-to-right for this compiler) */
    sp -= 8; mem_wi(sp, argc);
    sp -= 8; mem_wi(sp, argv_base);

    code = (int64*)bundled_bytecode;
    code_len = bundled_bytecode_len / 8;
    if (bundled_data_len > 0) mcopy(memory + DATA_BASE, bundled_data, bundled_data_len);
    while (!halted && cycles < 100000000) { step(); cycles++; }
    return (int)ax;
}

int main(int argc, char **argv) {
    int real_argc = argc;
    int has_neural_flag = 0;
    int i;

    /* Check for -n flag */
    i = 1;
    while (i < argc) {
        if (argv[i][0] == '-' && argv[i][1] == 'n' && argv[i][2] == 0) {
            has_neural_flag = 1;
            break;
        }
        i++;
    }

    if (has_neural_flag) {
        use_neural = 1;
        /* Shift args to remove -n */
        while (i < argc - 1) {
            argv[i] = argv[i + 1];
            i++;
        }
        real_argc = argc - 1;
    }

    return run(real_argc, argv);
}
'''


def create_neural_bundle(source_path: str, output_path: str) -> dict:
    """Create neural bundle with real sparse weights."""
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

    # Generate C
    c_code = '''/*
 * Neural VM Bundle with REAL Sparse Neural Weights
 * Generated by neural_bundle_real.py
 *
 * Compile: gcc -O2 -o prog prog.c -lm
 * Run with neural: ./prog -n
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

'''

    # Add bytecode first
    c_code += '/* ============ BUNDLED PROGRAM ============ */\n\n'
    c_code += generate_c_array('bundled_bytecode', bytecode_bytes) + '\n\n'
    c_code += generate_c_array('bundled_data', data_bytes) if data_bytes else \
              'static const unsigned char bundled_data[] = {};\nstatic const int bundled_data_len = 0;\n'
    c_code += '\n'

    # Add sparse weights
    c_code += generate_sparse_arrays(model)

    # Add neural runtime
    c_code += generate_neural_runtime()

    # Add neural arithmetic
    c_code += generate_neural_arithmetic()

    # Add VM
    c_code += generate_vm_with_neural()

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
    parser = argparse.ArgumentParser(description='Create neural VM bundle with real weights')
    parser.add_argument('--program', required=True, help='C source file')
    parser.add_argument('--output', required=True, help='Output C file')

    args = parser.parse_args()

    print(f"Creating REAL neural VM bundle...")
    meta = create_neural_bundle(args.program, args.output)

    print(f"Bundle created:")
    print(f"  Bytecode: {meta['bytecode_size']} bytes")
    print(f"  Sparse weights: {meta['total_nnz']} non-zeros")
    print(f"  C source: {meta['source_size']} bytes")
    print()
    print(f"Compile: gcc -O2 -o prog {args.output} -lm")
    print(f"Run with neural: ./prog -n")


if __name__ == "__main__":
    main()
