#!/usr/bin/env python3
"""
Freestanding Neural VM Bundler

Creates a standalone executable with NO libc dependency.
Uses raw system calls for I/O.

Supports:
- Linux x86_64 (syscalls)
- macOS x86_64/arm64 (syscalls)
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


def generate_sparse_weight_c(name: str, layer: SparseLinear) -> str:
    """Generate C code for a sparse weight matrix."""
    if layer.nnz() == 0:
        return f"/* {name}: empty */\n"

    rows = layer.indices[0].tolist()
    cols = layer.indices[1].tolist()
    vals = layer.values.tolist()

    # Pack as simple arrays
    code = f"/* {name}: {layer.out_features}x{layer.in_features}, nnz={layer.nnz()} */\n"
    code += f"static const short {name}_r[] = {{{','.join(str(r) for r in rows)}}};\n"
    code += f"static const short {name}_c[] = {{{','.join(str(c) for c in cols)}}};\n"
    code += f"static const int {name}_n = {layer.nnz()};\n"
    # For fixed-point: scale by 256
    fixed_vals = [int(v * 256) for v in vals]
    code += f"static const short {name}_v[] = {{{','.join(str(v) for v in fixed_vals)}}};\n\n"
    return code


def generate_freestanding_runtime() -> str:
    """Generate freestanding runtime with raw syscalls."""
    return '''/*
 * Freestanding Neural VM Bundle
 *
 * NO libc dependency - uses raw system calls.
 * Compile with: clang -nostdlib -O2 -o prog prog.c
 *
 * On macOS: clang -nostdlib -e _start -O2 -o prog prog.c
 * On Linux: gcc -nostdlib -O2 -o prog prog.c
 */

typedef long long int64;
typedef unsigned long long uint64;
typedef unsigned char byte;
typedef short int16;

/* ============ SYSTEM CALLS ============ */

#ifdef __APPLE__
/* macOS syscalls (add 0x2000000 to syscall number) */
#define SYS_exit   0x2000001
#define SYS_read   0x2000003
#define SYS_write  0x2000004

#ifdef __aarch64__
/* ARM64 macOS */
static inline int64 syscall3(int64 num, int64 a1, int64 a2, int64 a3) {
    register int64 x16 __asm__("x16") = num;
    register int64 x0 __asm__("x0") = a1;
    register int64 x1 __asm__("x1") = a2;
    register int64 x2 __asm__("x2") = a3;
    __asm__ volatile("svc #0x80" : "+r"(x0) : "r"(x16), "r"(x1), "r"(x2) : "memory");
    return x0;
}
static inline void syscall1(int64 num, int64 a1) {
    register int64 x16 __asm__("x16") = num;
    register int64 x0 __asm__("x0") = a1;
    __asm__ volatile("svc #0x80" : : "r"(x16), "r"(x0));
}
#else
/* x86_64 macOS */
static inline int64 syscall3(int64 num, int64 a1, int64 a2, int64 a3) {
    int64 ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(num), "D"(a1), "S"(a2), "d"(a3) : "rcx", "r11", "memory");
    return ret;
}
static inline void syscall1(int64 num, int64 a1) {
    __asm__ volatile("syscall" : : "a"(num), "D"(a1) : "rcx", "r11", "memory");
}
#endif

#else
/* Linux syscalls */
#define SYS_exit   60
#define SYS_read   0
#define SYS_write  1

static inline int64 syscall3(int64 num, int64 a1, int64 a2, int64 a3) {
    int64 ret;
    __asm__ volatile("syscall" : "=a"(ret) : "a"(num), "D"(a1), "S"(a2), "d"(a3) : "rcx", "r11", "memory");
    return ret;
}
static inline void syscall1(int64 num, int64 a1) {
    __asm__ volatile("syscall" : : "a"(num), "D"(a1) : "rcx", "r11", "memory");
}
#endif

/* ============ I/O PRIMITIVES ============ */

static void sys_exit(int code) {
    syscall1(SYS_exit, code);
    __builtin_unreachable();
}

static int sys_write(int fd, const void *buf, int len) {
    return (int)syscall3(SYS_write, fd, (int64)buf, len);
}

static int sys_read(int fd, void *buf, int len) {
    return (int)syscall3(SYS_read, fd, (int64)buf, len);
}

static void putchar_raw(int c) {
    byte ch = (byte)c;
    sys_write(1, &ch, 1);
}

static int getchar_raw(void) {
    byte ch;
    int n = sys_read(0, &ch, 1);
    if (n <= 0) return 0;  /* EOF -> 0 */
    return ch;
}

/* ============ MEMORY (static allocation) ============ */

#define MEM_SIZE (1024 * 1024)  /* 1 MB */
static byte memory[MEM_SIZE];

#define DATA_BASE 0x10000
#define STACK_BASE 0x90000

/* Simple memcpy */
static void mcopy(void *dst, const void *src, int n) {
    byte *d = (byte *)dst;
    const byte *s = (const byte *)src;
    while (n-- > 0) *d++ = *s++;
}

/* Simple memset */
static void mset(void *dst, byte val, int n) {
    byte *d = (byte *)dst;
    while (n-- > 0) *d++ = val;
}

/* ============ FIXED-POINT MATH ============ */

/* 8.8 fixed point: multiply two fixed-point numbers */
static int fmul(int a, int b) {
    return (a * b) >> 8;
}

/* Approximate SiLU in fixed point: x * sigmoid(x) */
/* sigmoid(x) ≈ 0.5 + 0.25*x for small x, saturates to 0/1 */
static int silu_fixed(int x) {
    int sig;
    if (x > 512) {
        sig = 256;  /* sigmoid ≈ 1 */
    } else if (x < -512) {
        sig = 0;    /* sigmoid ≈ 0 */
    } else {
        sig = 128 + (x >> 2);  /* linear approximation */
    }
    return fmul(x, sig);
}

/* ============ SPARSE MATMUL ============ */

/* Sparse matrix-vector: out[row] += val * in[col] */
static void spmv(int n, const short *rows, const short *cols, const short *vals,
                 const int *x, int *out) {
    int i;
    i = 0;
    while (i < n) {
        out[rows[i]] += fmul(vals[i], x[cols[i]]);
        i = i + 1;
    }
}

'''


def generate_vm_freestanding() -> str:
    """Generate VM for freestanding environment."""
    return '''
/* ============ VM STATE ============ */

static int64 *code;
static int code_len;
static int64 pc, sp, bp, ax;
static int halted;

/* Memory access */
static int64 mem_ri(int64 a) { return *(int64 *)(memory + a); }
static void mem_wi(int64 a, int64 v) { *(int64 *)(memory + a) = v; }
static byte mem_rb(int64 a) { return memory[a]; }
static void mem_wb(int64 a, byte v) { memory[a] = v; }

/* Opcodes */
enum {
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV, LI, LC, SI, SC, PSH,
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE, SHL, SHR, ADD, SUB, MUL, DIV, MOD,
    OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCMP, EXIT_OP
};
#define GETC_OP 64
#define PUTC_OP 65

/* Run one step */
static int step(void) {
    int64 op, imm, a;

    if (halted) return 1;
    if (pc < 0 || pc >= code_len * 8) { halted = 1; return 1; }

    op = code[pc / 8] & 0xFF;
    imm = code[pc / 8] >> 8;
    if (imm >= (1LL << 55)) imm = imm - (1LL << 56);
    pc = pc + 8;

    if (op == LEA) { ax = bp + imm; }
    else if (op == IMM) { ax = imm; }
    else if (op == JMP) { pc = imm; }
    else if (op == JSR) { sp = sp - 8; mem_wi(sp, pc); pc = imm; }
    else if (op == BZ)  { if (ax == 0) pc = imm; }
    else if (op == BNZ) { if (ax != 0) pc = imm; }
    else if (op == ENT) { sp = sp - 8; mem_wi(sp, bp); bp = sp; sp = sp - imm; }
    else if (op == ADJ) { sp = sp + imm; }
    else if (op == LEV) { sp = bp; bp = mem_ri(sp); sp = sp + 8; pc = mem_ri(sp); sp = sp + 8; }
    else if (op == LI)  { ax = mem_ri(ax); }
    else if (op == LC)  { ax = mem_rb(ax); }
    else if (op == SI)  { a = mem_ri(sp); sp = sp + 8; mem_wi(a, ax); }
    else if (op == SC)  { a = mem_ri(sp); sp = sp + 8; mem_wb(a, (byte)ax); }
    else if (op == PSH) { sp = sp - 8; mem_wi(sp, ax); }
    else if (op == ADD) { a = mem_ri(sp); sp = sp + 8; ax = a + ax; }
    else if (op == SUB) { a = mem_ri(sp); sp = sp + 8; ax = a - ax; }
    else if (op == MUL) { a = mem_ri(sp); sp = sp + 8; ax = a * ax; }
    else if (op == DIV) { a = mem_ri(sp); sp = sp + 8; ax = ax ? a / ax : 0; }
    else if (op == MOD) { a = mem_ri(sp); sp = sp + 8; ax = ax ? a % ax : 0; }
    else if (op == AND) { a = mem_ri(sp); sp = sp + 8; ax = a & ax; }
    else if (op == OR)  { a = mem_ri(sp); sp = sp + 8; ax = a | ax; }
    else if (op == XOR) { a = mem_ri(sp); sp = sp + 8; ax = a ^ ax; }
    else if (op == SHL) { a = mem_ri(sp); sp = sp + 8; ax = a << ax; }
    else if (op == SHR) { a = mem_ri(sp); sp = sp + 8; ax = (int64)((uint64)a >> ax); }
    else if (op == EQ)  { a = mem_ri(sp); sp = sp + 8; ax = (a == ax); }
    else if (op == NE)  { a = mem_ri(sp); sp = sp + 8; ax = (a != ax); }
    else if (op == LT)  { a = mem_ri(sp); sp = sp + 8; ax = (a < ax); }
    else if (op == GT)  { a = mem_ri(sp); sp = sp + 8; ax = (a > ax); }
    else if (op == LE)  { a = mem_ri(sp); sp = sp + 8; ax = (a <= ax); }
    else if (op == GE)  { a = mem_ri(sp); sp = sp + 8; ax = (a >= ax); }
    else if (op == GETC_OP) { ax = getchar_raw(); }
    else if (op == PUTC_OP) { putchar_raw((int)mem_ri(sp)); }
    else if (op == EXIT_OP) { halted = 1; return (int)mem_ri(sp); }

    return 0;
}

/* Initialize and run */
static int run(void) {
    int cycles;

    /* Init state */
    sp = STACK_BASE + 0x40000 - 8;
    bp = sp;
    pc = 0;
    ax = 0;
    halted = 0;

    /* Load bytecode */
    code = (int64 *)bundled_bytecode;
    code_len = bundled_bytecode_len / 8;

    /* Copy data section */
    if (bundled_data_len > 0) {
        mcopy(memory + DATA_BASE, bundled_data, bundled_data_len);
    }

    /* Execute */
    cycles = 0;
    while (!halted && cycles < 100000000) {
        step();
        cycles = cycles + 1;
    }

    return (int)ax;
}

/* Entry point - must not be static and must be first */
__attribute__((visibility("default")))
__attribute__((used))
void _start(void) {
    int ret;
    ret = run();
    sys_exit(ret);
}

/* Alternative entry for different linkers */
__attribute__((visibility("default")))
__attribute__((used))
void start(void) {
    _start();
}

int main(void) {
    return run();
}
'''


def create_freestanding_bundle(source_path: str, output_path: str) -> dict:
    """Create freestanding bundle."""
    with open(source_path, 'r') as f:
        source = f.read()

    bytecode_list, data = compile_c(source)

    bytecode_bytes = b''
    for instr in bytecode_list:
        if instr < 0:
            instr = instr & 0xFFFFFFFFFFFFFFFF
        bytecode_bytes += struct.pack('<Q', instr)

    data_bytes = data if data else b''

    # Get sparse weights
    model = SparseVM()

    # Generate C
    c_code = generate_freestanding_runtime()

    # Add sparse weights
    c_code += '\n/* ============ SPARSE WEIGHTS ============ */\n\n'
    total_nnz = 0
    for name, module in model.named_modules():
        if isinstance(module, SparseLinear) and module.nnz() > 0:
            safe_name = name.replace('.', '_')
            c_code += generate_sparse_weight_c(safe_name, module)
            total_nnz += module.nnz()

    # Add bytecode
    c_code += '\n/* ============ BUNDLED PROGRAM ============ */\n\n'
    c_code += generate_c_array('bundled_bytecode', bytecode_bytes) + '\n\n'
    c_code += generate_c_array('bundled_data', data_bytes) if data_bytes else \
              'static const unsigned char bundled_data[] = {};\nstatic const int bundled_data_len = 0;\n'

    # Add VM
    c_code += generate_vm_freestanding()

    with open(output_path, 'w') as f:
        f.write(c_code)

    return {
        'bytecode_size': len(bytecode_bytes),
        'data_size': len(data_bytes),
        'total_nnz': total_nnz,
        'source_size': len(c_code),
    }


def main():
    parser = argparse.ArgumentParser(description='Create freestanding neural VM bundle')
    parser.add_argument('--program', required=True, help='C source file')
    parser.add_argument('--output', required=True, help='Output C file')

    args = parser.parse_args()

    print(f"Creating freestanding neural VM bundle...")
    meta = create_freestanding_bundle(args.program, args.output)

    print(f"Bundle created:")
    print(f"  Bytecode: {meta['bytecode_size']} bytes")
    print(f"  Sparse weights: {meta['total_nnz']} non-zeros")
    print(f"  C source: {meta['source_size']} bytes")
    print()
    print("To compile (macOS arm64):")
    print(f"  clang -nostdlib -e _start -O2 -o prog {args.output}")
    print()
    print("To compile (Linux x86_64):")
    print(f"  gcc -nostdlib -O2 -o prog {args.output}")


if __name__ == "__main__":
    main()
