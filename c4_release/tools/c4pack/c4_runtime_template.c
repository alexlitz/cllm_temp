/*
 * c4_runtime_template.c -- the standalone C4 VM runtime for c4pack.
 *
 * This is a self-contained, non-neural, non-torch C4-ISA interpreter.  It runs
 * the exact bytecode that src.compiler.compile_c emits, byte-for-byte identical
 * to the Python reference VM (c4_min.libprog_corpus.RefVM):
 *
 *   - PC and all branch / JSR / JMP targets are INSTRUCTION INDICES.
 *   - words are 32-bit two's-complement (SI/LI mask to 0xFFFFFFFF); SC/LC are
 *     byte store / SIGNED-char load; SHR is an ARITHMETIC (sign-filling) shift.
 *   - the stack descends by SLOT (8) per push; frame locals at BP-8/-16/...,
 *     params at BP+16/+24/...; ENT/ADJ immediates are BYTE sizes.
 *   - syscalls (OPEN/READ/CLOS/PRTF) PEEK their args off the stack and do NOT
 *     pop -- the compiler's trailing ADJ reclaims the pushed slots.
 *   - PRTF implements the c4 printf subset (%d %u %x %c %s %%), 32-bit.
 *
 * The bytecode + data segment are appended by c4pack.py as:
 *     static const long PROG_CODE[][2];   (op, imm)
 *     static const int  PROG_CODE_LEN;
 *     static const unsigned char PROG_DATA[];
 *     static const int  PROG_DATA_LEN;
 *
 * No external deps beyond the C standard library.  Compile with:
 *     gcc -O2 -static c4prog.c -o c4prog
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

/* ---- VM opcodes (src.compiler.Op) ------------------------------------- */
enum {
    LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV, LI, LC, SI, SC, PSH,
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE, SHL, SHR, ADD, SUB, MUL, DIV, MOD,
    OPEN, READ, CLOS, PRTF, MALC, FREE, MSET, MCMP, EXIT, NOP
};
#define GETCHAR 64
#define PUTCHAR 65

/* ---- memory model (matches RefVM) ------------------------------------- */
#define SLOT       8
#define STACK_TOP  0x100000
#define DATA_BASE  0x10000
#define HEAP_BASE  0x200000

/* Flat byte-addressable memory.  16 MB covers data + heap + descending stack
 * (stack top 0x100000, heap grows up from 0x200000). */
#define MEM_BYTES  (16 * 1024 * 1024)
static unsigned char MEM[MEM_BYTES];
static long heap_ptr = HEAP_BASE;

static unsigned mem_lw(long a) {           /* 32-bit little-endian word load */
    return (unsigned)MEM[a]
         | ((unsigned)MEM[a+1] << 8)
         | ((unsigned)MEM[a+2] << 16)
         | ((unsigned)MEM[a+3] << 24);
}
static void mem_sw(long a, unsigned v) {   /* 32-bit little-endian word store */
    MEM[a]   = (unsigned char)(v & 0xFF);
    MEM[a+1] = (unsigned char)((v >> 8) & 0xFF);
    MEM[a+2] = (unsigned char)((v >> 16) & 0xFF);
    MEM[a+3] = (unsigned char)((v >> 24) & 0xFF);
}

/* fd table: map VM fds to OS fds (0/1/2 pass through). */
static int cstr_len(long a) { int n = 0; while (MEM[a + n]) n++; return n; }

/* ---- the c4 printf subset (%d %u %x %c %s %%), byte-exact to RefVM ----- */
static void do_printf(long *sp) {
    /* The compiler always emits ADJ (n_args+1)*8 after PRTF; but from within
     * the runtime we recover n_args by scanning the format for conversions,
     * exactly as RefVM's _count_format_args does (fmt is the deepest slot). */
    /* First find fmt ptr: it is the deepest pushed slot.  We derive the pushed
     * depth from the trailing ADJ the caller emitted; c4pack passes it via the
     * global g_prtf_pushed set by the dispatcher. */
    extern int g_prtf_pushed;
    int n_pushed = g_prtf_pushed;
    int n_args = n_pushed - 1; if (n_args < 0) n_args = 0;
    long fmt_ptr = (long)mem_lw((long)(*sp) + (long)n_args * SLOT);
    long fp = fmt_ptr;
    int ai = 0;
    unsigned char c;
    while ((c = MEM[fp++]) != 0) {
        if (c != '%') { putchar(c); continue; }
        c = MEM[fp++];
        if (c == 0) { putchar('%'); break; }
        if (c == '%') { putchar('%'); continue; }
        {
            /* arg ai (call order) sits at depth (n_args-1-ai) above fmt base */
            long slot = (long)(*sp) + (long)(n_args - 1 - ai) * SLOT;
            unsigned arg = mem_lw(slot);
            if (c == 'd') {
                int sv = (int)arg;               /* signed 32 */
                printf("%d", sv); ai++;
            } else if (c == 'u') {
                printf("%u", arg); ai++;
            } else if (c == 'x') {
                printf("%x", arg); ai++;
            } else if (c == 'c') {
                putchar((int)(arg & 0xFF)); ai++;
            } else if (c == 's') {
                long s = (long)arg;
                unsigned char ch;
                while ((ch = MEM[s++]) != 0) putchar(ch);
                ai++;
            } else {
                putchar('%'); putchar(c);         /* unknown -> verbatim */
            }
        }
    }
}
int g_prtf_pushed = 1;

/* Read the trailing ADJ immediate that follows a syscall to get pushed slots. */
extern const long PROG_CODE[][2];
extern const int  PROG_CODE_LEN;
extern const unsigned char PROG_DATA[];
extern const int  PROG_DATA_LEN;

static int pushed_slots(long i) {
    if (i + 1 < PROG_CODE_LEN && PROG_CODE[i+1][0] == ADJ)
        return (int)(PROG_CODE[i+1][1] / SLOT);
    return 1;
}

int main(int argc, char **argv) {
    long sp, bp, pc;
    unsigned ax = 0;
    int i;

    /* load data segment at DATA_BASE */
    for (i = 0; i < PROG_DATA_LEN; i++) MEM[DATA_BASE + i] = PROG_DATA[i];

    sp = STACK_TOP;
    bp = STACK_TOP;
    pc = 0;

    /* --- argv setup: place argv strings on the heap, build the argv[] vector,
     * then push argc + argv-pointer as the two params main() reads at BP+16,
     * BP+24.  main is entered via the startup stub `JSR main; EXIT`, so pushing
     * the two params here (before JSR runs) makes them frame params. */
    {
        long argv_vec, str_area;
        int a;
        str_area = heap_ptr;
        /* copy each argv string into memory */
        long ptrs[256];
        int nargs = argc; if (nargs > 255) nargs = 255;
        for (a = 0; a < nargs; a++) {
            long dst = heap_ptr;
            const char *s = argv[a];
            int L = 0;
            while (s[L]) { MEM[heap_ptr++] = (unsigned char)s[L]; L++; }
            MEM[heap_ptr++] = 0;
            ptrs[a] = dst;
        }
        /* align heap */
        if (heap_ptr & 7) heap_ptr += 8 - (heap_ptr & 7);
        /* build argv[] vector of 32-bit pointers */
        argv_vec = heap_ptr;
        for (a = 0; a < nargs; a++) { mem_sw(heap_ptr, (unsigned)ptrs[a]); heap_ptr += SLOT; }
        (void)str_area;
        /* push params: main(argc, argv).  The compiler pushes call args
         * left-to-right (PSH argc; PSH argv), so the FIRST arg (argc) is the
         * DEEPEST slot, read at BP+24, and argv is read at BP+16.  Mirror that:
         * push argc first (deeper), then argv (top). */
        sp -= SLOT; mem_sw(sp, (unsigned)nargs);      /* param0: argc (deep, BP+24) */
        sp -= SLOT; mem_sw(sp, (unsigned)argv_vec);   /* param1: argv (top,  BP+16) */
    }

    /* --- execute --- */
    for (;;) {
        long op, imm;
        if (pc < 0 || pc >= PROG_CODE_LEN) break;
        op  = PROG_CODE[pc][0];
        imm = PROG_CODE[pc][1];
        pc++;
        switch (op) {
        case IMM: ax = (unsigned)imm; break;
        case LEA: ax = (unsigned)(bp + imm); break;
        case PSH: sp -= SLOT; mem_sw(sp, ax); break;
        case ADD: ax = mem_lw(sp) + ax; sp += SLOT; break;
        case SUB: ax = mem_lw(sp) - ax; sp += SLOT; break;
        case MUL: ax = mem_lw(sp) * ax; sp += SLOT; break;
        case DIV: { unsigned v = mem_lw(sp); ax = ax ? (unsigned)((int)v / (int)ax) : 0; sp += SLOT; } break;
        case MOD: { unsigned v = mem_lw(sp); ax = ax ? (unsigned)((int)v % (int)ax) : 0; sp += SLOT; } break;
        case OR:  ax = mem_lw(sp) | ax; sp += SLOT; break;
        case XOR: ax = mem_lw(sp) ^ ax; sp += SLOT; break;
        case AND: ax = mem_lw(sp) & ax; sp += SLOT; break;
        case SHL: ax = mem_lw(sp) << ax; sp += SLOT; break;
        case SHR: { int sv = (int)mem_lw(sp); ax = (unsigned)(sv >> ax); sp += SLOT; } break;
        case EQ:  ax = (mem_lw(sp) == ax) ? 1 : 0; sp += SLOT; break;
        case NE:  ax = (mem_lw(sp) != ax) ? 1 : 0; sp += SLOT; break;
        case LT:  ax = ((int)mem_lw(sp) <  (int)ax) ? 1 : 0; sp += SLOT; break;
        case GT:  ax = ((int)mem_lw(sp) >  (int)ax) ? 1 : 0; sp += SLOT; break;
        case LE:  ax = ((int)mem_lw(sp) <= (int)ax) ? 1 : 0; sp += SLOT; break;
        case GE:  ax = ((int)mem_lw(sp) >= (int)ax) ? 1 : 0; sp += SLOT; break;
        case LI:  ax = mem_lw((long)ax); break;
        case LC:  { unsigned char b = MEM[(long)ax]; ax = (b & 0x80) ? (unsigned)(b - 0x100) : b; } break;
        case SI:  { long a = (long)mem_lw(sp); sp += SLOT; mem_sw(a, ax); } break;
        case SC:  { long a = (long)mem_lw(sp); sp += SLOT; MEM[a] = (unsigned char)(ax & 0xFF); } break;
        case JMP: pc = imm; break;
        case BZ:  if (ax == 0) pc = imm; break;
        case BNZ: if (ax != 0) pc = imm; break;
        case JSR: sp -= SLOT; mem_sw(sp, (unsigned)pc); pc = imm; break;
        case ENT: sp -= SLOT; mem_sw(sp, (unsigned)bp); bp = sp; sp -= imm; break;
        case ADJ: sp += imm; break;
        case LEV: sp = bp; bp = (long)mem_lw(sp); sp += SLOT; pc = (long)mem_lw(sp); sp += SLOT; break;
        case NOP: break;
        case EXIT: goto done;

        case OPEN: {
            long name_ptr = (long)mem_lw(sp + 1 * SLOT);       /* under flags */
            long flags    = (long)mem_lw(sp + 0 * SLOT);
            char path[4096]; int L = cstr_len(name_ptr);
            if (L > 4095) L = 4095;
            memcpy(path, &MEM[name_ptr], L); path[L] = 0;
            ax = (unsigned)open(path, (int)flags);
            break;
        }
        case READ: {
            long n   = (long)mem_lw(sp + 0 * SLOT);
            long buf = (long)mem_lw(sp + 1 * SLOT);
            long fd  = (long)mem_lw(sp + 2 * SLOT);
            ssize_t got;
            static unsigned char tmp[65536];
            long want = n; if (want > (long)sizeof(tmp)) want = sizeof(tmp);
            got = read((int)fd, tmp, (size_t)want);
            if (got < 0) got = 0;
            { long k; for (k = 0; k < got; k++) MEM[buf + k] = tmp[k]; }
            ax = (unsigned)got;
            break;
        }
        case CLOS: {
            long fd = (long)mem_lw(sp + 0 * SLOT);
            ax = (unsigned)close((int)fd);
            break;
        }
        case PRTF: {
            g_prtf_pushed = pushed_slots(pc - 1);
            do_printf(&sp);
            break;
        }
        case MALC: {
            long n = (long)mem_lw(sp);   /* malloc arg is on stack top */
            ax = (unsigned)heap_ptr;
            heap_ptr += n;
            if (heap_ptr & 7) heap_ptr += 8 - (heap_ptr & 7);
            break;
        }
        case FREE: break;
        case MSET: {
            long v = (long)mem_lw(sp + 0 * SLOT);
            long c = (long)mem_lw(sp + 1 * SLOT);
            long p = (long)mem_lw(sp + 2 * SLOT);
            long k; for (k = 0; k < v; k++) MEM[p + k] = (unsigned char)(c & 0xFF);
            ax = (unsigned)p;
            break;
        }
        case MCMP: {
            long n = (long)mem_lw(sp + 0 * SLOT);
            long b = (long)mem_lw(sp + 1 * SLOT);
            long a = (long)mem_lw(sp + 2 * SLOT);
            int r = 0; long k;
            for (k = 0; k < n; k++) { if (MEM[a+k] != MEM[b+k]) { r = (int)MEM[a+k] - (int)MEM[b+k]; break; } }
            ax = (unsigned)r;
            break;
        }
        case GETCHAR: {
            int ch = getchar();
            ax = (ch == EOF) ? (unsigned)(-1) : (unsigned)(ch & 0xFF);
            break;
        }
        case PUTCHAR: {
            long v = (long)mem_lw(sp);   /* arg on stack top */
            putchar((int)(v & 0xFF));
            ax = (unsigned)v;
            break;
        }
        default:
            fprintf(stderr, "c4: unknown op %ld at pc %ld\n", op, pc - 1);
            goto done;
        }
    }
done:
    fflush(stdout);
    return (int)(ax & 0xFF);
}
