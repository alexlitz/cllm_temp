/*
 * Simple C4 VM Runtime - Non-Neural, Full 32-bit Arithmetic
 *
 * This runtime uses standard C integer arithmetic (no neural encoding).
 * Supports full 32-bit signed integer range (-2^31 to 2^31-1).
 *
 * Expected to be concatenated with:
 *   - int program_code[][2] (bytecode)
 *   - int program_code_len
 *   - char program_data[]
 *   - int program_data_len
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Memory implemented as hash table for sparse access */
#define HASH_SIZE (1 << 18)  /* 256K entries */
#define HASH_MASK (HASH_SIZE - 1)

typedef struct {
    int key;
    int value;
    int occupied;
} HashEntry;

static HashEntry *memory_table = NULL;

static void mem_init(void) {
    if (!memory_table) {
        memory_table = (HashEntry *)calloc(HASH_SIZE, sizeof(HashEntry));
    } else {
        memset(memory_table, 0, HASH_SIZE * sizeof(HashEntry));
    }
}

static inline int mem_get(int addr) {
    unsigned int h = (unsigned int)(addr * 2654435761U) & HASH_MASK;
    while (1) {
        if (!memory_table[h].occupied) return 0;
        if (memory_table[h].key == addr) return memory_table[h].value;
        h = (h + 1) & HASH_MASK;
    }
}

static inline void mem_set(int addr, int value) {
    unsigned int h = (unsigned int)(addr * 2654435761U) & HASH_MASK;
    while (1) {
        if (!memory_table[h].occupied) {
            memory_table[h].key = addr;
            memory_table[h].value = value;
            memory_table[h].occupied = 1;
            return;
        }
        if (memory_table[h].key == addr) {
            memory_table[h].value = value;
            return;
        }
        h = (h + 1) & HASH_MASK;
    }
}

int main(void) {
    int sp, bp, ax, pc;
    int heap_ptr;
    int steps;
    int idx, op, imm, a;

    /* Initialize memory */
    mem_init();

    /* Load data segment into memory at 0x10000 */
    for (int i = 0; i < program_data_len; i++) {
        mem_set(0x10000 + i, (unsigned char)program_data[i]);
    }

    /* Initialize VM state */
    sp = 0x10000;
    bp = 0x10000;
    ax = 0;
    pc = 0;
    heap_ptr = 0x200000;
    steps = 0;

    /* Execute bytecode */
    while (steps < 1000000) {
        idx = pc / 8;
        if (idx < 0 || idx >= program_code_len) break;

        op = program_code[idx][0];
        imm = program_code[idx][1];
        pc += 8;

        switch (op) {
            case 0:  /* LEA */
                ax = bp + imm;
                break;
            case 1:  /* IMM */
                ax = imm;
                break;
            case 2:  /* JMP */
                pc = imm;
                break;
            case 3:  /* JSR */
                sp -= 8;
                mem_set(sp, pc);
                pc = imm;
                break;
            case 4:  /* BZ */
                if (ax == 0) pc = imm;
                break;
            case 5:  /* BNZ */
                if (ax != 0) pc = imm;
                break;
            case 6:  /* ENT */
                sp -= 8;
                mem_set(sp, bp);
                bp = sp;
                sp -= imm;
                break;
            case 7:  /* ADJ */
                sp += imm;
                break;
            case 8:  /* LEV */
                sp = bp;
                bp = mem_get(sp);
                sp += 8;
                pc = mem_get(sp);
                sp += 8;
                break;
            case 9:  /* LI */
                ax = mem_get(ax);
                break;
            case 10: /* LC */
                ax = mem_get(ax) & 0xFF;
                break;
            case 11: /* SI */
                a = mem_get(sp);
                sp += 8;
                mem_set(a, ax);
                break;
            case 12: /* SC */
                a = mem_get(sp);
                sp += 8;
                mem_set(a, ax & 0xFF);
                break;
            case 13: /* PSH */
                sp -= 8;
                mem_set(sp, ax);
                break;
            case 14: /* OR */
                ax = mem_get(sp) | ax;
                sp += 8;
                break;
            case 15: /* XOR */
                ax = mem_get(sp) ^ ax;
                sp += 8;
                break;
            case 16: /* AND */
                ax = mem_get(sp) & ax;
                sp += 8;
                break;
            case 17: /* EQ */
                ax = (mem_get(sp) == ax) ? 1 : 0;
                sp += 8;
                break;
            case 18: /* NE */
                ax = (mem_get(sp) != ax) ? 1 : 0;
                sp += 8;
                break;
            case 19: /* LT */
                ax = (mem_get(sp) < ax) ? 1 : 0;
                sp += 8;
                break;
            case 20: /* GT */
                ax = (mem_get(sp) > ax) ? 1 : 0;
                sp += 8;
                break;
            case 21: /* LE */
                ax = (mem_get(sp) <= ax) ? 1 : 0;
                sp += 8;
                break;
            case 22: /* GE */
                ax = (mem_get(sp) >= ax) ? 1 : 0;
                sp += 8;
                break;
            case 23: /* SHL */
                ax = mem_get(sp) << ax;
                sp += 8;
                break;
            case 24: /* SHR */
                ax = mem_get(sp) >> ax;
                sp += 8;
                break;
            case 25: /* ADD */
                ax = mem_get(sp) + ax;
                sp += 8;
                break;
            case 26: /* SUB */
                ax = mem_get(sp) - ax;
                sp += 8;
                break;
            case 27: /* MUL */
                ax = mem_get(sp) * ax;
                sp += 8;
                break;
            case 28: /* DIV */
                if (ax != 0) {
                    ax = mem_get(sp) / ax;
                } else {
                    ax = 0;
                }
                sp += 8;
                break;
            case 29: /* MOD */
                if (ax != 0) {
                    ax = mem_get(sp) % ax;
                } else {
                    ax = 0;
                }
                sp += 8;
                break;
            case 34: /* MALC (malloc) */
                a = mem_get(sp);
                ax = heap_ptr;
                heap_ptr += a;
                if (heap_ptr & 7) heap_ptr += 8 - (heap_ptr & 7);
                break;
            case 35: /* FREE */
                break;
            case 38: /* EXIT */
                goto done;
            case 65: /* PUTCHAR */
                putchar(mem_get(sp) & 0xFF);
                ax = mem_get(sp);
                break;
            default:
                break;
        }

        steps++;
    }

done:
    free(memory_table);
    return ax & 0xFF;  /* Return exit code */
}
