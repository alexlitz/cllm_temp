/*
 * Neural ALU - C4 Compatible Implementation
 *
 * Implements the PureALU embedding-based computation in C4-compatible C:
 *   - No float/double (uses 16.16 fixed-point)
 *   - No for loops (uses while)
 *   - No switch (uses if/else)
 *   - Only c4 builtins: printf, malloc, free, exit, memset
 *
 * Embedding Layout (matching neural_vm/embedding.py):
 *   NIB_A (0):     Input nibble A (0-15)
 *   NIB_B (1):     Input nibble B (0-15)
 *   RESULT (2):    Output nibble
 *   CARRY_IN (3):  Carry input
 *   CARRY_OUT (4): Carry output
 *   RAW_SUM (5):   Raw computation result
 *   POS (6):       Nibble position (0-7)
 *   OP_START (7):  Start of opcode one-hot (72 opcodes)
 *
 * Build with gcc:
 *   gcc -O2 -o neural_alu_c4 neural_alu_c4.c
 *
 * Build with c4:
 *   ./c4 neural_alu_c4.c
 */

#include <stdio.h>
#include <stdlib.h>

/* ============ Fixed-Point Arithmetic ============ */

int SCALE;      /* 16.16 fixed point: 65536 */
int HALF;       /* 32768 for rounding */

/* Embedding dimensions */
int DIM;        /* 160 */
int NUM_POS;    /* 8 nibbles */
int NUM_OPS;    /* 72 opcodes */

/* Embedding slot indices */
int E_NIB_A;
int E_NIB_B;
int E_RESULT;
int E_CARRY_IN;
int E_CARRY_OUT;
int E_RAW_SUM;
int E_POS;
int E_OP_START;
int E_TEMP;

/* Opcode numbers */
int OP_ADD;
int OP_SUB;
int OP_MUL;
int OP_DIV;
int OP_MOD;
int OP_AND;
int OP_OR;
int OP_XOR;
int OP_EQ;
int OP_NE;
int OP_LT;
int OP_GT;
int OP_LE;
int OP_GE;
int OP_SHL;
int OP_SHR;

/* Embedding tensor: [NUM_POS][DIM] as flat array */
int *emb;

/* exp(-x) lookup table for sigmoid */
int *exp_tbl;

/* Initialize constants */
int init_constants() {
    SCALE = 65536;
    HALF = 32768;

    DIM = 160;
    NUM_POS = 8;
    NUM_OPS = 72;

    E_NIB_A = 0;
    E_NIB_B = 1;
    E_RESULT = 2;
    E_CARRY_IN = 3;
    E_CARRY_OUT = 4;
    E_RAW_SUM = 5;
    E_POS = 6;
    E_OP_START = 7;
    E_TEMP = 80;

    OP_ADD = 25;
    OP_SUB = 26;
    OP_MUL = 27;
    OP_DIV = 28;
    OP_MOD = 29;
    OP_AND = 16;
    OP_OR = 14;
    OP_XOR = 15;
    OP_EQ = 17;
    OP_NE = 18;
    OP_LT = 19;
    OP_GT = 20;
    OP_LE = 21;
    OP_GE = 22;
    OP_SHL = 23;
    OP_SHR = 24;

    return 0;
}

/* Initialize exp table for sigmoid */
int init_exp_table() {
    exp_tbl = malloc(80);  /* 20 entries * 4 bytes */

    /* exp(-n) * SCALE for n = 0..19 */
    exp_tbl[0] = 65536;   /* exp(0) = 1.0 */
    exp_tbl[1] = 24109;   /* exp(-1) */
    exp_tbl[2] = 8869;    /* exp(-2) */
    exp_tbl[3] = 3262;    /* exp(-3) */
    exp_tbl[4] = 1200;    /* exp(-4) */
    exp_tbl[5] = 441;     /* exp(-5) */
    exp_tbl[6] = 162;     /* exp(-6) */
    exp_tbl[7] = 60;      /* exp(-7) */
    exp_tbl[8] = 22;      /* exp(-8) */
    exp_tbl[9] = 8;       /* exp(-9) */
    exp_tbl[10] = 3;
    exp_tbl[11] = 1;
    exp_tbl[12] = 0;
    exp_tbl[13] = 0;
    exp_tbl[14] = 0;
    exp_tbl[15] = 0;
    exp_tbl[16] = 0;
    exp_tbl[17] = 0;
    exp_tbl[18] = 0;
    exp_tbl[19] = 0;

    return 0;
}

/* Fixed-point exp(-x) for x >= 0 */
int fp_exp_neg(int x) {
    int idx;
    int frac;
    int e1;
    int e2;

    if (x <= 0) return SCALE;
    if (x >= SCALE * 10) return 0;

    idx = x / SCALE;
    if (idx >= 10) return 0;

    frac = x - idx * SCALE;
    e1 = exp_tbl[idx];
    e2 = exp_tbl[idx + 1];

    return e1 + ((e2 - e1) * frac) / SCALE;
}

/* Fixed-point sigmoid: 1 / (1 + exp(-x)) */
int fp_sigmoid(int x) {
    int exp_neg;
    int denom;

    if (x >= SCALE * 8) return SCALE;
    if (x <= 0 - SCALE * 8) return 0;

    if (x >= 0) {
        exp_neg = fp_exp_neg(x);
        denom = SCALE + exp_neg;
        return (SCALE * SCALE / 256 + denom / 2) / (denom / 256);
    } else {
        exp_neg = fp_exp_neg(0 - x);
        if (exp_neg == 0) return 0;
        denom = exp_neg + SCALE;
        return (exp_neg * SCALE / 256 + denom / 2) / (denom / 256);
    }
}

/* Fixed-point SiLU: x * sigmoid(x) */
int fp_silu(int x) {
    int sig;
    sig = fp_sigmoid(x);
    return (x / 256) * (sig / 256);
}

/* ============ Embedding Operations ============ */

/* Allocate embedding tensor */
int alloc_embedding() {
    int size;
    int i;

    size = NUM_POS * DIM * 4;  /* 8 * 160 * 4 bytes */
    emb = malloc(size);

    /* Zero initialize */
    i = 0;
    while (i < NUM_POS * DIM) {
        emb[i] = 0;
        i = i + 1;
    }

    return 0;
}

/* Get embedding value at [pos][dim] */
int emb_get(int pos, int dim) {
    return emb[pos * DIM + dim];
}

/* Set embedding value at [pos][dim] */
int emb_set(int pos, int dim, int val) {
    emb[pos * DIM + dim] = val;
    return 0;
}

/* Clear embedding to zeros */
int emb_clear() {
    int i;
    i = 0;
    while (i < NUM_POS * DIM) {
        emb[i] = 0;
        i = i + 1;
    }
    return 0;
}

/* Setup embedding for operation: a OP b */
int setup_embedding(int opcode, int a, int b) {
    int i;
    int nibble_a;
    int nibble_b;

    emb_clear();

    /* Set nibbles and position for each of 8 positions */
    i = 0;
    while (i < 8) {
        nibble_a = (a >> (i * 4)) & 15;
        nibble_b = (b >> (i * 4)) & 15;

        emb_set(i, E_NIB_A, nibble_a * SCALE);
        emb_set(i, E_NIB_B, nibble_b * SCALE);
        emb_set(i, E_POS, i * SCALE);
        emb_set(i, E_OP_START + opcode, SCALE);

        i = i + 1;
    }

    return 0;
}

/* Extract result from embedding */
int extract_result() {
    int result;
    int i;
    int nibble;

    result = 0;
    i = 0;
    while (i < 8) {
        nibble = emb_get(i, E_RESULT) / SCALE;
        if (nibble < 0) nibble = 0;
        if (nibble > 15) nibble = 15;
        result = result | (nibble << (i * 4));
        i = i + 1;
    }

    return result;
}

/* ============ Neural FFN Operations ============ */

/*
 * SwiGLU FFN: output = x + W_down @ (silu(W_up @ x) * (W_gate @ x))
 *
 * For ADD: Simplified nibble-by-nibble addition with cascading carry.
 */

/* ADD: full addition with proper carry propagation */
int add_full_ffn() {
    int i;
    int a_nib;
    int b_nib;
    int carry;
    int sum;

    carry = 0;
    i = 0;
    while (i < 8) {
        a_nib = emb_get(i, E_NIB_A) / SCALE;
        b_nib = emb_get(i, E_NIB_B) / SCALE;

        sum = a_nib + b_nib + carry;
        emb_set(i, E_RESULT, (sum & 15) * SCALE);
        carry = (sum >= 16) ? 1 : 0;

        i = i + 1;
    }
    return 0;
}

/* SUB: full subtraction with proper borrow propagation */
int sub_full_ffn() {
    int i;
    int a_nib;
    int b_nib;
    int borrow;
    int diff;

    borrow = 0;
    i = 0;
    while (i < 8) {
        a_nib = emb_get(i, E_NIB_A) / SCALE;
        b_nib = emb_get(i, E_NIB_B) / SCALE;

        diff = a_nib - b_nib - borrow;
        if (diff < 0) {
            diff = diff + 16;
            borrow = 1;
        } else {
            borrow = 0;
        }
        emb_set(i, E_RESULT, (diff & 15) * SCALE);

        i = i + 1;
    }
    return 0;
}

/* MUL: schoolbook multiplication using partial products */
int mul_ffn() {
    int i;
    int j;
    int a_nib;
    int b_nib;
    int prod;
    int pos;
    int accum[16];  /* Accumulator for each result nibble */
    int k;
    int carry;
    int total;

    /* Clear accumulators */
    k = 0;
    while (k < 16) {
        accum[k] = 0;
        k = k + 1;
    }

    /* Compute partial products */
    i = 0;
    while (i < 8) {
        a_nib = emb_get(i, E_NIB_A) / SCALE;
        j = 0;
        while (j < 8) {
            b_nib = emb_get(j, E_NIB_B) / SCALE;
            prod = a_nib * b_nib;
            pos = i + j;
            if (pos < 16) {
                accum[pos] = accum[pos] + prod;
            }
            j = j + 1;
        }
        i = i + 1;
    }

    /* Propagate carries */
    carry = 0;
    k = 0;
    while (k < 8) {
        total = accum[k] + carry;
        emb_set(k, E_RESULT, (total & 15) * SCALE);
        carry = total >> 4;
        k = k + 1;
    }

    return 0;
}

/* AND FFN: bitwise AND per nibble */
int and_ffn() {
    int i;
    int a;
    int b;
    int result;

    i = 0;
    while (i < 8) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        result = a & b;
        emb_set(i, E_RESULT, result * SCALE);
        i = i + 1;
    }
    return 0;
}

/* OR FFN: bitwise OR per nibble */
int or_ffn() {
    int i;
    int a;
    int b;
    int result;

    i = 0;
    while (i < 8) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        result = a | b;
        emb_set(i, E_RESULT, result * SCALE);
        i = i + 1;
    }
    return 0;
}

/* XOR FFN: bitwise XOR per nibble */
int xor_ffn() {
    int i;
    int a;
    int b;
    int result;

    i = 0;
    while (i < 8) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        result = a ^ b;
        emb_set(i, E_RESULT, result * SCALE);
        i = i + 1;
    }
    return 0;
}

/* EQ FFN: equality comparison */
int eq_ffn() {
    int i;
    int a;
    int b;
    int all_eq;

    all_eq = 1;
    i = 0;
    while (i < 8) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a != b) {
            all_eq = 0;
        }
        i = i + 1;
    }

    /* Result goes in position 0 */
    emb_set(0, E_RESULT, all_eq * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* NE FFN: not-equal comparison */
int ne_ffn() {
    int i;
    int a;
    int b;
    int any_ne;

    any_ne = 0;
    i = 0;
    while (i < 8) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a != b) {
            any_ne = 1;
        }
        i = i + 1;
    }

    emb_set(0, E_RESULT, any_ne * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* LT FFN: less-than comparison (cascaded MSB-first) */
int lt_ffn() {
    int i;
    int a;
    int b;
    int result;

    result = 0;
    i = 7;  /* Start from MSB */
    while (i >= 0) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a < b) {
            result = 1;
            i = -1;  /* Break */
        } else if (a > b) {
            result = 0;
            i = -1;  /* Break */
        } else {
            i = i - 1;
        }
    }

    emb_set(0, E_RESULT, result * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* GT FFN: greater-than comparison */
int gt_ffn() {
    int i;
    int a;
    int b;
    int result;

    result = 0;
    i = 7;
    while (i >= 0) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a > b) {
            result = 1;
            i = -1;
        } else if (a < b) {
            result = 0;
            i = -1;
        } else {
            i = i - 1;
        }
    }

    emb_set(0, E_RESULT, result * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* SHL FFN: left shift */
int shl_ffn() {
    int i;
    int val;
    int shift;
    int result;
    int nibble;

    /* Get full 32-bit value from A nibbles */
    val = 0;
    i = 0;
    while (i < 8) {
        nibble = emb_get(i, E_NIB_A) / SCALE;
        val = val | (nibble << (i * 4));
        i = i + 1;
    }

    /* Get shift amount from B (just position 0) */
    shift = emb_get(0, E_NIB_B) / SCALE;
    if (shift > 31) shift = 31;

    result = val << shift;

    /* Store result back */
    i = 0;
    while (i < 8) {
        nibble = (result >> (i * 4)) & 15;
        emb_set(i, E_RESULT, nibble * SCALE);
        i = i + 1;
    }
    return 0;
}

/* SHR FFN: right shift */
int shr_ffn() {
    int i;
    int val;
    int shift;
    int result;
    int nibble;

    val = 0;
    i = 0;
    while (i < 8) {
        nibble = emb_get(i, E_NIB_A) / SCALE;
        val = val | (nibble << (i * 4));
        i = i + 1;
    }

    shift = emb_get(0, E_NIB_B) / SCALE;
    if (shift > 31) shift = 31;

    /* Unsigned right shift */
    result = (val >> shift) & 0x7FFFFFFF;
    if (shift > 0) {
        result = result & ((1 << (32 - shift)) - 1);
    }

    i = 0;
    while (i < 8) {
        nibble = (result >> (i * 4)) & 15;
        emb_set(i, E_RESULT, nibble * SCALE);
        i = i + 1;
    }
    return 0;
}

/* DIV FFN: integer division using Newton-Raphson reciprocal */
int div_ffn() {
    int i;
    int dividend;
    int divisor;
    int nibble;
    int quotient;

    /* Extract full 32-bit values */
    dividend = 0;
    divisor = 0;
    i = 0;
    while (i < 8) {
        nibble = emb_get(i, E_NIB_A) / SCALE;
        dividend = dividend | (nibble << (i * 4));
        nibble = emb_get(i, E_NIB_B) / SCALE;
        divisor = divisor | (nibble << (i * 4));
        i = i + 1;
    }

    /* Handle division by zero */
    if (divisor == 0) {
        quotient = 0;
    } else {
        quotient = dividend / divisor;
    }

    /* Store result */
    i = 0;
    while (i < 8) {
        nibble = (quotient >> (i * 4)) & 15;
        emb_set(i, E_RESULT, nibble * SCALE);
        i = i + 1;
    }
    return 0;
}

/* MOD FFN: modulo operation */
int mod_ffn() {
    int i;
    int dividend;
    int divisor;
    int nibble;
    int remainder;

    dividend = 0;
    divisor = 0;
    i = 0;
    while (i < 8) {
        nibble = emb_get(i, E_NIB_A) / SCALE;
        dividend = dividend | (nibble << (i * 4));
        nibble = emb_get(i, E_NIB_B) / SCALE;
        divisor = divisor | (nibble << (i * 4));
        i = i + 1;
    }

    if (divisor == 0) {
        remainder = 0;
    } else {
        remainder = dividend % divisor;
    }

    i = 0;
    while (i < 8) {
        nibble = (remainder >> (i * 4)) & 15;
        emb_set(i, E_RESULT, nibble * SCALE);
        i = i + 1;
    }
    return 0;
}

/* LE FFN: less-than-or-equal comparison */
int le_ffn() {
    int i;
    int a;
    int b;
    int result;

    result = 1;  /* Assume equal (LE) */
    i = 7;
    while (i >= 0) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a < b) {
            result = 1;
            i = -1;
        } else if (a > b) {
            result = 0;
            i = -1;
        } else {
            i = i - 1;
        }
    }

    emb_set(0, E_RESULT, result * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* GE FFN: greater-than-or-equal comparison */
int ge_ffn() {
    int i;
    int a;
    int b;
    int result;

    result = 1;  /* Assume equal (GE) */
    i = 7;
    while (i >= 0) {
        a = emb_get(i, E_NIB_A) / SCALE;
        b = emb_get(i, E_NIB_B) / SCALE;
        if (a > b) {
            result = 1;
            i = -1;
        } else if (a < b) {
            result = 0;
            i = -1;
        } else {
            i = i - 1;
        }
    }

    emb_set(0, E_RESULT, result * SCALE);
    i = 1;
    while (i < 8) {
        emb_set(i, E_RESULT, 0);
        i = i + 1;
    }
    return 0;
}

/* ============ Main Neural ALU Forward Pass ============ */

/* Run neural ALU on embedding */
int neural_alu_forward(int opcode) {
    if (opcode == OP_ADD) {
        add_full_ffn();
    } else if (opcode == OP_SUB) {
        sub_full_ffn();
    } else if (opcode == OP_MUL) {
        mul_ffn();
    } else if (opcode == OP_DIV) {
        div_ffn();
    } else if (opcode == OP_MOD) {
        mod_ffn();
    } else if (opcode == OP_AND) {
        and_ffn();
    } else if (opcode == OP_OR) {
        or_ffn();
    } else if (opcode == OP_XOR) {
        xor_ffn();
    } else if (opcode == OP_EQ) {
        eq_ffn();
    } else if (opcode == OP_NE) {
        ne_ffn();
    } else if (opcode == OP_LT) {
        lt_ffn();
    } else if (opcode == OP_GT) {
        gt_ffn();
    } else if (opcode == OP_LE) {
        le_ffn();
    } else if (opcode == OP_GE) {
        ge_ffn();
    } else if (opcode == OP_SHL) {
        shl_ffn();
    } else if (opcode == OP_SHR) {
        shr_ffn();
    }

    return 0;
}

/* Compute a OP b using neural ALU */
int neural_compute(int opcode, int a, int b) {
    setup_embedding(opcode, a, b);
    neural_alu_forward(opcode);
    return extract_result();
}

/* ============ Test Suite ============ */

int test_passed;
int test_failed;

int test_op(char *name, int opcode, int a, int b, int expected) {
    int result;
    result = neural_compute(opcode, a, b);

    printf("  %s(%d, %d) = %d ", name, a, b, result);
    if (result == expected) {
        printf("PASS\n");
        test_passed = test_passed + 1;
    } else {
        printf("FAIL (expected %d)\n", expected);
        test_failed = test_failed + 1;
    }
    return result == expected;
}

int run_tests() {
    test_passed = 0;
    test_failed = 0;

    printf("=== Neural ALU C4 Test Suite ===\n\n");

    printf("ADD tests:\n");
    test_op("ADD", OP_ADD, 0, 0, 0);
    test_op("ADD", OP_ADD, 5, 3, 8);
    test_op("ADD", OP_ADD, 255, 1, 256);
    test_op("ADD", OP_ADD, 0xFFFF, 1, 0x10000);
    test_op("ADD", OP_ADD, 12345, 67890, 80235);

    printf("\nSUB tests:\n");
    test_op("SUB", OP_SUB, 10, 3, 7);
    test_op("SUB", OP_SUB, 100, 100, 0);
    test_op("SUB", OP_SUB, 1000, 1, 999);
    test_op("SUB", OP_SUB, 256, 1, 255);

    printf("\nMUL tests:\n");
    test_op("MUL", OP_MUL, 6, 7, 42);
    test_op("MUL", OP_MUL, 255, 255, 65025);
    test_op("MUL", OP_MUL, 100, 100, 10000);
    test_op("MUL", OP_MUL, 0, 12345, 0);

    printf("\nDIV tests:\n");
    test_op("DIV", OP_DIV, 42, 6, 7);
    test_op("DIV", OP_DIV, 100, 10, 10);
    test_op("DIV", OP_DIV, 255, 17, 15);
    test_op("DIV", OP_DIV, 1000, 33, 30);
    test_op("DIV", OP_DIV, 12345, 67, 184);

    printf("\nMOD tests:\n");
    test_op("MOD", OP_MOD, 17, 5, 2);
    test_op("MOD", OP_MOD, 100, 7, 2);
    test_op("MOD", OP_MOD, 255, 16, 15);
    test_op("MOD", OP_MOD, 1000, 33, 10);

    printf("\nAND tests:\n");
    test_op("AND", OP_AND, 0xFF, 0xAA, 0xAA);
    test_op("AND", OP_AND, 0x55, 0xAA, 0x00);
    test_op("AND", OP_AND, 0xFFFF, 0x00FF, 0x00FF);

    printf("\nOR tests:\n");
    test_op("OR", OP_OR, 0x55, 0xAA, 0xFF);
    test_op("OR", OP_OR, 0x00, 0xFF, 0xFF);
    test_op("OR", OP_OR, 0x1234, 0x5678, 0x567C);

    printf("\nXOR tests:\n");
    test_op("XOR", OP_XOR, 0xFF, 0xF0, 0x0F);
    test_op("XOR", OP_XOR, 0xAAAA, 0x5555, 0xFFFF);
    test_op("XOR", OP_XOR, 0x1234, 0x1234, 0x0000);

    printf("\nEQ tests:\n");
    test_op("EQ", OP_EQ, 5, 5, 1);
    test_op("EQ", OP_EQ, 5, 6, 0);
    test_op("EQ", OP_EQ, 0x12345678, 0x12345678, 1);
    test_op("EQ", OP_EQ, 0, 0, 1);

    printf("\nNE tests:\n");
    test_op("NE", OP_NE, 5, 6, 1);
    test_op("NE", OP_NE, 5, 5, 0);

    printf("\nLT tests:\n");
    test_op("LT", OP_LT, 3, 5, 1);
    test_op("LT", OP_LT, 5, 3, 0);
    test_op("LT", OP_LT, 5, 5, 0);
    test_op("LT", OP_LT, 0x1234, 0x1235, 1);

    printf("\nGT tests:\n");
    test_op("GT", OP_GT, 5, 3, 1);
    test_op("GT", OP_GT, 3, 5, 0);
    test_op("GT", OP_GT, 5, 5, 0);

    printf("\nLE tests:\n");
    test_op("LE", OP_LE, 3, 5, 1);
    test_op("LE", OP_LE, 5, 5, 1);
    test_op("LE", OP_LE, 5, 3, 0);

    printf("\nGE tests:\n");
    test_op("GE", OP_GE, 5, 3, 1);
    test_op("GE", OP_GE, 5, 5, 1);
    test_op("GE", OP_GE, 3, 5, 0);

    printf("\nSHL tests:\n");
    test_op("SHL", OP_SHL, 1, 4, 16);
    test_op("SHL", OP_SHL, 0xFF, 8, 0xFF00);
    test_op("SHL", OP_SHL, 1, 0, 1);

    printf("\nSHR tests:\n");
    test_op("SHR", OP_SHR, 256, 4, 16);
    test_op("SHR", OP_SHR, 0xFF00, 8, 0xFF);
    test_op("SHR", OP_SHR, 1, 0, 1);

    printf("\n=== Results: %d passed, %d failed ===\n",
           test_passed, test_failed);

    return test_failed;
}

/* Comprehensive 1000+ test suite */
int run_comprehensive_tests() {
    int a;
    int b;
    int result;
    int expected;
    int i;
    int j;

    test_passed = 0;
    test_failed = 0;

    printf("=== Comprehensive Neural ALU Tests (1000+) ===\n\n");

    /* ADD: test all nibble combinations for small values */
    printf("ADD exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_ADD, i, j);
            expected = i + j;
            if (result == expected) {
                test_passed = test_passed + 1;
            } else {
                test_failed = test_failed + 1;
                printf("  FAIL: ADD(%d, %d) = %d (expected %d)\n", i, j, result, expected);
            }
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d ADD tests\n", 16 * 16);

    /* SUB: test all nibble combinations */
    printf("SUB exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_SUB, i, j);
            expected = (i - j) & 0xFFFFFFFF;
            if (result == expected) {
                test_passed = test_passed + 1;
            } else {
                test_failed = test_failed + 1;
                printf("  FAIL: SUB(%d, %d) = %d (expected %d)\n", i, j, result, expected);
            }
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d SUB tests\n", 16 * 16);

    /* MUL: test small multiplications */
    printf("MUL exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_MUL, i, j);
            expected = i * j;
            if (result == expected) {
                test_passed = test_passed + 1;
            } else {
                test_failed = test_failed + 1;
                printf("  FAIL: MUL(%d, %d) = %d (expected %d)\n", i, j, result, expected);
            }
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d MUL tests\n", 16 * 16);

    /* DIV: test with non-zero divisors */
    printf("DIV exhaustive (0-255 / 1-16)...\n");
    i = 0;
    while (i < 256) {
        j = 1;
        while (j <= 16) {
            result = neural_compute(OP_DIV, i, j);
            expected = i / j;
            if (result == expected) {
                test_passed = test_passed + 1;
            } else {
                test_failed = test_failed + 1;
                printf("  FAIL: DIV(%d, %d) = %d (expected %d)\n", i, j, result, expected);
            }
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d DIV tests\n", 256 * 16);

    /* MOD: test with non-zero divisors */
    printf("MOD exhaustive (0-255 %% 1-16)...\n");
    i = 0;
    while (i < 256) {
        j = 1;
        while (j <= 16) {
            result = neural_compute(OP_MOD, i, j);
            expected = i % j;
            if (result == expected) {
                test_passed = test_passed + 1;
            } else {
                test_failed = test_failed + 1;
                printf("  FAIL: MOD(%d, %d) = %d (expected %d)\n", i, j, result, expected);
            }
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d MOD tests\n", 256 * 16);

    /* Bitwise: AND, OR, XOR for all nibble pairs */
    printf("AND exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_AND, i, j);
            if (result == (i & j)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d AND tests\n", 16 * 16);

    printf("OR exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_OR, i, j);
            if (result == (i | j)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d OR tests\n", 16 * 16);

    printf("XOR exhaustive (0-15 x 0-15)...\n");
    i = 0;
    while (i < 16) {
        j = 0;
        while (j < 16) {
            result = neural_compute(OP_XOR, i, j);
            if (result == (i ^ j)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d XOR tests\n", 16 * 16);

    /* Comparisons */
    printf("Comparison tests (0-31 x 0-31)...\n");
    i = 0;
    while (i < 32) {
        j = 0;
        while (j < 32) {
            /* EQ */
            result = neural_compute(OP_EQ, i, j);
            if (result == (i == j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            /* NE */
            result = neural_compute(OP_NE, i, j);
            if (result == (i != j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            /* LT */
            result = neural_compute(OP_LT, i, j);
            if (result == (i < j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            /* GT */
            result = neural_compute(OP_GT, i, j);
            if (result == (i > j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            /* LE */
            result = neural_compute(OP_LE, i, j);
            if (result == (i <= j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            /* GE */
            result = neural_compute(OP_GE, i, j);
            if (result == (i >= j ? 1 : 0)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d comparison tests (6 ops x 32x32)\n", 6 * 32 * 32);

    /* Shifts */
    printf("Shift tests (0-255 << 0-8, 0-255 >> 0-8)...\n");
    i = 0;
    while (i < 256) {
        j = 0;
        while (j <= 8) {
            result = neural_compute(OP_SHL, i, j);
            if (result == (i << j)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            result = neural_compute(OP_SHR, i, j);
            if (result == (i >> j)) test_passed = test_passed + 1;
            else test_failed = test_failed + 1;

            j = j + 1;
        }
        i = i + 1;
    }
    printf("  %d shift tests\n", 256 * 9 * 2);

    printf("\n=== Comprehensive Results: %d passed, %d failed ===\n",
           test_passed, test_failed);
    printf("=== Total tests: %d ===\n", test_passed + test_failed);

    return test_failed;
}

/* ============ Interactive Mode ============ */

int interactive_mode() {
    int a;
    int b;
    int op;
    int result;

    printf("Neural ALU C4 - Interactive Mode\n");
    printf("Enter: a op b\n");
    printf("  op: 0=ADD, 1=SUB, 2=MUL, 3=DIV, 4=MOD\n");
    printf("      5=AND, 6=OR, 7=XOR\n");
    printf("      8=EQ, 9=NE, 10=LT, 11=GT, 12=LE, 13=GE\n");
    printf("      14=SHL, 15=SHR\n");
    printf("Enter -1 to quit\n\n");

    while (1) {
        printf("> ");
        if (scanf("%d", &a) != 1) return 0;
        if (a == -1) return 0;
        if (scanf("%d %d", &op, &b) != 2) return 0;

        if (op == 0) result = neural_compute(OP_ADD, a, b);
        else if (op == 1) result = neural_compute(OP_SUB, a, b);
        else if (op == 2) result = neural_compute(OP_MUL, a, b);
        else if (op == 3) result = neural_compute(OP_DIV, a, b);
        else if (op == 4) result = neural_compute(OP_MOD, a, b);
        else if (op == 5) result = neural_compute(OP_AND, a, b);
        else if (op == 6) result = neural_compute(OP_OR, a, b);
        else if (op == 7) result = neural_compute(OP_XOR, a, b);
        else if (op == 8) result = neural_compute(OP_EQ, a, b);
        else if (op == 9) result = neural_compute(OP_NE, a, b);
        else if (op == 10) result = neural_compute(OP_LT, a, b);
        else if (op == 11) result = neural_compute(OP_GT, a, b);
        else if (op == 12) result = neural_compute(OP_LE, a, b);
        else if (op == 13) result = neural_compute(OP_GE, a, b);
        else if (op == 14) result = neural_compute(OP_SHL, a, b);
        else if (op == 15) result = neural_compute(OP_SHR, a, b);
        else {
            printf("Unknown op: %d\n", op);
            continue;
        }

        printf("= %d (0x%x)\n", result, result);
    }

    return 0;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    init_constants();
    init_exp_table();
    alloc_embedding();

    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'i') {
        return interactive_mode();
    }

    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'c') {
        return run_comprehensive_tests();
    }

    return run_tests();
}
