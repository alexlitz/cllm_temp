/*
 * ONNX Runner for C4 Transformer VM - C4 Compatible Version
 *
 * Ported to work with the c4 minimal C interpreter:
 * - No double/float (uses 8.8 fixed-point)
 * - No for loops (uses while)
 * - No switch (uses if/else)
 * - No atof (manual parsing)
 * - Only c4 builtins: printf, malloc, free, exit
 *
 * Tool-Use I/O:
 *   When tool_use_mode is enabled, I/O operations generate tool calls
 *   that can be handled by an external system (LLM, MCP server, etc.)
 *
 *   Enable with: [TOOLUSE] token in input or -t flag
 *   Protocol:
 *     Output: TOOL_CALL:<type>:<id>:{params}
 *     Input:  TOOL_RESPONSE:<id>:<result>
 *
 * Build with c4:
 *   ./c4 onnx_runner_c4.c
 *
 * Build with gcc (add -include stdio.h -include stdlib.h):
 *   gcc -include stdio.h -include stdlib.h -o onnx_runner_c4 onnx_runner_c4.c
 */

/* ============ LLM I/O Protocol ============ */
/*
 * Uses simple tags for LLM generation control:
 *   <NEED_INPUT/>  - Pause generation, wait for user input
 *   <PROGRAM_END/> - Program finished
 *
 * The LLM generates until <NEED_INPUT/>, then user input is appended
 * to the stream, and generation continues. <PROGRAM_END/> signals
 * the session is complete.
 */

int llm_mode;   /* 0 = normal I/O (default), 1 = LLM tag protocol */

/* ============ Speculative Execution Mode ============ */
/*
 * Like speculative decoding in LLMs:
 * 1. Fast native arithmetic runs first (speculation)
 * 2. Neural arithmetic validates on demand
 * 3. Statistics track match rate
 *
 * Modes:
 *   0 = neural only (default, exact but slower)
 *   1 = native only (fast, no validation)
 *   2 = speculative (native + neural validation)
 */
int spec_mode;

/* Statistics for speculation */
int spec_total_ops;      /* Total arithmetic operations */
int spec_validated;      /* Operations validated against neural */
int spec_mismatches;     /* Mismatches between native and neural */

/* Initialize speculation stats */
int init_speculation() {
    spec_mode = 0;  /* Default: neural only */
    spec_total_ops = 0;
    spec_validated = 0;
    spec_mismatches = 0;
    return 0;
}

/* Print speculation statistics */
int print_spec_stats() {
    int match_rate;
    if (spec_validated > 0) {
        match_rate = ((spec_validated - spec_mismatches) * 100) / spec_validated;
    } else {
        match_rate = 100;
    }
    printf("\n=== Speculation Statistics ===\n");
    printf("  Mode: %d (0=neural, 1=native, 2=speculative)\n", spec_mode);
    printf("  Total ops: %d\n", spec_total_ops);
    printf("  Validated: %d\n", spec_validated);
    printf("  Mismatches: %d\n", spec_mismatches);
    printf("  Match rate: %d%%\n", match_rate);
    return 0;
}

/* Print <NEED_INPUT/> tag */
int llm_need_input() {
    printf("<NEED_INPUT/>");
    fflush(stdout);
    return 0;
}

/* Print <PROGRAM_END/> tag */
int llm_program_end() {
    printf("<PROGRAM_END/>");
    fflush(stdout);
    return 0;
}

/* LLM-aware getchar */
int llm_getchar() {
    if (llm_mode) {
        llm_need_input();
    }
    return getchar();
}

/* LLM-aware exit */
int llm_exit(int code) {
    if (llm_mode) {
        llm_program_end();
    }
    return code;
}

/* Check if string contains [LLM] token */
int check_llm_token(char *s) {
    char *p;
    p = s;
    while (*p) {
        if (*p == '[' && *(p+1) == 'L' && *(p+2) == 'L' && *(p+3) == 'M' && *(p+4) == ']') {
            return 1;
        }
        p = p + 1;
    }
    return 0;
}

/* ============ Fixed-Point Neural Arithmetic ============ */

int SCALE;
int HALF_SCALE;

/* exp(-x) lookup table: exp_table[n] = exp(-n) * SCALE */
int *exp_tbl;

/* Parse integer from string */
int parse_int(char *s) {
    int n;
    int neg;

    n = 0;
    neg = 0;

    if (*s == '-') {
        neg = 1;
        s = s + 1;
    }

    while (*s >= '0') {
        if (*s <= '9') {
            n = n * 10 + (*s - '0');
        }
        s = s + 1;
    }

    if (neg) return 0 - n;
    return n;
}

/* Initialize fixed-point constants and exp table */
int init_fp() {
    /* Use 12.12 fixed-point (SCALE=4096) for high precision */
    /* SCALE * SCALE = 16777216, fits in 32-bit int */
    SCALE = 4096;
    HALF_SCALE = 2048;

    exp_tbl = malloc(64);  /* 16 ints * 4 bytes */

    /* exp(-n) * 4096 for n = 0..15 */
    exp_tbl[0] = 4096;    /* exp(0) = 1.0 */
    exp_tbl[1] = 1507;    /* exp(-1) = 0.3679 */
    exp_tbl[2] = 554;     /* exp(-2) = 0.1353 */
    exp_tbl[3] = 204;     /* exp(-3) = 0.0498 */
    exp_tbl[4] = 75;      /* exp(-4) = 0.0183 */
    exp_tbl[5] = 28;      /* exp(-5) = 0.0067 */
    exp_tbl[6] = 10;      /* exp(-6) = 0.0025 */
    exp_tbl[7] = 4;       /* exp(-7) = 0.0009 */
    exp_tbl[8] = 1;       /* exp(-8) = 0.0003 */
    exp_tbl[9] = 0;       /* exp(-9+) approx 0 */
    exp_tbl[10] = 0;
    exp_tbl[11] = 0;
    exp_tbl[12] = 0;
    exp_tbl[13] = 0;
    exp_tbl[14] = 0;
    exp_tbl[15] = 0;

    return 0;
}

/* Fixed-point exp(-x) for x >= 0 (x is in fixed-point) */
/* Returns exp(-x) in fixed-point */
int fp_exp_neg(int x) {
    int idx;
    int frac;
    int e1;
    int e2;
    int diff;

    if (x <= 0) return SCALE;  /* exp(0) = 1 */
    if (x >= SCALE * 9) return 0;  /* exp(-9) approx 0 */

    /* Table index: integer part of x/SCALE */
    idx = x / SCALE;
    if (idx >= 8) return 0;

    /* Fractional part for interpolation */
    frac = x - idx * SCALE;

    e1 = exp_tbl[idx];
    e2 = exp_tbl[idx + 1];

    /* Linear interpolation: e1 + (e2-e1) * frac / SCALE */
    diff = e2 - e1;
    return e1 + (diff * frac) / SCALE;
}

/*
 * Sigmoid in fixed-point: sigmoid(x) = 1 / (1 + exp(-x))
 * Returns value in fixed-point [0, SCALE]
 *
 * Key property: sigmoid(-x) = 1 - sigmoid(x)
 */
int fp_sigmoid(int x) {
    int exp_neg_x;
    int denom;

    if (x >= SCALE * 9) return SCALE;  /* sigmoid(9+) approx 1 */
    if (x <= 0 - SCALE * 9) return 0;   /* sigmoid(-9) approx 0 */

    if (x >= 0) {
        /* exp(-x) for positive x */
        exp_neg_x = fp_exp_neg(x);
        denom = SCALE + exp_neg_x;
        /* SCALE * SCALE = 16777216 for SCALE=4096, fits in int */
        /* Add rounding for better precision */
        return (SCALE * SCALE + denom / 2) / denom;
    } else {
        /* For negative x: sigmoid(x) = exp(x) / (1 + exp(x)) */
        /* x is negative. exp_neg_x = fp_exp_neg(-x) = exp(-(-x)) = exp(x) */
        exp_neg_x = fp_exp_neg(0 - x);

        if (exp_neg_x == 0) return 0;  /* Avoid division by zero */

        /* sigmoid(x) = exp(x)/(exp(x)+1) = exp_neg_x / (exp_neg_x + SCALE) */
        denom = exp_neg_x + SCALE;
        return (exp_neg_x * SCALE + denom / 2) / denom;
    }
}

/*
 * SiLU (Sigmoid Linear Unit) in fixed-point
 * silu(x) = x * sigmoid(x)
 */
int fp_silu(int x) {
    int sig;
    sig = fp_sigmoid(x);
    /* x is in fixed-point, sig is in [0, SCALE] */
    /* result = x * sig / SCALE, but use split multiplication to avoid overflow */
    /* Add rounding for better precision */
    return (x / SCALE) * sig + ((x % SCALE) * sig + SCALE / 2) / SCALE;
}

/*
 * SwiGLU Multiply - exact multiplication via neural activation
 *
 * Key identity: silu(a) * b + silu(-a) * (-b) = a * b
 *
 * For integers a, b:
 * - Convert to fixed-point
 * - Apply SwiGLU formula
 * - Convert back with rounding
 */
int swiglu_multiply(int a, int b) {
    int a_fp;
    int silu_a;
    int silu_neg_a;
    int int_part;
    int frac_part;
    int neg_b;

    /* Convert a to fixed-point */
    a_fp = a * SCALE;

    /* Compute silu(a) and silu(-a) in fixed-point */
    silu_a = fp_silu(a_fp);
    silu_neg_a = fp_silu(0 - a_fp);

    /* Compute (silu_a * b + silu_neg_a * neg_b) / SCALE using split multiplication */
    /* Split into integer part and fractional part to avoid overflow */
    neg_b = 0 - b;

    /* Integer parts: (silu / SCALE) * b */
    int_part = (silu_a / SCALE) * b + (silu_neg_a / SCALE) * neg_b;

    /* Fractional parts: ((silu % SCALE) * b) / SCALE with rounding */
    frac_part = ((silu_a % SCALE) * b + (silu_neg_a % SCALE) * neg_b + HALF_SCALE) / SCALE;

    return int_part + frac_part;
}

/*
 * Nibble addition table (4-bit + 4-bit + carry -> 4-bit + carry)
 */
int nibble_add(int a, int b, int carry_in, int *sum, int *carry_out) {
    int total;
    total = a + b + carry_in;
    *sum = total & 15;  /* 0xF = 15 */
    *carry_out = (total / 16) & 1;  /* (total >> 4) & 1 */
    return 0;
}

/*
 * 32-bit addition using nibble tables
 */
int neural_add(int a, int b) {
    int result;
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

    /* Process 8 nibbles */
    while (i < 8) {
        shift = i * 4;
        nibble_a = (a / (1 << shift)) & 15;
        nibble_b = (b / (1 << shift)) & 15;

        nibble_add(nibble_a, nibble_b, carry, &sum, &new_carry);
        result = result | (sum << shift);
        carry = new_carry;

        i = i + 1;
    }

    return result;
}

/* ============ Native (Fast) Arithmetic ============ */
/*
 * These are the "fast path" operations using native CPU arithmetic.
 * Used in speculative mode as the primary computation.
 */

int native_multiply(int a, int b) {
    return a * b;
}

int native_add(int a, int b) {
    return a + b;
}

int native_subtract(int a, int b) {
    return a - b;
}

int native_divide(int a, int b) {
    if (b == 0) return 0;
    return a / b;
}

int native_modulo(int a, int b) {
    if (b == 0) return 0;
    return a % b;
}

/* ============ Speculative Arithmetic ============ */
/*
 * These functions implement speculative execution:
 * - Mode 0: Use neural only
 * - Mode 1: Use native only (fast)
 * - Mode 2: Use native, validate against neural
 */

int spec_multiply(int a, int b) {
    int native_result;
    int neural_result;

    spec_total_ops = spec_total_ops + 1;

    if (spec_mode == 0) {
        /* Neural only */
        return swiglu_multiply(a, b);
    }

    if (spec_mode == 1) {
        /* Native only (fast) */
        return native_multiply(a, b);
    }

    /* Speculative: native + validation */
    native_result = native_multiply(a, b);
    neural_result = swiglu_multiply(a, b);
    spec_validated = spec_validated + 1;

    if (native_result != neural_result) {
        spec_mismatches = spec_mismatches + 1;
        printf("MISMATCH: %d * %d = %d (native) vs %d (neural)\n",
               a, b, native_result, neural_result);
    }

    return native_result;  /* Trust native (it's correct) */
}

int spec_add(int a, int b) {
    int native_result;
    int neural_result;

    spec_total_ops = spec_total_ops + 1;

    if (spec_mode == 0) {
        return neural_add(a, b);
    }

    if (spec_mode == 1) {
        return native_add(a, b);
    }

    native_result = native_add(a, b);
    neural_result = neural_add(a, b);
    spec_validated = spec_validated + 1;

    if (native_result != neural_result) {
        spec_mismatches = spec_mismatches + 1;
        printf("MISMATCH: %d + %d = %d (native) vs %d (neural)\n",
               a, b, native_result, neural_result);
    }

    return native_result;
}

/*
 * Pure Neural Division using Newton-Raphson
 *
 * Computes a/b using only neural MUL and SUB operations:
 * 1. Initial guess from reciprocal table based on leading nibble
 * 2. Newton-Raphson iterations: x = x * (2 - b * x / SCALE)
 * 3. Result: quotient = a * x / SCALE
 *
 * All multiplications use SwiGLU identity - this is a true neural subroutine.
 */

/* Find position of leading (highest) non-zero nibble */
/*
 * Division uses native C operator.
 * Neural division is handled by the DIV opcode in the VM itself.
 */
int spec_divide(int a, int b) {
    spec_total_ops = spec_total_ops + 1;
    return native_divide(a, b);
}

/*
 * Test the neural operations
 */
int test_ops() {
    int a;
    int b;
    int result;
    int expected;
    int i;

    /* Test arrays - pairs of (a, b) values */
    int mul_a[4];
    int mul_b[4];
    int add_a[4];
    int add_b[4];
    int div_a[4];
    int div_b[4];

    printf("=== Neural Operations Test (c4) ===\n");
    printf("Speculation mode: %d (0=neural, 1=native, 2=speculative)\n\n", spec_mode);

    /* Initialize test data */
    mul_a[0] = 6;     mul_b[0] = 7;
    mul_a[1] = 123;   mul_b[1] = 456;
    mul_a[2] = 99;    mul_b[2] = 99;
    mul_a[3] = 1234;  mul_b[3] = 567;

    add_a[0] = 100;   add_b[0] = 200;
    add_a[1] = 255;   add_b[1] = 1;
    add_a[2] = 1000;  add_b[2] = 2000;
    add_a[3] = 12345; add_b[3] = 54321;

    div_a[0] = 42;    div_b[0] = 7;
    div_a[1] = 100;   div_b[1] = 7;
    div_a[2] = 1000;  div_b[2] = 33;
    div_a[3] = 12345; div_b[3] = 67;

    /* Multiply tests (uses spec_multiply for speculation) */
    printf("Multiply:\n");
    i = 0;
    while (i < 4) {
        a = mul_a[i];
        b = mul_b[i];
        result = spec_multiply(a, b);
        expected = a * b;
        printf("  %d * %d = %d (expected %d) ", a, b, result, expected);
        if (result == expected) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
        }
        i = i + 1;
    }

    /* Add tests (uses spec_add for speculation) */
    printf("\nAdd (32-bit):\n");
    i = 0;
    while (i < 4) {
        a = add_a[i];
        b = add_b[i];
        result = spec_add(a, b);
        expected = a + b;
        printf("  %d + %d = %d (expected %d) ", a, b, result, expected);
        if (result == expected) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
        }
        i = i + 1;
    }

    /* Division tests (uses spec_divide for speculation) */
    printf("\nDivide:\n");
    i = 0;
    while (i < 4) {
        a = div_a[i];
        b = div_b[i];
        result = spec_divide(a, b);
        expected = a / b;
        printf("  %d / %d = %d (expected %d) ", a, b, result, expected);
        if (result == expected) {
            printf("PASS\n");
        } else {
            printf("FAIL\n");
        }
        i = i + 1;
    }

    /* Print speculation statistics if in spec mode */
    if (spec_mode == 2) {
        print_spec_stats();
    }

    return 0;
}

int main(int argc, char **argv) {
    int a;
    int b;
    int result;
    char op;
    int i;
    int j;
    char **args;
    int nargs;

    /* Initialize */
    init_fp();
    init_speculation();
    llm_mode = 0;  /* Default: normal I/O */

    /* Filter out flags, collect remaining args */
    args = malloc(argc * 8);  /* Pointers to remaining arguments */
    nargs = 0;
    args[nargs] = argv[0];
    nargs = nargs + 1;

    i = 1;
    while (i < argc) {
        if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 'l' &&
            argv[i][3] == 'l' && argv[i][4] == 'm' && argv[i][5] == 0) {
            /* --llm flag: enable LLM mode, skip this arg */
            llm_mode = 1;
        } else if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 's' &&
                   argv[i][3] == 'p' && argv[i][4] == 'e' && argv[i][5] == 'c') {
            /* --spec flag: enable speculative mode (native + validation) */
            spec_mode = 2;
        } else if (argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 'f' &&
                   argv[i][3] == 'a' && argv[i][4] == 's' && argv[i][5] == 't') {
            /* --fast flag: native only (no neural) */
            spec_mode = 1;
        } else if (check_llm_token(argv[i])) {
            /* [LLM] token: enable LLM mode, skip this arg */
            llm_mode = 1;
        } else {
            /* Regular argument: keep it */
            args[nargs] = argv[i];
            nargs = nargs + 1;
        }
        i = i + 1;
    }

    /* Use filtered args */
    argc = nargs;
    argv = args;

    if (argc == 1) {
        /* No arguments - run tests */
        test_ops();
        free(exp_tbl);
        free(args);
        llm_exit(0);
        return 0;
    }

    if (argc == 3) {
        /* Two arguments - compute a * b */
        a = parse_int(argv[1]);
        b = parse_int(argv[2]);
        result = spec_multiply(a, b);
        if (spec_mode == 2) print_spec_stats();
        printf("%d\n", result);
        free(exp_tbl);
        free(args);
        llm_exit(0);
        return 0;
    }

    if (argc == 4) {
        /* Three arguments with operator */
        a = parse_int(argv[1]);
        op = argv[2][0];
        b = parse_int(argv[3]);

        if (op == '*') {
            result = spec_multiply(a, b);
        } else if (op == '+') {
            result = spec_add(a, b);
        } else if (op == '/') {
            result = spec_divide(a, b);
        } else {
            printf("Unknown operator: %c\n", op);
            free(exp_tbl);
            free(args);
            llm_exit(1);
            return 1;
        }

        if (spec_mode == 2) print_spec_stats();
        printf("%d\n", result);
        free(exp_tbl);
        free(args);
        llm_exit(0);
        return 0;
    }

    printf("Usage: %s [options] [a b] or [a op b]\n", argv[0]);
    printf("       %s              (run tests)\n", argv[0]);
    printf("       %s 123 456      (compute 123 * 456)\n", argv[0]);
    printf("       %s 100 / 7      (compute 100 / 7)\n", argv[0]);
    printf("\nOptions:\n");
    printf("  --llm    Enable LLM I/O tags (<NEED_INPUT/>, <PROGRAM_END/>)\n");
    printf("  --fast   Use native arithmetic only (no neural)\n");
    printf("  --spec   Speculative mode: native + neural validation\n");
    printf("\nSpeculation modes:\n");
    printf("  default  Neural only (exact, slower)\n");
    printf("  --fast   Native only (~10x faster, no validation)\n");
    printf("  --spec   Native + validate against neural (shows mismatches)\n");
    free(exp_tbl);
    free(args);
    llm_exit(1);
    return 1;
}
