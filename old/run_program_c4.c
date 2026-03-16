/*
 * Run Program - C4 Compatible
 *
 * Simple expression evaluator that uses ONNX model for arithmetic.
 * Mimics the Python run_program.py interface.
 *
 * Usage:
 *   ./run_program_c4 -e '6 * 7'                              # Uses built-in neural ops
 *   ./run_program_c4 -m models/swiglu_mul.onnx -e '6 * 7'    # Uses ONNX model
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -include fcntl.h -include unistd.h \
 *       -w -o run_program_c4 run_program_c4.c
 */

/* ============ ONNX Runtime State ============ */

int MAX_NODES;
int MAX_VARS;
int *node_op;
int *node_in1;
int *node_in2;
int *node_out;
int node_count;
int *var_hash;
int *var_val;
int var_count;
char *onnx_buf;
int onnx_len;
int onnx_pos;
int onnx_loaded;
int onnx_a_idx;
int onnx_b_idx;
int onnx_result_idx;

/* ============ Fixed-Point Neural Arithmetic ============ */

int SCALE;
int *exp_tbl;

int init_neural() {
    /* Use 12.12 fixed-point (SCALE=4096) for high precision */
    /* SCALE * SCALE = 16777216, still fits in 32-bit int */
    SCALE = 4096;
    exp_tbl = malloc(64);
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
    return 0;
}

int fp_exp_neg(int x) {
    int idx; int frac; int e1; int e2;
    if (x <= 0) return SCALE;
    if (x >= SCALE * 9) return 0;
    idx = x / SCALE;
    if (idx >= 8) return 0;
    frac = x - idx * SCALE;
    e1 = exp_tbl[idx];
    e2 = exp_tbl[idx + 1];
    return e1 + ((e2 - e1) * frac) / SCALE;
}

int fp_sigmoid(int x) {
    int exp_neg_x; int denom;
    if (x >= SCALE * 9) return SCALE;
    if (x <= 0 - SCALE * 9) return 0;
    if (x >= 0) {
        exp_neg_x = fp_exp_neg(x);
        denom = SCALE + exp_neg_x;
        /* Add rounding for better precision */
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
    /* Use split multiplication to avoid overflow, with rounding */
    return (x / SCALE) * sig + ((x % SCALE) * sig + SCALE / 2) / SCALE;
}

/* SwiGLU multiply (built-in): silu(a)*b + silu(-a)*(-b) = a*b */
int builtin_mul(int a, int b) {
    int a_fp; int silu_a; int silu_neg_a;
    int int_part; int frac_part;
    int half_scale; int neg_b;

    half_scale = SCALE / 2;
    a_fp = a * SCALE;
    silu_a = fp_silu(a_fp);
    silu_neg_a = fp_silu(0 - a_fp);

    /* Compute (silu_a * b + silu_neg_a * neg_b) / SCALE using split multiplication */
    neg_b = 0 - b;

    /* Integer parts: (silu / SCALE) * b */
    int_part = (silu_a / SCALE) * b + (silu_neg_a / SCALE) * neg_b;

    /* Fractional parts with rounding */
    frac_part = ((silu_a % SCALE) * b + (silu_neg_a % SCALE) * neg_b + half_scale) / SCALE;

    return int_part + frac_part;
}

/* ============ ONNX Protobuf Parser ============ */

int str_hash(char *s, int len) {
    int h; int i;
    h = 0; i = 0;
    while (i < len) {
        h = h * 31 + s[i];
        i = i + 1;
    }
    return h & 0x7FFFFFFF;
}

int str_eq(char *a, int alen, char *b, int blen) {
    int i;
    if (alen != blen) return 0;
    i = 0;
    while (i < alen) {
        if (a[i] != b[i]) return 0;
        i = i + 1;
    }
    return 1;
}

int find_or_create_var(char *name, int len) {
    int h; int i;
    h = str_hash(name, len);
    i = 0;
    while (i < var_count) {
        if (var_hash[i] == h) return i;
        i = i + 1;
    }
    var_hash[var_count] = h;
    var_val[var_count] = 0;
    var_count = var_count + 1;
    return var_count - 1;
}

int read_varint() {
    int result; int shift; int b;
    result = 0; shift = 0;
    while (onnx_pos < onnx_len) {
        b = onnx_buf[onnx_pos] & 255;
        onnx_pos = onnx_pos + 1;
        result = result | ((b & 127) << shift);
        if ((b & 128) == 0) return result;
        shift = shift + 7;
    }
    return result;
}

int skip_field(int wire_type) {
    int len;
    if (wire_type == 0) { read_varint(); }
    else if (wire_type == 1) { onnx_pos = onnx_pos + 8; }
    else if (wire_type == 2) { len = read_varint(); onnx_pos = onnx_pos + len; }
    else if (wire_type == 5) { onnx_pos = onnx_pos + 4; }
    return 0;
}

int parse_node(int end_pos) {
    int tag; int field; int wire; int len; int op;
    int in1; int in2; int out;
    char *str_start;

    op = 0; in1 = -1; in2 = -1; out = -1;

    while (onnx_pos < end_pos) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;

        if (field == 1 && wire == 2) {
            len = read_varint();
            str_start = onnx_buf + onnx_pos;
            if (in1 == -1) in1 = find_or_create_var(str_start, len);
            else in2 = find_or_create_var(str_start, len);
            onnx_pos = onnx_pos + len;
        } else if (field == 2 && wire == 2) {
            len = read_varint();
            str_start = onnx_buf + onnx_pos;
            out = find_or_create_var(str_start, len);
            onnx_pos = onnx_pos + len;
        } else if (field == 4 && wire == 2) {
            len = read_varint();
            str_start = onnx_buf + onnx_pos;
            if (str_eq(str_start, len, "Sigmoid", 7)) op = 1;
            else if (str_eq(str_start, len, "Mul", 3)) op = 2;
            else if (str_eq(str_start, len, "Neg", 3)) op = 3;
            else if (str_eq(str_start, len, "Add", 3)) op = 4;
            onnx_pos = onnx_pos + len;
        } else {
            skip_field(wire);
        }
    }

    if (op > 0 && out >= 0) {
        node_op[node_count] = op;
        node_in1[node_count] = in1;
        node_in2[node_count] = in2;
        node_out[node_count] = out;
        node_count = node_count + 1;
    }
    return 0;
}

int parse_graph(int end_pos) {
    int tag; int field; int wire; int len;
    while (onnx_pos < end_pos) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;
        if (field == 1 && wire == 2) {
            len = read_varint();
            parse_node(onnx_pos + len);
        } else {
            skip_field(wire);
        }
    }
    return 0;
}

int parse_model() {
    int tag; int field; int wire; int len;
    while (onnx_pos < onnx_len) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;
        if (field == 7 && wire == 2) {
            len = read_varint();
            parse_graph(onnx_pos + len);
        } else {
            skip_field(wire);
        }
    }
    return 0;
}

int execute_onnx_graph() {
    int i; int op; int v1; int v2; int result;
    i = 0;
    while (i < node_count) {
        op = node_op[i];
        v1 = var_val[node_in1[i]];
        if (node_in2[i] >= 0) v2 = var_val[node_in2[i]];
        else v2 = 0;

        if (op == 1) result = fp_sigmoid(v1);
        else if (op == 2) result = (v1 / SCALE) * v2 + ((v1 % SCALE) * v2 + SCALE / 2) / SCALE;
        else if (op == 3) result = 0 - v1;
        else if (op == 4) result = v1 + v2;
        else result = 0;

        var_val[node_out[i]] = result;
        i = i + 1;
    }
    return 0;
}

int load_onnx(char *filename) {
    int fd;

    MAX_NODES = 32;
    MAX_VARS = 64;
    node_op = malloc(MAX_NODES * 4);
    node_in1 = malloc(MAX_NODES * 4);
    node_in2 = malloc(MAX_NODES * 4);
    node_out = malloc(MAX_NODES * 4);
    var_hash = malloc(MAX_VARS * 4);
    var_val = malloc(MAX_VARS * 4);
    node_count = 0;
    var_count = 0;

    fd = open(filename, 0);
    if (fd < 0) return -1;

    onnx_buf = malloc(8192);
    onnx_len = read(fd, onnx_buf, 8192);
    close(fd);
    onnx_pos = 0;

    parse_model();

    /* Save input/output variable indices */
    onnx_a_idx = find_or_create_var("a", 1);
    onnx_b_idx = find_or_create_var("b", 1);
    onnx_result_idx = find_or_create_var("result", 6);

    onnx_loaded = 1;
    return 0;
}

/* ONNX-based multiplication */
int onnx_mul(int a, int b) {
    int i;
    int half_scale;

    half_scale = SCALE / 2;

    /* Clear all variable values */
    i = 0;
    while (i < var_count) { var_val[i] = 0; i = i + 1; }

    /* Set inputs: a in fixed-point, b as integer (used as multiplier) */
    var_val[onnx_a_idx] = a * SCALE;
    var_val[onnx_b_idx] = b;

    execute_onnx_graph();

    /* Result is already an integer since b was used as integer multiplier */
    return var_val[onnx_result_idx];
}

/* Neural multiply - uses ONNX if loaded, otherwise built-in */
int neural_mul(int a, int b) {
    if (onnx_loaded) return onnx_mul(a, b);
    return builtin_mul(a, b);
}

/* Division via repeated subtraction */
int neural_div(int a, int b) {
    int result; int neg;
    if (b == 0) return 0;
    neg = 0;
    if (a < 0) { a = 0 - a; neg = neg + 1; }
    if (b < 0) { b = 0 - b; neg = neg + 1; }
    result = 0;
    while (a >= b) { a = a - b; result = result + 1; }
    if (neg == 1) return 0 - result;
    return result;
}

/* Nibble-based addition */
int neural_add(int a, int b) {
    int result; int carry; int i; int shift;
    int nibble_a; int nibble_b; int sum; int total;
    result = 0;
    carry = 0;
    i = 0;
    while (i < 8) {
        shift = i * 4;
        nibble_a = (a / (1 << shift)) & 15;
        nibble_b = (b / (1 << shift)) & 15;
        total = nibble_a + nibble_b + carry;
        sum = total & 15;
        carry = (total / 16) & 1;
        result = result | (sum << shift);
        i = i + 1;
    }
    return result;
}

/* ============ Expression Parser ============ */

char *expr;
int expr_pos;

int is_digit(int c) { return c >= '0' && c <= '9'; }
int is_space(int c) { return c == ' ' || c == '\t'; }

int skip_space() {
    while (is_space(expr[expr_pos])) expr_pos = expr_pos + 1;
    return 0;
}

int parse_number() {
    int num; int neg;
    num = 0;
    neg = 0;
    skip_space();
    if (expr[expr_pos] == '-') {
        neg = 1;
        expr_pos = expr_pos + 1;
        skip_space();
    }
    while (is_digit(expr[expr_pos])) {
        num = num * 10 + (expr[expr_pos] - '0');
        expr_pos = expr_pos + 1;
    }
    if (neg) return 0 - num;
    return num;
}

int parse_op() {
    int op;
    skip_space();
    op = expr[expr_pos];
    if (op == '+' || op == '-' || op == '*' || op == '/') {
        expr_pos = expr_pos + 1;
        return op;
    }
    return 0;
}

/* Evaluate simple expression: num op num [op num ...] */
int eval_expr(char *e) {
    int a; int b; int op; int result;

    expr = e;
    expr_pos = 0;

    result = parse_number();

    while (1) {
        op = parse_op();
        if (op == 0) break;

        b = parse_number();

        if (op == '+') {
            result = neural_add(result, b);
        } else if (op == '-') {
            result = result - b;
        } else if (op == '*') {
            result = neural_mul(result, b);
        } else if (op == '/') {
            result = neural_div(result, b);
        }
    }

    return result;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    int i;
    int result;
    int verbose;
    char *expression;
    char *model_file;

    init_neural();
    onnx_loaded = 0;

    expression = 0;
    model_file = 0;
    verbose = 0;

    i = 1;
    while (i < argc) {
        if (argv[i][0] == '-' && argv[i][1] == 'e') {
            i = i + 1;
            if (i < argc) {
                expression = argv[i];
            }
        } else if (argv[i][0] == '-' && argv[i][1] == 'm') {
            i = i + 1;
            if (i < argc) {
                model_file = argv[i];
            }
        } else if (argv[i][0] == '-' && argv[i][1] == 'v') {
            verbose = 1;
        }
        i = i + 1;
    }

    if (expression == 0) {
        printf("Usage: %s [-m model.onnx] -e 'expression'\n", argv[0]);
        printf("Examples:\n");
        printf("  %s -e '6 * 7'                           # Built-in neural ops\n", argv[0]);
        printf("  %s -m models/swiglu_mul.onnx -e '6 * 7' # Use ONNX model\n", argv[0]);
        printf("  %s -e '123 + 456'                       # Returns 579\n", argv[0]);
        printf("  %s -e '100 / 7'                         # Returns 14\n", argv[0]);
        free(exp_tbl);
        return 1;
    }

    /* Load ONNX model if specified */
    if (model_file) {
        if (load_onnx(model_file) < 0) {
            printf("Error: Could not load %s\n", model_file);
            free(exp_tbl);
            return 1;
        }
        if (verbose) {
            printf("Loaded ONNX model: %d nodes\n", node_count);
        }
    }

    result = eval_expr(expression);

    if (verbose) {
        printf("Result: %d\n", result);
    }

    free(exp_tbl);
    return result & 255;
}
