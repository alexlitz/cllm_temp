/*
 * Minimal ONNX Runtime for C4
 *
 * Parses and executes the SwiGLU ONNX model using only c4-compatible C.
 * This is a real ONNX runtime that reads the protobuf binary format,
 * extracts the computation graph, and executes it.
 *
 * Supported ONNX ops: Sigmoid, Mul, Neg, Add
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -include fcntl.h -include unistd.h \
 *       -o onnx_runtime_c4 onnx_runtime_c4.c
 *
 * Usage:
 *   ./onnx_runtime_c4 models/swiglu_mul.onnx 6 7
 *   ./onnx_runtime_c4 models/swiglu_mul.onnx 123 456 debug   # shows parsed graph
 */

/* Fixed-point constants (8.8 format) */
int SCALE;
int *exp_tbl;

/* Node structure for ONNX graph */
int MAX_NODES;
int MAX_VARS;

/* Node arrays - each node has: op_type, input1, input2, output */
int *node_op;      /* 0=none, 1=Sigmoid, 2=Mul, 3=Neg, 4=Add */
int *node_in1;     /* index into vars */
int *node_in2;     /* index into vars (-1 if unused) */
int *node_out;     /* index into vars */
int node_count;

/* Variable storage: name hash -> value */
int *var_hash;     /* hash of variable name */
int *var_val;      /* fixed-point value */
int var_count;

/* File buffer */
char *buf;
int buf_len;
int buf_pos;

/* ============ String/Hash Utilities ============ */

int str_hash(char *s, int len) {
    int h;
    int i;
    h = 0;
    i = 0;
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

/* ============ Variable Management ============ */

int find_or_create_var(char *name, int len) {
    int h;
    int i;

    h = str_hash(name, len);
    i = 0;
    while (i < var_count) {
        if (var_hash[i] == h) return i;
        i = i + 1;
    }

    /* Create new variable */
    var_hash[var_count] = h;
    var_val[var_count] = 0;
    var_count = var_count + 1;
    return var_count - 1;
}

/* ============ Fixed-Point Math ============ */

int init_fp() {
    /* Use 12.12 fixed-point (SCALE=4096) for high precision */
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
    int idx;
    int frac;
    int e1;
    int e2;

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
    int exp_neg_x;
    int denom;

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

/* ============ Protobuf Parsing ============ */

int read_varint() {
    int result;
    int shift;
    int b;

    result = 0;
    shift = 0;

    while (buf_pos < buf_len) {
        b = buf[buf_pos] & 255;
        buf_pos = buf_pos + 1;
        result = result | ((b & 127) << shift);
        if ((b & 128) == 0) return result;
        shift = shift + 7;
    }
    return result;
}

int skip_field(int wire_type) {
    int len;

    if (wire_type == 0) {
        read_varint();
    } else if (wire_type == 1) {
        buf_pos = buf_pos + 8;
    } else if (wire_type == 2) {
        len = read_varint();
        buf_pos = buf_pos + len;
    } else if (wire_type == 5) {
        buf_pos = buf_pos + 4;
    }
    return 0;
}

/* Parse a NodeProto and add to graph */
int parse_node(int end_pos) {
    int tag;
    int field;
    int wire;
    int len;
    int op;
    int in1;
    int in2;
    int out;
    char *str_start;

    op = 0;
    in1 = -1;
    in2 = -1;
    out = -1;

    while (buf_pos < end_pos) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;

        if (field == 1 && wire == 2) {
            /* input */
            len = read_varint();
            str_start = buf + buf_pos;
            if (in1 == -1) {
                in1 = find_or_create_var(str_start, len);
            } else {
                in2 = find_or_create_var(str_start, len);
            }
            buf_pos = buf_pos + len;
        } else if (field == 2 && wire == 2) {
            /* output */
            len = read_varint();
            str_start = buf + buf_pos;
            out = find_or_create_var(str_start, len);
            buf_pos = buf_pos + len;
        } else if (field == 4 && wire == 2) {
            /* op_type */
            len = read_varint();
            str_start = buf + buf_pos;

            /* Match op_type string */
            if (str_eq(str_start, len, "Sigmoid", 7)) {
                op = 1;
            } else if (str_eq(str_start, len, "Mul", 3)) {
                op = 2;
            } else if (str_eq(str_start, len, "Neg", 3)) {
                op = 3;
            } else if (str_eq(str_start, len, "Add", 3)) {
                op = 4;
            }
            buf_pos = buf_pos + len;
        } else {
            skip_field(wire);
        }
    }

    /* Add node if we found a valid op */
    if (op > 0 && out >= 0) {
        node_op[node_count] = op;
        node_in1[node_count] = in1;
        node_in2[node_count] = in2;
        node_out[node_count] = out;
        node_count = node_count + 1;
    }

    return 0;
}

/* Parse GraphProto to extract nodes */
int parse_graph(int end_pos) {
    int tag;
    int field;
    int wire;
    int len;

    while (buf_pos < end_pos) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;

        if (field == 1 && wire == 2) {
            /* node - NodeProto */
            len = read_varint();
            parse_node(buf_pos + len);
        } else {
            skip_field(wire);
        }
    }

    return 0;
}

/* Parse ModelProto to find graph */
int parse_model() {
    int tag;
    int field;
    int wire;
    int len;

    while (buf_pos < buf_len) {
        tag = read_varint();
        field = tag / 8;
        wire = tag & 7;

        if (field == 7 && wire == 2) {
            /* graph - GraphProto */
            len = read_varint();
            parse_graph(buf_pos + len);
        } else {
            skip_field(wire);
        }
    }

    return 0;
}

/* ============ Graph Execution ============ */

int execute_graph() {
    int i;
    int op;
    int v1;
    int v2;
    int result;

    i = 0;
    while (i < node_count) {
        op = node_op[i];
        v1 = var_val[node_in1[i]];

        if (node_in2[i] >= 0) {
            v2 = var_val[node_in2[i]];
        } else {
            v2 = 0;
        }

        if (op == 1) {
            /* Sigmoid */
            result = fp_sigmoid(v1);
        } else if (op == 2) {
            /* Mul: (v1 * v2) / SCALE, but avoid overflow */
            /* Split with rounding: (v1 / SCALE) * v2 + ((v1 % SCALE) * v2 + SCALE/2) / SCALE */
            result = (v1 / SCALE) * v2 + ((v1 % SCALE) * v2 + SCALE / 2) / SCALE;
        } else if (op == 3) {
            /* Neg */
            result = 0 - v1;
        } else if (op == 4) {
            /* Add */
            result = v1 + v2;
        } else {
            result = 0;
        }

        var_val[node_out[i]] = result;
        i = i + 1;
    }

    return 0;
}

/* ============ Integer Parsing ============ */

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

/* ============ Main ============ */

int main(int argc, char **argv) {
    int fd;
    int i;
    int a_idx;
    int b_idx;
    int result_idx;
    int a_val;
    int b_val;
    int result;

    if (argc < 4) {
        printf("Usage: %s <model.onnx> <a> <b>\n", argv[0]);
        return 1;
    }

    /* Initialize */
    init_fp();

    MAX_NODES = 32;
    MAX_VARS = 64;

    node_op = malloc(MAX_NODES * 4);
    node_in1 = malloc(MAX_NODES * 4);
    node_in2 = malloc(MAX_NODES * 4);
    node_out = malloc(MAX_NODES * 4);
    node_count = 0;

    var_hash = malloc(MAX_VARS * 4);
    var_val = malloc(MAX_VARS * 4);
    var_count = 0;

    /* Read ONNX file */
    fd = open(argv[1], 0);
    if (fd < 0) {
        printf("Could not open %s\n", argv[1]);
        return 1;
    }

    buf = malloc(8192);
    buf_len = read(fd, buf, 8192);
    close(fd);
    buf_pos = 0;

    /* Parse model */
    parse_model();

    /* Debug: show parsed graph */
    if (argc > 4) {
        printf("Parsed %d nodes from ONNX:\n", node_count);
        i = 0;
        while (i < node_count) {
            if (node_op[i] == 1) printf("  [%d] Sigmoid\n", i);
            else if (node_op[i] == 2) printf("  [%d] Mul\n", i);
            else if (node_op[i] == 3) printf("  [%d] Neg\n", i);
            else if (node_op[i] == 4) printf("  [%d] Add\n", i);
            i = i + 1;
        }
        printf("\n");
    }

    /* Set up inputs */
    a_val = parse_int(argv[2]);
    b_val = parse_int(argv[3]);

    /* Find input variable indices (a and b are single-char names) */
    a_idx = find_or_create_var("a", 1);
    b_idx = find_or_create_var("b", 1);
    result_idx = find_or_create_var("result", 6);

    /* Set input values: a in fixed-point, b as integer (used as multiplier) */
    var_val[a_idx] = a_val * SCALE;
    var_val[b_idx] = b_val;

    /* Execute graph */
    execute_graph();

    /* Get result - already an integer since b was used as integer multiplier */
    result = var_val[result_idx];

    printf("%d\n", result);

    /* Cleanup */
    free(buf);
    free(node_op);
    free(node_in1);
    free(node_in2);
    free(node_out);
    free(var_hash);
    free(var_val);
    free(exp_tbl);

    return 0;
}
