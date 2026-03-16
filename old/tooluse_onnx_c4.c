/*
 * Tool-Use ONNX Runtime - C4 Compatible
 *
 * Runs ONNX models with tool-use I/O protocol.
 * I/O operations generate tool calls that are handled externally.
 *
 * Protocol:
 *   Output: TOOL_CALL:<type>:<id>:<params_json>
 *   Input:  TOOL_RESPONSE:<id>:<result>
 *
 * Usage:
 *   ./tooluse_onnx_c4 -m model.onnx       # Interactive mode
 *   ./tooluse_onnx_c4 -m model.onnx -b    # Batch mode (raw I/O)
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -include fcntl.h -include unistd.h \
 *       -include string.h -w -o tooluse_onnx_c4 tooluse_onnx_c4.c
 */

/* ============ Tool Call Protocol ============ */

int tool_call_id;
int tool_use_enabled;
char *tool_response_buf;

int init_tooluse() {
    tool_call_id = 0;
    tool_use_enabled = 0;
    tool_response_buf = malloc(4096);
    return 0;
}

int next_call_id() {
    tool_call_id = tool_call_id + 1;
    return tool_call_id;
}

/* Emit a tool call and wait for response */
int emit_tool_call(char *call_type, int id, char *params) {
    printf("TOOL_CALL:%s:%d:{%s}\n", call_type, id, params);
    fflush(stdout);
    return 0;
}

/* Read tool response - returns the result value */
int read_tool_response(int expected_id) {
    char line[256];
    int id;
    int result;
    char *p;
    int i;

    /* Read line from stdin */
    if (fgets(line, 256, stdin) == 0) return -1;

    /* Parse TOOL_RESPONSE:<id>:<result> */
    p = line;

    /* Skip "TOOL_RESPONSE:" prefix */
    i = 0;
    while (i < 14 && *p) { p = p + 1; i = i + 1; }

    /* Parse id */
    id = 0;
    while (*p >= '0' && *p <= '9') {
        id = id * 10 + (*p - '0');
        p = p + 1;
    }
    if (*p == ':') p = p + 1;

    /* Parse result */
    result = 0;
    if (*p == '-') {
        p = p + 1;
        while (*p >= '0' && *p <= '9') {
            result = result * 10 + (*p - '0');
            p = p + 1;
        }
        result = 0 - result;
    } else {
        while (*p >= '0' && *p <= '9') {
            result = result * 10 + (*p - '0');
            p = p + 1;
        }
    }

    return result;
}

/* Tool-use putchar */
int tu_putchar(int c) {
    int id;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:putchar:%d:{\"char\":%d}\n", id, c & 255);
        fflush(stdout);
        /* Don't wait for response on output */
        return c;
    } else {
        putchar(c);
        return c;
    }
}

/* Tool-use getchar */
int tu_getchar() {
    int id;
    int result;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:getchar:%d:{}\n", id);
        fflush(stdout);
        result = read_tool_response(id);
        return result;
    } else {
        return getchar();
    }
}

/* Tool-use print string */
int tu_print_str(char *s) {
    int id;
    int i;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:print:%d:{\"text\":\"", id);
        /* Escape the string */
        while (*s) {
            if (*s == '"') printf("\\\"");
            else if (*s == '\\') printf("\\\\");
            else if (*s == '\n') printf("\\n");
            else if (*s == '\t') printf("\\t");
            else putchar(*s);
            s = s + 1;
        }
        printf("\"}\n");
        fflush(stdout);
        return 0;
    } else {
        while (*s) {
            putchar(*s);
            s = s + 1;
        }
        return 0;
    }
}

/* Tool-use file open */
int tu_open(char *path, int mode) {
    int id;
    int result;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:file_open:%d:{\"path\":\"", id);
        while (*path) {
            if (*path == '"') printf("\\\"");
            else putchar(*path);
            path = path + 1;
        }
        printf("\",\"mode\":\"%s\"}\n", mode == 0 ? "r" : "w");
        fflush(stdout);
        result = read_tool_response(id);
        return result;
    } else {
        return open(path, mode);
    }
}

/* Tool-use file read */
int tu_read(int fd, char *buf, int size) {
    int id;
    int result;
    int i;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:file_read:%d:{\"fd\":%d,\"size\":%d}\n", id, fd, size);
        fflush(stdout);
        /* For tool use, we need to read the data differently */
        /* Response format: TOOL_RESPONSE:<id>:<bytes_read>:<base64_data> */
        result = read_tool_response(id);
        /* TODO: Actually read the data into buf */
        return result;
    } else {
        return read(fd, buf, size);
    }
}

/* Tool-use file close */
int tu_close(int fd) {
    int id;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:file_close:%d:{\"fd\":%d}\n", id, fd);
        fflush(stdout);
        return read_tool_response(id);
    } else {
        return close(fd);
    }
}

/* Tool-use exit */
int tu_exit(int code) {
    int id;
    if (tool_use_enabled) {
        id = next_call_id();
        printf("TOOL_CALL:exit:%d:{\"code\":%d}\n", id, code);
        fflush(stdout);
    }
    return code;
}

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
    SCALE = 4096;
    exp_tbl = malloc(64);
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

int builtin_mul(int a, int b) {
    int a_fp; int silu_a; int silu_neg_a;
    int int_part; int frac_part;
    int half_scale; int neg_b;

    half_scale = SCALE / 2;
    a_fp = a * SCALE;
    silu_a = fp_silu(a_fp);
    silu_neg_a = fp_silu(0 - a_fp);
    neg_b = 0 - b;
    int_part = (silu_a / SCALE) * b + (silu_neg_a / SCALE) * neg_b;
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

    onnx_a_idx = find_or_create_var("a", 1);
    onnx_b_idx = find_or_create_var("b", 1);
    onnx_result_idx = find_or_create_var("result", 6);

    onnx_loaded = 1;
    return 0;
}

int onnx_mul(int a, int b) {
    int i;
    int half_scale;

    half_scale = SCALE / 2;
    i = 0;
    while (i < var_count) { var_val[i] = 0; i = i + 1; }

    var_val[onnx_a_idx] = a * SCALE;
    var_val[onnx_b_idx] = b;

    execute_onnx_graph();

    return var_val[onnx_result_idx];
}

int neural_mul(int a, int b) {
    if (onnx_loaded) return onnx_mul(a, b);
    return builtin_mul(a, b);
}

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

/* ============ Interactive Mode ============ */

char *input_buf;

int read_input_line() {
    int c;
    int i;
    i = 0;
    c = tu_getchar();
    while (c != '\n' && c >= 0) {
        input_buf[i] = c;
        i = i + 1;
        c = tu_getchar();
    }
    input_buf[i] = 0;
    if (c < 0 && i == 0) return -1;
    return i;
}

int run_interactive() {
    int n;
    int result;

    input_buf = malloc(256);

    tu_print_str("=====================================\n");
    tu_print_str("  Tool-Use ONNX Calculator\n");
    tu_print_str("=====================================\n");
    tu_print_str("Enter expressions (e.g. '6 * 7')\n");
    tu_print_str("Type 'quit' to exit\n\n");

    while (1) {
        tu_print_str("> ");
        n = read_input_line();

        if (n < 0) break;
        if (n == 0) continue;

        /* Check for quit */
        if (input_buf[0] == 'q') break;

        result = eval_expr(input_buf);

        tu_print_str("= ");
        /* Print result digit by digit */
        if (result < 0) {
            tu_putchar('-');
            result = 0 - result;
        }
        if (result == 0) {
            tu_putchar('0');
        } else {
            char digits[16];
            int di;
            int dr;
            di = 0;
            dr = result;
            while (dr > 0) {
                digits[di] = (dr % 10) + '0';
                dr = dr / 10;
                di = di + 1;
            }
            while (di > 0) {
                di = di - 1;
                tu_putchar(digits[di]);
            }
        }
        tu_putchar('\n');
    }

    tu_print_str("=====================================\n");
    free(input_buf);
    return tu_exit(0);
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    int i;
    int result;
    int verbose;
    int interactive;
    char *expression;
    char *model_file;

    init_neural();
    init_tooluse();
    onnx_loaded = 0;

    expression = 0;
    model_file = 0;
    verbose = 0;
    interactive = 0;

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
        } else if (argv[i][0] == '-' && argv[i][1] == 'i') {
            interactive = 1;
        } else if (argv[i][0] == '-' && argv[i][1] == 't') {
            tool_use_enabled = 1;
        }
        i = i + 1;
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

    /* Interactive mode */
    if (interactive) {
        return run_interactive();
    }

    /* Expression mode */
    if (expression == 0) {
        printf("Usage: %s [options] -e 'expression'\n", argv[0]);
        printf("Options:\n");
        printf("  -m model.onnx   Load ONNX model for arithmetic\n");
        printf("  -i              Interactive mode\n");
        printf("  -t              Enable tool-use protocol\n");
        printf("  -v              Verbose output\n");
        printf("\nExamples:\n");
        printf("  %s -e '6 * 7'                 # Built-in neural ops\n", argv[0]);
        printf("  %s -m model.onnx -e '6 * 7'   # Use ONNX model\n", argv[0]);
        printf("  %s -i -t                       # Interactive with tool-use\n", argv[0]);
        free(exp_tbl);
        return 1;
    }

    result = eval_expr(expression);

    if (verbose) {
        printf("Result: %d\n", result);
    }

    free(exp_tbl);
    return result & 255;
}
