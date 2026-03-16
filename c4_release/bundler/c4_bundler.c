/*
 * C4 Neural Bundler - C4 Compatible
 *
 * Bundles:
 * - Model weights (.c4onnx)
 * - ONNX runtime
 * - Target C program
 *
 * Into a single C4-compatible C file.
 *
 * Usage:
 *   ./c4_bundler model.c4onnx program.c > output.c
 *   gcc -o output output.c  (or: ./c4 output.c)
 *
 * Build:
 *   gcc -o c4_bundler c4_bundler.c
 */

/* C4-compatible function declarations */
int open(char *path, int mode);
int read(int fd, char *buf, int n);
int close(int fd);
char *malloc(int size);
int printf(char *fmt, ...);
int putchar(int c);
int exit(int code);

/* Globals */
char *model_data;
int model_len;
char *prog_data;
int prog_len;
char hex_chars[17];

int init_hex() {
    hex_chars[0] = '0'; hex_chars[1] = '1'; hex_chars[2] = '2'; hex_chars[3] = '3';
    hex_chars[4] = '4'; hex_chars[5] = '5'; hex_chars[6] = '6'; hex_chars[7] = '7';
    hex_chars[8] = '8'; hex_chars[9] = '9'; hex_chars[10] = 'a'; hex_chars[11] = 'b';
    hex_chars[12] = 'c'; hex_chars[13] = 'd'; hex_chars[14] = 'e'; hex_chars[15] = 'f';
    return 0;
}

int read_file(char *path, char **out_data, int *out_len) {
    int fd;
    int n;
    int total;
    char tmp[1024];
    char *buf;
    char *newbuf;
    int cap;
    int i;

    fd = open(path, 0);
    if (fd < 0) {
        printf("/* Error: cannot open %s */\n", path);
        return -1;
    }

    cap = 8192;
    buf = malloc(cap);
    total = 0;

    n = read(fd, tmp, 1024);
    while (n > 0) {
        if (total + n > cap) {
            cap = cap * 2;
            newbuf = malloc(cap);
            i = 0;
            while (i < total) {
                newbuf[i] = buf[i];
                i = i + 1;
            }
            buf = newbuf;
        }
        i = 0;
        while (i < n) {
            buf[total + i] = tmp[i];
            i = i + 1;
        }
        total = total + n;
        n = read(fd, tmp, 1024);
    }

    close(fd);
    buf[total] = 0;
    *out_data = buf;
    *out_len = total;
    return 0;
}

int print_hex_byte(int b) {
    int hi;
    int lo;
    b = b & 255;
    hi = (b / 16) & 15;
    lo = b & 15;
    putchar('0');
    putchar('x');
    putchar(hex_chars[hi]);
    putchar(hex_chars[lo]);
    return 0;
}

int print_model_array() {
    int i;

    printf("char embedded_model[] = {\n");

    i = 0;
    while (i < model_len) {
        if ((i & 15) == 0) printf("    ");
        print_hex_byte(model_data[i]);
        if (i + 1 < model_len) putchar(',');
        if ((i & 15) == 15 || i + 1 == model_len) putchar('\n');
        i = i + 1;
    }

    printf("};\n");
    printf("int embedded_model_len;\n\n");
    return 0;
}

int print_runtime() {
    printf("/* ===== C4 ONNX RUNTIME ===== */\n\n");

    printf("int SCALE;\n");
    printf("int MAX_TENSORS;\n");
    printf("int MAX_NODES;\n");
    printf("int MAX_DIMS;\n");
    printf("int MAX_NAME;\n");
    printf("int MAX_IO;\n\n");

    printf("char *tensor_names;\n");
    printf("int *tensor_ndims;\n");
    printf("int *tensor_dims;\n");
    printf("int *tensor_size;\n");
    printf("int *tensor_data_base;\n");
    printf("int *tensor_data_pool;\n");
    printf("int tensor_data_ptr;\n");
    printf("int num_tensors;\n\n");

    printf("int *node_op;\n");
    printf("int *node_num_inputs;\n");
    printf("int *node_inputs;\n");
    printf("int *node_num_outputs;\n");
    printf("int *node_outputs;\n");
    printf("int num_nodes;\n\n");

    printf("int embed_ptr;\n\n");

    printf("int neural_init() {\n");
    printf("    SCALE = 65536;\n");
    printf("    MAX_TENSORS = 256;\n");
    printf("    MAX_NODES = 512;\n");
    printf("    MAX_DIMS = 8;\n");
    printf("    MAX_NAME = 64;\n");
    printf("    MAX_IO = 16;\n");
    printf("    embedded_model_len = %d;\n\n", model_len);

    printf("    tensor_names = malloc(MAX_TENSORS * MAX_NAME);\n");
    printf("    tensor_ndims = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_dims = malloc(MAX_TENSORS * MAX_DIMS * 4);\n");
    printf("    tensor_size = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_data_base = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_data_pool = malloc(500000 * 4);\n");
    printf("    tensor_data_ptr = 0;\n\n");

    printf("    node_op = malloc(MAX_NODES * 4);\n");
    printf("    node_num_inputs = malloc(MAX_NODES * 4);\n");
    printf("    node_inputs = malloc(MAX_NODES * MAX_IO * 4);\n");
    printf("    node_num_outputs = malloc(MAX_NODES * 4);\n");
    printf("    node_outputs = malloc(MAX_NODES * MAX_IO * 4);\n\n");

    printf("    num_tensors = 0;\n");
    printf("    num_nodes = 0;\n");
    printf("    embed_ptr = 0;\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int read_embed_int() {\n");
    printf("    int v;\n");
    printf("    v = (embedded_model[embed_ptr] & 255);\n");
    printf("    v = v | ((embedded_model[embed_ptr + 1] & 255) << 8);\n");
    printf("    v = v | ((embedded_model[embed_ptr + 2] & 255) << 16);\n");
    printf("    v = v | ((embedded_model[embed_ptr + 3] & 255) << 24);\n");
    printf("    embed_ptr = embed_ptr + 4;\n");
    printf("    return v;\n");
    printf("}\n\n");

    printf("int load_model() {\n");
    printf("    int magic; int version; int n_tensors; int n_nodes;\n");
    printf("    int i; int j; int name_len; int ndims; int dtype; int size;\n");
    printf("    char *name_ptr;\n\n");

    printf("    magic = read_embed_int();\n");
    printf("    if (magic != 1481526863) {\n");
    printf("        printf(\"Bad magic: %%d\\n\", magic);\n");
    printf("        return -1;\n");
    printf("    }\n\n");

    printf("    version = read_embed_int();\n");
    printf("    n_tensors = read_embed_int();\n");
    printf("    n_nodes = read_embed_int();\n\n");

    printf("    i = 0;\n");
    printf("    while (i < n_tensors) {\n");
    printf("        name_len = read_embed_int();\n");
    printf("        name_ptr = tensor_names + i * MAX_NAME;\n");
    printf("        j = 0;\n");
    printf("        while (j < name_len) {\n");
    printf("            name_ptr[j] = embedded_model[embed_ptr + j];\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        name_ptr[j] = 0;\n");
    printf("        embed_ptr = embed_ptr + name_len;\n\n");

    printf("        ndims = read_embed_int();\n");
    printf("        tensor_ndims[i] = ndims;\n");
    printf("        j = 0;\n");
    printf("        while (j < ndims) {\n");
    printf("            tensor_dims[i * MAX_DIMS + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        dtype = read_embed_int();\n");
    printf("        size = read_embed_int();\n");
    printf("        tensor_size[i] = size;\n");
    printf("        tensor_data_base[i] = tensor_data_ptr;\n\n");

    printf("        j = 0;\n");
    printf("        while (j < size) {\n");
    printf("            tensor_data_pool[tensor_data_ptr] = read_embed_int();\n");
    printf("            tensor_data_ptr = tensor_data_ptr + 1;\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_tensors = n_tensors;\n\n");

    printf("    i = 0;\n");
    printf("    while (i < n_nodes) {\n");
    printf("        node_op[i] = read_embed_int();\n");
    printf("        node_num_inputs[i] = read_embed_int();\n");
    printf("        j = 0;\n");
    printf("        while (j < node_num_inputs[i]) {\n");
    printf("            node_inputs[i * MAX_IO + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        node_num_outputs[i] = read_embed_int();\n");
    printf("        j = 0;\n");
    printf("        while (j < node_num_outputs[i]) {\n");
    printf("            node_outputs[i * MAX_IO + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_nodes = n_nodes;\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int fp_mul(int a, int b) {\n");
    printf("    return (a / 256) * (b / 256);\n");
    printf("}\n\n");

    printf("int *get_tensor_data(int idx) {\n");
    printf("    return tensor_data_pool + tensor_data_base[idx];\n");
    printf("}\n\n");

    printf("int OP_ADD; int OP_MUL; int OP_SUB; int OP_MATMUL; int OP_RELU; int OP_IDENTITY;\n\n");

    printf("int init_ops() {\n");
    printf("    OP_ADD = 1; OP_MUL = 2; OP_SUB = 3; OP_MATMUL = 5; OP_RELU = 6; OP_IDENTITY = 13;\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int exec_node(int ni) {\n");
    printf("    int op; int in0; int in1; int out0;\n");
    printf("    int *a; int *b; int *c;\n");
    printf("    int i; int size;\n");
    printf("    int m; int k; int n; int mi; int ki; int nj; int sum;\n\n");

    printf("    op = node_op[ni];\n");
    printf("    in0 = node_inputs[ni * MAX_IO];\n");
    printf("    in1 = node_inputs[ni * MAX_IO + 1];\n");
    printf("    out0 = node_outputs[ni * MAX_IO];\n\n");

    printf("    a = get_tensor_data(in0);\n");
    printf("    b = get_tensor_data(in1);\n");
    printf("    c = get_tensor_data(out0);\n");
    printf("    size = tensor_size[out0];\n\n");

    printf("    if (op == OP_ADD) {\n");
    printf("        i = 0; while (i < size) { c[i] = a[i] + b[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    if (op == OP_MUL) {\n");
    printf("        i = 0; while (i < size) { c[i] = fp_mul(a[i], b[i]); i = i + 1; }\n");
    printf("    }\n");
    printf("    if (op == OP_SUB) {\n");
    printf("        i = 0; while (i < size) { c[i] = a[i] - b[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    if (op == OP_RELU) {\n");
    printf("        i = 0; while (i < size) { c[i] = (a[i] > 0) ? a[i] : 0; i = i + 1; }\n");
    printf("    }\n");
    printf("    if (op == OP_IDENTITY) {\n");
    printf("        i = 0; while (i < size) { c[i] = a[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    if (op == OP_MATMUL) {\n");
    printf("        m = tensor_dims[in0 * MAX_DIMS];\n");
    printf("        k = tensor_dims[in0 * MAX_DIMS + 1];\n");
    printf("        n = tensor_dims[in1 * MAX_DIMS + 1];\n");
    printf("        if (tensor_ndims[in0] == 1) { m = 1; k = tensor_dims[in0 * MAX_DIMS]; }\n");
    printf("        mi = 0;\n");
    printf("        while (mi < m) {\n");
    printf("            nj = 0;\n");
    printf("            while (nj < n) {\n");
    printf("                sum = 0; ki = 0;\n");
    printf("                while (ki < k) {\n");
    printf("                    sum = sum + fp_mul(a[mi * k + ki], b[ki * n + nj]);\n");
    printf("                    ki = ki + 1;\n");
    printf("                }\n");
    printf("                c[mi * n + nj] = sum;\n");
    printf("                nj = nj + 1;\n");
    printf("            }\n");
    printf("            mi = mi + 1;\n");
    printf("        }\n");
    printf("    }\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int run_model() {\n");
    printf("    int i; i = 0;\n");
    printf("    while (i < num_nodes) { exec_node(i); i = i + 1; }\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int find_tensor(char *name) {\n");
    printf("    int i; int j; int match; char *tn;\n");
    printf("    i = 0;\n");
    printf("    while (i < num_tensors) {\n");
    printf("        tn = tensor_names + i * MAX_NAME;\n");
    printf("        match = 1; j = 0;\n");
    printf("        while (name[j] && tn[j]) {\n");
    printf("            if (name[j] != tn[j]) match = 0;\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        if (match && name[j] == tn[j]) return i;\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    return -1;\n");
    printf("}\n\n");

    printf("int neural_infer(int *input, int in_sz, int *output, int out_sz) {\n");
    printf("    int *in_data; int *out_data; int i;\n");
    printf("    int in_idx; int out_idx;\n");
    printf("    in_idx = find_tensor(\"x\");\n");
    printf("    out_idx = find_tensor(\"y\");\n");
    printf("    if (in_idx < 0) in_idx = num_tensors - 2;\n");
    printf("    if (out_idx < 0) out_idx = num_tensors - 1;\n");
    printf("    in_data = get_tensor_data(in_idx);\n");
    printf("    i = 0;\n");
    printf("    while (i < in_sz && i < tensor_size[in_idx]) {\n");
    printf("        in_data[i] = input[i]; i = i + 1;\n");
    printf("    }\n");
    printf("    run_model();\n");
    printf("    out_data = get_tensor_data(out_idx);\n");
    printf("    i = 0;\n");
    printf("    while (i < out_sz && i < tensor_size[out_idx]) {\n");
    printf("        output[i] = out_data[i]; i = i + 1;\n");
    printf("    }\n");
    printf("    return 0;\n");
    printf("}\n\n");

    printf("int neural_setup() {\n");
    printf("    neural_init();\n");
    printf("    init_ops();\n");
    printf("    return load_model();\n");
    printf("}\n\n");

    return 0;
}

int print_program() {
    printf("/* ===== USER PROGRAM ===== */\n\n");
    printf("%s", prog_data);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("/* Usage: c4_bundler model.c4onnx program.c > output.c */\n");
        return 1;
    }

    init_hex();

    if (read_file(argv[1], &model_data, &model_len) != 0) {
        return 1;
    }

    if (read_file(argv[2], &prog_data, &prog_len) != 0) {
        return 1;
    }

    printf("/*\n");
    printf(" * C4 Neural Bundle\n");
    printf(" *\n");
    printf(" * Model: %d bytes\n", model_len);
    printf(" * Program: %d bytes\n", prog_len);
    printf(" *\n");
    printf(" * Build: gcc -o output output.c\n");
    printf(" * Or:    ./c4 output.c\n");
    printf(" */\n\n");

    /* C4-style declarations */
    printf("/* Function declarations */\n");
    printf("int printf(char *fmt, ...);\n");
    printf("int putchar(int c);\n");
    printf("char *malloc(int size);\n\n");

    print_model_array();
    print_runtime();
    print_program();

    return 0;
}
