/*
 * Neural C4 Bundle Generator
 *
 * Creates a standalone C file that combines:
 * - Embedded neural model (ONNX weights)
 * - ONNX runtime (C4 compatible)
 * - C4 compiler/interpreter
 *
 * The output can take a C4 C source file and execute it
 * with access to neural model inference functions.
 *
 * Usage:
 *   ./neural_c4_bundle model.c4onnx > output.c
 *   gcc -o neural_c4 output.c
 *   ./neural_c4 program.c
 *
 * Build:
 *   gcc -o neural_c4_bundle neural_c4_bundle.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *read_file(char *path, int *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    int len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = 0;
    fclose(f);
    *out_len = len;
    return buf;
}

void print_hex(int b) {
    char hex[] = "0123456789abcdef";
    b = b & 255;
    printf("0x%c%c", hex[(b >> 4) & 15], hex[b & 15]);
}

void print_byte_array(char *name, unsigned char *data, int len) {
    printf("static unsigned char %s[] = {\n", name);
    int i = 0;
    while (i < len) {
        if ((i % 16) == 0) printf("    ");
        print_hex(data[i]);
        if (i + 1 < len) printf(",");
        if ((i % 16) == 15 || i + 1 == len) printf("\n");
        i++;
    }
    printf("};\n");
    printf("static int %s_len = %d;\n\n", name, len);
}

void print_c4_compatible_runtime(void) {
    printf("/* ===== C4-COMPATIBLE ONNX RUNTIME ===== */\n\n");

    /* All C4 compatible - no sizeof, simple types */
    printf("int SCALE;\n");
    printf("int FP_SHIFT;\n\n");

    printf("int MAX_TENSORS;\n");
    printf("int MAX_NODES;\n");
    printf("int MAX_DIMS;\n");
    printf("int MAX_NAME;\n");
    printf("int MAX_IO;\n\n");

    printf("char *tensor_names;  /* tensor_names[i * MAX_NAME] */\n");
    printf("int *tensor_ndims;\n");
    printf("int *tensor_dims;    /* tensor_dims[i * MAX_DIMS + j] */\n");
    printf("int *tensor_size;\n");
    printf("int *tensor_data_base;  /* offset into tensor_data_pool */\n");
    printf("int *tensor_data_pool;  /* all tensor data */\n");
    printf("int tensor_data_ptr;\n");
    printf("int num_tensors;\n\n");

    printf("int *node_op;\n");
    printf("int *node_num_inputs;\n");
    printf("int *node_inputs;    /* node_inputs[i * MAX_IO + j] */\n");
    printf("int *node_num_outputs;\n");
    printf("int *node_outputs;   /* node_outputs[i * MAX_IO + j] */\n");
    printf("int num_nodes;\n\n");

    printf("int neural_init() {\n");
    printf("    SCALE = 65536;\n");
    printf("    FP_SHIFT = 16;\n");
    printf("    MAX_TENSORS = 256;\n");
    printf("    MAX_NODES = 512;\n");
    printf("    MAX_DIMS = 8;\n");
    printf("    MAX_NAME = 64;\n");
    printf("    MAX_IO = 16;\n\n");

    printf("    tensor_names = malloc(MAX_TENSORS * MAX_NAME);\n");
    printf("    tensor_ndims = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_dims = malloc(MAX_TENSORS * MAX_DIMS * 4);\n");
    printf("    tensor_size = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_data_base = malloc(MAX_TENSORS * 4);\n");
    printf("    tensor_data_pool = malloc(1000000 * 4);  /* 4MB for tensor data */\n");
    printf("    tensor_data_ptr = 0;\n\n");

    printf("    node_op = malloc(MAX_NODES * 4);\n");
    printf("    node_num_inputs = malloc(MAX_NODES * 4);\n");
    printf("    node_inputs = malloc(MAX_NODES * MAX_IO * 4);\n");
    printf("    node_num_outputs = malloc(MAX_NODES * 4);\n");
    printf("    node_outputs = malloc(MAX_NODES * MAX_IO * 4);\n\n");

    printf("    num_tensors = 0;\n");
    printf("    num_nodes = 0;\n");
    printf("    return 0;\n");
    printf("}\n\n");

    /* Read int from embedded data - C4 compatible */
    printf("int embed_ptr;\n\n");

    printf("int read_embed_int() {\n");
    printf("    int v;\n");
    printf("    v = embedded_model[embed_ptr];\n");
    printf("    v = v | (embedded_model[embed_ptr + 1] << 8);\n");
    printf("    v = v | (embedded_model[embed_ptr + 2] << 16);\n");
    printf("    v = v | (embedded_model[embed_ptr + 3] << 24);\n");
    printf("    embed_ptr = embed_ptr + 4;\n");
    printf("    return v;\n");
    printf("}\n\n");

    printf("int load_embedded_model() {\n");
    printf("    int magic;\n");
    printf("    int version;\n");
    printf("    int n_tensors;\n");
    printf("    int n_nodes;\n");
    printf("    int i;\n");
    printf("    int j;\n");
    printf("    int name_len;\n");
    printf("    int ndims;\n");
    printf("    int data_type;\n");
    printf("    int size;\n");
    printf("    char *name_ptr;\n\n");

    printf("    embed_ptr = 0;\n");
    printf("    magic = read_embed_int();\n");
    printf("    if (magic != 1481526863) {\n");  /* 0x584E4E4F */
    printf("        printf(\"Error: invalid magic %%d\\n\", magic);\n");
    printf("        return -1;\n");
    printf("    }\n\n");

    printf("    version = read_embed_int();\n");
    printf("    n_tensors = read_embed_int();\n");
    printf("    n_nodes = read_embed_int();\n\n");

    printf("    /* Load tensors */\n");
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
    printf("        tensor_ndims[i] = ndims;\n\n");

    printf("        j = 0;\n");
    printf("        while (j < ndims) {\n");
    printf("            tensor_dims[i * MAX_DIMS + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        data_type = read_embed_int();\n");
    printf("        size = read_embed_int();\n");
    printf("        tensor_size[i] = size;\n");
    printf("        tensor_data_base[i] = tensor_data_ptr;\n\n");

    printf("        j = 0;\n");
    printf("        while (j < size) {\n");
    printf("            tensor_data_pool[tensor_data_ptr] = read_embed_int();\n");
    printf("            tensor_data_ptr = tensor_data_ptr + 1;\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_tensors = n_tensors;\n\n");

    printf("    /* Load nodes */\n");
    printf("    i = 0;\n");
    printf("    while (i < n_nodes) {\n");
    printf("        node_op[i] = read_embed_int();\n");
    printf("        node_num_inputs[i] = read_embed_int();\n\n");

    printf("        j = 0;\n");
    printf("        while (j < node_num_inputs[i]) {\n");
    printf("            node_inputs[i * MAX_IO + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        node_num_outputs[i] = read_embed_int();\n\n");

    printf("        j = 0;\n");
    printf("        while (j < node_num_outputs[i]) {\n");
    printf("            node_outputs[i * MAX_IO + j] = read_embed_int();\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_nodes = n_nodes;\n\n");

    printf("    return 0;\n");
    printf("}\n\n");

    /* Fixed-point ops */
    printf("int fp_mul(int a, int b) {\n");
    printf("    return (a / 256) * (b / 256);  /* Approximate for C4 */\n");
    printf("}\n\n");

    printf("int fp_div(int a, int b) {\n");
    printf("    if (b == 0) return 0;\n");
    printf("    return (a * 256) / (b / 256);\n");
    printf("}\n\n");

    /* Operation enum values */
    printf("int OP_GEMM; int OP_ADD; int OP_MUL; int OP_SUB; int OP_DIV;\n");
    printf("int OP_MATMUL; int OP_RELU; int OP_SIGMOID; int OP_SILU; int OP_TANH;\n");
    printf("int OP_SOFTMAX; int OP_RESHAPE; int OP_TRANSPOSE; int OP_IDENTITY;\n\n");

    printf("int init_ops() {\n");
    printf("    OP_GEMM = 0; OP_ADD = 1; OP_MUL = 2; OP_SUB = 3; OP_DIV = 4;\n");
    printf("    OP_MATMUL = 5; OP_RELU = 6; OP_SIGMOID = 7; OP_SILU = 8; OP_TANH = 9;\n");
    printf("    OP_SOFTMAX = 10; OP_RESHAPE = 11; OP_TRANSPOSE = 12; OP_IDENTITY = 13;\n");
    printf("    return 0;\n");
    printf("}\n\n");

    /* Get tensor data pointer */
    printf("int *get_tensor_data(int idx) {\n");
    printf("    return tensor_data_pool + tensor_data_base[idx];\n");
    printf("}\n\n");

    /* Execute single node */
    printf("int exec_node(int node_idx) {\n");
    printf("    int op;\n");
    printf("    int in0; int in1; int out0;\n");
    printf("    int i; int size;\n");
    printf("    int *a; int *b; int *c;\n");
    printf("    int m; int k; int n; int mi; int ki; int ni; int sum;\n\n");

    printf("    op = node_op[node_idx];\n");
    printf("    in0 = node_inputs[node_idx * MAX_IO];\n");
    printf("    in1 = node_inputs[node_idx * MAX_IO + 1];\n");
    printf("    out0 = node_outputs[node_idx * MAX_IO];\n\n");

    printf("    a = get_tensor_data(in0);\n");
    printf("    b = get_tensor_data(in1);\n");
    printf("    c = get_tensor_data(out0);\n");
    printf("    size = tensor_size[out0];\n\n");

    printf("    if (op == OP_ADD) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) { c[i] = a[i] + b[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    else if (op == OP_MUL) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) { c[i] = fp_mul(a[i], b[i]); i = i + 1; }\n");
    printf("    }\n");
    printf("    else if (op == OP_SUB) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) { c[i] = a[i] - b[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    else if (op == OP_RELU) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) { c[i] = (a[i] > 0) ? a[i] : 0; i = i + 1; }\n");
    printf("    }\n");
    printf("    else if (op == OP_IDENTITY) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) { c[i] = a[i]; i = i + 1; }\n");
    printf("    }\n");
    printf("    else if (op == OP_MATMUL) {\n");
    printf("        m = tensor_dims[in0 * MAX_DIMS];\n");
    printf("        k = tensor_dims[in0 * MAX_DIMS + 1];\n");
    printf("        n = tensor_dims[in1 * MAX_DIMS + 1];\n");
    printf("        if (tensor_ndims[in0] == 1) { m = 1; k = tensor_dims[in0 * MAX_DIMS]; }\n");
    printf("        mi = 0;\n");
    printf("        while (mi < m) {\n");
    printf("            ni = 0;\n");
    printf("            while (ni < n) {\n");
    printf("                sum = 0;\n");
    printf("                ki = 0;\n");
    printf("                while (ki < k) {\n");
    printf("                    sum = sum + fp_mul(a[mi * k + ki], b[ki * n + ni]);\n");
    printf("                    ki = ki + 1;\n");
    printf("                }\n");
    printf("                c[mi * n + ni] = sum;\n");
    printf("                ni = ni + 1;\n");
    printf("            }\n");
    printf("            mi = mi + 1;\n");
    printf("        }\n");
    printf("    }\n\n");

    printf("    return 0;\n");
    printf("}\n\n");

    /* Run model */
    printf("int run_neural_model() {\n");
    printf("    int i;\n");
    printf("    i = 0;\n");
    printf("    while (i < num_nodes) {\n");
    printf("        exec_node(i);\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    return 0;\n");
    printf("}\n\n");

    /* Find tensor by name - for user programs */
    printf("int find_tensor(char *name) {\n");
    printf("    int i; int j; int match; char *tname;\n");
    printf("    i = 0;\n");
    printf("    while (i < num_tensors) {\n");
    printf("        tname = tensor_names + i * MAX_NAME;\n");
    printf("        match = 1; j = 0;\n");
    printf("        while (name[j] && tname[j]) {\n");
    printf("            if (name[j] != tname[j]) match = 0;\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        if (match && name[j] == tname[j]) return i;\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    return -1;\n");
    printf("}\n\n");

    /* User-callable inference function */
    printf("int neural_infer(int *input, int input_size, int *output, int output_size) {\n");
    printf("    int *in_data; int *out_data; int i;\n");
    printf("    int in_idx; int out_idx;\n\n");

    printf("    in_idx = find_tensor(\"x\");\n");
    printf("    out_idx = find_tensor(\"y\");\n");
    printf("    if (in_idx < 0) in_idx = num_tensors - 2;\n");
    printf("    if (out_idx < 0) out_idx = num_tensors - 1;\n\n");

    printf("    in_data = get_tensor_data(in_idx);\n");
    printf("    i = 0;\n");
    printf("    while (i < input_size && i < tensor_size[in_idx]) {\n");
    printf("        in_data[i] = input[i];\n");
    printf("        i = i + 1;\n");
    printf("    }\n\n");

    printf("    run_neural_model();\n\n");

    printf("    out_data = get_tensor_data(out_idx);\n");
    printf("    i = 0;\n");
    printf("    while (i < output_size && i < tensor_size[out_idx]) {\n");
    printf("        output[i] = out_data[i];\n");
    printf("        i = i + 1;\n");
    printf("    }\n\n");

    printf("    return 0;\n");
    printf("}\n\n");
}

void print_c4_compiler(void) {
    printf("/* ===== EMBEDDED C4 COMPILER ===== */\n\n");

    /* The C4 source with minor modifications for embedding */
    printf("char *c4_p, *c4_lp, *c4_data;\n");
    printf("int *c4_e, *c4_le, *c4_id, *c4_sym;\n");
    printf("int c4_tk, c4_ival, c4_ty, c4_loc, c4_line, c4_src, c4_debug;\n");
    printf("int c4_poolsz;\n\n");

    /* Tokens */
    printf("enum {\n");
    printf("  C4_Num = 128, C4_Fun, C4_Sys, C4_Glo, C4_Loc, C4_Id,\n");
    printf("  C4_Char, C4_Else, C4_Enum, C4_If, C4_Int, C4_Return, C4_Sizeof, C4_While,\n");
    printf("  C4_Assign, C4_Cond, C4_Lor, C4_Lan, C4_Or, C4_Xor, C4_And,\n");
    printf("  C4_Eq, C4_Ne, C4_Lt, C4_Gt, C4_Le, C4_Ge, C4_Shl, C4_Shr,\n");
    printf("  C4_Add, C4_Sub, C4_Mul, C4_Div, C4_Mod, C4_Inc, C4_Dec, C4_Brak\n");
    printf("};\n\n");

    /* Opcodes */
    printf("enum {\n");
    printf("  C4_LEA, C4_IMM, C4_JMP, C4_JSR, C4_BZ, C4_BNZ, C4_ENT, C4_ADJ, C4_LEV,\n");
    printf("  C4_LI, C4_LC, C4_SI, C4_SC, C4_PSH,\n");
    printf("  C4_OR, C4_XOR, C4_AND, C4_EQ, C4_NE, C4_LT, C4_GT, C4_LE, C4_GE,\n");
    printf("  C4_SHL, C4_SHR, C4_ADD, C4_SUB, C4_MUL, C4_DIV, C4_MOD,\n");
    printf("  C4_OPEN, C4_READ, C4_CLOS, C4_PRTF, C4_MALC, C4_FREE, C4_MSET, C4_MCMP, C4_EXIT,\n");
    printf("  C4_NINF  /* Neural inference syscall */\n");
    printf("};\n\n");

    /* Identifier fields */
    printf("enum { C4_Tk, C4_Hash, C4_Name, C4_Class, C4_Type, C4_Val, C4_HClass, C4_HType, C4_HVal, C4_Idsz };\n\n");

    /* Types */
    printf("enum { C4_CHAR, C4_INT, C4_PTR };\n\n");

    /* Lexer - simplified for embedding */
    printf("int c4_next() {\n");
    printf("    char *pp;\n");
    printf("    while ((c4_tk = *c4_p)) {\n");
    printf("        ++c4_p;\n");
    printf("        if (c4_tk == '\\n') { ++c4_line; }\n");
    printf("        else if (c4_tk == '#') { while (*c4_p != 0 && *c4_p != '\\n') ++c4_p; }\n");
    printf("        else if ((c4_tk >= 'a' && c4_tk <= 'z') || (c4_tk >= 'A' && c4_tk <= 'Z') || c4_tk == '_') {\n");
    printf("            pp = c4_p - 1;\n");
    printf("            while ((*c4_p >= 'a' && *c4_p <= 'z') || (*c4_p >= 'A' && *c4_p <= 'Z') ||\n");
    printf("                   (*c4_p >= '0' && *c4_p <= '9') || *c4_p == '_') ++c4_p;\n");
    printf("            c4_id = c4_sym;\n");
    printf("            while (c4_id[C4_Tk]) {\n");
    printf("                if (!memcmp((char *)c4_id[C4_Name], pp, c4_p - pp) && !((char *)c4_id[C4_Name])[c4_p - pp]) {\n");
    printf("                    c4_tk = c4_id[C4_Tk]; return 0;\n");
    printf("                }\n");
    printf("                c4_id = c4_id + C4_Idsz;\n");
    printf("            }\n");
    printf("            c4_id[C4_Name] = (int)pp;\n");
    printf("            c4_id[C4_Hash] = 0;\n");
    printf("            c4_tk = c4_id[C4_Tk] = C4_Id;\n");
    printf("            return 0;\n");
    printf("        }\n");
    printf("        else if (c4_tk >= '0' && c4_tk <= '9') {\n");
    printf("            c4_ival = c4_tk - '0';\n");
    printf("            if (c4_ival) { while (*c4_p >= '0' && *c4_p <= '9') c4_ival = c4_ival * 10 + *c4_p++ - '0'; }\n");
    printf("            else if (*c4_p == 'x' || *c4_p == 'X') {\n");
    printf("                ++c4_p;\n");
    printf("                while ((*c4_p >= '0' && *c4_p <= '9') || (*c4_p >= 'a' && *c4_p <= 'f') || (*c4_p >= 'A' && *c4_p <= 'F'))\n");
    printf("                    c4_ival = c4_ival * 16 + (*c4_p++ & 15) + (*c4_p > '9' ? 9 : 0);\n");
    printf("            }\n");
    printf("            c4_tk = C4_Num;\n");
    printf("            return 0;\n");
    printf("        }\n");
    printf("        else if (c4_tk == '/') {\n");
    printf("            if (*c4_p == '/') { ++c4_p; while (*c4_p != 0 && *c4_p != '\\n') ++c4_p; }\n");
    printf("            else { c4_tk = C4_Div; return 0; }\n");
    printf("        }\n");
    printf("        else if (c4_tk == '\\'' || c4_tk == '\"') {\n");
    printf("            pp = c4_data;\n");
    printf("            while (*c4_p != 0 && *c4_p != c4_tk) {\n");
    printf("                if ((c4_ival = *c4_p++) == '\\\\') {\n");
    printf("                    if ((c4_ival = *c4_p++) == 'n') c4_ival = '\\n';\n");
    printf("                }\n");
    printf("                if (c4_tk == '\"') *c4_data++ = c4_ival;\n");
    printf("            }\n");
    printf("            ++c4_p;\n");
    printf("            if (c4_tk == '\"') c4_ival = (int)pp; else c4_tk = C4_Num;\n");
    printf("            return 0;\n");
    printf("        }\n");
    printf("        else if (c4_tk == '=') { if (*c4_p == '=') { ++c4_p; c4_tk = C4_Eq; } else c4_tk = C4_Assign; return 0; }\n");
    printf("        else if (c4_tk == '+') { if (*c4_p == '+') { ++c4_p; c4_tk = C4_Inc; } else c4_tk = C4_Add; return 0; }\n");
    printf("        else if (c4_tk == '-') { if (*c4_p == '-') { ++c4_p; c4_tk = C4_Dec; } else c4_tk = C4_Sub; return 0; }\n");
    printf("        else if (c4_tk == '!') { if (*c4_p == '=') { ++c4_p; c4_tk = C4_Ne; } return 0; }\n");
    printf("        else if (c4_tk == '<') { if (*c4_p == '=') { ++c4_p; c4_tk = C4_Le; } else if (*c4_p == '<') { ++c4_p; c4_tk = C4_Shl; } else c4_tk = C4_Lt; return 0; }\n");
    printf("        else if (c4_tk == '>') { if (*c4_p == '=') { ++c4_p; c4_tk = C4_Ge; } else if (*c4_p == '>') { ++c4_p; c4_tk = C4_Shr; } else c4_tk = C4_Gt; return 0; }\n");
    printf("        else if (c4_tk == '|') { if (*c4_p == '|') { ++c4_p; c4_tk = C4_Lor; } else c4_tk = C4_Or; return 0; }\n");
    printf("        else if (c4_tk == '&') { if (*c4_p == '&') { ++c4_p; c4_tk = C4_Lan; } else c4_tk = C4_And; return 0; }\n");
    printf("        else if (c4_tk == '^') { c4_tk = C4_Xor; return 0; }\n");
    printf("        else if (c4_tk == '%%') { c4_tk = C4_Mod; return 0; }\n");
    printf("        else if (c4_tk == '*') { c4_tk = C4_Mul; return 0; }\n");
    printf("        else if (c4_tk == '[') { c4_tk = C4_Brak; return 0; }\n");
    printf("        else if (c4_tk == '?') { c4_tk = C4_Cond; return 0; }\n");
    printf("        else if (c4_tk != ' ' && c4_tk != '\\t' && c4_tk != '\\r') return 0;\n");
    printf("    }\n");
    printf("    return 0;\n");
    printf("}\n\n");

    /* Simple expression parser - outputs to c4_e */
    printf("int c4_expr(int lev);\n");
    printf("int c4_stmt();\n\n");

    /* This is a simplified C4 - full implementation would be much longer */
    printf("/* Note: Full C4 compiler implementation would go here */\n");
    printf("/* For brevity, this is a stub that can be expanded */\n\n");

    printf("int c4_run_source(char *src) {\n");
    printf("    printf(\"C4 interpreter stub - source loaded\\n\");\n");
    printf("    printf(\"Neural model has %%d tensors, %%d nodes\\n\", num_tensors, num_nodes);\n");
    printf("    return 0;\n");
    printf("}\n\n");
}

void print_main(void) {
    printf("/* ===== MAIN ENTRY POINT ===== */\n\n");

    printf("int main(int argc, char **argv) {\n");
    printf("    char *src;\n");
    printf("    int src_len;\n");
    printf("    int fd;\n\n");

    printf("    /* Initialize neural model */\n");
    printf("    neural_init();\n");
    printf("    init_ops();\n\n");

    printf("    if (load_embedded_model() != 0) {\n");
    printf("        printf(\"Failed to load model\\n\");\n");
    printf("        return 1;\n");
    printf("    }\n\n");

    printf("    printf(\"Neural model loaded: %%d tensors, %%d nodes\\n\", num_tensors, num_nodes);\n\n");

    printf("    if (argc < 2) {\n");
    printf("        /* No source file - run test */\n");
    printf("        int input[4];\n");
    printf("        int output[4];\n");
    printf("        int i;\n\n");

    printf("        printf(\"No source file provided. Running test inference...\\n\");\n\n");

    printf("        input[0] = 1 * SCALE;\n");
    printf("        input[1] = 2 * SCALE;\n");
    printf("        input[2] = 3 * SCALE;\n");
    printf("        input[3] = 4 * SCALE;\n\n");

    printf("        neural_infer(input, 4, output, 4);\n\n");

    printf("        printf(\"Input: \");\n");
    printf("        i = 0;\n");
    printf("        while (i < 4) { printf(\"%%d \", input[i] / SCALE); i = i + 1; }\n");
    printf("        printf(\"\\n\");\n\n");

    printf("        printf(\"Output: \");\n");
    printf("        i = 0;\n");
    printf("        while (i < 4) { printf(\"%%d \", output[i] / SCALE); i = i + 1; }\n");
    printf("        printf(\"\\n\");\n\n");

    printf("        return 0;\n");
    printf("    }\n\n");

    printf("    /* Load and run source file */\n");
    printf("    fd = open(argv[1], 0);\n");
    printf("    if (fd < 0) {\n");
    printf("        printf(\"Cannot open %%s\\n\", argv[1]);\n");
    printf("        return 1;\n");
    printf("    }\n\n");

    printf("    src = malloc(100000);\n");
    printf("    src_len = read(fd, src, 99999);\n");
    printf("    src[src_len] = 0;\n");
    printf("    close(fd);\n\n");

    printf("    return c4_run_source(src);\n");
    printf("}\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.c4onnx > output.c\n", argv[0]);
        return 1;
    }

    int model_len;
    unsigned char *model = (unsigned char *)read_file(argv[1], &model_len);
    if (!model) return 1;

    printf("/*\n");
    printf(" * Neural C4 Bundle - Generated by neural_c4_bundle\n");
    printf(" *\n");
    printf(" * Contains:\n");
    printf(" *   - Embedded ONNX model (%d bytes)\n", model_len);
    printf(" *   - C4-compatible ONNX runtime\n");
    printf(" *   - C4 compiler stub (can be expanded)\n");
    printf(" *\n");
    printf(" * Build: gcc -o neural_c4 output.c\n");
    printf(" * Run:   ./neural_c4 [program.c]\n");
    printf(" */\n\n");

    printf("#include <stdio.h>\n");
    printf("#include <stdlib.h>\n");
    printf("#include <string.h>\n");
    printf("#include <fcntl.h>\n");
    printf("#include <unistd.h>\n\n");

    /* Embed model */
    print_byte_array("embedded_model", model, model_len);

    /* Print C4-compatible runtime */
    print_c4_compatible_runtime();

    /* Print C4 compiler */
    print_c4_compiler();

    /* Print main */
    print_main();

    return 0;
}
