/*
 * ONNX Model Bundler
 *
 * Bundles a C4 ONNX model, the runtime, and user program into a single C file.
 *
 * Usage:
 *   ./onnx_bundler model.c4onnx [program.c] > output.c
 *
 * If no program is specified, generates a simple test harness.
 *
 * Build:
 *   gcc -o onnx_bundler onnx_bundler.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read entire file into buffer */
char *read_file(char *path, int *out_len) {
    FILE *f;
    char *buf;
    int len;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return 0;
    }

    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);

    buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = 0;

    fclose(f);
    *out_len = len;
    return buf;
}

/* Print byte as hex */
void print_hex(int b) {
    char hex[] = "0123456789abcdef";
    b = b & 255;
    printf("0x%c%c", hex[(b >> 4) & 15], hex[b & 15]);
}

/* Print C array of bytes */
void print_byte_array(char *name, unsigned char *data, int len) {
    int i;
    int per_line = 16;

    printf("static unsigned char %s[] = {\n", name);

    i = 0;
    while (i < len) {
        if ((i % per_line) == 0) {
            printf("    ");
        }

        print_hex(data[i]);

        if (i + 1 < len) {
            printf(",");
        }

        if ((i % per_line) == per_line - 1 || i + 1 == len) {
            printf("\n");
        }

        i = i + 1;
    }

    printf("};\n");
    printf("static int %s_len = %d;\n\n", name, len);
}

/* Print embedded ONNX runtime */
void print_runtime(void) {
    printf("/* ===== EMBEDDED ONNX RUNTIME ===== */\n\n");

    /* Fixed-point constants */
    printf("#define SCALE 65536\n");
    printf("#define FP_SHIFT 16\n\n");

    /* Limits */
    printf("#define MAX_TENSORS 256\n");
    printf("#define MAX_NODES 512\n");
    printf("#define MAX_DIMS 8\n");
    printf("#define MAX_NAME 64\n");
    printf("#define MAX_IO 16\n\n");

    /* Tensor storage */
    printf("static char tensor_names[MAX_TENSORS][MAX_NAME];\n");
    printf("static int tensor_ndims[MAX_TENSORS];\n");
    printf("static int tensor_dims[MAX_TENSORS][MAX_DIMS];\n");
    printf("static int tensor_size[MAX_TENSORS];\n");
    printf("static int *tensor_data[MAX_TENSORS];\n");
    printf("static int num_tensors;\n\n");

    /* Node storage */
    printf("static int node_op[MAX_NODES];\n");
    printf("static int node_num_inputs[MAX_NODES];\n");
    printf("static int node_inputs[MAX_NODES][MAX_IO];\n");
    printf("static int node_num_outputs[MAX_NODES];\n");
    printf("static int node_outputs[MAX_NODES][MAX_IO];\n");
    printf("static int num_nodes;\n\n");

    /* Read int from embedded data */
    printf("static int read_int_embed(unsigned char **p) {\n");
    printf("    unsigned char *d = *p;\n");
    printf("    int v = d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24);\n");
    printf("    *p = d + 4;\n");
    printf("    return v;\n");
    printf("}\n\n");

    /* Load model from embedded data */
    printf("static int load_embedded_model(void) {\n");
    printf("    unsigned char *p = embedded_model;\n");
    printf("    int magic, version, n_tensors, n_nodes;\n");
    printf("    int i, j, name_len, ndims, data_type, size;\n\n");

    printf("    magic = read_int_embed(&p);\n");
    printf("    if (magic != 0x584E4E4F) {\n");
    printf("        printf(\"Error: invalid magic\\n\");\n");
    printf("        return -1;\n");
    printf("    }\n\n");

    printf("    version = read_int_embed(&p);\n");
    printf("    n_tensors = read_int_embed(&p);\n");
    printf("    n_nodes = read_int_embed(&p);\n\n");

    printf("    /* Load tensors */\n");
    printf("    i = 0;\n");
    printf("    while (i < n_tensors) {\n");
    printf("        name_len = read_int_embed(&p);\n");
    printf("        j = 0;\n");
    printf("        while (j < name_len && j < MAX_NAME - 1) {\n");
    printf("            tensor_names[i][j] = p[j];\n");
    printf("            j = j + 1;\n");
    printf("        }\n");
    printf("        tensor_names[i][j] = 0;\n");
    printf("        p = p + name_len;\n\n");

    printf("        ndims = read_int_embed(&p);\n");
    printf("        tensor_ndims[i] = ndims;\n\n");

    printf("        j = 0;\n");
    printf("        while (j < ndims) {\n");
    printf("            tensor_dims[i][j] = read_int_embed(&p);\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        data_type = read_int_embed(&p);\n");
    printf("        size = read_int_embed(&p);\n");
    printf("        tensor_size[i] = size;\n\n");

    printf("        tensor_data[i] = (int *)malloc(size * sizeof(int));\n");
    printf("        j = 0;\n");
    printf("        while (j < size) {\n");
    printf("            tensor_data[i][j] = read_int_embed(&p);\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_tensors = n_tensors;\n\n");

    printf("    /* Load nodes */\n");
    printf("    i = 0;\n");
    printf("    while (i < n_nodes) {\n");
    printf("        node_op[i] = read_int_embed(&p);\n\n");

    printf("        node_num_inputs[i] = read_int_embed(&p);\n");
    printf("        j = 0;\n");
    printf("        while (j < node_num_inputs[i]) {\n");
    printf("            node_inputs[i][j] = read_int_embed(&p);\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        node_num_outputs[i] = read_int_embed(&p);\n");
    printf("        j = 0;\n");
    printf("        while (j < node_num_outputs[i]) {\n");
    printf("            node_outputs[i][j] = read_int_embed(&p);\n");
    printf("            j = j + 1;\n");
    printf("        }\n\n");

    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    num_nodes = n_nodes;\n\n");

    printf("    return 0;\n");
    printf("}\n\n");

    /* Operation implementations */
    printf("/* Fixed-point multiply */\n");
    printf("static int fp_mul(int a, int b) {\n");
    printf("    long long r = (long long)a * b;\n");
    printf("    return (int)(r >> FP_SHIFT);\n");
    printf("}\n\n");

    printf("/* Fixed-point divide */\n");
    printf("static int fp_div(int a, int b) {\n");
    printf("    if (b == 0) return 0;\n");
    printf("    long long r = ((long long)a << FP_SHIFT) / b;\n");
    printf("    return (int)r;\n");
    printf("}\n\n");

    /* Operation enum */
    printf("enum {\n");
    printf("    OP_GEMM=0, OP_ADD=1, OP_MUL=2, OP_SUB=3, OP_DIV=4,\n");
    printf("    OP_MATMUL=5, OP_RELU=6, OP_SIGMOID=7, OP_SILU=8, OP_TANH=9,\n");
    printf("    OP_SOFTMAX=10, OP_RESHAPE=11, OP_TRANSPOSE=12, OP_IDENTITY=13\n");
    printf("};\n\n");

    /* Execute node */
    printf("static void exec_node(int node_idx) {\n");
    printf("    int op = node_op[node_idx];\n");
    printf("    int in0, in1, out0;\n");
    printf("    int i, size;\n");
    printf("    int *a, *b, *c;\n\n");

    printf("    in0 = (node_num_inputs[node_idx] > 0) ? node_inputs[node_idx][0] : 0;\n");
    printf("    in1 = (node_num_inputs[node_idx] > 1) ? node_inputs[node_idx][1] : 0;\n");
    printf("    out0 = (node_num_outputs[node_idx] > 0) ? node_outputs[node_idx][0] : 0;\n\n");

    printf("    a = tensor_data[in0];\n");
    printf("    b = tensor_data[in1];\n");
    printf("    c = tensor_data[out0];\n");
    printf("    size = tensor_size[out0];\n\n");

    printf("    if (op == OP_ADD) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = a[i] + b[i];\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_MUL) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = fp_mul(a[i], b[i]);\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_SUB) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = a[i] - b[i];\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_DIV) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = fp_div(a[i], b[i]);\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_RELU) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = (a[i] > 0) ? a[i] : 0;\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_IDENTITY) {\n");
    printf("        i = 0;\n");
    printf("        while (i < size) {\n");
    printf("            c[i] = a[i];\n");
    printf("            i = i + 1;\n");
    printf("        }\n");
    printf("    } else if (op == OP_MATMUL) {\n");
    printf("        /* MatMul: C[m,n] = A[m,k] * B[k,n] */\n");
    printf("        int m, k, n, mi, ki, ni;\n");
    printf("        int a_ndims = tensor_ndims[in0];\n");
    printf("        int b_ndims = tensor_ndims[in1];\n");
    printf("        int sum;\n");
    printf("        m = (a_ndims >= 2) ? tensor_dims[in0][a_ndims-2] : 1;\n");
    printf("        k = tensor_dims[in0][a_ndims-1];\n");
    printf("        n = tensor_dims[in1][b_ndims-1];\n");
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
    printf("    }\n");
    printf("}\n\n");

    /* Run model */
    printf("static void run_model(void) {\n");
    printf("    int i = 0;\n");
    printf("    while (i < num_nodes) {\n");
    printf("        exec_node(i);\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("}\n\n");

    /* Find tensor by name */
    printf("static int find_tensor(char *name) {\n");
    printf("    int i = 0;\n");
    printf("    while (i < num_tensors) {\n");
    printf("        if (strcmp(tensor_names[i], name) == 0) return i;\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    return -1;\n");
    printf("}\n\n");
}

/* Print default test harness */
void print_test_harness(void) {
    printf("/* ===== DEFAULT TEST HARNESS ===== */\n\n");

    printf("int main(int argc, char **argv) {\n");
    printf("    int i;\n");
    printf("    int input_idx, output_idx;\n\n");

    printf("    if (load_embedded_model() != 0) {\n");
    printf("        return 1;\n");
    printf("    }\n\n");

    printf("    printf(\"Model loaded: %%d tensors, %%d nodes\\n\", num_tensors, num_nodes);\n\n");

    printf("    input_idx = find_tensor(\"x\");\n");
    printf("    output_idx = find_tensor(\"y\");\n\n");

    printf("    if (input_idx < 0) input_idx = num_tensors - 2;\n");
    printf("    if (output_idx < 0) output_idx = num_tensors - 1;\n\n");

    printf("    printf(\"Input: %%s (%%d values)\\n\", tensor_names[input_idx], tensor_size[input_idx]);\n");
    printf("    printf(\"Output: %%s (%%d values)\\n\", tensor_names[output_idx], tensor_size[output_idx]);\n\n");

    printf("    /* Set test input: 1.0, 2.0, 3.0, 4.0 */\n");
    printf("    i = 0;\n");
    printf("    while (i < tensor_size[input_idx] && i < 4) {\n");
    printf("        tensor_data[input_idx][i] = (i + 1) * SCALE;\n");
    printf("        i = i + 1;\n");
    printf("    }\n\n");

    printf("    run_model();\n\n");

    printf("    printf(\"Output values: \");\n");
    printf("    i = 0;\n");
    printf("    while (i < tensor_size[output_idx] && i < 8) {\n");
    printf("        printf(\"%%d \", tensor_data[output_idx][i] / SCALE);\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    printf(\"\\n\");\n\n");

    printf("    return 0;\n");
    printf("}\n");
}

int main(int argc, char **argv) {
    char *model_path;
    char *program_path;
    unsigned char *model_data;
    char *program_data;
    int model_len;
    int program_len;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.c4onnx [program.c]\n", argv[0]);
        return 1;
    }

    model_path = argv[1];
    program_path = (argc > 2) ? argv[2] : 0;

    /* Read model */
    model_data = (unsigned char *)read_file(model_path, &model_len);
    if (!model_data) {
        return 1;
    }

    /* Read program if provided */
    program_data = 0;
    if (program_path) {
        program_data = read_file(program_path, &program_len);
    }

    /* Generate output */
    printf("/*\n");
    printf(" * Bundled ONNX Model - Generated by onnx_bundler\n");
    printf(" *\n");
    printf(" * Model: %s (%d bytes)\n", model_path, model_len);
    if (program_path) {
        printf(" * Program: %s\n", program_path);
    }
    printf(" */\n\n");

    printf("#include <stdio.h>\n");
    printf("#include <stdlib.h>\n");
    printf("#include <string.h>\n\n");

    /* Embedded model data */
    print_byte_array("embedded_model", model_data, model_len);

    /* Runtime code */
    print_runtime();

    /* Program or test harness */
    if (program_data) {
        printf("/* ===== USER PROGRAM ===== */\n\n");
        printf("%s\n", program_data);
    } else {
        print_test_harness();
    }

    return 0;
}
