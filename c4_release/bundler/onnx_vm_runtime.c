/* ================================================================
 * ONNX VM Runtime — All computation through ONNX subgraph execution
 *
 * This file is concatenated into bundled C output by neural_bundler.
 * It expects these arrays to be defined before it:
 *   - char embedded_model[]    (the .c4onnx v3 model bytes)
 *   - int embedded_model_len
 *   - int program_code[][2]    (bytecode: {op, imm} pairs)
 *   - int program_code_len
 *   - char program_data[]      (string/data segment)
 *   - int program_data_len
 *
 * Architecture:
 *   Part 1: ONNX runtime (from vm/onnx_runtime_c4.c, without main/tests)
 *   Part 2: Subgraph parsing + run_subgraph()
 *   Part 3: Neural ops + NVal + VM execution
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ Fixed-Point Arithmetic (16.16) ============ */

int SCALE;
int HALF;

int init_fp() {
    SCALE = 65536;
    HALF = 32768;
    return 0;
}

int fp_mul(int a, int b) {
    return (a / 256) * (b / 256);
}

int fp_div(int a, int b) {
    if (b == 0) return 0;
    return (a * 256) / (b / 256);
}

int fp_pow2(int x) {
    int int_part;
    int frac;
    int f2;
    int f3;
    int val;
    int neg;

    if (x >= 0) {
        int_part = x / 65536;
    } else {
        int_part = 0 - ((0 - x + 65535) / 65536);
    }
    frac = x - int_part * 65536;

    f2 = fp_mul(frac, frac);
    f3 = fp_mul(f2, frac);
    val = 65536
        + fp_mul(45426, frac)
        + fp_mul(15743, f2)
        + fp_mul(3638, f3);

    if (int_part >= 0) {
        if (int_part >= 20) return 0x7FFFFFFF;
        return val << int_part;
    } else {
        neg = 0 - int_part;
        if (neg >= 30) return 0;
        return val >> neg;
    }
}

int fp_exp(int x) {
    int y;
    y = fp_mul(x, 94548);
    if (y > 20 * 65536) return 0x7FFFFFFF;
    if (y < 0 - 30 * 65536) return 0;
    return fp_pow2(y);
}

int fp_sigmoid(int x) {
    int enx;
    int denom;

    if (x >= SCALE * 8) return SCALE;
    if (x <= 0 - SCALE * 8) return 0;
    enx = fp_exp(0 - x);
    denom = SCALE + enx;
    if (denom == 0) return SCALE;
    return fp_div(SCALE, denom);
}

int fp_silu(int x) {
    return fp_mul(x, fp_sigmoid(x));
}

int fp_relu(int x) {
    if (x < 0) return 0;
    return x;
}

int fp_tanh(int x) {
    int sig;
    sig = fp_sigmoid(x * 2);
    return sig * 2 - SCALE;
}

/* ============ Tensor Structure ============ */

int MAX_TENSORS;
int MAX_DIMS;
int MAX_NAME;

int **tensor_data;
int *tensor_size;
int *tensor_ndims;
int **tensor_dims;
char **tensor_names;
int num_tensors;

int **tensor_sparse_idx;
int **tensor_sparse_val;
int *tensor_nnz;

/* ============ Operation Types ============ */

int OP_GEMM;
int OP_ADD_NN;
int OP_MUL_NN;
int OP_SUB_NN;
int OP_DIV_NN;
int OP_MATMUL;
int OP_RELU;
int OP_SIGMOID_NN;
int OP_SILU_NN;
int OP_TANH_NN;
int OP_SOFTMAX;
int OP_RESHAPE;
int OP_TRANSPOSE;
int OP_IDENTITY;
int OP_CONCAT;
int OP_SLICE;
int OP_SCALE_NN;

int init_ops() {
    OP_GEMM = 0;
    OP_ADD_NN = 1;
    OP_MUL_NN = 2;
    OP_SUB_NN = 3;
    OP_DIV_NN = 4;
    OP_MATMUL = 5;
    OP_RELU = 6;
    OP_SIGMOID_NN = 7;
    OP_SILU_NN = 8;
    OP_TANH_NN = 9;
    OP_SOFTMAX = 10;
    OP_RESHAPE = 11;
    OP_TRANSPOSE = 12;
    OP_IDENTITY = 13;
    OP_CONCAT = 14;
    OP_SLICE = 15;
    OP_SCALE_NN = 16;
    return 0;
}

/* Node structure */
int *node_op;
int *node_num_inputs;
int **node_inputs;
int *node_num_outputs;
int **node_outputs;
int num_nodes;

/* ============ Tensor Operations ============ */

int init_tensors() {
    int i;

    MAX_TENSORS = 256;
    MAX_DIMS = 8;
    MAX_NAME = 64;

    tensor_data = malloc(MAX_TENSORS * sizeof(int *));
    tensor_size = malloc(MAX_TENSORS * sizeof(int));
    tensor_ndims = malloc(MAX_TENSORS * sizeof(int));
    tensor_dims = malloc(MAX_TENSORS * sizeof(int *));
    tensor_names = malloc(MAX_TENSORS * sizeof(char *));
    tensor_sparse_idx = malloc(MAX_TENSORS * sizeof(int *));
    tensor_sparse_val = malloc(MAX_TENSORS * sizeof(int *));
    tensor_nnz = malloc(MAX_TENSORS * sizeof(int));

    i = 0;
    while (i < MAX_TENSORS) {
        tensor_data[i] = 0;
        tensor_size[i] = 0;
        tensor_ndims[i] = 0;
        tensor_dims[i] = malloc(MAX_DIMS * sizeof(int));
        tensor_names[i] = malloc(MAX_NAME);
        tensor_names[i][0] = 0;
        tensor_sparse_idx[i] = 0;
        tensor_sparse_val[i] = 0;
        tensor_nnz[i] = 0;
        i = i + 1;
    }

    num_tensors = 0;
    return 0;
}

int init_nodes() {
    int i;
    int MAX_NODES;
    int MAX_IO;

    MAX_NODES = 512;
    MAX_IO = 8;

    node_op = malloc(MAX_NODES * sizeof(int));
    node_num_inputs = malloc(MAX_NODES * sizeof(int));
    node_inputs = malloc(MAX_NODES * sizeof(int *));
    node_num_outputs = malloc(MAX_NODES * sizeof(int));
    node_outputs = malloc(MAX_NODES * sizeof(int *));

    i = 0;
    while (i < MAX_NODES) {
        node_inputs[i] = malloc(MAX_IO * sizeof(int));
        node_outputs[i] = malloc(MAX_IO * sizeof(int));
        i = i + 1;
    }

    num_nodes = 0;
    return 0;
}

int alloc_tensor(int idx, int ndims, int *dims) {
    int size;
    int i;

    size = 1;
    i = 0;
    while (i < ndims) {
        tensor_dims[idx][i] = dims[i];
        size = size * dims[i];
        i = i + 1;
    }

    tensor_ndims[idx] = ndims;
    tensor_size[idx] = size;
    tensor_data[idx] = malloc(size * sizeof(int));

    i = 0;
    while (i < size) {
        tensor_data[idx][i] = 0;
        i = i + 1;
    }

    return 0;
}

/* ============ Core ONNX Operations ============ */

int op_add(int out_idx, int a_idx, int b_idx) {
    int i;
    int size;
    size = tensor_size[out_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = tensor_data[a_idx][i % tensor_size[a_idx]] + tensor_data[b_idx][i % tensor_size[b_idx]];
        i = i + 1;
    }
    return 0;
}

int op_mul(int out_idx, int a_idx, int b_idx) {
    int i;
    int size;
    size = tensor_size[out_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_mul(tensor_data[a_idx][i % tensor_size[a_idx]], tensor_data[b_idx][i % tensor_size[b_idx]]);
        i = i + 1;
    }
    return 0;
}

int op_sub(int out_idx, int a_idx, int b_idx) {
    int i;
    int size;
    size = tensor_size[out_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = tensor_data[a_idx][i % tensor_size[a_idx]] - tensor_data[b_idx][i % tensor_size[b_idx]];
        i = i + 1;
    }
    return 0;
}

int op_div(int out_idx, int a_idx, int b_idx) {
    int i;
    int size;
    size = tensor_size[out_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_div(tensor_data[a_idx][i % tensor_size[a_idx]], tensor_data[b_idx][i % tensor_size[b_idx]]);
        i = i + 1;
    }
    return 0;
}

int op_matmul(int out_idx, int a_idx, int b_idx) {
    int m;
    int k;
    int n;
    int i;
    int j;
    int l;
    int sum;
    int *a;
    int *b;
    int *c;

    m = tensor_dims[a_idx][0];
    k = tensor_dims[a_idx][1];
    n = tensor_dims[b_idx][1];

    a = tensor_data[a_idx];
    b = tensor_data[b_idx];
    c = tensor_data[out_idx];

    i = 0;
    while (i < m) {
        j = 0;
        while (j < n) {
            sum = 0;
            l = 0;
            while (l < k) {
                sum = sum + fp_mul(a[i * k + l], b[l * n + j]);
                l = l + 1;
            }
            c[i * n + j] = sum;
            j = j + 1;
        }
        i = i + 1;
    }

    return 0;
}

int op_gemm(int out_idx, int a_idx, int b_idx, int c_idx, int alpha, int beta) {
    int m;
    int k;
    int n;
    int i;
    int j;
    int l;
    int sum;
    int *a;
    int *b;
    int *c;
    int *y;

    m = tensor_dims[a_idx][0];
    k = tensor_dims[a_idx][1];
    n = tensor_dims[b_idx][1];

    a = tensor_data[a_idx];
    b = tensor_data[b_idx];
    c = tensor_data[c_idx];
    y = tensor_data[out_idx];

    i = 0;
    while (i < m) {
        j = 0;
        while (j < n) {
            sum = 0;
            l = 0;
            while (l < k) {
                sum = sum + fp_mul(a[i * k + l], b[l * n + j]);
                l = l + 1;
            }
            y[i * n + j] = fp_mul(alpha, sum) + fp_mul(beta, c[j % tensor_size[c_idx]]);
            j = j + 1;
        }
        i = i + 1;
    }

    return 0;
}

int op_relu(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_relu(tensor_data[in_idx][i]);
        i = i + 1;
    }
    return 0;
}

int op_sigmoid(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_sigmoid(tensor_data[in_idx][i]);
        i = i + 1;
    }
    return 0;
}

int op_silu(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_silu(tensor_data[in_idx][i]);
        i = i + 1;
    }
    return 0;
}

int op_tanh(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_tanh(tensor_data[in_idx][i]);
        i = i + 1;
    }
    return 0;
}

int op_softmax(int out_idx, int in_idx) {
    int batch_size;
    int last_dim;
    int ndims;
    int b;
    int i;
    int max_val;
    int sum;
    int offset;

    ndims = tensor_ndims[in_idx];
    last_dim = tensor_dims[in_idx][ndims - 1];
    batch_size = tensor_size[in_idx] / last_dim;

    b = 0;
    while (b < batch_size) {
        offset = b * last_dim;

        max_val = tensor_data[in_idx][offset];
        i = 1;
        while (i < last_dim) {
            if (tensor_data[in_idx][offset + i] > max_val) {
                max_val = tensor_data[in_idx][offset + i];
            }
            i = i + 1;
        }

        sum = 0;
        i = 0;
        while (i < last_dim) {
            tensor_data[out_idx][offset + i] =
                fp_exp(tensor_data[in_idx][offset + i] - max_val);
            sum = sum + tensor_data[out_idx][offset + i];
            i = i + 1;
        }

        if (sum > 0) {
            i = 0;
            while (i < last_dim) {
                tensor_data[out_idx][offset + i] =
                    fp_div(tensor_data[out_idx][offset + i], sum);
                i = i + 1;
            }
        }

        b = b + 1;
    }

    return 0;
}

int op_reshape(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = tensor_data[in_idx][i];
        i = i + 1;
    }
    return 0;
}

int op_identity(int out_idx, int in_idx) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = tensor_data[in_idx][i];
        i = i + 1;
    }
    return 0;
}

int op_concat(int out_idx, int *in_indices, int num_in) {
    int i;
    int j;
    int offset;
    int size;

    offset = 0;
    i = 0;
    while (i < num_in) {
        size = tensor_size[in_indices[i]];
        j = 0;
        while (j < size) {
            tensor_data[out_idx][offset + j] = tensor_data[in_indices[i]][j];
            j = j + 1;
        }
        offset = offset + size;
        i = i + 1;
    }

    return 0;
}

int op_slice(int out_idx, int in_idx, int start, int end) {
    int i;
    int len;
    len = end - start;
    i = 0;
    while (i < len) {
        tensor_data[out_idx][i] = tensor_data[in_idx][start + i];
        i = i + 1;
    }
    return 0;
}

int op_scale(int out_idx, int in_idx, int scalar) {
    int i;
    int size;
    size = tensor_size[in_idx];
    i = 0;
    while (i < size) {
        tensor_data[out_idx][i] = fp_mul(tensor_data[in_idx][i], scalar);
        i = i + 1;
    }
    return 0;
}

int op_matmul_sparse(int out_idx, int in_idx, int *sp_indices, int *sp_values, int nnz, int ncols, int out_size) {
    int i;
    int row;
    int col;
    int *out;
    int *in;

    out = tensor_data[out_idx];
    in = tensor_data[in_idx];

    i = 0;
    while (i < out_size) { out[i] = 0; i = i + 1; }

    i = 0;
    while (i < nnz) {
        row = sp_indices[i] / ncols;
        col = sp_indices[i] % ncols;
        out[col] = out[col] + fp_mul(in[row], sp_values[i]);
        i = i + 1;
    }

    return 0;
}

/* ============ Node Execution ============ */

int execute_node(int node_idx) {
    int op;
    int out;
    int in0;
    int in1;
    int in2;

    op = node_op[node_idx];
    out = node_outputs[node_idx][0];
    in0 = node_inputs[node_idx][0];

    if (node_num_inputs[node_idx] > 1) {
        in1 = node_inputs[node_idx][1];
    } else {
        in1 = 0;
    }

    if (node_num_inputs[node_idx] > 2) {
        in2 = node_inputs[node_idx][2];
    } else {
        in2 = 0;
    }

    if (op == OP_ADD_NN) {
        op_add(out, in0, in1);
    } else if (op == OP_MUL_NN) {
        op_mul(out, in0, in1);
    } else if (op == OP_SUB_NN) {
        op_sub(out, in0, in1);
    } else if (op == OP_DIV_NN) {
        op_div(out, in0, in1);
    } else if (op == OP_MATMUL) {
        op_matmul(out, in0, in1);
    } else if (op == OP_GEMM) {
        op_gemm(out, in0, in1, in2, SCALE, SCALE);
    } else if (op == OP_RELU) {
        op_relu(out, in0);
    } else if (op == OP_SIGMOID_NN) {
        op_sigmoid(out, in0);
    } else if (op == OP_SILU_NN) {
        op_silu(out, in0);
    } else if (op == OP_TANH_NN) {
        op_tanh(out, in0);
    } else if (op == OP_SOFTMAX) {
        op_softmax(out, in0);
    } else if (op == OP_RESHAPE) {
        op_reshape(out, in0);
    } else if (op == OP_IDENTITY) {
        op_identity(out, in0);
    } else if (op == OP_CONCAT) {
        op_concat(out, node_inputs[node_idx], node_num_inputs[node_idx]);
    } else if (op == OP_SLICE) {
        op_slice(out, in0, in1, in2);
    } else if (op == OP_SCALE_NN) {
        op_scale(out, in0, in1);
    } else {
        printf("Unknown op: %d\n", op);
        return -1;
    }

    return 0;
}

int run_model() {
    int i;
    i = 0;
    while (i < num_nodes) {
        execute_node(i);
        i = i + 1;
    }
    return 0;
}

/* ============ Utility Functions ============ */

int read_int_from(char **p) {
    char *d;
    int v;
    d = *p;
    v = (d[0] & 0xFF) | ((d[1] & 0xFF) << 8) | ((d[2] & 0xFF) << 16) | ((d[3] & 0xFF) << 24);
    *p = d + 4;
    return v;
}

int str_match(char *a, char *b) {
    while (*a && *b) {
        if (*a != *b) return 0;
        a = a + 1;
        b = b + 1;
    }
    return *a == *b;
}

/* ============ Subgraph Storage & Execution ============ */

int MAX_SUBGRAPHS;
int MAX_SG_NODES;
int MAX_SG_TEMPS;
int SCRATCH_BASE;     /* tensor index where scratch tensors start */
int SCRATCH_SIZE;     /* max elements per scratch tensor */

char sg_names[16][64];
int sg_n_in[16];
int sg_n_out[16];
int sg_n_temp[16];
int sg_temp_sizes[16][16];  /* temp_sizes[sg][temp] */
int sg_num_nodes[16];

/* Per-subgraph node storage: up to 16 subgraphs x 8 nodes */
int sg_node_op[16][8];
int sg_node_nin[16][8];
int sg_node_in[16][8][4];   /* up to 4 inputs per node */
int sg_node_nout[16][8];
int sg_node_out[16][8][4];  /* up to 4 outputs per node */
int num_subgraphs;

int init_subgraphs() {
    int i;
    int j;
    int dims1[1];

    MAX_SUBGRAPHS = 16;
    MAX_SG_NODES = 8;
    MAX_SG_TEMPS = 16;
    SCRATCH_BASE = 200;
    SCRATCH_SIZE = 512;
    num_subgraphs = 0;

    /* Pre-allocate scratch tensors at indices 200+ */
    i = 0;
    while (i < MAX_SG_TEMPS) {
        dims1[0] = SCRATCH_SIZE;
        alloc_tensor(SCRATCH_BASE + i, 1, dims1);
        i = i + 1;
    }

    return 0;
}

int load_subgraphs(char **p, int n_sg) {
    int i;
    int j;
    int k;
    int name_len;
    int n_nodes;
    int n_in;
    int n_out;

    num_subgraphs = n_sg;
    i = 0;
    while (i < n_sg) {
        name_len = read_int_from(p);
        j = 0;
        while (j < name_len && j < 63) {
            sg_names[i][j] = (*p)[j];
            j = j + 1;
        }
        sg_names[i][j] = 0;
        *p = *p + name_len;

        sg_n_in[i] = read_int_from(p);
        sg_n_out[i] = read_int_from(p);
        sg_n_temp[i] = read_int_from(p);

        j = 0;
        while (j < sg_n_temp[i]) {
            sg_temp_sizes[i][j] = read_int_from(p);
            j = j + 1;
        }

        n_nodes = read_int_from(p);
        sg_num_nodes[i] = n_nodes;

        j = 0;
        while (j < n_nodes) {
            sg_node_op[i][j] = read_int_from(p);

            n_in = read_int_from(p);
            sg_node_nin[i][j] = n_in;
            k = 0;
            while (k < n_in) {
                sg_node_in[i][j][k] = read_int_from(p);
                k = k + 1;
            }

            n_out = read_int_from(p);
            sg_node_nout[i][j] = n_out;
            k = 0;
            while (k < n_out) {
                sg_node_out[i][j][k] = read_int_from(p);
                k = k + 1;
            }

            j = j + 1;
        }

        i = i + 1;
    }

    return 0;
}

int find_subgraph(char *name) {
    int i;
    i = 0;
    while (i < num_subgraphs) {
        if (str_match(sg_names[i], name)) return i;
        i = i + 1;
    }
    return -1;
}

/* Resolve a subgraph reference to a global tensor index.
 * ref >= 0: global tensor (weight)
 * ref < 0: local_idx = (-ref) - 1
 *   local_idx 0..n_in-1 = inputs (mapped to scratch slots)
 *   local_idx n_in..n_in+n_temp-1 = temps
 *   local_idx n_in+n_temp.. = outputs */
int resolve_ref(int ref, int scratch_base) {
    if (ref >= 0) return ref;
    return scratch_base + ((-ref) - 1);
}

/* Run a subgraph.
 * in_bufs[]: array of int* buffers for each input
 * in_sizes[]: size of each input buffer
 * out_bufs[]: array of int* buffers for each output
 * out_sizes[]: size of each output buffer */
int run_subgraph(int sg_idx, int **in_bufs, int *in_sizes, int num_in,
                 int **out_bufs, int *out_sizes, int num_out) {
    int i;
    int j;
    int n_local;
    int scratch;
    int op;
    int ref;
    int tidx;
    int dims1[1];
    int dims2[2];

    /* Total local slots: inputs + temps + outputs */
    n_local = sg_n_in[sg_idx] + sg_n_temp[sg_idx] + sg_n_out[sg_idx];
    scratch = SCRATCH_BASE;  /* use scratch tensors starting at index 200 */

    /* Copy input data into scratch tensor slots */
    i = 0;
    while (i < num_in) {
        /* Resize scratch tensor to match input */
        dims1[0] = in_sizes[i];
        tensor_ndims[scratch + i] = 1;
        tensor_dims[scratch + i][0] = in_sizes[i];
        tensor_size[scratch + i] = in_sizes[i];
        j = 0;
        while (j < in_sizes[i]) {
            tensor_data[scratch + i][j] = in_bufs[i][j];
            j = j + 1;
        }
        i = i + 1;
    }

    /* Set up temp tensor sizes */
    i = 0;
    while (i < sg_n_temp[sg_idx]) {
        tidx = scratch + sg_n_in[sg_idx] + i;
        dims1[0] = sg_temp_sizes[sg_idx][i];
        tensor_ndims[tidx] = 1;
        tensor_dims[tidx][0] = sg_temp_sizes[sg_idx][i];
        tensor_size[tidx] = sg_temp_sizes[sg_idx][i];
        j = 0;
        while (j < sg_temp_sizes[sg_idx][i]) {
            tensor_data[tidx][j] = 0;
            j = j + 1;
        }
        i = i + 1;
    }

    /* Set up output tensor sizes */
    i = 0;
    while (i < num_out) {
        tidx = scratch + sg_n_in[sg_idx] + sg_n_temp[sg_idx] + i;
        dims1[0] = out_sizes[i];
        tensor_ndims[tidx] = 1;
        tensor_dims[tidx][0] = out_sizes[i];
        tensor_size[tidx] = out_sizes[i];
        j = 0;
        while (j < out_sizes[i]) {
            tensor_data[tidx][j] = 0;
            j = j + 1;
        }
        i = i + 1;
    }

    /* Execute subgraph nodes */
    i = 0;
    while (i < sg_num_nodes[sg_idx]) {
        /* Build a temporary node for execute_node */
        /* We reuse node slot 0 temporarily */
        node_op[0] = sg_node_op[sg_idx][i];
        node_num_inputs[0] = sg_node_nin[sg_idx][i];
        j = 0;
        while (j < sg_node_nin[sg_idx][i]) {
            node_inputs[0][j] = resolve_ref(sg_node_in[sg_idx][i][j], scratch);
            j = j + 1;
        }
        node_num_outputs[0] = sg_node_nout[sg_idx][i];
        j = 0;
        while (j < sg_node_nout[sg_idx][i]) {
            node_outputs[0][j] = resolve_ref(sg_node_out[sg_idx][i][j], scratch);
            j = j + 1;
        }

        /* Handle MatMul specially for sparse weights */
        if (node_op[0] == OP_MATMUL) {
            ref = sg_node_in[sg_idx][i][1];  /* weight tensor ref */
            if (ref >= 0 && tensor_sparse_idx[ref] != 0) {
                /* Use sparse matmul */
                op_matmul_sparse(node_outputs[0][0], node_inputs[0][0],
                    tensor_sparse_idx[ref], tensor_sparse_val[ref],
                    tensor_nnz[ref], tensor_dims[ref][1],
                    tensor_size[node_outputs[0][0]]);
                i = i + 1;
                continue;
            }
            /* For matmul, set up 2D dims on input: [1, size] */
            tidx = node_inputs[0][0];
            dims2[0] = 1;
            dims2[1] = tensor_size[tidx];
            tensor_ndims[tidx] = 2;
            tensor_dims[tidx][0] = 1;
            tensor_dims[tidx][1] = dims2[1];
            /* Output is [1, weight_cols] */
            tidx = node_outputs[0][0];
            tensor_ndims[tidx] = 2;
            tensor_dims[tidx][0] = 1;
            tensor_dims[tidx][1] = tensor_dims[node_inputs[0][1]][1];
            tensor_size[tidx] = tensor_dims[node_inputs[0][1]][1];
        }

        execute_node(0);
        i = i + 1;
    }

    /* Copy output data from scratch tensor slots */
    i = 0;
    while (i < num_out) {
        tidx = scratch + sg_n_in[sg_idx] + sg_n_temp[sg_idx] + i;
        j = 0;
        while (j < out_sizes[i]) {
            out_bufs[i][j] = tensor_data[tidx][j];
            j = j + 1;
        }
        i = i + 1;
    }

    return 0;
}

/* ============ Embedded Model Loading ============ */

int load_embedded_model() {
    char *p;
    int magic;
    int version;
    int n_tensors;
    int n_subgraphs;
    int i;
    int j;
    int name_len;
    int ndims;
    int dims[8];
    int storage_type;
    int data_size;
    int nnz;

    p = (char *)embedded_model;

    magic = read_int_from(&p);
    if (magic != 0x584E4E4F) {
        printf("Error: invalid model magic\n");
        return -1;
    }

    version = read_int_from(&p);
    n_tensors = read_int_from(&p);
    n_subgraphs = read_int_from(&p);

    /* Load tensors */
    i = 0;
    while (i < n_tensors) {
        name_len = read_int_from(&p);
        j = 0;
        while (j < name_len && j < 63) {
            tensor_names[i][j] = p[j];
            j = j + 1;
        }
        tensor_names[i][j] = 0;
        p = p + name_len;

        ndims = read_int_from(&p);
        j = 0;
        while (j < ndims) {
            dims[j] = read_int_from(&p);
            j = j + 1;
        }

        if (version >= 3) {
            storage_type = read_int_from(&p);
        } else {
            read_int_from(&p);
            storage_type = 0;
        }

        if (storage_type == 1) {
            /* Sparse COO */
            nnz = read_int_from(&p);
            alloc_tensor(i, ndims, dims);
            tensor_nnz[i] = nnz;
            tensor_sparse_idx[i] = malloc(nnz * sizeof(int));
            tensor_sparse_val[i] = malloc(nnz * sizeof(int));
            j = 0;
            while (j < nnz) {
                tensor_sparse_idx[i][j] = read_int_from(&p);
                j = j + 1;
            }
            j = 0;
            while (j < nnz) {
                tensor_sparse_val[i][j] = read_int_from(&p);
                j = j + 1;
            }
            /* Reconstruct dense data */
            j = 0;
            while (j < nnz) {
                tensor_data[i][tensor_sparse_idx[i][j]] = tensor_sparse_val[i][j];
                j = j + 1;
            }
        } else {
            /* Dense */
            data_size = read_int_from(&p);
            alloc_tensor(i, ndims, dims);
            j = 0;
            while (j < data_size) {
                tensor_data[i][j] = read_int_from(&p);
                j = j + 1;
            }
        }

        i = i + 1;
    }
    num_tensors = n_tensors;

    /* Parse subgraphs (v3) */
    if (version >= 3 && n_subgraphs > 0) {
        load_subgraphs(&p, n_subgraphs);
    }

    return 0;
}

/* ============ Subgraph Indices (cached after model load) ============ */

int sg_b2n;
int sg_n2b;
int sg_nib_add;
int sg_nib_mul;
int sg_nib_and;
int sg_nib_or;
int sg_nib_xor;
int sg_byte_is_zero;
int sg_nib_is_neg;
int sg_baked_fetch;
int sg_baked_data;
int baked_mode;
int baked_data_mode;

/* Tensor indices for shift tables */
int t_shl_res;
int t_shl_over;
int t_shr_res;
int t_shr_under;

int find_tensor_by_name(char *name) {
    int i;
    i = 0;
    while (i < num_tensors) {
        if (str_match(tensor_names[i], name)) return i;
        i = i + 1;
    }
    return -1;
}

int cache_subgraph_indices() {
    sg_b2n = find_subgraph("b2n");
    sg_n2b = find_subgraph("n2b");
    sg_nib_add = find_subgraph("nib_add");
    sg_nib_mul = find_subgraph("nib_mul");
    sg_nib_and = find_subgraph("nib_and");
    sg_nib_or = find_subgraph("nib_or");
    sg_nib_xor = find_subgraph("nib_xor");
    sg_byte_is_zero = find_subgraph("byte_is_zero");
    sg_nib_is_neg = find_subgraph("nib_is_neg");
    sg_baked_fetch = find_subgraph("baked_fetch");
    baked_mode = (sg_baked_fetch >= 0) ? 1 : 0;
    sg_baked_data = find_subgraph("baked_data");
    baked_data_mode = (sg_baked_data >= 0) ? 1 : 0;
    t_shl_res = find_tensor_by_name("shl_result");
    t_shl_over = find_tensor_by_name("shl_overflow");
    t_shr_res = find_tensor_by_name("shr_result");
    t_shr_under = find_tensor_by_name("shr_underflow");
    return 0;
}

/* ============ Neural Byte/Nibble Operations (via subgraphs) ============ */

int FP_ONE;   /* 65536 */
int FP_HALF;  /* 32768 */

int nn_argmax(int *v, int n) {
    int best;
    int i;
    best = 0;
    i = 1;
    while (i < n) {
        if (v[i] > v[best]) best = i;
        i = i + 1;
    }
    return best;
}

void encode_byte(int *out, int val) {
    int i;
    i = 0;
    while (i < 256) { out[i] = 0; i = i + 1; }
    out[val & 0xFF] = FP_ONE;
}

int decode_byte(int *v) {
    return nn_argmax(v, 256);
}

/* Byte to nibbles via b2n subgraph */
void neural_b2n(int *high, int *low, int *byte_oh) {
    int *in_bufs[1];
    int in_sizes[1];
    int *out_bufs[2];
    int out_sizes[2];

    in_bufs[0] = byte_oh;
    in_sizes[0] = 256;
    out_bufs[0] = high;
    out_bufs[1] = low;
    out_sizes[0] = 16;
    out_sizes[1] = 16;

    run_subgraph(sg_b2n, in_bufs, in_sizes, 1, out_bufs, out_sizes, 2);
}

/* Nibbles to byte via n2b subgraph */
void neural_n2b(int *byte_oh, int *high, int *low) {
    int *in_bufs[2];
    int in_sizes[2];
    int *out_bufs[1];
    int out_sizes[1];

    in_bufs[0] = high;
    in_bufs[1] = low;
    in_sizes[0] = 16;
    in_sizes[1] = 16;
    out_bufs[0] = byte_oh;
    out_sizes[0] = 256;

    run_subgraph(sg_n2b, in_bufs, in_sizes, 2, out_bufs, out_sizes, 1);
}

/* Nibble add with carry via nib_add subgraph */
void neural_nib_add(int *sum_out, int *cout, int *a, int *b, int *cin) {
    int *in_bufs[3];
    int in_sizes[3];
    int *out_bufs[2];
    int out_sizes[2];

    in_bufs[0] = a;
    in_bufs[1] = b;
    in_bufs[2] = cin;
    in_sizes[0] = 16;
    in_sizes[1] = 16;
    in_sizes[2] = 2;
    out_bufs[0] = sum_out;
    out_bufs[1] = cout;
    out_sizes[0] = 16;
    out_sizes[1] = 2;

    run_subgraph(sg_nib_add, in_bufs, in_sizes, 3, out_bufs, out_sizes, 2);
}

/* Nibble bitwise op via subgraph */
void neural_nib_op(int *result, int *a, int *b, int sg_idx) {
    int *in_bufs[2];
    int in_sizes[2];
    int *out_bufs[1];
    int out_sizes[1];

    in_bufs[0] = a;
    in_bufs[1] = b;
    in_sizes[0] = 16;
    in_sizes[1] = 16;
    out_bufs[0] = result;
    out_sizes[0] = 16;

    run_subgraph(sg_idx, in_bufs, in_sizes, 2, out_bufs, out_sizes, 1);
}

/* Nibble multiply via nib_mul subgraph */
void neural_nib_mul(int *lo_out, int *hi_out, int *a, int *b) {
    int *in_bufs[2];
    int in_sizes[2];
    int *out_bufs[2];
    int out_sizes[2];

    in_bufs[0] = a;
    in_bufs[1] = b;
    in_sizes[0] = 16;
    in_sizes[1] = 16;
    out_bufs[0] = lo_out;
    out_bufs[1] = hi_out;
    out_sizes[0] = 16;
    out_sizes[1] = 16;

    run_subgraph(sg_nib_mul, in_bufs, in_sizes, 2, out_bufs, out_sizes, 2);
}

/* ============ NVal: 32-bit integer as 4 one-hot byte vectors ============ */

int *nval_new() {
    int *v;
    int i;
    v = malloc(1024 * 4);
    i = 0;
    while (i < 1024) { v[i] = 0; i = i + 1; }
    v[0] = FP_ONE;
    v[256] = FP_ONE;
    v[512] = FP_ONE;
    v[768] = FP_ONE;
    return v;
}

void nval_encode(int *v, int val) {
    int i;
    i = 0;
    while (i < 4) {
        encode_byte(v + i * 256, (val >> (i * 8)) & 0xFF);
        i = i + 1;
    }
}

int nval_decode(int *v) {
    int i;
    int val;
    val = 0;
    i = 0;
    while (i < 4) {
        val = val | (decode_byte(v + i * 256) << (i * 8));
        i = i + 1;
    }
    return val;
}

void nval_copy(int *dst, int *src) {
    int i;
    i = 0;
    while (i < 1024) { dst[i] = src[i]; i = i + 1; }
}

/* ============ Neural 32-bit Operations ============ */

void neural_add(int *result, int *a, int *b) {
    int ah[16], al[16], bh[16], bl[16];
    int sum_l[16], sum_h[16], carry[2];
    int i;

    carry[0] = FP_ONE; carry[1] = 0;

    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        neural_b2n(bh, bl, b + i * 256);
        neural_nib_add(sum_l, carry, al, bl, carry);
        neural_nib_add(sum_h, carry, ah, bh, carry);
        neural_n2b(result + i * 256, sum_h, sum_l);
        i = i + 1;
    }
}

void neural_negate(int *result, int *x) {
    int xh[16], xl[16], oh[16], ol[16], rh[16], rl[16];
    int sum_l[16], sum_h[16], carry[2];
    int ones[256];
    int zero_nib[16];
    int *inv;
    int i;

    inv = malloc(1024 * 4);
    encode_byte(ones, 0xFF);

    i = 0;
    while (i < 4) {
        neural_b2n(xh, xl, x + i * 256);
        neural_b2n(oh, ol, ones);
        neural_nib_op(rh, xh, oh, sg_nib_xor);
        neural_nib_op(rl, xl, ol, sg_nib_xor);
        neural_n2b(inv + i * 256, rh, rl);
        i = i + 1;
    }

    carry[0] = 0; carry[1] = FP_ONE;
    i = 0;
    while (i < 16) { zero_nib[i] = 0; i = i + 1; }
    zero_nib[0] = FP_ONE;

    i = 0;
    while (i < 4) {
        neural_b2n(xh, xl, inv + i * 256);
        neural_nib_add(sum_l, carry, xl, zero_nib, carry);
        neural_nib_add(sum_h, carry, xh, zero_nib, carry);
        neural_n2b(result + i * 256, sum_h, sum_l);
        i = i + 1;
    }
    free(inv);
}

void neural_sub(int *result, int *a, int *b) {
    int *neg_b;
    neg_b = nval_new();
    neural_negate(neg_b, b);
    neural_add(result, a, neg_b);
    free(neg_b);
}

void neural_bitwise(int *result, int *a, int *b, int sg_idx) {
    int ah[16], al[16], bh[16], bl[16], rh[16], rl[16];
    int i;
    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        neural_b2n(bh, bl, b + i * 256);
        neural_nib_op(rh, ah, bh, sg_idx);
        neural_nib_op(rl, al, bl, sg_idx);
        neural_n2b(result + i * 256, rh, rl);
        i = i + 1;
    }
}

void neural_shl1(int *result, int *x) {
    int h[16], l[16];
    int new_l[16], new_h[16], l_over[16], h_over[16];
    int carry[16];
    int i;
    int j;
    int k;
    int lv;
    int hv;
    int sr;
    int so;
    int *shl_r;
    int *shl_o;

    shl_r = tensor_data[t_shl_res];
    shl_o = tensor_data[t_shl_over];

    j = 0;
    while (j < 16) { carry[j] = 0; j = j + 1; }
    carry[0] = FP_ONE;

    i = 0;
    while (i < 4) {
        neural_b2n(h, l, x + i * 256);

        j = 0;
        while (j < 16) {
            new_l[j] = 0; l_over[j] = 0;
            new_h[j] = 0; h_over[j] = 0;
            j = j + 1;
        }
        j = 0;
        while (j < 16) {
            lv = l[j]; hv = h[j];
            k = 0;
            while (k < 16) {
                sr = shl_r[1 * 256 + j * 16 + k];
                so = shl_o[1 * 256 + j * 16 + k];
                new_l[k] = new_l[k] + fp_mul(lv, sr);
                l_over[k] = l_over[k] + fp_mul(lv, so);
                new_h[k] = new_h[k] + fp_mul(hv, sr);
                h_over[k] = h_over[k] + fp_mul(hv, so);
                k = k + 1;
            }
            j = j + 1;
        }

        neural_nib_op(new_l, new_l, carry, sg_nib_or);
        neural_nib_op(new_h, new_h, l_over, sg_nib_or);
        j = 0;
        while (j < 16) { carry[j] = h_over[j]; j = j + 1; }

        neural_n2b(result + i * 256, new_h, new_l);
        i = i + 1;
    }
}

void neural_shr1(int *result, int *x) {
    int h[16], l[16];
    int new_l[16], new_h[16], l_under[16], h_under[16];
    int carry[16];
    int i;
    int j;
    int k;
    int lv;
    int hv;
    int sr;
    int su;
    int *shr_r;
    int *shr_u;

    shr_r = tensor_data[t_shr_res];
    shr_u = tensor_data[t_shr_under];

    j = 0;
    while (j < 16) { carry[j] = 0; j = j + 1; }
    carry[0] = FP_ONE;

    i = 3;
    while (i >= 0) {
        neural_b2n(h, l, x + i * 256);

        j = 0;
        while (j < 16) {
            new_l[j] = 0; l_under[j] = 0;
            new_h[j] = 0; h_under[j] = 0;
            j = j + 1;
        }
        j = 0;
        while (j < 16) {
            lv = l[j]; hv = h[j];
            k = 0;
            while (k < 16) {
                sr = shr_r[1 * 256 + j * 16 + k];
                su = shr_u[1 * 256 + j * 16 + k];
                new_l[k] = new_l[k] + fp_mul(lv, sr);
                l_under[k] = l_under[k] + fp_mul(lv, su);
                new_h[k] = new_h[k] + fp_mul(hv, sr);
                h_under[k] = h_under[k] + fp_mul(hv, su);
                k = k + 1;
            }
            j = j + 1;
        }

        neural_nib_op(new_h, new_h, carry, sg_nib_or);
        neural_nib_op(new_l, new_l, h_under, sg_nib_or);
        j = 0;
        while (j < 16) { carry[j] = l_under[j]; j = j + 1; }

        neural_n2b(result + i * 256, new_h, new_l);
        i = i - 1;
    }
}

void neural_mul(int *result, int *a, int *b) {
    int *a_nib;
    int *b_nib;
    int *acc;
    int carry[2];
    int prod_lo[16];
    int prod_hi[16];
    int sum_tmp[16];
    int ah[16], al[16], bh[16], bl[16];
    int zero_nib_l[16];
    int i;
    int j;
    int k;
    int pos;

    a_nib = malloc(128 * 4);
    b_nib = malloc(128 * 4);
    acc = malloc(128 * 4);

    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        k = 0;
        while (k < 16) {
            a_nib[(i * 2) * 16 + k] = al[k];
            a_nib[(i * 2 + 1) * 16 + k] = ah[k];
            k = k + 1;
        }
        neural_b2n(bh, bl, b + i * 256);
        k = 0;
        while (k < 16) {
            b_nib[(i * 2) * 16 + k] = bl[k];
            b_nib[(i * 2 + 1) * 16 + k] = bh[k];
            k = k + 1;
        }
        i = i + 1;
    }

    i = 0;
    while (i < 128) { acc[i] = 0; i = i + 1; }
    i = 0;
    while (i < 8) { acc[i * 16] = FP_ONE; i = i + 1; }

    i = 0;
    while (i < 16) { zero_nib_l[i] = 0; i = i + 1; }
    zero_nib_l[0] = FP_ONE;

    i = 0;
    while (i < 8) {
        j = 0;
        while (j < 8) {
            pos = i + j;
            if (pos < 8) {
                neural_nib_mul(prod_lo, prod_hi, a_nib + i * 16, b_nib + j * 16);

                carry[0] = FP_ONE; carry[1] = 0;
                neural_nib_add(sum_tmp, carry, acc + pos * 16, prod_lo, carry);
                k = 0;
                while (k < 16) { acc[pos * 16 + k] = sum_tmp[k]; k = k + 1; }

                if (pos + 1 < 8) {
                    neural_nib_add(sum_tmp, carry, acc + (pos + 1) * 16, prod_hi, carry);
                    k = 0;
                    while (k < 16) { acc[(pos + 1) * 16 + k] = sum_tmp[k]; k = k + 1; }

                    pos = pos + 2;
                    while (pos < 8) {
                        if (carry[1] > FP_HALF) {
                            neural_nib_add(sum_tmp, carry, acc + pos * 16, zero_nib_l, carry);
                            k = 0;
                            while (k < 16) { acc[pos * 16 + k] = sum_tmp[k]; k = k + 1; }
                        }
                        pos = pos + 1;
                    }
                }
            }
            j = j + 1;
        }
        i = i + 1;
    }

    i = 0;
    while (i < 4) {
        neural_n2b(result + i * 256, acc + (i * 2 + 1) * 16, acc + (i * 2) * 16);
        i = i + 1;
    }
    free(a_nib);
    free(b_nib);
    free(acc);
}

int neural_is_zero(int *x) {
    int h0[16], l0[16], h1[16], l1[16], h2[16], l2[16], h3[16], l3[16];
    int h01[16], l01[16], h_all[16], l_all[16], h_final[16], l_final[16];
    int or_byte[256];
    int result[2];
    int *in_bufs[1];
    int in_sizes[1];
    int *out_bufs[1];
    int out_sizes[1];

    /* OR all 4 bytes together using b2n + nib_or */
    neural_b2n(h0, l0, x);
    neural_b2n(h1, l1, x + 256);
    neural_nib_op(h01, h0, h1, sg_nib_or);
    neural_nib_op(l01, l0, l1, sg_nib_or);

    neural_b2n(h2, l2, x + 512);
    neural_nib_op(h_all, h01, h2, sg_nib_or);
    neural_nib_op(l_all, l01, l2, sg_nib_or);

    neural_b2n(h3, l3, x + 768);
    neural_nib_op(h_final, h_all, h3, sg_nib_or);
    neural_nib_op(l_final, l_all, l3, sg_nib_or);

    /* Recombine into single byte */
    neural_n2b(or_byte, h_final, l_final);

    /* Check if or_byte is zero via byte_is_zero subgraph */
    in_bufs[0] = or_byte;
    in_sizes[0] = 256;
    out_bufs[0] = result;
    out_sizes[0] = 2;
    run_subgraph(sg_byte_is_zero, in_bufs, in_sizes, 1, out_bufs, out_sizes, 1);

    return result[0] > FP_HALF ? 1 : 0;
}

int neural_is_negative(int *x) {
    int hi_nib[16], lo_nib[16];
    int result[2];
    int *in_bufs[1];
    int in_sizes[1];
    int *out_bufs[1];
    int out_sizes[1];

    /* Get hi nibble of MSB (byte 3) */
    neural_b2n(hi_nib, lo_nib, x + 768);

    /* Check if hi nibble >= 8 via nib_is_neg subgraph */
    in_bufs[0] = hi_nib;
    in_sizes[0] = 16;
    out_bufs[0] = result;
    out_sizes[0] = 2;
    run_subgraph(sg_nib_is_neg, in_bufs, in_sizes, 1, out_bufs, out_sizes, 1);

    return result[0] > FP_HALF ? 1 : 0;
}

int neural_compare(int *a, int *b) {
    int *diff;
    int z;
    int n;
    diff = nval_new();
    neural_sub(diff, a, b);
    z = neural_is_zero(diff);
    n = neural_is_negative(diff);
    free(diff);
    if (z) return 0;
    if (n) return 0 - 1;
    return 1;
}

void neural_div(int *result, int *a, int *b) {
    int av;
    int bv;
    int a_neg;
    int b_neg;
    int *a_abs;
    int *b_abs;
    int *quotient;
    int *remainder;
    int *one;
    int *bit_mask;
    int *masked;
    int i;
    int bit_set;
    int cmp;

    av = nval_decode(a);
    bv = nval_decode(b);
    if (bv == 0) { nval_encode(result, 0); return; }

    a_neg = av < 0;
    b_neg = bv < 0;

    a_abs = nval_new();
    b_abs = nval_new();
    if (a_neg) neural_negate(a_abs, a); else nval_copy(a_abs, a);
    if (b_neg) neural_negate(b_abs, b); else nval_copy(b_abs, b);

    quotient = nval_new();
    remainder = nval_new();
    one = nval_new();
    bit_mask = nval_new();
    masked = nval_new();
    nval_encode(quotient, 0);
    nval_encode(remainder, 0);
    nval_encode(one, 1);

    i = 31;
    while (i >= 0) {
        neural_shl1(remainder, remainder);

        nval_encode(bit_mask, 1 << i);
        neural_bitwise(masked, a_abs, bit_mask, sg_nib_and);
        bit_set = !neural_is_zero(masked);

        if (bit_set) {
            neural_bitwise(remainder, remainder, one, sg_nib_or);
        }

        cmp = neural_compare(remainder, b_abs);
        if (cmp >= 0) {
            neural_sub(remainder, remainder, b_abs);
            neural_bitwise(quotient, quotient, bit_mask, sg_nib_or);
        }

        i = i - 1;
    }

    if (a_neg != b_neg) neural_negate(result, quotient);
    else nval_copy(result, quotient);

    free(a_abs); free(b_abs); free(quotient); free(remainder);
    free(one); free(bit_mask); free(masked);
}

void neural_mod(int *result, int *a, int *b) {
    int *q;
    int *qb;
    q = nval_new();
    qb = nval_new();
    neural_div(q, a, b);
    neural_mul(qb, q, b);
    neural_sub(result, a, qb);
    free(q); free(qb);
}

void neural_shl(int *result, int *x, int *amt) {
    int shift;
    int i;
    nval_copy(result, x);
    shift = nval_decode(amt);
    i = 0;
    while (i < shift && i < 32) {
        neural_shl1(result, result);
        i = i + 1;
    }
}

void neural_shr(int *result, int *x, int *amt) {
    int shift;
    int i;
    nval_copy(result, x);
    shift = nval_decode(amt);
    i = 0;
    while (i < shift && i < 32) {
        neural_shr1(result, result);
        i = i + 1;
    }
}

void neural_mask_byte(int *result, int *x) {
    int i;
    int j;
    j = 0;
    while (j < 256) { result[j] = x[j]; j = j + 1; }
    i = 1;
    while (i < 4) {
        j = 0;
        while (j < 256) { result[i * 256 + j] = 0; j = j + 1; }
        result[i * 256] = FP_ONE;
        i = i + 1;
    }
}

/* ============ VM State ============ */

int *reg_ax;
int *reg_sp;
int *reg_bp;
int *reg_pc;
int vm_halted;
int vm_exit_code;

int MEM_SIZE;
int *vm_memory;
int heap_ptr;

void vm_push(int *val) {
    int *eight;
    int addr;
    eight = nval_new();
    nval_encode(eight, 8);
    neural_sub(reg_sp, reg_sp, eight);
    addr = nval_decode(reg_sp);
    if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(val);
    free(eight);
}

void vm_pop(int *val) {
    int addr;
    int *eight;
    addr = nval_decode(reg_sp);
    nval_encode(val, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
    eight = nval_new();
    nval_encode(eight, 8);
    neural_add(reg_sp, reg_sp, eight);
    free(eight);
}

/* ============ VM Execution ============ */

int vm_step() {
    int pc_val;
    int instr_idx;
    int op;
    int imm;
    int *eight;
    int *imm_enc;
    int *ax;
    int *a;
    int cmp;
    int addr;
    int c;
    int size;
    int ptr;
    int *val;
    int *masked;
    int nib[128];
    int nh[16];
    int nl[16];
    int sg_out[1025];
    int *sg_ib[1];
    int sg_is[1];
    int *sg_ob[1];
    int sg_os[1];
    int bi;

    if (vm_halted) return 0;

    pc_val = nval_decode(reg_pc);
    instr_idx = pc_val / 8;

    if (instr_idx < 0 || instr_idx >= program_code_len) {
        vm_halted = 1;
        return 0;
    }

    imm_enc = nval_new();

    if (baked_mode) {
        /* Decompose PC NVal → nibbles, run baked_fetch subgraph */
        bi = 0;
        while (bi < 4) {
            neural_b2n(nh, nl, reg_pc + bi * 256);
            memcpy(nib + (bi * 2) * 16, nl, 16 * sizeof(int));
            memcpy(nib + (bi * 2 + 1) * 16, nh, 16 * sizeof(int));
            bi = bi + 1;
        }
        sg_ib[0] = nib; sg_is[0] = 128;
        sg_ob[0] = sg_out; sg_os[0] = 1025;
        run_subgraph(sg_baked_fetch, sg_ib, sg_is, 1, sg_ob, sg_os, 1);
        op = (sg_out[0] + FP_HALF) / SCALE;
        memcpy(imm_enc, sg_out + 1, 1024 * sizeof(int));
    } else {
        op = program_code[instr_idx][0];
        imm = program_code[instr_idx][1];
    }

    eight = nval_new();
    nval_encode(eight, 8);
    neural_add(reg_pc, reg_pc, eight);
    free(eight);

    if (!baked_mode) {
        nval_encode(imm_enc, imm);
    }
    ax = nval_new();
    nval_copy(ax, reg_ax);
    a = nval_new();

    if (op == 0) { /* LEA */
        neural_add(reg_ax, reg_bp, imm_enc);
    }
    else if (op == 1) { /* IMM */
        nval_copy(reg_ax, imm_enc);
    }
    else if (op == 2) { /* JMP */
        nval_copy(reg_pc, imm_enc);
    }
    else if (op == 3) { /* JSR */
        vm_push(reg_pc);
        nval_copy(reg_pc, imm_enc);
    }
    else if (op == 4) { /* BZ */
        if (neural_is_zero(ax)) nval_copy(reg_pc, imm_enc);
    }
    else if (op == 5) { /* BNZ */
        if (!neural_is_zero(ax)) nval_copy(reg_pc, imm_enc);
    }
    else if (op == 6) { /* ENT */
        vm_push(reg_bp);
        nval_copy(reg_bp, reg_sp);
        neural_sub(reg_sp, reg_sp, imm_enc);
    }
    else if (op == 7) { /* ADJ */
        neural_add(reg_sp, reg_sp, imm_enc);
    }
    else if (op == 8) { /* LEV */
        nval_copy(reg_sp, reg_bp);
        vm_pop(reg_bp);
        vm_pop(reg_pc);
    }
    else if (op == 9) { /* LI */
        addr = nval_decode(ax);
        if (baked_data_mode && addr >= 0x10000 && addr < 0x10000 + program_data_len) {
            bi = 0;
            while (bi < 4) {
                neural_b2n(nh, nl, ax + bi * 256);
                memcpy(nib + (bi * 2) * 16, nl, 16 * sizeof(int));
                memcpy(nib + (bi * 2 + 1) * 16, nh, 16 * sizeof(int));
                bi = bi + 1;
            }
            sg_ib[0] = nib; sg_is[0] = 128;
            sg_ob[0] = sg_out; sg_os[0] = 1;
            run_subgraph(sg_baked_data, sg_ib, sg_is, 1, sg_ob, sg_os, 1);
            nval_encode(reg_ax, (sg_out[0] + FP_HALF) / SCALE);
        } else {
            nval_encode(reg_ax, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
        }
    }
    else if (op == 10) { /* LC */
        addr = nval_decode(ax);
        val = nval_new();
        if (baked_data_mode && addr >= 0x10000 && addr < 0x10000 + program_data_len) {
            bi = 0;
            while (bi < 4) {
                neural_b2n(nh, nl, ax + bi * 256);
                memcpy(nib + (bi * 2) * 16, nl, 16 * sizeof(int));
                memcpy(nib + (bi * 2 + 1) * 16, nh, 16 * sizeof(int));
                bi = bi + 1;
            }
            sg_ib[0] = nib; sg_is[0] = 128;
            sg_ob[0] = sg_out; sg_os[0] = 1;
            run_subgraph(sg_baked_data, sg_ib, sg_is, 1, sg_ob, sg_os, 1);
            nval_encode(val, (sg_out[0] + FP_HALF) / SCALE);
        } else {
            nval_encode(val, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
        }
        neural_mask_byte(reg_ax, val);
        free(val);
    }
    else if (op == 11) { /* SI */
        vm_pop(a);
        addr = nval_decode(a);
        if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(ax);
    }
    else if (op == 12) { /* SC */
        vm_pop(a);
        addr = nval_decode(a);
        masked = nval_new();
        neural_mask_byte(masked, ax);
        if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(masked);
        free(masked);
    }
    else if (op == 13) { /* PSH */
        vm_push(ax);
    }
    else if (op == 14) { /* OR */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, sg_nib_or);
    }
    else if (op == 15) { /* XOR */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, sg_nib_xor);
    }
    else if (op == 16) { /* AND */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, sg_nib_and);
    }
    else if (op == 17) { /* EQ */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp == 0 ? 1 : 0);
    }
    else if (op == 18) { /* NE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp != 0 ? 1 : 0);
    }
    else if (op == 19) { /* LT */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp < 0 ? 1 : 0);
    }
    else if (op == 20) { /* GT */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp > 0 ? 1 : 0);
    }
    else if (op == 21) { /* LE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp <= 0 ? 1 : 0);
    }
    else if (op == 22) { /* GE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp >= 0 ? 1 : 0);
    }
    else if (op == 23) { /* SHL */
        vm_pop(a);
        neural_shl(reg_ax, a, ax);
    }
    else if (op == 24) { /* SHR */
        vm_pop(a);
        neural_shr(reg_ax, a, ax);
    }
    else if (op == 25) { /* ADD */
        vm_pop(a);
        neural_add(reg_ax, a, ax);
    }
    else if (op == 26) { /* SUB */
        vm_pop(a);
        neural_sub(reg_ax, a, ax);
    }
    else if (op == 27) { /* MUL */
        vm_pop(a);
        neural_mul(reg_ax, a, ax);
    }
    else if (op == 28) { /* DIV */
        vm_pop(a);
        neural_div(reg_ax, a, ax);
    }
    else if (op == 29) { /* MOD */
        vm_pop(a);
        neural_mod(reg_ax, a, ax);
    }
    else if (op == 34) { /* MALC */
        int *size_nval;
        int *seven_nval;
        int *sum_nval;
        int *three_nval;
        int *aligned_nval;

        addr = nval_decode(reg_sp);
        size = (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
        ptr = heap_ptr;

        /* Neural: (size + 7) / 8  via add then shift-right-3 */
        size_nval = nval_new();
        seven_nval = nval_new();
        sum_nval = nval_new();
        three_nval = nval_new();
        aligned_nval = nval_new();

        nval_encode(size_nval, size);
        nval_encode(seven_nval, 7);
        neural_add(sum_nval, size_nval, seven_nval);
        nval_encode(three_nval, 3);
        neural_shr(aligned_nval, sum_nval, three_nval);
        heap_ptr = heap_ptr + nval_decode(aligned_nval);

        free(size_nval);
        free(seven_nval);
        free(sum_nval);
        free(three_nval);
        free(aligned_nval);

        if (heap_ptr >= MEM_SIZE) heap_ptr = MEM_SIZE - 1;
        nval_encode(reg_ax, ptr);
    }
    else if (op == 35) { /* FREE */ }
    else if (op == 36) { /* MSET - memset(ptr, val, size) */
        int sp_val = nval_decode(reg_sp);
        ptr = vm_memory[sp_val + 16];
        c = vm_memory[sp_val + 8];
        size = vm_memory[sp_val];
        addr = 0;
        while (addr < size) {
            if (ptr + addr >= 0 && ptr + addr < MEM_SIZE)
                vm_memory[ptr + addr] = c & 0xFF;
            addr = addr + 1;
        }
        nval_encode(reg_ax, ptr);
    }
    else if (op == 37) { /* MCMP - memcmp(p1, p2, size) */
        int sp_val = nval_decode(reg_sp);
        int p1 = vm_memory[sp_val + 16];
        int p2 = vm_memory[sp_val + 8];
        size = vm_memory[sp_val];
        c = 0;
        addr = 0;
        while (addr < size && c == 0) {
            int a_val = (p1+addr >= 0 && p1+addr < MEM_SIZE) ? vm_memory[p1+addr] : 0;
            int b_val = (p2+addr >= 0 && p2+addr < MEM_SIZE) ? vm_memory[p2+addr] : 0;
            c = (a_val & 0xFF) - (b_val & 0xFF);
            addr = addr + 1;
        }
        nval_encode(reg_ax, c);
    }
    else if (op == 38) { /* EXIT */
        vm_exit_code = nval_decode(ax);
        vm_halted = 1;
        free(imm_enc); free(ax); free(a);
        return 0;
    }
    else if (op == 64) { /* GETCHAR */
        c = getchar();
        nval_encode(reg_ax, c);
    }
    else if (op == 65) { /* PUTCHAR */
        addr = nval_decode(reg_sp);
        c = (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
        putchar(c & 0xFF);
        fflush(stdout);
    }
    else {
        printf("Unknown opcode: %d at PC=%d\n", op, pc_val);
        vm_halted = 1;
        free(imm_enc); free(ax); free(a);
        return 0;
    }

    free(imm_enc);
    free(ax);
    free(a);
    return 1;
}

/* ============ Main ============ */

int main() {
    int i;

    FP_ONE = 65536;
    FP_HALF = 32768;
    MEM_SIZE = 0x20000;
    heap_ptr = 0x18000;

    init_fp();
    init_ops();
    init_tensors();
    init_nodes();
    init_subgraphs();

    vm_memory = malloc(MEM_SIZE * 4);
    i = 0;
    while (i < MEM_SIZE) { vm_memory[i] = 0; i = i + 1; }

    if (load_embedded_model() != 0) {
        printf("Failed to load neural model\n");
        return 1;
    }

    cache_subgraph_indices();

    /* Grow scratch tensors if any subgraph needs larger buffers */
    {
        int si;
        int ti;
        int max_needed;
        int dims1[1];
        max_needed = SCRATCH_SIZE;
        if (max_needed < 2048) max_needed = 2048;
        si = 0;
        while (si < num_subgraphs) {
            ti = 0;
            while (ti < sg_n_temp[si]) {
                if (sg_temp_sizes[si][ti] > max_needed)
                    max_needed = sg_temp_sizes[si][ti];
                ti = ti + 1;
            }
            si = si + 1;
        }
        if (max_needed > SCRATCH_SIZE) {
            SCRATCH_SIZE = max_needed;
            i = 0;
            while (i < MAX_SG_TEMPS) {
                free(tensor_data[SCRATCH_BASE + i]);
                dims1[0] = SCRATCH_SIZE;
                tensor_data[SCRATCH_BASE + i] = malloc(SCRATCH_SIZE * sizeof(int));
                tensor_size[SCRATCH_BASE + i] = SCRATCH_SIZE;
                tensor_dims[SCRATCH_BASE + i][0] = SCRATCH_SIZE;
                i = i + 1;
            }
        }
    }

    reg_ax = nval_new();
    reg_sp = nval_new();
    reg_bp = nval_new();
    reg_pc = nval_new();
    nval_encode(reg_ax, 0);
    nval_encode(reg_sp, 0x10000);
    nval_encode(reg_bp, 0x10000);
    nval_encode(reg_pc, 0);
    vm_halted = 0;
    vm_exit_code = 0;

    i = 0;
    while (i < program_data_len) {
        vm_memory[0x10000 + i] = program_data[i];
        i = i + 1;
    }

    while (vm_step()) {}

    return vm_exit_code & 0xFF;
}
