/*
 * Text to C4 ONNX Binary Converter - C4 Compatible Version
 *
 * Simplified for C4 compiler compatibility.
 * C4 limitations: no sizeof, limited pointer arithmetic
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global storage */
int MAX_NAME;
int MAX_LINE;
int MAX_TENSORS;
int MAX_DIMS;
int MAX_DATA;
int MAX_NODES;

char *line;
int line_pos;

char *names;      /* names[i * MAX_NAME] = tensor name i */
int *ndims;       /* ndims[i] = number of dims for tensor i */
int *dims;        /* dims[i * MAX_DIMS + j] = dim j of tensor i */
int *sizes;       /* sizes[i] = size of tensor i */
int *data;        /* data[base + j] = value j of tensor i, base computed */
int *data_base;   /* data_base[i] = starting index in data[] */
int num_tensors;

int *node_op;
int *node_nin;
int *node_in;     /* node_in[i * 8 + j] = input j of node i */
int *node_nout;
int *node_out;    /* node_out[i * 8 + j] = output j of node i */
int num_nodes;

int data_ptr;     /* current position in data array */

int init() {
    MAX_NAME = 64;
    MAX_LINE = 100000;
    MAX_TENSORS = 256;
    MAX_DIMS = 4;
    MAX_DATA = 50000;
    MAX_NODES = 512;

    line = malloc(MAX_LINE);
    names = malloc(MAX_TENSORS * MAX_NAME);
    ndims = malloc(MAX_TENSORS * 4);
    dims = malloc(MAX_TENSORS * MAX_DIMS * 4);
    sizes = malloc(MAX_TENSORS * 4);
    data = malloc(MAX_DATA * 4);
    data_base = malloc(MAX_TENSORS * 4);

    node_op = malloc(MAX_NODES * 4);
    node_nin = malloc(MAX_NODES * 4);
    node_in = malloc(MAX_NODES * 8 * 4);
    node_nout = malloc(MAX_NODES * 4);
    node_out = malloc(MAX_NODES * 8 * 4);

    num_tensors = 0;
    num_nodes = 0;
    data_ptr = 0;

    return 0;
}

int skip_ws() {
    while (line[line_pos] == 32 || line[line_pos] == 9) {
        line_pos = line_pos + 1;
    }
    return 0;
}

int read_int_line() {
    int val;
    int neg;

    skip_ws();
    neg = 0;
    if (line[line_pos] == 45) {  /* '-' */
        neg = 1;
        line_pos = line_pos + 1;
    }

    val = 0;
    while (line[line_pos] >= 48 && line[line_pos] <= 57) {
        val = val * 10 + (line[line_pos] - 48);
        line_pos = line_pos + 1;
    }

    if (neg) val = 0 - val;
    return val;
}

int read_word(char *buf) {
    int i;
    skip_ws();
    i = 0;
    while (line[line_pos] != 32 && line[line_pos] != 9 &&
           line[line_pos] != 10 && line[line_pos] != 0) {
        buf[i] = line[line_pos];
        i = i + 1;
        line_pos = line_pos + 1;
    }
    buf[i] = 0;
    return i;
}

FILE *g_outfile;

int write_int_f(int val) {
    char buf[4];
    buf[0] = val & 255;
    buf[1] = (val / 256) & 255;
    buf[2] = (val / 65536) & 255;
    buf[3] = (val / 16777216) & 255;
    fwrite(buf, 1, 4, g_outfile);
    return 0;
}

FILE *g_infile;

int load_text(char *filename) {
    char word[64];
    int version;
    int n_tensors;
    int n_nodes;
    int i;
    int j;
    int nd;
    int sz;
    int op;
    int nin;
    int nout;

    g_infile = fopen(filename, "r");
    if (!g_infile) {
        printf("Error: cannot open input\n");
        return -1;
    }

    /* Read ONNX 1 */
    fgets(line, MAX_LINE, g_infile);
    line_pos = 0;
    read_word(word);
    version = read_int_line();
    printf("Version: %d\n", version);

    /* Read TENSORS n */
    fgets(line, MAX_LINE, g_infile);
    line_pos = 0;
    read_word(word);
    n_tensors = read_int_line();
    printf("Tensors: %d\n", n_tensors);

    /* Read tensors */
    i = 0;
    while (i < n_tensors) {
        fgets(line, MAX_LINE, g_infile);
        line_pos = 0;
        read_word(word);  /* T */

        read_word(names + i * MAX_NAME);
        nd = read_int_line();
        ndims[i] = nd;

        j = 0;
        while (j < nd) {
            dims[i * MAX_DIMS + j] = read_int_line();
            j = j + 1;
        }

        sz = read_int_line();
        sizes[i] = sz;
        data_base[i] = data_ptr;

        j = 0;
        while (j < sz && data_ptr < MAX_DATA) {
            data[data_ptr] = read_int_line();
            data_ptr = data_ptr + 1;
            j = j + 1;
        }

        i = i + 1;
    }
    num_tensors = n_tensors;

    /* Read NODES n */
    fgets(line, MAX_LINE, g_infile);
    line_pos = 0;
    read_word(word);
    n_nodes = read_int_line();
    printf("Nodes: %d\n", n_nodes);

    /* Read nodes */
    i = 0;
    while (i < n_nodes) {
        fgets(line, MAX_LINE, g_infile);
        line_pos = 0;
        read_word(word);  /* N */

        op = read_int_line();
        node_op[i] = op;

        nin = read_int_line();
        node_nin[i] = nin;

        j = 0;
        while (j < nin) {
            node_in[i * 8 + j] = read_int_line();
            j = j + 1;
        }

        nout = read_int_line();
        node_nout[i] = nout;

        j = 0;
        while (j < nout) {
            node_out[i * 8 + j] = read_int_line();
            j = j + 1;
        }

        i = i + 1;
    }
    num_nodes = n_nodes;

    fclose(g_infile);
    return 0;
}

int save_binary(char *filename) {
    int i;
    int j;
    int name_len;
    char *name;

    g_outfile = fopen(filename, "wb");
    if (!g_outfile) {
        printf("Error: cannot create output\n");
        return -1;
    }

    /* Header */
    write_int_f(1481526863);  /* 0x584E4E4F = "ONNX" */
    write_int_f(1);
    write_int_f(num_tensors);
    write_int_f(num_nodes);

    /* Tensors */
    i = 0;
    while (i < num_tensors) {
        name = names + i * MAX_NAME;
        name_len = strlen(name);
        write_int_f(name_len);
        fwrite(name, 1, name_len, g_outfile);

        write_int_f(ndims[i]);
        j = 0;
        while (j < ndims[i]) {
            write_int_f(dims[i * MAX_DIMS + j]);
            j = j + 1;
        }

        write_int_f(0);  /* data_type */
        write_int_f(sizes[i]);

        j = 0;
        while (j < sizes[i]) {
            write_int_f(data[data_base[i] + j]);
            j = j + 1;
        }

        i = i + 1;
    }

    /* Nodes */
    i = 0;
    while (i < num_nodes) {
        write_int_f(node_op[i]);
        write_int_f(node_nin[i]);

        j = 0;
        while (j < node_nin[i]) {
            write_int_f(node_in[i * 8 + j]);
            j = j + 1;
        }

        write_int_f(node_nout[i]);

        j = 0;
        while (j < node_nout[i]) {
            write_int_f(node_out[i * 8 + j]);
            j = j + 1;
        }

        i = i + 1;
    }

    fclose(g_outfile);
    printf("Wrote output\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: text_to_c4onnx input.txt output.c4onnx\n");
        return 1;
    }

    init();

    if (load_text(argv[1]) != 0) {
        return 1;
    }

    if (save_binary(argv[2]) != 0) {
        return 1;
    }

    printf("Done!\n");
    return 0;
}
