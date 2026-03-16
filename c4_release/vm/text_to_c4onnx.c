/*
 * Text to C4 ONNX Binary Converter
 *
 * C4-compatible code that reads the text format from onnx_to_text.py
 * and writes the binary format for onnx_runtime_c4.c
 *
 * Text Format:
 *   ONNX 1
 *   TENSORS <n>
 *   T <name> <ndims> <d0> ... <size> <v0> ...
 *   NODES <n>
 *   N <op> <num_in> <in0> ... <num_out> <out0> ...
 *   END
 *
 * Build: gcc -o text_to_c4onnx text_to_c4onnx.c
 * Or:    ./c4 text_to_c4onnx.c
 *
 * Usage: ./text_to_c4onnx input.txt output.c4onnx
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Buffer sizes */
int MAX_NAME;
int MAX_LINE;
int MAX_TENSORS;
int MAX_DIMS;
int MAX_DATA;
int MAX_NODES;
int MAX_IO;

/* Tensor storage */
char **tensor_names;
int *tensor_ndims;
int **tensor_dims;
int *tensor_size;
int **tensor_data;
int num_tensors;

/* Node storage */
int *node_op;
int *node_num_inputs;
int **node_inputs;
int *node_num_outputs;
int **node_outputs;
int num_nodes;

/* Line buffer */
char *line;
int line_pos;

int init_storage() {
    int i;

    MAX_NAME = 128;
    MAX_LINE = 1000000;  /* 1MB line buffer for large tensor data */
    MAX_TENSORS = 512;
    MAX_DIMS = 8;
    MAX_DATA = 100000;
    MAX_NODES = 1024;
    MAX_IO = 16;

    line = malloc(MAX_LINE);

    tensor_names = malloc(MAX_TENSORS * sizeof(char *));
    tensor_ndims = malloc(MAX_TENSORS * sizeof(int));
    tensor_dims = malloc(MAX_TENSORS * sizeof(int *));
    tensor_size = malloc(MAX_TENSORS * sizeof(int));
    tensor_data = malloc(MAX_TENSORS * sizeof(int *));

    i = 0;
    while (i < MAX_TENSORS) {
        tensor_names[i] = malloc(MAX_NAME);
        tensor_dims[i] = malloc(MAX_DIMS * sizeof(int));
        tensor_data[i] = malloc(MAX_DATA * sizeof(int));
        i = i + 1;
    }

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

    num_tensors = 0;
    num_nodes = 0;

    return 0;
}

/* Skip whitespace in line buffer */
int skip_ws() {
    while (line[line_pos] == ' ' || line[line_pos] == '\t') {
        line_pos = line_pos + 1;
    }
    return 0;
}

/* Read integer from line buffer */
int read_int_from_line() {
    int val;
    int neg;

    skip_ws();

    neg = 0;
    if (line[line_pos] == '-') {
        neg = 1;
        line_pos = line_pos + 1;
    }

    val = 0;
    while (line[line_pos] >= '0' && line[line_pos] <= '9') {
        val = val * 10 + (line[line_pos] - '0');
        line_pos = line_pos + 1;
    }

    if (neg) {
        val = 0 - val;
    }

    return val;
}

/* Read word from line buffer */
int read_word(char *buf) {
    int i;

    skip_ws();

    i = 0;
    while (line[line_pos] != ' ' && line[line_pos] != '\t' &&
           line[line_pos] != '\n' && line[line_pos] != 0 && i < MAX_NAME - 1) {
        buf[i] = line[line_pos];
        i = i + 1;
        line_pos = line_pos + 1;
    }
    buf[i] = 0;

    return i;
}

/* Write 32-bit integer in little-endian */
int write_int(FILE *f, int val) {
    unsigned char buf[4];

    buf[0] = val & 255;
    buf[1] = (val >> 8) & 255;
    buf[2] = (val >> 16) & 255;
    buf[3] = (val >> 24) & 255;

    fwrite(buf, 1, 4, f);
    return 0;
}

int load_text(char *filename) {
    FILE *f;
    char word[128];
    int version;
    int n_tensors;
    int n_nodes;
    int i;
    int j;
    int ndims;
    int size;
    int op;
    int num_in;
    int num_out;

    f = fopen(filename, "r");
    if (!f) {
        printf("Error: cannot open %s\n", filename);
        return -1;
    }

    /* Read header: ONNX 1 */
    fgets(line, MAX_LINE, f);
    line_pos = 0;
    read_word(word);
    if (strcmp(word, "ONNX") != 0) {
        printf("Error: invalid header\n");
        fclose(f);
        return -1;
    }
    version = read_int_from_line();
    printf("Version: %d\n", version);

    /* Read TENSORS n */
    fgets(line, MAX_LINE, f);
    line_pos = 0;
    read_word(word);  /* TENSORS */
    n_tensors = read_int_from_line();
    printf("Tensors: %d\n", n_tensors);

    /* Read each tensor */
    i = 0;
    while (i < n_tensors) {
        fgets(line, MAX_LINE, f);
        line_pos = 0;

        read_word(word);  /* T */
        if (word[0] != 'T') {
            printf("Error: expected T at tensor %d\n", i);
            break;
        }

        read_word(tensor_names[i]);
        ndims = read_int_from_line();
        tensor_ndims[i] = ndims;

        j = 0;
        while (j < ndims) {
            tensor_dims[i][j] = read_int_from_line();
            j = j + 1;
        }

        size = read_int_from_line();
        tensor_size[i] = size;

        /* Read data values */
        j = 0;
        while (j < size && j < MAX_DATA) {
            tensor_data[i][j] = read_int_from_line();
            j = j + 1;
        }

        i = i + 1;
    }
    num_tensors = n_tensors;

    /* Read NODES n */
    fgets(line, MAX_LINE, f);
    line_pos = 0;
    read_word(word);  /* NODES */
    n_nodes = read_int_from_line();
    printf("Nodes: %d\n", n_nodes);

    /* Read each node */
    i = 0;
    while (i < n_nodes) {
        fgets(line, MAX_LINE, f);
        line_pos = 0;

        read_word(word);  /* N */
        if (word[0] != 'N') {
            printf("Error: expected N at node %d\n", i);
            break;
        }

        op = read_int_from_line();
        node_op[i] = op;

        num_in = read_int_from_line();
        node_num_inputs[i] = num_in;

        j = 0;
        while (j < num_in) {
            node_inputs[i][j] = read_int_from_line();
            j = j + 1;
        }

        num_out = read_int_from_line();
        node_num_outputs[i] = num_out;

        j = 0;
        while (j < num_out) {
            node_outputs[i][j] = read_int_from_line();
            j = j + 1;
        }

        i = i + 1;
    }
    num_nodes = n_nodes;

    fclose(f);
    return 0;
}

int save_binary(char *filename) {
    FILE *f;
    int i;
    int j;
    int name_len;

    f = fopen(filename, "wb");
    if (!f) {
        printf("Error: cannot create %s\n", filename);
        return -1;
    }

    /* Header */
    write_int(f, 0x584E4E4F);  /* "ONNX" magic */
    write_int(f, 1);            /* Version */
    write_int(f, num_tensors);
    write_int(f, num_nodes);

    /* Tensors */
    i = 0;
    while (i < num_tensors) {
        name_len = strlen(tensor_names[i]);
        write_int(f, name_len);
        fwrite(tensor_names[i], 1, name_len, f);

        write_int(f, tensor_ndims[i]);
        j = 0;
        while (j < tensor_ndims[i]) {
            write_int(f, tensor_dims[i][j]);
            j = j + 1;
        }

        write_int(f, 0);  /* data_type */
        write_int(f, tensor_size[i]);

        j = 0;
        while (j < tensor_size[i]) {
            write_int(f, tensor_data[i][j]);
            j = j + 1;
        }

        i = i + 1;
    }

    /* Nodes */
    i = 0;
    while (i < num_nodes) {
        write_int(f, node_op[i]);
        write_int(f, node_num_inputs[i]);

        j = 0;
        while (j < node_num_inputs[i]) {
            write_int(f, node_inputs[i][j]);
            j = j + 1;
        }

        write_int(f, node_num_outputs[i]);

        j = 0;
        while (j < node_num_outputs[i]) {
            write_int(f, node_outputs[i][j]);
            j = j + 1;
        }

        i = i + 1;
    }

    fclose(f);
    printf("Wrote %s\n", filename);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s input.txt output.c4onnx\n", argv[0]);
        return 1;
    }

    init_storage();

    if (load_text(argv[1]) != 0) {
        return 1;
    }

    if (save_binary(argv[2]) != 0) {
        return 1;
    }

    printf("Conversion complete!\n");
    return 0;
}
