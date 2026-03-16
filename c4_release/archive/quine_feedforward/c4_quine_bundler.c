/*
 * C4 Neural Quine Bundler
 *
 * Creates a self-reproducing program that includes neural model weights.
 * The output program prints its own complete source code.
 *
 * Usage:
 *   ./c4_quine_bundler model.c4onnx > quine.c
 *   gcc -o quine quine.c
 *   ./quine > quine_copy.c
 *   diff quine.c quine_copy.c  # Should be identical
 *
 * Build:
 *   gcc -o c4_quine_bundler c4_quine_bundler.c
 */

int open(char *path, int mode);
int read(int fd, char *buf, int n);
int close(int fd);
char *malloc(int size);
int printf(char *fmt, ...);
int putchar(int c);

char *model_data;
int model_len;
char hex_chars[17];

int init_hex() {
    hex_chars[0] = '0'; hex_chars[1] = '1'; hex_chars[2] = '2'; hex_chars[3] = '3';
    hex_chars[4] = '4'; hex_chars[5] = '5'; hex_chars[6] = '6'; hex_chars[7] = '7';
    hex_chars[8] = '8'; hex_chars[9] = '9'; hex_chars[10] = 'a'; hex_chars[11] = 'b';
    hex_chars[12] = 'c'; hex_chars[13] = 'd'; hex_chars[14] = 'e'; hex_chars[15] = 'f';
    return 0;
}

int read_file(char *path, char **out_data, int *out_len) {
    int fd; int n; int total; char tmp[1024];
    char *buf; char *newbuf; int cap; int i;

    fd = open(path, 0);
    if (fd < 0) { printf("/* Error */\n"); return -1; }

    cap = 8192;
    buf = malloc(cap);
    total = 0;

    n = read(fd, tmp, 1024);
    while (n > 0) {
        if (total + n > cap) {
            cap = cap * 2;
            newbuf = malloc(cap);
            i = 0;
            while (i < total) { newbuf[i] = buf[i]; i = i + 1; }
            buf = newbuf;
        }
        i = 0;
        while (i < n) { buf[total + i] = tmp[i]; i = i + 1; }
        total = total + n;
        n = read(fd, tmp, 1024);
    }
    close(fd);
    *out_data = buf;
    *out_len = total;
    return 0;
}

int main(int argc, char **argv) {
    int i;
    int b;

    if (argc < 2) {
        printf("/* Usage: c4_quine_bundler model.c4onnx > quine.c */\n");
        return 1;
    }

    init_hex();
    if (read_file(argv[1], &model_data, &model_len) != 0) return 1;

    /* Generate the quine source code */
    printf("/* Neural Quine - prints its own source code */\n");
    printf("int printf(char *fmt, ...);\n");
    printf("char *malloc(int size);\n");
    printf("int putchar(int c);\n\n");

    /* Embed model as array */
    printf("char M[] = {\n");
    i = 0;
    while (i < model_len) {
        if ((i & 15) == 0) printf("    ");
        b = model_data[i] & 255;
        printf("0x%c%c", hex_chars[b >> 4], hex_chars[b & 15]);
        if (i + 1 < model_len) putchar(',');
        if ((i & 15) == 15 || i + 1 == model_len) putchar('\n');
        i = i + 1;
    }
    printf("};\n");
    printf("int ML = %d;\n\n", model_len);

    /* The quine data - contains the program structure */
    printf("char *Q[] = {\n");

    /* Part 0: Header */
    printf("    \"/* Neural Quine - prints its own source code */\\n\"\n");
    printf("    \"int printf(char *fmt, ...);\\n\"\n");
    printf("    \"char *malloc(int size);\\n\"\n");
    printf("    \"int putchar(int c);\\n\\n\",\n\n");

    /* Part 1: Model array header */
    printf("    \"char M[] = {\\n\",\n\n");

    /* Part 2: Model array footer + ML */
    printf("    \"};\\n\"\n");
    printf("    \"int ML = %d;\\n\\n\",\n\n", model_len);

    /* Part 3: Q array header */
    printf("    \"char *Q[] = {\\n\",\n\n");

    /* Part 4: Q array footer */
    printf("    \"};\\n\\n\",\n\n");

    /* Part 5: Helper functions and main - this is the tricky part */
    printf("    \"char H[] = \\\"0123456789abcdef\\\";\\n\"\n");
    printf("    \"int ph(int b) { b = b & 255; printf(\\\"0x%%c%%c\\\", H[b >> 4], H[b & 15]); return 0; }\\n\"\n");
    printf("    \"int pq(char *s) { int i; i = 0; while (s[i]) { if (s[i] == 10) printf(\\\"\\\\\\\\n\\\"); else if (s[i] == 34) printf(\\\"\\\\\\\\\\\\\\\"\\\" \\\"); else if (s[i] == 92) printf(\\\"\\\\\\\\\\\\\\\\\\\"); else putchar(s[i]); i = i + 1; } return 0; }\\n\"\n");
    printf("    \"int main() {\\n\"\n");
    printf("    \"    int i; int j;\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[0]);\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[1]);\\n\"\n");
    printf("    \"    i = 0;\\n\"\n");
    printf("    \"    while (i < ML) {\\n\"\n");
    printf("    \"        if ((i & 15) == 0) printf(\\\"    \\\");\\n\"\n");
    printf("    \"        ph(M[i]);\\n\"\n");
    printf("    \"        if (i + 1 < ML) putchar(44);\\n\"\n");
    printf("    \"        if ((i & 15) == 15 || i + 1 == ML) putchar(10);\\n\"\n");
    printf("    \"        i = i + 1;\\n\"\n");
    printf("    \"    }\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[2]);\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[3]);\\n\"\n");
    printf("    \"    j = 0;\\n\"\n");
    printf("    \"    while (j < 6) {\\n\"\n");
    printf("    \"        printf(\\\"    \\\\\\\"\\\");\\n\"\n");
    printf("    \"        pq(Q[j]);\\n\"\n");
    printf("    \"        if (j < 5) printf(\\\"\\\\\\\",\\\\n\\\\n\\\");\\n\"\n");
    printf("    \"        else printf(\\\"\\\\\\\"\\\\n\\\");\\n\"\n");
    printf("    \"        j = j + 1;\\n\"\n");
    printf("    \"    }\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[4]);\\n\"\n");
    printf("    \"    printf(\\\"%%s\\\", Q[5]);\\n\"\n");
    printf("    \"    return 0;\\n\"\n");
    printf("    \"}\\n\"\n");

    printf("};\n\n");

    /* Actual code */
    printf("char H[] = \"0123456789abcdef\";\n");
    printf("int ph(int b) { b = b & 255; printf(\"0x%%c%%c\", H[b >> 4], H[b & 15]); return 0; }\n");
    printf("int pq(char *s) { int i; i = 0; while (s[i]) { if (s[i] == 10) printf(\"\\\\n\"); else if (s[i] == 34) printf(\"\\\\\\\"\"); else if (s[i] == 92) printf(\"\\\\\\\\\"); else putchar(s[i]); i = i + 1; } return 0; }\n");
    printf("int main() {\n");
    printf("    int i; int j;\n");
    printf("    printf(\"%%s\", Q[0]);\n");
    printf("    printf(\"%%s\", Q[1]);\n");
    printf("    i = 0;\n");
    printf("    while (i < ML) {\n");
    printf("        if ((i & 15) == 0) printf(\"    \");\n");
    printf("        ph(M[i]);\n");
    printf("        if (i + 1 < ML) putchar(44);\n");
    printf("        if ((i & 15) == 15 || i + 1 == ML) putchar(10);\n");
    printf("        i = i + 1;\n");
    printf("    }\n");
    printf("    printf(\"%%s\", Q[2]);\n");
    printf("    printf(\"%%s\", Q[3]);\n");
    printf("    j = 0;\n");
    printf("    while (j < 6) {\n");
    printf("        printf(\"    \\\"\");\n");
    printf("        pq(Q[j]);\n");
    printf("        if (j < 5) printf(\"\\\",\\n\\n\");\n");
    printf("        else printf(\"\\\"\\n\");\n");
    printf("        j = j + 1;\n");
    printf("    }\n");
    printf("    printf(\"%%s\", Q[4]);\n");
    printf("    printf(\"%%s\", Q[5]);\n");
    printf("    return 0;\n");
    printf("}\n");

    return 0;
}
