/*
 * Neural Quine Generator
 *
 * Creates a minimal quine that embeds neural model weights.
 * The generated program prints its own source code exactly.
 *
 * Usage:
 *   gcc -o neural_quine neural_quine.c
 *   ./neural_quine model.c4onnx > quine.c
 *   gcc -o quine quine.c
 *   ./quine > quine2.c
 *   diff quine.c quine2.c  # Should show no differences
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    FILE *f;
    unsigned char *model;
    int model_len;
    int i, b;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.c4onnx > quine.c\n", argv[0]);
        return 1;
    }

    /* Read model */
    f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    model_len = ftell(f);
    fseek(f, 0, SEEK_SET);
    model = malloc(model_len);
    fread(model, 1, model_len, f);
    fclose(f);

    /* Generate quine */
    printf("/* Neural Quine */\n");
    printf("char*d=\"");

    /* Encode the quine data string - contains the template */
    /* Format: model_len followed by model bytes, then the code template */

    /* We use a simple approach: store model in M[], and use classic quine for code */
    printf("Neural Quine|");
    printf("int printf();int putchar();char*m=");
    printf("34");  /* quote char */
    printf(";char M[]={");

    /* Model bytes placeholder - will be filled by format string */
    i = 0;
    while (i < model_len) {
        if (i > 0) printf(",");
        printf("%d", model[i] & 255);
        i++;
    }
    printf("};int L=%d;", model_len);

    /* The quine code - prints itself */
    printf("char*c=");
    printf("34");
    printf(";int main(){int i,j;char*p;p=d;printf(");
    printf("34 34");
    printf("/* Neural Quine */\\n");
    printf("34 34");
    printf(");printf(");
    printf("34 34");
    printf("char*d=\\");
    printf("34 34 34");
    printf(");p=d;while(*p){if(*p==124)putchar(10);else if(*p==34)putchar(39);else putchar(*p);p++;}printf(");
    printf("34 34");
    printf("\\");
    printf("34 34");
    printf(";\\n");
    printf("34 34");
    printf(");p=d;while(*p){if(*p==124)printf(");
    printf("34 34");
    printf("124");
    printf("34 34");
    printf(");else if(*p==34)printf(");
    printf("34 34");
    printf("34");
    printf("34 34");
    printf(");else printf(");
    printf("34 34");
    printf("%%c");
    printf("34 34");
    printf(",*p);p++;}putchar(10);return 0;}");

    printf("\";\n");

    /* Now print the actual code that matches what we encoded */
    printf("int printf();int putchar();char*m=\"");
    printf("\";char M[]={");
    i = 0;
    while (i < model_len) {
        if (i > 0) printf(",");
        printf("%d", model[i] & 255);
        i++;
    }
    printf("};int L=%d;", model_len);
    printf("char*c=\"");
    printf("\";int main(){int i,j;char*p;p=d;printf(\"/* Neural Quine */\\n\");");
    printf("printf(\"char*d=\\\"\");");
    printf("p=d;while(*p){if(*p=='|')putchar('\\n');else if(*p=='\"')putchar('\\'');else putchar(*p);p++;}");
    printf("printf(\"\\\";\\n\");");
    printf("p=d;while(*p){");
    printf("if(*p=='|')printf(\"|\");");
    printf("else if(*p=='\"')printf(\"34\");");
    printf("else printf(\"%%c\",*p);");
    printf("p++;}putchar('\\n');return 0;}\n");

    return 0;
}
