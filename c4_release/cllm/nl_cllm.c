/* nl-cllm: number lines via neural VM */
int main() {
    int c;
    int line;
    int at_start;
    int d;

    line = 1;
    at_start = 1;

    c = getchar();
    while (c >= 0) {
        if (at_start) {
            /* Print line number */
            if (line >= 100) {
                d = line / 100;
                putchar('0' + d);
            } else {
                putchar(32);
            }
            if (line >= 10) {
                d = (line / 10) % 10;
                putchar('0' + d);
            } else {
                putchar(32);
            }
            putchar('0' + line % 10);
            putchar(9);
            at_start = 0;
        }
        putchar(c);
        if (c == 10) {
            line = line + 1;
            at_start = 1;
        }
        c = getchar();
    }
    return 0;
}
