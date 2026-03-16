/* tail-cllm: output last 10 lines via neural VM */
int main() {
    int *buf;
    int *starts;
    int c;
    int pos;
    int lines;
    int i;
    int start;

    buf = malloc(65536);
    starts = malloc(8192);
    pos = 0;
    lines = 0;
    starts[0] = 0;

    c = getchar();
    while (c >= 0) {
        if (pos < 8192) {
            buf[pos] = c;
            if (c == 10) {
                lines = lines + 1;
                if (lines < 1024) starts[lines] = pos + 1;
            }
            pos = pos + 1;
        }
        c = getchar();
    }

    /* Output last 10 lines */
    start = lines - 10;
    if (start < 0) start = 0;
    i = starts[start];
    while (i < pos) {
        putchar(buf[i]);
        i = i + 1;
    }
    return 0;
}
