/* rev-cllm: reverse each line - all arithmetic via neural VM */
int main() {
    int c;
    int *buf;
    int len;
    buf = malloc(2048);
    len = 0;
    c = getchar();
    while (c >= 0) {
        if (c == 10) {
            while (len > 0) {
                len = len - 1;
                putchar(buf[len]);
            }
            putchar(10);
            len = 0;
        } else {
            if (len < 256) {
                buf[len] = c;
                len = len + 1;
            }
        }
        c = getchar();
    }
    while (len > 0) {
        len = len - 1;
        putchar(buf[len]);
    }
    return 0;
}
