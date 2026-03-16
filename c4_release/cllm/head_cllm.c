/* head-cllm: output first 10 lines - all arithmetic via neural VM */
int main() {
    int c;
    int lines;
    lines = 0;
    c = getchar();
    while (c >= 0) {
        if (lines >= 10) return 0;
        putchar(c);
        if (c == 10) lines = lines + 1;
        c = getchar();
    }
    return 0;
}
