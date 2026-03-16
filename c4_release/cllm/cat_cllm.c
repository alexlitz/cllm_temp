/* cat-cllm: copy stdin to stdout - all arithmetic via neural VM */
int main() {
    int c;
    c = getchar();
    while (c >= 0) {
        putchar(c);
        c = getchar();
    }
    return 0;
}
