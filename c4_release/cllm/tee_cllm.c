/* tee-cllm: copy stdin to stdout via neural VM (no file output in VM) */
int main() {
    int c;
    c = getchar();
    while (c >= 0) {
        putchar(c);
        c = getchar();
    }
    return 0;
}
