int main(int argc, char **argv) {
    int i;
    char *s;

    i = 0;
    while (i < argc) {
        s = *(char **)(argv + i * 8);
        while (*s) { putchar(*s); s = s + 1; }
        putchar(10);
        i = i + 1;
    }
    return 0;
}
