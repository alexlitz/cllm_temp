int main() {
    char *buf;
    char *buf2;
    int i;

    buf = malloc(16);
    memset(buf, 65, 10);
    i = 0;
    while (i < 10) {
        putchar(*(buf + i));
        i = i + 1;
    }
    putchar(10);
    buf2 = malloc(16);
    memset(buf2, 65, 10);
    if (memcmp(buf, buf2, 10) == 0) { putchar(89); }
    putchar(10);
    return 0;
}
