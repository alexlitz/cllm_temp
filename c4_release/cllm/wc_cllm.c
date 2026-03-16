/* wc-cllm: count lines, words, chars - all arithmetic via neural VM */
int main() {
    int c;
    int lines;
    int words;
    int chars;
    int in_word;
    int d;

    lines = 0;
    words = 0;
    chars = 0;
    in_word = 0;

    c = getchar();
    while (c >= 0) {
        chars = chars + 1;
        if (c == 10) {
            lines = lines + 1;
            in_word = 0;
        } else if (c == 32 || c == 9) {
            in_word = 0;
        } else {
            if (in_word == 0) {
                words = words + 1;
                in_word = 1;
            }
        }
        c = getchar();
    }

    /* Print counts using putchar */
    if (lines >= 100) { d = lines / 100; putchar(48 + d); }
    if (lines >= 10) { d = (lines / 10) % 10; putchar(48 + d); }
    putchar(48 + lines % 10);
    putchar(32);
    if (words >= 100) { d = words / 100; putchar(48 + d); }
    if (words >= 10) { d = (words / 10) % 10; putchar(48 + d); }
    putchar(48 + words % 10);
    putchar(32);
    if (chars >= 100) { d = chars / 100; putchar(48 + d); }
    if (chars >= 10) { d = (chars / 10) % 10; putchar(48 + d); }
    putchar(48 + chars % 10);
    putchar(10);

    return 0;
}
