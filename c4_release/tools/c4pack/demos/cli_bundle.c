/* cli_bundle.c -- an echo/cat/yes multitool in ONE c4 program.
 *
 * The classic streaming CLI trio bundled into one binary, dispatched by the
 * first argument:  cli_bundle echo A B C  |  cli_bundle cat  |  cli_bundle yes S N
 *
 * This is the NATIVE (argv + getchar/putchar) equivalent of the neural
 * echo-cat-yes CLI bundle -- a single self-contained c4 program packaged by
 * c4pack into one standalone binary.  Uses the int* argv workaround (the c4
 * compiler's char** handling) matching test_programs/echo.c.
 */

int streq(char *a, char *b) {
    while (*a && *b) {
        if (*a != *b) return 0;
        a = a + 1;
        b = b + 1;
    }
    if (*a == 0) {
        if (*b == 0) return 1;
    }
    return 0;
}

int do_echo(int argc, int *ap) {
    int i;
    char *s;
    i = 2;
    while (i < argc) {
        if (i > 2) putchar(32);
        s = (char *)ap[i];
        while (*s) {
            putchar(*s);
            s = s + 1;
        }
        i = i + 1;
    }
    putchar(10);
    return 0;
}

int do_cat() {
    int c;
    while ((c = getchar()) != -1) {
        putchar(c);
    }
    return 0;
}

int str_to_int(char *s) {
    int n;
    n = 0;
    while (*s >= 48) {
        if (*s > 57) return n;
        n = n * 10 + (*s - 48);
        s = s + 1;
    }
    return n;
}

int do_yes(int argc, int *ap) {
    char *s;
    int n;
    int k;
    char *word;
    if (argc > 2) word = (char *)ap[2]; else word = (char *)0;
    if (argc > 3) n = str_to_int((char *)ap[3]); else n = 3;
    k = 0;
    while (k < n) {
        if (word) {
            s = word;
            while (*s) { putchar(*s); s = s + 1; }
        } else {
            putchar(121);  /* 'y' */
        }
        putchar(10);
        k = k + 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int *ap;
    char *cmd;
    ap = (int *)argv;
    if (argc < 2) {
        putchar(10);
        return 0;
    }
    cmd = (char *)ap[1];
    if (streq(cmd, "echo")) return do_echo(argc, ap);
    if (streq(cmd, "cat")) return do_cat();
    if (streq(cmd, "yes")) return do_yes(argc, ap);
    /* unknown: print the command back */
    while (*cmd) { putchar(*cmd); cmd = cmd + 1; }
    putchar(10);
    return 0;
}
