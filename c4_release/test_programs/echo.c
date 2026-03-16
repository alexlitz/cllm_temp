/* echo.c - Echo command line arguments
 *
 * Usage: echo [args...]
 * Prints all arguments separated by spaces, followed by newline.
 * Uses int* workaround for char** compiler bug.
 */

int main(int argc, char **argv) {
    int i;
    char *s;
    int *ap;

    ap = (int *)argv;
    i = 1;
    while (i < argc) {
        if (i > 1) {
            putchar(32);
        }
        /* Print each character of argument */
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
