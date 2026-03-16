/* yes.c - Repeatedly output a line
 *
 * Usage: yes [string]
 * Outputs "y" (or string if provided) repeatedly until killed.
 * For testing, we limit to 100 iterations.
 */

int main(int argc, char **argv) {
    char *msg;
    int count;
    int i;

    /* Default message is "y" */
    if (argc > 1) {
        msg = argv[1];
    } else {
        msg = "y";
    }

    /* Output 100 times (limited for testing) */
    count = 100;
    i = 0;
    while (i < count) {
        /* Print message */
        char *s;
        s = msg;
        while (*s) {
            putchar(*s);
            s = s + 1;
        }
        putchar('\n');
        i = i + 1;
    }

    return 0;
}
