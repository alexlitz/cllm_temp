/* cat.c - Concatenate and print input
 *
 * Usage: cat
 * Reads from stdin and writes to stdout until EOF.
 * This tests the streaming I/O system.
 */

int main() {
    int c;

    /* Read until EOF (-1) */
    while ((c = getchar()) != -1) {
        putchar(c);
    }

    return 0;
}
