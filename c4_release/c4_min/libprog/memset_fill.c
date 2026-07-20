/* memset_fill.c — memset fill + verify.
 *
 * Fills a heap buffer with a byte value, then reads every byte back and prints
 * it, proving the memset loop wrote each cell (a loop of SC, §747-749).
 *
 * Exercises: malloc, memset (fill 6 bytes with '*'=42), printf (%c) in a verify loop.
 */
int main() {
    char *p;
    int i;

    p = malloc(8);
    memset(p, 42, 6);          /* fill 6 bytes with '*' */
    i = 0;
    while (i < 6) {
        printf("%c", *(p + i));
        i = i + 1;
    }
    printf("%c", 10);
    return 0;
}
