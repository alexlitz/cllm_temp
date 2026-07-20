/* memcmp_ne.c — memcmp unequal case.
 *
 * Fills two buffers, then perturbs one byte so memcmp finds a difference and
 * returns non-zero -> the program prints "NE".  Exercises the memcmp loop
 * taking the early-exit (differing-byte) branch.
 *
 * Exercises: malloc (x2), memset (x2), SC (byte poke), memcmp (unequal), printf.
 */
int main() {
    char *a;
    char *b;

    a = malloc(8);
    b = malloc(8);
    memset(a, 88, 5);          /* 'X' x5 */
    memset(b, 88, 5);
    *(b + 2) = 89;             /* perturb b[2] = 'Y' */
    if (memcmp(a, b, 5) == 0) {
        printf("EQ\n");
    } else {
        printf("NE\n");
    }
    return 0;
}
