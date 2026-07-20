/* memcmp_eq.c — memcmp equal case.
 *
 * Fills two buffers identically and compares them: memcmp returns 0, so the
 * program prints "EQ".  Exercises the memcmp loop (LC + subtract, §747-749)
 * running to completion with no differing byte.
 *
 * Exercises: malloc (x2), memset (x2), memcmp (equal -> 0), printf.
 */
int main() {
    char *a;
    char *b;

    a = malloc(8);
    b = malloc(8);
    memset(a, 90, 5);          /* 'Z' x5 */
    memset(b, 90, 5);
    if (memcmp(a, b, 5) == 0) {
        printf("EQ\n");
    } else {
        printf("NE\n");
    }
    return 0;
}
