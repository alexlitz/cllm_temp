/* malloc_free_reuse.c — malloc + free reuse pattern.
 *
 * Allocates a buffer, fills it, prints a marker, frees it, allocates again and
 * reuses.  The c4 free() is a bump-allocator no-op (§689-691: overwriting with
 * zero is the free), so this exercises the free() call path + a second malloc.
 *
 * Exercises: malloc (x2), free, memset, printf (%c + %d).
 */
int main() {
    char *a;
    char *b;

    a = malloc(8);
    memset(a, 66, 4);          /* 'B' x4 */
    printf("%c%c%c%c\n", *a, *(a + 1), *(a + 2), *(a + 3));
    free(a);

    b = malloc(8);
    memset(b, 67, 3);          /* 'C' x3 */
    printf("%c%c%c\n", *b, *(b + 1), *(b + 2));
    printf("%d\n", 42);
    return 0;
}
