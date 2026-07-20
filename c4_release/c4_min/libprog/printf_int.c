/* printf_int.c — printf integer + char formatting.
 *
 * Exercises the c4 printf format subset: %d (signed decimal), %c (char), and
 * literal bytes, across several calls with different args.
 *
 * Exercises: printf (%d, %c, literals) — no heap.
 */
int main() {
    printf("n=%d\n", 7);
    printf("m=%d\n", 123);
    printf("%c%c%c\n", 65, 66, 67);
    return 0;
}
