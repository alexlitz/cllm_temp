/* malloc_printf.c — malloc a buffer, memset it, report the fill via %d + %c.
 *
 * Ties the heap library to formatted output: allocate, fill with a value, then
 * print that value back both as a decimal (%d) and as the character it fills
 * with (%c), reading the byte out of the heap buffer.
 *
 * Exercises: malloc, memset, printf (%d + %c reading a heap byte).
 */
int main() {
    char *p;

    p = malloc(4);
    memset(p, 72, 3);          /* 'H' x3 */
    printf("byte=%d char=%c\n", *p, *p);
    printf("%c%c%c\n", *p, *(p + 1), *(p + 2));
    return 0;
}
