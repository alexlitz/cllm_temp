/* printf_str.c — printf %s (string pointer) formatting.
 *
 * Prints a data-segment string literal via %s, mixed with %d and literals.
 * Exercises the printf %s path (the arg is a pointer to a C string in the data
 * segment, resolved by the runner).
 *
 * Exercises: printf (%s, %d, literals), string literal in data segment.
 */
int main() {
    printf("hi %s #%d\n", "world", 3);
    return 0;
}
