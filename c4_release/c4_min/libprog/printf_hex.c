/* printf_hex.c — printf %x (hex) formatting.
 *
 * Prints values in hex with %x, plus %d and literals.  Exercises the printf %x
 * path (lowercase hex, no leading 0x, matching the c4 / classic printf subset).
 *
 * Exercises: printf (%x, %d, literals) — no heap.
 */
int main() {
    printf("%x\n", 255);
    printf("%x %d\n", 16, 16);
    return 0;
}
