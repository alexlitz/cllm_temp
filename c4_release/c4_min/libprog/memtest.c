/* memtest.c — malloc + memset + memcmp exerciser (ported from cllm/memtest_cllm.c).
 *
 * PORT NOTE: the original cllm/memtest_cllm.c used putchar(), which was REMOVED
 * from this branch's ISA (the only visible-output channel left is PRTF / printf,
 * §Tool Use Mode).  Every putchar(x) is rewritten as printf("%c", x) so the
 * program's stdout is byte-for-byte identical while going through the surviving
 * I/O opcode.  The malloc/memset/memcmp calls are the c4 runtime-library
 * routines (compiled to bytecode, run neurally — NOT tool calls).
 *
 * Exercises: malloc (two allocations), memset (fill 10 bytes with 'A'), memcmp
 * (two equal buffers -> 0), printf (%c visible output).
 */
int main() {
    char *buf;
    char *buf2;
    int i;

    buf = malloc(16);
    memset(buf, 65, 10);        /* fill 10 bytes with 'A' (65) */
    i = 0;
    while (i < 10) {
        printf("%c", *(buf + i));   /* was putchar(*(buf+i)) */
        i = i + 1;
    }
    printf("%c", 10);               /* newline (was putchar(10)) */

    buf2 = malloc(16);
    memset(buf2, 65, 10);
    if (memcmp(buf, buf2, 10) == 0) { printf("%c", 89); }  /* 'Y' if equal */
    printf("%c", 10);
    return 0;
}
