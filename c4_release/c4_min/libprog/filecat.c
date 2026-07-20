/* filecat.c — open + read + close + print a file (cat-style).
 *
 * Opens a named file, reads its bytes into a heap buffer, closes it, then
 * prints the buffer.  This is the file-IO tool-call path (OPEN/READ/CLOS are
 * TOOL_CALLs, §Tool Use Mode) plus a malloc'd buffer and a print loop.
 *
 * PORT NOTE: written in the file-IO model of this branch (OPEN/READ/CLOS/PRTF);
 * the golden build wraps the c4 open()/read()/close() around a real file so gcc
 * reproduces the same bytes.  The c4 harness seeds the same file into the stub
 * filesystem.  putchar is expressed as printf("%c", ...).
 *
 * Exercises: OPEN, READ, CLOS (file-IO tool calls), malloc, printf.
 */
int main() {
    char *buf;
    int fd;
    int n;
    int i;

    buf = malloc(32);
    fd = open("greeting.txt", 0);   /* 0 = O_RDONLY */
    n = read(fd, buf, 16);
    close(fd);

    i = 0;
    while (i < n) {
        printf("%c", *(buf + i));
        i = i + 1;
    }
    return 0;
}
