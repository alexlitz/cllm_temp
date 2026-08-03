/* Bit-exact IEEE-754 single ground truth: read (op, a_bits, b_bits) lines on
 * stdin, print the raw 32-bit result bits of the float op.  gcc round-to-nearest
 * even is the reference the c4 megablocks + the Python oracle must match. */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float    fval(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

int main(void) {
    char opname[8];
    uint32_t ab, bb;
    while (scanf("%7s %u %u", opname, &ab, &bb) == 3) {
        float a = fval(ab), b = fval(bb), r = 0.0f;
        if      (!strcmp(opname, "F_ADD")) r = a + b;
        else if (!strcmp(opname, "F_SUB")) r = a - b;
        else if (!strcmp(opname, "F_MUL")) r = a * b;
        else if (!strcmp(opname, "F_DIV")) r = a / b;
        printf("%u\n", bits(r));
    }
    return 0;
}
