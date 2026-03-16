/*
 * ASCII Mandelbrot Set - C4 Compatible
 *
 * Renders the Mandelbrot set using fixed-point arithmetic.
 * Uses only c4-compatible C: no floats, while loops, if/else.
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -o mandelbrot_c4 mandelbrot_c4.c
 *
 * Usage:
 *   ./mandelbrot_c4           # 60x30 default
 *   ./mandelbrot_c4 80 40     # custom size
 */

/* Fixed-point scale (10.10 format for Mandelbrot precision) */
int SCALE;

/* Mandelbrot iteration */
int mandel_iter(int cx, int cy, int max_iter) {
    int zx;
    int zy;
    int zx2;
    int zy2;
    int tmp;
    int iter;

    zx = 0;
    zy = 0;
    iter = 0;

    while (iter < max_iter) {
        zx2 = (zx * zx) / SCALE;
        zy2 = (zy * zy) / SCALE;

        /* Check escape: |z|^2 > 4 */
        if (zx2 + zy2 > 4 * SCALE) {
            return iter;
        }

        /* z = z^2 + c */
        tmp = zx2 - zy2 + cx;
        zy = 2 * zx * zy / SCALE + cy;
        zx = tmp;

        iter = iter + 1;
    }

    return max_iter;
}

/* Character mapping for iteration count */
int iter_to_char(int iter, int max_iter) {
    char *chars;
    int idx;

    if (iter == max_iter) return '#';

    chars = " .:-=+*%@#";
    idx = (iter * 10) / max_iter;
    if (idx > 9) idx = 9;

    return chars[idx];
}

/* Parse integer from string */
int parse_int(char *s) {
    int n;
    n = 0;
    while (*s >= '0') {
        if (*s <= '9') {
            n = n * 10 + (*s - '0');
        }
        s = s + 1;
    }
    return n;
}

int main(int argc, char **argv) {
    int width;
    int height;
    int max_iter;
    int x_min;
    int x_max;
    int y_min;
    int y_max;
    int row;
    int col;
    int cx;
    int cy;
    int iter;
    int ch;
    int dx;
    int dy;

    /* Initialize fixed-point scale */
    SCALE = 1024;

    /* Default dimensions */
    width = 60;
    height = 30;

    /* Parse command line args */
    if (argc > 1) {
        width = parse_int(argv[1]);
    }
    if (argc > 2) {
        height = parse_int(argv[2]);
    }

    max_iter = 50;

    /* Mandelbrot view: x in [-2.5, 1.0], y in [-1.0, 1.0] */
    x_min = 0 - 2560;  /* -2.5 * 1024 */
    x_max = 1024;      /* 1.0 * 1024 */
    y_min = 0 - 1024;  /* -1.0 * 1024 */
    y_max = 1024;      /* 1.0 * 1024 */

    dx = (x_max - x_min) / width;
    dy = (y_max - y_min) / height;

    /* Print header */
    printf("Mandelbrot Set %dx%d (C4 Fixed-Point)\n", width, height);

    /* Print top border */
    printf("+");
    col = 0;
    while (col < width) {
        printf("-");
        col = col + 1;
    }
    printf("+\n");

    /* Render rows */
    row = 0;
    while (row < height) {
        cy = y_max - row * dy;

        printf("|");

        col = 0;
        while (col < width) {
            cx = x_min + col * dx;

            iter = mandel_iter(cx, cy, max_iter);
            ch = iter_to_char(iter, max_iter);

            printf("%c", ch);

            col = col + 1;
        }

        printf("|\n");
        row = row + 1;
    }

    /* Print bottom border */
    printf("+");
    col = 0;
    while (col < width) {
        printf("-");
        col = col + 1;
    }
    printf("+\n");

    return 0;
}
