/*
 * Color Mandelbrot PNG Generator - C4 Compatible
 *
 * Generates a valid PNG file with the Mandelbrot set in color.
 * Uses RGB color palette based on iteration count.
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -o mandelbrot_color_png_c4 mandelbrot_color_png_c4.c
 *
 * Usage:
 *   ./mandelbrot_color_png_c4 > mandelbrot_color.png
 *   ./mandelbrot_color_png_c4 128 128 > mandelbrot128.png
 */

int putbyte(int b) {
    printf("%c", b & 255);
    return 0;
}

int put32be(int v) {
    putbyte((v >> 24) & 255);
    putbyte((v >> 16) & 255);
    putbyte((v >> 8) & 255);
    putbyte(v & 255);
    return 0;
}

/* CRC32 */
int *crc_table;
int crc_initialized;

int init_crc_table() {
    int n;
    int c;
    int k;

    if (crc_initialized) return 0;

    crc_table = malloc(256 * 4);
    n = 0;
    while (n < 256) {
        c = n;
        k = 0;
        while (k < 8) {
            if (c & 1) {
                c = 0xEDB88320 ^ ((c >> 1) & 0x7FFFFFFF);
            } else {
                c = (c >> 1) & 0x7FFFFFFF;
            }
            k = k + 1;
        }
        crc_table[n] = c;
        n = n + 1;
    }
    crc_initialized = 1;
    return 0;
}

int update_crc(int crc, int byte) {
    int idx;
    idx = (crc ^ byte) & 255;
    return crc_table[idx] ^ ((crc >> 8) & 0x00FFFFFF);
}

/* Adler32 */
int adler_a;
int adler_b;

int adler_init() {
    adler_a = 1;
    adler_b = 0;
    return 0;
}

int adler_update(int byte) {
    adler_a = (adler_a + byte) % 65521;
    adler_b = (adler_b + adler_a) % 65521;
    return 0;
}

int adler_get() {
    return (adler_b << 16) | adler_a;
}

int chunk_crc;

int chunk_start(char *type) {
    chunk_crc = 0xFFFFFFFF;
    chunk_crc = update_crc(chunk_crc, type[0]);
    chunk_crc = update_crc(chunk_crc, type[1]);
    chunk_crc = update_crc(chunk_crc, type[2]);
    chunk_crc = update_crc(chunk_crc, type[3]);
    putbyte(type[0]);
    putbyte(type[1]);
    putbyte(type[2]);
    putbyte(type[3]);
    return 0;
}

int chunk_byte(int b) {
    chunk_crc = update_crc(chunk_crc, b);
    putbyte(b);
    return 0;
}

int chunk_end() {
    int crc;
    crc = chunk_crc ^ 0xFFFFFFFF;
    put32be(crc);
    return 0;
}

/* Fixed-point */
int SCALE;

/* Color palette (RGB triplets) */
int *palette_r;
int *palette_g;
int *palette_b;
int palette_size;

int init_palette(int max_iter) {
    int i;
    int t;

    palette_size = max_iter + 1;
    palette_r = malloc(palette_size * 4);
    palette_g = malloc(palette_size * 4);
    palette_b = malloc(palette_size * 4);

    i = 0;
    while (i < max_iter) {
        /* Create a nice gradient: blue -> cyan -> green -> yellow -> red */
        t = (i * 1024) / max_iter;

        if (t < 256) {
            /* Blue to cyan */
            palette_r[i] = 0;
            palette_g[i] = t;
            palette_b[i] = 255;
        } else if (t < 512) {
            /* Cyan to green */
            palette_r[i] = 0;
            palette_g[i] = 255;
            palette_b[i] = 255 - (t - 256);
        } else if (t < 768) {
            /* Green to yellow */
            palette_r[i] = (t - 512);
            palette_g[i] = 255;
            palette_b[i] = 0;
        } else {
            /* Yellow to red */
            palette_r[i] = 255;
            palette_g[i] = 255 - (t - 768);
            palette_b[i] = 0;
        }
        i = i + 1;
    }

    /* Black for points in the set */
    palette_r[max_iter] = 0;
    palette_g[max_iter] = 0;
    palette_b[max_iter] = 0;

    return 0;
}

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

        if (zx2 + zy2 > 4 * SCALE) {
            return iter;
        }

        tmp = zx2 - zy2 + cx;
        zy = 2 * zx * zy / SCALE + cy;
        zx = tmp;

        iter = iter + 1;
    }

    return max_iter;
}

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
    int dx;
    int dy;
    int raw_size;
    int block_size;
    int blocks;
    int remaining;
    int i;
    int block;
    int is_last;
    int this_size;
    int y;
    int x;
    int rgb_state;
    int cx;
    int cy;
    int iter;

    SCALE = 1024;
    crc_initialized = 0;
    init_crc_table();

    width = 64;
    height = 64;

    if (argc > 1) {
        width = parse_int(argv[1]);
    }
    if (argc > 2) {
        height = parse_int(argv[2]);
    }

    max_iter = 100;
    init_palette(max_iter);

    x_min = 0 - 2560;
    x_max = 1024;
    y_min = 0 - 1024;
    y_max = 1024;

    dx = (x_max - x_min) / width;
    dy = (y_max - y_min) / height;

    /* PNG signature */
    putbyte(137);
    putbyte(80);
    putbyte(78);
    putbyte(71);
    putbyte(13);
    putbyte(10);
    putbyte(26);
    putbyte(10);

    /* IHDR */
    put32be(13);
    chunk_start("IHDR");
    chunk_byte((width >> 24) & 255);
    chunk_byte((width >> 16) & 255);
    chunk_byte((width >> 8) & 255);
    chunk_byte(width & 255);
    chunk_byte((height >> 24) & 255);
    chunk_byte((height >> 16) & 255);
    chunk_byte((height >> 8) & 255);
    chunk_byte(height & 255);
    chunk_byte(8);   /* Bit depth */
    chunk_byte(2);   /* Color type: RGB */
    chunk_byte(0);
    chunk_byte(0);
    chunk_byte(0);
    chunk_end();

    /* IDAT */
    /* Each row: 1 filter byte + width * 3 (RGB) */
    raw_size = height * (1 + width * 3);
    block_size = 65535;
    blocks = (raw_size + block_size - 1) / block_size;

    put32be(2 + blocks * 5 + raw_size + 4);
    chunk_start("IDAT");

    chunk_byte(0x78);
    chunk_byte(0x01);

    adler_init();

    remaining = raw_size;
    y = 0;
    x = -1;
    rgb_state = 0;
    iter = 0;

    block = 0;
    while (block < blocks) {
        is_last = (block == blocks - 1) ? 1 : 0;
        this_size = remaining;
        if (this_size > block_size) {
            this_size = block_size;
        }

        chunk_byte(is_last);
        chunk_byte(this_size & 255);
        chunk_byte((this_size >> 8) & 255);
        chunk_byte((0 - this_size - 1) & 255);
        chunk_byte(((0 - this_size - 1) >> 8) & 255);

        i = 0;
        while (i < this_size) {
            if (x == -1) {
                chunk_byte(0);
                adler_update(0);
                x = 0;
                rgb_state = 0;
            } else {
                if (rgb_state == 0) {
                    cy = y_max - y * dy;
                    cx = x_min + x * dx;
                    iter = mandel_iter(cx, cy, max_iter);
                    chunk_byte(palette_r[iter]);
                    adler_update(palette_r[iter]);
                    rgb_state = 1;
                } else if (rgb_state == 1) {
                    chunk_byte(palette_g[iter]);
                    adler_update(palette_g[iter]);
                    rgb_state = 2;
                } else {
                    chunk_byte(palette_b[iter]);
                    adler_update(palette_b[iter]);
                    rgb_state = 0;
                    x = x + 1;
                    if (x >= width) {
                        x = -1;
                        y = y + 1;
                    }
                }
            }
            i = i + 1;
        }

        remaining = remaining - this_size;
        block = block + 1;
    }

    i = adler_get();
    chunk_byte((i >> 24) & 255);
    chunk_byte((i >> 16) & 255);
    chunk_byte((i >> 8) & 255);
    chunk_byte(i & 255);

    chunk_end();

    /* IEND */
    put32be(0);
    chunk_start("IEND");
    chunk_end();

    free(crc_table);
    free(palette_r);
    free(palette_g);
    free(palette_b);
    return 0;
}
