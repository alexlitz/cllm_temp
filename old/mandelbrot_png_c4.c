/*
 * Mandelbrot PNG Generator - C4 Compatible
 *
 * Generates a valid PNG file with the Mandelbrot set.
 * Uses uncompressed DEFLATE blocks to avoid complex compression.
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -o mandelbrot_png_c4 mandelbrot_png_c4.c
 *
 * Usage:
 *   ./mandelbrot_png_c4 > mandelbrot.png
 *   ./mandelbrot_png_c4 64 64 > mandelbrot64.png
 */

/* PNG uses big-endian, we need byte output */
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

int put16le(int v) {
    putbyte(v & 255);
    putbyte((v >> 8) & 255);
    return 0;
}

/* CRC32 calculation */
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

/* Adler32 for zlib */
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

/* Global chunk buffer for CRC calculation */
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

/* Fixed-point for Mandelbrot */
int SCALE;

int mandel_color(int cx, int cy, int max_iter) {
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
            /* Map iteration to grayscale */
            return 255 - (iter * 255 / max_iter);
        }

        tmp = zx2 - zy2 + cx;
        zy = 2 * zx * zy / SCALE + cy;
        zx = tmp;

        iter = iter + 1;
    }

    return 0;  /* In set = black */
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
    int dx;
    int dy;
    int row;
    int col;
    int cx;
    int cy;
    int gray;
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

    /* Initialize */
    SCALE = 1024;
    crc_initialized = 0;
    init_crc_table();

    /* Default dimensions (small for demo) */
    width = 32;
    height = 32;

    if (argc > 1) {
        width = parse_int(argv[1]);
    }
    if (argc > 2) {
        height = parse_int(argv[2]);
    }

    max_iter = 50;

    /* Mandelbrot view */
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

    /* IHDR chunk */
    put32be(13);  /* Length */
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
    chunk_byte(0);   /* Color type: grayscale */
    chunk_byte(0);   /* Compression */
    chunk_byte(0);   /* Filter */
    chunk_byte(0);   /* Interlace */
    chunk_end();

    /* Calculate IDAT size */
    /* Each row: 1 filter byte + width pixels */
    raw_size = height * (1 + width);

    /* Uncompressed DEFLATE: 2 byte zlib header + blocks + 4 byte adler32 */
    /* Each block: 1 type + 4 size bytes + data (max 65535 per block) */
    block_size = 65535;
    blocks = (raw_size + block_size - 1) / block_size;

    /* Total IDAT data size */
    /* 2 (zlib header) + blocks * 5 (block headers) + raw_size + 4 (adler32) */
    put32be(2 + blocks * 5 + raw_size + 4);

    chunk_start("IDAT");

    /* Zlib header: CMF=0x78 (deflate, 32K window), FLG=0x01 (no dict, check bits) */
    chunk_byte(0x78);
    chunk_byte(0x01);

    /* Initialize Adler32 */
    adler_init();

    /* Write uncompressed DEFLATE blocks */
    remaining = raw_size;
    y = 0;
    x = -1;  /* -1 means we need to write filter byte */

    block = 0;
    while (block < blocks) {
        is_last = (block == blocks - 1) ? 1 : 0;
        this_size = remaining;
        if (this_size > block_size) {
            this_size = block_size;
        }

        /* Block header: BFINAL (1 bit) + BTYPE=00 (2 bits) = 0x00 or 0x01 */
        chunk_byte(is_last);

        /* LEN and NLEN (little-endian) */
        chunk_byte(this_size & 255);
        chunk_byte((this_size >> 8) & 255);
        chunk_byte((0 - this_size - 1) & 255);
        chunk_byte(((0 - this_size - 1) >> 8) & 255);

        /* Block data */
        i = 0;
        while (i < this_size) {
            if (x == -1) {
                /* Filter byte: 0 = None */
                chunk_byte(0);
                adler_update(0);
                x = 0;
            } else {
                /* Pixel data */
                cy = y_max - y * dy;
                cx = x_min + x * dx;
                gray = mandel_color(cx, cy, max_iter);
                chunk_byte(gray);
                adler_update(gray);
                x = x + 1;
                if (x >= width) {
                    x = -1;
                    y = y + 1;
                }
            }
            i = i + 1;
        }

        remaining = remaining - this_size;
        block = block + 1;
    }

    /* Adler32 checksum (big-endian) */
    i = adler_get();
    chunk_byte((i >> 24) & 255);
    chunk_byte((i >> 16) & 255);
    chunk_byte((i >> 8) & 255);
    chunk_byte(i & 255);

    chunk_end();

    /* IEND chunk */
    put32be(0);
    chunk_start("IEND");
    chunk_end();

    free(crc_table);
    return 0;
}
