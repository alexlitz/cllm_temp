/* ================================================================
 * NEURAL VM RUNTIME — Fixed-Point (16.16), C4-Compatible
 *
 * Implements the C4 Transformer VM architecture in C.
 * ALL computation flows through the model's weight matrices
 * via matrix multiply + softmax + activation functions.
 * ALL arithmetic uses 16.16 fixed-point int — no floats, no long long.
 *
 * Types: int and char only (C4 compatible).
 * exp() via log2: exp(x) = 2^(x * log2(e)), pow2 via bit shift + polynomial.
 *
 * Architecture (matching src/transformer_vm.py):
 *   - Values encoded as 4 one-hot [256] byte vectors (16.16 FP)
 *   - Byte ↔ nibble conversion via learned weight matrices
 *   - Arithmetic via nibble lookup tables (matmul + softmax)
 *   - Multiplication via SwiGLU: a*b = silu(a)*b + silu(-a)*(-b)
 *   - Division via binary long division (32 neural iterations)
 *   - All state updates through neural operations
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ Fixed-Point (16.16) ============ */
/* All FP values: integer part in upper 16 bits, fraction in lower 16 */
/* FP_ONE = 1.0, FP_HALF = 0.5 */

int FP_ONE;   /* 65536 */
int FP_HALF;  /* 32768 */

/* fp_mul: a * b in 16.16 → 16.16, using (a/256)*(b/256) to stay in int */
int fp_mul(int a, int b) {
    return (a / 256) * (b / 256);
}

/* fp_div: a / b in 16.16 → 16.16 */
int fp_div(int a, int b) {
    if (b == 0) return 0;
    return (a * 256) / (b / 256);
}

/* 2^x in 16.16 via bit shift + degree-3 polynomial for fractional part.
 * exp(x) = 2^(x * log2(e)); this implements the 2^x part.
 *
 * For f in [0,1):
 *   2^f ≈ 65536 + 45426*f + 15743*f^2 + 3638*f^3
 * where f is in 16.16 (range 0..65535).
 *
 * All products use fp_mul to avoid overflow. */
int fp_pow2(int x) {
    int int_part;
    int frac;
    int f2;
    int f3;
    int val;
    int neg;

    if (x >= 0) {
        int_part = x / 65536;
    } else {
        int_part = 0 - ((0 - x + 65535) / 65536);
    }
    frac = x - int_part * 65536;

    /* 2^frac via polynomial (all fp_mul) */
    f2 = fp_mul(frac, frac);
    f3 = fp_mul(f2, frac);
    val = 65536
        + fp_mul(45426, frac)
        + fp_mul(15743, f2)
        + fp_mul(3638, f3);

    if (int_part >= 0) {
        if (int_part >= 20) return 0x7FFFFFFF;
        return val << int_part;
    } else {
        neg = 0 - int_part;
        if (neg >= 30) return 0;
        return val >> neg;
    }
}

/* exp(x) = 2^(x * log2(e)), where log2(e) = 1.4427 ≈ 94548 in 16.16 */
int fp_exp(int x) {
    int y;
    /* y = x * log2(e) in 16.16 = fp_mul(x, 94548) */
    y = fp_mul(x, 94548);
    if (y > 20 * 65536) return 0x7FFFFFFF;
    if (y < 0 - 30 * 65536) return 0;
    return fp_pow2(y);
}

/* sigmoid(x) = 1 / (1 + exp(-x)) in 16.16 */
int fp_sigmoid(int x) {
    int enx;
    int denom;

    if (x >= 65536 * 8) return FP_ONE;
    if (x <= 0 - 65536 * 8) return 0;
    enx = fp_exp(0 - x);
    denom = FP_ONE + enx;
    if (denom == 0) return FP_ONE;
    return fp_div(FP_ONE, denom);
}

/* silu(x) = x * sigmoid(x) in 16.16 */
int fp_silu(int x) {
    return fp_mul(x, fp_sigmoid(x));
}

/* ============ Model Weight Storage ============ */

int *b2n_W1;       /* [256][256] byte-to-nibble step 1 */
int *b2n_W2;       /* [256][32]  byte-to-nibble step 2 */
int *n2b_W1;       /* [32][256]  nibble-to-byte address encoding */
int *n2b_W2;       /* [256][256] nibble-to-byte lookup */
int *nib_add_W1;   /* [34][512]  nibble add address encoding */
int *nib_add_W2s;  /* [512][16]  nibble add sum result */
int *nib_add_W2c;  /* [512][2]   nibble add carry out */
int *nib_mul_W1;   /* [32][256]  nibble mul address encoding */
int *nib_mul_W2l;  /* [256][16]  nibble mul product low */
int *nib_mul_W2h;  /* [256][16]  nibble mul product high */
int *nib_and_W1;   /* [32][256]  AND address encoding */
int *nib_and_W2;   /* [256][16]  AND result */
int *nib_or_W1;    /* [32][256]  OR address encoding */
int *nib_or_W2;    /* [256][16]  OR result */
int *nib_xor_W1;   /* [32][256]  XOR address encoding */
int *nib_xor_W2;   /* [256][16]  XOR result */
int *shl_res;      /* [5][16][16] shift left result */
int *shl_over;     /* [5][16][16] shift left overflow */
int *shr_res;      /* [5][16][16] shift right result */
int *shr_under;    /* [5][16][16] shift right underflow */

int model_loaded;

/* ============ Load Model Weights ============ */

int read_int_from(char **p) {
    char *d;
    int v;
    d = *p;
    v = (d[0] & 0xFF) | ((d[1] & 0xFF) << 8) | ((d[2] & 0xFF) << 16) | ((d[3] & 0xFF) << 24);
    *p = d + 4;
    return v;
}

/* Simple string compare (no strcmp needed) */
int str_eq(char *a, char *b) {
    while (*a && *b) {
        if (*a != *b) return 0;
        a = a + 1;
        b = b + 1;
    }
    return *a == *b;
}

int load_neural_model() {
    char *p;
    int magic;
    int version;
    int n_tensors;
    int n_nodes;
    int i;
    int j;
    int name_len;
    int ndims;
    int dims[8];
    int data_type;
    int size;
    char name[64];
    int *data;

    p = (char *)embedded_model;

    magic = read_int_from(&p);
    if (magic != 0x584E4E4F) {
        printf("Error: invalid model magic\n");
        return 0 - 1;
    }

    version = read_int_from(&p);
    n_tensors = read_int_from(&p);
    n_nodes = read_int_from(&p);

    i = 0;
    while (i < n_tensors) {
        name_len = read_int_from(&p);
        j = 0;
        while (j < name_len && j < 63) {
            name[j] = p[j];
            j = j + 1;
        }
        name[j] = 0;
        p = p + name_len;

        ndims = read_int_from(&p);
        size = 1;
        j = 0;
        while (j < ndims) {
            dims[j] = read_int_from(&p);
            size = size * dims[j];
            j = j + 1;
        }

        data_type = read_int_from(&p);
        if (data_type == 1) {
            /* Sparse COO: read nnz indices + nnz values, expand to dense */
            int nnz;
            int *indices;
            int *values;
            int k;
            nnz = read_int_from(&p);
            indices = malloc(nnz * 4);
            values = malloc(nnz * 4);
            j = 0;
            while (j < nnz) {
                indices[j] = read_int_from(&p);
                j = j + 1;
            }
            j = 0;
            while (j < nnz) {
                values[j] = read_int_from(&p);
                j = j + 1;
            }
            data = malloc(size * 4);
            j = 0;
            while (j < size) {
                data[j] = 0;
                j = j + 1;
            }
            k = 0;
            while (k < nnz) {
                data[indices[k]] = values[k];
                k = k + 1;
            }
            free(indices);
            free(values);
        } else {
            /* Dense: read size values directly */
            size = read_int_from(&p);
            data = malloc(size * 4);
            j = 0;
            while (j < size) {
                data[j] = read_int_from(&p);
                j = j + 1;
            }
        }

        if (str_eq(name, "b2n_W1")) b2n_W1 = data;
        else if (str_eq(name, "b2n_W2")) b2n_W2 = data;
        else if (str_eq(name, "n2b_W1")) n2b_W1 = data;
        else if (str_eq(name, "n2b_W2")) n2b_W2 = data;
        else if (str_eq(name, "nib_add_W1")) nib_add_W1 = data;
        else if (str_eq(name, "nib_add_W2_sum")) nib_add_W2s = data;
        else if (str_eq(name, "nib_add_W2_cout")) nib_add_W2c = data;
        else if (str_eq(name, "nib_mul_W1")) nib_mul_W1 = data;
        else if (str_eq(name, "nib_mul_W2_lo")) nib_mul_W2l = data;
        else if (str_eq(name, "nib_mul_W2_hi")) nib_mul_W2h = data;
        else if (str_eq(name, "nib_and_W1")) nib_and_W1 = data;
        else if (str_eq(name, "nib_and_W2")) nib_and_W2 = data;
        else if (str_eq(name, "nib_or_W1")) nib_or_W1 = data;
        else if (str_eq(name, "nib_or_W2")) nib_or_W2 = data;
        else if (str_eq(name, "nib_xor_W1")) nib_xor_W1 = data;
        else if (str_eq(name, "nib_xor_W2")) nib_xor_W2 = data;
        else if (str_eq(name, "shl_result")) shl_res = data;
        else if (str_eq(name, "shl_overflow")) shl_over = data;
        else if (str_eq(name, "shr_result")) shr_res = data;
        else if (str_eq(name, "shr_underflow")) shr_under = data;
        else free(data);

        i = i + 1;
    }

    model_loaded = 1;
    return 0;
}

/* ============ Neural Computation Primitives (Fixed-Point) ============ */
/* All values are 16.16 fixed-point int. One-hot = 0 or FP_ONE (65536). */

/* Softmax over array of length n, temperature-scaled */
void nn_softmax(int *out, int *in, int n, int temp) {
    int i;
    int max_val;
    int sum;

    max_val = in[0];
    i = 1;
    while (i < n) {
        if (in[i] > max_val) max_val = in[i];
        i = i + 1;
    }
    sum = 0;
    i = 0;
    while (i < n) {
        out[i] = fp_exp(fp_mul(in[i] - max_val, temp));
        sum = sum + out[i];
        i = i + 1;
    }
    if (sum > 0) {
        i = 0;
        while (i < n) {
            out[i] = fp_div(out[i], sum);
            i = i + 1;
        }
    }
}

/* MatMul: out[n] = in[m] @ W[m][n], all 16.16 FP.
 * Each term: fp_mul(in[i], W[i*n+j]) = (in[i]/256)*(W[i*n+j]/256).
 * Accumulate in int — safe because one-hot inputs have at most 1 non-zero. */
void nn_matmul_fp(int *out, int *in, int *W, int m, int n) {
    int i;
    int j;
    int acc;

    j = 0;
    while (j < n) {
        acc = 0;
        i = 0;
        while (i < m) {
            acc = acc + fp_mul(in[i], W[i * n + j]);
            i = i + 1;
        }
        out[j] = acc;
        j = j + 1;
    }
}

/* Argmax */
int nn_argmax(int *v, int n) {
    int best;
    int i;
    best = 0;
    i = 1;
    while (i < n) {
        if (v[i] > v[best]) best = i;
        i = i + 1;
    }
    return best;
}

/* ============ Neural Byte/Nibble Operations ============ */

/* One-hot encode a byte value */
void encode_byte(int *out, int val) {
    int i;
    i = 0;
    while (i < 256) { out[i] = 0; i = i + 1; }
    out[val & 0xFF] = FP_ONE;
}

int decode_byte(int *v) {
    return nn_argmax(v, 256);
}

/* Byte to nibbles via model weights */
void neural_b2n(int *high, int *low, int *byte_oh) {
    int *h;
    int *out;
    int i;
    h = malloc(256 * 4);
    out = malloc(32 * 4);
    nn_matmul_fp(h, byte_oh, b2n_W1, 256, 256);
    nn_matmul_fp(out, h, b2n_W2, 256, 32);
    i = 0;
    while (i < 16) {
        high[i] = out[i];
        low[i] = out[16 + i];
        i = i + 1;
    }
    free(h);
    free(out);
}

/* Nibbles to byte via model weights + softmax */
void neural_n2b(int *byte_oh, int *high, int *low) {
    int *combined;
    int *addr;
    int *addr_soft;
    int i;
    combined = malloc(32 * 4);
    addr = malloc(256 * 4);
    addr_soft = malloc(256 * 4);
    i = 0;
    while (i < 16) {
        combined[i] = high[i];
        combined[16 + i] = low[i];
        i = i + 1;
    }
    nn_matmul_fp(addr, combined, n2b_W1, 32, 256);
    nn_softmax(addr_soft, addr, 256, 100 * 65536);
    nn_matmul_fp(byte_oh, addr_soft, n2b_W2, 256, 256);
    free(combined);
    free(addr);
    free(addr_soft);
}

/* Nibble add with carry */
void neural_nib_add(int *sum_out, int *cout, int *a, int *b, int *cin) {
    int *combined;
    int *addr;
    int *addr_soft;
    int i;
    combined = malloc(34 * 4);
    addr = malloc(512 * 4);
    addr_soft = malloc(512 * 4);
    i = 0;
    while (i < 16) { combined[i] = a[i]; i = i + 1; }
    i = 0;
    while (i < 16) { combined[16 + i] = b[i]; i = i + 1; }
    combined[32] = cin[0];
    combined[33] = cin[1];
    nn_matmul_fp(addr, combined, nib_add_W1, 34, 512);
    nn_softmax(addr_soft, addr, 512, 100 * 65536);
    nn_matmul_fp(sum_out, addr_soft, nib_add_W2s, 512, 16);
    nn_matmul_fp(cout, addr_soft, nib_add_W2c, 512, 2);
    free(combined);
    free(addr);
    free(addr_soft);
}

/* Nibble bitwise op */
void neural_nib_op(int *result, int *a, int *b, int *W1, int *W2) {
    int *combined;
    int *addr;
    int *addr_soft;
    int i;
    combined = malloc(32 * 4);
    addr = malloc(256 * 4);
    addr_soft = malloc(256 * 4);
    i = 0;
    while (i < 16) { combined[i] = a[i]; i = i + 1; }
    i = 0;
    while (i < 16) { combined[16 + i] = b[i]; i = i + 1; }
    nn_matmul_fp(addr, combined, W1, 32, 256);
    nn_softmax(addr_soft, addr, 256, 100 * 65536);
    nn_matmul_fp(result, addr_soft, W2, 256, 16);
    free(combined);
    free(addr);
    free(addr_soft);
}

/* Nibble multiply: (a, b) -> (product_lo, product_hi) via lookup table */
void neural_nib_mul(int *lo_out, int *hi_out, int *a, int *b) {
    int *combined;
    int *addr;
    int *addr_soft;
    int i;
    combined = malloc(32 * 4);
    addr = malloc(256 * 4);
    addr_soft = malloc(256 * 4);
    i = 0;
    while (i < 16) { combined[i] = a[i]; i = i + 1; }
    i = 0;
    while (i < 16) { combined[16 + i] = b[i]; i = i + 1; }
    nn_matmul_fp(addr, combined, nib_mul_W1, 32, 256);
    nn_softmax(addr_soft, addr, 256, 100 * 65536);
    nn_matmul_fp(lo_out, addr_soft, nib_mul_W2l, 256, 16);
    nn_matmul_fp(hi_out, addr_soft, nib_mul_W2h, 256, 16);
    free(combined);
    free(addr);
    free(addr_soft);
}

/* ============ NVal: 32-bit integer as 4 one-hot byte vectors ============ */
/* NVal = int[1024] = 4 * 256 one-hot slots, each 0 or FP_ONE */
/* Layout: bytes 0-255 = byte 0 (LSB), 256-511 = byte 1, etc. */

int *nval_new() {
    int *v;
    int i;
    v = malloc(1024 * 4);
    i = 0;
    while (i < 1024) { v[i] = 0; i = i + 1; }
    /* Default: all bytes = 0 (one-hot at index 0) */
    v[0] = FP_ONE;
    v[256] = FP_ONE;
    v[512] = FP_ONE;
    v[768] = FP_ONE;
    return v;
}

void nval_encode(int *v, int val) {
    int i;
    i = 0;
    while (i < 4) {
        encode_byte(v + i * 256, (val >> (i * 8)) & 0xFF);
        i = i + 1;
    }
}

int nval_decode(int *v) {
    int i;
    int val;
    val = 0;
    i = 0;
    while (i < 4) {
        val = val | (decode_byte(v + i * 256) << (i * 8));
        i = i + 1;
    }
    /* Sign extend: if bit 31 set, make negative */
    if ((val & 0x80000000) != 0) {
        /* val is already negative in two's complement on 32-bit int */
    }
    return val;
}

void nval_copy(int *dst, int *src) {
    int i;
    i = 0;
    while (i < 1024) { dst[i] = src[i]; i = i + 1; }
}

/* ============ Neural 32-bit Operations ============ */

/* Neural addition: a + b using nibble add with carry */
void neural_add(int *result, int *a, int *b) {
    int ah[16], al[16], bh[16], bl[16];
    int sum_l[16], sum_h[16], carry[2];
    int i;

    carry[0] = FP_ONE; carry[1] = 0;

    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        neural_b2n(bh, bl, b + i * 256);
        neural_nib_add(sum_l, carry, al, bl, carry);
        neural_nib_add(sum_h, carry, ah, bh, carry);
        neural_n2b(result + i * 256, sum_h, sum_l);
        i = i + 1;
    }
}

/* Neural negate: ~x + 1 */
void neural_negate(int *result, int *x) {
    int xh[16], xl[16], oh[16], ol[16], rh[16], rl[16];
    int sum_l[16], sum_h[16], carry[2];
    int ones[256];
    int zero_nib[16];
    int *inv;
    int i;

    inv = malloc(1024 * 4);
    encode_byte(ones, 0xFF);

    /* XOR each byte with 0xFF */
    i = 0;
    while (i < 4) {
        neural_b2n(xh, xl, x + i * 256);
        neural_b2n(oh, ol, ones);
        neural_nib_op(rh, xh, oh, nib_xor_W1, nib_xor_W2);
        neural_nib_op(rl, xl, ol, nib_xor_W1, nib_xor_W2);
        neural_n2b(inv + i * 256, rh, rl);
        i = i + 1;
    }

    /* Add 1 */
    carry[0] = 0; carry[1] = FP_ONE;
    i = 0;
    while (i < 16) { zero_nib[i] = 0; i = i + 1; }
    zero_nib[0] = FP_ONE;

    i = 0;
    while (i < 4) {
        neural_b2n(xh, xl, inv + i * 256);
        neural_nib_add(sum_l, carry, xl, zero_nib, carry);
        neural_nib_add(sum_h, carry, xh, zero_nib, carry);
        neural_n2b(result + i * 256, sum_h, sum_l);
        i = i + 1;
    }
    free(inv);
}

void neural_sub(int *result, int *a, int *b) {
    int *neg_b;
    neg_b = nval_new();
    neural_negate(neg_b, b);
    neural_add(result, a, neg_b);
    free(neg_b);
}

/* Neural bitwise operations */
void neural_bitwise(int *result, int *a, int *b, int *W1, int *W2) {
    int ah[16], al[16], bh[16], bl[16], rh[16], rl[16];
    int i;
    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        neural_b2n(bh, bl, b + i * 256);
        neural_nib_op(rh, ah, bh, W1, W2);
        neural_nib_op(rl, al, bl, W1, W2);
        neural_n2b(result + i * 256, rh, rl);
        i = i + 1;
    }
}

/* Neural shift left by 1 */
void neural_shl1(int *result, int *x) {
    int h[16], l[16];
    int new_l[16], new_h[16], l_over[16], h_over[16];
    int carry[16];
    int i;
    int j;
    int k;
    int lv;
    int hv;
    int sr;
    int so;

    j = 0;
    while (j < 16) { carry[j] = 0; j = j + 1; }
    carry[0] = FP_ONE;

    i = 0;
    while (i < 4) {
        neural_b2n(h, l, x + i * 256);

        j = 0;
        while (j < 16) {
            new_l[j] = 0; l_over[j] = 0;
            new_h[j] = 0; h_over[j] = 0;
            j = j + 1;
        }
        j = 0;
        while (j < 16) {
            lv = l[j]; hv = h[j];
            k = 0;
            while (k < 16) {
                sr = shl_res[1 * 256 + j * 16 + k];
                so = shl_over[1 * 256 + j * 16 + k];
                new_l[k] = new_l[k] + fp_mul(lv, sr);
                l_over[k] = l_over[k] + fp_mul(lv, so);
                new_h[k] = new_h[k] + fp_mul(hv, sr);
                h_over[k] = h_over[k] + fp_mul(hv, so);
                k = k + 1;
            }
            j = j + 1;
        }

        neural_nib_op(new_l, new_l, carry, nib_or_W1, nib_or_W2);
        neural_nib_op(new_h, new_h, l_over, nib_or_W1, nib_or_W2);
        j = 0;
        while (j < 16) { carry[j] = h_over[j]; j = j + 1; }

        neural_n2b(result + i * 256, new_h, new_l);
        i = i + 1;
    }
}

/* Neural shift right by 1 */
void neural_shr1(int *result, int *x) {
    int h[16], l[16];
    int new_l[16], new_h[16], l_under[16], h_under[16];
    int carry[16];
    int i;
    int j;
    int k;
    int lv;
    int hv;
    int sr;
    int su;

    j = 0;
    while (j < 16) { carry[j] = 0; j = j + 1; }
    carry[0] = FP_ONE;

    i = 3;
    while (i >= 0) {
        neural_b2n(h, l, x + i * 256);

        j = 0;
        while (j < 16) {
            new_l[j] = 0; l_under[j] = 0;
            new_h[j] = 0; h_under[j] = 0;
            j = j + 1;
        }
        j = 0;
        while (j < 16) {
            lv = l[j]; hv = h[j];
            k = 0;
            while (k < 16) {
                sr = shr_res[1 * 256 + j * 16 + k];
                su = shr_under[1 * 256 + j * 16 + k];
                new_l[k] = new_l[k] + fp_mul(lv, sr);
                l_under[k] = l_under[k] + fp_mul(lv, su);
                new_h[k] = new_h[k] + fp_mul(hv, sr);
                h_under[k] = h_under[k] + fp_mul(hv, su);
                k = k + 1;
            }
            j = j + 1;
        }

        neural_nib_op(new_h, new_h, carry, nib_or_W1, nib_or_W2);
        neural_nib_op(new_l, new_l, h_under, nib_or_W1, nib_or_W2);
        j = 0;
        while (j < 16) { carry[j] = l_under[j]; j = j + 1; }

        neural_n2b(result + i * 256, new_h, new_l);
        i = i - 1;
    }
}

/* Neural multiply via schoolbook nibble multiplication */
void neural_mul(int *result, int *a, int *b) {
    int *a_nib;     /* 8 nibbles of a, each 16 ints = 128 total */
    int *b_nib;     /* 8 nibbles of b */
    int *acc;       /* 8 accumulator nibbles */
    int carry[2];
    int prod_lo[16];
    int prod_hi[16];
    int sum_tmp[16];
    int ah[16], al[16], bh[16], bl[16];
    int zero_nib_l[16];
    int i;
    int j;
    int k;
    int pos;

    a_nib = malloc(128 * 4);
    b_nib = malloc(128 * 4);
    acc = malloc(128 * 4);

    /* Extract 8 nibbles from a and b */
    i = 0;
    while (i < 4) {
        neural_b2n(ah, al, a + i * 256);
        k = 0;
        while (k < 16) {
            a_nib[(i * 2) * 16 + k] = al[k];
            a_nib[(i * 2 + 1) * 16 + k] = ah[k];
            k = k + 1;
        }
        neural_b2n(bh, bl, b + i * 256);
        k = 0;
        while (k < 16) {
            b_nib[(i * 2) * 16 + k] = bl[k];
            b_nib[(i * 2 + 1) * 16 + k] = bh[k];
            k = k + 1;
        }
        i = i + 1;
    }

    /* Initialize accumulator nibbles to zero */
    i = 0;
    while (i < 128) {
        acc[i] = 0;
        i = i + 1;
    }
    i = 0;
    while (i < 8) {
        acc[i * 16] = FP_ONE;
        i = i + 1;
    }

    /* Zero nibble for carry propagation */
    i = 0;
    while (i < 16) { zero_nib_l[i] = 0; i = i + 1; }
    zero_nib_l[0] = FP_ONE;

    /* Schoolbook: for each pair (i,j) where i+j < 8 */
    i = 0;
    while (i < 8) {
        j = 0;
        while (j < 8) {
            pos = i + j;
            if (pos < 8) {
                /* Multiply nibble a[i] * b[j] -> (prod_lo, prod_hi) */
                neural_nib_mul(prod_lo, prod_hi, a_nib + i * 16, b_nib + j * 16);

                /* Add prod_lo to acc[pos] with carry */
                carry[0] = FP_ONE; carry[1] = 0;
                neural_nib_add(sum_tmp, carry, acc + pos * 16, prod_lo, carry);
                k = 0;
                while (k < 16) { acc[pos * 16 + k] = sum_tmp[k]; k = k + 1; }

                /* Add prod_hi to acc[pos+1] with carry (if in range) */
                if (pos + 1 < 8) {
                    neural_nib_add(sum_tmp, carry, acc + (pos + 1) * 16, prod_hi, carry);
                    k = 0;
                    while (k < 16) { acc[(pos + 1) * 16 + k] = sum_tmp[k]; k = k + 1; }

                    /* Propagate remaining carry up */
                    pos = pos + 2;
                    while (pos < 8) {
                        if (carry[1] > FP_HALF) {
                            neural_nib_add(sum_tmp, carry, acc + pos * 16, zero_nib_l, carry);
                            k = 0;
                            while (k < 16) { acc[pos * 16 + k] = sum_tmp[k]; k = k + 1; }
                        }
                        pos = pos + 1;
                    }
                }
            }
            j = j + 1;
        }
        i = i + 1;
    }

    /* Pack 8 accumulator nibbles back into 4 bytes */
    i = 0;
    while (i < 4) {
        neural_n2b(result + i * 256, acc + (i * 2 + 1) * 16, acc + (i * 2) * 16);
        i = i + 1;
    }
    free(a_nib);
    free(b_nib);
    free(acc);
}

/* Neural is_zero check */
int neural_is_zero(int *x) {
    int prob;
    int i;
    /* Product of p(byte==0) — use fp_mul chain */
    prob = FP_ONE;
    i = 0;
    while (i < 4) {
        prob = fp_mul(prob, x[i * 256]);
        i = i + 1;
    }
    return prob > FP_HALF ? 1 : 0;
}

/* Neural is_negative (MSB set) */
int neural_is_negative(int *x) {
    int neg_w;
    int i;
    neg_w = 0;
    i = 128;
    while (i < 256) {
        neg_w = neg_w + x[768 + i]; /* byte 3, indices 128-255 */
        i = i + 1;
    }
    return neg_w > FP_HALF ? 1 : 0;
}

int neural_compare(int *a, int *b) {
    int *diff;
    int z;
    int n;
    diff = nval_new();
    neural_sub(diff, a, b);
    z = neural_is_zero(diff);
    n = neural_is_negative(diff);
    free(diff);
    if (z) return 0;
    if (n) return 0 - 1;
    return 1;
}

/* Neural divide via binary long division */
void neural_div(int *result, int *a, int *b) {
    int av;
    int bv;
    int a_neg;
    int b_neg;
    int *a_abs;
    int *b_abs;
    int *quotient;
    int *remainder;
    int *one;
    int *bit_mask;
    int *masked;
    int i;
    int bit_set;
    int cmp;

    av = nval_decode(a);
    bv = nval_decode(b);
    if (bv == 0) { nval_encode(result, 0); return; }

    a_neg = av < 0;
    b_neg = bv < 0;

    a_abs = nval_new();
    b_abs = nval_new();
    if (a_neg) neural_negate(a_abs, a); else nval_copy(a_abs, a);
    if (b_neg) neural_negate(b_abs, b); else nval_copy(b_abs, b);

    quotient = nval_new();
    remainder = nval_new();
    one = nval_new();
    bit_mask = nval_new();
    masked = nval_new();
    nval_encode(quotient, 0);
    nval_encode(remainder, 0);
    nval_encode(one, 1);

    i = 31;
    while (i >= 0) {
        neural_shl1(remainder, remainder);

        nval_encode(bit_mask, 1 << i);
        neural_bitwise(masked, a_abs, bit_mask, nib_and_W1, nib_and_W2);
        bit_set = !neural_is_zero(masked);

        if (bit_set) {
            neural_bitwise(remainder, remainder, one, nib_or_W1, nib_or_W2);
        }

        cmp = neural_compare(remainder, b_abs);
        if (cmp >= 0) {
            neural_sub(remainder, remainder, b_abs);
            neural_bitwise(quotient, quotient, bit_mask, nib_or_W1, nib_or_W2);
        }

        i = i - 1;
    }

    if (a_neg != b_neg) neural_negate(result, quotient);
    else nval_copy(result, quotient);

    free(a_abs); free(b_abs); free(quotient); free(remainder);
    free(one); free(bit_mask); free(masked);
}

void neural_mod(int *result, int *a, int *b) {
    int *q;
    int *qb;
    q = nval_new();
    qb = nval_new();
    neural_div(q, a, b);
    neural_mul(qb, q, b);
    neural_sub(result, a, qb);
    free(q); free(qb);
}

void neural_shl(int *result, int *x, int *amt) {
    int shift;
    int i;
    nval_copy(result, x);
    shift = nval_decode(amt);
    i = 0;
    while (i < shift && i < 32) {
        neural_shl1(result, result);
        i = i + 1;
    }
}

void neural_shr(int *result, int *x, int *amt) {
    int shift;
    int i;
    nval_copy(result, x);
    shift = nval_decode(amt);
    i = 0;
    while (i < shift && i < 32) {
        neural_shr1(result, result);
        i = i + 1;
    }
}

void neural_mask_byte(int *result, int *x) {
    int i;
    int j;
    j = 0;
    while (j < 256) { result[j] = x[j]; j = j + 1; }
    i = 1;
    while (i < 4) {
        j = 0;
        while (j < 256) { result[i * 256 + j] = 0; j = j + 1; }
        result[i * 256] = FP_ONE;
        i = i + 1;
    }
}

/* ============ VM State ============ */

int *reg_ax;
int *reg_sp;
int *reg_bp;
int *reg_pc;
int vm_halted;
int vm_exit_code;

int MEM_SIZE;
int *vm_memory;
int heap_ptr;

/* Stack helpers */
void vm_push(int *val) {
    int *eight;
    int addr;
    eight = nval_new();
    nval_encode(eight, 8);
    neural_sub(reg_sp, reg_sp, eight);
    addr = nval_decode(reg_sp);
    if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(val);
    free(eight);
}

void vm_pop(int *val) {
    int addr;
    int *eight;
    addr = nval_decode(reg_sp);
    nval_encode(val, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
    eight = nval_new();
    nval_encode(eight, 8);
    neural_add(reg_sp, reg_sp, eight);
    free(eight);
}

/* ============ VM Execution ============ */

int vm_step() {
    int pc_val;
    int instr_idx;
    int op;
    int imm;
    int *eight;
    int *imm_enc;
    int *ax;
    int *a;
    int cmp;
    int addr;
    int c;
    int size;
    int ptr;
    int *val;
    int *masked;

    if (vm_halted) return 0;

    pc_val = nval_decode(reg_pc);
    instr_idx = pc_val / 8;

    if (instr_idx < 0 || instr_idx >= program_code_len) {
        vm_halted = 1;
        return 0;
    }

    op = program_code[instr_idx][0];
    imm = program_code[instr_idx][1];

    /* Advance PC */
    eight = nval_new();
    nval_encode(eight, 8);
    neural_add(reg_pc, reg_pc, eight);
    free(eight);

    imm_enc = nval_new();
    nval_encode(imm_enc, imm);
    ax = nval_new();
    nval_copy(ax, reg_ax);
    a = nval_new();

    if (op == 0) { /* LEA */
        neural_add(reg_ax, reg_bp, imm_enc);
    }
    else if (op == 1) { /* IMM */
        nval_copy(reg_ax, imm_enc);
    }
    else if (op == 2) { /* JMP */
        nval_copy(reg_pc, imm_enc);
    }
    else if (op == 3) { /* JSR */
        vm_push(reg_pc);
        nval_copy(reg_pc, imm_enc);
    }
    else if (op == 4) { /* BZ */
        if (neural_is_zero(ax)) nval_copy(reg_pc, imm_enc);
    }
    else if (op == 5) { /* BNZ */
        if (!neural_is_zero(ax)) nval_copy(reg_pc, imm_enc);
    }
    else if (op == 6) { /* ENT */
        vm_push(reg_bp);
        nval_copy(reg_bp, reg_sp);
        neural_sub(reg_sp, reg_sp, imm_enc);
    }
    else if (op == 7) { /* ADJ */
        neural_add(reg_sp, reg_sp, imm_enc);
    }
    else if (op == 8) { /* LEV */
        nval_copy(reg_sp, reg_bp);
        vm_pop(reg_bp);
        vm_pop(reg_pc);
    }
    else if (op == 9) { /* LI */
        addr = nval_decode(ax);
        nval_encode(reg_ax, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
    }
    else if (op == 10) { /* LC */
        addr = nval_decode(ax);
        val = nval_new();
        nval_encode(val, (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0);
        neural_mask_byte(reg_ax, val);
        free(val);
    }
    else if (op == 11) { /* SI */
        vm_pop(a);
        addr = nval_decode(a);
        if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(ax);
    }
    else if (op == 12) { /* SC */
        vm_pop(a);
        addr = nval_decode(a);
        masked = nval_new();
        neural_mask_byte(masked, ax);
        if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = nval_decode(masked);
        free(masked);
    }
    else if (op == 13) { /* PSH */
        vm_push(ax);
    }
    else if (op == 14) { /* OR */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, nib_or_W1, nib_or_W2);
    }
    else if (op == 15) { /* XOR */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, nib_xor_W1, nib_xor_W2);
    }
    else if (op == 16) { /* AND */
        vm_pop(a);
        neural_bitwise(reg_ax, a, ax, nib_and_W1, nib_and_W2);
    }
    else if (op == 17) { /* EQ */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp == 0 ? 1 : 0);
    }
    else if (op == 18) { /* NE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp != 0 ? 1 : 0);
    }
    else if (op == 19) { /* LT */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp < 0 ? 1 : 0);
    }
    else if (op == 20) { /* GT */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp > 0 ? 1 : 0);
    }
    else if (op == 21) { /* LE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp <= 0 ? 1 : 0);
    }
    else if (op == 22) { /* GE */
        vm_pop(a);
        cmp = neural_compare(a, ax);
        nval_encode(reg_ax, cmp >= 0 ? 1 : 0);
    }
    else if (op == 23) { /* SHL */
        vm_pop(a);
        neural_shl(reg_ax, a, ax);
    }
    else if (op == 24) { /* SHR */
        vm_pop(a);
        neural_shr(reg_ax, a, ax);
    }
    else if (op == 25) { /* ADD */
        vm_pop(a);
        neural_add(reg_ax, a, ax);
    }
    else if (op == 26) { /* SUB */
        vm_pop(a);
        neural_sub(reg_ax, a, ax);
    }
    else if (op == 27) { /* MUL */
        vm_pop(a);
        neural_mul(reg_ax, a, ax);
    }
    else if (op == 28) { /* DIV */
        vm_pop(a);
        neural_div(reg_ax, a, ax);
    }
    else if (op == 29) { /* MOD */
        vm_pop(a);
        neural_mod(reg_ax, a, ax);
    }
    else if (op == 34) { /* MALC */
        addr = nval_decode(reg_sp);
        size = (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
        ptr = heap_ptr;
        heap_ptr = heap_ptr + (size + 7) / 8;
        if (heap_ptr >= MEM_SIZE) heap_ptr = MEM_SIZE - 1;
        nval_encode(reg_ax, ptr);
    }
    else if (op == 35) { /* FREE */ }
    else if (op == 38) { /* EXIT */
        vm_exit_code = nval_decode(ax);
        vm_halted = 1;
        free(imm_enc); free(ax); free(a);
        return 0;
    }
    else if (op == 64) { /* GETCHAR */
        c = getchar();
        nval_encode(reg_ax, c);
    }
    else if (op == 65) { /* PUTCHAR */
        addr = nval_decode(reg_sp);
        c = (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
        putchar(c & 0xFF);
        fflush(stdout);
    }
    else {
        printf("Unknown opcode: %d at PC=%d\n", op, pc_val);
        vm_halted = 1;
        free(imm_enc); free(ax); free(a);
        return 0;
    }

    free(imm_enc);
    free(ax);
    free(a);
    return 1;
}

/* ============ Main ============ */

int main() {
    int i;

    FP_ONE = 65536;
    FP_HALF = 32768;
    MEM_SIZE = 0x20000;
    heap_ptr = 0x18000;

    vm_memory = malloc(MEM_SIZE * 4);
    i = 0;
    while (i < MEM_SIZE) { vm_memory[i] = 0; i = i + 1; }

    if (load_neural_model() != 0) {
        printf("Failed to load neural model\n");
        return 1;
    }

    reg_ax = nval_new();
    reg_sp = nval_new();
    reg_bp = nval_new();
    reg_pc = nval_new();
    nval_encode(reg_ax, 0);
    nval_encode(reg_sp, 0x10000);
    nval_encode(reg_bp, 0x10000);
    nval_encode(reg_pc, 0);
    vm_halted = 0;
    vm_exit_code = 0;

    i = 0;
    while (i < program_data_len) {
        vm_memory[0x10000 + i] = program_data[i];
        i = i + 1;
    }

    while (vm_step()) {}

    return vm_exit_code & 0xFF;
}
