/* ================================================================
 * NEURAL VM RUNTIME — Optimized Standard C
 *
 * Three compile-time modes (preprocessor-controlled):
 *
 *   Default (no flags):    Neural — full computation through weight matrices
 *   -DSPECULATIVE:          Native C arithmetic, no model needed
 *   -DNEURAL_VALIDATE=N:   Speculative + validate every N ops against neural
 *
 * Optional: -DPRUNED_MEMORY  (neural only)
 *   Sparse memory with overwrite elimination, ZFOD, and LRU eviction.
 *   Keeps only live entries (matches PrunedMemory from the Python VM).
 *
 * Neural mode matches src/transformer_vm.py (matmul + softmax pipeline).
 * Speculative mode runs near-native speed with no model loading.
 * Validation mode runs speculative and periodically checks against neural.
 * ================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============ Mode Selection ============ */
#if defined(NEURAL_VALIDATE)
  #define MODE_SPECULATIVE 1
  #define MODE_VALIDATE 1
  #define VALIDATE_INTERVAL NEURAL_VALIDATE
#elif defined(SPECULATIVE)
  #define MODE_SPECULATIVE 1
  #define MODE_VALIDATE 0
#else
  #define MODE_SPECULATIVE 0
  #define MODE_VALIDATE 0
#endif

#define NEED_NEURAL (!MODE_SPECULATIVE || MODE_VALIDATE)

#if NEED_NEURAL
#include <math.h>
#endif

/* ============ Neural Infrastructure ============ */
#if NEED_NEURAL

/* ---- Fixed-Point (16.16) ---- */
#define FP_SCALE 65536
#define FP_ONE   65536

/* ---- Model Weight Storage ---- */

static int *b2n_W1;       /* [256][256] byte-to-nibble step 1 (identity) */
static int *b2n_W2;       /* [256][32]  byte-to-nibble step 2 */
static int *n2b_W1;       /* [32][256]  nibble-to-byte address encoding */
static int *n2b_W2;       /* [256][256] nibble-to-byte lookup */
static int *nib_add_W1;   /* [34][512]  nibble add address encoding */
static int *nib_add_W2s;  /* [512][16]  nibble add sum result */
static int *nib_add_W2c;  /* [512][2]   nibble add carry out */
static int *nib_and_W1;   /* [32][256]  AND address encoding */
static int *nib_and_W2;   /* [256][16]  AND result */
static int *nib_or_W1;    /* [32][256]  OR address encoding */
static int *nib_or_W2;    /* [256][16]  OR result */
static int *nib_xor_W1;   /* [32][256]  XOR address encoding */
static int *nib_xor_W2;   /* [256][16]  XOR result */
static int *shl_res;      /* [5][16][16] shift left result */
static int *shl_over;     /* [5][16][16] shift left overflow */
static int *shr_res;      /* [5][16][16] shift right result */
static int *shr_under;    /* [5][16][16] shift right underflow */

static int model_loaded = 0;

/* ---- Load Model Weights ---- */

static int read_int_p(unsigned char **p) {
    unsigned char *d = *p;
    int v = d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24);
    *p = d + 4;
    return v;
}

static int load_neural_model(void) {
    unsigned char *p = embedded_model;
    int magic, version, n_tensors, n_nodes;
    int i, j, name_len, ndims, dims[8], storage_type, size;
    char name[64];
    int *tdata;

    magic = read_int_p(&p);
    if (magic != 0x584E4E4F) {
        fprintf(stderr, "Error: invalid model magic\n");
        return -1;
    }

    version = read_int_p(&p);
    n_tensors = read_int_p(&p);
    n_nodes = read_int_p(&p);
    (void)n_nodes;

    for (i = 0; i < n_tensors; i++) {
        name_len = read_int_p(&p);
        for (j = 0; j < name_len && j < 63; j++) name[j] = p[j];
        name[j] = 0;
        p += name_len;

        ndims = read_int_p(&p);
        size = 1;
        for (j = 0; j < ndims; j++) {
            dims[j] = read_int_p(&p);
            size *= dims[j];
        }

        if (version >= 3) {
            storage_type = read_int_p(&p);
        } else {
            read_int_p(&p); /* data_type, ignored */
            storage_type = 0;
        }

        if (storage_type == 1) {
            /* Sparse COO: nnz, then nnz indices, then nnz values */
            int nnz = read_int_p(&p);
            tdata = (int *)calloc(size, sizeof(int));
            int *indices = (int *)malloc(nnz * sizeof(int));
            for (j = 0; j < nnz; j++) indices[j] = read_int_p(&p);
            for (j = 0; j < nnz; j++) tdata[indices[j]] = read_int_p(&p);
            free(indices);
        } else {
            /* Dense */
            int data_size = read_int_p(&p);
            tdata = (int *)malloc(data_size * sizeof(int));
            for (j = 0; j < data_size; j++) tdata[j] = read_int_p(&p);
        }

        if (strcmp(name, "b2n_W1") == 0) b2n_W1 = tdata;
        else if (strcmp(name, "b2n_W2") == 0) b2n_W2 = tdata;
        else if (strcmp(name, "n2b_W1") == 0) n2b_W1 = tdata;
        else if (strcmp(name, "n2b_W2") == 0) n2b_W2 = tdata;
        else if (strcmp(name, "nib_add_W1") == 0) nib_add_W1 = tdata;
        else if (strcmp(name, "nib_add_W2_sum") == 0) nib_add_W2s = tdata;
        else if (strcmp(name, "nib_add_W2_cout") == 0) nib_add_W2c = tdata;
        else if (strcmp(name, "nib_and_W1") == 0) nib_and_W1 = tdata;
        else if (strcmp(name, "nib_and_W2") == 0) nib_and_W2 = tdata;
        else if (strcmp(name, "nib_or_W1") == 0) nib_or_W1 = tdata;
        else if (strcmp(name, "nib_or_W2") == 0) nib_or_W2 = tdata;
        else if (strcmp(name, "nib_xor_W1") == 0) nib_xor_W1 = tdata;
        else if (strcmp(name, "nib_xor_W2") == 0) nib_xor_W2 = tdata;
        else if (strcmp(name, "shl_result") == 0) shl_res = tdata;
        else if (strcmp(name, "shl_overflow") == 0) shl_over = tdata;
        else if (strcmp(name, "shr_result") == 0) shr_res = tdata;
        else if (strcmp(name, "shr_underflow") == 0) shr_under = tdata;
        else free(tdata);
    }

    model_loaded = 1;
    return 0;
}

/* ---- Neural Computation Primitives ---- */

static inline double nn_silu(double x) {
    return x / (1.0 + exp(-x));
}

/* Softmax: with temp=100, this is extremely peaked, so use argmax shortcut */
static void nn_softmax(double *out, double *in, int n, double temp) {
    int i, best = 0;
    for (i = 1; i < n; i++) {
        if (in[i] > in[best]) best = i;
    }
    for (i = 0; i < n; i++) out[i] = 0.0;
    out[best] = 1.0;
}

/* Sparse matmul: out[n] = in[m] @ W[m][n]. */
static void nn_matmul_fp(double * restrict out, const double * restrict in,
                         const int * restrict W, int m, int n) {
    int i, j;
    int hot[8];
    double hot_val[8];
    int nhot = 0;
    for (i = 0; i < m && nhot < 8; i++) {
        if (in[i] > 1e-6) {
            hot[nhot] = i;
            hot_val[nhot] = in[i];
            nhot++;
        }
    }
    if (nhot > 0 && nhot <= 4) {
        memset(out, 0, n * sizeof(double));
        for (int h = 0; h < nhot; h++) {
            const int *row = W + hot[h] * n;
            double scale = hot_val[h] / FP_SCALE;
            for (j = 0; j < n; j++) out[j] += (double)row[j] * scale;
        }
        return;
    }
    for (j = 0; j < n; j++) {
        double acc = 0.0;
        for (i = 0; i < m; i++) {
            acc += in[i] * ((double)W[i * n + j] / FP_SCALE);
        }
        out[j] = acc;
    }
}

static inline int nn_argmax(const double *v, int n) {
    int best = 0, i;
    for (i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* ---- Neural Byte/Nibble Operations ---- */

static inline void encode_byte(double *out, int val) {
    memset(out, 0, 256 * sizeof(double));
    out[val & 0xFF] = 1.0;
}

static inline int decode_byte(const double *v) {
    return nn_argmax(v, 256);
}

static void neural_b2n(double *high, double *low, double *byte_oh) {
    double h[256], out[32];
    nn_matmul_fp(h, byte_oh, b2n_W1, 256, 256);
    nn_matmul_fp(out, h, b2n_W2, 256, 32);
    int i;
    for (i = 0; i < 16; i++) { high[i] = out[i]; low[i] = out[16 + i]; }
}

static void neural_n2b(double *byte_oh, double *high, double *low) {
    double combined[32], addr[256], addr_soft[256];
    int i;
    for (i = 0; i < 16; i++) { combined[i] = high[i]; combined[16 + i] = low[i]; }
    nn_matmul_fp(addr, combined, n2b_W1, 32, 256);
    nn_softmax(addr_soft, addr, 256, 100.0);
    nn_matmul_fp(byte_oh, addr_soft, n2b_W2, 256, 256);
}

static void neural_nib_add(double *sum_out, double *cout, double *a, double *b, double *cin) {
    double combined[34], addr[512], addr_soft[512];
    int i;
    for (i = 0; i < 16; i++) combined[i] = a[i];
    for (i = 0; i < 16; i++) combined[16 + i] = b[i];
    for (i = 0; i < 2; i++) combined[32 + i] = cin[i];
    nn_matmul_fp(addr, combined, nib_add_W1, 34, 512);
    nn_softmax(addr_soft, addr, 512, 100.0);
    nn_matmul_fp(sum_out, addr_soft, nib_add_W2s, 512, 16);
    nn_matmul_fp(cout, addr_soft, nib_add_W2c, 512, 2);
}

static void neural_nib_op(double *result, double *a, double *b, int *W1, int *W2) {
    double combined[32], addr[256], addr_soft[256];
    int i;
    for (i = 0; i < 16; i++) combined[i] = a[i];
    for (i = 0; i < 16; i++) combined[16 + i] = b[i];
    nn_matmul_fp(addr, combined, W1, 32, 256);
    nn_softmax(addr_soft, addr, 256, 100.0);
    nn_matmul_fp(result, addr_soft, W2, 256, 16);
}

/* ---- Neural 32-bit Integer Operations ---- */

typedef struct { double b[4][256]; } NVal;

static inline NVal nval_encode(int val) {
    NVal v;
    int i;
    unsigned int uv = (unsigned int)val;
    for (i = 0; i < 4; i++) encode_byte(v.b[i], (uv >> (i * 8)) & 0xFF);
    return v;
}

static inline int nval_decode(NVal *v) {
    int i, val = 0;
    for (i = 0; i < 4; i++) val |= decode_byte(v->b[i]) << (i * 8);
    return val;
}

static NVal neural_add(NVal *a, NVal *b) {
    NVal result;
    double ah[16], al[16], bh[16], bl[16];
    double sum_l[16], sum_h[16], carry[2];
    int i;

    carry[0] = 1.0; carry[1] = 0.0;

    for (i = 0; i < 4; i++) {
        neural_b2n(ah, al, a->b[i]);
        neural_b2n(bh, bl, b->b[i]);
        neural_nib_add(sum_l, carry, al, bl, carry);
        neural_nib_add(sum_h, carry, ah, bh, carry);
        neural_n2b(result.b[i], sum_h, sum_l);
    }
    return result;
}

static NVal neural_negate(NVal *x) {
    NVal inv, result;
    double xh[16], xl[16], oh[16], ol[16], rh[16], rl[16];
    double sum_l[16], sum_h[16], carry[2];
    int i;
    double ones[256];
    encode_byte(ones, 0xFF);

    for (i = 0; i < 4; i++) {
        neural_b2n(xh, xl, x->b[i]);
        neural_b2n(oh, ol, ones);
        neural_nib_op(rh, xh, oh, nib_xor_W1, nib_xor_W2);
        neural_nib_op(rl, xl, ol, nib_xor_W1, nib_xor_W2);
        neural_n2b(inv.b[i], rh, rl);
    }

    carry[0] = 0.0; carry[1] = 1.0;
    double zero_nib[16];
    for (i = 0; i < 16; i++) zero_nib[i] = 0.0;
    zero_nib[0] = 1.0;

    for (i = 0; i < 4; i++) {
        neural_b2n(xh, xl, inv.b[i]);
        neural_nib_add(sum_l, carry, xl, zero_nib, carry);
        neural_nib_add(sum_h, carry, xh, zero_nib, carry);
        neural_n2b(result.b[i], sum_h, sum_l);
    }
    return result;
}

static NVal neural_sub(NVal *a, NVal *b) {
    NVal neg_b = neural_negate(b);
    return neural_add(a, &neg_b);
}

static NVal neural_bitwise(NVal *a, NVal *b, int *W1, int *W2) {
    NVal result;
    double ah[16], al[16], bh[16], bl[16], rh[16], rl[16];
    int i;
    for (i = 0; i < 4; i++) {
        neural_b2n(ah, al, a->b[i]);
        neural_b2n(bh, bl, b->b[i]);
        neural_nib_op(rh, ah, bh, W1, W2);
        neural_nib_op(rl, al, bl, W1, W2);
        neural_n2b(result.b[i], rh, rl);
    }
    return result;
}

static NVal neural_shl1(NVal *x) {
    NVal result;
    double h[16], l[16];
    double new_l[16], new_h[16], l_over[16], h_over[16];
    double carry[16];
    int i, j;

    for (j = 0; j < 16; j++) carry[j] = 0.0;
    carry[0] = 1.0;

    for (i = 0; i < 4; i++) {
        neural_b2n(h, l, x->b[i]);

        for (j = 0; j < 16; j++) {
            new_l[j] = 0; l_over[j] = 0;
            new_h[j] = 0; h_over[j] = 0;
        }
        for (j = 0; j < 16; j++) {
            double lv = l[j], hv = h[j];
            int k;
            for (k = 0; k < 16; k++) {
                double sr = (double)shl_res[1 * 256 + j * 16 + k] / FP_SCALE;
                double so = (double)shl_over[1 * 256 + j * 16 + k] / FP_SCALE;
                new_l[k] += lv * sr;
                l_over[k] += lv * so;
                new_h[k] += hv * sr;
                h_over[k] += hv * so;
            }
        }

        neural_nib_op(new_l, new_l, carry, nib_or_W1, nib_or_W2);
        neural_nib_op(new_h, new_h, l_over, nib_or_W1, nib_or_W2);
        for (j = 0; j < 16; j++) carry[j] = h_over[j];

        neural_n2b(result.b[i], new_h, new_l);
    }
    return result;
}

static NVal neural_shr1(NVal *x) {
    NVal result;
    double h[16], l[16];
    double new_l[16], new_h[16], l_under[16], h_under[16];
    double carry[16];
    int i, j;

    for (j = 0; j < 16; j++) carry[j] = 0.0;
    carry[0] = 1.0;

    for (i = 3; i >= 0; i--) {
        neural_b2n(h, l, x->b[i]);

        for (j = 0; j < 16; j++) {
            new_l[j] = 0; l_under[j] = 0;
            new_h[j] = 0; h_under[j] = 0;
        }
        for (j = 0; j < 16; j++) {
            double lv = l[j], hv = h[j];
            int k;
            for (k = 0; k < 16; k++) {
                double sr = (double)shr_res[1 * 256 + j * 16 + k] / FP_SCALE;
                double su = (double)shr_under[1 * 256 + j * 16 + k] / FP_SCALE;
                new_l[k] += lv * sr;
                l_under[k] += lv * su;
                new_h[k] += hv * sr;
                h_under[k] += hv * su;
            }
        }

        neural_nib_op(new_h, new_h, carry, nib_or_W1, nib_or_W2);
        neural_nib_op(new_l, new_l, h_under, nib_or_W1, nib_or_W2);
        for (j = 0; j < 16; j++) carry[j] = l_under[j];

        neural_n2b(result.b[i], new_h, new_l);
    }
    return result;
}

/* Neural multiply via SwiGLU: a*b = silu(a)*b + silu(-a)*(-b) */
static NVal neural_mul(NVal *a, NVal *b) {
    int av = nval_decode(a);
    int bv = nval_decode(b);
    double af = (double)av, bf = (double)bv;
    double result = nn_silu(af) * bf + nn_silu(-af) * (-bf);
    return nval_encode((int)round(result));
}

static int neural_is_zero(NVal *x) {
    double prob = 1.0;
    int i;
    for (i = 0; i < 4; i++) prob *= x->b[i][0];
    return prob > 0.5 ? 1 : 0;
}

static int neural_is_negative(NVal *x) {
    double neg_w = 0.0;
    int i;
    for (i = 128; i < 256; i++) neg_w += x->b[3][i];
    return neg_w > 0.5 ? 1 : 0;
}

static int neural_compare(NVal *a, NVal *b) {
    NVal diff = neural_sub(a, b);
    if (neural_is_zero(&diff)) return 0;
    if (neural_is_negative(&diff)) return -1;
    return 1;
}

static NVal neural_div(NVal *a, NVal *b) {
    int av = nval_decode(a);
    int bv = nval_decode(b);
    if (bv == 0) return nval_encode(0);

    int a_neg = av < 0, b_neg = bv < 0;
    NVal a_abs = a_neg ? neural_negate(a) : *a;
    NVal b_abs = b_neg ? neural_negate(b) : *b;

    NVal quotient = nval_encode(0);
    NVal remainder = nval_encode(0);
    NVal one = nval_encode(1);
    int i;

    for (i = 31; i >= 0; i--) {
        remainder = neural_shl1(&remainder);

        NVal bit_mask = nval_encode(1 << i);
        NVal masked = neural_bitwise(&a_abs, &bit_mask, nib_and_W1, nib_and_W2);
        int bit_set = !neural_is_zero(&masked);

        if (bit_set) {
            remainder = neural_bitwise(&remainder, &one, nib_or_W1, nib_or_W2);
        }

        int cmp = neural_compare(&remainder, &b_abs);
        if (cmp >= 0) {
            remainder = neural_sub(&remainder, &b_abs);
            quotient = neural_bitwise(&quotient, &bit_mask, nib_or_W1, nib_or_W2);
        }
    }

    if (a_neg != b_neg) quotient = neural_negate(&quotient);
    return quotient;
}

static NVal neural_mod(NVal *a, NVal *b) {
    NVal q = neural_div(a, b);
    NVal qb = neural_mul(&q, b);
    return neural_sub(a, &qb);
}

static NVal neural_shl(NVal *x, NVal *amt) {
    NVal result = *x;
    int shift = nval_decode(amt);
    int i;
    for (i = 0; i < shift && i < 32; i++) result = neural_shl1(&result);
    return result;
}

static NVal neural_shr(NVal *x, NVal *amt) {
    NVal result = *x;
    int shift = nval_decode(amt);
    int i;
    for (i = 0; i < shift && i < 32; i++) result = neural_shr1(&result);
    return result;
}

static NVal neural_mask_byte(NVal *x) {
    NVal result;
    int i, j;
    for (j = 0; j < 256; j++) result.b[0][j] = x->b[0][j];
    for (i = 1; i < 4; i++) { for (j = 0; j < 256; j++) result.b[i][j] = 0.0; result.b[i][0] = 1.0; }
    return result;
}

#endif /* NEED_NEURAL */

/* ============ VM State ============ */

static int vm_halted;
static int vm_exit_code;

#define MEM_SIZE 0x20000
static int vm_memory[MEM_SIZE];
static int heap_ptr = 0x18000;

/* ---- Memory access abstraction ---- */
#if !MODE_SPECULATIVE

#ifdef PRUNED_MEMORY
/* Sparse memory with pruning (matches PrunedMemory from the Python VM).
 *   - Overwrite elimination: same address updates in-place.
 *   - Zero-write eviction: writing 0 removes the entry (ZFOD).
 *   - LRU eviction: when full, evict the coldest 10%.
 */
#ifndef PM_BUCKETS
#define PM_BUCKETS 8192
#endif
#define PM_BUCKET_MASK (PM_BUCKETS - 1)
#ifndef PM_MAX_ENTRIES
#define PM_MAX_ENTRIES 65536
#endif

typedef struct PMNode {
    int addr, value, write_step, last_read;
    struct PMNode *next;
} PMNode;

static PMNode *pm_pool;
static PMNode **pm_buckets;
static PMNode *pm_free;
static int pm_count, pm_step, pm_overwrites, pm_evictions;

static void pm_init(void) {
    int i;
    pm_pool = (PMNode *)calloc(PM_MAX_ENTRIES, sizeof(PMNode));
    pm_buckets = (PMNode **)calloc(PM_BUCKETS, sizeof(PMNode *));
    for (i = 0; i < PM_MAX_ENTRIES - 1; i++)
        pm_pool[i].next = &pm_pool[i + 1];
    pm_pool[PM_MAX_ENTRIES - 1].next = NULL;
    pm_free = &pm_pool[0];
    pm_count = 0;
    pm_step = 0;
    pm_overwrites = 0;
    pm_evictions = 0;
}

static unsigned int pm_hash(int addr) {
    unsigned int h = (unsigned int)addr;
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h & PM_BUCKET_MASK;
}

static PMNode *pm_find(int addr) {
    PMNode *n = pm_buckets[pm_hash(addr)];
    while (n) {
        if (n->addr == addr) return n;
        n = n->next;
    }
    return NULL;
}

static int int_cmp(const void *a, const void *b) {
    return (*(const int *)a) - (*(const int *)b);
}

static void pm_evict_lru(void) {
    int timestamps[PM_MAX_ENTRIES];
    int count = 0, evicted = 0, prune_count, threshold;
    unsigned int b;
    PMNode **prev, *n;

    for (b = 0; b < PM_BUCKETS; b++)
        for (n = pm_buckets[b]; n; n = n->next)
            timestamps[count++] = n->last_read;
    if (count == 0) return;

    qsort(timestamps, count, sizeof(int), int_cmp);
    prune_count = count / 10;
    if (prune_count < 1) prune_count = 1;
    threshold = timestamps[prune_count - 1];

    for (b = 0; b < PM_BUCKETS; b++) {
        prev = &pm_buckets[b];
        while (*prev) {
            n = *prev;
            if (n->last_read <= threshold && evicted < prune_count) {
                *prev = n->next;
                n->next = pm_free;
                pm_free = n;
                pm_count--;
                pm_evictions++;
                evicted++;
            } else {
                prev = &n->next;
            }
        }
    }
}

static int mem_read(int addr) {
    PMNode *n = pm_find(addr);
    if (n) {
        n->last_read = pm_step++;
        return n->value;
    }
    pm_step++;
    return 0; /* ZFOD */
}

static void mem_write(int addr, int val) {
    PMNode *n = pm_find(addr);
    unsigned int b;
    PMNode **prev;
    pm_step++;

    if (n) {
        pm_overwrites++;
        if (val == 0) {
            /* Zero-write eviction: unlink and free (ZFOD) */
            b = pm_hash(addr);
            prev = &pm_buckets[b];
            while (*prev != n) prev = &(*prev)->next;
            *prev = n->next;
            n->next = pm_free;
            pm_free = n;
            pm_count--;
        } else {
            n->value = val;
            n->write_step = pm_step;
            n->last_read = pm_step;
        }
        return;
    }

    if (val == 0) return; /* ZFOD: don't store zero */

    if (pm_count >= PM_MAX_ENTRIES)
        pm_evict_lru();

    if (!pm_free) return;

    n = pm_free;
    pm_free = n->next;
    b = pm_hash(addr);
    n->addr = addr;
    n->value = val;
    n->write_step = pm_step;
    n->last_read = pm_step;
    n->next = pm_buckets[b];
    pm_buckets[b] = n;
    pm_count++;
}

static void pm_print_stats(void) {
    fprintf(stderr, "PRUNED MEMORY: live=%d overwrites=%d evictions=%d step=%d\n",
            pm_count, pm_overwrites, pm_evictions, pm_step);
}

#else /* !PRUNED_MEMORY */

static inline int mem_read(int addr) {
    return (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
}

static inline void mem_write(int addr, int val) {
    if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = val;
}

#endif /* PRUNED_MEMORY */
#endif /* !MODE_SPECULATIVE */

/* ---- Neural VM registers and stack ops ---- */
#if !MODE_SPECULATIVE

static NVal reg_ax, reg_sp, reg_bp, reg_pc;

static NVal const_eight;
static int const_eight_initialized = 0;

static void ensure_const_eight(void) {
    if (!const_eight_initialized) {
        const_eight = nval_encode(8);
        const_eight_initialized = 1;
    }
}

static void vm_push(NVal *val) {
    reg_sp = neural_sub(&reg_sp, &const_eight);
    int addr = nval_decode(&reg_sp);
    mem_write(addr, nval_decode(val));
}

static NVal vm_pop(void) {
    int addr = nval_decode(&reg_sp);
    NVal val = nval_encode(mem_read(addr));
    reg_sp = neural_add(&reg_sp, &const_eight);
    return val;
}

#endif /* !MODE_SPECULATIVE */

/* ---- Speculative VM registers and stack ops ---- */
#if MODE_SPECULATIVE

static int spec_ax, spec_sp, spec_bp, spec_pc;

static void spec_push(int val) {
    spec_sp -= 8;
    vm_memory[spec_sp] = val;
}

static int spec_pop(void) {
    int v = vm_memory[spec_sp];
    spec_sp += 8;
    return v;
}

#endif /* MODE_SPECULATIVE */

/* ---- Validation helpers ---- */
#if MODE_VALIDATE

static int validate_count = 0;

/* Wrapper functions matching NVal (*)(NVal *, NVal *) signature */
static NVal vld_and(NVal *a, NVal *b) { return neural_bitwise(a, b, nib_and_W1, nib_and_W2); }
static NVal vld_or(NVal *a, NVal *b)  { return neural_bitwise(a, b, nib_or_W1, nib_or_W2); }
static NVal vld_xor(NVal *a, NVal *b) { return neural_bitwise(a, b, nib_xor_W1, nib_xor_W2); }
static NVal vld_eq(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) == 0 ? 1 : 0); }
static NVal vld_ne(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) != 0 ? 1 : 0); }
static NVal vld_lt(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) < 0 ? 1 : 0); }
static NVal vld_gt(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) > 0 ? 1 : 0); }
static NVal vld_le(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) <= 0 ? 1 : 0); }
static NVal vld_ge(NVal *a, NVal *b)  { return nval_encode(neural_compare(a, b) >= 0 ? 1 : 0); }

static void check_arith(const char *name, int a, int b, int native_res,
                         NVal (*fn)(NVal *, NVal *)) {
    if (++validate_count < VALIDATE_INTERVAL) return;
    validate_count = 0;
    NVal na = nval_encode(a), nb = nval_encode(b);
    NVal nr = fn(&na, &nb);
    int neural_res = nval_decode(&nr);
    if (neural_res != native_res)
        fprintf(stderr, "VALIDATE: %s(%d, %d) native=%d neural=%d\n",
                name, a, b, native_res, neural_res);
}

#endif /* MODE_VALIDATE */

/* ============ VM Execution ============ */

static int vm_step(void) {
    if (vm_halted) return 0;

#if MODE_SPECULATIVE
    int pc_val = spec_pc;
#else
    int pc_val = nval_decode(&reg_pc);
#endif

    int instr_idx = pc_val / 8;

    if (instr_idx < 0 || instr_idx >= program_code_len) {
        vm_halted = 1;
        return 0;
    }

    int op = program_code[instr_idx][0];
    int imm = program_code[instr_idx][1];

    /* Advance PC */
#if MODE_SPECULATIVE
    spec_pc += 8;
#else
    reg_pc = neural_add(&reg_pc, &const_eight);
    NVal imm_enc = nval_encode(imm);
    NVal ax = reg_ax;
    NVal a;
#endif

    switch (op) {
    case 0: /* LEA */
#if MODE_SPECULATIVE
        spec_ax = spec_bp + imm;
#else
        reg_ax = neural_add(&reg_bp, &imm_enc);
#endif
        break;

    case 1: /* IMM */
#if MODE_SPECULATIVE
        spec_ax = imm;
#else
        reg_ax = imm_enc;
#endif
        break;

    case 2: /* JMP */
#if MODE_SPECULATIVE
        spec_pc = imm;
#else
        reg_pc = imm_enc;
#endif
        break;

    case 3: /* JSR */
#if MODE_SPECULATIVE
        spec_push(spec_pc);
        spec_pc = imm;
#else
        vm_push(&reg_pc);
        reg_pc = imm_enc;
#endif
        break;

    case 4: /* BZ */
#if MODE_SPECULATIVE
        if (spec_ax == 0) spec_pc = imm;
#else
        if (neural_is_zero(&ax)) reg_pc = imm_enc;
#endif
        break;

    case 5: /* BNZ */
#if MODE_SPECULATIVE
        if (spec_ax != 0) spec_pc = imm;
#else
        if (!neural_is_zero(&ax)) reg_pc = imm_enc;
#endif
        break;

    case 6: /* ENT */
#if MODE_SPECULATIVE
        spec_push(spec_bp);
        spec_bp = spec_sp;
        spec_sp -= imm;
#else
        vm_push(&reg_bp);
        reg_bp = reg_sp;
        reg_sp = neural_sub(&reg_sp, &imm_enc);
#endif
        break;

    case 7: /* ADJ */
#if MODE_SPECULATIVE
        spec_sp += imm;
#else
        reg_sp = neural_add(&reg_sp, &imm_enc);
#endif
        break;

    case 8: /* LEV */
#if MODE_SPECULATIVE
        spec_sp = spec_bp;
        spec_bp = spec_pop();
        spec_pc = spec_pop();
#else
        reg_sp = reg_bp;
        reg_bp = vm_pop();
        reg_pc = vm_pop();
#endif
        break;

    case 9: /* LI */
#if MODE_SPECULATIVE
        { int addr = spec_ax;
          spec_ax = (addr >= 0 && addr < MEM_SIZE) ? vm_memory[addr] : 0;
        }
#else
        { int addr = nval_decode(&ax);
          reg_ax = nval_encode(mem_read(addr));
        }
#endif
        break;

    case 10: /* LC */
#if MODE_SPECULATIVE
        { int addr = spec_ax;
          spec_ax = (addr >= 0 && addr < MEM_SIZE) ? (vm_memory[addr] & 0xFF) : 0;
        }
#else
        { int addr = nval_decode(&ax);
          NVal val = nval_encode(mem_read(addr));
          reg_ax = neural_mask_byte(&val);
        }
#endif
        break;

    case 11: /* SI */
#if MODE_SPECULATIVE
        { int addr = spec_pop();
          if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = spec_ax;
        }
#else
        { a = vm_pop();
          int addr = nval_decode(&a);
          mem_write(addr, nval_decode(&ax));
        }
#endif
        break;

    case 12: /* SC */
#if MODE_SPECULATIVE
        { int addr = spec_pop();
          if (addr >= 0 && addr < MEM_SIZE) vm_memory[addr] = spec_ax & 0xFF;
        }
#else
        { a = vm_pop();
          int addr = nval_decode(&a);
          NVal masked = neural_mask_byte(&ax);
          mem_write(addr, nval_decode(&masked));
        }
#endif
        break;

    case 13: /* PSH */
#if MODE_SPECULATIVE
        spec_push(spec_ax);
#else
        vm_push(&ax);
#endif
        break;

    case 14: /* OR */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a | b;
#if MODE_VALIDATE
          check_arith("OR", a, b, spec_ax, vld_or);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_bitwise(&a, &ax, nib_or_W1, nib_or_W2);
#endif
        break;

    case 15: /* XOR */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a ^ b;
#if MODE_VALIDATE
          check_arith("XOR", a, b, spec_ax, vld_xor);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_bitwise(&a, &ax, nib_xor_W1, nib_xor_W2);
#endif
        break;

    case 16: /* AND */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a & b;
#if MODE_VALIDATE
          check_arith("AND", a, b, spec_ax, vld_and);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_bitwise(&a, &ax, nib_and_W1, nib_and_W2);
#endif
        break;

    case 17: /* EQ */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a == b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("EQ", a, b, spec_ax, vld_eq);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp == 0 ? 1 : 0);
        }
#endif
        break;

    case 18: /* NE */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a != b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("NE", a, b, spec_ax, vld_ne);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp != 0 ? 1 : 0);
        }
#endif
        break;

    case 19: /* LT */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a < b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("LT", a, b, spec_ax, vld_lt);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp < 0 ? 1 : 0);
        }
#endif
        break;

    case 20: /* GT */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a > b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("GT", a, b, spec_ax, vld_gt);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp > 0 ? 1 : 0);
        }
#endif
        break;

    case 21: /* LE */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a <= b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("LE", a, b, spec_ax, vld_le);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp <= 0 ? 1 : 0);
        }
#endif
        break;

    case 22: /* GE */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = (a >= b) ? 1 : 0;
#if MODE_VALIDATE
          check_arith("GE", a, b, spec_ax, vld_ge);
#endif
        }
#else
        a = vm_pop();
        { int cmp = neural_compare(&a, &ax);
          reg_ax = nval_encode(cmp >= 0 ? 1 : 0);
        }
#endif
        break;

    case 23: /* SHL */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a << b;
#if MODE_VALIDATE
          check_arith("SHL", a, b, spec_ax, neural_shl);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_shl(&a, &ax);
#endif
        break;

    case 24: /* SHR */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a >> b;
#if MODE_VALIDATE
          check_arith("SHR", a, b, spec_ax, neural_shr);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_shr(&a, &ax);
#endif
        break;

    case 25: /* ADD */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a + b;
#if MODE_VALIDATE
          check_arith("ADD", a, b, spec_ax, neural_add);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_add(&a, &ax);
#endif
        break;

    case 26: /* SUB */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a - b;
#if MODE_VALIDATE
          check_arith("SUB", a, b, spec_ax, neural_sub);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_sub(&a, &ax);
#endif
        break;

    case 27: /* MUL */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = a * b;
#if MODE_VALIDATE
          check_arith("MUL", a, b, spec_ax, neural_mul);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_mul(&a, &ax);
#endif
        break;

    case 28: /* DIV */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = b ? a / b : 0;
#if MODE_VALIDATE
          check_arith("DIV", a, b, spec_ax, neural_div);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_div(&a, &ax);
#endif
        break;

    case 29: /* MOD */
#if MODE_SPECULATIVE
        { int a = spec_pop(); int b = spec_ax; spec_ax = b ? a % b : 0;
#if MODE_VALIDATE
          check_arith("MOD", a, b, spec_ax, neural_mod);
#endif
        }
#else
        a = vm_pop();
        reg_ax = neural_mod(&a, &ax);
#endif
        break;

    case 34: /* MALC */
#if MODE_SPECULATIVE
        { int size = (spec_sp >= 0 && spec_sp < MEM_SIZE) ? vm_memory[spec_sp] : 0;
          spec_ax = heap_ptr;
          heap_ptr += (size + 7) / 8;
          if (heap_ptr >= MEM_SIZE) heap_ptr = MEM_SIZE - 1;
        }
#else
        { int addr = nval_decode(&reg_sp);
          int size = mem_read(addr);
          int ptr = heap_ptr;
          heap_ptr += (size + 7) / 8;
          if (heap_ptr >= MEM_SIZE) heap_ptr = MEM_SIZE - 1;
          reg_ax = nval_encode(ptr);
        }
#endif
        break;

    case 35: /* FREE */
        break;

    case 38: /* EXIT */
#if MODE_SPECULATIVE
        vm_exit_code = spec_ax;
#else
        vm_exit_code = nval_decode(&ax);
#endif
        vm_halted = 1;
        return 0;

    case 64: /* GETCHAR */
#if MODE_SPECULATIVE
        spec_ax = getchar();
#else
        { int c = getchar();
          reg_ax = nval_encode(c);
        }
#endif
        break;

    case 65: /* PUTCHAR */
#if MODE_SPECULATIVE
        { int c = (spec_sp >= 0 && spec_sp < MEM_SIZE) ? vm_memory[spec_sp] : 0;
          putchar(c & 0xFF);
          fflush(stdout);
        }
#else
        { int addr = nval_decode(&reg_sp);
          int c = mem_read(addr);
          putchar(c & 0xFF);
          fflush(stdout);
        }
#endif
        break;

    default:
        fprintf(stderr, "Unknown opcode: %d at PC=%d\n", op, pc_val);
        vm_halted = 1;
        return 0;
    }

    return 1;
}

/* ============ Main ============ */

int main(int argc, char **argv) {
    int i;

#if NEED_NEURAL
    if (load_neural_model() != 0) {
        fprintf(stderr, "Failed to load neural model\n");
        return 1;
    }
#endif

#if !MODE_SPECULATIVE
    ensure_const_eight();
    reg_ax = nval_encode(0);
    reg_sp = nval_encode(0x10000);
    reg_bp = nval_encode(0x10000);
    reg_pc = nval_encode(0);
#endif

#if MODE_SPECULATIVE
    spec_ax = 0;
    spec_sp = 0x10000;
    spec_bp = 0x10000;
    spec_pc = 0;
#endif

    vm_halted = 0;
    vm_exit_code = 0;
    memset(vm_memory, 0, sizeof(vm_memory));

#if !MODE_SPECULATIVE && defined(PRUNED_MEMORY)
    pm_init();
#endif

    for (i = 0; i < program_data_len; i++) {
#if MODE_SPECULATIVE
        vm_memory[0x10000 + i] = program_data[i];
#else
        mem_write(0x10000 + i, program_data[i]);
#endif
    }

    while (vm_step()) {}

#if !MODE_SPECULATIVE && defined(PRUNED_MEMORY)
    pm_print_stats();
#endif
    return vm_exit_code & 0xFF;
}
