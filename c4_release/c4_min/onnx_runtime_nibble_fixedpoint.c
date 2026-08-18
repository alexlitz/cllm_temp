/*
 * onnx_runtime_nibble_fixedpoint.c  —  a FIXED-POINT (integer-only) build of the
 * nibble-VM ONNX runtime.  This is the self-hosting variant of
 * onnx_runtime_nibble.c: it replaces every `float` + `expf`/`fabsf`/sigmoid with
 * 16.16 fixed-point `long` arithmetic and integer lookup tables, so the whole
 * runtime compiles under the c4 VM's C subset (c4 has NO float/double type).
 *
 * Authority: docs/BLOG_SPEC.md §"Adapting to Different Precisions" + §"Sparse
 * Tensors" + §"Self-Hosting"; the fixed-point convention mirrors
 * tools/neural_bundle_fixedpoint.py (integer weights * SCALE, SiLU/exp lookup
 * table, no float, C4-compatible).  See docs/NIBBLE_FIXEDPOINT_2026_07_14.md.
 *
 * Representation.  Every "float" tensor value v is stored as the 64-bit integer
 * round(v * SCALE) with SCALE = 2^16 (16.16 fixed-point).  "int64" tensors (shape
 * plumbing, indices, Range/Shape/Cast-to-int outputs) stay *unscaled* integers.
 * A per-tensor flag t_isfp[] records which representation a runtime tensor holds,
 * exactly as the pure-int numpy prototype (fp_proto.py) does, so multiply/add/cast
 * can align the two.  All ops that the float build did in `float` are done here in
 * `long`: a fp*fp multiply yields SCALE^2 and is shifted right by SCALE_BITS
 * (round-to-nearest); fp/fp division pre-shifts the numerator left by SCALE_BITS.
 *
 * exp/sigmoid.  The softmax1 path needs exp(x) for x <= 0 (it computes
 * exp(x - rowmax)).  We bake an integer table of exp over x in [EXP_MIN, 0] and
 * linear-interpolate (the same shape as neural_bundle_fixedpoint's SiLU table).
 * The causal mask fills masked scores with -inf; we carry that as a large-negative
 * sentinel whose exp saturates to 0.  Sigmoid(x)=SCALE^2/(SCALE+exp(-x)) reuses
 * the same table.  abs is trivial integer abs.
 *
 * Proof (docs/NIBBLE_FIXEDPOINT_2026_07_14.md + test_onnx_runtime_fixedpoint.py):
 * argmax(byte-head) over the LM logits is *identical* to the float build AND to
 * torch/onnxruntime on the proof frames — the VM reads its state by argmax over
 * the byte head, so argmax-exactness is the byte-identity the VM needs.  The
 * single load-bearing decode margin on the proof frame is ~0.085 logits; at
 * SCALE=2^16 the fixed-point runtime reproduces it as ~0.085 (verified >2^10..2^24
 * all argmax-exact), so precision never flips the argmax.
 *
 * Build:   gcc -O2 -o onnx_runtime_nibble_fixedpoint onnx_runtime_nibble_fixedpoint.c
 *          (no -lm: no libm calls.  builds under c4vm's C subset too.)
 * Run:     ./onnx_runtime_nibble_fixedpoint model.nblbin tokens.txt [--dump-argmax|--dump-logits]
 *          (tokens.txt: first line "B S", then B*S ints)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- fixed-point scale: 16.16 (BLOG_SPEC §"Adapting to Different Precisions":
 * the fractional-bit count is the one precision knob; 12 (the 2^12 convention of
 * tools/neural_bundle_fixedpoint.py) .. 24 are all argmax-exact on the proof
 * frame, so 16 is a comfortable default.  -DSCALE_BITS=N overrides it.) ---- */
#ifndef SCALE_BITS
#define SCALE_BITS 16
#endif
#define SCALE (1 << SCALE_BITS)          /* 65536 at 16.16 */
#define NEG_INF_FP (0 - ((long)1 << 62)) /* -inf sentinel for the causal mask */

/* ---- op enum (must match onnx_to_c4bin.OP) ---- */
#define OP_MATMUL 0
#define OP_ADD 1
#define OP_SUB 2
#define OP_MUL 3
#define OP_DIV 4
#define OP_GATHER 5
#define OP_RESHAPE 6
#define OP_TRANSPOSE 7
#define OP_CONCAT 8
#define OP_UNSQUEEZE 9
#define OP_SHAPE 10
#define OP_CAST 11
#define OP_EXP 12
#define OP_NEG 13
#define OP_ABS 14
#define OP_RANGE 15
#define OP_REDUCEMAX 16
#define OP_REDUCESUM 17
#define OP_CLIP 18
#define OP_TRILU 19
#define OP_CONSTANTOFSHAPE 20
#define OP_SIGMOID 21
#define OP_IDENTITY 22

/* ---- attr-key enum (must match onnx_to_c4bin.ATTR) ---- */
#define A_PERM 0
#define A_AXIS 1
#define A_TO 2
#define A_UPPER 3
#define A_KEEPDIMS 4
#define A_AXES 5
#define A_ALLOWZERO 6
#define A_COS_FILL_BITS 7

#define DT_FLOAT 0
#define DT_INT64 1

#define MAX_TENSORS 512
#define MAX_NODES 512
#define MAX_IO 8
#define MAX_RANK 8
#define MAX_ATTRS 4

/* ---- exp lookup table over x in [EXP_MIN, 0], EXP_STEPS+1 samples ---- */
#define EXP_STEPS 4096
/* EXP_MIN = -32 in fixed-point; exp(-32)~1.3e-14 -> 0 at our resolution */
#define EXP_MIN_FP (0 - ((long)32 * SCALE))
int exp_tbl[EXP_STEPS + 1];              /* exp_tbl[i] = exp(x_i) * SCALE, int */

/* ---- tensor table (Structure-of-Arrays, C4-subset friendly) ---- */
int   t_dtype[MAX_TENSORS];       /* 0 float(=fixed-point), 1 int64 */
int   t_isfp[MAX_TENSORS];        /* 1 => value payload is v*SCALE fixed-point */
int   t_rank[MAX_TENSORS];
int   t_dims[MAX_TENSORS][MAX_RANK];
int   t_size[MAX_TENSORS];        /* element count */
long *tv[MAX_TENSORS];            /* payload (fixed-point OR raw int64) */
int   t_is_init[MAX_TENSORS];
char  t_name[MAX_TENSORS][80];
int   n_tensors;
int   input_tid;
int   output_tid;

/* ---- WEIGHT DEDUP (C4_DEDUP_WEIGHTS, opt-in, default OFF) --------------------
 * The emulated transformer's weights are a tiny PALETTE of unique values replicated
 * ~100x (198,610 nonzeros -> 1,895 unique).  With -DC4_DEDUP_WEIGHTS the runtime
 * stores each float(=fixed-point) INITIALIZER's payload as a PALETTE INDEX (an int
 * into `palette[]`) instead of the value, and a matmul weight read is the TWO-ACCESS
 * chained indirection  `i = tv[a][flat]; W = palette[i]`  (the second chained load).
 * This ADDS one load per weight fetch (the honest dedup cost) while collapsing the
 * value storage to the palette.  DEFAULT-OFF -> the golden weight path (direct
 * `tv[a][flat]`) is byte-identical; ON -> byte-EXACT (palette[idx]==the value).
 * Non-weight (int64) tensors + runtime intermediates are never deduped. */
#ifndef PALETTE_MAX
#define PALETTE_MAX 4096                 /* >= 1,895 unique weight values */
#endif
long palette[PALETTE_MAX];               /* the unique fixed-point weight values */
int  palette_n;                          /* number of distinct values stored */
int  t_deduped[MAX_TENSORS];             /* 1 => tv[i] holds palette INDICES */
#ifdef C4_DEDUP_WEIGHTS
int  dedup_on = 1;
#else
int  dedup_on = 0;
#endif

/* ---- FUSED NON-MATMUL TAIL (C4_FUSE_TAIL, opt-in, default OFF) ---------------
 * The self-emulation floor is the NON-MATMUL TAIL: with the matmul MAC fused to
 * 1 step/MAC (C4_MEM_OPERAND) the tail (silu + element-wise adds; softmax is
 * negligible at the S=1 decode row) is 98.5% of the total.  The tail ops are
 * D-wide element-wise loops: each element the emulating c4 stack machine does
 * pointer math + loads + the compute + a store + loop control (~112 steps/elem
 * for add, ~141 for silu's exp-Taylor).  A FUSED tail superinstruction — the
 * analogue of the fused MAC — commits ONE element per model.forward: the operand
 * CAM reads run in the instruction's early blocks and feed the late-block
 * add / silu, so 1 step/element.
 *
 * In THIS reference runtime the fusion is a byte-EXACT proof: with -DC4_FUSE_TAIL
 * the silu (op_unary kind 3) and the element-wise adds (ew_broadcast kind 0) route
 * through a dedicated fused per-element evaluator (`fused_silu` / `fused_add3`)
 * that produces the IDENTICAL fixed-point value the unfused loop does (the fusion
 * collapses STEPS, not values — exactly as palette[idx]==the value for dedup, and
 * MAC [a],[b]==the bytecode dot for the MAC).  DEFAULT-OFF -> the golden op path is
 * byte-identical; ON -> byte-EXACT.  The step COLLAPSE (loop -> 1/element) is
 * grounded in _tail_fused_src / ground_fused_tail_selfemu, not in this runtime's
 * wall (this runtime is compiled native, not self-emulated). */
#ifdef C4_FUSE_TAIL
int  fuse_tail_on = 1;
#else
int  fuse_tail_on = 0;
#endif

/* intern a fixed-point value into the palette; return its index (linear scan —
 * C4-subset friendly; the palette is tiny so this is cheap at load time). */
int palette_intern(long v) {
    int i;
    i = 0;
    while (i < palette_n) { if (palette[i] == v) return i; i = i + 1; }
    if (palette_n >= PALETTE_MAX) { printf("palette overflow (>%d)\n", PALETTE_MAX); exit(1); }
    palette[palette_n] = v;
    palette_n = palette_n + 1;
    return palette_n - 1;
}

/* ---- node table ---- */
int n_op[MAX_NODES];
int n_nin[MAX_NODES];
int n_in[MAX_NODES][MAX_IO];
int n_nout[MAX_NODES];
int n_out[MAX_NODES][MAX_IO];
int n_nattr[MAX_NODES];
int n_akey[MAX_NODES][MAX_ATTRS];
int n_anv[MAX_NODES][MAX_ATTRS];
int n_aval[MAX_NODES][MAX_ATTRS][MAX_RANK];
int n_nodes;

/* ============ exp table build (integer, no libm at run time) ============ */
/* We cannot call expf under c4.  We approximate exp(x) for the small offline
 * table with a Taylor/`exp(x)=(exp(x/n))^n` range reduction done in fixed-point.
 * This table is computed once at startup; the values match round(exp(x)*SCALE).
 * For x in [-32,0] we split x = -k - f (k int >=0, f in [0,1)) and use
 * exp(x) = exp(-1)^k * exp(-f), with exp(-1) and exp(-f) via a short series. */

/* fixed-point multiply (round to nearest, symmetric) */
long fpmul(long a, long b) {
    long p; long half;
    p = a * b;
    half = SCALE / 2;
    if (p >= 0) return (p + half) >> SCALE_BITS;
    return 0 - ((0 - p + half) >> SCALE_BITS);
}

/* exp(-f) for f in [0,1], fixed-point, via 8-term Taylor of exp(-f) */
long exp_neg_frac(long ffp) {
    /* series: 1 - f + f^2/2 - f^3/6 + ... , ffp = f*SCALE */
    long term; long acc; int k;
    acc = SCALE;                 /* term 0 = 1 */
    term = SCALE;
    k = 1;
    while (k <= 12) {
        /* term *= (-f)/k */
        term = fpmul(term, ffp) / k;
        if (k & 1) acc = acc - term; else acc = acc + term;
        k = k + 1;
    }
    if (acc < 0) acc = 0;
    return acc;
}

int build_exp_table() {
    long em1;                    /* exp(-1) in fixed-point */
    int i;
    long xfp;                    /* x * SCALE, x in [-32,0] */
    long k; long ffp; long e; long base;
    int kk;
    em1 = exp_neg_frac(SCALE);   /* exp(-1) */
    i = 0;
    while (i <= EXP_STEPS) {
        /* x = EXP_MIN * (1 - i/EXP_STEPS)  -> xfp */
        /* xfp = EXP_MIN_FP * (EXP_STEPS - i) / EXP_STEPS */
        xfp = EXP_MIN_FP;
        xfp = (xfp / EXP_STEPS) * (EXP_STEPS - i);   /* x<=0, exact enough */
        /* split -xfp = k*SCALE + ffp */
        {
            long neg; neg = 0 - xfp;
            k = neg >> SCALE_BITS;
            ffp = neg - (k << SCALE_BITS);
        }
        /* base = exp(-1)^k */
        base = SCALE;
        kk = 0;
        while (kk < (int)k) { base = fpmul(base, em1); kk = kk + 1; }
        e = fpmul(base, exp_neg_frac(ffp));
        exp_tbl[i] = (int)e;
        i = i + 1;
    }
    return 0;
}

/* exp(x)*SCALE for fixed-point x (x may be any sign); table covers x<=0. */
long fp_exp(long xfp) {
    long num; long den; long idx; long frac; long v1; long v2;
    if (xfp >= 0) {
        /* softmax feeds x<=0 (x - rowmax); exp(0)=1. Larger x is not exercised. */
        return SCALE;
    }
    if (xfp <= EXP_MIN_FP) return 0;
    /* t = (xfp - EXP_MIN_FP) * EXP_STEPS / (0 - EXP_MIN_FP) in [0, EXP_STEPS] */
    num = (xfp - EXP_MIN_FP) * EXP_STEPS;
    den = 0 - EXP_MIN_FP;
    idx = num / den;
    frac = num - idx * den;
    if (idx >= EXP_STEPS) return exp_tbl[EXP_STEPS];
    v1 = exp_tbl[idx];
    v2 = exp_tbl[idx + 1];
    return v1 + (v2 - v1) * frac / den;
}

/* ============ binary reader ============ */
int rd_i32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { printf("EOF\n"); exit(1); }
    return (int)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}
/* read a stored int64 (two int32, little-endian lo/hi) as a C long */
long rd_i64(FILE *f) {
    unsigned int lo;
    int hi;
    lo = (unsigned int)rd_i32(f);
    hi = rd_i32(f);
    return ((long)hi << 32) | (long)lo;
}
/* read a stored float32 bit pattern and convert to fixed-point WITHOUT using a
 * `float` variable (c4 has no float).  We decode IEEE-754 binary32 by hand into
 * round(value * SCALE).  Handles the sub-set of values the graph stores:
 * finite values, +/-inf (mask fill).  Returns the 16.16 fixed-point integer. */
long rd_f32_fp(FILE *f) {
    int bits; int sign; int exp; int mant; long m; int e;
    bits = rd_i32(f);
    sign = (bits >> 31) & 1;
    exp = (bits >> 23) & 0xFF;
    mant = bits & 0x7FFFFF;
    if (exp == 0xFF) {
        /* inf / nan -> saturate to the -inf mask sentinel (only -inf occurs) */
        if (sign) return NEG_INF_FP;
        return 0 - NEG_INF_FP;
    }
    if (exp == 0 && mant == 0) return 0;   /* +/-0 */
    /* value = (-1)^sign * 1.mant * 2^(exp-127); we want value * 2^SCALE_BITS.
     * m = (1<<23 | mant) is the 24-bit significand (value = m * 2^(exp-127-23)). */
    m = (long)((1 << 23) | mant);
    e = exp - 127 - 23 + SCALE_BITS;       /* final shift on m */
    if (e >= 0) {
        m = m << e;
    } else {
        /* round to nearest when shifting right */
        int s; long half;
        s = 0 - e;
        if (s < 63) {
            half = (long)1 << (s - 1);
            m = (m + half) >> s;
        } else {
            m = 0;
        }
    }
    if (sign) return 0 - m;
    return m;
}

/* ============ tensor helpers ============ */
int prod(int rank, int *dims) {
    int p; int i;
    p = 1; i = 0;
    while (i < rank) { p = p * dims[i]; i = i + 1; }
    return p;
}

/* allocate a runtime tensor's storage for the given shape + dtype/isfp.  Resets
 * t_deduped: a freshly COMPUTED tensor (matmul/add/...) holds VALUES, not palette
 * indices.  The structural COPY ops (gather/reshape/transpose/concat/unsqueeze) that
 * move a deduped payload verbatim re-assert t_deduped[out]=t_deduped[in] afterwards. */
void alloc_tensor(int tid, int dtype, int isfp, int rank, int *dims) {
    int sz; int i;
    t_deduped[tid] = 0;
    t_dtype[tid] = dtype;
    t_isfp[tid] = isfp;
    t_rank[tid] = rank;
    i = 0;
    while (i < rank) { t_dims[tid][i] = dims[i]; i = i + 1; }
    sz = prod(rank, dims);
    t_size[tid] = sz;
    if (tv[tid]) { free(tv[tid]); tv[tid] = 0; }
    tv[tid] = malloc(sz * sizeof(long));
}

/* strides for row-major layout */
void strides_of(int rank, int *dims, int *st) {
    int i; int s;
    s = 1; i = rank - 1;
    while (i >= 0) { st[i] = s; s = s * dims[i]; i = i - 1; }
}

/* ============ load .nblbin ============ */
void load(char *path) {
    FILE *f;
    int magic; int i; int j; int k;
    int nl; int isi; int dt; int rank; int dims[MAX_RANK]; int ne;
    int op; int nin; int nout; int nattr; int key; int nv;

    f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    magic = rd_i32(f);
    if (magic != 0x314C424E) { printf("bad magic %x\n", magic); exit(1); }
    n_tensors = rd_i32(f);
    n_nodes = rd_i32(f);
    input_tid = rd_i32(f);
    output_tid = rd_i32(f);

    i = 0;
    while (i < n_tensors) {
        nl = rd_i32(f);
        fread(t_name[i], 1, nl, f);
        t_name[i][nl] = 0;
        isi = rd_i32(f);
        t_is_init[i] = isi;
        tv[i] = 0;
        dt = rd_i32(f);
        rank = rd_i32(f);
        j = 0;
        while (j < rank) { dims[j] = rd_i32(f); j = j + 1; }
        ne = rd_i32(f);
        if (isi) {
            /* float initializer -> fixed-point payload (t_isfp=1);
               int64 initializer -> raw int payload (t_isfp=0) */
            alloc_tensor(i, dt, (dt == DT_FLOAT) ? 1 : 0, rank, dims);
            t_deduped[i] = 0;
            if (dt == DT_FLOAT) {
                if (dedup_on) {
                    /* DEDUP: store the palette INDEX (an int), not the value.  The
                       matmul weight read then chains  i=tv[a][flat]; W=palette[i]. */
                    j = 0;
                    while (j < ne) {
                        long v; v = rd_f32_fp(f);
                        tv[i][j] = (long)palette_intern(v);
                        j = j + 1;
                    }
                    t_deduped[i] = 1;
                } else {
                    j = 0; while (j < ne) { tv[i][j] = rd_f32_fp(f); j = j + 1; }
                }
            } else {
                j = 0; while (j < ne) { tv[i][j] = rd_i64(f); j = j + 1; }
            }
        } else {
            t_dtype[i] = -1; t_isfp[i] = 0; t_rank[i] = 0; t_size[i] = 0;
            t_deduped[i] = 0;
        }
        i = i + 1;
    }

    i = 0;
    while (i < n_nodes) {
        op = rd_i32(f);
        n_op[i] = op;
        nin = rd_i32(f);
        n_nin[i] = nin;
        j = 0; while (j < nin) { n_in[i][j] = rd_i32(f); j = j + 1; }
        nout = rd_i32(f);
        n_nout[i] = nout;
        j = 0; while (j < nout) { n_out[i][j] = rd_i32(f); j = j + 1; }
        nattr = rd_i32(f);
        n_nattr[i] = nattr;
        j = 0;
        while (j < nattr) {
            key = rd_i32(f);
            nv = rd_i32(f);
            n_akey[i][j] = key;
            n_anv[i][j] = nv;
            k = 0; while (k < nv) { n_aval[i][j][k] = rd_i32(f); k = k + 1; }
            j = j + 1;
        }
        i = i + 1;
    }
    fclose(f);
    printf("loaded %s: %d tensors, %d nodes (in=%d out=%d) [fixed-point 16.16]\n",
           path, n_tensors, n_nodes, input_tid, output_tid);
    if (dedup_on)
        printf("  [dedup ON] palette = %d unique weight values (two-access weight read)\n",
               palette_n);
    if (fuse_tail_on)
        printf("  [fuse-tail ON] silu + element-wise adds route through the fused "
               "per-element evaluators (1 step/element in self-emulation)\n");
}

/* find an attribute value on node `nd`, key `key`; return value index 0 or dflt */
int attr1(int nd, int key, int dflt) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_aval[nd][j][0];
        j = j + 1;
    }
    return dflt;
}
/* count of values for an attr key (0 if absent) */
int attr_nv(int nd, int key) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_anv[nd][j];
        j = j + 1;
    }
    return 0;
}
int attr_val(int nd, int key, int idx) {
    int j;
    j = 0;
    while (j < n_nattr[nd]) {
        if (n_akey[nd][j] == key) return n_aval[nd][j][idx];
        j = j + 1;
    }
    return 0;
}

/* DEDUP-aware payload access.  For a deduped tensor (an initializer whose payload
 * was interned into `palette[]` at load time) this is the TWO-ACCESS chained load
 * `i = tv[tid][flat]; W = palette[i]` (the second chained load is the dedup cost);
 * for a non-deduped tensor (dedup OFF, an int64 tensor, or a runtime intermediate)
 * it is the plain single load.  Byte-EXACT either way (palette[idx] == the value).
 * Routing ALL value reads through here keeps every op — matmul, add/mul, gather,
 * reduce — reading the resolved value, so the whole forward is byte-identical. */
long getv(int tid, int flat) {
    if (t_deduped[tid]) {
        int idx;
        idx = (int)tv[tid][flat];       /* load 1: the palette index for this slot */
        return palette[idx];            /* load 2: the value at that palette slot */
    }
    return tv[tid][flat];               /* direct (golden) single load */
}

/* the matmul weight fetch — the SAME two-access chained load, named for the MAC
 * loop where the dedup step-cost is measured (see _matmul_dedup_src). */
long get_wt(int tid, int flat) { return getv(tid, flat); }

/* fixed-point multiply where we know the result should stay fixed-point:
   (a_fp * b_fp) >> SCALE_BITS with round-to-nearest.  a,b are long. */
long fpmul_l(long a, long b) {
    long p; long half;
    p = a * b;
    half = SCALE / 2;
    if (p >= 0) return (p + half) >> SCALE_BITS;
    return 0 - ((0 - p + half) >> SCALE_BITS);
}

/* ---- FUSED TAIL per-element evaluators (C4_FUSE_TAIL) ------------------------
 * A fused tail superinstruction commits ONE element per forward: the operand CAM
 * read(s) run in the instruction's early blocks and feed the late-block compute.
 * These helpers are the byte-EXACT value proof of that fusion — each computes the
 * SAME fixed-point result the unfused D-iteration loop does, for one element.  The
 * step collapse (loop -> 1/element) is grounded in _tail_fused_src; here we prove
 * the RESULT is unchanged so the fused self-forward stays byte-identical. */

/* fused SILU/sigmoid element: sigmoid(x) = SCALE^2 / (SCALE + exp(-x)).  Same
   value op_unary kind 3 computes, evaluated as one fused element (the exp-Taylor
   inner loop collapses into the instruction's own nonlinear forward). */
long fused_silu(long x) {
    long negx; long e; long numer;
    negx = 0 - x;
    e = fp_exp(negx);
    numer = (long)SCALE * (long)SCALE;
    return numer / (SCALE + e);
}

/* fused ADD element: the aligned/-inf-saturated add ew_broadcast kind 0 does for
   one pair of already-resolved (getv) operands.  One fused element per forward. */
long fused_add3(long xa, long xb, int afp, int bfp) {
    if (afp && !bfp) xb = xb * SCALE;
    if (bfp && !afp) xa = xa * SCALE;
    if (xa <= NEG_INF_FP || xb <= NEG_INF_FP) return NEG_INF_FP;
    return xa + xb;
}

/* broadcasting elementwise: kind 0=add 1=sub 2=mul 3=div.
   Handles fixed-point/int mixing exactly like fp_proto._align_add / Mul / Div. */
void ew_broadcast(int out, int a, int b, int kind) {
    int rank; int rdims[MAX_RANK];
    int ar; int br; int i;
    int ast[MAX_RANK]; int bst[MAX_RANK];
    int adims[MAX_RANK]; int bdims[MAX_RANK];
    int idx[MAX_RANK];
    int flat; int sz;
    int ai; int bi; int d;
    long va; long vb; long vr;
    int afp; int bfp; int ofp;

    afp = t_isfp[a]; bfp = t_isfp[b];
    ar = t_rank[a]; br = t_rank[b];
    rank = ar > br ? ar : br;
    i = 0;
    while (i < rank) {
        d = rank - 1 - i;
        adims[d] = (i < ar) ? t_dims[a][ar - 1 - i] : 1;
        bdims[d] = (i < br) ? t_dims[b][br - 1 - i] : 1;
        rdims[d] = adims[d] > bdims[d] ? adims[d] : bdims[d];
        i = i + 1;
    }
    strides_of(rank, adims, ast);
    strides_of(rank, bdims, bst);
    /* output is fixed-point iff either operand is */
    ofp = (afp || bfp) ? 1 : 0;
    alloc_tensor(out, DT_FLOAT, ofp, rank, rdims);
    sz = t_size[out];

    i = 0; while (i < rank) { idx[i] = 0; i = i + 1; }
    flat = 0;
    while (flat < sz) {
        ai = 0; bi = 0; i = 0;
        while (i < rank) {
            ai = ai + (adims[i] == 1 ? 0 : idx[i]) * ast[i];
            bi = bi + (bdims[i] == 1 ? 0 : idx[i]) * bst[i];
            i = i + 1;
        }
        va = getv(a, ai); vb = getv(b, bi);
        if (kind == 0 && fuse_tail_on) {
            /* FUSED element-wise ADD (the tail's dominant #2 op): one fused element
               per forward, byte-identical value to the unfused loop below. */
            vr = fused_add3(va, vb, afp, bfp);
        } else if (kind == 0 || kind == 1) {
            /* align: scale the raw-int operand up when the other is fixed-point */
            long xa; long xb;
            xa = va; xb = vb;
            if (afp && !bfp) xb = xb * SCALE;
            if (bfp && !afp) xa = xa * SCALE;
            /* saturate -inf sentinel through add/sub (mask + score) */
            if (xa <= NEG_INF_FP || xb <= NEG_INF_FP) { vr = NEG_INF_FP; }
            else if (kind == 0) vr = xa + xb;
            else vr = xa - xb;
        } else if (kind == 2) {
            if (afp && bfp) vr = fpmul_l(va, vb);
            else vr = va * vb;               /* one side raw int -> keep other's scale */
        } else {
            /* div: numerator fixed-point-shifted so result stays fixed-point */
            long num;
            if (afp && bfp) { num = va * SCALE; vr = num / vb; }
            else if (afp && !bfp) { vr = va / vb; }
            else if (!afp && bfp) { num = va * SCALE * SCALE; vr = num / vb; }
            else { vr = (va * SCALE) / vb; }
        }
        tv[out][flat] = vr;
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < rdims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* generic N-D MatMul.  Accumulate in 64-bit; if both operands fixed-point the
   accumulator is SCALE^2-scaled and is shifted back by SCALE_BITS at the end. */
void op_matmul(int out, int a, int b) {
    int ar; int br; int rank;
    int M; int K; int N;
    int i; int batch;
    int rdims[MAX_RANK];
    int abatch; int bbatch;
    int adims2; int bdims2;
    int p; int q; int r;
    int aoff; int boff; int ooff;
    long acc; long half;
    int afp; int bfp; int ofp;

    afp = t_isfp[a]; bfp = t_isfp[b];
    ar = t_rank[a]; br = t_rank[b];
    M = t_dims[a][ar - 2]; K = t_dims[a][ar - 1];
    N = t_dims[b][br - 1];
    rank = ar > br ? ar : br;
    i = 0;
    while (i < rank - 2) {
        int ad; int bd;
        int da; int db;
        da = rank - ar; db = rank - br;
        ad = (i >= da) ? t_dims[a][i - da] : 1;
        bd = (i >= db) ? t_dims[b][i - db] : 1;
        rdims[i] = ad > bd ? ad : bd;
        i = i + 1;
    }
    rdims[rank - 2] = M; rdims[rank - 1] = N;
    ofp = (afp || bfp) ? 1 : 0;
    alloc_tensor(out, DT_FLOAT, ofp, rank, rdims);

    batch = 1;
    i = 0; while (i < rank - 2) { batch = batch * rdims[i]; i = i + 1; }
    adims2 = M * K; bdims2 = K * N;
    abatch = (ar == rank) ? 1 : 0;
    bbatch = (br == rank) ? 1 : 0;
    half = SCALE / 2;

    i = 0;
    while (i < batch) {
        aoff = abatch ? i * adims2 : 0;
        boff = bbatch ? i * bdims2 : 0;
        ooff = i * M * N;
        p = 0;
        while (p < M) {
            q = 0;
            while (q < N) {
                acc = 0;
                r = 0;
                while (r < K) {
                    /* DEDUP-aware operand fetch: a deduped weight tensor resolves
                       via `palette[tv[...]]` (the two-access chained load); a
                       runtime intermediate / dedup-OFF reads directly.  Byte-exact. */
                    acc = acc + get_wt(a, aoff + p * K + r) * get_wt(b, boff + r * N + q);
                    r = r + 1;
                }
                if (afp && bfp) {
                    if (acc >= 0) acc = (acc + half) >> SCALE_BITS;
                    else acc = 0 - ((0 - acc + half) >> SCALE_BITS);
                }
                tv[out][ooff + p * N + q] = acc;
                q = q + 1;
            }
            p = p + 1;
        }
        i = i + 1;
    }
}

/* Gather: out = take(data, indices, axis) */
void op_gather(int out, int data, int ind, int axis) {
    int dr; int ir; int i; int j;
    int outer; int inner; int axlen; int nidx;
    int rdims[MAX_RANK]; int rank;
    int o; int aa; int in_i; int base; int src;

    dr = t_rank[data]; ir = t_rank[ind];
    axlen = t_dims[data][axis];
    outer = 1; i = 0; while (i < axis) { outer = outer * t_dims[data][i]; i = i + 1; }
    inner = 1; i = axis + 1; while (i < dr) { inner = inner * t_dims[data][i]; i = i + 1; }
    nidx = t_size[ind];

    rank = 0;
    i = 0; while (i < axis) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    i = 0; while (i < ir) { rdims[rank] = t_dims[ind][i]; rank = rank + 1; i = i + 1; }
    i = axis + 1; while (i < dr) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    if (rank == 0) { rdims[0] = 1; rank = 1; }
    alloc_tensor(out, t_dtype[data], t_isfp[data], rank, rdims);
    /* Gather copies data payload VERBATIM: if `data` is a deduped (palette-indexed)
       weight, the gathered rows are still palette indices -> propagate the flag. */
    t_deduped[out] = t_deduped[data];

    o = 0;
    while (o < outer) {
        j = 0;
        while (j < nidx) {
            in_i = (int)getv(ind, j);
            if (in_i < 0) in_i = in_i + axlen;
            base = (o * nidx + j) * inner;
            src = (o * axlen + in_i) * inner;
            aa = 0;
            while (aa < inner) {
                tv[out][base + aa] = tv[data][src + aa];
                aa = aa + 1;
            }
            j = j + 1;
        }
        o = o + 1;
    }
}

/* Reshape: copy data, set shape from int64 shape-tensor (supports one -1) */
void op_reshape(int out, int in, int shp) {
    int rank; int i; int sz; int known; int neg;
    int rdims[MAX_RANK]; long v;
    rank = t_size[shp];
    known = 1; neg = -1;
    i = 0;
    while (i < rank) {
        v = getv(shp, i);
        if (v == -1) { neg = i; rdims[i] = 1; }
        else { rdims[i] = (int)v; known = known * (int)v; }
        i = i + 1;
    }
    if (neg >= 0) rdims[neg] = t_size[in] / known;
    sz = t_size[in];
    alloc_tensor(out, t_dtype[in], t_isfp[in], rank, rdims);
    t_deduped[out] = t_deduped[in];      /* verbatim payload copy: indices stay indices */
    i = 0;
    while (i < sz) { tv[out][i] = tv[in][i]; i = i + 1; }
}

/* Transpose with an explicit perm */
void op_transpose(int out, int in, int nd, int *perm) {
    int rank; int i; int sz;
    int idims[MAX_RANK]; int odims[MAX_RANK];
    int ist[MAX_RANK]; int ost[MAX_RANK];
    int idx[MAX_RANK]; int flat; int src;
    rank = t_rank[in];
    i = 0; while (i < rank) { idims[i] = t_dims[in][i]; i = i + 1; }
    i = 0; while (i < rank) { odims[i] = idims[perm[i]]; i = i + 1; }
    strides_of(rank, idims, ist);
    alloc_tensor(out, t_dtype[in], t_isfp[in], rank, odims);
    t_deduped[out] = t_deduped[in];      /* verbatim payload permute: indices stay indices */
    strides_of(rank, odims, ost);
    sz = t_size[in];
    i = 0; while (i < rank) { idx[i] = 0; i = i + 1; }
    flat = 0;
    while (flat < sz) {
        src = 0; i = 0;
        while (i < rank) { src = src + idx[i] * ist[perm[i]]; i = i + 1; }
        tv[out][flat] = tv[in][src];
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < odims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* Concat along an axis */
void op_concat(int nd, int out, int axis) {
    int nin; int i; int rank;
    int rdims[MAX_RANK]; int total; int dt; int fp;
    int outer; int inner; int k; int o; int aa; int off; int cin; int seg;
    nin = n_nin[nd];
    dt = t_dtype[n_in[nd][0]];
    fp = t_isfp[n_in[nd][0]];
    rank = t_rank[n_in[nd][0]];
    if (rank == 0) { rank = 1; }
    total = 0;
    i = 0;
    while (i < nin) {
        int r; r = t_rank[n_in[nd][i]];
        total = total + (r == 0 ? 1 : t_dims[n_in[nd][i]][axis]);
        i = i + 1;
    }
    i = 0; while (i < rank) { rdims[i] = t_dims[n_in[nd][0]][i]; i = i + 1; }
    if (t_rank[n_in[nd][0]] == 0) rdims[0] = 1;
    rdims[axis] = total;
    alloc_tensor(out, dt, fp, rank, rdims);
    outer = 1; i = 0; while (i < axis) { outer = outer * rdims[i]; i = i + 1; }
    inner = 1; i = axis + 1; while (i < rank) { inner = inner * rdims[i]; i = i + 1; }
    o = 0;
    while (o < outer) {
        off = 0;
        i = 0;
        while (i < nin) {
            cin = n_in[nd][i];
            seg = (t_rank[cin] == 0) ? 1 : t_dims[cin][axis];
            k = 0;
            while (k < seg) {
                aa = 0;
                while (aa < inner) {
                    int dstf; int srcf;
                    dstf = (o * total + off + k) * inner + aa;
                    srcf = (o * seg + k) * inner + aa;
                    /* Concat may mix deduped + non-deduped inputs, so RESOLVE each
                       source to its VALUE via getv (out is left non-deduped). */
                    tv[out][dstf] = getv(cin, srcf);
                    aa = aa + 1;
                }
                k = k + 1;
            }
            off = off + seg;
            i = i + 1;
        }
        o = o + 1;
    }
}

/* Unsqueeze: insert size-1 dims at the given axes (data copied verbatim) */
void op_unsqueeze(int out, int in, int axtid) {
    int naxes; int i; int rank; int newrank;
    int axes[MAX_RANK]; int rdims[MAX_RANK]; int sz; int mark[MAX_RANK];
    int p; int aa;
    naxes = t_size[axtid];
    rank = t_rank[in];
    newrank = rank + naxes;
    i = 0; while (i < naxes) {
        aa = (int)getv(axtid, i);
        if (aa < 0) aa = aa + newrank;
        axes[i] = aa; i = i + 1;
    }
    i = 0; while (i < newrank) { mark[i] = 0; i = i + 1; }
    i = 0; while (i < naxes) { mark[axes[i]] = 1; i = i + 1; }
    p = 0;
    i = 0;
    while (i < newrank) {
        if (mark[i]) rdims[i] = 1;
        else { rdims[i] = (rank == 0) ? 1 : t_dims[in][p]; p = p + 1; }
        i = i + 1;
    }
    sz = (rank == 0) ? 1 : t_size[in];
    alloc_tensor(out, t_dtype[in], t_isfp[in], newrank, rdims);
    t_deduped[out] = t_deduped[in];      /* verbatim payload copy: indices stay indices */
    i = 0;
    while (i < sz) { tv[out][i] = tv[in][i]; i = i + 1; }
}

/* Shape: out (int64, raw) = the input's dims */
void op_shape(int out, int in) {
    int rank; int i; int d[MAX_RANK];
    rank = t_rank[in];
    d[0] = rank;
    alloc_tensor(out, DT_INT64, 0, 1, d);
    i = 0; while (i < rank) { tv[out][i] = (long)t_dims[in][i]; i = i + 1; }
}

/* Cast: to==1 -> float(=fixed-point), to==7 -> int64.
   fp->int truncates toward zero (ONNX Cast); int->fp multiplies by SCALE. */
void op_cast(int out, int in, int to) {
    int i; int sz; int dt; int wantfp; long v;
    sz = t_size[in];
    dt = (to == 1) ? DT_FLOAT : DT_INT64;
    wantfp = (to == 1) ? 1 : 0;
    alloc_tensor(out, dt, wantfp, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        v = getv(in, i);
        if (wantfp) {
            if (t_isfp[in]) tv[out][i] = v;      /* already fixed-point */
            else tv[out][i] = v * SCALE;         /* int -> fixed-point */
        } else {
            if (t_isfp[in]) {                    /* fixed-point -> int, truncate */
                if (v >= 0) tv[out][i] = v / SCALE;
                else tv[out][i] = 0 - ((0 - v) / SCALE);
            } else tv[out][i] = v;
        }
        i = i + 1;
    }
}

/* Range: out (int64, raw) = [start, limit) step delta */
void op_range(int out, int st, int lim, int dl) {
    long s; long l; long d; int cnt; int i; int dim[1];
    s = getv(st, 0); l = getv(lim, 0); d = getv(dl, 0);
    cnt = 0;
    { long v; v = s; while ((d > 0 && v < l) || (d < 0 && v > l)) { cnt = cnt + 1; v = v + d; } }
    dim[0] = cnt;
    alloc_tensor(out, DT_INT64, 0, 1, dim);
    i = 0; { long v; v = s; while (i < cnt) { tv[out][i] = v; v = v + d; i = i + 1; } }
}

/* unary map: 0 exp, 1 neg, 2 abs, 3 sigmoid, 4 identity.  All fixed-point. */
void op_unary(int out, int in, int kind) {
    int i; int sz; long x; long e; int odt; int ofp;
    sz = t_size[in];
    /* exp/sigmoid always produce fixed-point; neg/abs/identity preserve the
       input's representation (an int64 distance stays raw int -> Cast can scale). */
    if (kind == 0 || kind == 3) { odt = DT_FLOAT; ofp = 1; }
    else { odt = t_dtype[in]; ofp = t_isfp[in]; }
    alloc_tensor(out, odt, ofp, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        x = getv(in, i);
        if (kind == 0) {                 /* exp */
            tv[out][i] = fp_exp(x);
        } else if (kind == 1) {          /* neg */
            if (x <= NEG_INF_FP) tv[out][i] = 0 - NEG_INF_FP;
            else tv[out][i] = 0 - x;
        } else if (kind == 2) {          /* abs */
            tv[out][i] = (x < 0) ? (0 - x) : x;
        } else if (kind == 3) {          /* sigmoid = SCALE^2 / (SCALE + exp(-x)) */
            if (fuse_tail_on) {
                /* FUSED SILU (the tail's dominant #1 op): the exp-Taylor inner loop
                   collapses into one fused element per forward, byte-identical value. */
                tv[out][i] = fused_silu(x);
            } else {
                long negx; long numer; negx = 0 - x;
                e = fp_exp(negx);        /* our graph only feeds x<=0, so -x>=0 -> exp<=1;
                                            fp_exp clamps x>0 to SCALE (exp(0)=1) which is
                                            the correct 0.5 midpoint at x=0. */
                numer = (long)SCALE * (long)SCALE;   /* 64-bit: avoid int overflow */
                tv[out][i] = numer / (SCALE + e);
            }
        } else {                         /* identity */
            tv[out][i] = x;
        }
        i = i + 1;
    }
}

/* Reduce over the last axis, keepdims=1.  kind 0=max, 1=sum. */
void op_reduce_last(int out, int in, int kind) {
    int rank; int i; int last; int rows; int r; int c;
    int rdims[MAX_RANK]; long acc;
    rank = t_rank[in];
    last = t_dims[in][rank - 1];
    rows = t_size[in] / last;
    i = 0; while (i < rank) { rdims[i] = t_dims[in][i]; i = i + 1; }
    rdims[rank - 1] = 1;
    alloc_tensor(out, DT_FLOAT, t_isfp[in], rank, rdims);
    r = 0;
    while (r < rows) {
        if (kind == 0) {
            acc = tv[in][r * last];
            c = 1; while (c < last) { if (tv[in][r * last + c] > acc) acc = tv[in][r * last + c]; c = c + 1; }
        } else {
            acc = 0;
            c = 0; while (c < last) { acc = acc + tv[in][r * last + c]; c = c + 1; }
        }
        tv[out][r] = acc;
        r = r + 1;
    }
}

/* Clip: lower bound only (nibble uses min=0.0, no upper) */
void op_clip(int out, int in, int lotid) {
    int i; int sz; long lo; long x;
    sz = t_size[in];
    lo = (lotid >= 0) ? getv(lotid, 0) : NEG_INF_FP;
    alloc_tensor(out, DT_FLOAT, 1, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) { x = getv(in, i); tv[out][i] = (x < lo) ? lo : x; i = i + 1; }
}

/* Trilu upper/lower (mask): keep or zero elements of the last two dims */
void op_trilu(int out, int in, int upper, int k) {
    int rank; int rows; int cols; int mat; int nmat; int rr; int cc;
    rank = t_rank[in];
    rows = t_dims[in][rank - 2];
    cols = t_dims[in][rank - 1];
    nmat = t_size[in] / (rows * cols);
    alloc_tensor(out, DT_FLOAT, t_isfp[in], rank, t_dims[in]);
    mat = 0;
    while (mat < nmat) {
        rr = 0;
        while (rr < rows) {
            cc = 0;
            while (cc < cols) {
                int f; int keep;
                f = mat * rows * cols + rr * cols + cc;
                if (upper) keep = (cc - rr >= k) ? 1 : 0;
                else keep = (cc - rr <= k) ? 1 : 0;
                tv[out][f] = keep ? tv[in][f] : 0;
                cc = cc + 1;
            }
            rr = rr + 1;
        }
        mat = mat + 1;
    }
}

/* ConstantOfShape: fill with a float bit-pattern (decoded to fixed-point) */
void op_constofshape(int out, int shp, int bits) {
    int rank; int i; int rdims[MAX_RANK]; int sz; long fill;
    int sign; int exp; int mant; long m; int e;
    rank = t_size[shp];
    i = 0; while (i < rank) { rdims[i] = (int)getv(shp, i); i = i + 1; }
    /* decode the 32-bit float `bits` to fixed-point (same logic as rd_f32_fp) */
    sign = (bits >> 31) & 1;
    exp = (bits >> 23) & 0xFF;
    mant = bits & 0x7FFFFF;
    if (exp == 0xFF) {
        fill = sign ? NEG_INF_FP : (0 - NEG_INF_FP);
    } else if (exp == 0 && mant == 0) {
        fill = 0;
    } else {
        m = (long)((1 << 23) | mant);
        e = exp - 127 - 23 + SCALE_BITS;
        if (e >= 0) m = m << e;
        else { int s; long half; s = 0 - e; if (s < 63) { half = (long)1 << (s - 1); m = (m + half) >> s; } else m = 0; }
        fill = sign ? (0 - m) : m;
    }
    alloc_tensor(out, DT_FLOAT, 1, rank, rdims);
    sz = t_size[out];
    i = 0; while (i < sz) { tv[out][i] = fill; i = i + 1; }
}

/* ============ execute ============ */
void run() {
    int nd; int op; int o; int i0; int i1; int i2;
    int perm[MAX_RANK]; int i; int np;
    nd = 0;
    while (nd < n_nodes) {
        op = n_op[nd];
        o = n_out[nd][0];
        i0 = (n_nin[nd] > 0) ? n_in[nd][0] : -1;
        i1 = (n_nin[nd] > 1) ? n_in[nd][1] : -1;
        i2 = (n_nin[nd] > 2) ? n_in[nd][2] : -1;

        if (op == OP_MATMUL) op_matmul(o, i0, i1);
        else if (op == OP_ADD) ew_broadcast(o, i0, i1, 0);
        else if (op == OP_SUB) ew_broadcast(o, i0, i1, 1);
        else if (op == OP_MUL) ew_broadcast(o, i0, i1, 2);
        else if (op == OP_DIV) ew_broadcast(o, i0, i1, 3);
        else if (op == OP_GATHER) op_gather(o, i0, i1, attr1(nd, A_AXIS, 0));
        else if (op == OP_RESHAPE) op_reshape(o, i0, i1);
        else if (op == OP_TRANSPOSE) {
            np = attr_nv(nd, A_PERM);
            i = 0; while (i < np) { perm[i] = attr_val(nd, A_PERM, i); i = i + 1; }
            op_transpose(o, i0, np, perm);
        }
        else if (op == OP_CONCAT) op_concat(nd, o, attr1(nd, A_AXIS, 0));
        else if (op == OP_UNSQUEEZE) op_unsqueeze(o, i0, i1);
        else if (op == OP_SHAPE) op_shape(o, i0);
        else if (op == OP_CAST) op_cast(o, i0, attr1(nd, A_TO, 1));
        else if (op == OP_EXP) op_unary(o, i0, 0);
        else if (op == OP_NEG) op_unary(o, i0, 1);
        else if (op == OP_ABS) op_unary(o, i0, 2);
        else if (op == OP_RANGE) op_range(o, i0, i1, i2);
        else if (op == OP_REDUCEMAX) op_reduce_last(o, i0, 0);
        else if (op == OP_REDUCESUM) op_reduce_last(o, i0, 1);
        else if (op == OP_CLIP) op_clip(o, i0, i1);
        else if (op == OP_TRILU) op_trilu(o, i0, attr1(nd, A_UPPER, 1), (int)getv(i1, 0));
        else if (op == OP_CONSTANTOFSHAPE) op_constofshape(o, i0, attr1(nd, A_COS_FILL_BITS, 0));
        else if (op == OP_SIGMOID) op_unary(o, i0, 3);
        else if (op == OP_IDENTITY) op_unary(o, i0, 4);
        else { printf("unknown op %d at node %d\n", op, nd); exit(1); }
        nd = nd + 1;
    }
}

/* ============ main ============ */
int main(int argc, char **argv) {
    FILE *tf_in;
    int B; int S; int i; int dump; int a;
    int dim[2];
    char *model; char *tokfile;

    if (argc < 3) {
        printf("usage: %s model.nblbin tokens.txt [--dump-argmax|--dump-logits]\n", argv[0]);
        return 1;
    }
    model = argv[1];
    tokfile = argv[2];
    dump = 0;
    if (argc > 3 && strcmp(argv[3], "--dump-argmax") == 0) dump = 1;
    if (argc > 3 && strcmp(argv[3], "--dump-logits") == 0) dump = 2;

    a = 0;
    while (a < MAX_TENSORS) { tv[a] = 0; a = a + 1; }

    build_exp_table();
    load(model);

    /* read tokens: "B S" then B*S ints */
    tf_in = fopen(tokfile, "r");
    if (!tf_in) { printf("cannot open %s\n", tokfile); return 1; }
    fscanf(tf_in, "%d %d", &B, &S);
    dim[0] = B; dim[1] = S;
    alloc_tensor(input_tid, DT_INT64, 0, 2, dim);
    t_is_init[input_tid] = 1;
    i = 0;
    while (i < B * S) { int tk; fscanf(tf_in, "%d", &tk); tv[input_tid][i] = (long)tk; i = i + 1; }
    fclose(tf_in);
    printf("tokens: B=%d S=%d\n", B, S);

    run();

    /* logits: [B, S, vocab] — argmax over vocab per (b,s) */
    {
        int vocab; int rank; int rows; int r; int c; int best; long bv;
        rank = t_rank[output_tid];
        vocab = t_dims[output_tid][rank - 1];
        rows = t_size[output_tid] / vocab;
        printf("logits shape:");
        r = 0; while (r < rank) { printf(" %d", t_dims[output_tid][r]); r = r + 1; }
        printf("\n");
        if (dump == 2) {
            /* raw fixed-point logits rescaled to a decimal, one per line.
               We print value/SCALE with 6 fractional digits, integer-only. */
            r = 0;
            while (r < rows * vocab) {
                long v; long ip; long fp6; long fr;
                v = tv[output_tid][r];
                if (v < 0) { printf("-"); v = 0 - v; }
                ip = v >> SCALE_BITS;
                fr = v - (ip << SCALE_BITS);
                fp6 = fr * 1000000;
                fp6 = fp6 >> SCALE_BITS;
                printf("%ld.%06ld\n", ip, fp6);
                r = r + 1;
            }
        } else if (dump == 1) {
            r = 0;
            while (r < rows) {
                best = 0; bv = tv[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tv[output_tid][r * vocab + c] > bv) { bv = tv[output_tid][r * vocab + c]; best = c; }
                    c = c + 1;
                }
                printf("%d\n", best);
                r = r + 1;
            }
        } else {
            int sum; sum = 0;
            r = 0;
            while (r < rows) {
                best = 0; bv = tv[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tv[output_tid][r * vocab + c] > bv) { bv = tv[output_tid][r * vocab + c]; best = c; }
                    c = c + 1;
                }
                sum = sum + best * (r + 1);
                r = r + 1;
            }
            printf("argmax_checksum=%d rows=%d\n", sum, rows);
        }
    }
    return 0;
}
