/*
 * onnx_runtime_nibble_incremental.c  —  SELF-CONTAINED sparse (COO) runtime with
 * a WINDOWED KV-CACHED INCREMENTAL block-stack forward (kills the O(S^2)).
 *
 * This is the self-contained (`.incbin`-embedded model) sparse runtime PLUS an
 * incremental decode path (`--incremental`, serve mode) that makes the per-step
 * wall FLAT in the stream length S instead of growing O(S^2).
 *
 * THE O(S^2) PROBLEM the base sparse runtime has: the corpus driver runs one VM
 * step as `x = embed[stream]; overlay(x); hidden = blocks(x)` and reads only the
 * LAST row `hidden[0,-1]`.  Each step APPENDS a 30-token register frame to the
 * stream, so S grows by 30/step; the base runtime re-runs the WHOLE block stack
 * over ALL S rows every step (attention is O(S^2), FFN O(S)).  ~1.7s@S31 growing
 * to ~45s@S211 — quine (S=1651) / mandelbrot intractable.
 *
 * THE FIX — windowed KV-cached incremental forward (BYTE-EXACT).  The stream is
 * append-only: verified (torch, max|delta|=0 over ALL 242 blocks) that between
 * step N (S_prev rows) and step N+1, rows [0 .. S_prev-2] have BYTE-IDENTICAL
 * block outputs at EVERY block — only the previous query row S_prev-1 changes
 * (the overlay clears its all-ROLE one-hot when it stops being last) and the 30
 * new rows are added.  Blocks are causal, so a stable prefix row's K/V is fixed
 * forever.  Each step therefore recomputes only the TAIL rows [S_prev-1 .. S-1]
 * (31 rows: 1 changed + 30 new), attending over a CACHED prefix K/V (rows
 * 0..S_prev-2) concatenated with the tail's own K/V.  Per-step: attention is
 * O(31 * S) (LINEAR, not S^2); Q/K/V/O projections + FFN are O(31) (CONSTANT in
 * S).  After the step, tail rows [S_prev-1 .. S-2] become the new stable prefix
 * (K/V appended to the cache); the new last row S-1 is recomputed next step.
 * Byte-exact vs the full-recompute at row -1 (max|delta| ~1e-19, decode argmax
 * identical) because the same COO ascending-r matmul + same softmax1 fp order.
 *
 * The incremental forward (incr_forward) is a HAND-WRITTEN port of
 * blogspec_model.Attn.forward + FFN.forward (softmax1 + ALiBi + SwiGLU) that
 * reads the per-block Q/K/V/O + FFN up/gate/down COO weights and the alibi-slope
 * / scale / bias constants straight out of the loaded graph (the block stack is a
 * perfectly regular 242 x 69-node pattern, so the weight tensor ids are DERIVED
 * from the node structure at load, not hardcoded).
 *
 * SIMD: the hot dense loops (COO scatter accumulation, softmax1 exp/reduce,
 * attn@V, FFN gate/up/down) are written so `gcc -O3 -march=native` auto-
 * vectorises them; the fp ACCUMULATION ORDER is preserved (no -ffast-math
 * reassociation) so byte-exactness holds.
 *
 * MULTICORE: the 242-block loop is sequential (data dependency), but WITHIN a
 * block the 23 attention heads and the projection/FFN row loops are parallelised
 * over cores.  Two mechanisms, either static-linkable to keep ONE self-contained
 * binary:
 *   - OpenMP (`-fopenmp`): `#pragma omp parallel for` on the head/row loops.  Needs
 *     static libgomp (or a libgomp .so dep); falls back to serial without -fopenmp.
 *   - pthreads (`-DUSE_PTHREADS -pthread`): a tiny fork-join parallel-for over the
 *     23 heads (glibc static pthreads works where static libgomp is absent).  This
 *     is the SELF-CONTAINED multicore path on toolchains without static libgomp.
 * Byte-exactness holds either way: each head writes a disjoint output slice, so the
 * per-head fp reduction order is unchanged by the parallel split.
 *
 * ---- (inherited from the sparse runtime) ----
 * THE DIFFERENCE from the dense runtime (onnx_runtime_nibble.c lines ~202-214):
 * the dense runtime RECONSTRUCTS every COO-sparse weight initializer into a dense
 * buffer and does an O(M*K*N) dense MatMul — the WRONG compute for a ~99.99%
 * sparse model.  Here a COO-sparse RHS weight is KEPT sparse (parallel val/row/col
 * triples, in stored flat-index order) and op_matmul_coo iterates ONLY the nnz
 * nonzeros: for each activation row it scatters nnz terms, so the inner-loop count
 * is O(M*nnz), NOT O(M*K*N).  Non-matmul consumers of a sparse tensor (none in the
 * block-stack graph, but kept general) still read a zero-filled DENSE payload, so
 * the whole 23-op forward is intact and every other op is bit-identical to dense.
 *
 * BYTE-EXACT vs the dense runtime.  The load-bearing subtlety at VM residual
 * magnitudes ~1e29 is fp ACCUMULATION ORDER: adding the same terms in a different
 * order flips low bits and can flip a decode argmax.  The dense MatMul accumulates
 * out[p][q] = sum_r A[p][r]*W[r][q] over r = 0..K-1 IN ORDER.  The COO nonzeros of
 * a row-major weight W[K,N] are stored sorted by flat index f = r*N + q (ascending)
 * — i.e. r-major, q-minor.  A single scatter pass over the COO array in stored
 * order therefore delivers, for each fixed output column q, its contributions in
 * ASCENDING-r order — EXACTLY the dense reduction order.  Skipped zero terms add
 * +0.0 in dense (no-op in fp), so iterating only the nonzeros in stored order is
 * bit-identical to the dense O(K) reduction.  (op_matmul_coo asserts this ordering
 * by scattering in stored COO order, never re-sorting per column.)
 *
 * float parity: native `float`, argmax(byte-head)-identical to torch/ORT — the VM
 * decodes its state by argmax over the byte head.  Same convention as the dense
 * runtime.
 *
 * Build:   gcc -O2 -static -lm -o rt_sparse onnx_runtime_nibble_sparse.c
 * Run:     ./rt_sparse model.nblbin input [--dump-argmax|--dump-logits]
 *                                          [--residual-in [--serve]]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_PTHREADS
#include <pthread.h>
#include <unistd.h>
#endif

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

/* Sized for the ONE UNIFIED full-op VM graph (dim~1633, 305 blocks): the full op
 * set (incl. the ~300 DIV/MOD blocks) yields ~33k nodes / ~34k distinct tensors.
 * The old lean/compact graph (dim=1564, 42 blocks: 4461 tensors / 3186 nodes) and
 * the tiny proof model (dim=104, 2 blocks) both fit far inside — one runtime for
 * all.  These tables are BSS globals (~20 MB), never on the stack. */
#define MAX_TENSORS 65536
#define MAX_NODES 65536
#define MAX_IO 8
#define MAX_RANK 8
#define MAX_ATTRS 4

/* A COO-sparse weight's zero-filled DENSE payload is reconstructed ONLY when its
 * numel is <= this (biases a non-matmul op reads).  Larger weights are MatMul-RHS
 * only (directly or via Identity), so their dense buffer is never read — skip it. */
#define COO_DENSE_MAX 1048576

/* ---- tensor table (Structure-of-Arrays, C4-subset friendly) ---- */
int   t_dtype[MAX_TENSORS];       /* 0 float, 1 int64 */
int   t_rank[MAX_TENSORS];
int   t_dims[MAX_TENSORS][MAX_RANK];
int   t_size[MAX_TENSORS];        /* element count */
float *tf[MAX_TENSORS];           /* float payload (dtype 0) */
long  *ti[MAX_TENSORS];           /* int64 payload (dtype 1) */
int   t_is_init[MAX_TENSORS];
char  t_name[MAX_TENSORS][80];
int   n_tensors;
int   input_tid;
int   output_tid;

/* ---- SPARSE (COO) payload — the load-bearing difference from the dense runtime.
 * A 2-D weight that arrives COO-sparse (is_init==2) is kept SPARSE: its nonzeros
 * are parallel (val, row, col) triples in STORED flat-index order (row-major:
 * ascending r, then ascending q within a row) and t_is_coo[tid]=1.  op_matmul
 * routes such a RHS to the O(M*nnz) sparse path; every other consumer reads the
 * zero-filled dense payload tf[tid] (also reconstructed at load).  Stored order
 * is preserved so the per-column reduction order matches dense (byte-exact). */
int   t_is_coo[MAX_TENSORS];      /* 1 => weight stored as COO (sparse matmul RHS) */
int   t_nnz[MAX_TENSORS];         /* number of nonzeros */
int  *t_coo_row[MAX_TENSORS];     /* nnz row indices (dim -2), stored order */
int  *t_coo_col[MAX_TENSORS];     /* nnz col indices (dim -1), stored order */
float *t_coo_val[MAX_TENSORS];    /* nnz float values, stored order */

/* diagnostic counters: the ACTUAL matmul inner-iteration counts executed this run,
 * so a measurement can print the realised sparse work (SPARSE_ITERS) vs the dense
 * work it replaced (DENSE_EQUIV_ITERS = M*K*N over the sparse-RHS matmuls). */
long  SPARSE_ITERS;
long  DENSE_EQUIV_ITERS;

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

/* ============ binary reader — MEMORY cursor over the EMBEDDED model blob ========
 * This self-contained build has the .nblbin bytes linked in via `.incbin`
 * (see the _model_blob asm at the bottom); the loader reads from that byte cursor,
 * so the binary needs NO external model file. */
extern const unsigned char _model_blob[];
extern const unsigned char _model_blob_end[];
const unsigned char *g_blob;
long g_pos;

int rd_i32(void) {
    const unsigned char *b;
    b = g_blob + g_pos;
    g_pos += 4;
    return (int)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}
/* read a stored int64 (two int32, little-endian lo/hi) as a C long */
long rd_i64(void) {
    unsigned int lo;
    int hi;
    lo = (unsigned int)rd_i32();
    hi = rd_i32();
    return ((long)hi << 32) | (long)lo;
}
float rd_f32(void) {
    int bits;
    float x;
    bits = rd_i32();
    memcpy(&x, &bits, 4);
    return x;
}
/* copy `n` raw bytes from the cursor into dst, advancing the cursor. */
void rd_bytes(void *dst, int n) {
    memcpy(dst, g_blob + g_pos, n);
    g_pos += n;
}

/* ============ tensor helpers ============ */
int prod(int rank, int *dims) {
    int p; int i;
    p = 1; i = 0;
    while (i < rank) { p = p * dims[i]; i = i + 1; }
    return p;
}

/* allocate a runtime tensor's storage for the given shape + dtype.  A dense
 * (re)allocation clears any COO-alias status the tensor id previously held. */
void alloc_tensor(int tid, int dtype, int rank, int *dims) {
    int sz; int i;
    t_is_coo[tid] = 0;                 /* dense alloc -> not a COO alias */
    t_dtype[tid] = dtype;
    t_rank[tid] = rank;
    i = 0;
    while (i < rank) { t_dims[tid][i] = dims[i]; i = i + 1; }
    sz = prod(rank, dims);
    t_size[tid] = sz;
    if (tf[tid]) { free(tf[tid]); tf[tid] = 0; }
    if (ti[tid]) { free(ti[tid]); ti[tid] = 0; }
    if (dtype == DT_FLOAT) tf[tid] = malloc(sz * sizeof(float));
    else ti[tid] = malloc(sz * sizeof(long));
}

/* strides for row-major layout */
void strides_of(int rank, int *dims, int *st) {
    int i; int s;
    s = 1; i = rank - 1;
    while (i >= 0) { st[i] = s; s = s * dims[i]; i = i - 1; }
}

/* ============ load the EMBEDDED .nblbin (memory cursor, no file) ============ */
void load(char *path) {
    int magic; int i; int j; int k;
    int nl; int isi; int dt; int rank; int dims[MAX_RANK]; int ne;
    int op; int nin; int nout; int nattr; int key; int nv;

    (void)path;                       /* embedded blob — path ignored */
    g_blob = _model_blob;
    g_pos = 0;
    magic = rd_i32();
    if (magic != 0x314C424E) { printf("bad magic %x\n", magic); exit(1); }
    n_tensors = rd_i32();
    n_nodes = rd_i32();
    input_tid = rd_i32();
    output_tid = rd_i32();

    i = 0;
    while (i < n_tensors) {
        nl = rd_i32();
        rd_bytes(t_name[i], nl);
        t_name[i][nl] = 0;
        isi = rd_i32();
        t_is_init[i] = isi;
        tf[i] = 0; ti[i] = 0;
        t_is_coo[i] = 0; t_nnz[i] = 0;
        t_coo_row[i] = 0; t_coo_col[i] = 0; t_coo_val[i] = 0;
        dt = rd_i32();
        rank = rd_i32();
        j = 0;
        while (j < rank) { dims[j] = rd_i32(); j = j + 1; }
        ne = rd_i32();
        if (isi == 1) {
            /* dense initializer: ne stored values */
            alloc_tensor(i, dt, rank, dims);
            if (dt == DT_FLOAT) {
                j = 0; while (j < ne) { tf[i][j] = rd_f32(); j = j + 1; }
            } else {
                j = 0; while (j < ne) { ti[i][j] = rd_i64(); j = j + 1; }
            }
        } else if (isi == 2) {
            /* COO-sparse float initializer: ne = DENSE numel, then nnz, then
             * nnz int64 flat-indices (ASCENDING = row-major stored order) + nnz
             * float values.  KEEP IT SPARSE: store parallel (row,col,val) triples
             * in the SAME stored order (so op_matmul_coo's per-column reduction
             * order == dense; op_identity propagates the COO alias to MatMul RHS).
             *
             * The dense payload tf[i] is reconstructed ONLY for SMALL tensors
             * (numel <= COO_DENSE_MAX): those are biases (e.g. blocks.34.ffn.b_gate,
             * [4320]) that a non-matmul op (Add) consumes and therefore need a dense
             * read.  Large weight matrices are consumed ONLY as MatMul RHS (directly
             * or through Identity), so their dense payload is never read — skipping
             * it saves the ~12 GB / ~3 B-float dense reconstruction and its zero-fill
             * time.  If a large sparse tensor WERE read densely, tf[i]==0 would be a
             * loud NULL-deref, not a silent wrong answer. */
            int nnz; long *s_idx; int lastdim; long fidx; int want_dense;
            t_dtype[i] = DT_FLOAT; t_rank[i] = rank;
            { int dd; dd = 0; while (dd < rank) { t_dims[i][dd] = dims[dd]; dd = dd + 1; } }
            t_size[i] = ne;
            want_dense = (ne <= COO_DENSE_MAX) ? 1 : 0;
            if (want_dense) {
                tf[i] = malloc(ne * sizeof(float));
                j = 0; while (j < ne) { tf[i][j] = 0.0f; j = j + 1; }
            }
            lastdim = dims[rank - 1];             /* N (col stride) */
            nnz = rd_i32();
            s_idx = malloc(nnz * sizeof(long));
            j = 0;
            while (j < nnz) { s_idx[j] = rd_i64(); j = j + 1; }
            t_is_coo[i] = 1;
            t_nnz[i] = nnz;
            t_coo_row[i] = malloc(nnz * sizeof(int));
            t_coo_col[i] = malloc(nnz * sizeof(int));
            t_coo_val[i] = malloc(nnz * sizeof(float));
            j = 0;
            while (j < nnz) {
                fidx = s_idx[j];
                t_coo_row[i][j] = (int)(fidx / lastdim);
                t_coo_col[i][j] = (int)(fidx - (fidx / lastdim) * lastdim);
                j = j + 1;
            }
            j = 0;
            while (j < nnz) {
                float v; v = rd_f32();
                t_coo_val[i][j] = v;
                if (want_dense) tf[i][s_idx[j]] = v;   /* scatter into dense payload */
                j = j + 1;
            }
            free(s_idx);
        } else {
            t_dtype[i] = -1; t_rank[i] = 0; t_size[i] = 0;
        }
        i = i + 1;
    }

    i = 0;
    while (i < n_nodes) {
        op = rd_i32();
        n_op[i] = op;
        nin = rd_i32();
        n_nin[i] = nin;
        j = 0; while (j < nin) { n_in[i][j] = rd_i32(); j = j + 1; }
        nout = rd_i32();
        n_nout[i] = nout;
        j = 0; while (j < nout) { n_out[i][j] = rd_i32(); j = j + 1; }
        nattr = rd_i32();
        n_nattr[i] = nattr;
        j = 0;
        while (j < nattr) {
            key = rd_i32();
            nv = rd_i32();
            n_akey[i][j] = key;
            n_anv[i][j] = nv;
            k = 0; while (k < nv) { n_aval[i][j][k] = rd_i32(); k = k + 1; }
            j = j + 1;
        }
        i = i + 1;
    }
    /* diagnostics on stderr so --residual-in / --dump-logits keep a clean stdout */
    fprintf(stderr, "loaded EMBEDDED model: %ld bytes, %d tensors, %d nodes "
            "(in=%d out=%d)\n", (long)(_model_blob_end - _model_blob),
            n_tensors, n_nodes, input_tid, output_tid);
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

/* read tensor element t[flat] as float regardless of dtype */
float getf(int tid, int flat) {
    if (t_dtype[tid] == DT_FLOAT) return tf[tid][flat];
    return (float)ti[tid][flat];
}
long geti(int tid, int flat) {
    if (t_dtype[tid] == DT_INT64) return ti[tid][flat];
    return (long)tf[tid][flat];
}

/* broadcasting elementwise: op 0=add 1=sub 2=mul 3=div. writes out (float). */
void ew_broadcast(int out, int a, int b, int kind) {
    int rank; int rdims[MAX_RANK];
    int ar; int br; int i;
    int ast[MAX_RANK]; int bst[MAX_RANK];
    int adims[MAX_RANK]; int bdims[MAX_RANK];
    int idx[MAX_RANK];
    int flat; int sz;
    int ai; int bi; int d;
    float va; float vb; float vr;

    ar = t_rank[a]; br = t_rank[b];
    rank = ar > br ? ar : br;
    /* right-align dims (pad with 1 on the left) */
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
    alloc_tensor(out, DT_FLOAT, rank, rdims);
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
        va = getf(a, ai); vb = getf(b, bi);
        if (kind == 0) vr = va + vb;
        else if (kind == 1) vr = va - vb;
        else if (kind == 2) vr = va * vb;
        else vr = va / vb;
        tf[out][flat] = vr;
        /* increment odometer */
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < rdims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* ===== SPARSE (COO) MatMul: out[...,M,N] = A[...,M,K] @ Wcoo[K,N] ==============
 * The RHS weight W[K,N] is stored COO (val,row,col triples in row-major stored
 * order: ascending flat = r*N+q).  A is a DENSE activation of rank ar, with M =
 * A.dims[-2], K = A.dims[-1]; its leading dims are the batch.  W is 2-D and
 * broadcast across every batch slice.  Output rank = ar, dims = A.dims[:-1] + [N].
 *
 * For each (batch, output-row p) we zero the N-wide output row, then scatter EVERY
 * nonzero in STORED order:  out[p][col] += A[p][row]*val.  Because the COO array
 * is row-major (all q of r=0, then all q of r=1, ...), each out[p][q] receives its
 * contributions in ascending-r order — identical to the dense sum_r loop, so the
 * fp accumulation is BIT-EXACT to op_matmul on the reconstructed-dense weight.
 * O(batch*M*nnz) inner iterations instead of O(batch*M*K*N). */
void op_matmul_coo(int out, int a, int b) {
    int ar; int rank; int M; int K; int N; int nnz;
    int i; int batch; int rdims[MAX_RANK];
    int *row; int *col; float *val;
    int bt; int p; int j; int aoff; int ooff; int base;
    float av;

    ar = t_rank[a];
    M = t_dims[a][ar - 2]; K = t_dims[a][ar - 1];
    N = t_dims[b][t_rank[b] - 1];
    rank = ar;                                   /* W is 2-D, broadcast; out rank=ar */
    i = 0; while (i < rank - 2) { rdims[i] = t_dims[a][i]; i = i + 1; }
    rdims[rank - 2] = M; rdims[rank - 1] = N;
    alloc_tensor(out, DT_FLOAT, rank, rdims);

    batch = 1;
    i = 0; while (i < rank - 2) { batch = batch * rdims[i]; i = i + 1; }

    nnz = t_nnz[b];
    row = t_coo_row[b]; col = t_coo_col[b]; val = t_coo_val[b];

    /* zero the whole output first (dense sets each out[p][q] once; here we scatter
     * additively, so pre-zeroing gives the identical starting value 0.0f). */
    i = 0; while (i < batch * M * N) { tf[out][i] = 0.0f; i = i + 1; }

    bt = 0;
    while (bt < batch) {
        aoff = bt * M * K;
        ooff = bt * M * N;
        p = 0;
        while (p < M) {
            base = ooff + p * N;
            j = 0;
            while (j < nnz) {
                av = tf[a][aoff + p * K + row[j]];
                tf[out][base + col[j]] = tf[out][base + col[j]] + av * val[j];
                j = j + 1;
            }
            p = p + 1;
        }
        bt = bt + 1;
    }
    SPARSE_ITERS = SPARSE_ITERS + (long)batch * (long)M * (long)nnz;
    DENSE_EQUIV_ITERS = DENSE_EQUIV_ITERS + (long)batch * (long)M * (long)K * (long)N;
}

/* generic N-D MatMul: batched over leading dims, contract last of A with 2nd-last of B.
 * If the RHS is a COO-sparse weight, dispatch to the O(nnz) sparse path (byte-exact). */
void op_matmul(int out, int a, int b) {
    if (t_is_coo[b]) { op_matmul_coo(out, a, b); return; }
    int ar; int br; int rank;
    int M; int K; int N;
    int i; int batch;
    int rdims[MAX_RANK];
    int abatch; int bbatch;
    int adims2; int bdims2;
    int p; int q; int r;
    int aoff; int boff; int ooff;
    float acc;

    ar = t_rank[a]; br = t_rank[b];
    M = t_dims[a][ar - 2]; K = t_dims[a][ar - 1];
    N = t_dims[b][br - 1];
    /* output rank = max(ar,br); leading batch dims come from the higher-rank
       operand (nibble graph only ever broadcasts a 2-D weight against 4-D acts
       or matches batch dims exactly). */
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
    alloc_tensor(out, DT_FLOAT, rank, rdims);

    batch = 1;
    i = 0; while (i < rank - 2) { batch = batch * rdims[i]; i = i + 1; }
    adims2 = M * K; bdims2 = K * N;
    /* does A / B carry the batch dim? (broadcast a 2-D operand across batch) */
    abatch = (ar == rank) ? 1 : 0;
    bbatch = (br == rank) ? 1 : 0;

    i = 0;
    while (i < batch) {
        aoff = abatch ? i * adims2 : 0;
        boff = bbatch ? i * bdims2 : 0;
        ooff = i * M * N;
        p = 0;
        while (p < M) {
            q = 0;
            while (q < N) {
                acc = 0.0f;
                r = 0;
                while (r < K) {
                    acc = acc + tf[a][aoff + p * K + r] * tf[b][boff + r * N + q];
                    r = r + 1;
                }
                tf[out][ooff + p * N + q] = acc;
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
    int o; int a; int in_i; int base; int src;

    dr = t_rank[data]; ir = t_rank[ind];
    axlen = t_dims[data][axis];
    outer = 1; i = 0; while (i < axis) { outer = outer * t_dims[data][i]; i = i + 1; }
    inner = 1; i = axis + 1; while (i < dr) { inner = inner * t_dims[data][i]; i = i + 1; }
    nidx = t_size[ind];

    /* output shape = data[:axis] + ind.shape + data[axis+1:] */
    rank = 0;
    i = 0; while (i < axis) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    i = 0; while (i < ir) { rdims[rank] = t_dims[ind][i]; rank = rank + 1; i = i + 1; }
    i = axis + 1; while (i < dr) { rdims[rank] = t_dims[data][i]; rank = rank + 1; i = i + 1; }
    if (rank == 0) { rdims[0] = 1; rank = 1; }
    alloc_tensor(out, t_dtype[data], rank, rdims);

    o = 0;
    while (o < outer) {
        j = 0;
        while (j < nidx) {
            in_i = (int)geti(ind, j);
            if (in_i < 0) in_i = in_i + axlen;
            base = (o * nidx + j) * inner;
            src = (o * axlen + in_i) * inner;
            a = 0;
            while (a < inner) {
                if (t_dtype[data] == DT_FLOAT) tf[out][base + a] = tf[data][src + a];
                else ti[out][base + a] = ti[data][src + a];
                a = a + 1;
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
        v = geti(shp, i);
        if (v == -1) { neg = i; rdims[i] = 1; }
        else { rdims[i] = (int)v; known = known * (int)v; }
        i = i + 1;
    }
    if (neg >= 0) rdims[neg] = t_size[in] / known;
    sz = t_size[in];
    alloc_tensor(out, t_dtype[in], rank, rdims);
    i = 0;
    while (i < sz) {
        if (t_dtype[in] == DT_FLOAT) tf[out][i] = tf[in][i];
        else ti[out][i] = ti[in][i];
        i = i + 1;
    }
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
    alloc_tensor(out, t_dtype[in], rank, odims);
    strides_of(rank, odims, ost);
    sz = t_size[in];
    i = 0; while (i < rank) { idx[i] = 0; i = i + 1; }
    flat = 0;
    while (flat < sz) {
        src = 0; i = 0;
        while (i < rank) { src = src + idx[i] * ist[perm[i]]; i = i + 1; }
        if (t_dtype[in] == DT_FLOAT) tf[out][flat] = tf[in][src];
        else ti[out][flat] = ti[in][src];
        i = rank - 1;
        while (i >= 0) {
            idx[i] = idx[i] + 1;
            if (idx[i] < odims[i]) { i = -1; }
            else { idx[i] = 0; i = i - 1; }
        }
        flat = flat + 1;
    }
}

/* Concat along an axis (all int64 here — shape plumbing) */
void op_concat(int nd, int out, int axis) {
    int nin; int i; int j; int rank;
    int rdims[MAX_RANK]; int total; int dt;
    int outer; int inner; int k; int o; int a; int off; int cin; int seg;
    nin = n_nin[nd];
    dt = t_dtype[n_in[nd][0]];
    rank = t_rank[n_in[nd][0]];
    if (rank == 0) { rank = 1; }
    /* sum the axis dim */
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
    alloc_tensor(out, dt, rank, rdims);
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
                a = 0;
                while (a < inner) {
                    int dstf; int srcf;
                    dstf = (o * total + off + k) * inner + a;
                    srcf = (o * seg + k) * inner + a;
                    if (dt == DT_FLOAT) tf[out][dstf] = tf[cin][srcf];
                    else ti[out][dstf] = ti[cin][srcf];
                    a = a + 1;
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
    int p; int a;
    naxes = t_size[axtid];
    rank = t_rank[in];
    newrank = rank + naxes;
    i = 0; while (i < naxes) {
        a = (int)geti(axtid, i);
        if (a < 0) a = a + newrank;
        axes[i] = a; i = i + 1;
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
    alloc_tensor(out, t_dtype[in], newrank, rdims);
    i = 0;
    while (i < sz) {
        if (t_dtype[in] == DT_FLOAT) tf[out][i] = (rank == 0) ? getf(in, 0) : tf[in][i];
        else ti[out][i] = (rank == 0) ? geti(in, 0) : ti[in][i];
        i = i + 1;
    }
}

void op_unary(int out, int in, int kind);   /* fwd decl (defined below) */

/* Identity that PRESERVES COO sparsity.  ONNX often routes a weight through an
 * Identity node before the MatMul (onnx::MatMul_* -> Identity -> MatMul).  If the
 * input is a COO-sparse weight, make the output a COO ALIAS (share the same
 * row/col/val arrays + dims + nnz, is_coo=1, no dense buffer) so the downstream
 * MatMul reads it via the O(nnz) sparse path instead of a reconstructed-dense
 * O(K*N) copy.  Aliasing is safe: COO arrays are never freed mid-run, and these
 * Identity outputs are consumed only as MatMul RHS.  For a non-COO input this
 * falls back to a plain dense copy (op_unary identity). */
void op_identity(int out, int in) {
    if (t_is_coo[in]) {
        int i;
        t_dtype[out] = DT_FLOAT;
        t_rank[out] = t_rank[in];
        i = 0; while (i < t_rank[in]) { t_dims[out][i] = t_dims[in][i]; i = i + 1; }
        t_size[out] = t_size[in];
        if (tf[out]) { free(tf[out]); tf[out] = 0; }
        if (ti[out]) { free(ti[out]); ti[out] = 0; }
        t_is_coo[out] = 1;
        t_nnz[out] = t_nnz[in];
        t_coo_row[out] = t_coo_row[in];      /* share (never freed) */
        t_coo_col[out] = t_coo_col[in];
        t_coo_val[out] = t_coo_val[in];
        return;
    }
    op_unary(out, in, 4);
}

/* Shape: out (int64) = the input's dims */
void op_shape(int out, int in) {
    int rank; int i; int d[MAX_RANK];
    rank = t_rank[in];
    d[0] = rank;
    alloc_tensor(out, DT_INT64, 1, d);
    i = 0; while (i < rank) { ti[out][i] = (long)t_dims[in][i]; i = i + 1; }
}

/* Cast: to==1 -> float32, to==7 -> int64 (onnx TensorProto codes) */
void op_cast(int out, int in, int to) {
    int i; int sz; int dt;
    sz = t_size[in];
    dt = (to == 1) ? DT_FLOAT : DT_INT64;
    alloc_tensor(out, dt, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        if (dt == DT_FLOAT) tf[out][i] = getf(in, i);
        else ti[out][i] = geti(in, i);
        i = i + 1;
    }
}

/* Range: out (int64) = [start, limit) step delta.  scalars are int64 tensors. */
void op_range(int out, int st, int lim, int dl) {
    long s; long l; long d; int cnt; int i; int dim[1];
    s = geti(st, 0); l = geti(lim, 0); d = geti(dl, 0);
    cnt = 0;
    { long v; v = s; while ((d > 0 && v < l) || (d < 0 && v > l)) { cnt = cnt + 1; v = v + d; } }
    dim[0] = cnt;
    alloc_tensor(out, DT_INT64, 1, dim);
    i = 0; { long v; v = s; while (i < cnt) { ti[out][i] = v; v = v + d; i = i + 1; } }
}

/* unary float map: 0 exp, 1 neg, 2 abs, 3 sigmoid, 4 identity */
void op_unary(int out, int in, int kind) {
    int i; int sz; float x;
    sz = t_size[in];
    alloc_tensor(out, DT_FLOAT, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) {
        x = getf(in, i);
        if (kind == 0) tf[out][i] = expf(x);
        else if (kind == 1) tf[out][i] = -x;
        else if (kind == 2) tf[out][i] = fabsf(x);
        else if (kind == 3) tf[out][i] = 1.0f / (1.0f + expf(-x));
        else tf[out][i] = x;
        i = i + 1;
    }
}

/* Reduce over the last axis (nibble graph reduces last-axis only), keepdims=1.
   kind 0=max, 1=sum. */
void op_reduce_last(int out, int in, int kind) {
    int rank; int i; int last; int rows; int r; int c;
    int rdims[MAX_RANK]; float acc;
    rank = t_rank[in];
    last = t_dims[in][rank - 1];
    rows = t_size[in] / last;
    i = 0; while (i < rank) { rdims[i] = t_dims[in][i]; i = i + 1; }
    rdims[rank - 1] = 1;
    alloc_tensor(out, DT_FLOAT, rank, rdims);
    r = 0;
    while (r < rows) {
        if (kind == 0) {
            acc = tf[in][r * last];
            c = 1; while (c < last) { if (tf[in][r * last + c] > acc) acc = tf[in][r * last + c]; c = c + 1; }
        } else {
            acc = 0.0f;
            c = 0; while (c < last) { acc = acc + tf[in][r * last + c]; c = c + 1; }
        }
        tf[out][r] = acc;
        r = r + 1;
    }
}

/* Clip: lower bound only (nibble uses min=0.0, no upper).  in2 optional. */
void op_clip(int out, int in, int lotid) {
    int i; int sz; float lo; float x;
    sz = t_size[in];
    lo = (lotid >= 0) ? getf(lotid, 0) : -1e30f;
    alloc_tensor(out, DT_FLOAT, t_rank[in], t_dims[in]);
    i = 0;
    while (i < sz) { x = getf(in, i); tf[out][i] = (x < lo) ? lo : x; i = i + 1; }
}

/* Trilu upper: keep elements where col-row >= k, else 0.  Operates on the last
   two dims of an N-D tensor (here rank 2, the [S,S] mask). */
void op_trilu(int out, int in, int upper, int k) {
    int rank; int rows; int cols; int i; int mat; int nmat; int rr; int cc;
    rank = t_rank[in];
    rows = t_dims[in][rank - 2];
    cols = t_dims[in][rank - 1];
    nmat = t_size[in] / (rows * cols);
    alloc_tensor(out, DT_FLOAT, rank, t_dims[in]);
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
                tf[out][f] = keep ? tf[in][f] : 0.0f;
                cc = cc + 1;
            }
            rr = rr + 1;
        }
        mat = mat + 1;
    }
}

/* ConstantOfShape: fill a tensor of the given int64 shape with a float bit-pattern */
void op_constofshape(int out, int shp, int bits) {
    int rank; int i; int rdims[MAX_RANK]; int sz; float fill;
    rank = t_size[shp];
    i = 0; while (i < rank) { rdims[i] = (int)geti(shp, i); i = i + 1; }
    memcpy(&fill, &bits, 4);
    alloc_tensor(out, DT_FLOAT, rank, rdims);
    sz = t_size[out];
    i = 0; while (i < sz) { tf[out][i] = fill; i = i + 1; }
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
        else if (op == OP_TRILU) op_trilu(o, i0, attr1(nd, A_UPPER, 1), (int)geti(i1, 0));
        else if (op == OP_CONSTANTOFSHAPE) op_constofshape(o, i0, attr1(nd, A_COS_FILL_BITS, 0));
        else if (op == OP_SIGMOID) op_unary(o, i0, 3);
        else if (op == OP_IDENTITY) op_identity(o, i0);
        else { printf("unknown op %d at node %d\n", op, nd); exit(1); }
        nd = nd + 1;
    }
}

/* =====================================================================
 * WINDOWED KV-CACHED INCREMENTAL BLOCK-STACK FORWARD  (the O(S^2) fix)
 * =====================================================================
 * The block stack is a perfectly regular pattern: 2248 leading IDENTITY
 * nodes (they alias the COO weights onto the MatMul-RHS tensor ids, no
 * compute), then 242 identical 69-node blocks (attn softmax1+ALiBi + SwiGLU
 * FFN).  We discover the per-block weight tensor ids from the FIRST block's
 * node structure, then run a hand-written incremental forward that is
 * byte-exact to the full graph at the last row.  See file header. */

#define NB_MAX 512          /* max blocks (242 today) */
#define BLK_NODES 69         /* nodes per block (regular pattern) */

int   g_incr_ready;          /* 1 once the block layout is discovered */
int   g_nblocks;             /* number of transformer blocks */
int   g_H;                   /* heads */
int   g_HD;                  /* head dim */
int   g_D;                   /* d_model = H*HD */
float g_scale;               /* attention score scale */
int   g_slope_tid;           /* alibi-slope tensor id ([1,H,1,1]) */
/* per-block COO weight tensor ids + FFN bias ids */
int   g_wq[NB_MAX], g_wk[NB_MAX], g_wv[NB_MAX], g_wo[NB_MAX];
int   g_wup[NB_MAX], g_wgate[NB_MAX], g_wdown[NB_MAX];
int   g_bup[NB_MAX], g_bgate[NB_MAX], g_bdown[NB_MAX];
int   g_ffn_hidden[NB_MAX];  /* FFN hidden width PER BLOCK (varies! DIV/MOD megablocks) */
int   g_ffn_hidden_max;      /* max over blocks (for scratch sizing) */

/* ---- per-block KV cache (the stable prefix, rows 0..prefix_len-1) ---- */
/* K/V stored [H][cap*HD] (row-major per head: pos-major, hd-minor). */
float *g_Kc[NB_MAX];         /* cached K, per block */
float *g_Vc[NB_MAX];         /* cached V, per block */
int    g_cap[NB_MAX];        /* allocated capacity (in positions) */
int    g_prefix_len;         /* number of cached (stable) prefix positions */
int    g_prev_S;             /* S of the previous step (0 = fresh) */

/* find the 2nd input of a node relative to the first block base */
static int _blk_node_in(int blk_base, int off, int which) {
    int nd = blk_base + off;
    return (n_nin[nd] > which) ? n_in[nd][which] : -1;
}

/* the bias operand of an Add(matmul_out, bias) node: the input that is NOT the
 * output of the preceding MatMul node `mm`.  Returns -1 if no such operand. */
static int _bias_of_add(int addn, int mm) {
    int mmout = (mm >= 0 && n_nout[mm] > 0) ? n_out[mm][0] : -2;
    if (n_nin[addn] < 2) return -1;
    if (n_in[addn][0] != mmout) return n_in[addn][0];
    return n_in[addn][1];
}

/* discover the regular block layout + weight ids.  Returns 1 on success.
 * FIRST runs the prologue IDENTITY nodes: they alias the COO weights onto the
 * MatMul-RHS tensor ids (onnx::MatMul_* -> weight), so after this the per-block
 * weight ids carry their COO row/col/val + dims.  Cheap (pure pointer aliasing,
 * no compute). */
int incr_discover(void) {
    int base; int b;
    /* the first transformer block starts at the first non-IDENTITY node */
    base = 0;
    while (base < n_nodes && n_op[base] == OP_IDENTITY) base = base + 1;
    if (base >= n_nodes) return 0;
    g_nblocks = (n_nodes - base) / BLK_NODES;
    if (g_nblocks <= 0 || g_nblocks > NB_MAX) return 0;
    if ((n_nodes - base) % BLK_NODES != 0) return 0;

    /* run the prologue IDENTITY nodes so the COO aliases are populated */
    { int nd; for (nd = 0; nd < base; nd = nd + 1)
        if (n_op[nd] == OP_IDENTITY) op_identity(n_out[nd][0], n_in[nd][0]); }

    /* per-block weight ids, from the regular node offsets (validated against
     * the graph dump):  Q=off6.in1 K=off18.in1 V=off20.in1 O=off57.in1
     * FFN up=off59.in1 gate=off61.in1 down=off66.in1; biases from the
     * following Add nodes (the non-MatMul-output operand). */
    g_ffn_hidden_max = 0;
    b = 0;
    while (b < g_nblocks) {
        int bb;
        bb = base + b * BLK_NODES;
        g_wq[b] = _blk_node_in(bb, 6, 1);
        g_wk[b] = _blk_node_in(bb, 18, 1);
        g_wv[b] = _blk_node_in(bb, 20, 1);
        g_wo[b] = _blk_node_in(bb, 57, 1);
        g_wup[b] = _blk_node_in(bb, 59, 1);
        g_wgate[b] = _blk_node_in(bb, 61, 1);
        g_wdown[b] = _blk_node_in(bb, 66, 1);
        /* biases: the non-MatMul-output operand of the Add.  May be absent
         * (folded-away zero: is_init==0, size 0) or COO — coo_bias() reads them
         * generically.  We keep the id; a NULL/absent bias reads as +0. */
        g_bup[b] = _bias_of_add(bb + 60, bb + 59);
        g_bgate[b] = _bias_of_add(bb + 62, bb + 61);
        g_bdown[b] = _bias_of_add(bb + 67, bb + 66);
        /* per-block FFN hidden = last dim of the (COO) up weight [D, hidden] */
        if (g_wup[b] < 0) return 0;
        g_ffn_hidden[b] = t_dims[g_wup[b]][t_rank[g_wup[b]] - 1];
        if (g_ffn_hidden[b] > g_ffn_hidden_max) g_ffn_hidden_max = g_ffn_hidden[b];
        b = b + 1;
    }
    /* dims: Q weight is [D,D] (COO after aliasing). */
    if (g_wq[0] < 0) return 0;
    g_D = t_dims[g_wq[0]][0];
    /* alibi slope tensor: node off33 (Mul_1) input 0 = onnx::Mul_2978x [1,H,1,1] */
    g_slope_tid = _blk_node_in(base, 33, 0);
    if (g_slope_tid < 0) return 0;
    g_H = t_dims[g_slope_tid][1];
    g_HD = g_D / g_H;
    /* scale constant: node off25 (Mul) input 1 = Constant_10 (scalar dense) */
    {
        int sc = _blk_node_in(base, 25, 1);
        g_scale = (sc >= 0 && tf[sc]) ? tf[sc][0] : 1.0f;
    }
    g_incr_ready = 1;
    g_prefix_len = 0;
    g_prev_S = 0;
    for (b = 0; b < g_nblocks; b = b + 1) {
        g_Kc[b] = 0; g_Vc[b] = 0; g_cap[b] = 0;
    }
    return 1;
}

/* row projection:  out[t][*] = sum_r act[t][r] * W[r][*] for t in [0,T).
 * act is [T,K] (row-major); out is [T,N].  W may be:
 *   (a) COO (t_is_coo): scatter the nnz in STORED order (ascending flat r*N+q),
 *       so each out[t][q] accumulates ascending-r == the dense reduction order.
 *   (b) DENSE (is_init==1, a small weight to_sparse_onnx kept dense): a plain
 *       row-major dense matmul, ALSO ascending-r per column (same order).
 * Both are byte-exact to the graph's op_matmul.  Parallelise over the T rows
 * (each writes a disjoint out row). */
static void coo_project(const float *act, int T, int K, int Wt,
                        float *out, int N) {
    int t;
    if (t_is_coo[Wt]) {
        int nnz = t_nnz[Wt];
        int *row = t_coo_row[Wt]; int *col = t_coo_col[Wt]; float *val = t_coo_val[Wt];
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (t = 0; t < T; t = t + 1) {
            const float *arow = act + (long)t * K;
            float *orow = out + (long)t * N;
            int q; int j;
            for (q = 0; q < N; q = q + 1) orow[q] = 0.0f;
            for (j = 0; j < nnz; j = j + 1)
                orow[col[j]] += arow[row[j]] * val[j];
        }
        SPARSE_ITERS = SPARSE_ITERS + (long)T * (long)nnz;
        DENSE_EQUIV_ITERS = DENSE_EQUIV_ITERS + (long)T * (long)K * (long)N;
    } else {
        /* dense fallback: out[t][q] = sum_r act[t][r] * W[r*N+q], ascending r */
        const float *W = tf[Wt];
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (t = 0; t < T; t = t + 1) {
            const float *arow = act + (long)t * K;
            float *orow = out + (long)t * N;
            int q; int r;
            for (q = 0; q < N; q = q + 1) orow[q] = 0.0f;
            for (r = 0; r < K; r = r + 1) {
                float a = arow[r];
                const float *wr = W + (long)r * N;
                for (q = 0; q < N; q = q + 1) orow[q] += a * wr[q];
            }
        }
        DENSE_EQUIV_ITERS = DENSE_EQUIV_ITERS + (long)T * (long)K * (long)N;
        SPARSE_ITERS = SPARSE_ITERS + (long)T * (long)K * (long)N;
    }
}

/* grow a per-block K/V cache to hold at least `need` positions */
static void cache_reserve(int b, int need) {
    if (g_cap[b] >= need) return;
    {
        int newcap = g_cap[b] ? g_cap[b] : 64;
        while (newcap < need) newcap = newcap * 2;
        g_Kc[b] = realloc(g_Kc[b], (long)newcap * g_D * sizeof(float));
        g_Vc[b] = realloc(g_Vc[b], (long)newcap * g_D * sizeof(float));
        g_cap[b] = newcap;
    }
}

/* one head's softmax1+ALiBi attention over the tail rows: Qt[t] (head h) attends
 * over cached K/V[0..SK-1], writes attnout[t][h*HD..].  Each head is independent
 * (disjoint output slice + private scratch) so this is parallel-safe. */
static void attn_head(int h, int b, int P, int T, int SK,
                      const float *Qt, float *attnout) {
    float slope = tf[g_slope_tid][h];
    int hoff = h * g_HD;
    float *sch = malloc((long)SK * sizeof(float));
    int t;
    for (t = 0; t < T; t = t + 1) {
        int qpos = P + t;
        const float *qv = Qt + (long)t * g_D + hoff;
        int k; float m; float denom; int hd;
        m = 0.0f;                   /* softmax1 sink: clamp(max,0) starts at 0 */
        for (k = 0; k < SK; k = k + 1) {
            if (k > qpos) { sch[k] = -1e30f; continue; }
            {
                const float *kv = g_Kc[b] + (long)k * g_D + hoff;
                float dot = 0.0f;
                for (hd = 0; hd < g_HD; hd = hd + 1) dot += qv[hd] * kv[hd];
                {
                    float dist = (float)(qpos - k);   /* |q-k|, k<=q */
                    sch[k] = dot * g_scale - slope * dist;
                }
                if (sch[k] > m) m = sch[k];
            }
        }
        denom = expf(-m);           /* softmax1: exp(x-m)/(exp(-m)+sum exp(x-m)) */
        for (k = 0; k <= qpos; k = k + 1) {
            sch[k] = expf(sch[k] - m);
            denom += sch[k];
        }
        {
            float *outv = attnout + (long)t * g_D + hoff;
            for (hd = 0; hd < g_HD; hd = hd + 1) outv[hd] = 0.0f;
            for (k = 0; k <= qpos; k = k + 1) {   /* attn@V, ascending-k order */
                float w = sch[k] / denom;
                const float *vv = g_Vc[b] + (long)k * g_D + hoff;
                for (hd = 0; hd < g_HD; hd = hd + 1) outv[hd] += w * vv[hd];
            }
        }
    }
    free(sch);
}

#ifdef USE_PTHREADS
/* fork-join parallel-for over the g_H heads (self-contained, static-linkable) */
typedef struct { int b, P, T, SK; const float *Qt; float *attnout;
                 int h0, h1; } _attn_arg;
static void *_attn_worker(void *a) {
    _attn_arg *w = (_attn_arg *)a; int h;
    for (h = w->h0; h < w->h1; h = h + 1)
        attn_head(h, w->b, w->P, w->T, w->SK, w->Qt, w->attnout);
    return 0;
}
static int _nthreads(void) {
    static int n = 0;
    if (n == 0) {
        char *e = getenv("INCR_THREADS");
        n = e ? atoi(e) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
    }
    return n;
}
#endif

static void attn_all_heads(int b, int P, int T, int SK,
                           const float *Qt, float *attnout) {
#ifdef USE_PTHREADS
    int nth = _nthreads();
    if (nth > g_H) nth = g_H;
    if (nth > 1) {
        pthread_t th[256]; _attn_arg args[256]; int i;
        int per = (g_H + nth - 1) / nth;
        for (i = 0; i < nth; i = i + 1) {
            args[i].b = b; args[i].P = P; args[i].T = T; args[i].SK = SK;
            args[i].Qt = Qt; args[i].attnout = attnout;
            args[i].h0 = i * per; args[i].h1 = (i + 1) * per;
            if (args[i].h1 > g_H) args[i].h1 = g_H;
            pthread_create(&th[i], 0, _attn_worker, &args[i]);
        }
        for (i = 0; i < nth; i = i + 1) pthread_join(th[i], 0);
        return;
    }
#endif
    {
        int h;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (h = 0; h < g_H; h = h + 1)
            attn_head(h, b, P, T, SK, Qt, attnout);
    }
}

/* the incremental block-stack forward over a residual [1,S,D] input already in
 * tf[input_tid].  Recomputes only the TAIL rows [P .. S-1] where P is the stable
 * prefix length (= prev_S-1, 0 on a fresh stream), reading each block's cached
 * prefix K/V.  Writes the exact hidden row -1 into tf[output_tid] row S-1 (only
 * that row is read by the driver).  Byte-exact vs run() at row -1.  Returns 0. */
void run_incremental(int S) {
    int P; int T; int b;
    float *xin;                 /* current block-input residual for tail rows [T,D] */
    float *xout;                /* current block-output residual for tail rows [T,D] */
    float *Kt; float *Vt; float *Qt; float *attnout;

    if (!g_incr_ready) { run(); return; }

    /* prefix length: rows [0..P-1] are stable (cached).  On a fresh stream (or a
     * reset when S went backwards) P=0 and caches are empty. */
    if (g_prev_S == 0 || S <= g_prev_S) {
        P = 0; g_prefix_len = 0;
        for (b = 0; b < g_nblocks; b = b + 1) g_cap[b] = g_cap[b]; /* keep alloc */
    } else {
        P = g_prev_S - 1;       /* verified: rows 0..prev_S-2 immutable */
    }
    if (P != g_prefix_len) {
        /* the cache holds exactly g_prefix_len positions; if the requested prefix
         * differs (shouldn't in steady state) fall back to a full rebuild. */
        P = 0; g_prefix_len = 0;
    }
    T = S - P;

    xin = malloc((long)T * g_D * sizeof(float));
    xout = malloc((long)T * g_D * sizeof(float));
    Qt = malloc((long)T * g_D * sizeof(float));
    Kt = malloc((long)T * g_D * sizeof(float));
    Vt = malloc((long)T * g_D * sizeof(float));
    attnout = malloc((long)T * g_D * sizeof(float));

    /* copy the tail rows of the input residual (positions P..S-1) */
    {
        int t;
        for (t = 0; t < T; t = t + 1)
            memcpy(xin + (long)t * g_D, tf[input_tid] + (long)(P + t) * g_D,
                   g_D * sizeof(float));
    }

    for (b = 0; b < g_nblocks; b = b + 1) {
        int SK = P + T;         /* total keys = prefix + tail */
        int h; int t;
        /* Q/K/V projections for the tail rows (COO, ascending-r, byte-exact) */
        coo_project(xin, T, g_D, g_wq[b], Qt, g_D);
        coo_project(xin, T, g_D, g_wk[b], Kt, g_D);
        coo_project(xin, T, g_D, g_wv[b], Vt, g_D);

        /* append the tail K/V to the cache buffer so K/V is contiguous per head.
         * cache buffers hold rows [0..P-1]; we write tail rows into [P..SK-1]. */
        cache_reserve(b, SK);
        for (t = 0; t < T; t = t + 1) {
            memcpy(g_Kc[b] + (long)(P + t) * g_D, Kt + (long)t * g_D, g_D * sizeof(float));
            memcpy(g_Vc[b] + (long)(P + t) * g_D, Vt + (long)t * g_D, g_D * sizeof(float));
        }

        /* per-head attention: Qt[t] attends over K/V[0..SK-1], softmax1+ALiBi,
         * causal (mask k>q).  Parallelised over heads (disjoint output slices)
         * via OpenMP or self-contained pthreads. */
        attn_all_heads(b, P, T, SK, Qt, attnout);
        (void)h; (void)t;

        /* output projection W_o (COO) + residual: xout = xin + attnout @ W_o */
        coo_project(attnout, T, g_D, g_wo[b], xout, g_D);
        for (t = 0; t < T * g_D; t = t + 1) xout[t] = xout[t] + xin[t];

        /* SwiGLU FFN:  up = xout@Wup + b_up; gate = xout@Wgate + b_gate;
         *              h = silu(up)*gate;  xout' = xout + h@Wdown + b_down.
         * FFN hidden width is PER-BLOCK (DIV/MOD megablocks are wide); biases may
         * be absent (folded-away zero) -> read via the NULL-guarded g_bup etc. */
        {
            int hid = g_ffn_hidden[b];
            const float *bup = (g_bup[b] >= 0) ? tf[g_bup[b]] : 0;
            const float *bgate = (g_bgate[b] >= 0) ? tf[g_bgate[b]] : 0;
            const float *bdown = (g_bdown[b] >= 0) ? tf[g_bdown[b]] : 0;
            float *up = malloc((long)T * hid * sizeof(float));
            float *gate = malloc((long)T * hid * sizeof(float));
            float *hh = malloc((long)T * hid * sizeof(float));
            float *dn = malloc((long)T * g_D * sizeof(float));
            coo_project(xout, T, g_D, g_wup[b], up, hid);
            coo_project(xout, T, g_D, g_wgate[b], gate, hid);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (t = 0; t < T; t = t + 1) {
                int j;
                for (j = 0; j < hid; j = j + 1) {
                    float u = up[(long)t * hid + j] + (bup ? bup[j] : 0.0f);
                    float ga = gate[(long)t * hid + j] + (bgate ? bgate[j] : 0.0f);
                    float su = u / (1.0f + expf(-u));       /* SiLU(up) */
                    hh[(long)t * hid + j] = su * ga;
                }
            }
            coo_project(hh, T, hid, g_wdown[b], dn, g_D);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (t = 0; t < T; t = t + 1) {
                int j;
                for (j = 0; j < g_D; j = j + 1)
                    xout[(long)t * g_D + j] += dn[(long)t * g_D + j]
                                             + (bdown ? bdown[j] : 0.0f);
            }
            free(up); free(gate); free(hh); free(dn);
        }

        /* this block's output tail becomes the next block's input tail */
        { float *tmp = xin; xin = xout; xout = tmp; }
    }

    /* Materialise the hidden output tensor [1,S,D].  The single-step driver reads
     * only row -1; the SPECULATIVE (batched-K) driver reads the K frame-end rows,
     * all of which live in the freshly-computed tail [P..S-1].  So we write the
     * EXACT block-stack output for EVERY tail row (positions P..S-1) into the
     * output tensor, and zero the stable-prefix rows [0..P-1] (never read).  This
     * is BYTE-EXACT: each tail row's value in xin is the same value the row would
     * have as row -1 of a stream truncated there (causal attention → a row's
     * output depends only on rows <= it), so row (P+i) here == the single-step
     * decode of a stream that ended at P+i.  The row-(S-1) copy is bit-identical
     * to the prior single-row behaviour. */
    {
        int rdim[3]; rdim[0] = 1; rdim[1] = S; rdim[2] = g_D;
        int t;
        alloc_tensor(output_tid, DT_FLOAT, 3, rdim);
        if (P > 0) memset(tf[output_tid], 0, (long)P * g_D * sizeof(float));
        for (t = 0; t < T; t = t + 1)
            memcpy(tf[output_tid] + (long)(P + t) * g_D, xin + (long)t * g_D,
                   g_D * sizeof(float));
    }

    /* update the stable prefix for the next step: rows [0..S-2] are now stable.
     * The cache buffers already hold K/V for rows [0..SK-1] (=[0..S-1]); the new
     * stable prefix is S-1 positions (drop the current last row, which will be
     * recomputed next step).  We simply record the new prefix length; the last
     * row's cached K/V slot is overwritten next step. */
    g_prefix_len = S - 1;
    g_prev_S = S;

    free(xin); free(xout); free(Qt); free(Kt); free(Vt); free(attnout);
}

/* ============ main ============ */
int main(int argc, char **argv) {
    FILE *tf_in;
    int B; int S; int i; int dump; int a;
    int residual_in; int stats; int incremental;
    int dim[2];
    char *model; char *tokfile;

    if (argc < 3) {
        printf("usage: %s model.nblbin input.txt "
               "[--dump-argmax|--dump-logits] [--residual-in] [--incremental] "
               "[--stats]\n", argv[0]);
        return 1;
    }
    model = argv[1];
    tokfile = argv[2];
    dump = 0;
    residual_in = 0;
    stats = 0;
    incremental = 0;
    /* scan all trailing flags (order-independent) */
    a = 3;
    while (a < argc) {
        if (strcmp(argv[a], "--dump-argmax") == 0) dump = 1;
        else if (strcmp(argv[a], "--dump-logits") == 0) dump = 2;
        else if (strcmp(argv[a], "--residual-in") == 0) residual_in = 1;
        else if (strcmp(argv[a], "--incremental") == 0) incremental = 1;
        else if (strcmp(argv[a], "--stats") == 0) stats = 1;
        a = a + 1;
    }
    SPARSE_ITERS = 0; DENSE_EQUIV_ITERS = 0;

    i = 0; a = 0;
    while (a < MAX_TENSORS) { tf[a] = 0; ti[a] = 0; a = a + 1; }

    load(model);

    /* discover the regular block layout for the incremental path (no-op if the
     * graph is not the regular block stack — then --incremental falls back to the
     * full run() per frame). */
    if (incremental) {
        if (!incr_discover()) {
            fprintf(stderr, "incremental: block layout NOT regular; "
                    "falling back to full-recompute run()\n");
            incremental = 0;
        } else {
            fprintf(stderr, "incremental: %d blocks, H=%d HD=%d D=%d "
                    "ffn_hidden<=%d scale=%.9g  (windowed KV-cache, O(S) per step)\n",
                    g_nblocks, g_H, g_HD, g_D, g_ffn_hidden_max, g_scale);
        }
    }

    if (residual_in) {
        /* residual-in mode: input is a BINARY file — int32 B,S,D header then
         * B*S*D float32 values.  The input tensor is the pre-embedded /
         * program-overlaid residual [B,S,D] the corpus driver feeds; the C
         * runtime runs the block stack and dumps the hidden residual.  Output is
         * written as raw little-endian float32 bytes (no text) for speed. */
        int D; int n; int serve;
        /* --serve + tokfile "-": load the graph ONCE, then loop reading residual
         * frames from stdin (int32 B,S,D header + B*S*D float32) and writing the
         * hidden residual to stdout, until EOF.  This is the persistent-process
         * mode the corpus driver uses so the ~2 MB graph load is amortized over the
         * whole autoregressive run (no per-step re-load / re-spawn). */
        serve = (strcmp(tokfile, "-") == 0) ? 1 : 0;
        if (serve) {
            while (fread(&B, sizeof(int), 1, stdin) == 1) {
                fread(&S, sizeof(int), 1, stdin);
                fread(&D, sizeof(int), 1, stdin);
                {
                    int rdim[3]; rdim[0] = B; rdim[1] = S; rdim[2] = D;
                    alloc_tensor(input_tid, DT_FLOAT, 3, rdim);
                }
                t_is_init[input_tid] = 1;
                n = B * S * D;
                fread(tf[input_tid], sizeof(float), n, stdin);
                if (incremental) run_incremental(S); else run();
                { int sz; sz = t_size[output_tid];
                  fwrite(tf[output_tid], sizeof(float), sz, stdout); }
                fflush(stdout);
            }
            return 0;
        }
        tf_in = fopen(tokfile, "rb");
        if (!tf_in) { printf("cannot open %s\n", tokfile); return 1; }
        fread(&B, sizeof(int), 1, tf_in);
        fread(&S, sizeof(int), 1, tf_in);
        fread(&D, sizeof(int), 1, tf_in);
        {
            int rdim[3]; rdim[0] = B; rdim[1] = S; rdim[2] = D;
            alloc_tensor(input_tid, DT_FLOAT, 3, rdim);
        }
        t_is_init[input_tid] = 1;
        n = B * S * D;
        fread(tf[input_tid], sizeof(float), n, tf_in);
        fclose(tf_in);
        if (incremental) run_incremental(S); else run();
        /* dump the block-stack output residual as raw float32 (stdout binary) */
        {
            int sz; sz = t_size[output_tid];
            fwrite(tf[output_tid], sizeof(float), sz, stdout);
        }
        if (stats) {
            fprintf(stderr, "SPARSE_MATMUL_ITERS %ld  DENSE_EQUIV_ITERS %ld  "
                    "ratio %.2f\n", SPARSE_ITERS, DENSE_EQUIV_ITERS,
                    (double)DENSE_EQUIV_ITERS / (double)(SPARSE_ITERS > 0 ? SPARSE_ITERS : 1));
        }
        return 0;
    }

    /* read tokens: "B S" then B*S ints */
    tf_in = fopen(tokfile, "r");
    if (!tf_in) { printf("cannot open %s\n", tokfile); return 1; }
    fscanf(tf_in, "%d %d", &B, &S);
    dim[0] = B; dim[1] = S;
    alloc_tensor(input_tid, DT_INT64, 2, dim);
    t_is_init[input_tid] = 1;
    i = 0;
    while (i < B * S) { int tk; fscanf(tf_in, "%d", &tk); ti[input_tid][i] = (long)tk; i = i + 1; }
    fclose(tf_in);
    fprintf(stderr, "tokens: B=%d S=%d\n", B, S);

    run();

    /* logits: [B, S, vocab] — argmax over vocab per (b,s) */
    {
        int vocab; int rank; int rows; int r; int c; int best; float bv;
        rank = t_rank[output_tid];
        vocab = t_dims[output_tid][rank - 1];
        rows = t_size[output_tid] / vocab;
        fprintf(stderr, "logits shape:");
        r = 0; while (r < rank) { fprintf(stderr, " %d", t_dims[output_tid][r]); r = r + 1; }
        fprintf(stderr, "\n");
        if (dump == 2) {
            /* raw logits, one float per line (%.9g for full precision) */
            r = 0;
            while (r < rows * vocab) { printf("%.9g\n", tf[output_tid][r]); r = r + 1; }
        } else if (dump == 1) {
            r = 0;
            while (r < rows) {
                best = 0; bv = tf[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tf[output_tid][r * vocab + c] > bv) { bv = tf[output_tid][r * vocab + c]; best = c; }
                    c = c + 1;
                }
                printf("%d\n", best);
                r = r + 1;
            }
        } else {
            /* print last-position argmax + a checksum of all argmaxes */
            int sum; sum = 0;
            r = 0;
            while (r < rows) {
                best = 0; bv = tf[output_tid][r * vocab];
                c = 1;
                while (c < vocab) {
                    if (tf[output_tid][r * vocab + c] > bv) { bv = tf[output_tid][r * vocab + c]; best = c; }
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

/* ============ embedded model blob ============================================
 * The full-ISA COO .nblbin is linked in as a static byte array via GNU-as
 * `.incbin`.  The (bare) path is supplied at compile time with
 *   -DMODEL_BLOB_PATH=/abs/path/blockstack.nblbin
 * so `prog` needs NO external model file — the weights + graph are IN the binary.
 * `_model_blob` / `_model_blob_end` bracket the bytes (see the memory-cursor
 * reader above).  STRINGIZE the bare path so the .incbin gets a quoted string. */
#ifndef MODEL_BLOB_PATH
#error "compile with -DMODEL_BLOB_PATH=/path/to/blockstack.nblbin"
#endif
#define _STR2(x) #x
#define _STR(x) _STR2(x)
__asm__(
    ".section .rodata\n"
    ".global _model_blob\n"
    ".align 16\n"
    "_model_blob:\n"
    ".incbin \"" _STR(MODEL_BLOB_PATH) "\"\n"
    ".global _model_blob_end\n"
    "_model_blob_end:\n"
    ".previous\n"
);
