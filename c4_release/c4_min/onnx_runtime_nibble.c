/*
 * onnx_runtime_nibble.c  —  a C ONNX runtime retargeted to the NIBBLE VM graph.
 *
 * The in-repo C ONNX runtime (vm/onnx_runtime_c4.c) was written for the broken
 * 48k export and knew ~17 ops over 2-D fixed-point tensors.  The nibble VM's
 * exported graph (c4_min/export_onnx.py: softmax1 + ALiBi + SwiGLU, a *vanilla*
 * decode-only transformer) needs 23 op types over general N-D tensors, incl. the
 * 14 the old runtime lacked:
 *
 *     Abs Cast Clip ConstantOfShape Exp Gather Neg Range
 *     ReduceMax ReduceSum Shape Trilu Unsqueeze   (+ Constant, folded at export)
 *
 * This runtime implements all 23 (Gather=embedding lookup; Exp/ReduceMax/
 * ReduceSum/Div = softmax1; Abs/Range/Neg/Sub = ALiBi; Trilu/ConstantOfShape =
 * causal mask; Cast/Shape/Unsqueeze/Concat = the traced shape plumbing) over
 * N-D float tensors, and runs the flat .nblbin graph produced by
 * c4_min/onnx_to_c4bin.py from the real ONNX file.
 *
 * It is a genuine graph interpreter: the shape-plumbing ops produce concrete
 * int shapes at run time (nothing is baked), so any [B,S] token frame runs.
 *
 * float parity: this reference build uses native `float`, giving results that
 * are argmax(byte-head)-identical to torch/onnxruntime on the proof frame — the
 * VM reads its state by argmax over the byte head, so that is the meaningful
 * "byte-identical" (torch vs ORT already differ ~3e-5 in raw fp, never in
 * argmax).  A fixed-point 16.16 build (for eventual self-hosting under the
 * integer-only c4 VM) is future work; see docs/NIBBLE_ONNX_RUNTIME_2026_07_14.md.
 *
 * Build:   gcc -O2 -lm -o onnx_runtime_nibble onnx_runtime_nibble.c
 * Run:     ./onnx_runtime_nibble model.nblbin tokens.txt [--dump-argmax]
 *          (tokens.txt: first line "B S", then B*S ints)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* Sized for the COMPACT c4_min VM graph (dim=1564, 42 blocks): 4461 distinct
 * tensors / 3186 real nodes after Constant folding.  The tiny proof model
 * (dim=104, 2 blocks) fits far inside these bounds too — one runtime for both. */
#define MAX_TENSORS 8192
#define MAX_NODES 6144
#define MAX_IO 8
#define MAX_RANK 8
#define MAX_ATTRS 4

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
float rd_f32(FILE *f) {
    int bits;
    float x;
    bits = rd_i32(f);
    memcpy(&x, &bits, 4);
    return x;
}

/* ============ tensor helpers ============ */
int prod(int rank, int *dims) {
    int p; int i;
    p = 1; i = 0;
    while (i < rank) { p = p * dims[i]; i = i + 1; }
    return p;
}

/* allocate a runtime tensor's storage for the given shape + dtype */
void alloc_tensor(int tid, int dtype, int rank, int *dims) {
    int sz; int i;
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
        tf[i] = 0; ti[i] = 0;
        dt = rd_i32(f);
        rank = rd_i32(f);
        j = 0;
        while (j < rank) { dims[j] = rd_i32(f); j = j + 1; }
        ne = rd_i32(f);
        if (isi == 1) {
            /* dense initializer: ne stored values */
            alloc_tensor(i, dt, rank, dims);
            if (dt == DT_FLOAT) {
                j = 0; while (j < ne) { tf[i][j] = rd_f32(f); j = j + 1; }
            } else {
                j = 0; while (j < ne) { ti[i][j] = rd_i64(f); j = j + 1; }
            }
        } else if (isi == 2) {
            /* COO-sparse float initializer: ne = DENSE numel, then nnz, then
             * nnz int64 flat-indices + nnz float values.  Reconstruct dense
             * (zero-filled) so the compute path is byte-identical to dense. */
            int nnz; long *s_idx;
            alloc_tensor(i, DT_FLOAT, rank, dims);
            j = 0; while (j < ne) { tf[i][j] = 0.0f; j = j + 1; }
            nnz = rd_i32(f);
            s_idx = malloc(nnz * sizeof(long));
            j = 0;
            while (j < nnz) { s_idx[j] = rd_i64(f); j = j + 1; }
            j = 0;
            while (j < nnz) { tf[i][s_idx[j]] = rd_f32(f); j = j + 1; }
            free(s_idx);
        } else {
            t_dtype[i] = -1; t_rank[i] = 0; t_size[i] = 0;
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
    /* diagnostics on stderr so --residual-in / --dump-logits keep a clean stdout */
    fprintf(stderr, "loaded %s: %d tensors, %d nodes (in=%d out=%d)\n",
            path, n_tensors, n_nodes, input_tid, output_tid);
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

/* generic N-D MatMul: batched over leading dims, contract last of A with 2nd-last of B */
void op_matmul(int out, int a, int b) {
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
        else if (op == OP_IDENTITY) op_unary(o, i0, 4);
        else { printf("unknown op %d at node %d\n", op, nd); exit(1); }
        nd = nd + 1;
    }
}

/* ============ main ============ */
int main(int argc, char **argv) {
    FILE *tf_in;
    int B; int S; int i; int dump; int a;
    int residual_in;
    int dim[2];
    char *model; char *tokfile;

    if (argc < 3) {
        printf("usage: %s model.nblbin input.txt "
               "[--dump-argmax|--dump-logits] [--residual-in]\n", argv[0]);
        return 1;
    }
    model = argv[1];
    tokfile = argv[2];
    dump = 0;
    residual_in = 0;
    /* scan all trailing flags (order-independent) */
    a = 3;
    while (a < argc) {
        if (strcmp(argv[a], "--dump-argmax") == 0) dump = 1;
        else if (strcmp(argv[a], "--dump-logits") == 0) dump = 2;
        else if (strcmp(argv[a], "--residual-in") == 0) residual_in = 1;
        a = a + 1;
    }

    i = 0; a = 0;
    while (a < MAX_TENSORS) { tf[a] = 0; ti[a] = 0; a = a + 1; }

    load(model);

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
                run();
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
        run();
        /* dump the block-stack output residual as raw float32 (stdout binary) */
        {
            int sz; sz = t_size[output_tid];
            fwrite(tf[output_tid], sizeof(float), sz, stdout);
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
