#!/usr/bin/env python3
"""
Bundle autoregressive VM model + quine source into a standalone C file.

Takes .arvm model + quine source, outputs a single C file that:
1. Embeds model weights as C arrays (sparse COO for matrices, dense for vectors)
2. Embeds compiled C4 bytecode and data section
3. Contains fixed-point (16.16) transformer inference runtime
4. Runs autoregressive generation loop
5. PUTCHAR output reproduces the quine source

Every character flows through the transformer: embed -> attention -> FFN -> head -> argmax.

Usage:
    python tools/bundle_autoregressive_quine.py \\
        --model model.arvm \\
        --program cllm/quine_cllm.c \\
        --output build/bundled/quine_autoregressive.c
"""

import argparse
import struct
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from tools.export_autoregressive import load_arvm

FIXED_POINT_SCALE = 65536


# =========================================================================
# Weight array formatting
# =========================================================================

def float_to_fp(val):
    """Convert float to 16.16 fixed-point int."""
    return int(round(float(val) * FIXED_POINT_SCALE))


def float_to_ieee(val):
    """Convert float to IEEE 754 fp32 bit pattern as signed int."""
    return struct.unpack('<i', struct.pack('<f', float(val)))[0]


def format_int_array(values, per_line=12):
    """Format list of ints as C array initializer body (indented lines)."""
    if not values:
        return '    0'
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append('    ' + ', '.join(str(v) for v in chunk))
    return ',\n'.join(lines)


def emit_dense(name, array):
    """Emit 1D or 2D array as dense fixed-point C array.

    Returns C code string.
    """
    flat = np.asarray(array).flatten().astype(np.float64)
    fp_vals = [float_to_fp(v) for v in flat]
    shape = np.asarray(array).shape
    shape_str = 'x'.join(str(s) for s in shape)
    return (
        f'/* {name}: {shape_str} */\n'
        f'static int {name}[{len(fp_vals)}] = {{\n'
        f'{format_int_array(fp_vals)}\n'
        f'}};'
    )


def emit_sparse(name, matrix):
    """Emit 2D weight as sparse COO (flat index + value) C arrays.

    Returns (code_string, nnz, dim1).
    """
    mat = np.asarray(matrix)
    flat = mat.flatten().astype(np.float64)
    fp_vals = [float_to_fp(v) for v in flat]
    nonzero = [(i, v) for i, v in enumerate(fp_vals) if v != 0]
    nnz = len(nonzero)
    dim0 = mat.shape[0]
    dim1 = mat.shape[1] if mat.ndim > 1 else 1
    arr_size = max(nnz, 1)

    if nnz == 0:
        rows_body = '    0'
        vals_body = '    0'
    else:
        rows = [idx for idx, _ in nonzero]
        vals = [val for _, val in nonzero]
        rows_body = format_int_array(rows)
        vals_body = format_int_array(vals)

    code = (
        f'/* {name}: {dim0}x{dim1}, {nnz} nonzero of {dim0 * dim1} */\n'
        f'static int {name}_rows[{arr_size}] = {{\n{rows_body}\n}};\n'
        f'static int {name}_vals[{arr_size}] = {{\n{vals_body}\n}};'
    )
    return code, nnz, dim1


def emit_c4_dense_init(name, array, indent='    ', fp_convert=None):
    """Generate C4 init code: malloc + memset(0) + set nonzero entries."""
    if fp_convert is None:
        fp_convert = float_to_fp
    flat = np.asarray(array).flatten().astype(np.float64)
    fp_vals = [fp_convert(v) for v in flat]
    size = len(fp_vals)
    lines = []
    lines.append(f'{indent}{name} = malloc({size} * sizeof(int));')
    lines.append(f'{indent}memset({name}, 0, {size} * sizeof(int));')
    for i, v in enumerate(fp_vals):
        if v != 0:
            lines.append(f'{indent}{name}[{i}] = {v};')
    return '\n'.join(lines)


def emit_c4_sparse_init(rows_name, vals_name, matrix, indent='    ', fp_convert=None):
    """Generate C4 init code for sparse COO arrays.

    rows_name/vals_name: target array expressions (e.g. 'head_w_rows'
    or 'layer_wq_rows[0]').

    Returns (code_string, nnz, dim1).
    """
    if fp_convert is None:
        fp_convert = float_to_fp
    mat = np.asarray(matrix)
    flat = mat.flatten().astype(np.float64)
    fp_vals = [fp_convert(v) for v in flat]
    nonzero = [(i, v) for i, v in enumerate(fp_vals) if v != 0]
    nnz = len(nonzero)
    dim1 = mat.shape[1] if mat.ndim > 1 else 1
    arr_size = max(nnz, 1)

    lines = []
    lines.append(f'{indent}{rows_name} = malloc({arr_size} * sizeof(int));')
    lines.append(f'{indent}{vals_name} = malloc({arr_size} * sizeof(int));')
    if nnz == 0:
        lines.append(f'{indent}{rows_name}[0] = 0;')
        lines.append(f'{indent}{vals_name}[0] = 0;')
    else:
        for k, (idx, val) in enumerate(nonzero):
            lines.append(f'{indent}{rows_name}[{k}] = {idx};')
            lines.append(f'{indent}{vals_name}[{k}] = {val};')
    return '\n'.join(lines), nnz, dim1


# =========================================================================
# C code generation
# =========================================================================

def generate_weights_and_tables(weights):
    """Generate all weight arrays and lookup tables.

    Returns (arrays_code, tables_code).
    """
    n_layers = weights['n_layers']
    array_parts = []
    table_parts = []

    # Embedding (dense)
    array_parts.append(emit_dense('embed_data', weights['embed_weight']))

    # Per-layer sparse weight metadata: {c_name: [(nnz, dim1), ...]}
    SPARSE_KEYS = [
        ('W_q', 'w_q'), ('W_k', 'w_k'), ('W_v', 'w_v'), ('W_o', 'w_o'),
        ('W_up', 'w_up'), ('W_gate', 'w_gate'), ('W_down', 'w_down'),
    ]
    sparse_meta = {c: [] for _, c in SPARSE_KEYS}

    for i in range(n_layers):
        layer = weights['layers'][i]
        pfx = f'l{i}'

        # ALiBi slopes (dense 1D)
        array_parts.append(emit_dense(f'{pfx}_alibi', layer['alibi_slopes']))

        # Attention + FFN matrices (sparse)
        for wkey, cname in SPARSE_KEYS:
            code, nnz, dim1 = emit_sparse(f'{pfx}_{cname}', layer[wkey])
            array_parts.append(code)
            sparse_meta[cname].append((nnz, dim1))

        # FFN biases (dense)
        array_parts.append(emit_dense(f'{pfx}_b_up', layer['b_up']))
        array_parts.append(emit_dense(f'{pfx}_b_gate', layer['b_gate']))
        array_parts.append(emit_dense(f'{pfx}_b_down', layer['b_down']))

    # Head weight (sparse) + bias (dense)
    head_code, head_nnz, head_dim1 = emit_sparse('head_w', weights['head_weight'])
    array_parts.append(head_code)
    array_parts.append(emit_dense('head_bias', weights['head_bias']))

    # --- Lookup tables ---
    TABLE_MAP = {
        'w_q': 'wq', 'w_k': 'wk', 'w_v': 'wv', 'w_o': 'wo',
        'w_up': 'w_up', 'w_gate': 'w_gate', 'w_down': 'w_down',
    }

    table_parts.append('/* Per-layer sparse weight lookup tables */')
    for cname, tname in TABLE_MAP.items():
        meta = sparse_meta[cname]
        rows_ptrs = ', '.join(f'l{i}_{cname}_rows' for i in range(n_layers))
        vals_ptrs = ', '.join(f'l{i}_{cname}_vals' for i in range(n_layers))
        nnz_vals = ', '.join(str(m[0]) for m in meta)
        dim1_vals = ', '.join(str(m[1]) for m in meta)
        table_parts.append(f'static int *layer_{tname}_rows[] = {{{rows_ptrs}}};')
        table_parts.append(f'static int *layer_{tname}_vals[] = {{{vals_ptrs}}};')
        table_parts.append(f'static int layer_{tname}_nnz[] = {{{nnz_vals}}};')
        table_parts.append(f'static int layer_{tname}_dim1[] = {{{dim1_vals}}};')

    # Dense lookup tables
    table_parts.append('')
    table_parts.append('/* Per-layer dense weight lookup tables */')
    for dname in ['alibi', 'b_up', 'b_gate', 'b_down']:
        ptrs = ', '.join(f'l{i}_{dname}' for i in range(n_layers))
        table_parts.append(f'static int *layer_{dname}[] = {{{ptrs}}};')

    # Head sparse metadata
    table_parts.append('')
    table_parts.append(f'static int head_w_nnz = {head_nnz};')
    table_parts.append(f'static int head_w_dim1 = {head_dim1};')

    return '\n\n'.join(array_parts), '\n'.join(table_parts)


def generate_bytecode_arrays(bytecode, data):
    """Generate C arrays for bytecode (op/imm pairs) and data section."""
    parts = []

    # Bytecode as separate op and imm arrays
    ops = []
    imms = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        # Sign-extend if needed (Python ints are arbitrary precision)
        if imm >= (1 << 55):
            imm -= (1 << 56)
        ops.append(op)
        imms.append(imm)

    parts.append(
        f'/* Compiled bytecode: {len(bytecode)} instructions */\n'
        f'static int bytecode_ops[{len(ops)}] = {{\n'
        f'{format_int_array(ops)}\n'
        f'}};\n'
        f'static int bytecode_imms[{len(imms)}] = {{\n'
        f'{format_int_array(imms)}\n'
        f'}};\n'
        f'static int bytecode_len = {len(bytecode)};'
    )

    # Data section
    if data:
        data_vals = list(data) if isinstance(data, (bytes, bytearray)) else data
        parts.append(
            f'/* Data section: {len(data_vals)} bytes */\n'
            f'static unsigned char data_section[{len(data_vals)}] = {{\n'
            f'{format_int_array(data_vals)}\n'
            f'}};\n'
            f'static int data_section_len = {len(data_vals)};'
        )
    else:
        parts.append(
            'static unsigned char data_section[1] = {0};\n'
            'static int data_section_len = 0;'
        )

    return '\n\n'.join(parts)


# =========================================================================
# Static C code sections (no Python interpolation needed)
# =========================================================================

C_HEADER = """\
/* ======================================================================
 * Autoregressive Neural Quine
 *
 * Self-contained C file that runs a decoder-only transformer to execute
 * C4 bytecode autoregressively. PUTCHAR steps emit the quine source.
 *
 * Fixed-point (16.16) transformer inference.
 * Every character flows through: embed -> attention -> FFN -> head -> argmax.
 *
 * Generated by tools/bundle_autoregressive_quine.py
 * ====================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
"""

C_TOKEN_DEFINES = """\
/* ==================== Token Constants ==================== */

#define TOK_SEP        256
#define TOK_REG_PC     257
#define TOK_REG_AX     258
#define TOK_REG_SP     259
#define TOK_REG_BP     260
#define TOK_MEM        261
#define TOK_STEP_END   262
#define TOK_HALT       263
#define TOK_CODE_START 264
#define TOK_CODE_END   265
#define TOK_DATA_START 266
#define TOK_DATA_END   267
#define OP_PUTCHAR     65
"""

C_FP_MATH = """\
/* ==================== Fixed-Point Math (16.16) ==================== */

static int fp_mul(int a, int b) {
    return (a / 256) * (b / 256);
}

static int fp_div(int a, int b) {
    if (b == 0) return 0;
    return (a * 256) / (b / 256);
}

static int fp_pow2(int x) {
    int int_part, frac, f2, f3, val, neg;
    if (x >= 0)
        int_part = x / FP_SCALE;
    else
        int_part = -((-x + FP_SCALE - 1) / FP_SCALE);
    frac = x - int_part * FP_SCALE;
    f2 = fp_mul(frac, frac);
    f3 = fp_mul(f2, frac);
    val = FP_SCALE
        + fp_mul(45426, frac)
        + fp_mul(15743, f2)
        + fp_mul(3638, f3);
    if (int_part >= 0) {
        if (int_part >= 20) return 0x7FFFFFFF;
        return val << int_part;
    } else {
        neg = -int_part;
        if (neg >= 30) return 0;
        return val >> neg;
    }
}

static int fp_exp(int x) {
    int y = fp_mul(x, 94548); /* x * log2(e) in FP16.16 */
    if (y > 20 * FP_SCALE) return 0x7FFFFFFF;
    if (y < -30 * FP_SCALE) return 0;
    return fp_pow2(y);
}

static int fp_sigmoid(int x) {
    int enx, denom;
    if (x >= 8 * FP_SCALE) return FP_SCALE;
    if (x <= -8 * FP_SCALE) return 0;
    enx = fp_exp(-x);
    denom = FP_SCALE + enx;
    if (denom == 0) return FP_SCALE;
    return fp_div(FP_SCALE, denom);
}

static int fp_silu(int x) {
    return fp_mul(x, fp_sigmoid(x));
}
"""

C_RUNTIME = """\
/* ==================== Runtime Functions ==================== */

/* Sparse matrix-vector multiply: out += W @ in
 * W stored as COO: rows[k] = flat index, vals[k] = FP16.16 value
 * n_cols = number of columns in W (inner dimension)
 */
static void sparse_matvec(const int *rows, const int *vals, int nnz,
                          int n_cols, const int *in, int *out) {
    int k;
    for (k = 0; k < nnz; k++) {
        int flat = rows[k];
        int r = flat / n_cols;
        int c = flat % n_cols;
        out[r] += fp_mul(vals[k], in[c]);
    }
}

/* Embed token: lookup row in dense embedding table */
static void embed_token(int token_id, int *out) {
    int i;
    if (token_id >= 0 && token_id < VOCAB_SIZE) {
        memcpy(out, embed_data + token_id * D_MODEL, D_MODEL * sizeof(int));
    } else {
        for (i = 0; i < D_MODEL; i++)
            out[i] = 0;
    }
}

/* softmax1: softmax with +1 anchor (ZFOD semantics)
 * out[i] = exp(x[i]) / (1 + sum(exp(x[j])))
 * The +1 anchor means when all scores are very negative, output -> 0.
 */
static void fp_softmax1(int *scores, int n) {
    int max_val = 0; /* anchor at 0 */
    int i, sum;
    for (i = 0; i < n; i++)
        if (scores[i] > max_val) max_val = scores[i];
    sum = fp_exp(-max_val); /* anchor term: exp(0 - max) */
    for (i = 0; i < n; i++) {
        scores[i] = fp_exp(scores[i] - max_val);
        sum += scores[i];
    }
    if (sum == 0) sum = 1;
    for (i = 0; i < n; i++)
        scores[i] = fp_div(scores[i], sum);
}

/* Multi-head attention with ALiBi + softmax1 + causal mask.
 * x_all: [seq_len * D_MODEL] — hidden states for all positions.
 * Computes attention for all positions, writes back with residual.
 */
static void ar_attention(int layer, int *x_all, int seq_len) {
    int d = D_MODEL;
    int H = N_HEADS;
    int HD = HEAD_DIM;
    int h, i, j, k;
    int scale = FP_SCALE / 8; /* 1/sqrt(64) = 0.125 for HEAD_DIM=64 */
    int *slopes = layer_alibi[layer];
    int *Q, *K, *V, *attn_out, *scores, *proj;

    Q = (int *)calloc(seq_len * d, sizeof(int));
    K = (int *)calloc(seq_len * d, sizeof(int));
    V = (int *)calloc(seq_len * d, sizeof(int));
    attn_out = (int *)calloc(seq_len * d, sizeof(int));
    scores = (int *)malloc(seq_len * sizeof(int));

    /* Q, K, V projections for all positions */
    for (i = 0; i < seq_len; i++) {
        sparse_matvec(layer_wq_rows[layer], layer_wq_vals[layer],
                      layer_wq_nnz[layer], layer_wq_dim1[layer],
                      x_all + i * d, Q + i * d);
        sparse_matvec(layer_wk_rows[layer], layer_wk_vals[layer],
                      layer_wk_nnz[layer], layer_wk_dim1[layer],
                      x_all + i * d, K + i * d);
        sparse_matvec(layer_wv_rows[layer], layer_wv_vals[layer],
                      layer_wv_nnz[layer], layer_wv_dim1[layer],
                      x_all + i * d, V + i * d);
    }

    /* Per-head attention with ALiBi */
    for (h = 0; h < H; h++) {
        int slope = slopes[h];
        int h_off = h * HD;

        for (i = 0; i < seq_len; i++) {
            /* Scores: Q[i] . K[j] for j <= i (causal) */
            for (j = 0; j <= i; j++) {
                int dot = 0;
                for (k = 0; k < HD; k++)
                    dot += fp_mul(Q[i * d + h_off + k], K[j * d + h_off + k]);
                dot = fp_mul(dot, scale);
                /* ALiBi bias: -slope * |i - j| */
                dot -= fp_mul(slope, (i - j) * FP_SCALE);
                scores[j] = dot;
            }

            /* softmax1 over [0..i] */
            fp_softmax1(scores, i + 1);

            /* Weighted sum of V */
            for (k = 0; k < HD; k++) {
                int s = 0;
                for (j = 0; j <= i; j++)
                    s += fp_mul(scores[j], V[j * d + h_off + k]);
                attn_out[i * d + h_off + k] = s;
            }
        }
    }

    /* Output projection + residual: x += W_o @ attn_out */
    proj = (int *)calloc(d, sizeof(int));
    for (i = 0; i < seq_len; i++) {
        memset(proj, 0, d * sizeof(int));
        sparse_matvec(layer_wo_rows[layer], layer_wo_vals[layer],
                      layer_wo_nnz[layer], layer_wo_dim1[layer],
                      attn_out + i * d, proj);
        for (j = 0; j < d; j++)
            x_all[i * d + j] += proj[j];
    }

    free(Q); free(K); free(V);
    free(attn_out); free(scores); free(proj);
}

/* SwiGLU FFN with residual:
 *   x += W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down
 */
static void ar_ffn(int layer, int *x_all, int seq_len) {
    int d = D_MODEL;
    int hid = HIDDEN_DIM;
    int i, j;
    int *up_out, *gate_out, *down_out;

    up_out = (int *)malloc(hid * sizeof(int));
    gate_out = (int *)malloc(hid * sizeof(int));
    down_out = (int *)malloc(d * sizeof(int));

    for (i = 0; i < seq_len; i++) {
        int *x = x_all + i * d;

        /* up = silu(W_up @ x + b_up) */
        memcpy(up_out, layer_b_up[layer], hid * sizeof(int));
        sparse_matvec(layer_w_up_rows[layer], layer_w_up_vals[layer],
                      layer_w_up_nnz[layer], layer_w_up_dim1[layer],
                      x, up_out);
        for (j = 0; j < hid; j++)
            up_out[j] = fp_silu(up_out[j]);

        /* gate = W_gate @ x + b_gate */
        memcpy(gate_out, layer_b_gate[layer], hid * sizeof(int));
        sparse_matvec(layer_w_gate_rows[layer], layer_w_gate_vals[layer],
                      layer_w_gate_nnz[layer], layer_w_gate_dim1[layer],
                      x, gate_out);

        /* hidden = up * gate (element-wise) */
        for (j = 0; j < hid; j++)
            up_out[j] = fp_mul(up_out[j], gate_out[j]);

        /* down = W_down @ hidden + b_down */
        memcpy(down_out, layer_b_down[layer], d * sizeof(int));
        sparse_matvec(layer_w_down_rows[layer], layer_w_down_vals[layer],
                      layer_w_down_nnz[layer], layer_w_down_dim1[layer],
                      up_out, down_out);

        /* residual */
        for (j = 0; j < d; j++)
            x[j] += down_out[j];
    }

    free(up_out); free(gate_out); free(down_out);
}

/* Full transformer forward pass: tokens[0..seq_len-1] -> logits[VOCAB_SIZE]
 * Computes logits for the LAST position only (for generation).
 * x_all holds all positions' hidden states (needed for attention).
 */
static void ar_forward(const int *tokens, int seq_len,
                       int *logits, int *x_all) {
    int i, layer;

    /* Embed all tokens */
    for (i = 0; i < seq_len; i++)
        embed_token(tokens[i], x_all + i * D_MODEL);

    /* Transformer blocks */
    for (layer = 0; layer < N_LAYERS; layer++) {
        ar_attention(layer, x_all, seq_len);
        ar_ffn(layer, x_all, seq_len);
    }

    /* Output head: logits = head_w @ x_last + head_bias */
    memcpy(logits, head_bias, VOCAB_SIZE * sizeof(int));
    sparse_matvec(head_w_rows, head_w_vals, head_w_nnz, head_w_dim1,
                  x_all + (seq_len - 1) * D_MODEL, logits);
}

/* Argmax over fixed-point logits */
static int fp_argmax(const int *logits, int n) {
    int best = 0, i;
    for (i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}
"""

C_GENERATION = """\
/* ==================== Context Builder ==================== */

/* Build context tokens from bytecode and data section.
 * Format: [CODE_START] op0 imm0_b0..b3 op1 imm1_b0..b3 ... [CODE_END]
 *         [DATA_START] data_bytes... [DATA_END]
 */
static int build_context(int *tokens, int max_tokens) {
    int pos = 0;
    int i;

    tokens[pos++] = TOK_CODE_START;

    for (i = 0; i < bytecode_len && pos + 5 < max_tokens; i++) {
        int op = bytecode_ops[i];
        unsigned int uimm = (unsigned int)bytecode_imms[i];
        tokens[pos++] = op & 0xFF;
        tokens[pos++] = (int)(uimm & 0xFF);
        tokens[pos++] = (int)((uimm >> 8) & 0xFF);
        tokens[pos++] = (int)((uimm >> 16) & 0xFF);
        tokens[pos++] = (int)((uimm >> 24) & 0xFF);
    }

    tokens[pos++] = TOK_CODE_END;
    tokens[pos++] = TOK_DATA_START;

    for (i = 0; i < data_section_len && pos < max_tokens - 1; i++)
        tokens[pos++] = (int)data_section[i];

    tokens[pos++] = TOK_DATA_END;

    return pos;
}

/* ==================== Register Extraction ==================== */

/* Extract register value from the last completed step.
 * Scans backward (up to 40 tokens) for the marker, reads 4 LE bytes.
 */
static int extract_reg(const int *tokens, int seq_len, int marker) {
    int i, j, limit;
    limit = seq_len - 40;
    if (limit < 0) limit = 0;
    for (i = seq_len - 1; i >= limit; i--) {
        if (tokens[i] == marker && i + 4 < seq_len) {
            int val = 0;
            for (j = 0; j < 4; j++)
                val |= (tokens[i + 1 + j] & 0xFF) << (j * 8);
            return val;
        }
    }
    return -1;
}

/* ==================== Generation Loop ==================== */

static int generate(int *tokens, int ctx_len) {
    int seq_len = ctx_len;
    int max_gen = 20000;
    int prev_pc = -1;
    int *logits, *x_all;
    int step;

    logits = (int *)malloc(VOCAB_SIZE * sizeof(int));
    x_all = (int *)malloc(MAX_SEQ_LEN * D_MODEL * sizeof(int));

    for (step = 0; step < max_gen && seq_len < MAX_SEQ_LEN - 1; step++) {
        int next_tok;

        /* Full forward pass (recompute all positions) */
        ar_forward(tokens, seq_len, logits, x_all);

        /* Greedy decode: argmax */
        next_tok = fp_argmax(logits, VOCAB_SIZE);
        tokens[seq_len++] = next_tok;

        /* Halt detection */
        if (next_tok == TOK_HALT) break;

        /* PUTCHAR detection: when STEP_END is emitted, check if the
         * instruction at prev_pc was PUTCHAR. If so, output AX byte.
         */
        if (next_tok == TOK_STEP_END) {
            if (prev_pc >= 0 && prev_pc < bytecode_len) {
                if (bytecode_ops[prev_pc] == OP_PUTCHAR) {
                    int ax = extract_reg(tokens, seq_len, TOK_REG_AX);
                    if (ax >= 0) {
                        putchar(ax & 0xFF);
                        fflush(stdout);
                    }
                }
            }
            /* Update prev_pc from current step's REG_PC */
            prev_pc = extract_reg(tokens, seq_len, TOK_REG_PC);
        }
    }

    free(logits);
    free(x_all);
    return seq_len;
}

/* ==================== Main ==================== */

int main(void) {
    int *tokens;
    int ctx_len;

    tokens = (int *)calloc(MAX_SEQ_LEN, sizeof(int));
    if (!tokens) {
        fprintf(stderr, "Failed to allocate token buffer\\n");
        return 1;
    }

    ctx_len = build_context(tokens, MAX_SEQ_LEN);
    fprintf(stderr, "Context: %d tokens\\n", ctx_len);
    fprintf(stderr, "Generating (full recompute per step)...\\n");

    generate(tokens, ctx_len);

    free(tokens);
    return 0;
}
"""


# =========================================================================
# C4-compatible code generation
# =========================================================================

C4_HEADER = """\
/* ======================================================================
 * Autoregressive Neural Quine — C4 Compatible
 *
 * Self-contained C file that runs a decoder-only transformer to execute
 * C4 bytecode autoregressively. PUTCHAR steps emit the quine source.
 *
 * Fixed-point (16.16) transformer inference.
 * Every character flows through: embed -> attention -> FFN -> head -> argmax.
 *
 * Generated by tools/bundle_autoregressive_quine.py --c4
 * ====================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
"""

C4_FP_MATH = """\
/* ==================== Fixed-Point Math (16.16) ==================== */

int fp_mul(int a, int b) {
    return (a / 256) * (b / 256);
}

int fp_div(int a, int b) {
    if (b == 0) return 0;
    return (a * 256) / (b / 256);
}

int fp_pow2(int x) {
    int int_part;
    int frac;
    int f2;
    int f3;
    int val;
    int neg;

    if (x >= 0) {
        int_part = x / FP_SCALE;
    } else {
        int_part = 0 - ((0 - x + FP_SCALE - 1) / FP_SCALE);
    }
    frac = x - int_part * FP_SCALE;

    f2 = fp_mul(frac, frac);
    f3 = fp_mul(f2, frac);
    val = FP_SCALE
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

int fp_exp(int x) {
    int y;
    y = fp_mul(x, 94548);
    if (y > 20 * FP_SCALE) return 0x7FFFFFFF;
    if (y < 0 - 30 * FP_SCALE) return 0;
    return fp_pow2(y);
}

int fp_sigmoid(int x) {
    int enx;
    int denom;

    if (x >= 8 * FP_SCALE) return FP_SCALE;
    if (x <= 0 - 8 * FP_SCALE) return 0;
    enx = fp_exp(0 - x);
    denom = FP_SCALE + enx;
    if (denom == 0) return FP_SCALE;
    return fp_div(FP_SCALE, denom);
}

int fp_silu(int x) {
    return fp_mul(x, fp_sigmoid(x));
}
"""

C4_SOFTFLOAT = """\
/* ==================== Soft-Float (IEEE 754 fp32) ==================== */

/* All values are IEEE 754 bit patterns stored in int.
 * No hardware float or long long needed — pure 32-bit int arithmetic.
 */

/* Soft-float constants (set in init_config) */
int SF_ONE;
int SF_TWO;
int SF_HALF;
int SF_EIGHT;
int SF_NEG_EIGHT;
int SF_LOG2E;
int SF_LN2;
int SF_INV6;
int SF_INV24;
int SF_EXP_HI;
int SF_EXP_LO;
int SF_ATTN_SCALE;

int sf_neg(int a) {
    return a ^ 0x80000000;
}

int sf_abs(int a) {
    return a & 0x7FFFFFFF;
}

int sf_gt(int a, int b) {
    int sa;
    int sb;

    /* Both zero (either sign) */
    if ((a & 0x7FFFFFFF) == 0 && (b & 0x7FFFFFFF) == 0) return 0;

    sa = a < 0;
    sb = b < 0;

    /* Different signs */
    if (sa == 0 && sb) return 1;
    if (sa && sb == 0) return 0;

    /* Same sign positive: larger int = larger float */
    if (sa == 0) return a > b;

    /* Same sign negative: smaller int = larger float */
    return a < b;
}

int sf_ge(int a, int b) {
    int sa;
    int sb;

    if ((a & 0x7FFFFFFF) == 0 && (b & 0x7FFFFFFF) == 0) return 1;

    sa = a < 0;
    sb = b < 0;

    if (sa == 0 && sb) return 1;
    if (sa && sb == 0) return 0;

    if (sa == 0) return a >= b;
    return a <= b;
}

int sf_mul(int a, int b) {
    int sa;
    int sb;
    int ea;
    int eb;
    int ma;
    int mb;
    int sign;
    int exp_out;
    int ah;
    int al;
    int bh;
    int bl;
    int hi;
    int mid;

    /* Zero check */
    if ((a & 0x7FFFFFFF) == 0) return 0;
    if ((b & 0x7FFFFFFF) == 0) return 0;

    sa = (a >> 31) & 1;
    sb = (b >> 31) & 1;
    sign = sa ^ sb;

    ea = (a >> 23) & 0xFF;
    eb = (b >> 23) & 0xFF;
    ma = (a & 0x7FFFFF) | 0x800000;
    mb = (b & 0x7FFFFF) | 0x800000;

    /* Split 24-bit mantissas into 12+12 for overflow-safe multiply */
    ah = ma >> 12;
    al = ma & 0xFFF;
    bh = mb >> 12;
    bl = mb & 0xFFF;

    hi = ah * bh;
    mid = ah * bl + al * bh;
    hi = hi + (mid >> 12);

    /* Normalize */
    exp_out = ea + eb - 127;
    if (hi >= 0x800000) {
        /* Product >= 2.0: hi is already the 24-bit mantissa */
        exp_out = exp_out + 1;
        if (exp_out >= 255) return (sign << 31) | 0x7F800000;
        if (exp_out <= 0) return 0;
        return (sign << 31) | (exp_out << 23) | (hi & 0x7FFFFF);
    }
    /* Product in [1.0, 2.0): shift left to get 24-bit mantissa */
    if (exp_out >= 255) return (sign << 31) | 0x7F800000;
    if (exp_out <= 0) return 0;
    return (sign << 31) | (exp_out << 23) | ((hi << 1) & 0x7FFFFF);
}

int sf_add(int a, int b) {
    int sa;
    int sb;
    int ea;
    int eb;
    int ma;
    int mb;
    int sign;
    int exp_out;
    int mant_out;
    int shift;
    int tmp;

    /* Zero handling */
    if ((a & 0x7FFFFFFF) == 0) return b;
    if ((b & 0x7FFFFFFF) == 0) return a;

    sa = (a >> 31) & 1;
    sb = (b >> 31) & 1;
    ea = (a >> 23) & 0xFF;
    eb = (b >> 23) & 0xFF;
    ma = (a & 0x7FFFFF) | 0x800000;
    mb = (b & 0x7FFFFF) | 0x800000;

    /* Ensure |a| >= |b| by comparing (exp, mantissa) */
    if (ea < eb || (ea == eb && ma < mb)) {
        tmp = sa; sa = sb; sb = tmp;
        tmp = ea; ea = eb; eb = tmp;
        tmp = ma; ma = mb; mb = tmp;
    }

    sign = sa;
    exp_out = ea;

    /* Align exponents */
    shift = ea - eb;
    if (shift > 24) shift = 24;
    mb = mb >> shift;

    if (sa == sb) {
        /* Same sign: add */
        mant_out = ma + mb;
        if (mant_out >= 0x1000000) {
            mant_out = mant_out >> 1;
            exp_out = exp_out + 1;
        }
    } else {
        /* Different signs: subtract */
        mant_out = ma - mb;
        if (mant_out == 0) return 0;
        /* Normalize: shift left until leading 1 at bit 23 */
        while (mant_out < 0x800000 && exp_out > 1) {
            mant_out = mant_out << 1;
            exp_out = exp_out - 1;
        }
    }

    if (exp_out >= 255) return (sign << 31) | 0x7F800000;
    if (exp_out <= 0) return 0;

    return (sign << 31) | (exp_out << 23) | (mant_out & 0x7FFFFF);
}

int sf_sub(int a, int b) {
    return sf_add(a, sf_neg(b));
}

int sf_from_int(int x) {
    int sign;
    int abs_x;
    int exp_adj;
    int mant;

    if (x == 0) return 0;

    sign = 0;
    abs_x = x;
    if (x < 0) {
        sign = 1;
        abs_x = 0 - x;
    }

    /* Normalize: shift until leading 1 at bit 23 */
    exp_adj = 0;
    mant = abs_x;
    while (mant >= 0x1000000) {
        mant = mant >> 1;
        exp_adj = exp_adj + 1;
    }
    while (mant < 0x800000) {
        mant = mant << 1;
        exp_adj = exp_adj - 1;
    }

    exp_adj = exp_adj + 23 + 127;
    if (exp_adj >= 255) return (sign << 31) | 0x7F800000;
    if (exp_adj <= 0) return 0;
    return (sign << 31) | (exp_adj << 23) | (mant & 0x7FFFFF);
}

int sf_to_int(int a) {
    int sign;
    int exp;
    int mant;
    int shift;

    if ((a & 0x7FFFFFFF) == 0) return 0;

    sign = (a >> 31) & 1;
    exp = (a >> 23) & 0xFF;
    mant = (a & 0x7FFFFF) | 0x800000;

    shift = exp - 127 - 23;
    if (shift >= 0) {
        if (shift > 7) {
            if (sign) return 0 - 0x7FFFFFFF;
            return 0x7FFFFFFF;
        }
        mant = mant << shift;
    } else {
        shift = 0 - shift;
        if (shift >= 24) return 0;
        mant = mant >> shift;
    }

    if (sign) return 0 - mant;
    return mant;
}

int sf_ldexp(int a, int n) {
    int exp;

    if ((a & 0x7FFFFFFF) == 0) return 0;

    exp = ((a >> 23) & 0xFF) + n;
    if (exp >= 255) return (a & 0x80000000) | 0x7F800000;
    if (exp <= 0) return 0;

    return (a & 0x807FFFFF) | (exp << 23);
}

int sf_div(int a, int b) {
    int sb;
    int eb;
    int x;
    int two;
    int i;

    /* Division by zero */
    if ((b & 0x7FFFFFFF) == 0) return 0;
    /* Zero numerator */
    if ((a & 0x7FFFFFFF) == 0) return 0;

    /* Newton-Raphson reciprocal: x = x * (2 - b*x), 5 iterations.
     * Initial estimate: flip exponent of |b|.
     */
    sb = (b >> 31) & 1;
    eb = (b >> 23) & 0xFF;

    /* Initial guess: 2^(-(eb-127)) = 2^(127-eb), biased exp = 254-eb */
    x = ((254 - eb) << 23) | 0x000000;

    two = SF_TWO;
    i = 0;
    while (i < 5) {
        /* x = x * (2 - |b| * x) */
        x = sf_mul(x, sf_sub(two, sf_mul(b & 0x7FFFFFFF, x)));
        i = i + 1;
    }

    /* result = a * (1/|b|), then apply b's sign */
    x = sf_mul(a, x);
    if (sb) x = sf_neg(x);
    return x;
}

int sf_exp(int x) {
    int ni;
    int n;
    int f;
    int result;

    /* Clamp to avoid overflow/underflow */
    if (sf_gt(x, SF_EXP_HI)) return 0x7F800000;
    if (sf_gt(SF_EXP_LO, x)) return 0;

    /* n = trunc(x * log2(e)) */
    ni = sf_to_int(sf_mul(x, SF_LOG2E));
    n = sf_from_int(ni);

    /* f = x - n * ln(2), so f is in [-ln2/2, ln2/2] */
    f = sf_sub(x, sf_mul(n, SF_LN2));

    /* Horner polynomial: exp(f) = 1 + f*(1 + f*(1/2 + f*(1/6 + f/24))) */
    result = sf_mul(f, SF_INV24);
    result = sf_add(SF_INV6, result);
    result = sf_mul(f, result);
    result = sf_add(SF_HALF, result);
    result = sf_mul(f, result);
    result = sf_add(SF_ONE, result);
    result = sf_mul(f, result);
    result = sf_add(SF_ONE, result);

    /* Multiply by 2^n */
    result = sf_ldexp(result, ni);
    return result;
}

int sf_sigmoid(int x) {
    int enx;
    int denom;

    /* Clamp */
    if (sf_ge(x, SF_EIGHT)) return SF_ONE;
    if (sf_ge(SF_NEG_EIGHT, x)) return 0;

    /* 1 / (1 + exp(-x)) */
    enx = sf_exp(sf_neg(x));
    denom = sf_add(SF_ONE, enx);
    if ((denom & 0x7FFFFFFF) == 0) return SF_ONE;
    return sf_div(SF_ONE, denom);
}

int sf_silu(int x) {
    return sf_mul(x, sf_sigmoid(x));
}
"""

C4_INIT_TOKENS = """\
int init_tokens() {
    TOK_SEP = 256;
    TOK_REG_PC = 257;
    TOK_REG_AX = 258;
    TOK_REG_SP = 259;
    TOK_REG_BP = 260;
    TOK_MEM = 261;
    TOK_STEP_END = 262;
    TOK_HALT = 263;
    TOK_CODE_START = 264;
    TOK_CODE_END = 265;
    TOK_DATA_START = 266;
    TOK_DATA_END = 267;
    OP_PUTCHAR = 65;
    return 0;
}
"""

C4_RUNTIME = """\
/* ==================== Runtime Functions ==================== */

int sparse_matvec(int *rows, int *vals, int nnz,
                  int n_cols, int *in, int *out) {
    int k;
    int flat;
    int r;
    int c;

    k = 0;
    while (k < nnz) {
        flat = rows[k];
        r = flat / n_cols;
        c = flat % n_cols;
        out[r] = out[r] + fp_mul(vals[k], in[c]);
        k = k + 1;
    }
    return 0;
}

int embed_token(int token_id, int *out) {
    int i;
    int *src;

    if (token_id >= 0 && token_id < VOCAB_SIZE) {
        src = embed_data + token_id * D_MODEL;
        i = 0;
        while (i < D_MODEL) {
            out[i] = src[i];
            i = i + 1;
        }
    } else {
        i = 0;
        while (i < D_MODEL) {
            out[i] = 0;
            i = i + 1;
        }
    }
    return 0;
}

int fp_softmax1(int *scores, int n) {
    int max_val;
    int i;
    int sum;

    max_val = 0;
    i = 0;
    while (i < n) {
        if (scores[i] > max_val) max_val = scores[i];
        i = i + 1;
    }
    sum = fp_exp(0 - max_val);
    i = 0;
    while (i < n) {
        scores[i] = fp_exp(scores[i] - max_val);
        sum = sum + scores[i];
        i = i + 1;
    }
    if (sum == 0) sum = 1;
    i = 0;
    while (i < n) {
        scores[i] = fp_div(scores[i], sum);
        i = i + 1;
    }
    return 0;
}

int ar_attention(int layer, int *x_all, int seq_len) {
    int d;
    int H;
    int HD;
    int h;
    int i;
    int j;
    int k;
    int scale;
    int *slopes;
    int *Q;
    int *K;
    int *V;
    int *attn_out;
    int *scores;
    int *proj;
    int slope;
    int h_off;
    int dot;
    int s;

    d = D_MODEL;
    H = N_HEADS;
    HD = HEAD_DIM;
    scale = FP_SCALE / 8;
    slopes = layer_alibi[layer];

    Q = malloc(seq_len * d * sizeof(int));
    memset(Q, 0, seq_len * d * sizeof(int));
    K = malloc(seq_len * d * sizeof(int));
    memset(K, 0, seq_len * d * sizeof(int));
    V = malloc(seq_len * d * sizeof(int));
    memset(V, 0, seq_len * d * sizeof(int));
    attn_out = malloc(seq_len * d * sizeof(int));
    memset(attn_out, 0, seq_len * d * sizeof(int));
    scores = malloc(seq_len * sizeof(int));

    /* Q, K, V projections for all positions */
    i = 0;
    while (i < seq_len) {
        sparse_matvec(layer_wq_rows[layer], layer_wq_vals[layer],
                      layer_wq_nnz[layer], layer_wq_dim1[layer],
                      x_all + i * d, Q + i * d);
        sparse_matvec(layer_wk_rows[layer], layer_wk_vals[layer],
                      layer_wk_nnz[layer], layer_wk_dim1[layer],
                      x_all + i * d, K + i * d);
        sparse_matvec(layer_wv_rows[layer], layer_wv_vals[layer],
                      layer_wv_nnz[layer], layer_wv_dim1[layer],
                      x_all + i * d, V + i * d);
        i = i + 1;
    }

    /* Per-head attention with ALiBi */
    h = 0;
    while (h < H) {
        slope = slopes[h];
        h_off = h * HD;

        i = 0;
        while (i < seq_len) {
            /* Scores: Q[i] . K[j] for j <= i (causal) */
            j = 0;
            while (j <= i) {
                dot = 0;
                k = 0;
                while (k < HD) {
                    dot = dot + fp_mul(Q[i * d + h_off + k], K[j * d + h_off + k]);
                    k = k + 1;
                }
                dot = fp_mul(dot, scale);
                dot = dot - fp_mul(slope, (i - j) * FP_SCALE);
                scores[j] = dot;
                j = j + 1;
            }

            fp_softmax1(scores, i + 1);

            /* Weighted sum of V */
            k = 0;
            while (k < HD) {
                s = 0;
                j = 0;
                while (j <= i) {
                    s = s + fp_mul(scores[j], V[j * d + h_off + k]);
                    j = j + 1;
                }
                attn_out[i * d + h_off + k] = s;
                k = k + 1;
            }
            i = i + 1;
        }
        h = h + 1;
    }

    /* Output projection + residual */
    proj = malloc(d * sizeof(int));
    i = 0;
    while (i < seq_len) {
        memset(proj, 0, d * sizeof(int));
        sparse_matvec(layer_wo_rows[layer], layer_wo_vals[layer],
                      layer_wo_nnz[layer], layer_wo_dim1[layer],
                      attn_out + i * d, proj);
        j = 0;
        while (j < d) {
            x_all[i * d + j] = x_all[i * d + j] + proj[j];
            j = j + 1;
        }
        i = i + 1;
    }

    free(Q); free(K); free(V);
    free(attn_out); free(scores); free(proj);
    return 0;
}

int ar_ffn(int layer, int *x_all, int seq_len) {
    int d;
    int hid;
    int i;
    int j;
    int *up_out;
    int *gate_out;
    int *down_out;
    int *x;

    d = D_MODEL;
    hid = HIDDEN_DIM;

    up_out = malloc(hid * sizeof(int));
    gate_out = malloc(hid * sizeof(int));
    down_out = malloc(d * sizeof(int));

    i = 0;
    while (i < seq_len) {
        x = x_all + i * d;

        /* up = silu(W_up @ x + b_up) */
        j = 0;
        while (j < hid) {
            up_out[j] = layer_b_up[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_up_rows[layer], layer_w_up_vals[layer],
                      layer_w_up_nnz[layer], layer_w_up_dim1[layer],
                      x, up_out);
        j = 0;
        while (j < hid) {
            up_out[j] = fp_silu(up_out[j]);
            j = j + 1;
        }

        /* gate = W_gate @ x + b_gate */
        j = 0;
        while (j < hid) {
            gate_out[j] = layer_b_gate[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_gate_rows[layer], layer_w_gate_vals[layer],
                      layer_w_gate_nnz[layer], layer_w_gate_dim1[layer],
                      x, gate_out);

        /* hidden = up * gate */
        j = 0;
        while (j < hid) {
            up_out[j] = fp_mul(up_out[j], gate_out[j]);
            j = j + 1;
        }

        /* down = W_down @ hidden + b_down */
        j = 0;
        while (j < d) {
            down_out[j] = layer_b_down[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_down_rows[layer], layer_w_down_vals[layer],
                      layer_w_down_nnz[layer], layer_w_down_dim1[layer],
                      up_out, down_out);

        /* residual */
        j = 0;
        while (j < d) {
            x[j] = x[j] + down_out[j];
            j = j + 1;
        }
        i = i + 1;
    }

    free(up_out); free(gate_out); free(down_out);
    return 0;
}

int ar_forward(int *tokens, int seq_len, int *logits, int *x_all) {
    int i;
    int layer;

    /* Embed all tokens */
    i = 0;
    while (i < seq_len) {
        embed_token(tokens[i], x_all + i * D_MODEL);
        i = i + 1;
    }

    /* Transformer blocks */
    layer = 0;
    while (layer < N_LAYERS) {
        ar_attention(layer, x_all, seq_len);
        ar_ffn(layer, x_all, seq_len);
        layer = layer + 1;
    }

    /* Output head: logits = head_w @ x_last + head_bias */
    i = 0;
    while (i < VOCAB_SIZE) {
        logits[i] = head_bias[i];
        i = i + 1;
    }
    sparse_matvec(head_w_rows, head_w_vals, head_w_nnz, head_w_dim1,
                  x_all + (seq_len - 1) * D_MODEL, logits);
    return 0;
}

int fp_argmax(int *logits, int n) {
    int best;
    int i;

    best = 0;
    i = 1;
    while (i < n) {
        if (logits[i] > logits[best]) best = i;
        i = i + 1;
    }
    return best;
}
"""

C4_GENERATION = """\
/* ==================== Context Builder ==================== */

int build_context(int *tokens, int max_tokens) {
    int pos;
    int i;
    int op;
    int uimm;

    pos = 0;
    tokens[pos] = TOK_CODE_START;
    pos = pos + 1;

    i = 0;
    while (i < bytecode_len && pos + 5 < max_tokens) {
        op = bytecode_ops[i];
        uimm = bytecode_imms[i];
        tokens[pos] = op & 0xFF;
        pos = pos + 1;
        tokens[pos] = uimm & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 8) & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 16) & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 24) & 0xFF;
        pos = pos + 1;
        i = i + 1;
    }

    tokens[pos] = TOK_CODE_END;
    pos = pos + 1;
    tokens[pos] = TOK_DATA_START;
    pos = pos + 1;

    i = 0;
    while (i < data_section_len && pos < max_tokens - 1) {
        tokens[pos] = data_section[i];
        pos = pos + 1;
        i = i + 1;
    }

    tokens[pos] = TOK_DATA_END;
    pos = pos + 1;

    return pos;
}

/* ==================== Register Extraction ==================== */

int extract_reg(int *tokens, int seq_len, int marker) {
    int i;
    int j;
    int limit;
    int val;

    limit = seq_len - 40;
    if (limit < 0) limit = 0;
    i = seq_len - 1;
    while (i >= limit) {
        if (tokens[i] == marker && i + 4 < seq_len) {
            val = 0;
            j = 0;
            while (j < 4) {
                val = val | ((tokens[i + 1 + j] & 0xFF) << (j * 8));
                j = j + 1;
            }
            return val;
        }
        i = i - 1;
    }
    return 0 - 1;
}

/* ==================== Generation Loop ==================== */

int generate(int *tokens, int ctx_len) {
    int seq_len;
    int max_gen;
    int prev_pc;
    int *logits;
    int *x_all;
    int step;
    int next_tok;
    int ax;

    seq_len = ctx_len;
    max_gen = 20000;
    prev_pc = 0 - 1;

    logits = malloc(VOCAB_SIZE * sizeof(int));
    x_all = malloc(MAX_SEQ_LEN * D_MODEL * sizeof(int));

    step = 0;
    while (step < max_gen && seq_len < MAX_SEQ_LEN - 1) {
        /* Full forward pass */
        ar_forward(tokens, seq_len, logits, x_all);

        /* Greedy decode */
        next_tok = fp_argmax(logits, VOCAB_SIZE);
        tokens[seq_len] = next_tok;
        seq_len = seq_len + 1;

        /* Halt detection */
        if (next_tok == TOK_HALT) {
            free(logits);
            free(x_all);
            return seq_len;
        }

        /* PUTCHAR detection */
        if (next_tok == TOK_STEP_END) {
            if (prev_pc >= 0 && prev_pc < bytecode_len) {
                if (bytecode_ops[prev_pc] == OP_PUTCHAR) {
                    ax = extract_reg(tokens, seq_len, TOK_REG_AX);
                    if (ax >= 0) {
                        printf("%c", ax & 0xFF);
                    }
                }
            }
            prev_pc = extract_reg(tokens, seq_len, TOK_REG_PC);
        }
        step = step + 1;
    }

    free(logits);
    free(x_all);
    return seq_len;
}
"""

C4_SF_RUNTIME = """\
/* ==================== Runtime Functions (Soft-Float) ==================== */

int sparse_matvec(int *rows, int *vals, int nnz,
                  int n_cols, int *in, int *out) {
    int k;
    int flat;
    int r;
    int c;

    k = 0;
    while (k < nnz) {
        flat = rows[k];
        r = flat / n_cols;
        c = flat % n_cols;
        out[r] = sf_add(out[r], sf_mul(vals[k], in[c]));
        k = k + 1;
    }
    return 0;
}

int embed_token(int token_id, int *out) {
    int i;
    int *src;

    if (token_id >= 0 && token_id < VOCAB_SIZE) {
        src = embed_data + token_id * D_MODEL;
        i = 0;
        while (i < D_MODEL) {
            out[i] = src[i];
            i = i + 1;
        }
    } else {
        i = 0;
        while (i < D_MODEL) {
            out[i] = 0;
            i = i + 1;
        }
    }
    return 0;
}

int sf_softmax1(int *scores, int n) {
    int max_val;
    int i;
    int sum;

    max_val = 0;
    i = 0;
    while (i < n) {
        if (sf_gt(scores[i], max_val)) max_val = scores[i];
        i = i + 1;
    }
    sum = sf_exp(sf_neg(max_val));
    i = 0;
    while (i < n) {
        scores[i] = sf_exp(sf_sub(scores[i], max_val));
        sum = sf_add(sum, scores[i]);
        i = i + 1;
    }
    if ((sum & 0x7FFFFFFF) == 0) sum = SF_ONE;
    i = 0;
    while (i < n) {
        scores[i] = sf_div(scores[i], sum);
        i = i + 1;
    }
    return 0;
}

int ar_attention(int layer, int *x_all, int seq_len) {
    int d;
    int H;
    int HD;
    int h;
    int i;
    int j;
    int k;
    int *slopes;
    int *Q;
    int *K;
    int *V;
    int *attn_out;
    int *scores;
    int *proj;
    int slope;
    int h_off;
    int dot;
    int s;

    d = D_MODEL;
    H = N_HEADS;
    HD = HEAD_DIM;
    slopes = layer_alibi[layer];

    Q = malloc(seq_len * d * sizeof(int));
    memset(Q, 0, seq_len * d * sizeof(int));
    K = malloc(seq_len * d * sizeof(int));
    memset(K, 0, seq_len * d * sizeof(int));
    V = malloc(seq_len * d * sizeof(int));
    memset(V, 0, seq_len * d * sizeof(int));
    attn_out = malloc(seq_len * d * sizeof(int));
    memset(attn_out, 0, seq_len * d * sizeof(int));
    scores = malloc(seq_len * sizeof(int));

    /* Q, K, V projections for all positions */
    i = 0;
    while (i < seq_len) {
        sparse_matvec(layer_wq_rows[layer], layer_wq_vals[layer],
                      layer_wq_nnz[layer], layer_wq_dim1[layer],
                      x_all + i * d, Q + i * d);
        sparse_matvec(layer_wk_rows[layer], layer_wk_vals[layer],
                      layer_wk_nnz[layer], layer_wk_dim1[layer],
                      x_all + i * d, K + i * d);
        sparse_matvec(layer_wv_rows[layer], layer_wv_vals[layer],
                      layer_wv_nnz[layer], layer_wv_dim1[layer],
                      x_all + i * d, V + i * d);
        i = i + 1;
    }

    /* Per-head attention with ALiBi */
    h = 0;
    while (h < H) {
        slope = slopes[h];
        h_off = h * HD;

        i = 0;
        while (i < seq_len) {
            /* Scores: Q[i] . K[j] for j <= i (causal) */
            j = 0;
            while (j <= i) {
                dot = 0;
                k = 0;
                while (k < HD) {
                    dot = sf_add(dot, sf_mul(Q[i * d + h_off + k], K[j * d + h_off + k]));
                    k = k + 1;
                }
                dot = sf_mul(dot, SF_ATTN_SCALE);
                /* ALiBi bias: -slope * |i - j| */
                dot = sf_sub(dot, sf_mul(slope, sf_from_int(i - j)));
                scores[j] = dot;
                j = j + 1;
            }

            sf_softmax1(scores, i + 1);

            /* Weighted sum of V */
            k = 0;
            while (k < HD) {
                s = 0;
                j = 0;
                while (j <= i) {
                    s = sf_add(s, sf_mul(scores[j], V[j * d + h_off + k]));
                    j = j + 1;
                }
                attn_out[i * d + h_off + k] = s;
                k = k + 1;
            }
            i = i + 1;
        }
        h = h + 1;
    }

    /* Output projection + residual */
    proj = malloc(d * sizeof(int));
    i = 0;
    while (i < seq_len) {
        memset(proj, 0, d * sizeof(int));
        sparse_matvec(layer_wo_rows[layer], layer_wo_vals[layer],
                      layer_wo_nnz[layer], layer_wo_dim1[layer],
                      attn_out + i * d, proj);
        j = 0;
        while (j < d) {
            x_all[i * d + j] = sf_add(x_all[i * d + j], proj[j]);
            j = j + 1;
        }
        i = i + 1;
    }

    free(Q); free(K); free(V);
    free(attn_out); free(scores); free(proj);
    return 0;
}

int ar_ffn(int layer, int *x_all, int seq_len) {
    int d;
    int hid;
    int i;
    int j;
    int *up_out;
    int *gate_out;
    int *down_out;
    int *x;

    d = D_MODEL;
    hid = HIDDEN_DIM;

    up_out = malloc(hid * sizeof(int));
    gate_out = malloc(hid * sizeof(int));
    down_out = malloc(d * sizeof(int));

    i = 0;
    while (i < seq_len) {
        x = x_all + i * d;

        /* up = silu(W_up @ x + b_up) */
        j = 0;
        while (j < hid) {
            up_out[j] = layer_b_up[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_up_rows[layer], layer_w_up_vals[layer],
                      layer_w_up_nnz[layer], layer_w_up_dim1[layer],
                      x, up_out);
        j = 0;
        while (j < hid) {
            up_out[j] = sf_silu(up_out[j]);
            j = j + 1;
        }

        /* gate = W_gate @ x + b_gate */
        j = 0;
        while (j < hid) {
            gate_out[j] = layer_b_gate[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_gate_rows[layer], layer_w_gate_vals[layer],
                      layer_w_gate_nnz[layer], layer_w_gate_dim1[layer],
                      x, gate_out);

        /* hidden = up * gate */
        j = 0;
        while (j < hid) {
            up_out[j] = sf_mul(up_out[j], gate_out[j]);
            j = j + 1;
        }

        /* down = W_down @ hidden + b_down */
        j = 0;
        while (j < d) {
            down_out[j] = layer_b_down[layer][j];
            j = j + 1;
        }
        sparse_matvec(layer_w_down_rows[layer], layer_w_down_vals[layer],
                      layer_w_down_nnz[layer], layer_w_down_dim1[layer],
                      up_out, down_out);

        /* residual */
        j = 0;
        while (j < d) {
            x[j] = sf_add(x[j], down_out[j]);
            j = j + 1;
        }
        i = i + 1;
    }

    free(up_out); free(gate_out); free(down_out);
    return 0;
}

int ar_forward(int *tokens, int seq_len, int *logits, int *x_all) {
    int i;
    int layer;

    /* Embed all tokens */
    i = 0;
    while (i < seq_len) {
        embed_token(tokens[i], x_all + i * D_MODEL);
        i = i + 1;
    }

    /* Transformer blocks */
    layer = 0;
    while (layer < N_LAYERS) {
        ar_attention(layer, x_all, seq_len);
        ar_ffn(layer, x_all, seq_len);
        layer = layer + 1;
    }

    /* Output head: logits = head_w @ x_last + head_bias */
    i = 0;
    while (i < VOCAB_SIZE) {
        logits[i] = head_bias[i];
        i = i + 1;
    }
    sparse_matvec(head_w_rows, head_w_vals, head_w_nnz, head_w_dim1,
                  x_all + (seq_len - 1) * D_MODEL, logits);
    return 0;
}

int sf_argmax(int *logits, int n) {
    int best;
    int i;

    best = 0;
    i = 1;
    while (i < n) {
        if (sf_gt(logits[i], logits[best])) best = i;
        i = i + 1;
    }
    return best;
}
"""

C4_SF_GENERATION = """\
/* ==================== Context Builder ==================== */

int build_context(int *tokens, int max_tokens) {
    int pos;
    int i;
    int op;
    int uimm;

    pos = 0;
    tokens[pos] = TOK_CODE_START;
    pos = pos + 1;

    i = 0;
    while (i < bytecode_len && pos + 5 < max_tokens) {
        op = bytecode_ops[i];
        uimm = bytecode_imms[i];
        tokens[pos] = op & 0xFF;
        pos = pos + 1;
        tokens[pos] = uimm & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 8) & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 16) & 0xFF;
        pos = pos + 1;
        tokens[pos] = (uimm >> 24) & 0xFF;
        pos = pos + 1;
        i = i + 1;
    }

    tokens[pos] = TOK_CODE_END;
    pos = pos + 1;
    tokens[pos] = TOK_DATA_START;
    pos = pos + 1;

    i = 0;
    while (i < data_section_len && pos < max_tokens - 1) {
        tokens[pos] = data_section[i];
        pos = pos + 1;
        i = i + 1;
    }

    tokens[pos] = TOK_DATA_END;
    pos = pos + 1;

    return pos;
}

/* ==================== Register Extraction ==================== */

int extract_reg(int *tokens, int seq_len, int marker) {
    int i;
    int j;
    int limit;
    int val;

    limit = seq_len - 40;
    if (limit < 0) limit = 0;
    i = seq_len - 1;
    while (i >= limit) {
        if (tokens[i] == marker && i + 4 < seq_len) {
            val = 0;
            j = 0;
            while (j < 4) {
                val = val | ((tokens[i + 1 + j] & 0xFF) << (j * 8));
                j = j + 1;
            }
            return val;
        }
        i = i - 1;
    }
    return 0 - 1;
}

/* ==================== Generation Loop ==================== */

int generate(int *tokens, int ctx_len) {
    int seq_len;
    int max_gen;
    int prev_pc;
    int *logits;
    int *x_all;
    int step;
    int next_tok;
    int ax;

    seq_len = ctx_len;
    max_gen = 20000;
    prev_pc = 0 - 1;

    logits = malloc(VOCAB_SIZE * sizeof(int));
    x_all = malloc(MAX_SEQ_LEN * D_MODEL * sizeof(int));

    step = 0;
    while (step < max_gen && seq_len < MAX_SEQ_LEN - 1) {
        /* Full forward pass */
        ar_forward(tokens, seq_len, logits, x_all);

        /* Greedy decode */
        next_tok = sf_argmax(logits, VOCAB_SIZE);
        tokens[seq_len] = next_tok;
        seq_len = seq_len + 1;

        /* Halt detection */
        if (next_tok == TOK_HALT) {
            free(logits);
            free(x_all);
            return seq_len;
        }

        /* PUTCHAR detection */
        if (next_tok == TOK_STEP_END) {
            if (prev_pc >= 0 && prev_pc < bytecode_len) {
                if (bytecode_ops[prev_pc] == OP_PUTCHAR) {
                    ax = extract_reg(tokens, seq_len, TOK_REG_AX);
                    if (ax >= 0) {
                        printf("%c", ax & 0xFF);
                    }
                }
            }
            prev_pc = extract_reg(tokens, seq_len, TOK_REG_PC);
        }
        step = step + 1;
    }

    free(logits);
    free(x_all);
    return seq_len;
}
"""


def generate_c4_globals(fp32=False):
    """Generate global variable declarations for C4."""
    lines = []
    lines.append('/* ==================== Global Variables ==================== */')
    lines.append('')

    lines.append('/* Model configuration */')
    config_vars = ['VOCAB_SIZE', 'D_MODEL', 'N_LAYERS', 'N_HEADS',
                   'HEAD_DIM', 'HIDDEN_DIM', 'MAX_SEQ_LEN']
    if not fp32:
        config_vars.append('FP_SCALE')
    for name in config_vars:
        lines.append(f'int {name};')
    lines.append('')

    lines.append('/* Token constants */')
    for name in ['TOK_SEP', 'TOK_REG_PC', 'TOK_REG_AX', 'TOK_REG_SP',
                 'TOK_REG_BP', 'TOK_MEM', 'TOK_STEP_END', 'TOK_HALT',
                 'TOK_CODE_START', 'TOK_CODE_END', 'TOK_DATA_START',
                 'TOK_DATA_END', 'OP_PUTCHAR']:
        lines.append(f'int {name};')
    lines.append('')

    lines.append('/* Embedding */')
    lines.append('int *embed_data;')
    lines.append('')

    lines.append('/* Output head */')
    lines.append('int *head_w_rows;')
    lines.append('int *head_w_vals;')
    lines.append('int *head_bias;')
    lines.append('int head_w_nnz;')
    lines.append('int head_w_dim1;')
    lines.append('')

    lines.append('/* Per-layer weight lookup tables */')
    for t in ['wq', 'wk', 'wv', 'wo', 'w_up', 'w_gate', 'w_down']:
        lines.append(f'int **layer_{t}_rows;')
        lines.append(f'int **layer_{t}_vals;')
        lines.append(f'int *layer_{t}_nnz;')
        lines.append(f'int *layer_{t}_dim1;')
    for d in ['alibi', 'b_up', 'b_gate', 'b_down']:
        lines.append(f'int **layer_{d};')
    lines.append('')

    lines.append('/* Bytecode */')
    lines.append('int *bytecode_ops;')
    lines.append('int *bytecode_imms;')
    lines.append('int bytecode_len;')
    lines.append('int *data_section;')
    lines.append('int data_section_len;')

    return '\n'.join(lines)


def generate_c4_init_config(weights, fp32=False):
    """Generate init_config() function for C4."""
    vocab_size = weights['vocab_size']
    d_model = weights['d_model']
    n_layers = weights['n_layers']
    n_heads = weights['n_heads']
    head_dim = d_model // n_heads
    hidden_dim = d_model * 4

    lines = [
        'int init_config() {',
        f'    VOCAB_SIZE = {vocab_size};',
        f'    D_MODEL = {d_model};',
        f'    N_LAYERS = {n_layers};',
        f'    N_HEADS = {n_heads};',
        f'    HEAD_DIM = {head_dim};',
        f'    HIDDEN_DIM = {hidden_dim};',
        f'    MAX_SEQ_LEN = 4096;',
    ]

    if fp32:
        lines.extend([
            f'    SF_ONE = {float_to_ieee(1.0)};',
            f'    SF_TWO = {float_to_ieee(2.0)};',
            f'    SF_HALF = {float_to_ieee(0.5)};',
            f'    SF_EIGHT = {float_to_ieee(8.0)};',
            f'    SF_NEG_EIGHT = {float_to_ieee(-8.0)};',
            f'    SF_LOG2E = {float_to_ieee(1.44269504)};',
            f'    SF_LN2 = {float_to_ieee(0.69314718)};',
            f'    SF_INV6 = {float_to_ieee(1.0/6.0)};',
            f'    SF_INV24 = {float_to_ieee(1.0/24.0)};',
            f'    SF_EXP_HI = {float_to_ieee(88.0)};',
            f'    SF_EXP_LO = {float_to_ieee(-88.0)};',
            f'    SF_ATTN_SCALE = {float_to_ieee(0.125)};',
        ])
    else:
        lines.append('    FP_SCALE = 65536;')

    lines.extend([
        '    return 0;',
        '}',
    ])

    return '\n'.join(lines)


def generate_c4_init_embed(weights, fp_convert=None):
    """Generate init_embed() function for C4."""
    lines = ['int init_embed() {']
    lines.append(emit_c4_dense_init('embed_data', weights['embed_weight'],
                                    fp_convert=fp_convert))
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_init_weights(weights, fp_convert=None):
    """Generate init_weights() for C4.

    Allocates all lookup tables, fills per-layer weights directly into them,
    and initializes head weights.
    """
    SPARSE_KEYS = [
        ('W_q', 'wq'), ('W_k', 'wk'), ('W_v', 'wv'), ('W_o', 'wo'),
        ('W_up', 'w_up'), ('W_gate', 'w_gate'), ('W_down', 'w_down'),
    ]
    n_layers = weights['n_layers']

    lines = ['int init_weights() {']

    # Allocate sparse lookup tables
    for _, tname in SPARSE_KEYS:
        lines.append(f'    layer_{tname}_rows = malloc({n_layers} * sizeof(int *));')
        lines.append(f'    layer_{tname}_vals = malloc({n_layers} * sizeof(int *));')
        lines.append(f'    layer_{tname}_nnz = malloc({n_layers} * sizeof(int));')
        lines.append(f'    layer_{tname}_dim1 = malloc({n_layers} * sizeof(int));')

    # Allocate dense lookup tables
    for dname in ['alibi', 'b_up', 'b_gate', 'b_down']:
        lines.append(f'    layer_{dname} = malloc({n_layers} * sizeof(int *));')

    # Fill per-layer weights directly into lookup tables
    for i in range(n_layers):
        layer = weights['layers'][i]
        lines.append(f'    /* Layer {i} */')

        # ALiBi slopes
        lines.append(emit_c4_dense_init(f'layer_alibi[{i}]',
                                        layer['alibi_slopes'],
                                        fp_convert=fp_convert))

        # Sparse weights
        for wkey, tname in SPARSE_KEYS:
            code, nnz, dim1 = emit_c4_sparse_init(
                f'layer_{tname}_rows[{i}]', f'layer_{tname}_vals[{i}]',
                layer[wkey], fp_convert=fp_convert)
            lines.append(code)
            lines.append(f'    layer_{tname}_nnz[{i}] = {nnz};')
            lines.append(f'    layer_{tname}_dim1[{i}] = {dim1};')

        # Dense biases
        for bkey, bname in [('b_up', 'b_up'), ('b_gate', 'b_gate'),
                            ('b_down', 'b_down')]:
            lines.append(emit_c4_dense_init(f'layer_{bname}[{i}]',
                                            layer[bkey],
                                            fp_convert=fp_convert))

    # Head
    lines.append('    /* Output head */')
    code, nnz, dim1 = emit_c4_sparse_init(
        'head_w_rows', 'head_w_vals', weights['head_weight'],
        fp_convert=fp_convert)
    lines.append(code)
    lines.append(f'    head_w_nnz = {nnz};')
    lines.append(f'    head_w_dim1 = {dim1};')
    lines.append(emit_c4_dense_init('head_bias', weights['head_bias'],
                                    fp_convert=fp_convert))

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_init_bytecode(bytecode, data):
    """Generate init_bytecode() for C4."""
    lines = ['int init_bytecode() {']

    # Bytecode ops and imms
    ops = []
    imms = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        ops.append(op)
        imms.append(imm)

    n = len(bytecode)
    lines.append(f'    bytecode_len = {n};')
    lines.append(f'    bytecode_ops = malloc({n} * sizeof(int));')
    lines.append(f'    bytecode_imms = malloc({n} * sizeof(int));')
    for i, (op, imm) in enumerate(zip(ops, imms)):
        lines.append(f'    bytecode_ops[{i}] = {op};')
        lines.append(f'    bytecode_imms[{i}] = {imm};')

    # Data section
    if data:
        data_vals = list(data) if isinstance(data, (bytes, bytearray)) else data
        n_data = len(data_vals)
        lines.append(f'    data_section_len = {n_data};')
        lines.append(f'    data_section = malloc({n_data} * sizeof(int));')
        for i, v in enumerate(data_vals):
            lines.append(f'    data_section[{i}] = {v};')
    else:
        lines.append('    data_section_len = 0;')
        lines.append('    data_section = malloc(sizeof(int));')
        lines.append('    data_section[0] = 0;')

    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_init_bytecode_quine(ops, imms, dslen, data_str=None):
    """Generate init_bytecode() with compact string-based data section.

    For the autoregressive quine, the data section is stored as a C string
    literal and copied via a loop, instead of individual byte assignments.
    This enables the quine's self-referential convergence (~1.2x expansion
    vs ~30x for per-byte assignments).

    Args:
        ops: list of opcode ints
        imms: list of immediate ints
        dslen: data section length (including null terminator)
        data_str: C-escaped string content, or None for ~ placeholder
    """
    lines = ['int init_bytecode() {']
    lines.append('    char *ds;')
    lines.append('    int i;')
    n = len(ops)
    lines.append(f'    bytecode_len = {n};')
    lines.append(f'    bytecode_ops = malloc({n} * sizeof(int));')
    lines.append(f'    bytecode_imms = malloc({n} * sizeof(int));')
    for i in range(n):
        lines.append(f'    bytecode_ops[{i}] = {ops[i]}; bytecode_imms[{i}] = {imms[i]};')
    if data_str is None:
        lines.append('    ds = ~;')
    else:
        lines.append(f'    ds = "{data_str}";')
    lines.append(f'    data_section_len = {dslen};')
    lines.append(f'    data_section = malloc({dslen} * sizeof(int));')
    lines.append('    i = 0;')
    lines.append(f'    while (i < {dslen}) {{')
    lines.append('        data_section[i] = *(ds + i);')
    lines.append('        i = i + 1;')
    lines.append('    }')
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_main():
    """Generate main() for C4."""
    lines = ['/* ==================== Main ==================== */', '']
    lines.append('int main() {')
    lines.append('    int *tokens;')
    lines.append('    int ctx_len;')
    lines.append('')
    lines.append('    init_config();')
    lines.append('    init_tokens();')
    lines.append('    init_embed();')
    lines.append('    init_weights();')
    lines.append('    init_bytecode();')
    lines.append('')
    lines.append('    tokens = malloc(MAX_SEQ_LEN * sizeof(int));')
    lines.append('    memset(tokens, 0, MAX_SEQ_LEN * sizeof(int));')
    lines.append('')
    lines.append('    ctx_len = build_context(tokens, MAX_SEQ_LEN);')
    lines.append('    printf("Context: %d tokens\\n", ctx_len);')
    lines.append('    printf("Generating (full recompute per step)...\\n");')
    lines.append('')
    lines.append('    generate(tokens, ctx_len);')
    lines.append('')
    lines.append('    free(tokens);')
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_main_quine():
    """Generate main() for C4 quine mode (no diagnostic output).

    Same as generate_c4_main but without printf diagnostics, so that
    stdout contains only the quine output (PUTCHAR characters from the
    transformer's generation loop).
    """
    lines = ['/* ==================== Main ==================== */', '']
    lines.append('int main() {')
    lines.append('    int *tokens;')
    lines.append('    int ctx_len;')
    lines.append('')
    lines.append('    init_config();')
    lines.append('    init_tokens();')
    lines.append('    init_embed();')
    lines.append('    init_weights();')
    lines.append('    init_bytecode();')
    lines.append('')
    lines.append('    tokens = malloc(MAX_SEQ_LEN * sizeof(int));')
    lines.append('    memset(tokens, 0, MAX_SEQ_LEN * sizeof(int));')
    lines.append('')
    lines.append('    ctx_len = build_context(tokens, MAX_SEQ_LEN);')
    lines.append('    generate(tokens, ctx_len);')
    lines.append('')
    lines.append('    free(tokens);')
    lines.append('    return 0;')
    lines.append('}')
    return '\n'.join(lines)


def generate_c4_file(weights, bytecode, data, fp32=False, quine=False):
    """Generate the complete C4-compatible bundled C source file."""
    fp_convert = float_to_ieee if fp32 else None

    # Sections before init_bytecode
    sections_before = [
        C4_HEADER,
        generate_c4_globals(fp32=fp32),
        C4_SOFTFLOAT if fp32 else C4_FP_MATH,
        '/* ==================== Initialization ==================== */',
        generate_c4_init_config(weights, fp32=fp32),
        C4_INIT_TOKENS,
        generate_c4_init_embed(weights, fp_convert=fp_convert),
        generate_c4_init_weights(weights, fp_convert=fp_convert),
    ]

    # Sections after init_bytecode
    sections_after = [
        C4_SF_RUNTIME if fp32 else C4_RUNTIME,
        C4_SF_GENERATION if fp32 else C4_GENERATION,
    ]

    if quine:
        sections_after.append(generate_c4_main_quine())
        part_a = '\n\n'.join(sections_before)
        part_c = '\n\n'.join(sections_after)
        return make_autoregressive_quine(part_a, part_c)
    else:
        all_sections = (sections_before +
                        [generate_c4_init_bytecode(bytecode, data)] +
                        sections_after +
                        [generate_c4_main()])
        return '\n\n'.join(all_sections)


# =========================================================================
# Autoregressive quine
# =========================================================================

# The quine C source program: compiled to C4 bytecode and run through the
# autoregressive transformer. Prints its data string char by char. When it
# encounters ~ (126), prints the C-escaped string in quotes (self-reference).
# Escapes: newline -> \n (92,110), backslash -> \\ (92,92), quote -> \" (92,34)
QUINE_SOURCE = """\
int main() {
    char *s;
    int i;
    int j;
    int c;
    s = "PLACEHOLDER";
    i = 0;
    while (*(s + i)) {
        if (*(s + i) == 126) {
            putchar(34);
            j = 0;
            while (*(s + j)) {
                c = *(s + j);
                if (c == 10) {
                    putchar(92);
                    putchar(110);
                } else {
                    if (c == 92) {
                        putchar(92);
                        putchar(92);
                    } else {
                        if (c == 34) {
                            putchar(92);
                            putchar(34);
                        } else {
                            putchar(c);
                        }
                    }
                }
                j = j + 1;
            }
            putchar(34);
        } else {
            putchar(*(s + i));
        }
        i = i + 1;
    }
    return 0;
}
"""


def c_escape(text):
    """Escape text for embedding in a C string literal.

    Handles: backslash -> \\\\, double-quote -> \\", newline -> \\n
    Order matters: backslash first to avoid double-escaping.
    """
    return (text
            .replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n'))


def make_autoregressive_quine(part_a, part_c):
    """Build autoregressive quine via fixed-point iteration.

    The quine source (a C program that prints its data string with
    self-reference at ~) is compiled to C4 bytecode. The bytecode is
    embedded in init_bytecode() with the data section stored as a
    compact string literal. When run through the transformer, each
    PUTCHAR produces one character of the full bundled file.

    Architecture:
      T = part_a + sep + init_bytecode + sep + part_c
      init_bytecode contains: ds = "c_escape(T_with_tilde)";
      T_with_tilde = T but with the ds string replaced by ~

    Convergence:
      |T_with_tilde| = |part_a| + |part_c| + |IB_fixed| + 2*digits(DSLEN) + 5
      This is constant once digits(DSLEN) stabilizes (1-2 iterations).

    Args:
        part_a: all sections before init_bytecode, joined with \\n\\n
        part_c: all sections after init_bytecode, joined with \\n\\n

    Returns:
        Complete quine C source file
    """
    import subprocess
    import tempfile

    # 1. Compile quine source with dummy string to get bytecode structure
    #    Ops and imms are invariant to string content (only data section changes).
    dummy_source = QUINE_SOURCE.replace('"PLACEHOLDER"', '"X"')
    dummy_code, _ = compile_c(dummy_source)

    ops = []
    imms = []
    for instr in dummy_code:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        ops.append(op)
        imms.append(imm)

    # 2. Fixed-point iteration on DSLEN
    #    DSLEN = len(T_with_tilde) + 1 (null terminator)
    dslen = len(part_a) + len(part_c) + 500  # initial overestimate
    for iteration in range(20):
        ib_template = generate_c4_init_bytecode_quine(ops, imms, dslen)
        t_with_tilde = part_a + '\n\n' + ib_template + '\n\n' + part_c
        new_dslen = len(t_with_tilde) + 1
        if new_dslen == dslen:
            break
        dslen = new_dslen
    else:
        raise RuntimeError("DSLEN did not converge after 20 iterations")

    # 3. Verify ~ appears exactly once in T_with_tilde
    tilde_count = t_with_tilde.count('~')
    if tilde_count != 1:
        raise RuntimeError(f"Expected 1 ~ in T_with_tilde, found {tilde_count}")

    # 4. Escape T_with_tilde for C string literal
    escaped = c_escape(t_with_tilde)

    # 5. Build final init_bytecode with the escaped string
    ib_final = generate_c4_init_bytecode_quine(ops, imms, dslen, data_str=escaped)

    # 6. Assemble final T
    t_final = part_a + '\n\n' + ib_final + '\n\n' + part_c

    # 7. Verify: compile quine source with actual string, check consistency
    actual_source = QUINE_SOURCE.replace('"PLACEHOLDER"', '"' + escaped + '"')
    actual_code, actual_data = compile_c(actual_source)

    actual_ops = [instr & 0xFF for instr in actual_code]
    actual_imms = []
    for instr in actual_code:
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        actual_imms.append(imm)

    if ops != actual_ops:
        raise RuntimeError("Bytecode ops changed with actual string content")
    if imms != actual_imms:
        raise RuntimeError("Bytecode imms changed with actual string content")

    expected_data = [ord(c) for c in t_with_tilde] + [0]
    actual_list = list(actual_data)
    # C4 compiler pads data section to 8-byte alignment; trim padding zeros
    if len(actual_list) > len(expected_data):
        padding = actual_list[len(expected_data):]
        if all(b == 0 for b in padding):
            actual_list = actual_list[:len(expected_data)]
    if actual_list != expected_data:
        # Find first difference for debugging
        for idx, (a, b) in enumerate(zip(actual_list, expected_data)):
            if a != b:
                raise RuntimeError(
                    f"Data section mismatch at byte {idx}: "
                    f"got {a} ({chr(a) if 32 <= a < 127 else '?'}), "
                    f"expected {b} ({chr(b) if 32 <= b < 127 else '?'})")
        raise RuntimeError(
            f"Data section length mismatch: expected {len(expected_data)} bytes, "
            f"got {len(actual_list)} bytes")

    print(f"  Autoregressive quine assembled:")
    print(f"    T_with_tilde: {len(t_with_tilde):,} chars")
    print(f"    Escaped string: {len(escaped):,} chars")
    print(f"    Final size: {len(t_final):,} chars")
    print(f"    Bytecode: {len(ops)} instructions")
    print(f"    Data section: {dslen:,} bytes")
    print(f"    DSLEN converged in {iteration + 1} iteration(s)")

    # 8. Optional: verify quine property by compiling+running quine source
    #    Add #include <stdio.h> for gcc (C4 compiler ignores includes)
    try:
        gcc_source = '#include <stdio.h>\n' + actual_source
        with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
            f.write(gcc_source)
            qs_path = f.name
        exe_path = qs_path.replace('.c', '')
        result = subprocess.run(['gcc', '-O2', '-o', exe_path, qs_path],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            result = subprocess.run([exe_path], capture_output=True, text=True,
                                    timeout=120)
            if result.stdout == t_final:
                print("    Quine property VERIFIED (gcc compile + run)")
            else:
                print(f"    Quine MISMATCH: output {len(result.stdout):,} chars, "
                      f"expected {len(t_final):,}")
                for i, (a, b) in enumerate(zip(result.stdout, t_final)):
                    if a != b:
                        print(f"      First diff at char {i}: "
                              f"got {repr(a)}, expected {repr(b)}")
                        break
                if len(result.stdout) != len(t_final):
                    print(f"      Length diff: {len(result.stdout)} vs {len(t_final)}")
        else:
            print(f"    Could not verify: gcc compilation failed")
            if result.stderr:
                print(f"      {result.stderr[:200]}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("    Could not verify: gcc not available or timeout")
    finally:
        for p in [qs_path, exe_path]:
            try:
                os.unlink(p)
            except OSError:
                pass

    return t_final


# =========================================================================
# Main bundler
# =========================================================================

def generate_c_file(weights, bytecode, data):
    """Generate the complete bundled C source file."""
    vocab_size = weights['vocab_size']
    d_model = weights['d_model']
    n_layers = weights['n_layers']
    n_heads = weights['n_heads']
    head_dim = d_model // n_heads
    hidden_dim = d_model * 4

    config = (
        f'/* ==================== Model Configuration ==================== */\n'
        f'#define VOCAB_SIZE  {vocab_size}\n'
        f'#define D_MODEL     {d_model}\n'
        f'#define N_LAYERS    {n_layers}\n'
        f'#define N_HEADS     {n_heads}\n'
        f'#define HEAD_DIM    {head_dim}\n'
        f'#define HIDDEN_DIM  {hidden_dim}\n'
        f'#define MAX_SEQ_LEN 4096\n'
        f'#define FP_SCALE    65536'
    )

    weight_arrays, weight_tables = generate_weights_and_tables(weights)
    bc_code = generate_bytecode_arrays(bytecode, data)

    sections = [
        C_HEADER,
        config,
        C_TOKEN_DEFINES,
        C_FP_MATH,
        '/* ==================== Model Weights ==================== */\n',
        weight_arrays,
        '',
        weight_tables,
        '',
        '/* ==================== Bytecode & Data ==================== */\n',
        bc_code,
        C_RUNTIME,
        C_GENERATION,
    ]

    return '\n\n'.join(sections)


def bundle(model_path, program_path, output_path, c4=False, fp32=False,
           quine=False):
    """Bundle model + program into a standalone C file.

    Args:
        model_path: Path to .arvm model file
        program_path: Path to C source file
        output_path: Path for generated C file
        c4: If True, generate C4-compatible output
        fp32: If True, use soft-float (IEEE 754) instead of fixed-point
        quine: If True, make the output a quine (prints its own source)

    Returns:
        output_path
    """
    if fp32 and not c4:
        print("Warning: --fp32 only applies to --c4 mode, ignoring")
        fp32 = False

    if quine and not c4:
        print("Warning: --quine only applies to --c4 mode, ignoring")
        quine = False

    # Load model
    print(f"Loading model: {model_path}")
    weights = load_arvm(model_path)
    print(f"  vocab_size={weights['vocab_size']}, d_model={weights['d_model']}, "
          f"n_layers={weights['n_layers']}, n_heads={weights['n_heads']}")

    # Count total nonzero weights
    total_nnz = 0
    total_params = 0
    for layer in weights['layers']:
        for key in ['W_q', 'W_k', 'W_v', 'W_o', 'W_up', 'W_gate', 'W_down']:
            w = layer[key]
            total_nnz += np.count_nonzero(w)
            total_params += w.size
    embed_nnz = np.count_nonzero(weights['embed_weight'])
    head_nnz = np.count_nonzero(weights['head_weight'])
    total_nnz += embed_nnz + head_nnz
    total_params += weights['embed_weight'].size + weights['head_weight'].size
    print(f"  Nonzero weights: {total_nnz:,} / {total_params:,} "
          f"({100 * total_nnz / max(total_params, 1):.1f}%)")

    # Compile program
    print(f"Compiling: {program_path}")
    with open(program_path) as f:
        source = f.read()
    code, data = compile_c(source)
    print(f"  {len(code)} instructions, {len(data)} data bytes")

    # Context size estimate
    ctx_tokens = 2 + len(code) * 5 + 2 + len(data)
    print(f"  Context prefix: {ctx_tokens} tokens")

    # Generate C file
    if c4:
        mode_parts = ["C4-compatible"]
        if fp32:
            mode_parts.append("soft-float fp32")
        if quine:
            mode_parts.append("quine")
        print(f"Generating {' '.join(mode_parts)} bundled C file...")
        c_code = generate_c4_file(weights, code, data, fp32=fp32, quine=quine)
    else:
        print("Generating bundled C file...")
        c_code = generate_c_file(weights, code, data)

    # Write output
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(c_code)

    print(f"Written: {output_path} ({len(c_code):,} bytes)")
    print(f"\nTo compile and run:")
    print(f"  gcc -O2 -o quine_ar {output_path}")
    print(f"  ./quine_ar")
    if c4:
        print(f"  (C4-compatible: ./c4 {output_path})")
    if quine:
        print(f"\nVerify quine property:")
        print(f"  ./quine_ar > /tmp/quine_out.c && diff {output_path} /tmp/quine_out.c")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Bundle autoregressive VM model + quine into standalone C file')
    parser.add_argument('--model', required=True, help='.arvm model file')
    parser.add_argument('--program', required=True, help='C source file to compile')
    parser.add_argument('--output', '-o', required=True, help='Output C file')
    parser.add_argument('--c4', action='store_true',
                        help='Generate C4-compatible output')
    parser.add_argument('--fp32', action='store_true',
                        help='Use soft-float (IEEE 754 fp32) instead of fixed-point')
    parser.add_argument('--quine', action='store_true',
                        help='Make the output a quine (prints its own source)')
    args = parser.parse_args()
    bundle(args.model, args.program, args.output, c4=args.c4, fp32=args.fp32,
           quine=args.quine)


if __name__ == '__main__':
    main()
