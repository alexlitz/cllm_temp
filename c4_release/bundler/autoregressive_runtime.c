/*
 * Autoregressive Neural VM Runtime
 *
 * Self-contained C runtime for the decoder-only transformer VM.
 * Hardcodes the transformer forward pass (embed -> attention+FFN x N -> head -> argmax).
 * Full recomputation per generation step (no KV cache).
 *
 * Expects these globals from the bundler:
 *   extern char embedded_model[];
 *   extern int embedded_model_len;
 *   extern int program_code[][2];   // {op, imm} pairs
 *   extern int program_code_len;
 *   extern char program_data[];
 *   extern int program_data_len;
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* Part 1: Token constants + model struct                                    */
/* ========================================================================= */

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
#define TOK_STACK0     268

#define MAX_SEQ_LEN    4096
#define TOKENS_PER_STEP 39

/* Opcodes (must match neural_vm/embedding.py) */
#define OP_IMM      1
#define OP_EXIT     38
#define OP_GETCHAR  64
#define OP_PUTCHAR  65

typedef struct {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int head_dim;
    int hidden_dim;   /* FFN hidden size (from header) */

    float *embed_weight;   /* [vocab_size * d_model] */

    /* Per-layer weights (n_layers of each) */
    float **alibi_slopes;  /* [n_layers][n_heads] */
    float **W_q;           /* [n_layers][d_model * d_model] */
    float **W_k;           /* [n_layers][d_model * d_model] */
    float **W_v;           /* [n_layers][d_model * d_model] */
    float **W_o;           /* [n_layers][d_model * d_model] */
    float **W_up;          /* [n_layers][hidden * d_model] */
    float **b_up;          /* [n_layers][hidden] */
    float **W_gate;        /* [n_layers][hidden * d_model] */
    float **b_gate;        /* [n_layers][hidden] */
    float **W_down;        /* [n_layers][d_model * hidden] */
    float **b_down;        /* [n_layers][d_model] */

    float *head_weight;    /* [vocab_size * d_model] */
    float *head_bias;      /* [vocab_size] */
} Model;

/* ========================================================================= */
/* Part 2: Math helpers (softmax1, silu)                                     */
/* ========================================================================= */

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax1(float *scores, int n) {
    /*
     * softmax with +1 anchor:
     *   out[i] = exp(x[i] - max) / (exp(0 - max) + sum(exp(x[j] - max)))
     *
     * The exp(0 - max) is the "+1" anchor term from kv_cache_eviction.py:softmax1().
     * When all scores are very negative, output approaches 0 (ZFOD semantics).
     */
    float max_val = 0.0f;  /* anchor value */
    int i;
    float sum, anchor_exp;

    for (i = 0; i < n; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    sum = 0.0f;
    anchor_exp = expf(0.0f - max_val);  /* exp(anchor - max) */
    for (i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    sum += anchor_exp;

    for (i = 0; i < n; i++) {
        scores[i] /= sum;
    }
}

/* ========================================================================= */
/* Part 3: Model loader (parse .arvm from byte buffer)                       */
/* ========================================================================= */

static unsigned int read_u32(const char **p) {
    unsigned int val;
    memcpy(&val, *p, 4);
    *p += 4;
    return val;
}

static float read_f32(const char **p) {
    float val;
    memcpy(&val, *p, 4);
    *p += 4;
    return val;
}

static float *read_tensor(const char **p, int expected_count) {
    unsigned int storage_type = read_u32(p);
    float *data;
    int i;

    if (storage_type == 0) {
        /* Dense */
        unsigned int count = read_u32(p);
        data = (float *)calloc(expected_count, sizeof(float));
        memcpy(data, *p, count * sizeof(float));
        *p += count * 4;
    } else {
        /* Sparse COO */
        unsigned int nnz = read_u32(p);
        data = (float *)calloc(expected_count, sizeof(float));
        for (i = 0; i < (int)nnz; i++) {
            unsigned int idx = read_u32(p);
            float val = read_f32(p);
            if ((int)idx < expected_count) {
                data[idx] = val;
            }
        }
    }
    return data;
}

static Model *load_model(const char *buf, int buf_len) {
    const char *p = buf;
    unsigned int magic, version;
    int i;
    Model *m;

    (void)buf_len;

    magic = read_u32(&p);
    if (magic != 0x4D565241) {
        fprintf(stderr, "Bad ARVM magic: 0x%08X\n", magic);
        return NULL;
    }
    version = read_u32(&p);
    if (version != 1 && version != 2) {
        fprintf(stderr, "Unsupported ARVM version: %u\n", version);
        return NULL;
    }

    m = (Model *)calloc(1, sizeof(Model));
    m->vocab_size = (int)read_u32(&p);
    m->d_model    = (int)read_u32(&p);
    m->n_layers   = (int)read_u32(&p);
    m->n_heads    = (int)read_u32(&p);
    m->head_dim   = m->d_model / m->n_heads;
    if (version >= 2) {
        m->hidden_dim = (int)read_u32(&p);
    } else {
        m->hidden_dim = m->d_model * 4;
    }

    /* embed.weight */
    m->embed_weight = read_tensor(&p, m->vocab_size * m->d_model);

    /* Allocate per-layer arrays */
    m->alibi_slopes = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_q    = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_k    = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_v    = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_o    = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_up   = (float **)calloc(m->n_layers, sizeof(float *));
    m->b_up   = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_gate = (float **)calloc(m->n_layers, sizeof(float *));
    m->b_gate = (float **)calloc(m->n_layers, sizeof(float *));
    m->W_down = (float **)calloc(m->n_layers, sizeof(float *));
    m->b_down = (float **)calloc(m->n_layers, sizeof(float *));

    for (i = 0; i < m->n_layers; i++) {
        int dd = m->d_model * m->d_model;
        int hd = m->hidden_dim * m->d_model;

        m->alibi_slopes[i] = read_tensor(&p, m->n_heads);
        m->W_q[i]    = read_tensor(&p, dd);
        m->W_k[i]    = read_tensor(&p, dd);
        m->W_v[i]    = read_tensor(&p, dd);
        m->W_o[i]    = read_tensor(&p, dd);
        m->W_up[i]   = read_tensor(&p, hd);
        m->b_up[i]   = read_tensor(&p, m->hidden_dim);
        m->W_gate[i] = read_tensor(&p, hd);
        m->b_gate[i] = read_tensor(&p, m->hidden_dim);
        m->W_down[i] = read_tensor(&p, hd);
        m->b_down[i] = read_tensor(&p, m->d_model);
    }

    /* head */
    m->head_weight = read_tensor(&p, m->vocab_size * m->d_model);
    m->head_bias   = read_tensor(&p, m->vocab_size);

    return m;
}

static void free_model(Model *m) {
    int i;
    if (!m) return;
    free(m->embed_weight);
    for (i = 0; i < m->n_layers; i++) {
        free(m->alibi_slopes[i]);
        free(m->W_q[i]); free(m->W_k[i]);
        free(m->W_v[i]); free(m->W_o[i]);
        free(m->W_up[i]); free(m->b_up[i]);
        free(m->W_gate[i]); free(m->b_gate[i]);
        free(m->W_down[i]); free(m->b_down[i]);
    }
    free(m->alibi_slopes);
    free(m->W_q); free(m->W_k);
    free(m->W_v); free(m->W_o);
    free(m->W_up); free(m->b_up);
    free(m->W_gate); free(m->b_gate);
    free(m->W_down); free(m->b_down);
    free(m->head_weight);
    free(m->head_bias);
    free(m);
}

/* ========================================================================= */
/* Part 4: Forward pass                                                      */
/* ========================================================================= */

/*
 * embed_token: token_id -> float[d_model]
 * Copies one row of the embedding table.
 */
static void embed_token(const Model *m, int token_id, float *out) {
    int d = m->d_model;
    if (token_id >= 0 && token_id < m->vocab_size) {
        memcpy(out, m->embed_weight + token_id * d, d * sizeof(float));
    } else {
        memset(out, 0, d * sizeof(float));
    }
}

/*
 * linear: out[i] = sum_j in[j] * W[i*in_dim + j]  (+ optional bias)
 * Matches PyTorch F.linear(x, W, b) where W is [out_dim, in_dim].
 */
static void linear(const float *in, const float *W, const float *bias,
                    int in_dim, int out_dim, float *out) {
    int i, j;
    for (i = 0; i < out_dim; i++) {
        float sum = bias ? bias[i] : 0.0f;
        const float *row = W + i * in_dim;
        for (j = 0; j < in_dim; j++) {
            sum += in[j] * row[j];
        }
        out[i] = sum;
    }
}

/*
 * attention: Multi-head attention with softmax1 + ALiBi + causal mask.
 *
 * x_all: [seq_len * d_model] - hidden states for ALL positions
 * This function computes attention for ALL positions and writes results
 * back into x_all (with residual connection).
 */
static void attention(const Model *m, int layer, float *x_all, int seq_len) {
    int d = m->d_model;
    int H = m->n_heads;
    int HD = m->head_dim;
    int h, i, j, k;
    float scale = 1.0f / sqrtf((float)HD);
    float *slopes = m->alibi_slopes[layer];

    /* Allocate Q, K, V for all positions */
    float *Q = (float *)malloc(seq_len * d * sizeof(float));
    float *K = (float *)malloc(seq_len * d * sizeof(float));
    float *V = (float *)malloc(seq_len * d * sizeof(float));
    float *attn_out = (float *)calloc(seq_len * d, sizeof(float));
    float *scores = (float *)malloc(seq_len * sizeof(float));

    /* Compute Q, K, V for all positions */
    for (i = 0; i < seq_len; i++) {
        linear(x_all + i * d, m->W_q[layer], NULL, d, d, Q + i * d);
        linear(x_all + i * d, m->W_k[layer], NULL, d, d, K + i * d);
        linear(x_all + i * d, m->W_v[layer], NULL, d, d, V + i * d);
    }

    /* Per-head attention */
    for (h = 0; h < H; h++) {
        float slope = slopes[h];
        int h_off = h * HD;

        for (i = 0; i < seq_len; i++) {
            /* Compute scores for position i attending to all j <= i */
            for (j = 0; j <= i; j++) {
                float dot = 0.0f;
                for (k = 0; k < HD; k++) {
                    dot += Q[i * d + h_off + k] * K[j * d + h_off + k];
                }
                dot *= scale;
                /* ALiBi bias: -slope * |i - j| */
                dot -= slope * (float)(i - j);
                scores[j] = dot;
            }

            /* softmax1 over [0..i] */
            softmax1(scores, i + 1);

            /* Weighted sum of V */
            for (k = 0; k < HD; k++) {
                float sum = 0.0f;
                for (j = 0; j <= i; j++) {
                    sum += scores[j] * V[j * d + h_off + k];
                }
                attn_out[i * d + h_off + k] = sum;
            }
        }
    }

    /* Output projection + residual: x_all[i] += W_o @ attn_out[i] */
    for (i = 0; i < seq_len; i++) {
        float *proj = (float *)malloc(d * sizeof(float));
        linear(attn_out + i * d, m->W_o[layer], NULL, d, d, proj);
        for (j = 0; j < d; j++) {
            x_all[i * d + j] += proj[j];
        }
        free(proj);
    }

    free(Q); free(K); free(V);
    free(attn_out); free(scores);
}

/*
 * ffn: SwiGLU feed-forward network with residual.
 *   x += W_down @ (silu(W_up @ x + b_up) * (W_gate @ x + b_gate)) + b_down
 *
 * Processes all positions.
 */
static void ffn(const Model *m, int layer, float *x_all, int seq_len) {
    int d = m->d_model;
    int hid = m->hidden_dim;
    int i, j;

    float *up_out   = (float *)malloc(hid * sizeof(float));
    float *gate_out = (float *)malloc(hid * sizeof(float));
    float *down_out = (float *)malloc(d * sizeof(float));

    for (i = 0; i < seq_len; i++) {
        float *x = x_all + i * d;

        /* up = silu(W_up @ x + b_up) */
        linear(x, m->W_up[layer], m->b_up[layer], d, hid, up_out);
        for (j = 0; j < hid; j++) {
            up_out[j] = silu(up_out[j]);
        }

        /* gate = W_gate @ x + b_gate */
        linear(x, m->W_gate[layer], m->b_gate[layer], d, hid, gate_out);

        /* hidden = up * gate */
        for (j = 0; j < hid; j++) {
            up_out[j] *= gate_out[j];
        }

        /* down = W_down @ hidden + b_down */
        linear(up_out, m->W_down[layer], m->b_down[layer], hid, d, down_out);

        /* residual */
        for (j = 0; j < d; j++) {
            x[j] += down_out[j];
        }
    }

    free(up_out); free(gate_out); free(down_out);
}

/*
 * forward: Full transformer forward pass.
 *   tokens[0..seq_len-1] -> logits[vocab_size]
 *
 * Returns logits for the LAST position only (for generation).
 * x_all holds hidden states for all positions (needed for attention).
 */
static void forward(const Model *m, const int *tokens, int seq_len,
                    float *logits, float *x_all) {
    int d = m->d_model;
    int i, layer;

    /* Embed all tokens */
    for (i = 0; i < seq_len; i++) {
        embed_token(m, tokens[i], x_all + i * d);
    }

    /* Transformer blocks */
    for (layer = 0; layer < m->n_layers; layer++) {
        attention(m, layer, x_all, seq_len);
        ffn(m, layer, x_all, seq_len);
    }

    /* Output head: logits = head_weight @ x_last + head_bias */
    linear(x_all + (seq_len - 1) * d, m->head_weight, m->head_bias,
           d, m->vocab_size, logits);
}

/* ========================================================================= */
/* Part 5: Context builder + generation loop                                 */
/* ========================================================================= */

/*
 * Build context tokens from program_code, program_data, and argv.
 * Format (matches run_vm.py:_build_context):
 *   [CODE_START] op0 imm0_b0 imm0_b1 imm0_b2 imm0_b3 op1 ... [CODE_END]
 *   [DATA_START] data_bytes... [DATA_END]
 *   argv0_chars... \0 argv1_chars... \0
 *
 * Returns number of context tokens written.
 */
static int build_context(int *tokens, int max_tokens, int argc, char **argv) {
    int pos = 0;
    int i, j;

    tokens[pos++] = TOK_CODE_START;

    for (i = 0; i < program_code_len && pos + 5 < max_tokens; i++) {
        int op  = program_code[i][0];
        int imm = program_code[i][1];
        tokens[pos++] = op & 0xFF;
        tokens[pos++] = (imm)       & 0xFF;
        tokens[pos++] = (imm >> 8)  & 0xFF;
        tokens[pos++] = (imm >> 16) & 0xFF;
        tokens[pos++] = (imm >> 24) & 0xFF;
    }

    tokens[pos++] = TOK_CODE_END;
    tokens[pos++] = TOK_DATA_START;

    for (i = 0; i < program_data_len && pos < max_tokens - 1; i++) {
        tokens[pos++] = ((unsigned char *)program_data)[i];
    }

    tokens[pos++] = TOK_DATA_END;

    /* Argv: each arg as byte tokens followed by null terminator */
    for (i = 0; i < argc && pos < max_tokens - 1; i++) {
        for (j = 0; argv[i][j] && pos < max_tokens - 1; j++) {
            tokens[pos++] = (unsigned char)argv[i][j];
        }
        if (pos < max_tokens) {
            tokens[pos++] = 0;  /* null terminator */
        }
    }

    return pos;
}

/*
 * extract_step_ax: Read AX register value from the last completed step.
 * Scans backward from step_end_pos to find REG_AX, reads 4 LE bytes.
 * Returns the 32-bit value, or -1 if not found.
 */
static int extract_step_ax(const int *tokens, int step_end_pos) {
    int i;
    int search_start = step_end_pos - TOKENS_PER_STEP;
    if (search_start < 0) search_start = 0;

    for (i = step_end_pos - 1; i >= search_start; i--) {
        if (tokens[i] == TOK_REG_AX && i + 4 <= step_end_pos) {
            unsigned int val = 0;
            int j;
            for (j = 0; j < 4; j++) {
                val |= (unsigned int)(tokens[i + 1 + j] & 0xFF) << (j * 8);
            }
            return (int)val;
        }
    }
    return -1;
}

/*
 * extract_step_pc: Read PC register value from the last completed step.
 * Scans backward from step_end_pos to find REG_PC, reads 4 LE bytes.
 * The PC value indexes into the bytecode section of the context, allowing
 * us to determine which opcode was executed.
 */
static int extract_step_pc(const int *tokens, int step_end_pos) {
    int i;
    int search_start = step_end_pos - TOKENS_PER_STEP;
    if (search_start < 0) search_start = 0;

    for (i = step_end_pos - 1; i >= search_start; i--) {
        if (tokens[i] == TOK_REG_PC && i + 4 <= step_end_pos) {
            unsigned int val = 0;
            int j;
            for (j = 0; j < 4; j++) {
                val |= (unsigned int)(tokens[i + 1 + j] & 0xFF) << (j * 8);
            }
            return (int)val;
        }
    }
    return -1;
}

/*
 * get_opcode_at_pc: Given a PC value (byte offset into the bytecode section
 * of the context), return the opcode at that position.
 *
 * The context format is: [CODE_START] op0 imm0_b0..b3 op1 imm1_b0..b3 ...
 * Each instruction is 5 tokens. PC value N maps to context position 1 + N
 * (skipping CODE_START). The opcode is at that position.
 *
 * PC values from bake_v2 are the token offset from CODE_START:
 *   instruction 0: PC=2 (CODE_START + op + imm_b0 → op at offset 1)
 *   instruction 1: PC=7 (op at offset 6)
 *   etc.
 *
 * Actually PC points to the first immediate byte, so the opcode is at PC-1:
 *   PC=2 → opcode at context[1] (tokens[1] after CODE_START)
 *
 * For now, look up in program_code directly. PC value N maps to instruction
 * index (N-2)/5 when N >= 2.
 */
static int get_opcode_for_pc(int pc_val) {
    int instr_idx;
    if (pc_val < 2) return -1;
    instr_idx = (pc_val - 2) / 5;
    if (instr_idx < 0 || instr_idx >= program_code_len) return -1;
    return program_code[instr_idx][0];
}

/*
 * generate: Run autoregressive generation loop with I/O handling.
 *
 * At each step boundary (STEP_END), checks the executed opcode via PC:
 *   - PUTCHAR: extract AX low byte, write to stdout
 *   - GETCHAR: would need to inject stdin byte into next step's AX
 *     (requires model support — placeholder for now)
 *
 * Returns total sequence length (tokens array is filled in-place).
 */
static int generate(const Model *m, int *tokens, int ctx_len) {
    int seq_len = ctx_len;
    int max_tokens = 100000 * TOKENS_PER_STEP;
    int step;
    float *logits;
    float *x_all;

    logits = (float *)malloc(m->vocab_size * sizeof(float));
    x_all  = (float *)malloc(MAX_SEQ_LEN * m->d_model * sizeof(float));

    for (step = 0; step < max_tokens && seq_len < MAX_SEQ_LEN - 1; step++) {
        int best_tok = 0;
        float best_val;
        int i;

        /* Full forward pass */
        forward(m, tokens, seq_len, logits, x_all);

        /* Argmax */
        best_val = logits[0];
        for (i = 1; i < m->vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_tok = i;
            }
        }

        tokens[seq_len++] = best_tok;

        /* Halt */
        if (best_tok == TOK_HALT) {
            break;
        }

        /* Step boundary: handle I/O */
        if (best_tok == TOK_STEP_END) {
            int pc_val = extract_step_pc(tokens, seq_len);
            int opcode = get_opcode_for_pc(pc_val);

            if (opcode == OP_PUTCHAR) {
                int ax_val = extract_step_ax(tokens, seq_len);
                if (ax_val >= 0) {
                    putchar(ax_val & 0xFF);
                    fflush(stdout);
                }
            }
            /* GETCHAR: when baked, the model will need the runtime to inject
             * an input byte. The mechanism would be:
             *   1. Model generates a step with GETCHAR opcode
             *   2. Runtime reads a byte from stdin via getchar()
             *   3. Runtime injects the byte value into the context
             *      (e.g., by appending special input tokens before
             *       the model generates the next step's AX value)
             *
             * This requires model-side support (special attention pattern
             * that reads from an input region of the context). Placeholder
             * until GETCHAR is baked into the transformer weights.
             */
        }
    }

    free(logits);
    free(x_all);

    return seq_len;
}

/* ========================================================================= */
/* Part 6: Exit code extraction + main                                       */
/* ========================================================================= */

static int decode_exit_code(const int *tokens, int seq_len) {
    int i;
    for (i = seq_len - 1; i >= 0; i--) {
        if (tokens[i] == TOK_REG_AX && i + 4 < seq_len) {
            unsigned int val = 0;
            int j;
            for (j = 0; j < 4; j++) {
                val |= (unsigned int)(tokens[i + 1 + j] & 0xFF) << (j * 8);
            }
            return (int)(val & 0xFF);  /* return low byte as process exit code */
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    Model *m;
    int *tokens;
    int ctx_len, seq_len;
    int exit_code;

    /* Load model from embedded bytes */
    m = load_model(embedded_model, embedded_model_len);
    if (!m) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    /* Allocate token buffer */
    tokens = (int *)calloc(MAX_SEQ_LEN, sizeof(int));

    /*
     * Build context from bytecode + data.
     * Argv is passed only when the bundler was invoked with --argv,
     * which patches the bytecode to read argv from the data section.
     * For programs that need argv in the context (model reads it via
     * attention), pass argc/argv here. Currently no-op (0 args).
     */
#ifdef ARVM_USE_ARGV
    ctx_len = build_context(tokens, MAX_SEQ_LEN, argc, argv);
#else
    (void)argc; (void)argv;
    ctx_len = build_context(tokens, MAX_SEQ_LEN, 0, NULL);
#endif

    /* Run generation */
    seq_len = generate(m, tokens, ctx_len);

    /* Extract exit code */
    exit_code = decode_exit_code(tokens, seq_len);

    free(tokens);
    free_model(m);

    return exit_code;
}
