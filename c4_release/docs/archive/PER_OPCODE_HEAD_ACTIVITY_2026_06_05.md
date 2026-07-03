# Per-Opcode Attention Head Activity Audit (2026-06-05)

This audit inspects every (block, head) attention cell in the compiled VM and
classifies each cell as DEAD or LIVE for each of the 34 canonical opcodes.
Dead-per-opcode cells are candidates for runtime head masking (skip the head's
Q/K/V/O matmuls when the cell is dead for the current step's opcode).

## Methodology

Compiled the full VM with `compile_full_vm(disk_cache=True)` and inspected the
baked attention matrices on every block:

- **n_blocks=30** (after `compile_full_vm` expansions; nominal L0..L17 plus
  inserted post-op blocks), **N_HEADS=8** per block, **head_dim=91** (D_MODEL=728).
- Total cells inspected: **244** (some blocks have fewer than 8 heads after
  `compact_dims` pruning).

For each (block b, head h) the audit computes from `block.attn.W_q/W_k/W_v/W_o`:

- `in_dims(b, h)`: dim names whose column has at least one non-zero entry in
  rows `[h*HD : (h+1)*HD]` of W_q or W_k or W_v.
- `out_dims(b, h)`: dim names whose row has at least one non-zero entry in
  columns `[h*HD : (h+1)*HD]` of W_o.
- `op_gates(b, h)`: subset of `in_dims` whose name matches the canonical
  `OP_*` prefix (and excludes derived relay dims like `OP_LI_RELAY`). When
  non-empty, the head only fires when at least one of those opcodes' one-hot
  is set; reading multiple OP_* dims acts as a disjunction under softmax-1.

Classification rules for opcode X:

1. **Always-dead** (DEAD for every X) — heads with empty `in_dims` (no Q/K/V
   activity) OR empty `out_dims` (no O scatter).
2. **Opcode-agnostic LIVE** — heads with non-empty `in_dims`/`out_dims` and
   empty `op_gates`. They fire on every step regardless of opcode. LIVE for
   all X.
3. **Opcode-gated** — heads with non-empty `op_gates`. LIVE for X iff
   `OP_X ∈ op_gates(b, h)`; DEAD for X otherwise.

## Headline Counts

| Metric                                     | Count |
| ------------------------------------------ | ----: |
| Total (block, head) cells inspected        |   244 |
| Zero-input heads (no Q/K/V activity)       |   168 |
| No-output heads (no W_o scatter)           |   169 |
| Active heads (in + out both non-empty)     |    75 |
| Opcode-agnostic active heads (always live) |    50 |
| Opcode-gated active heads                  |    25 |
| **Total DEAD (opcode, block, head) cells** | **6,538** |

Distribution across the 34 canonical opcodes: 184 (OP_JSR) to 194 (OP_NOP,
OP_PUTCHAR, OP_GETCHAR) DEAD cells per opcode. The variation lives entirely
inside the 25 opcode-gated heads — every other head is either always-dead or
always-live.

## Dead Heads per Block (Always-Dead Cluster)

Blocks 9, 11, 13–23, 25, 27–29 have **all 8 heads** zero-input — i.e., the
compiler placed FFN-only work there, and the attention modules are
weight-stripped. They contribute attention compute but zero useful output.
This is the dominant source of "dead heads".

Partially-dead blocks:

- block 1: heads {5, 6, 7} dead-input (3/8)
- block 2: heads {1..7} dead-input (7/8)
- block 4: heads {4, 5, 6, 7} dead-input (4/8)
- block 8: heads {6, 7} dead-input (2/8)
- block 10: heads {2..7} dead-input (6/8)
- block 12: heads {5, 6, 7} dead-input (3/8)
- block 24: heads {3..7} dead-input (5/8)

## Opcode-Gated Heads (the 25 cells that matter for per-opcode masking)

| Block | Head | OP_* gates                                                                                                                                                                                                                                                  |
| ----: | ---: | --- |
|     5 |    6 | OP_ENT                                                                                                                                                                                                                                                      |
|     6 |    0 | OP_JMP                                                                                                                                                                                                                                                      |
|     6 |    1 | OP_EXIT                                                                                                                                                                                                                                                     |
|     6 |    2 | OP_JMP                                                                                                                                                                                                                                                      |
|     6 |    3 | OP_JSR                                                                                                                                                                                                                                                      |
|     6 |    4 | OP_BZ, OP_BNZ                                                                                                                                                                                                                                               |
|     6 |    6 | OP_ADD, OP_ADJ, OP_AND, OP_DIV, OP_ENT, OP_EQ, OP_GE, OP_GT, OP_JSR, OP_LE, OP_LEV, OP_LT, OP_MOD, OP_MUL, OP_NE, OP_OR, OP_PSH, OP_SC, OP_SHL, OP_SHR, OP_SI, OP_SUB, OP_XOR                                                                                |
|     6 |    7 | OP_JSR                                                                                                                                                                                                                                                      |
|     7 |    0 | OP_LEA                                                                                                                                                                                                                                                      |
|     7 |    1 | OP_ADJ, OP_ENT, OP_LEA                                                                                                                                                                                                                                      |
|     7 |    5 | OP_AND, OP_JSR, OP_LC, OP_LEA, OP_LI, OP_OR, OP_SHR, OP_XOR                                                                                                                                                                                                 |
|     7 |    7 | OP_ENT, OP_JSR                                                                                                                                                                                                                                              |
|     8 |    4 | OP_IMM                                                                                                                                                                                                                                                      |
|     8 |    5 | OP_ADD, OP_ADJ, OP_AND, OP_BNZ, OP_BZ, OP_DIV, OP_ENT, OP_EQ, OP_EXIT, OP_GE, OP_GT, OP_IMM, OP_JMP, OP_JSR, OP_LC, OP_LE, OP_LEA, OP_LEV, OP_LI, OP_LT, OP_MOD, OP_MUL, OP_NE, OP_OR, OP_PSH, OP_SC, OP_SHL, OP_SHR, OP_SI, OP_SUB, OP_XOR                   |
|    10 |    0 | OP_LEV                                                                                                                                                                                                                                                      |
|    10 |    1 | OP_LEV                                                                                                                                                                                                                                                      |
|    12 |    1 | OP_IMM                                                                                                                                                                                                                                                      |
|    26 |    4 | OP_ENT, OP_JSR                                                                                                                                                                                                                                              |
|    26 |    5 | OP_ENT, OP_JSR                                                                                                                                                                                                                                              |
|    26 |    6 | OP_ENT, OP_JSR                                                                                                                                                                                                                                              |
|    26 |    7 | OP_ENT, OP_JSR                                                                                                                                                                                                                                              |
|    27 |    0 | OP_LEV                                                                                                                                                                                                                                                      |
|    27 |    1 | OP_LEV                                                                                                                                                                                                                                                      |
|    27 |    2 | OP_LEV                                                                                                                                                                                                                                                      |
|    27 |    3 | OP_LEV                                                                                                                                                                                                                                                      |

NB: blocks 6 head 6 and 8 head 5 read **every** non-control opcode dim. They
are "ALU broadcast" heads that gate on the disjunction of arithmetic/comparison
opcodes and cannot be masked for those opcodes. The remaining 23 gated heads
have narrow opcode coverage.

## Skippable Heads per Opcode

The interesting savings come from the 25 opcode-gated heads. For opcode X,
a gated head is *skippable* iff OP_X is NOT in its gate set — runtime can
zero-out its Q/K/V/O contribution.

**Top 3 opcodes by skippable head count** (sorted desc, ties broken by
alphabetical):

| Opcode      | Skippable gated heads | Of 25 gated total |
| ----------- | --------------------: | ----------------: |
| OP_NOP      |                    25 |              100% |
| OP_PUTCHAR  |                    25 |              100% |
| OP_GETCHAR  |                    25 |              100% |

These three opcodes do not appear in *any* gated head's opcode set, so 100%
of opcode-gated attention compute can be skipped on those steps.

**Bottom 3 opcodes by skippable head count** (i.e. opcodes that genuinely
need the most gated heads):

| Opcode  | Used gated heads | Skippable |
| ------- | ---------------: | --------: |
| OP_JSR  |               10 |        15 |
| OP_ENT  |                9 |        16 |
| OP_LEV  |                8 |        17 |

OP_JSR, OP_ENT and OP_LEV are the function-call / frame-setup opcodes —
they need multi-block stack relays (blocks 5, 6, 7, 10, 26, 27 all gate on
these). This matches expectations: stack frame opcodes are the most
attention-intensive.

## Per-Opcode DEAD Cell Detail

Every opcode has the same 169 no-output heads + 168 zero-input heads as a
shared dead baseline (heads dead-for-everyone). On top of that:

| Opcode      | Total DEAD | Gated-DEAD (X-specific) | Live |
| ----------- | ---------: | ----------------------: | ---: |
| OP_NOP      |        194 |                      25 |   50 |
| OP_PUTCHAR  |        194 |                      25 |   50 |
| OP_GETCHAR  |        194 |                      25 |   50 |
| OP_ADD      |        192 |                      23 |   52 |
| OP_BNZ/BZ   |        192 |                      23 |   52 |
| OP_DIV      |        192 |                      23 |   52 |
| OP_EQ/NE/LT/GT/LE/GE | 192 |                      23 |   52 |
| OP_EXIT     |        192 |                      23 |   52 |
| OP_LC/LI/SC/SI | 192    |                      23 |   52 |
| OP_MOD/MUL  |        192 |                      23 |   52 |
| OP_PSH/SHL/SUB | 192    |                      23 |   52 |
| OP_ADJ/AND/IMM/JMP/OR/SHR/XOR | 191 |             22 |   53 |
| OP_LEA      |        190 |                      21 |   54 |
| OP_LEV      |        186 |                      17 |   58 |
| OP_ENT      |        185 |                      16 |   59 |
| OP_JSR      |        184 |                      15 |   60 |

"Live" counts include the 50 opcode-agnostic heads + the opcode's gated
hits.

## Compute Savings Estimate

Per-head per-step attention compute (sliding window SEQ=35 tokens, HD=91):
`HD * SEQ^2 ≈ 111,475 MAC ops` per head per VM step (dominant softmax + score
matmul). Plus the Q/K/V/O linear contribution ~ `4 * HD * D_in` per token,
which scales linearly with seq.

For the **always-dead 168 zero-input heads × 30 steps × full SEQ**:

- Already addressable today by the `compact_dims` pass (which prunes
  zero-activity heads from W_q/W_k/W_v/W_o). The audit confirms `compact_dims`
  is doing the right thing — these 168 heads should already be skipped at
  inference.

For the **25 opcode-gated heads under runtime opcode masking**:

- **NOP/PUTCHAR/GETCHAR** steps: skip all 25 → **≈ 2.79 M MAC/step saved**
  on the score+softmax stage alone.
- **ALU/CMP opcode steps** (OP_ADD, OP_EQ, etc.): skip 23 → **≈ 2.56 M
  MAC/step**.
- **Memory ops** (OP_LI/LC/SI/SC): skip 23 → ≈ 2.56 M MAC/step.
- **Frame ops** (OP_JSR, OP_ENT, OP_LEV): skip 15–17 → ≈ 1.67–1.90 M
  MAC/step (these are the most expensive opcodes attention-wise).

Frequency-weighted aggregate (representative C4 workload with ~20% IMM,
~10% LEA/PSH, ~5% arithmetic, ~3% branches, ~3% calls): on the order of
**~2.6 M MAC/step saved** in average attention compute if runtime masks
opcode-gated heads.

## Action Items

1. The `compact_dims` already handles the 168 zero-input heads. Confirm via
   `attn._is_compact` after `compile_full_vm` — done by this audit (compact
   matrices are present where applicable).
2. The 25 opcode-gated heads are the masking target. Proposed runtime hook:
   pass current opcode one-hot into `AutoregressiveAttention.forward` and
   zero the head dim slice `Q/K/V[:, h*HD:(h+1)*HD]` for any (b, h) whose
   `op_gates` doesn't contain the current opcode. This requires a precomputed
   per-block `(opcode → head_mask)` table built from this audit.
3. The two "broadcast" heads (block 6 head 6, block 8 head 5) cover ~all
   non-control opcodes — masking them is only safe for OP_NOP / OP_LEV /
   OP_PUTCHAR / OP_GETCHAR / OP_EXIT (the heads' gate sets exclude these).

## Data Sources

- Raw audit JSON: `/tmp/head_activity_audit_v2.json` (244 cells, per-opcode
  DEAD/LIVE classification).
- Audit script: `/tmp/head_activity_audit_v2.py`.
- Opcode dim names: `c4_release/neural_vm/unified_compiler/ops/shared.py`
  `_opcode_name_map`.
- `Operation.opcodes` annotations: 25 ops annotated across
  `c4_release/neural_vm/unified_compiler/ops/{l5,l6,l9,l10,l13,l14,l15,l16,user_input,model}_ops.py`.
  Annotation coverage is currently sparse; the audit derives gating directly
  from baked W_q/W_k/W_v columns rather than the Operation.opcodes set, so
  results are robust to under-annotation.
