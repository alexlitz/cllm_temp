# Qwen R8 layer-by-layer divergence profile (2026-06-09)

After **I1** (commit `144cbdb8`) and **I3** (commit `3ffc514e`) landed, the
R8 production round-trip argmax sits at ~15 %. This doc profiles where in
the 36-block decoder stack the native VM and the exported Qwen3 model
diverge so future fix agents know exactly which layer to drill.

The diagnostic compiles the production VM with `C4_QWEN_EXPORT_COMPAT=1`,
exports it via `export_qwen3_dense`, loads through
`AutoModelForCausalLM.from_pretrained`, installs the
`NeuralVMEmbeddingWrapper`, and forwards
`int main(){return 42;}` (20-position decode window) through both models.
Per-block input/output hidden states are captured by forward hooks on
each `TransformerBlock` (native) and via `output_hidden_states=True` on
the HF `Qwen3Model`. Residual diffs are bucketed by `dim_registry` band
(`H1` = head-1 7-dim staging at dims 67-73, `H3` at 81-87, etc.).

Runner: `/tmp/qwen_layer_diff_diag.py` (CPU; not a regression gate, not
checked in).

## Per-layer diff table (max |Δ| over batch×seq, per-band)

| Layer | Max \|Δ\| | Mean \|Δ\| | Worst dim band | Note |
|-------|----------:|----------:|----------------|------|
| embed_out | 0.000 | 0.000 | – | I3 wrapper byte-identical |
| L0    | 1.0      | 0.0048 | NEXT_MEM | Flag bit flip; trivial |
| L1    | 1.0      | 0.0078 | NEXT_MEM | Flag bit flip; trivial |
| L2    | 1.0      | 0.0084 | NEXT_MEM | Flag bit flip; trivial |
| L3    | 2.47     | 0.0165 | ADDR_B2_LO | Addr-byte band starts drifting |
| L4    | 5.23     | 0.0310 | ADDR_B2_LO | Linear growth |
| L5    | 54.8     | 0.263  | OUTPUT_LO | Bytecode fetch / OP routing — 10× of L4 |
| **L6**    | **9 966.9**  | **7.225**  | **H1 (=H3 close)** | **FIRST CLIFF — attention H1/H3 explode ~180×** |
| L7    | 9 966.9  | 7.225  | H1 | propagation, no further attn change |
| L8    | 9 966.9  | 14.06  | H1 | POST_PRTF_SP_LO also active |
| L9    | 13 918   | 79.3   | H1 | All 3 used heads (H1/H2/H3) ~12K |
| L10   | 13 918   | 79.3   | H1 | flat |
| L11   | 15 094   | 130.6  | H3 | ADDR_B2_VALID joins top-3 |
| L12   | 15 094   | 130.6  | H3 | flat |
| L13   | 15 800   | 143.9  | H3 | H4 added to top-3 |
| L14   | 15 800   | 143.9  | H3 | flat |
| L15   | 15 800   | 143.9  | H3 | flat |
| **L16**   | **70 145**   | **325.7**  | **H1** | **SECOND CLIFF — attention H1 jumps ~5×** |
| L17–L31 | ~72 5xx   | ~341  | H1 | plateau, near-zero per-block growth |
| **L32**   | **5.95e11**  | **1.20e9** | **H1** | **THIRD CLIFF — H1/H2/H3 all blow up ~8M×** |
| L33   | 5.95e11  | 1.20e9 | H1 | flat |
| L34   | 5.95e11  | **1.21e9** | H1 | absolute peak mean |
| L35   | 7.28e9   | 1.46e7 | H1 | final-block partial collapse (~80×) |
| logits | 7.28e10 | 3.38e9 | – | argmax_match 0.25 on this window |

(`H1`, `H2`, `H3` are head-output staging bands at dims 60+h·7 — i.e.
they are the 7-dim slots that L0/L1 attention heads scatter into the
residual; large |Δ| there means the attention scores themselves diverged
from the native VM, not the FFN.)

## Pattern analysis

1. **Embedding is byte-identical** (`max=0.0`). The I3 wrapper installed
   by `install_neural_vm_embedding_wrapper` is doing its job — ADDR_KEY
   and MEM_STORE bands enter the stack matching the native VM exactly.
2. **L0-L5 is benign drift** (`max ≤ 55`). Some flag bits flip (NEXT_MEM
   at dim 259, OP_BZ, OP_SI) and ADDR_B2_LO accumulates linearly to
   ~5 over 5 layers. This is recoverable with rule-level fixes.
3. **L6 is the FIRST big cliff** — `max` jumps from 55 → 9 967 in the
   `H1`/`H3` attention-head staging bands. L6 is where the attention
   routing for JMP/EXIT/PSH relay first fires (see
   `compiler.py:591` "Compile L6 attention (JMP/EXIT relay …)"). The
   ~180× attention-head jump in a single layer matches the I1/I2
   signatures exactly: **production VM is still ALiBi, exported Qwen3
   runs RoPE; production VM has no per-head q_norm/k_norm**, so the very
   first attention layer where actual cross-position routing happens
   produces a Q·Kᵀ score that has zero structural relationship with the
   native VM's. Once H1 is wrong at L6 it stays wrong for the rest of
   the stack — the linear growth L7→L15 is just residual carry, not new
   divergence.
4. **L16 is a smaller second cliff** (~5× jump). L16 is the LEV
   routing layer; it re-runs an ALiBi-vs-RoPE-sensitive attention head.
5. **L17–L31 is a long plateau** at ~72 500 max. Nothing meaningful
   accumulates — most of those FFNs are gated off because their gate
   inputs (which sit on dims that *did* match the native VM) tell them
   to no-op, so the divergence sits in the H1 band without amplification.
6. **L32 is a third catastrophic cliff** — 8-million × jump to 5.95e11.
   This is in the late ALU/output-cleanup stack; the most likely cause
   is an FFN whose gate is computed on a band that became non-zero late
   (e.g. dim near OUTPUT_HI / TEMP) and the gate now fires when it
   shouldn't, multiplying a large H1 residual by a non-trivial weight
   column.
7. **Final softmax-sink token id / sink prepend has no measurable effect**
   on `embed_out` (`max=0.0` after shape-matching the qwen prepend), so
   D3 sink wiring is at most a minor contributor.

Robust bands (never appear in top-3 worst): `CLEAN_EMBED_LO`,
`CLEAN_EMBED_HI` (except as ties in L5), `ALU_LO`/`ALU_HI`, `CMP`,
`MUL_ACCUM`. The carry/clean-residual machinery is holding up; the rot
is concentrated in **attention-head output staging (`H1`, `H3`, `H2`,
`H4`)**.

## Top-3 biggest divergence layers + fix recommendations

1. **L6 — FIRST CLIFF, `H1` band 55 → 9 967 (~180× single-layer jump).**
   Root cause: production VM compiles with `positional_encoding="alibi"`
   (Blocker I2) and has no per-head `q_norm`/`k_norm` (Blocker I1).
   Exported Qwen3 runs RoPE + per-head Q/K RMSNorm. The L6 attention
   weights were baked for ALiBi geometry, so they immediately produce
   wrong scores when fed through Qwen3's attention.
   **Fix recommendation:** finish wiring **I2 in the production VM** —
   re-compile every attention block under `positional_encoding="rope"`
   (matching the exported `Qwen3Config.rope_theta=10000`), or pin
   `f7af0777`'s `alibi_to_rope_export` so it actually re-allocates RoPE
   caches AND re-bakes `W_q`/`W_k` to compensate for the dropped
   `−slope·|i−j|` term on layers 0-3 (the hybrid range). Then bake the
   I1 per-head compensator into the production VM the same way the tiny
   VM does. If only one of I1 or I2 lands without the other, L6's
   `max |Δ|` should drop by ~10× — if it doesn't, that's the regression
   signal for that fix agent. Verifier: re-run the diag and assert
   `L6 max < 1 000`.

2. **L32–L34 — THIRD CLIFF, `H1` band 73K → 5.95e11 (~8 million×).**
   The geometric explosion here cannot be ALiBi-vs-RoPE alone (that
   would compound linearly, not geometrically). Most likely an FFN gate
   in the ALU/output-cleanup stack (L32 is in the floor_ffn / div_mod /
   output-finalize range — see `c4_release/neural_vm/alu/ops/`) fires
   when fed the wrong H1 residual and multiplies by a large `W_down`
   column. **Fix recommendation:** dump `block.ffn.W_down[:, 69]`
   (H1 column) for L32/L33/L34, and identify which units have
   non-zero `W_down` writes into H1. Verify their gate predicates — if
   any of them reads a dim that is correct in the native VM but wrong
   in the qmodel forward, that's the amplifier. This is *only* worth
   chasing after L6 is fixed; if H1 enters L32 already correct, the
   explosion likely disappears.

3. **L16 — SECOND CLIFF, `H1` band 15.8K → 70.1K (~5× single-layer jump).**
   L16 is the LEV routing layer. Same root cause family as L6 (attention
   head bake assumes ALiBi geometry) but smaller magnitude because the
   incoming residual already has H1 saturated. **Fix recommendation:** if
   the L6 fix already lifts L6's `max |Δ|` below 1 000, L16 should fall
   to ~5× of that (~5 000) automatically — no separate fix needed.

## Bottom line for future fix agents

The **single biggest gap** is **L6 attention** — closing the ALiBi→RoPE
+ per-head q_norm/k_norm story there should propagate down the stack
and collapse the L16/L17-31 plateau as a side effect. The L32 explosion
is a downstream amplifier of the L6 wrongness; do not try to fix L32
first. Order of operations: **I2 production wiring** → **I1
production wiring** → re-run this diag → only then drill L32 if it
remains anomalous.
