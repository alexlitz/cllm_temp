# LOCAL (sliding-window) attention on the non-memory heads — 2026-07-21

**Branch** `local-attention-non-memory-heads` (off `kv-prune-hash-two-tier-2026-07-21 @ 17d6d4e4`).
**Device** `cuda:1` (24 GB), uncontended.  Do NOT touch main, do NOT merge.

## Motivation (#686 follow-up)

After #686 the fast path (`pf_speculative.verify_blocks` + `forward_hidden_cached`)
is the wall — **~80–83 % of the fast wall is the batched block forwards**, and each
block computes the FULL `Q@Kᵀ` score matrix `[H, Sq, Sk]` — **O(S²) per head** — even
though almost every head either (a) reaches only the current VM-step frame or (b)
has a zero value output.  The memory KV heads are the ONLY heads that must reach far
into the past.  So: give the LOCAL heads a **sliding window** (vanilla
Mistral/Longformer) of the last `W` keys — O(S·W) — and keep only the memory/stack/
LEV heads GLOBAL.  Each windowed head's true attention weight past `W` is **exactly
0**, so local == global — **byte-identical**.

## STEP 1 — head classification (MEASURED)

The complete pure-forward model (`build_pure_forward_complete_model`, the build the
fast path uses via `build_lib_model_streaming`) has **`n_heads = N_ROLES + 3 = 23`**
and **307 blocks**.  Only **3 of the 307 blocks have ANY non-zero attention output**
(`_zero_attn` zeros W_q/W_k/W_v/W_o on every other block → its attention output is
provably `x`).  Per-head empirical windows (max `|q_pos − k_pos|` carrying attention
weight ≥ 1e-6 at any query row, over a real run):

| block | name             | LIVE heads | role                | measured window | class  |
|------:|------------------|-----------:|---------------------|----------------:|--------|
| 0     | ingest+recompose | 0–19       | frame ingest (recency slope 6.0) | **≤ 28 tok** (< 1 frame) | **LOCAL** |
| 6     | mem-cam          | 20         | LI/LC KV memory read (slope 1.0) | 309→up to ~250 k tok | **GLOBAL** |
| 10    | stack-pop-cam    | 21, 22     | stack-pop + LEV return-PC (slope 1.0) | 159→up to ~250 k tok | **GLOBAL** |

Every OTHER head in EVERY block has a **zero value output** (W_v==0 or W_o==0) — its
attention output is `x` regardless of the window, so windowing it is free and
byte-identical.

**Classification rule** (`local_attention.classify_heads`, CONSERVATIVE): a LIVE
value head (W_v rows AND W_o cols non-zero for its head-dim slice) is windowed ONLY
if its ALiBi recency slope is the ingest slope `INGEST_RECENCY=6.0` (the PROVEN-local
frame-ingest heads).  EVERY other LIVE head is kept GLOBAL — the memory heads AND any
future far-reaching head whose slope is neither the ingest nor the memory slope.
Zero-value heads (attention output 0, window-invariant) are windowed regardless
(that is what wins on the 304 pure-passthrough blocks).  So **GLOBAL = live heads
that are NOT proven-local ingest heads.**  Result: **3 GLOBAL head-slots**
({6: [20], 10: [21, 22]}), **7058 / 7061 head-slots windowed (100.0 %)** across all
307 blocks.

**On the position-signature I/O heads.**  The brief anticipated position-signature
I/O heads.  The *complete* pure-forward model that the fast path builds
(`build_pure_forward_complete_model`) does NOT contain them — its I/O (OPEN/READ/
CLOS/PRTF) is the ONE class handled by the driver's TOOL_CALL boundary, not by baked
attention heads (see the `FILE_OPCODES` path).  Position-signature I/O heads live in
a SEPARATE build (`nibble_io_position.py`, used by chat/ELIZA), which the fast path
does not use.  The conservative rule handles them correctly if a build adds them: any
live head whose slope ≠ the ingest 6.0 is kept GLOBAL by default (windowing is opt-in
per proven-local slope, not opt-out), so a position-signature head is never wrongly
windowed.  `classify_heads` is empirical-per-build, not hard-coded to 3 heads.

### The ingest window CANNOT grow with stream length (why W=64 is safe)

Every VM step emits a full 30-token frame carrying all 20 register-byte ROLE
one-hots.  An ingest head for role `h` KEY/QUERY-matches ONLY role-`h` frame bytes;
among matches, recency ALiBi (slope 6.0) subtracts `6·dist`.  Because every frame
re-emits role `h`, a MORE-recent match always exists < 1 frame (≤ 30) back, and the
one-frame-older match is penalised `6·30 = 180` → `exp(−180) ≈ 10⁻⁷⁸` weight → far
below any decode margin.  **Measured worst window = 28 tokens over a 2401-token
stream** (did not grow); the chosen **W = 64 (≈ 2 VM steps)** covers it with 2.3×
margin.

## STEP 2 — the sliding-window implementation

`c4_min/local_attention.py`.  `windowed_forward` is a drop-in for
`SparseAttn.forward` (installed as a per-instance bound method by
`install_local_attention(model, window=W)`).  It splits the heads:

* **GLOBAL heads** (`_global_head_mask` True) — full causal over ALL keys (unchanged
  softmax1 + ALiBi + causal math).
* **LOCAL heads** — slice the key set to the contiguous suffix `k_pos ≥ q_pos.min()
  − W + 1` (the union of every local query's window), then apply a per-row window
  mask `(q_pos − k_pos) ≥ W → −∞` on top of the causal + ALiBi.  The score/softmax/
  weighted-V is **O(S·W)** instead of O(S·Sk).

The **KV cache is unchanged** — the full projected K/V of every position is still
stored (the global heads need it; the commit/eviction path is shared).  Only the
per-head *read* is windowed.  `uninstall_local_attention` reverts to the global
forward.  Opt-in via `bench_fast_path --local-window W`.

## STEP 3 — byte-identity proof (`_verify_local_attention.py`)

Ran the battery (add / mul / loop_n12 / nested_3x20) on the FAST PATH with GLOBAL vs
LOCAL attention (W=64), and captured the raw hidden L∞:

```
add_42       17 steps  global[PASS AX=42 acc=17]  local[PASS AX=42 acc=17]  IDENTICAL
mul_42       17 steps  global[PASS AX=42 acc=17]  local[PASS AX=42 acc=17]  IDENTICAL
loop_n12    195 steps  global[PASS AX=0  acc=195] local[PASS AX=0  acc=195] IDENTICAL
nested_3x20 1018 steps global[PASS AX=3  acc=1018] local[PASS AX=3 acc=1018] IDENTICAL
per-row hidden L-inf (global vs local): 0.000e+00   BYTE-IDENTICAL
VERDICT: PASS — local == global BYTE-IDENTICAL
```

The hidden L∞ is **exactly 0** — not just the decoded AX, the raw residual at every
query row is identical.  (Unit-tested `windowed_forward` vs the global forward on
synthetic attention: L∞=0 for window≥span all-local / all-global / mixed; >0 only
when the window genuinely cuts keys.)

## STEP 4 — the forward win (MEASURED, cuda:1 uncontended)

Isolated forward microbench (`_bench_local_forward.py`): ONE big-K
`forward_hidden_cached` over a span S against a cache, GLOBAL vs LOCAL (W=64):

| span S | cache | Sk   | GLOBAL ms/step | LOCAL ms/step | forward speedup | GPU util |
|-------:|------:|-----:|---------------:|--------------:|----------------:|---------:|
| 900    | 900   | 1800 | 79.2           | 57.8          | **1.37×**       | 95–100 % |
| 1900   | 1400  | 3300 | 121.4          | 84.4          | **1.44×**       | 99–100 % |

The win **grows with span** (O(S²) attention grows relative to the O(S) FFN GEMMs):
at the nested-6x100 span (S≈1900) the forward drops **121 → 84 ms/step (1.44×)** —
attention was ~30 % of the forward, and local attention removes almost all of it.
The remaining forward cost is the 307 sparse FFN GEMMs (orthogonal to attention).

Fast-path end-to-end, byte-exact with local attention (W=64), cuda:1 sparse_mm:

| program            | steps | K  | fast ms/step | GPU util | byte-exact | final AX |
|--------------------|------:|---:|-------------:|---------:|:----------:|:--------:|
| loop n=64          | 975   | 48 | 43.0         | 94 %     | ✅ 975/975  | 0 ✓ (== global 41.6) |
| **nested 6x100 (deep headline)** | **9217** | 48 | **43.8** | 95 % | ✅ 9217/9217 | **6 ✓** |

Both are byte-exact (`fast output == naive prefix: True (0 bytes)`, final AX correct,
`fast final AX == expected (32-bit ref): True`).

### The eviction interaction (honest caveat)
On a DEEP LOOP the KV eviction collapses the cache (nested-6x100: `max_cache=7` — the
loop revisits the same frame positions), so S per forward is only `K·30 + 7 ≈ 1447`
and the forward is FFN-GEMM-dominated at that small S — local and global are ~equal
(loop n=64: local 43.0 vs global 41.6 ms/step).  **The local-attention forward win
appears when the SPAN is large** — a big K, or a program whose cache genuinely grows
(malloc/growing-heap, or a big-K verify before eviction fires).  That is exactly the
regime the STEP-4 microbench isolates (S=900–1900 → 1.37–1.44×), and it is the regime
where global attention hits the O(S²) VRAM OOM that local removes.

### Note on the O(S²) VRAM wall
Global attention at S=1900/cache=1400 peaks at **19.65 GB** — the `[1,23,1900,3300]`
score matrix is why big-K global verify OOMs a 24 GB card.  Local attention removes
that transient (its score is `[·, S, W]`), so it is also the lever that lets a bigger
K fit; the *measured* peak here is set by the FFN hidden (dense_kernel), so the
reported VRAM delta is small, but the score-matrix OOM at big K is gone.

## Files

* `c4_min/local_attention.py` — classification + `windowed_forward` +
  `install/uninstall_local_attention` (the deliverable).
* `c4_min/bench_fast_path.py` — `--local-window W` flag (opt-in).
* `c4_min/_verify_local_attention.py` — byte-identity + window-growth proof.
* `c4_min/_bench_local_forward.py` — the isolated forward microbench (STEP 4).
* `c4_min/_probe_attn_windows.py` — the per-head window probe (STEP 1).

## Honesty

* **Fraction windowed: 100 % of head-slots** — only 3 head-slots (the memory KV
  heads) stay global.  No "ingest" head reached further than expected (worst 28 tok).
* The forward speedup is **1.37–1.44×** at these spans — real but bounded by the FFN
  GEMMs, which dominate the forward; attention was ~30 % of it.  The win grows with
  span, and it removes the O(S²) score-matrix VRAM wall (the big-K OOM).
* Local attention is **OFF by default** (`--local-window` opt-in); every default
  build is byte-identical (uninstall restores the exact global forward).
