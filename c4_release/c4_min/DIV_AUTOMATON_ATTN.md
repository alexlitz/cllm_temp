# CONST-DIVISOR DIVMOD as an ATTENTION-CAM AUTOMATON

*The divisibility-DFA transition table realised as a softmax1 **content-addressable
KV memory** — one attention lookup per step instead of a per-step FFN one-hot.*

Code: [`div_automaton_attn.py`](div_automaton_attn.py) (build + verify), gated by
[`test_div_automaton_attn.py`](test_div_automaton_attn.py). It READ/imports the
fp32-exact SwiGLU emitters from [`nibble_alu32.py`](nibble_alu32.py) (`_step_ge`,
`_ident`, `_clear`, `_guard`, `_truncate`, `_empty_spec`, `RELU_S`, `S`, `SILU_S`)
and reuses the binary per-bit-agreement CAM key/query of
[`qwen_full_vm._bake_memory_cam`](qwen_full_vm.py) /
[`shift_attention_bench`](shift_attention_bench.py) — no shared file is edited.

## The baseline it compresses

The FFN one-hot automaton ([`const_divmod_automaton.py`](const_divmod_automaton.py))
realises the same divisibility-DFA transition
`(q, R_new) = divmod(16·R + d, b)` — all baked constants for a compile-time `b` —
as a **per-step FFN one-hot**:

* **Block EQ** — the equality pulses `[d==e]`, `[R_nib==v]` (2 `_step_ge` units each);
* **Block RD** — the guarded readout: a `_guard` AND over the `b·16` reachable
  `(R, d)` rows routes the baked `(q, R_new)`;
* **Block COMMIT** — `R := R_new`, scatter `q`.

That is **2 FFN transition blocks + 1 commit = 3 blocks/step, 25 blocks**, and the
`b·16`-row table is **replicated as FFN guard units every step** (nz grows from
~6.8K at `b=2` to ~3.3M at `b=1024`).

## The idea — the transition table as a CAM

Because the transition table is a fixed set of `b·16` rows `(R, d) -> (q, R_new)`,
it is a perfect content-addressable memory. Bake it as an **attention KV memory**:

```
Keys   : one baked key row per reachable (R, d), encoded as a per-bit ±G agreement
         code on a RESERVED indicator band (so program state never cross-matches).
Query  : the CURRENT (R, d), same ±G code.
Value  : the baked (q, R_new) for that entry (nibble scalars).
```

One softmax1 attention head retrieves the matching row. With per-bit gain `G` the
exact-match net score is `+0.5·G²/√d` (softmax1 weight ≈ 1.0 in fp32) and every
1-bit mismatch is `≥ 2·G²/√d` **below** it (weight ≈ 0), so
`value = Σ_i w_i·V_i ≈ v_matched` is **byte-exact**. This is exactly the sharp-CAM
regime `_bake_memory_cam` runs in — minus the recency tie-break, because every
`(R, d)` key is **unique**.

The KV rows are **baked at compile time** (the transition table is a compile-time
constant), so its key/value rows are permanent context tokens the query attends to
— the `_bake_memory_cam` "store frames ride the token stream" mechanism, here with
the store log frozen at build time.

## The attention automaton cell — 1 attention lookup/step

Over a tiny residual whose bands are exactly the DFA planes:

* **Block QENC** (FFN glue) — write the ±G binary query code for the current
  `(R, d)`. A nibble bit `[x & 2^i]` is one `_step_ge` staircase (a nibble form
  `≤ 15·RELU_S`, no amplification); the query lane is `G·(2·bit − 1)`, plus the
  `−B` bias-lane constant.
* **Block CAM** (**ATTENTION**) — the softmax1 content-addressable lookup over the
  `b·16` baked key rows; the matched row's value `(q, R_new)` lands in `V_OUT`.
  **This single attention block replaces the FFN automaton's two FFN transition
  blocks (EQ + RD).**
* **Block COMMIT** (FFN glue) — `R := R_new` (SET) and scatter the emitted quotient
  digit `q = V_OUT[0]` into the assembled quotient's MSB-first place. As in the FFN
  automaton, the commit can't fold into CAM (the CAM writes `V_OUT` reading the
  block INPUT, so `R := R_new` must be a subsequent block).

**Attention-lookups-per-step = 1** (the target). Total depth = `3·in_nibs + 1`
blocks — the SAME block count as the FFN automaton, but the heavy `b·16` transition
table now lives in the **attention KV cache** (fixed weight, position-invariant,
shared across all 8 steps) instead of `b·16` FFN guard units *replicated every
step*.

## fp32 discipline — sharp hardmax, no snap

* The QENC staircase reads a nibble bit `[x & 2^i]`: each `_step_ge` argument is a
  nibble (0..15) scaled by `RELU_S` → `≤ RELU_S·16 ≈ 3200`, **five orders below the
  `2^24` fp32-integer limit**, with NO `16·R` amplification and no accumulation.
* The attention scores are per-bit `±G²` dots (`G=16` → net ~16 logits post-scale),
  so the CAM is a **hardmax in fp32**: the worst-case exact-match softmax1 weight
  **rounds to 1.0** across the whole battery (verified), and mismatching rows are
  ≥ ~48 logits below → weight ≈ 0.
* Values are baked integer nibbles. **Zero fp64 anywhere.**

## Results (real softmax1 attention + SwiGLU forward, 32-bit dividend)

Battery `{2,3,7,9,10,16,60,100,255,256,1000, small primes 11/13/17/97/251, powers
of two}`, each verified **byte-exact for both q and r** over edges
`{0,1,b−1,b,b+1,2³²−1}` + 2000 random 32-bit dividends, plus `b==0 → (0,0)`.

| `b` | blocks | attn/step | KV rows `b·16` | key dim | state nibs | FFN-glue nz | KV-table nz | worst exact wt | max \|kq\| |
|---:|---:|:---:|---:|:---:|:---:|---:|---:|:---:|---:|
| 2 | 25 | 1 | 32 | 6 | 1 | 3 340 | 238 | 1.0000 | 34 |
| 3 | 25 | 1 | 48 | 7 | 1 | 3 900 | 413 | 1.0000 | 38 |
| 7 | 25 | 1 | 112 | 8 | 1 | 4 204 | 1 097 | 1.0000 | 41 |
| 10 | 25 | 1 | 160 | 9 | 1 | 4 380 | 1 734 | 1.0000 | 44 |
| 16 | 25 | 1 | 256 | 9 | 1 | 4 380 | 2 784 | 1.0000 | 44 |
| 17 | 25 | 1 | 272 | 10 | 2 | 5 504 | 3 231 | 1.0000 | 47 |
| 100 | 25 | 1 | 1 600 | 12 | 2 | 6 368 | 23 532 | 1.0000 | 52 |
| 256 | 25 | 1 | 4 096 | 13 | 2 | 6 544 | 64 768 | 1.0000 | 54 |
| 1000 | 25 | 1 | 16 000 | 15 | 3 | 8 228 | 296 872 | 1.0000 | 59 |
| 1024 | 25 | 1 | 16 384 | 15 | 3 | 8 228 | 304 128 | 1.0000 | 59 |
| **0** | **17** | 1 | 16 | 6 | — | — | — | — | 34 → `(0,0)` ✓ |

**ALL BYTE-EXACT: True** (q and r) across the entire battery + edges, `b==0` incl.
Every row's worst-case exact-match softmax1 weight is **1.0000** (a clean hardmax);
key/query magnitudes are **34–59** (the sharp-CAM regime, well within fp32).

## Attention-CAM vs FFN one-hot — the honest verdict

The two automata are the **same 25-block, 1-lookup-per-step DFA**. What the
attention-CAM changes is *where the `b·16` transition table lives*:

* **FFN one-hot:** the table is `b·16` `_guard` AND units, and each of the 8 steps
  re-runs the EQ+RD blocks — so the table cost is paid as **2 FFN transition blocks
  per step**, and total FFN nz grows with `b·16` (6.8K at `b=2` → **3.3M** at
  `b=1024`).
* **Attention-CAM:** the table is a **single shared KV cache** (`b·16` key + value
  rows), consulted by ONE attention head per step. The per-step FFN shrinks to
  trivial glue (QENC + COMMIT), whose nz is **~3–8K and essentially flat in `b`**
  (it never carries the table). The table cost moves entirely to `KV-table nz`
  (238 at `b=2` → 304K at `b=1024`).

So the win is **not fewer blocks** (both are 25) — it is:

1. **The transition LOOKUP is genuinely ONE attention head** (the stated target),
   collapsing the two FFN transition blocks (EQ + RD) into one CAM block.
2. **The `b·16` table is amortised across all 8 steps as a shared KV memory** rather
   than replicated as FFN guard units per step — the FFN glue is flat in `b`, and
   the table is ~11× cheaper in raw nz at `b=1024` (304K KV entries vs 3.3M FFN
   units) because a KV row is `(key dim + value dim)` entries instead of `b·16`
   AND-gate weights.
3. **The CAM is a perfect hardmax** (worst exact-match weight 1.0000) in the same
   `_bake_memory_cam` sharp regime — no snap, no fp64.

**When to prefer which:** the attention-CAM is the right form when the transition
table is consulted *many times* (every step of every divmod) and you want it as a
shared, position-invariant KV memory — the natural "register/memory read" idiom of
this VM. The FFN one-hot is simpler (no attention block, pure SwiGLU) if you are
already paying FFN depth and don't want an extra head. Both are byte-exact and
fp32; the choice is an architecture-fit call, not a correctness one.

## Feasible `b` range

The KV cache holds `b·16` rows (one per reachable `(R,d)`), each a `(n_bits+1)`-dim
key + `(1+state_nibs)`-dim value. This stays a **sharp hardmax** and small for
**small-to-moderate `b`** (verified to `b=1024` → 16 384 rows, worst exact weight
still 1.0000). Past `b·16 ≤ 16384` → **`b ≤ 1024`** (`feasible_b_max()`) the KV
table dominates the memory — exactly as the FFN automaton's `b·16` guard table did
— while the digit-recurrence's cost is `b`-independent, so for **large `b`** the
digit-recurrence (74–122 blocks, `O(1)`-in-`b` width) is the better trade. The
whole battery (max `b=1024`) is inside the feasible range.

## Run

```
OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m c4_min.div_automaton_attn        # report
OMP_NUM_THREADS=4 PYTHONPATH=$(pwd) python -m pytest c4_min/test_div_automaton_attn.py -v
```
