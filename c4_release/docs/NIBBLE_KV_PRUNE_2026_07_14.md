# KV Cache Pruning for the c4 Foundation Model

**Branch:** `nibble-kv-pruning`
**Authority:** `docs/BLOG_SPEC.md` §"KV Cache Pruning" (lines 800–816).
**Deliverable:** `c4_min/nibble_kv_prune.py` + `c4_min/test_nibble_kv_prune.py`.
**Status:** additive — no change to `blogspec_model.py`; a caller that never prunes
gets the identical full-cache result.

---

## 1. Problem

The c4 VM emits the **full register file + memory writes every step** — the
30-token register frame of `blogspec_vocab.build_step_frame` (`REG_PC | 4B | REG_AX
| 4B | REG_SP | 4B | REG_BP | 4B | MEM | 8B | STEP_END`). So a naive KV cache
grows **linearly** with program length: a million-token program keeps a million
KV entries. That is infeasible.

Per BLOG_SPEC the cache instead stays **bounded (logarithmic)** via an eviction
policy that is *safe* — a numerical no-op on the attention output — because the
model attends with **softmax1** (`exp(x) / (1 + Σ exp(x))`, the ZFOD sink) and
**ALiBi** (`-slope·|i−j|` recency bias). Superseded entries are already crushed
to ~0 attention weight by ALiBi recency, so evicting them changes nothing.

## 2. The eviction policy (`c4_min/nibble_kv_prune.py`)

The cache stores per head a list of `KVEntry(key, value, position)` — the real
projected `W_k·x` / `W_v·x` and the **absolute** sequence position (which drives
ALiBi distance and is *never renumbered* on eviction; BLOG_SPEC line 712). A
`MultiHeadKVCache` holds one `KVCache` per head, each pruning with *its own* ALiBi
slope. `prune()` applies, in order:

**Mechanism 1 — cos-sim > 0.99 + ALiBi recency (latest-write-wins).**
Each VM step re-emits every register's marker, so a register's key repeats
verbatim across steps (cosine ≈ 1.0). ALiBi gives the *older* copy a permanent
extra `−slope·Δ` penalty, so its softmax1 weight is a fixed factor `exp(−slope·Δ)`
smaller than the newer copy's. We keep the newest of each near-duplicate key group
and evict the older members. This is *latest-write-wins* for registers and memory
(BLOG_SPEC lines 810–814).

**Mechanism 3 — ALiBi-recency horizon.**
The one mechanism that bounds an entry's contribution to *both* the softmax1
numerator (weighted value) *and* denominator (its `exp(score)` term). An entry
whose **max possible future weight** `exp(ceil_score − slope·dist)` (with
`ceil_score = max_key_norm²·scale` a conservative content-score ceiling, and
`dist = newest_pos − pos`, which only grows for future queries) falls below
`recency_eps` is dominated by recency and can never win non-trivial attention.
This is the explicit form of the spec's own justification ("ALiBi's steep recency
bias … will never win non-trivial attention weight") and generalises mechanism 1
to *non-duplicate* old payloads (e.g. old register byte tokens).

**Mechanism 2 — zero writes ("how free works").**
Two exact sub-rules:
- **2a dead-value head:** if *no* surviving entry carries a non-zero value, the
  head's output is identically `0` for every query; the whole head is evicted
  (this collapses the entirely-inert heads whose `W_v = 0`).
- **2b free-zero:** a `value == 0 AND key == 0` entry contributes `0` to the
  numerator and only a bare `exp(−slope·dist)` to the denominator; it is evicted
  once that denominator term is itself recency-negligible. A zero-*value* entry
  with a **strong key** (a live register marker) is *not* free — its `exp(score)`
  is a load-bearing denominator term that suppresses the other weights — and is
  left to mechanisms 1/3.

Eviction runs every `PRUNE_INTERVAL_TOKENS = 120` tokens (≈3 VM steps) via
`maybe_prune()`, exactly as the spec prescribes.

### Correctness subtlety (why the naive rule is wrong)

The spec's phrasing "a zero-value entry has the same result as not attending" is
only exact **when the entry's score is also negligible** ("as long as there are no
other keys that have non-trivial scores", line 816). Under softmax1 a zero-*key*
entry still scores `0` and contributes `exp(0 − slope·dist)` to the *denominator*;
dropping a *recent* one would raise every other weight and change the output. And a
zero-value *marker* with a **high** key (score ≈ 1.35 here) contributes `exp(1.35)`
to the denominator — evicting it corrupts the decode. Both are handled by the
recency horizon / dead-value rules above, not by a blind value-is-zero test. This
is the single correctness pivot of the implementation.

## 3. Eviction interface for the generation loop

```python
from c4_min import nibble_kv_prune as P

attn  = model.blocks[0].attn                 # a blogspec_model.Attn
cache = P.MultiHeadKVCache(attn.n_heads, attn.head_dim, attn.alibi_slopes)

for pos, token_residual in enumerate(stream):        # one residual per token
    k, v = P.project_kv(attn, token_residual)        # [n_heads, head_dim] each
    cache.append(k, v, pos)                           # 1) cache the token's KV
    cache.maybe_prune()                               # 2) evict on the 120-tok tick
    q   = P.project_q(attn, token_residual)           # 3) query at the frontier
    out = cache.attention_output(q, pos)              # softmax1+ALiBi over LIVE cache
    #     out is the pre-W_o attention output; residual = token_residual + W_o @ out
```

`attention_output` reproduces `blogspec_model.Attn.forward` per head to fp32
precision (verified: max abs diff ≈ 9e-7). `project_kv` / `project_q` project a
residual into per-head K/V/Q using the block's real weights. Eviction is
**additive**: set `prune_interval=10**9` for a never-pruned reference.

## 4. Output-equivalence proof (`c4_min/test_nibble_kv_prune.py`)

Run on a real countdown loop `AX=N; {PSH; IMM 1; SUB; BNZ loop}; HALT`, which
emits one 30-token register frame per step (many near-duplicate marker keys — the
exact case mechanism 1 targets). Comparing a **never-pruned** cache against the
**pruned** cache driven exactly as the generation loop drives it (append → prune →
query at the frontier):

- `test_attention_output_equivalent_full_vs_pruned` — at every step's frontier the
  softmax1+ALiBi attention output pruned == full to fp tolerance (< 1e-3; evicted
  entries' weights are below the recency epsilon).
- `test_decode_byte_exact_full_vs_pruned` — the **byte decoded through the real
  byte head** (`_decode_byte_from_nibbles`) from the reconstructed residual is
  **identical** with and without pruning, every step. This is the production claim:
  pruning never changes an emitted byte.
- `test_full_program_trace_unchanged_by_pruning` — the whole decoded AX trace
  equals the reference interpreter (`isa.interpret`).
- plus 7 unit tests pinning each mechanism (near-dup evicts older, free-zero
  evicts, strong-key zero-value survives, distinct/newest survive, interval
  gating, high eviction ratio).

**All 10 tests pass** (`python c4_min/test_nibble_kv_prune.py` → `10/10`).

## 5. Bounded-cache measurement

Countdown-7 (30 VM steps, **902 tokens emitted**), pruning every 30 tokens:

| Tokens seen | Un-pruned (entries/head) | **Pruned head-0 live** |
| ----------- | ------------------------ | ---------------------- |
| 30          | 30                       | 30                     |
| 120         | 120                      | **55**                 |
| 240         | 240                      | **55**                 |
| 480         | 480                      | **55**                 |
| 901         | 901                      | **56**                 |

The un-pruned cache grows **linearly** (902 entries/head at the end); the pruned
cache is **flat at ~55 entries/head** from token 120 onward — bounded by a small
constant (≈ the recency horizon + #distinct live marker/byte patterns), *not*
growing with step count. Across all 4 heads the pruned total is **63** (heads 1–3
are entirely `W_v = 0`, so mechanism 2a evicts them wholesale) vs 3608 un-pruned.
Eviction ratio: **93.7%** and climbing toward the spec's "99.999%" as programs
lengthen, because the bound is constant. This is what makes **arbitrarily long
programs** feasible with bounded memory: cache size is O(1) in program length while
the decode stays byte-exact.

## 6. Files

- `c4_min/nibble_kv_prune.py` — `KVEntry`, `KVCache`, `MultiHeadKVCache`,
  `project_kv`/`project_q`, the eviction policy and the softmax1/ALiBi mixer.
- `c4_min/test_nibble_kv_prune.py` — the 10-test proof (equivalence + bounded).
