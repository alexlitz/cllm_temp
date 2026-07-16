# Checklist #6 (hardening) — KV eviction is ROBUST + byte-exact under adversarial pressure

**Branch:** `chk6b-eviction-hardening` (off `chk6-kv-eviction-scale` @ `04df7c1c`)
**Owns:** `c4_min/nibble_kv_prune.py` (the eviction policy), `c4_min/test_kv_eviction_stress.py`
(new stress suite), `c4_min/test_nibble_kv_prune.py` (mechanism tests, updated).
**Builds on:** CHK-6 (`docs/KV_EVICTION_LONGRUN_2026_07_15.md`) — the happy-path proof
(byte-identical to a flat 114-entry/head cache at 42k tokens on the deep-loop corpus).

**Verdict:** CHK-6 proved the *happy path*; this deliverable HARDENS the policy. It
found and fixed a **real eviction-correctness bug** (un-gated near-duplicate
supersession diverged from the unbounded cache by **0.345** — enough to flip a byte
decode — on an adversarial-recency stream), and proves the fixed policy
**byte-identical to the unbounded reference** across all five stress axes, at
**250k-token** deep-loop scale, with the cache staying flat and RSS ≈ 4 GB.

---

## 1. The bug: un-gated mechanism-1 supersession is NOT output-exact

Mechanism 1 (near-duplicate eviction) originally dropped the OLDER of any two
entries whose keys have cos-sim > 0.99, **purely on cos-sim, with no recency gate**:

```python
for e in ordered:                       # newest-first
    if any(cosine_sim(e.key, k.key) > cos_threshold for k in survivors):
        continue                        # <-- dropped regardless of recency
    survivors.append(e)
```

The docstring justified this as *latest-write-wins*: "a register's key pattern
repeats verbatim across steps ... ALiBi gives the older copy an extra `−slope·Δ`
penalty, so its weight is a factor `exp(−slope·Δ)` smaller than the newer's." That
reasoning is a **ratio** bound, and it silently assumes **same key ⇒ same value**.
Both assumptions fail in general:

1. **A near-duplicate KEY does not imply a superseded VALUE.** Two entries can share
   a key *direction* (cos-sim > 0.99) yet carry **different values** — two distinct
   memory writes that alias a key direction, or a register marker whose payload
   changed step-to-step. Dropping the older one then removes a *live, differently-
   valued* contribution from the softmax1 numerator.
2. **The ratio being small ≠ the absolute weight being small.** If the newer copy
   is itself recent (large weight), the older copy — even at `exp(−slope·Δ)` of it —
   can still carry non-negligible **absolute** weight. Its `exp(score)` term is then
   load-bearing in the softmax1 **denominator**: removing it re-weights *every other*
   entry. This bites **even when the values match exactly** (a same-key/same-value
   pair both recency-live is NOT reducible to one entry under softmax1 — measured
   0.14 output change).

**Reproduction (the adversarial-recency stream).** Store an address-live value at
step 5, run 3000 unrelated register-churn steps (weak keys, random directions), then
read at the frontier. The un-gated rule dropped a recency-live entry (position 3000,
softmax1 weight **0.049** in the reference) because a *newer* churn key (position
3002) happened to alias its direction but held a *different* value → the bounded
frontier output diverged from the unbounded reference by **max |Δ| = 0.345**, which
would flip a decoded byte. This is exactly the store-early/load-late risk class
CHK-6b was chartered to close, surfacing on the *recency-window* half of it.

## 2. The fix: RECENCY-GATED supersession (one bound for mechanisms 1 and 3)

The single correct, output-exact rule for dropping an older near-duplicate is:

> Drop the older near-dup **only when its own maximum future softmax1 weight is
> below `recency_eps`** — i.e. apply the mechanism-3 recency-negligibility bound
> (using the entry's OWN key ceiling) to the *older* member of the pair.

Both the numerator (its weighted value) and denominator (`exp(score)`) contributions
are then `< recency_eps` for **every** future query, so removal is a byte-exact no-op
**regardless of value**. Implemented as a shared helper:

```python
def _max_future_weight(self, entry, newest_pos):
    dist = newest_pos - entry.position
    own_ceil = (entry.key_norm() ** 2) * self.score_scale     # best-case recall score
    return math.exp(min(0.0, own_ceil - self.slope * dist))   # bounds num AND denom

def _is_recency_negligible(self, entry, newest_pos):
    return self._max_future_weight(entry, newest_pos) < self.recency_eps
```

Mechanism 1 now gates the cos-sim drop on `_is_recency_negligible(e_old)`; mechanism
3 uses the same helper. This makes supersession exact in **every** case:

| case | behaviour after fix | exactness |
| --- | --- | --- |
| register markers ≥ 1 frame apart (CHK-6 stream) | older is recency-negligible → dropped | byte-identical, cache stays flat |
| address-live value **overwritten** by a newer same-address write | old write stays recallable ~119 tokens (head-0), matching the unbounded recall, THEN dropped | latest-write-wins, no premature drop, no resurrection |
| two recency-live near-dups, **different values** | **both kept** (both contribute) | fixes the 0.345 divergence |
| two recency-live near-dups, same value | both kept while live (denominator-live) | exact |

**Interaction with address-liveness protection.** An address-live entry that is
superseded by a newer same-address write leaves the cache the moment mechanism 1
finds it recency-negligible — even though mechanism 3's address-live protection would
otherwise keep it forever. So "**recency-stale but address-live and NOT superseded**"
→ KEEP (mechanism 3), while "**truly dead / overwritten**" → drop once ALiBi crushes
it (mechanism 1). That is precisely the keep-by-address-liveness-not-recency
guarantee the charter asked for.

## 3. Store-early / load-late (the #1 risk): result

An address-live value stored at step 5, then 8000 unrelated steps (other addresses +
register churn), then LOAD:

- The address-live store is **KEPT** (never recency-evicted) — `_is_address_live`
  fires (own aligned score `key_norm²·scale ≥ |ln recency_eps|`), so recency
  (mechanism 3) never drops it. `address_live_protected > 0`.
- The bounded cache **matches the unbounded recall at every LOAD** (max |Δ| within
  `recency_eps·value` tolerance), including the honest edge that once the store is
  past the ALiBi horizon **both** caches read ~0 (ZFOD) — the policy never recalls a
  value the reference wouldn't, nor drops one it would.
- The cache stays **bounded** (< seq_len/10) while the stream runs 8000+ steps.

**Boundary, documented.** Whether a stored value is *recallable* at a given distance
is a property of the model (ALiBi slope × address-key strength), **not** of eviction:
at head-0's slope 0.25 and address-key norm ~12, an exact-address load beats ALiBi
for ~`(own_ceil+|ln eps|)/slope ≈ 120–160` tokens, then the value is unrecallable in
BOTH the bounded and unbounded caches. Eviction is byte-identical to the unbounded
cache *at every distance*; it does not (and cannot) extend recall past what the
un-pruned model itself supports. A production memory head that must recall at
arbitrary distance sets a larger `sum(scale²)` (BLOG_SPEC line 410), which raises
`own_ceil` and both the recall horizon AND the address-live protection together — the
policy tracks it automatically (the horizon is derived from the head's own slope +
key ceiling, not a hard-coded window).

## 4. The five stress axes — all byte-identical to the unbounded cache

`c4_min/test_kv_eviction_stress.py`. Because the c4 foundation SLICE
(`blogspec_run._apply_op`) has **no SI/LI** (only IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ/
HALT), the memory-addressing risks are tested where they actually live — against the
eviction POLICY, driven with the per-head KV entries the real memory head projects
(a strong distinctive key per address; a weak byte-value key per churn token), mixed
through the SAME `nibble_kv_prune` softmax1+ALiBi attention the real
`blogspec_model.Attn` uses. Every axis asserts byte-identity to the unbounded
(never-pruned) reference AND (where a value is recalled) the decoded BYTE matches.

| # | axis | what it stresses | result |
| - | ---- | ---------------- | ------ |
| 1 | store-early / load-late | address-live keep vs recency; 8000 unrelated steps then LI | LI diff **≤ recency_eps·value**, store kept, cache bounded |
| 2 | 250k-token scale | ~8400-step self-restarting loop, real `run_program` | **byte-exact**, cache **flat ≈115/head**, RSS **≈4 GB**, ~250k evictions/head |
| 3 | capacity pressure | 40 distinct simultaneously-hot addresses | every live addr LI-exact + **byte-exact**; live cache = working-set + window, churn evicted |
| 4 | memory-op density | 32 addrs, overwrite + free(→0) + ZFOD | latest-write-wins, ZFOD reads 0 in both, **0 byte disagreements** |
| 5 | adversarial recency | near-dup markers (different values/step) + needed old memory key | frontier + old-LOAD both **byte-exact**; the case that exposed the bug |

**Tolerance is honest.** Eviction is exact *to* `recency_eps = 1e-6`: an entry whose
weight is just below the floor may be dropped, contributing up to
`recency_eps · value_magnitude` to the raw attention diff. These payloads reach 255
(a full byte), so the per-entry bound is `1e-6 · 255 ≈ 2.5e-4`; `_EXACT_TOL = 3e-4`.
That is orders of magnitude under any byte-head argmax margin — **the decoded byte
never flips** (asserted directly via `load_byte_agrees` / `_decode_all_bytes`).

## 5. Regression: CHK-6 happy path preserved

- `python -m pytest c4_min/test_kv_eviction_longrun.py -q` → **3 passed (348 s)**;
  the deep-loop corpus is still byte-identical, cache flat (now **115/head**, +1 from
  correctly keeping the recency-live near-dup the un-gated rule wrongly dropped — the
  attn diff actually *improved* from ~1.2e-4 to ~2–4e-7).
- `python -m pytest c4_min/test_nibble_kv_prune.py -q` → **11 passed** (10 original +
  1 new `test_near_duplicate_recency_live_is_kept`). Two mechanism tests were updated
  to assert the corrected, provably-exact behaviour (near-dup eviction is
  recency-gated; a recency-live near-dup is KEPT). The pre-fix versions asserted the
  un-gated drop, which is NOT output-exact.

## 6. Reproduce

```
export OMP_NUM_THREADS=4
# full stress report (per-axis table + traces):
python -m c4_min.test_kv_eviction_stress
# the pytest gates (5 policy stress + the 250k-token scale):
python -m pytest c4_min/test_kv_eviction_stress.py -q       # workers<=2 -- lean
# regression (CHK-6 happy path + mechanism proofs):
python -m pytest c4_min/test_kv_eviction_longrun.py c4_min/test_nibble_kv_prune.py -q
```

Memory discipline: the 250k scale peaks ≈ 4 GB RSS (the unbounded reference is
computed in query blocks — O(H·block·S), never O(H·S²)); the policy-level stresses
are < 1 GB. No dense/neural_vm model is built.

## 7. Files

- `c4_min/nibble_kv_prune.py` — **hardened**: `_max_future_weight` /
  `_is_recency_negligible` helpers; mechanism 1 recency-gated (the fix); mechanism 3
  refactored onto the shared helper. Address-liveness protection (`_is_address_live`)
  unchanged; the supersession-vs-protection interaction is now correct.
- `c4_min/test_kv_eviction_stress.py` — the five-axis stress suite (`_MemHead`
  faithful memory-head stand-in + the 250k `run_program` scale test).
- `c4_min/test_nibble_kv_prune.py` — mechanism tests updated for the corrected
  behaviour + one new recency-live-keep test.
- `docs/KV_EVICTION_LONGRUN_2026_07_15.md` — the CHK-6 happy-path proof (unchanged).
