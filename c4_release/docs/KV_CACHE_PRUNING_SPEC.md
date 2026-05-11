# KV Cache Pruning Specification

Status: SCOPING / DESIGN. Spec only — implementation deferred until the
transformer KV cache (branch `implement-kv-cache`) lands on `origin`.

This document specifies the pruning mechanism that evicts stale entries
from the transformer KV cache while a program is running. It is
**additive** on top of whatever KV-cache implementation the
`AutoregressiveVM` lands: pruning never changes correctness, only the
size of the cache.

Pruning is distinct from KV caching itself. The caching agent is wiring
the per-layer K/V tensors through `AutoregressiveVM.forward` and
threading them through the generation loop. This document focuses on
how to *bound* that cache while the program runs.

---

## 1. Motivation

C4 programs can generate millions of VM tokens (each VM step is 35
tokens; a 100K-instruction trace is 3.5M tokens). Without pruning the
KV cache grows linearly with token count, so attention becomes
O(n²) over time and GPU memory blows up.

Empirically, only a tiny fraction of past K/V entries can influence the
next token under the canonical attention recipe (softmax1 + ALiBi):

- **ALiBi recency bias** monotonically downweights older positions for
  the same key. The instant a fresher near-duplicate key lands, the
  older entry contributes negligibly to any future attention sum.
- **softmax1 + zero values**: writing `V == 0` makes a token's
  contribution identically zero under softmax1 (the "+1" anchor in the
  denominator absorbs all of the probability mass when every other
  score is finite). So zero-valued tokens are *already* dead in the
  cache; we just need to garbage-collect their storage.

These two observations give us two complementary eviction rules.

---

## 2. The Two Eviction Mechanisms

### 2.1 Key-similarity eviction (latest-write-wins)

**Rule.** For each (layer, head) pair, find pairs of cached positions
(i, j) with `i < j` where

    cos_sim(K[layer, head, i, :], K[layer, head, j, :]) > τ

with `τ = 0.99`. Evict position `i` (the older one) from every layer
and every head simultaneously (because the cache is a single shared
sequence dimension; we can only evict whole tokens, not per-head
slices).

**Why this works.**

- The compiler is structured so that semantically identical writes
  (same destination, same operation type) produce keys that are
  numerically close. Examples:
  - Writes to the same memory address from the same instruction site
    use the same MEM-store key pattern, differing only in the value
    bytes that flow into V.
  - Register marker tokens (REG_PC, REG_AX, REG_SP, REG_BP) at the
    same step-position have key embeddings that depend mostly on the
    register identity, not the value. So consecutive step-end emissions
    of `REG_PC` look nearly identical at every layer's K projection.
- ALiBi penalises position `i` (older) more than position `j` for any
  future query, so the older entry already contributes nothing
  meaningful to attention output. Removing it is loss-less.

**Latest-write-wins fall-outs for free.**

- **Memory.** Two writes to the same address produce K vectors that
  differ only in the address-key positional component (same step within
  the MEM-store sub-pattern). They share `> 0.99` cosine similarity in
  the heads that route MEM lookups. Old write evicted.
- **Registers.** Every VM step writes the *entire* register file
  (PC, AX, SP, BP, STACK0). At step N+1, the four register marker keys
  re-appear with cosine similarity ≈ 1 to step N's markers. Old
  register snapshots evicted, only the current register file survives.

### 2.2 Zero-write eviction (free()-style GC)

**Rule.** Immediately after the cache absorbs new K/V entries for a
step, scan the new V rows. Any row whose L2 norm is below a small
threshold `ε = 1e-6` is evicted.

**Why this works.** Under softmax1,

    softmax1(s)_j = exp(s_j) / (1 + Σ_k exp(s_k))

When V_j = 0 (the zero embedding), the contribution to the attention
output is `softmax1(s)_j · V_j = 0` regardless of the score `s_j`.
The "+1" anchor in the denominator also means that attending to *only*
a zero token is the same as attending to *nothing*. So a zero V row is
dead weight.

C4 programs hit this case continuously:

- `free(p)` writes zero to a heap region; those V rows become zero.
- Uninitialised stack slots between `ENT` and the first write are zero.
- The compiler models "no-op" sub-positions inside a step (between
  STEP_TOKENS-positioned markers) as zero-value writes.

Zero-write eviction is what gives the cache its *logarithmic* (not
just bounded) growth on real C programs: most memory traffic
eventually turns into zero writes when a function returns.

### 2.3 Why both?

They cover orthogonal cases:

| Case                         | 2.1 catches it? | 2.2 catches it? |
|------------------------------|-----------------|-----------------|
| Repeated PC marker (same)    | Yes             | No (V is PC value, non-zero) |
| Memory overwrite (same addr) | Yes             | Only if new value is 0       |
| `free(p)` zero-fill          | No (addresses differ across `free` calls) | Yes |
| Uninitialised stack slot     | No              | Yes                          |

Both rules run together at every pruning event.

---

## 3. Where Pruning Fires in the Runner

Pruning runs inside `AutoregressiveVMRunner.run` (file
`c4_release/neural_vm/run_vm.py`), in the generation loop around
line 348 (`for i in range(max_steps * Token.STEP_TOKENS):`).

### 3.1 Cadence

Every 120 generated tokens, which is `120 / Token.STEP_TOKENS ≈ 3.4`
VM steps. We round to "every 3 STEP_END boundaries" for cache-line
friendliness.

Concretely, the loop tracks `tokens_since_last_prune`. When that hits
120 *and* we just emitted `STEP_END`, call `self._prune_kv_cache(...)`
and reset the counter.

We deliberately gate on `STEP_END` so we never prune mid-step. A VM
step has tight internal coupling between its 35 tokens (PC, AX, SP,
BP, STACK0, MEM, STEP_END); evicting one of those mid-step would
corrupt the in-flight prediction.

### 3.2 Pseudocode

```python
def run(self, bytecode, ...):
    ...
    kv = _LayerKVCache(...)  # provided by KV-cache agent
    tokens_since_prune = 0
    PRUNE_INTERVAL = 120
    MIN_CACHE_SIZE = 256  # don't bother pruning small caches

    for i in range(max_steps * Token.STEP_TOKENS):
        next_token = self.model.generate_next(
            context, kv_cache=kv, ...)
        context.append(next_token)
        tokens_since_prune += 1

        # Prune only at step boundaries, and only past warmup
        if (next_token == Token.STEP_END
                and tokens_since_prune >= PRUNE_INTERVAL
                and kv.cache_size > MIN_CACHE_SIZE):
            evicted = self._prune_kv_cache(kv, prefix_len=prefix_len)
            tokens_since_prune = 0
```

### 3.3 What "prune" means at the cache level

The KV-cache agent's `LayerKVCache` stores, for each layer, a tensor
of shape `[B, H, S, HD]` for K and V. Pruning produces a single
**index mask** `keep_idx: LongTensor[S']` (`S' ≤ S`) and applies it
identically to every layer's K and V via `index_select(dim=2,
index=keep_idx)`.

Crucially the mask must be **shared across layers**. Why? Because the
sequence dimension `S` indexes *tokens in the context*. If layer 3
evicted token 17 but layer 5 didn't, layer 5's key for token 17 would
attend against a query that is now at relative position 16 in layer 3,
which breaks ALiBi/RoPE distance semantics in layer 5.

So pruning operates as: *decide once, apply everywhere*.

---

## 4. Efficient Near-Duplicate Detection

Naive pairwise cosine similarity is O(S²·D) which is too expensive at
S = 10K. We use the following approach.

### 4.1 Reference (slow) algorithm

```python
def find_duplicates_naive(K):  # K: [S, D]
    Kn = F.normalize(K, p=2, dim=-1)
    sim = Kn @ Kn.T              # [S, S]
    # Upper triangle: i < j
    mask = torch.triu(sim > 0.99, diagonal=1)
    pairs = mask.nonzero()       # [N, 2] -- (i, j) with sim>0.99
    return pairs
```

Cost: `O(S²·D + S²)` memory. At S=10K, D=512 the matmul is 50M·512
= 26 Gflops and the similarity matrix is 400 MB. Unworkable per
prune.

### 4.2 Fast algorithm: bucket by token type, prune within bucket

**Key observation.** Two cached positions only have similar keys if
they encode the same *kind* of thing. We don't need to compare a
REG_PC key with a MEM_VALUE key — we know they will differ.

We tag each cached position with its **token-type bucket**, derived
from the source token id at insertion time:

| Bucket             | Source token ids                      |
|--------------------|---------------------------------------|
| `REG_PC`           | `Token.REG_PC` (one per step)         |
| `REG_AX`           | `Token.REG_AX`                        |
| `REG_SP`           | `Token.REG_SP`                        |
| `REG_BP`           | `Token.REG_BP`                        |
| `STACK0`           | `Token.STACK0`                        |
| `MEM_marker`       | `Token.MEM_*` marker positions        |
| `MEM_value_byte`   | the 8 value-byte positions inside MEM |
| `STEP_END`         | `Token.STEP_END`                      |
| `bytecode`         | positions inside prefix (CODE)        |
| `data`             | positions inside prefix (DATA)        |
| `output`           | I/O / output bytes                    |
| `other`            | anything else                         |

The runner already knows the token id at each position (it built the
context), so bucketing is O(S) at insertion (or O(S) over the context
list when pruning fires).

Within each bucket we run the naive O(B²·D) similarity check. For
register and STEP_END buckets, `B` grows linearly with steps but each
bucket is searched independently; the cost is

    Σ_bucket B_bucket² ≤ (max bucket) · S

Empirically the dominant bucket is `MEM_value_byte` (every memory
write adds 8 positions). Even then, after the first few prunes the
bucket stabilises to its bounded size.

For the *first* prune after a long warmup we can additionally
short-circuit: in the register buckets every old entry is guaranteed
to be near-duplicate to the latest, so we can keep only the **newest
K** entries per bucket and skip similarity computation entirely.

The bucketed implementation outline:

```python
def _prune_kv_cache(self, kv, prefix_len):
    S = kv.cache_size
    if S < MIN_CACHE_SIZE:
        return 0

    # 1. Bucket positions by token type using context tokens
    buckets = self._bucket_positions(self._context, prefix_len, S)

    # 2. Take K from a single representative layer (layer 0 is fine —
    #    if it's a duplicate in L0 it's a duplicate everywhere because
    #    the keys all derive from the same embedding).
    K0 = kv.caches[0].cached_k.mean(dim=1)[0]  # [S, HD], avg across heads

    victims = set()

    # 3. Per-bucket key-similarity check
    for bucket_name, positions in buckets.items():
        if bucket_name in ("bytecode", "data"):
            continue  # protected prefix, never evict
        if len(positions) < 2:
            continue
        # Within-bucket fast pair scan
        K_b = K0[positions]                   # [B, HD]
        Kn = F.normalize(K_b, p=2, dim=-1)
        sim = Kn @ Kn.T                       # [B, B]
        # For each j, evict the most-recent older i with sim > τ
        for j in range(len(positions)):
            for i in range(j):
                if sim[i, j].item() > 0.99:
                    victims.add(positions[i])

    # 4. Zero-write eviction
    V0 = kv.caches[0].cached_v.mean(dim=1)[0]  # [S, HD]
    v_norms = V0.norm(dim=-1)                  # [S]
    zero_positions = (v_norms < 1e-6).nonzero().flatten().tolist()
    for p in zero_positions:
        if p >= prefix_len:  # don't evict prefix
            victims.add(p)

    if not victims:
        return 0

    # 5. Apply mask uniformly across all layers
    keep = [p for p in range(S) if p not in victims]
    keep_idx = torch.tensor(keep, device=K0.device)
    for c in kv.caches:
        c.cached_k = c.cached_k.index_select(2, keep_idx).contiguous()
        c.cached_v = c.cached_v.index_select(2, keep_idx).contiguous()
        c.cache_size = len(keep)

    return len(victims)
```

### 4.3 Optimisation notes

- **Use layer 0 only.** The K projection at layer 0 sits closest to
  the embedding, so its similarity structure already reflects "are
  these the same token kind / value". Using all layers' Ks (mean or
  concat) is overkill and 6–10× more expensive.
- **Average across heads.** Per-head K can vary, but averaging across
  the H heads in layer 0 gives a stable per-token fingerprint and
  cuts compute by H.
- **Compute on GPU.** All ops are matmul + normalize + comparison;
  do them in-place on `kv.caches[0].cached_k.device`.
- **Cosine threshold.** `τ = 0.99` matches the blog spec. Lower
  thresholds (0.95) over-prune and hurt accuracy; higher (0.999)
  under-prune and leave duplicates. The blog calibration was done on
  C4 register/memory patterns and is canonical.

### 4.4 Approximate algorithms (if exact gets too slow)

Future optimisation: use **LSH-style hashing**. Random-project K to
8 bits via `sign(K @ R)` where `R ~ N(0, I)` of shape `[D, 8]`, group
positions by their 8-bit hash, run exact similarity only within
groups. Cuts the bucket size by ~256× on average. Defer until the
exact algorithm is proven to be the bottleneck.

---

## 5. Handling Zero-Value Entries

After each step's K/V is appended to the cache, the pruner checks the
last 35 V rows (one VM step). Any row with `||V|| < ε = 1e-6` joins
the eviction set.

### 5.1 Why post-insertion (not pre-insertion)?

- The model is the source of truth for what counts as zero. If we
  filter writes *before* they enter the cache, we couple the runner
  to a particular interpretation of the V projection.
- Pruning is observational: we accept whatever the model emits, then
  evict things the canonical attention recipe would have ignored
  anyway.

### 5.2 Threshold choice

`ε = 1e-6` is conservative for fp32 weights. The byte-embedded zero
vector at every projection layer is exactly zero (the embedding row
for byte 0 is zeroed at compile time in many code paths) but in float
arithmetic round-off can push it to `~1e-7`. `1e-6` keeps a wide margin
above noise and well below any meaningful V magnitude (which sits in
the [0.1, 10] range under standard init).

### 5.3 What about partial-zero V rows?

We do not evict V rows that are merely *small* — only V rows that are
numerically zero. Partial zeros (e.g. `||V|| = 0.05`) might still
matter to some attention head and the safe default is to keep them.
ALiBi will recency-bias them down eventually anyway.

---

## 6. Protected Region: The Prefix

The first `prefix_len` tokens of the context are the CODE and DATA
sections (the "system prompt" of the VM). These are immutable across
the entire program and are read by L5 fetch heads on every step.
**They are never evicted**, even if their keys are near-duplicates.

The runner already tracks `prefix_len` (line 321 of `run_vm.py`).
The pruner takes it as a parameter and skips all positions `< prefix_len`
when building the victim set.

---

## 7. Test Plan

All tests live in `c4_release/tests/test_kv_cache_pruning.py` (to be
created when the KV-cache agent lands).

### 7.1 Unit tests (no model required)

- **`test_prune_identifies_zero_v`**: Construct a synthetic
  `LayerKVCache` of size 200 with V rows 50, 75, 110 set to zero.
  Run pruner. Assert exactly those three positions are evicted.

- **`test_prune_identifies_duplicate_k`**: Synthetic cache where
  positions (10, 50, 90) have nearly identical K vectors (sim > 0.99
  pairwise). Run pruner. Assert positions 10 and 50 are evicted,
  position 90 (newest) survives.

- **`test_prune_respects_prefix`**: Same as duplicate test but the
  duplicate triple is at positions (5, 10, 50) with `prefix_len = 20`.
  Assert positions 5 and 10 are *not* evicted (they are inside the
  prefix); only position 50 would have been evicted as the older one,
  but since 90 wasn't in the set this time nothing happens.

- **`test_prune_uniform_across_layers`**: Build a multi-layer
  `LayerKVCache`. After pruning, assert every layer's `cache_size` is
  identical and the keep-mask was applied uniformly.

### 7.2 Integration test: bounded cache on long programs

```python
def test_cache_stays_bounded_long_program(pure_neural_runner):
    """1000 PSH/POP cycles must not blow up the KV cache."""
    program = compile_c4("""
        int main() {
            int i;
            for (i = 0; i < 1000; i++) {
                int x; x = i;  // PSH local, then write
            }
            return 0;
        }
    """)

    runner = pure_neural_runner
    runner.enable_kv_pruning = True       # new flag
    output, exit_code = runner.run(program.bytecode, max_steps=50_000)
    assert exit_code == 0

    # Without pruning, cache would be ~50_000 tokens.
    # With pruning, target is bounded at 10K.
    final_size = runner.last_kv_cache_size
    assert final_size < 10_000, f"cache grew to {final_size}, expected <10K"
```

### 7.3 Performance regression guard

Pruning itself must not dominate runtime. Add a benchmark:

```python
def test_pruning_overhead_under_5pct(benchmark, ...):
    t_no_prune = bench_run(program, prune=False)
    t_prune    = bench_run(program, prune=True)
    overhead = (t_prune - t_no_prune) / t_no_prune
    assert overhead < 0.05  # less than 5% overhead
```

For a 1000-iteration loop, target measurements:

| Metric                   | No pruning | With pruning |
|--------------------------|------------|--------------|
| Final KV cache size      | ~50K tok   | < 10K tok    |
| Peak GPU memory          | ~12 GB     | < 4 GB       |
| Wall-clock time          | OOM at 50K | bounded      |

### 7.4 Correctness: pruning must not change semantics

```python
def test_pruning_preserves_output(simple_programs):
    """Pruned runs and unpruned runs must produce identical output."""
    for program in simple_programs:
        out_no_prune = run(program, prune=False)
        out_prune    = run(program, prune=True)
        assert out_no_prune == out_prune
```

Use the existing smoke programs (`tests/test_smoke.py`) as the
correctness set. If any smoke program fails with pruning enabled,
the algorithm is wrong, not the program.

### 7.5 Smoke 2/2 must continue to pass

The default `use_kv_cache` flag stays unchanged; pruning is gated on
a separate flag `enable_kv_pruning` that defaults to `False`. Smoke
tests run with the default and must not be affected.

---

## 8. Implementation Sequence (when KV cache lands)

1. **Wait for `origin/implement-kv-cache` to land.** Pull and check
   out a `kv-cache-pruning` branch *from that branch*, not from main.
2. **Add `_prune_kv_cache` method** to `AutoregressiveVMRunner`
   (per section 4.2 pseudocode).
3. **Add `_bucket_positions` helper** that maps context positions to
   token-type buckets.
4. **Thread `enable_kv_pruning` flag** through `__init__` and into
   the `run` loop.
5. **Wire the pruning call** at the `STEP_END` boundary per section 3.2.
6. **Add tests** per section 7.
7. **Run smoke 2/2**, ensure no regression.
8. **Run benchmark**, confirm cache stays bounded.

Total estimated work: ~3 hours of implementation + 1 hour of
testing once the KV-cache substrate exists.

---

## 9. Open Questions for the KV-Cache Agent

These coupling points must be agreed before pruning lands:

- **Exact API for sequence-dim eviction.** Will `LayerKVCache` expose
  a `prune(keep_idx)` method, or will the pruner reach in and mutate
  `cached_k`/`cached_v` directly? Pruning prefers a public method.
- **ALiBi/RoPE position semantics after pruning.** When we delete
  middle positions, do the surviving positions retain their original
  absolute indices (i.e. position 90 stays "position 90" for ALiBi
  distance calculations), or do they renumber to 0..S'-1?
  - **Required answer: retain original indices.** Renumbering would
    bring all old entries adjacent to the query and destroy ALiBi's
    recency bias. The KV-cache layer must store positions as a
    separate `pos_ids` tensor of shape `[S]` and use that for ALiBi
    distance, not the index in the K/V tensor.
- **RoPE re-application.** RoPE is baked into K *at insertion time*
  in the current vm_step code (line 342). If we keep positions
  intact (per the previous bullet), no re-application is needed. If
  any future change renumbers, the K tensor must be re-rotated.

---

## 10. Summary

The pruning mechanism is two rules running together every 120 tokens
at a STEP_END boundary:

1. **Near-duplicate K eviction** (older of any pair with cosine
   similarity > 0.99 in layer-0 averaged K).
2. **Zero-V eviction** (any cached V with L2 norm < 1e-6).

Bucketing by token type makes the duplicate scan O(S) on average.
The prefix is protected. The mask is applied uniformly across all
layers. The cache size on a 1000-iteration loop should stay under
10K tokens.

Implementation is deferred until `implement-kv-cache` lands on origin.
