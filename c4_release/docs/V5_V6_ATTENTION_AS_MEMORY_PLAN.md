# V5/V6/V10 Retirement Plan — Replace Runner Dicts with Attention-as-Memory

**Status**: Plan only (this file) + first implementation slice (KV cache pruning;
see `KV_CACHE_PRUNING_SPEC.md` §4.2 for the algorithm).
**Scope**: multi-week. This document covers the migration order; the pruner
implementation is the first concrete step (one slice of the broader project).
**Architectural directive**: `BLOG_SPEC.md` lines 3, 814–816 — "no auxiliary
memory or python variables"; KV cache pruning bounds cache size so attention
*is* the storage medium for memory and registers.

---

## 1. What gets retired

Three runner-side Python state mechanisms must die so that **attention is the
only memory** the VM has:

| Tag | Construct | File / lines | Role |
|-----|-----------|--------------|------|
| **V5** | `self._memory: dict[int, int]` (addr → uint8) | `run_vm.py:227, ~75 read/write sites` | Shadow byte memory for LI/LC, PRTF varargs, OPEN/READ stdin path |
| **V6** | `self._mem_history: dict[int, list[int]]` + `_mem_access_order` LRU + `max_mem_history=64` | `run_vm.py:252-253, 1688-1707` + `set_mem_history_end` shim in `neural_embedding.py:457` | 9-token MEM section per addr, LRU-evicted, re-injected into context at every step |
| **V10** | Context rebuild glue: `context[prefix_len:] = mem_flat + list(last_step)` + `set_mem_history_end()` | `run_vm.py:885-890, 1112-1117` | Per-step persistence shim that re-emits `_mem_history` into the live token stream so L15 attention finds historical MEM sections |

The replacement, per `BLOG_SPEC.md` §"KV Cache Pruning" / §"What Gets Evicted":

1. **Writes** (PSH/JSR/ENT/SI/SC) — the model already emits a MEM section at
   the end of each step. With the KV cache landed (commit `42e65f3`), those
   keys/values stay in the cache and are available to future attention
   queries.
2. **Reads** (LI/LC, POP-group, PRTF varargs, OPEN path string, READ stdin
   chunk) — L15 ADDR_KEY-equality attention (or L8 in the STACK0
   phase-2 path) returns the matching MEM val byte.
3. **Eviction** — the cache is bounded by the two-rule pruner in
   `KV_CACHE_PRUNING_SPEC.md` §2: cosine-similarity-`> 0.99` keeps "latest
   write wins", zero-V eviction garbage-collects `free()`.

After V5/V6/V10 die, there is no Python-side mirror of program memory at
all. The KV cache *is* the memory.

---

## 2. Retirement order

The order below intentionally retires V10 first (lightest coupling), then
the V6 LRU, then the V5 byte dict. V5 dies last because the runner-side I/O
shims (`_neural_prtf_emit`, `_neural_open_emit`, `_neural_read_emit`,
`_handle_skipped_io_op`) all currently consult `self._memory` to resolve
pointers — those shims migrate to "read from KV cache" only after
attention-as-memory is fully on for those paths.

### Phase A — KV cache pruning lands (this slice)

**Blocker today**: `_mem_history` exists *because* without pruning the
context grows linearly and L15 attention has to either O(n) over a giant
context or get truncated. The runner solves this by maintaining
`max_mem_history=64` and a context-window of 512 tokens, then re-injecting
the LRU subset every step.

**Replacement**: KV cache pruning per `KV_CACHE_PRUNING_SPEC.md` §4.2. Once
the per-layer K/V tensors are bounded by the two-rule pruner, the runner
can stop windowing/re-injection — old MEM sections that the pruner removed
were already attention-dead (ALiBi penalty + zero-V), so the live attention
output is unchanged.

**Deliverable**: `_prune_kv_cache` method on `AutoregressiveVMRunner`,
gated on a new flag `enable_kv_pruning` defaulting to `False`. Spec-only
in this slice; integration into pure_neural happens in Phase B.

### Phase B — V10 retirement (context rebuild glue)

**Today**, two sites in `_dispatch_step` rebuild the live context every
step:

```python
last_step = context[-(Token.STEP_TOKENS):]
mem_flat = []
for tokens in self._mem_history.values():
    mem_flat.extend(tokens)
context[prefix_len:] = mem_flat + list(last_step)
self.model.embed.set_mem_history_end(prefix_len + len(mem_flat))
```

This is V10 — it exists because:
- The dynamic context after `prefix_len` is *windowed* to "MEM sections we
  decided to keep + the last step".
- L15's K-side `MEM_STORE * 100` anchor + ADDR_KEY equality only matches
  positions where the embedding writes MEM_STORE; that injection is gated
  by `_mem_history_end`, so the runner must keep both in sync.

**Migration step** (after Phase A):

- Enable `enable_kv_pruning=True` in pure_neural. The pruner already bounds
  the K/V tensors; we no longer need the Python-side window.
- Delete the context-rewrite block in both pure_neural and handler-mode
  branches of `_dispatch_step` (`run_vm.py:885-890, 1112-1117`).
- Delete the windowing block in `run`
  (`run_vm.py:502-505`):
  ```python
  if len(context) > prefix_len + 512:
      generation_context = context[:prefix_len] + context[-512:]
  ```
- Delete `set_mem_history_end()` calls (`run_vm.py:455, 890, 1117`).
- Delete `_mem_history_end` field + `_inject_mem_store` guard in
  `neural_embedding.py:447-465` once no caller sets it. The
  `_inject_mem_store` walk over all MEM positions becomes unconditional
  because the KV cache + pruner now hold whatever subset is alive; the
  Python side no longer chooses.

**Test gate**:
- `test_autoregressive_kv_cache_byte_identical.py` must pass with the
  rewrite block removed (cache ON path produces same output as cache OFF).
- Smoke 2/2 (`test_smoke.py::TestSmokeBasic::test_imm_exit`) must pass.

### Phase C — V6 retirement (`_mem_history` LRU)

Once V10 is gone, `_mem_history` no longer feeds anything in the live
context — the only callers left are bookkeeping (`_track_mem_access`
appends to it) and the runner's debug introspection (none in the smoke
path).

**Migration step**:

- Delete `_track_mem_access` (`run_vm.py:1688-1707`).
- Delete `_inject_mem_section` (`run_vm.py:1661-1686`).
- Delete the V6 fields in `__init__` and `run` reset
  (`run_vm.py:252-253, 453-454`).
- Delete the constructor arg `max_mem_history` and its docstring entry.

**Test gate**: smoke 2/2 + full byte-identical KV cache suite.

### Phase D — V5 retirement (`_memory` byte dict)

This is the deepest cut. `_memory` has ~75 read/write sites:

- **Direct writes** (~10 sites): `_memory[addr] = byte`, `_mem_store_word`,
  data-section bootstrap in `run` (`run_vm.py:477, 480`), heap zero-fill
  in `_neural_read_emit` (`run_vm.py:1540`), etc.
- **Direct reads** (~15 sites): `_memory.get(addr, 0)`, `_mem_load_word`,
  used by:
  - LI/LC handler branch (`run_vm.py:1051, 1058`) — already disabled in
    pure_neural (L15 lookup handles it).
  - PRTF format-string walker (`run_vm.py:551, 1421-1422, 1746-1764`,
    `_read_string`) — reads the format string starting at `fmt_ptr`.
  - PRTF vararg slot resolution (`run_vm.py:1443, _mem_load_word`).
  - OPEN path-string walker (`_neural_open_emit` → `_read_string`).
  - READ stdin chunk write (`_neural_read_emit:1540`).
  - Heuristic checks (`if cand in self._memory`,
    `if s0 in self._memory`).

Each of these is a host-side I/O escape hatch — they cross the VM/host
boundary, so they aren't pure-neural anyway. Per `BLOG_SPEC.md` §"Tool Use
Mode" line 851, neural I/O mode handles these via the
THINK_START/THINK_END protocol; tool-call mode handles them via the
external runner. In *neither* mode does the runner need a Python-side
shadow memory once the model can read its own KV cache for the
format-string/path-pointer/buffer addresses.

**Migration step** (multi-week, parallel to the conversational-IO
neural-side completion):

1. **Format-string reads** (`_read_string`): rewrite to attend the model
   instead of reading `self._memory`. Concretely, the runner calls the
   model with a query token + addr (similar to a "fake LC instruction"),
   reads the byte off REG_AX, walks until 0. Slow but pure. Alternative:
   defer until the conversational-IO neural-side `_neural_prtf_emit`
   migration retires PRTF as a runner shim entirely.
2. **Stdin/READ buf writes**: in neural I/O mode (`conversational_io=True`),
   the model reads `USER_INPUT_START..USER_INPUT_END` via attention
   already (per `BLOG_SPEC.md`:851 nibble cascade). Delete the runner-side
   buffer write at `run_vm.py:1540` once the neural READ path is
   confirmed correct on the smoke set.
3. **Heuristic `cand in self._memory`** in `_neural_prtf_emit`,
   `_neural_open_emit`: replace with "does the model think this is a
   valid pointer" check (a small attention head that returns 1 if any
   MEM section has matching ADDR_KEY). Or simpler: in neural-I/O mode,
   never run these heuristics at all — the model emits the path/format
   string into the token stream directly.
4. **Data section bootstrap** (`run_vm.py:477, 480`): replace with
   "encode data bytes as MEM-section tokens in the prefix" so the model
   can attend them directly. The current bootstrap is a *write-only*
   shadow used only by the host-side shims (1)-(3), so it dies with
   them.

**Test gate**: full `test_smoke.py` + `test_smoke_pure_neural.py` pass
without any reference to `self._memory` in the runner.

---

## 3. How KV pruning composes with mem-attention

This section answers the question "do we double-evict?".

The two systems operate at different layers but are coherent:

| System | Layer | What it evicts |
|--------|-------|----------------|
| `_mem_history` LRU (V6, dying) | Python-side context token list | Oldest unique address beyond `max_mem_history` |
| KV cache pruning (this slice) | Per-layer K/V tensors | Position-level (older of any K cos-sim > 0.99 pair, or zero-V) |

After Phase A lands, both systems are active. They don't conflict because:

- The Python-side LRU controls which MEM sections appear in the *input*
  token stream. The model's embedding writes MEM_STORE only at those
  positions (gated by `_mem_history_end`).
- The pruner runs on the *output* of attention layers (K/V tensors).
  When a position the Python side kept gets pruned at the K/V level,
  the model still sees the token id in input, but the attention key for
  it is gone from the cache, so future queries miss it. That's fine: any
  K/V the pruner removes was already attention-dead.

After Phase B (V10 dies), the Python side no longer windows the input
context — every emitted MEM section stays in the token stream forever,
but the pruner removes its K/V entry once a fresher write replaces it.
End-state behavior matches the spec exactly:
> *"Eviction runs automatically every 120 tokens (~3 VM steps), keeping
> the cache at 1–10K tokens"* — `BLOG_SPEC.md`:816.

---

## 4. This-slice deliverable

This document plus the first implementation step:

**KV cache pruner** in `c4_release/neural_vm/kv_cache_pruning.py`, per
`KV_CACHE_PRUNING_SPEC.md` §4.2. The pruner:

1. Operates on a `LayerKVCache` (the structure landed in commit `42e65f3`).
2. Implements both eviction rules: key-similarity (`τ = 0.99`) bucketed by
   token type, plus zero-V (`ε = 1e-6`).
3. Respects the `prefix_len` protected region (CODE + DATA).
4. Applies the keep-mask uniformly across all layers.
5. Returns the number of evicted positions.
6. Is **not** wired into the `run` loop yet. Wiring is Phase B in this
   plan and requires careful integration testing on long programs;
   the slice here just makes the pruner available + tested.

Unit tests in `c4_release/tests/test_kv_cache_pruning.py` cover the four
spec §7.1 cases (zero-V identification, duplicate K identification,
prefix protection, layer-uniform mask application).

**What this slice does NOT do**:

- It does not delete V5/V6/V10. Those are Phases B-D above.
- It does not switch `enable_kv_pruning` on by default. The default
  remains off; the runner-side mem path is unchanged.
- It does not change `BLOG_SPEC.md` (canonical) or `KV_CACHE_PRUNING_SPEC.md`
  (the algorithm spec it implements).

---

## 5. Open questions

These don't block the pruner slice but are coupling points for Phase B
(V10 retirement):

- **When pruning removes middle positions, do ALiBi distances see
  surviving positions at their original indices or 0..S'-1?**
  - `KV_CACHE_PRUNING_SPEC.md` §9 requires "retain original indices".
    The current `LayerKVCache` in `kv_cache.py:114-151` does not store
    per-position absolute indices — it relies on the tensor's sequence
    dimension for ALiBi distance via the attention forward in
    `vm_step.py:260-470`.
  - **Required follow-up before Phase B**: add a `pos_ids` tensor to
    `TransformerKVCache` and thread it through ALiBi/RoPE distance
    computation. The pruner returns `keep_idx`; the cache also stores
    `pos_ids[keep_idx]` for future distance math.
  - Without this, pruning silently moves the older entries closer to
    the query in distance space and the recency bias breaks.

- **First-step prune cadence**. Spec says "every 120 tokens after warmup".
  In a fresh VM run, the first 120 tokens are mostly prefix (CODE/DATA)
  which the pruner protects. The first useful prune happens after
  ~3.4 VM steps. For very short programs (<3 steps) the pruner never
  fires, which is the desired behavior.

- **Where to call the pruner from**. Spec §3.2 says inside the `run`
  loop, gated on `STEP_END` boundary. The current loop is at
  `run_vm.py:499-668`; the natural insertion point is right after
  `_dispatch_step` returns False.

---

## 6. Cross-references

- `BLOG_SPEC.md` lines 3, 800-816 — canonical specification.
- `KV_CACHE_PRUNING_SPEC.md` — algorithm spec (lands today).
- `run_vm.py:225-260` — current V5/V6/V10 fields.
- `run_vm.py:879-890, 1102-1117` — V10 context rewrite glue (the two
  symmetric sites that re-emit `_mem_history` into the live context).
- `run_vm.py:1655-1714` — V5 byte-store/load helpers.
- `run_vm.py:1688-1707` — V6 LRU eviction logic.
- `neural_embedding.py:447-465` — V10's `_mem_history_end` consumer.
- `kv_cache.py:114-151` — `LayerKVCache` (the structure the pruner
  operates on).
- `key_similarity_eviction.py` — existing eviction primitive that
  works on the *context token list*, not per-layer K/V tensors. The
  pruner here is the per-layer K/V analogue.
- `STACK0_VIA_MEM_ATTENTION_PLAN.md` — parallel migration (STACK0 token
  slot → MEM attention) that retires another category of "read from
  Python state" by routing through MEM attention.
