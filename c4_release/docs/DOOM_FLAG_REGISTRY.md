# DOOM `C4_*` FLAG REGISTRY (c4_min)

Canonical inventory of every production `C4_*` environment flag read at runtime in
`c4_release/c4_min/` — the doom perf-fleet + pure-forward VM.  Companion to
[`FLAG_REGISTRY.md`](FLAG_REGISTRY.md), which covers `neural_vm/` + `tools/` ONLY and
has ZERO `c4_min`/doom coverage (the gap the 2026-08-05 audit found: **107 of 109
c4_min doom flags were undocumented** in the canonical registry).

Source of truth: the per-flag `*_enabled()` docstring at each `os.environ.get` site +
[`c4_min/DOOM_PARAM_PRECISION_FLAG_AUDIT_2026_08_05.md`](../c4_min/DOOM_PARAM_PRECISION_FLAG_AUDIT_2026_08_05.md)
+ the `REALTIME_SETUP.md` recipe.  Generated 2026-08-05; golden RE-BASELINED 2026-08-06
to `7d4afe61` (the `C4_BP_RESTORE_HIBYTE` DEFAULT-ON flip; `069cc32f` is the
`C4_BP_RESTORE_HIBYTE=0` escape-hatch build).

Scope note: this lists the ~113 PRODUCTION flags read in non-agent, non-bench, non-test
`c4_min` modules.  The `_agent_*` / `bench_*` scratch scripts read many more one-off
knobs (`C4_TEST_*`, `C4_RUN_*`, `C4_WALL6_*`, `C4_*_DEVICE`, `C4_PEAK_TFLOPS`,
`C4_DEDUP_WEIGHTS` [C-runtime `-D` define only], …) that are NOT production flags and
are deliberately omitted.

---

## Columns / conventions

- **DEFAULT** — value read when the flag is unset.  `unset→off` == `== "1"` style
  (absent ⇒ False).
- **DEPENDS** — flags that must ALSO be on for this one to have effect (the audit's
  undocumented dependency chains).  `—` = standalone.
- **GOLDEN** — impact on the byte-identity gate.  The current default golden fingerprint
  (`c4_min._fingerprint_build`, `build_compact_sparse_streaming`, non-recurrent long-div)
  is **`7d4afe61`** (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON); the pre-fix **`069cc32f`** is the
  `C4_BP_RESTORE_HIBYTE=0` escape-hatch build.
  - **`byte-exact`** — DEFAULT-OFF runtime/perf lever; the golden fingerprint is
    UNCHANGED whether the flag is on or off.  The overwhelming majority.  These are
    compute-fusion / schedule / attention levers proven L-inf=0.
  - **`doom-build`** — a BUILD-FAMILY selector for the DOOM/perf-fleet recurrent build
    (`build_lib_model_streaming` / `qwen_full_vm`).  It CHANGES the doom-build weights
    (a different, wider ISA — 32-bit ALU / wide addr / lean-divide / recurrent core),
    but does NOT touch the `build_compact_sparse_streaming` golden gate, so the golden
    is unchanged.  These pick the semantics of the doom run itself; several are
    mutually-exclusive or interacting build-family selectors.
  - **`golden-MOVING`** — flipping this MOVES the golden fingerprint itself (a
    hi/lo-nibble recompose width / wider OP_IS band / repacked block-0).  Now 3:
    `C4_BP_RESTORE_HIBYTE` (DEFAULT-ON — it OWNS the current `7d4afe61` golden, and
    `=0` rolls back to `069cc32f`), `C4_INGEST_WIDE`, `C4_DOOM_DRAWSPAN` (the latter two
    DEFAULT-OFF).  ON hashes are documented in `_fingerprint_build.py`.

---

## ★ The `C4_DOOM_FAST=1` composite (the one-toggle fast stack)

`C4_DOOM_FAST=1` turns on the whole byte-exact winning fast-doom stack with ONE flag
(module `c4_min/doom_fast.py`, helper `expand_doom_fast()`).  Every member is a
DEFAULT-OFF `byte-exact` flag below, so the composite is golden-safe (the default golden
`7d4afe61` unchanged unset AND set).  Members are set with `setdefault` semantics — a member you
set BY HAND (incl. `=0` to opt out) always wins.  The 19 members are:

```
C4_DEAD_BLOCK_FUSION=1  C4_FUSED_MEGABLOCK=1  C4_FFN_FUSED_HIDDEN=1
C4_FFN_WAVE_BATCH=1  C4_MEGABLOCK_BLOCK_K=512   # ← wave-batch is INERT without a bigger block_k
C4_DIRECT_CAM_BATCHED=1  C4_DIRECT_LOCAL_CAM=1  C4_DIRECT_CAM_VEC=1
C4_FLASH_ATTN=1  C4_BANDED_LOCAL_ATTN=1
C4_PRECOMPUTED_SCHEDULE=1  C4_ONCHIP_RESIDUAL=1  C4_RESIDENT_BATCH=1
C4_ATTN_MEGABLOCK=1  C4_BLOCK0_DK=1
C4_SCHED_FAST_BUILD=1  C4_SCHED_GPU_BUILD=1  C4_SCHED_PIPELINE=1  C4_SCHED_CACHE_RESOLVED=1
```

The composite sets ENV only; the attention levers still require the harness to CALL
`tight_attn_compose.install_composed(model)` on the built model (the flags are read at
forward time by the rebound `.forward`s).  See `c4_min/doom_fast.py` for the full
rationale + what it deliberately does NOT set (`C4_PF_CFM` build-mode, `C4_SCHED_CHUNK`
frame-shape knob, `C4_FFN_LINFOLD` honest-trade, `C4_MEGABLOCK_RESID` bf16).

---

## Dependency chains (the audit's undocumented deps)

```
C4_BLOCK0_DK          → C4_ATTN_MEGABLOCK
C4_ATTN_MEGABLOCK     → C4_PRECOMPUTED_SCHEDULE + C4_ONCHIP_RESIDUAL + C4_FUSED_MEGABLOCK
C4_ONCHIP_RESIDUAL    → C4_PRECOMPUTED_SCHEDULE
C4_RESIDENT_BATCH     → C4_PRECOMPUTED_SCHEDULE
C4_SCHED_GPU_BUILD    → C4_SCHED_FAST_BUILD + C4_ONCHIP_RESIDUAL
C4_SCHED_PIPELINE     → C4_SCHED_GPU_BUILD
C4_FFN_WAVE_BATCH     → a bigger C4_MEGABLOCK_BLOCK_K (INERT at the default 64; the stack uses 512)
C4_FAITHFUL_PRECOMPUTE_THREAD → the pipelined schedule builder (C4_SCHED_PIPELINE path)
```

## Overlapping megakernel levers + precedence

Several flags fuse/graph FFN spans and OVERLAP.  Precedence (from
`pf_speculative.py:1663` and the module docstrings):

- `C4_FUSED_MEGABLOCK` — the on-chip fused dead-FFN chain (the primary lever; the
  `C4_DOOM_FAST` member).
- `C4_FUSED_FFN_MEGAKERNEL`, `C4_GRAPH_MEGAKERNEL`, `C4_GRAPH_BLOCK0`,
  `C4_BLOCK0_FUSED_FFN` — earlier/alternative FFN-span fusion+graphing levers.  One
  "takes PRECEDENCE over `C4_GRAPH_MEGAKERNEL`" (see `pf_speculative.py`).  Do NOT stack
  these with `C4_FUSED_MEGABLOCK` expecting additive gains — they target the same span.
- `C4_MEGABLOCK_DOOM_LEAN` — the doom-lean variant selector for the megablock region.

## The `C4_WHOLE_STEP_GRAPH` vs `C4_WHOLESTEP_GRAPH` COLLISION (footgun)

Two DIFFERENT flags differing only by one underscore — a genuine naming footgun:

| flag | module | what it does |
|---|---|---|
| **`C4_WHOLE_STEP_GRAPH`**  | `whole_step_graph.py:80` | collapse **block-0's per-chunk S-chunk loop** (ingest-gather scatter + W_o + FFN) into ONE CUDA-graph replay + sync-free scatter per chunk. |
| **`C4_WHOLESTEP_GRAPH`** (no 2nd underscore) | `pf_kbatch.py:546` | collapse each maximal **FFN-only block SPAN** between attention blocks into ONE fixed-shape masked CUDA graph (the #880 whole-step launch-collapse). |

They are read by SEPARATE `*_enabled()` helpers in separate modules and mean DIFFERENT
things; there is NO alias between them (adding one would silently conflate two distinct
levers — DO NOT).  Both are DEFAULT-OFF `byte-exact` and neither is a `C4_DOOM_FAST`
member.  If you touch one, GREP for the other first.  A future rename should pick ONE
canonical name and alias the other with a deprecation shim (out of scope here — pure
docs task).

---

## Flag table

### The fast-doom composite + members

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_DOOM_FAST` | ★ composite: turn on the whole byte-exact fast-doom stack (19 members) with one toggle | unset→off | — | byte-exact |
| `C4_DEAD_BLOCK_FUSION` | fuse 238 dead-attention blocks → `output=x` (the "divfree" block-skip; no Q/K/V/O/KV) | unset→off | — | byte-exact |
| `C4_FUSED_MEGABLOCK` | on-chip fused dead-FFN chain — one CUDA-graph launch, residual resident in L2 | unset→off | — | byte-exact |
| `C4_FFN_FUSED_HIDDEN` | keep each dead-FFN block's `[Dff,K]` hidden on-chip (no HBM round-trip) | unset→off | `C4_FUSED_MEGABLOCK` | byte-exact |
| `C4_FFN_WAVE_BATCH` | horizontal wave-batching of the dead-FFN chain (49 pairs → ~27 wave-kernel pairs) | unset→off | bigger `C4_MEGABLOCK_BLOCK_K` (INERT at default 64) | byte-exact |
| `C4_FFN_LINFOLD` | VERTICAL linear-fold of consecutive dead-FFN blocks (honest FLOP-for-occupancy trade; fill-in) | unset→off | `C4_FUSED_MEGABLOCK` | byte-exact |
| `C4_MEGABLOCK_BLOCK_K` | dead-FFN megakernel K-tile size (perf knob; the stack uses 512, default 64) | (empty)→64 | — | byte-exact |
| `C4_MEGABLOCK_RESID` | resident residual dtype (`fp32` default byte-exact; `bf16` decode-safe ~0 gain; `fp16` raises) | fp32 | `C4_FUSED_MEGABLOCK` | byte-exact (fp32); bf16 non-golden research |
| `C4_MEGABLOCK_DOOM_LEAN` | doom-lean variant of the megablock region | unset→off | megablock path | byte-exact |
| `C4_ATTN_MEGABLOCK` | attention-block analog of the FFN megablock (fused `[D,K]` live-block region) | unset→off | `C4_PRECOMPUTED_SCHEDULE`+`C4_ONCHIP_RESIDUAL`+`C4_FUSED_MEGABLOCK` | byte-exact |
| `C4_BLOCK0_DK` | fold block-0 directly into the `[D,K]` fused region (kills the last input transpose, ~19%) | unset→off | `C4_ATTN_MEGABLOCK` | byte-exact |
| `C4_DIRECT_CAM_BATCHED` | batched direct-CAM gather (O(1) memory reads vs softmax-over-stores) | unset→off | — | byte-exact |
| `C4_DIRECT_LOCAL_CAM` | direct-CAM for the live LOCAL blocks | unset→off | — | byte-exact |
| `C4_DIRECT_CAM_VEC` | GPU-vectorized direct-CAM gather | unset→off | — | byte-exact |
| `C4_DIRECT_CAM_LIVE_LOCAL` | live-local direct-CAM path | 1 (on) | — | byte-exact |
| `C4_DIRECT_CAM_READ` | direct-CAM data read (the O(1) global-CAM read lever) | unset→off | — | byte-exact |
| `C4_DIRECT_CAM_VERIFY_ADDR` | address-verify guard on the direct-CAM read | unset→off | direct-cam | byte-exact |
| `C4_FLASH_ATTN` | flash/online-softmax1 attention (no `[Sq,Sk]` score tensor materialises) | unset→off | — | byte-exact |
| `C4_BANDED_LOCAL_ATTN` | banded sliding-window local attention (score only the last-W keys, ingest=30) | unset→off | — | byte-exact |
| `C4_BANDED_SQ_CHUNK` | banded-attn Sq chunk size | 8192 | `C4_BANDED_LOCAL_ATTN` | byte-exact |
| `C4_PRECOMPUTED_SCHEDULE` | precompute the whole K-batch schedule once → one graph-per-chunk dispatch | unset→off | — | byte-exact |
| `C4_ONCHIP_RESIDUAL` | residual-on-chip: kill the dense `W_o.linear` GEMMs, scatter-add W_o deltas | unset→off | `C4_PRECOMPUTED_SCHEDULE` | byte-exact |
| `C4_RESIDENT_BATCH` | whole-K-batch resident single-graph replay (no per-chunk `.copy_`) | unset→off | `C4_PRECOMPUTED_SCHEDULE` | byte-exact |
| `C4_SCHED_FAST_BUILD` | fast schedule build path | 1 (on) | — | byte-exact |
| `C4_SCHED_GPU_BUILD` | vectorized numpy + GPU-gather schedule build (collapse the 3-stage dict pipeline) | unset→off | `C4_SCHED_FAST_BUILD`+`C4_ONCHIP_RESIDUAL` | byte-exact |
| `C4_SCHED_PIPELINE` | double-buffered background-thread schedule builder (build overlaps dispatch) | unset→off | `C4_SCHED_GPU_BUILD` | byte-exact |
| `C4_SCHED_CACHE_RESOLVED` | cache the resolved CAM tables across frames (build→copy) | unset→off | schedule path | byte-exact |
| `C4_SCHED_CHUNK` | dispatch chunk size (frame-shape knob) | 4096 | schedule path | byte-exact |
| `C4_FAITHFUL_PRECOMPUTE_THREAD` | run the faithful value-verify precompute on a 2nd build thread | unset→off | pipelined builder | byte-exact |

### Other FFN / megakernel / graph fusion levers

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_FUSED_FFN_MEGAKERNEL` | earlier FFN-span fused megakernel | unset→off | — | byte-exact |
| `C4_GRAPH_MEGAKERNEL` | CUDA-graph the FFN megakernel span | unset→off | — | byte-exact |
| `C4_GRAPH_BLOCK0` | CUDA-graph block-0's S-chunked ingest attn+FFN | unset→off | — | byte-exact |
| `C4_BLOCK0_FUSED_FFN` | fuse block-0's dense FFN | unset→off | — | byte-exact |
| `C4_BLOCK0_SQ_CHUNK` | block-0 Sq chunk size | unset→off | — | byte-exact |
| `C4_BLOCK0_DROP_DEAD_KV` | drop dead KV rows in block-0 | 1 (on) | — | byte-exact |
| `C4_FUSED_DELTA_FFN` | fused-delta sparse-COO FFN kernel | unset→off | — | byte-exact |
| `C4_BLOCK_SPARSE_FFN` | block-sparse FFN forward | unset→off | — | byte-exact |
| `C4_WHOLE_STEP_GRAPH` | block-0 S-chunk-loop → one graph (see COLLISION above) | unset→off | — | byte-exact |
| `C4_WHOLESTEP_GRAPH` | FFN-only block-span → one masked graph (see COLLISION above) | unset→off | — | byte-exact |
| `C4_CODEGEN_FUSE` | codegen-fused kernel path | unset→off | — | byte-exact |
| `C4_CODEGEN_FUSE_ADDR_MASK` | addr-mask variant of codegen-fuse | unset→off | `C4_CODEGEN_FUSE` | byte-exact |
| `C4_MATERIALIZE_IDEMPOTENT` | idempotent materialize of sparse forward | 1 (on) | — | byte-exact |

### Direct-CAM / eviction / KV / block-skip

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_STEP_BLOCK_SKIP` | per-op live-block NAME schedule (older skip; NOT the megablock stack) | unset→off | — | byte-exact |
| `C4_BATCHED_BLOCK_SKIP` | batched precomputed per-op live-mask over blocks | unset→off | — | byte-exact |
| `C4_KV_STACK` | KV-backed stack pop (arbitrary depth, latest-write-wins); `=0` = 1-slot mirror | 1 (on) | — | byte-exact |
| `C4_FROZEN_ROW_SKIP` | skip frozen (unchanged) rows in the verify | unset→off | — | byte-exact |
| `C4_EXACT_EVICT` | O(steps) last-read+1 liveness KV eviction schedule | unset→off | — | byte-exact |
| `C4_EVICT_SCHEDULE` | eviction schedule path | unset→off | — | byte-exact |
| `C4_EVICT_TIMED` | drain/sync around each evict (timing) | unset→off | — | byte-exact (tooling) |
| `C4_STACK_POP_FREE` | pop-free stack schedule | unset→off | — | byte-exact |
| `C4_KV_PRUNE_FULL_COSINE` | full-cosine KV prune | unset→off | — | byte-exact |
| `C4_FAITHFUL_ATTN_EVICT` | faithful attention eviction | unset→off | — | byte-exact |
| `C4_FAITHFUL_PRECOMPUTE_CACHE` | cache the faithful precompute | unset→off | — | byte-exact |
| `C4_FAITHFUL_SINGLE_DISPATCH` | single-dispatch faithful verify | unset→off | — | byte-exact |
| `C4_HASH_CAM` | hash-CAM address resolution (faithful single dispatch) | unset→off | — | byte-exact |
| `C4_PACK_MEMCAM` | packed mem-CAM | unset→off | — | doom-build |
| `C4_UNIFY_CAM_HEAD` | unify the CAM heads | unset→off | — | doom-build |
| `C4_UNIFY_CAM_ONE` | unify to one CAM head | unset→off | — | doom-build |
| `C4_LIVE_HEADS_MEMO` | memoize the live-head set | 1 (on) | — | byte-exact |
| `C4_SELFEMU_DIRECT_CAM` | self-emulation direct-CAM (global-CAM softmax → O(1) gather) | unset→off | — | byte-exact |

### Speculative / decode / overlay / pos-sparse

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_BATCHED_DECODE` | batched speculative decode | unset→off | — | byte-exact |
| `C4_DRAFT_CMP32` | 32-bit cmp in the draft path | unset→off | — | byte-exact |
| `C4_GPU_VERIFY` | GPU verify path | unset→off | — | byte-exact |
| `C4_OVERLAY_BATCHED` | batched overlay | unset→off | — | byte-exact |
| `C4_OVERLAY_PRECOMPUTE` | precompute the overlay | 1 (on) | — | byte-exact |
| `C4_POS_SPARSE` | position-sparse bounded forward | unset→off | — | byte-exact |
| `C4_CUT_SPAN_CHUNK` | span-cut chunking | unset→off | — | byte-exact |
| `C4_QROW_CHUNK` | query-row chunking | ? | — | byte-exact |
| `C4_STREAM_EMBED` | stream the embed (lean) | unset→off | — | byte-exact |
| `C4_LEA_INT_SNAP` | integer-snap the LEA address in pos-sparse | unset→off | — | byte-exact |
| `C4_LEA_Q_SNAP` | LEA query snap (the 296-blocker fix) | 1 (on) | — | byte-exact |
| `C4_VANILLA_TOMBSTONE` | vanilla-tombstone KV | unset→off | — | byte-exact |

### Multi-GPU

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_MULTIGPU_KSPLIT` | split the K-batch across GPUs | unset→off | — | byte-exact |
| `C4_MULTIGPU_FRAMES` | frame-parallel across GPUs | unset→off | — | byte-exact |

### Precision (see audit Q3 — no unified `C4_PRECISION` selector exists)

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_FP64_ALL` | run ALL blocks in fp64 | unset→off | — | byte-exact (superset of selective) |
| `C4_FP64_BLOCK_NAMES` | comma-sep block names to run fp64 (override `FP_FRAGILE_BLOCK_NAMES=("lea-addr-nib",)`) | (`lea-addr-nib`) | — | byte-exact |
| `C4_FP32_ALU` | fp32 ALU baked path | unset→off | — | doom-build |
| `C4_FLOAT_OPS` | enable float ALU ops in the ISA | unset→off | — | doom-build (widens NUM_OPS) |
| `C4_FIXED_MAC_DISCRETE` | native fixed-point discrete MAC | unset→off | — | doom-build |

### Build-family selectors (DOOM/perf-fleet build — `doom-build`, golden gate unchanged)

These change the DOOM recurrent build's ISA/semantics (`build_lib_model_streaming` /
`qwen_full_vm`) but not the `build_compact_sparse_streaming` golden gate, so the default
golden `7d4afe61` is unchanged.  Several are MUTUALLY-EXCLUSIVE or interacting build-family selectors
(the audit's Q4(d)) — e.g. `C4_DIV_LONGDIV` overrides `C4_DIV_LEAN`.

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_VM_WIDTH32` | 32-bit VM word width | unset→off | — | doom-build |
| `C4_VM_TWO_LIMB` | two-limb wide VM | ? | — | doom-build |
| `C4_WIDTH_NARROW` | narrow-width variant | unset→off | — | doom-build |
| `C4_PC_WIDE` | wide PC register | unset→off | — | doom-build |
| `C4_SP_WIDE` | wide SP register | unset→off | — | doom-build |
| `C4_GLOBAL_ADDR32` | 32-bit global addressing | unset→off | — | doom-build |
| `C4_CODE_ADDR_BITS` | code address width (bits) | 12 | — | doom-build |
| `C4_MEM_ADDR_BITS` | memory address width (bits) | ? | — | doom-build |
| `C4_IMM_NIBS` | immediate operand nibble count | 5 | — | doom-build |
| `C4_CMP32` | 32-bit compare | unset→off | — | doom-build |
| `C4_CMP32_ORDER` | 32-bit compare byte-order fix (#826) | 1 (in doom scripts) | `C4_CMP32` | doom-build |
| `C4_SHIFT32` | 32-bit shift | unset→off | — | doom-build |
| `C4_DIVMOD_SIGNED` | signed divmod | unset→off | — | doom-build |
| `C4_DIV_LEAN` | lean divide gadget (default doom divide) | 1 (on) | — | doom-build |
| `C4_DIV_LONGDIV` | long-division divide gadget (OVERRIDES `C4_DIV_LEAN`) | unset→off | — | doom-build |
| `C4_LOGSINK_DIV` | fp64 log-sink divide variant | unset→off | — | doom-build |
| `C4_LOGSINK_DIV_FP32` | fp32 approximate-refine log-sink divide | unset→off | `C4_LOGSINK_DIV` path | doom-build |
| `C4_MUL_LOOKAHEAD` | lookahead multiply gadget | 1 (on) | — | doom-build |
| `C4_CONST_OPERAND` | const-operand fold | unset→off | — | doom-build |
| `C4_MEM_OPERAND` | memory-operand fold | unset→off | — | doom-build |
| `C4_KB_BATCHED` | KBatch-batched ALU | 1 (on) | — | doom-build |
| `C4_RECURRENT_CORE` | recurrent core (weight-tie the divmod loop) | unset→off | — | doom-build |
| `C4_BARREL_SHIFT` | barrel-shifter shift path | unset→off | — | doom-build |
| `C4_BARREL_UNIFY` | unify barrel-shift bitwise | unset→off | — | doom-build |
| `C4_TIGHT_SHIFT` | tight-nibble shifter | 1 (on) | — | doom-build |
| `C4_INGEST_GQA` | grouped-query ingest attention | unset→off | — | doom-build |

### Golden-MOVING (moves the fingerprint itself)

The current default golden is **`7d4afe61`** (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON, 2026-08-06).
The pre-fix golden **`069cc32f`** is the `C4_BP_RESTORE_HIBYTE=0` escape-hatch build.

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_BP_RESTORE_HIBYTE` | LEV/pop scalar recompose reads fp32-safe 5 nibbles (not fp64-only 8) — general-program-correctness fix; makes Mandelbrot fully byte-exact, doom stays byte-exact | **1 (on)** | — | **golden-MOVING** (DEFAULT-ON golden = `7d4afe61`; `=0` reverts to the pre-fix `069cc32f` escape-hatch build) |
| `C4_INGEST_WIDE` | 1-query/1-KV wide ingest — restructures block-0 to a 1-head packed-dim Attn | unset→off | — | **golden-MOVING** (ON hash ≠ default golden; packed-dim by construction) |
| `C4_DOOM_DRAWSPAN` | native DRAWSPAN render-macro opcode (47) — widens NUM_OPS 40→48 | unset→off | — | **golden-MOVING** (wider OP_IS band; ON `2f69350f…` was measured on the `069cc32f` base, shifts under the new default) |
| `C4_DOOM_DRAWCOL` | native DRAWCOL/DRAWSPANF gameplay render-macro opcodes (48/49) — 3D analog of the DRAWSPAN patch fold; widens NUM_OPS | unset→off | — | **golden-MOVING** (wider OP_IS band by construction; default-OFF golden `069cc32f`/`7d4afe61` unchanged) |

### Golden-migration CANDIDATES (general-correctness, DEFAULT-OFF today)

Levers that fix a general-program correctness gap and are NOT yet default-ON, but ARE
migration candidates — the same "gate-and-flip" precedent `C4_BP_RESTORE_HIBYTE` just set
(it moved `069cc32f → 7d4afe61`). Flipping one of these to DEFAULT-ON would MOVE the golden
again (the fingerprint changes because the ALU semantics change).

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_DIVMOD_SIGNED` (#790; canonical env flag — the doc/capstone also calls it `C4_SIGNED_DIVMOD`, an alias with NO separate env read) | signed trunc-toward-zero `DIV`/`MOD` (C4/C convention) instead of the golden UNSIGNED base-16 floor divide — negate each negative operand to its magnitude before the divide, conditionally negate quotient (`SGN_A^SGN_B`) / remainder (`SGN_A`) after, via a two's-complement carry chain over the operand/result nibble bands. **Closes the last C90 failure class** (signed-negative `DIV`/`MOD`, the only non-100% conformance class). | unset→off | — | **golden-migration candidate** (OFF → DIV/MOD stay unsigned-floor → golden byte-IDENTICAL `7d4afe61`; flipping DEFAULT-ON would MOVE the golden, like `C4_BP_RESTORE_HIBYTE` did) |

> Note: the CANONICAL env flag the code actually reads is `C4_DIVMOD_SIGNED`
> (`nibble_alu32.py::_signed_divmod_enabled`, `nibble_pure_forward_complete.py`) — it also
> appears in the Build-family selectors table above as the `doom-build` DIV/MOD-semantics
> knob. It is repeated here because, as the general-correctness lever that closes the last
> C90 class, it is a golden-migration candidate, not merely a doom-build selector.
> `C4_SIGNED_DIVMOD` is ONLY a general-scope doc/capstone alias name; there is no
> separate env read for it — always set `C4_DIVMOD_SIGNED=1`.

### CFM / build-mode / I/O / misc

| flag | purpose | default | depends | golden |
|---|---|---|---|---|
| `C4_PF_CFM` | LEAN CFM streaming build (~1.6 GB peak) — build-MODE, not a perf lever | unset→off | — | byte-exact (build-mode) |
| `C4_CFM_EMIT` | CFM emit path | unset→off | — | byte-exact |
| `C4_DOOM_DRAWSPAN` | (see golden-MOVING) | | | |
| `C4_ALLOW_DENSE_BUILD` | allow a dense (non-streaming) build (memory guard override) | unset→off | — | byte-exact (build-mode) |
| `C4_VERIFY_DENSE` | verify against the dense build | 1 (on) | — | byte-exact (tooling) |
| `C4_EXACT_STEPS` | exact-step verify | unset→off | — | byte-exact (tooling) |
| `C4_VM_CACHE_DIR` | disk cache dir for baked models | `/tmp/c4cache` | — | byte-exact (infra) |
| `C4_MULTIGPU_KSPLIT` | (see Multi-GPU) | | | |
| `C4_SELF_EMU_DEV` | self-emulation device | `cuda:1` | — | byte-exact (harness) |
| `C4_SELF_EMU_JSON` | self-emulation JSON output path | ? | — | byte-exact (harness) |
| `C4_SELF_EMU_REPS` | self-emulation timing reps | 3 | — | byte-exact (harness) |
| `C4_DIM_COALESCE`* | dim-coalesce (perf-fleet build; referenced in scripts) | — | — | doom-build |
| `C4_GENUINE_STRUCTURED_ATTN` | fuller-genuine memory read (model's own softmax1+ALiBi over the reconstructed physical winner K/V) instead of trusting the draft's store_log value atom (`genuine_structured_attn.py`) | unset→off | — | byte-exact `069cc32f` (flag-gated faithfulness lever) |
| `C4_MEM_EFF` | memory recall recency-horizon EFF (ALiBi recency window) override (`blogspec_memory.py`) | `500000.0` | — | byte-exact when horizon covers the program (sizing knob, not semantics) |
| `C4_CODE_CAM_SLOPE` | code-segment CAM ALiBi slope override for the LI/LC address query (`nibble_pure_forward_complete.py`) | unset→built-in | — | byte-exact (tuning knob; unset uses the baked slope) |
| `C4_BUILD_RSS_GUARD` / `C4_BUILD_RSS_CEIL_GB` | build-time RSS guard on/off and ceiling in GB — OOM safety on the streaming build (`compact_alloc.py`) | `1` (on) / (ceiling GB) | — | byte-exact (build memory-safety infra, output-neutral) |

`*` referenced in agent/bench scripts, not a production module read — included for
completeness of the audit cross-reference.

---

## Hardcoded levers that SHOULD be flags (future work — NOT fixed here)

The audit flagged two hardcoded precision constants that have no env override (Q3/Q4c).
Noted here as future "should-be-flag" items; NOT changed (would move the doom build):

- **`baked_attention._DTYPE = torch.float64`** (`baked_attention.py:56`) — the CAM
  division-staircase dtype is hardcoded fp64 (the division needs > fp32 unit step); no
  flag.  A future `C4_PRECISION` preset (audit Q3 design) would surface it.
- **`FP_FRAGILE_BLOCK_NAMES = ("lea-addr-nib",)`** (`pf_kbatch.py:74`) — the fp64-block
  set is env-OVERRIDABLE via `C4_FP64_BLOCK_NAMES`, but the POLICY ("only lea-addr-nib")
  is a module constant.  `analyze_fp64_blocks.py` proved exactly this one block is
  fp-fragile; a `C4_PRECISION` preset would make the policy a flag.

## Golden / safety

- The default golden `7d4afe61` (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON) re-confirmed UNCHANGED
  with `C4_DOOM_FAST` unset AND with all 19 members active (`c4_min._fingerprint_build`);
  the `C4_BP_RESTORE_HIBYTE=0` escape-hatch build is likewise `069cc32f…c0ca` in both
  states.  This registry + `doom_fast.py` add NO build-path / weight changes.
