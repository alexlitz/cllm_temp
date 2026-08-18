# SHALLOW-WIDE DIVMOD — FP32 vs FP64 DEFINITIVE VERDICT (#921/#916 consolidation)

MEASURED, off-build-path.  Golden `174ece66` UNTOUCHED (fingerprint verified before
AND after every measurement; no build-path file changed — only new `_*_921.py`
measurement tools + reused #916/#905 scripts).  Peak RSS 2.3 GB (< 4 GB).

Settles the shallow-wide + width-via-earlier-FFN + step-reorder thread: (1) whether an
FP32 single-pass whole-ISA build reaches ≤24 layers byte-exact, and (2) whether FP64
buys an even smaller host.

## THE ONE-LINE VERDICT

- **MANDATE 1 (FP32 ≤24): NO.** The shallowest whole-ISA FP32 single-pass divmod, with
  the full width-free parallel rewrite + free-reorder overlap + the droppable-carry
  schoolbook collapse, lands at **33 layers** (ASAP critical path).  Not ≤24, not ≤28.
  The blocker is precision, not glue: FP32's 2²⁴ exact-int ceiling forces (a) chunking
  the whole-value scalars and (b) a **refine pass** (FP64 can drop it) because the FP32
  reciprocal floor gives q0 off by up to **±43** on small-b/large-q, beyond ±1.
- **MANDATE 2 (FP64 smaller host): YES — but not smaller than 0.5B in params.** FP64
  needs no chunking and no refine → **24 layers** (knife-edge), so it fits a shallower
  host than the 33-layer FP32 build: smallest real dense-decoder host **Llama-3.2-3B
  (~3.6 B params)** vs FP32's **glm-edge-4b (~5.0 B params)** — FP64 wins by ~1.4 B.
  Neither approaches 0.5 B in PARAMS: the ISA's **width** (hidden ≥ 2624, inter ≥ 4320)
  is the binding constraint, not depth. "0.5B fits" in the thread means "fits a 24-LAYER
  budget" (Qwen2.5-0.5B has 24 layers); a literal 0.5B checkpoint (hidden 896) cannot
  host the ISA at ANY depth.

## MEASURED ASSEMBLED / STRUCTURAL NUMBERS

### Stored block counts of the actual constructions (`_measure_divmod_depths_921`)
| build | whole-ISA stored | divmod stored | base (non-divmod) | residual dim | max FFN units |
|---|---|---|---|---|---|
| radix-16 lean (fp32 DEFAULT) | 107 | 81 (+8 mul) | 18 | 2591 | 7920 |
| log-sink (fp64, `C4_LOGSINK_DIV=1`) | 153 | 127 | 26 | 2591 | 2422 |

The 127 log-sink blocks are **96 decompose-peel glue** (per-nibble MSB extract→snap→
reduce ×4 passes, a SEQUENTIAL running-remainder recurrence) + ~18 schoolbook ×2 + ~11
reciprocal core.  The greedy same-order antichain collapse only reaches **124** (from
127) — the current build's depth is genuine sequential glue, NOT collapsible width.

### The WIDTH-FREE PARALLEL rewrite (the strategy's actual construction, BUILT + measured)
Rewrite decompose to PARALLEL: all 8 nibbles read the CLEAN integer scalar directly
(`nib_c = floor(Q/16^c) − 16·floor(Q/16^(c+1))`), no running remainder.  All 8 extracts
then write DISTINCT places → collapse to ONE layer (verified genuinely independent, not
a tool artifact).  Drop the droppable carry rounds: `q·b` is only needed as a SCALAR, so
`Σ col_c·16^c` of the UNCARRIED partial products already equals `q·b` exactly (verified
0 mismatches / 300k) — schoolbook 8→4 layers.

| form | divmod stored | divmod ASAP | **whole-ISA ASAP** (free reorder, base overlap) | fits ≤24? | fits ≤28? |
|---|---|---|---|---|---|
| log-sink SERIAL (current build) | 127 | — | 108 | no | no |
| **fp64 parallel + direct schoolbook** | 43 | **16** | **24** | **YES (knife-edge)** | yes |
| **fp32 parallel + refine + direct schoolbook** | 59 | 25 | **33** | **NO** | no |

`whole-ISA ASAP` = longest-path over the true hazard DAG (RAW/WAR/WAW on residual
places), free reorder, unlimited width — the ABSOLUTE best-case the accumulation +
step-reorder strategy can achieve.  The FP64 critical path (traced): decode+reciprocal
overlap the base pipeline (fetch/mem/cmp/bitwise/mul all run concurrently L1–15) → qf →
parallel q-decompose (1 layer) → direct schoolbook q·b (4 layers) → parallel result
decompose → finalize → ax-mux = 24.

### Peak residual dim (anti-OOM proof)
Both forms: residual dim **2591** (< the radix-256/4096 GE-CAM 8200-dim OOM trap).  The
wide `q·b` partial-product bank is only **45 dims**, transient (FFN-intermediate), never
parked in the persistent residual — no OOM.  Peak RSS 2.3 GB throughout.

## PRECISION: the exact fp32-vs-fp64 boundary (`nibble_logsink_fp32` + measured here)
- **Parallel decompose of a 2³² scalar**: fp64 byte-exact **0/200000**; fp32 catastrophic
  **196137/200000** (a 2³² value exceeds fp32's 2²⁴ ceiling, so `floor(Q/16^c)` is
  already wrong).  fp32 CHUNKED (2×16-bit limbs, each < 2²⁴): byte-exact **0/300000**.
- **In-model reciprocal after 2 Newton steps** → worst |q0 − true|: fp64 = **1** (a single
  ±1 correction suffices → NO refine pass); fp32 = **43** (fp32 Newton hits its own
  precision floor; doesn't shrink with more steps) → needs a refine pass (2 blocks, NOT a
  43-deep ±1 ripple).
- **Full fp32-chunked parallel algorithm** (fp32 reciprocal + 2 Newton + 1 refine + ±1):
  byte-exact **0/500308** over the full 32-bit battery incl small-prime b / large-q /
  0xDEADBEEF/0x1234 / max/max.  **The FP32 divmod CORE is byte-exact — the ≤24 miss is
  purely the +9 depth the chunking+refine costs, NOT a correctness failure.**

## HONESTY GATE — what is NOT claimed
- The 24/33 are **ASAP critical-path projections over a free-reorder DAG built from the
  ACTUAL block weights**, NOT an assembled ≤24-layer bake.  Realizing 24 requires a
  scheduler that packs the independent blocks (unlimited width) AND the accumulation
  actually overlapping base + reciprocal — feasible per the DAG, but unproven as an
  assembled model.
- The **assembled 107-layer radix-16-lean fp32 model** (`_e2e_floor_r16lean_916`, RE-RUN
  this pass) is 8-bit fully byte-exact (42/42 programs, 31/31 divmod) but full-32-bit only
  **11/19 named, 477/800 random** — a whole-ISA STACK0/dividend-high-nibble truncation
  plumbing bug (#916), ORTHOGONAL to divmod depth (the divmod core is 875/875 standalone).
  So NO assembled whole-ISA model — fp32 OR fp64, at ANY depth — is full-32-bit byte-exact
  end-to-end until that plumbing bug is fixed.  A ≤24-layer parallel build would inherit
  the SAME 32-bit bug.  The ≤24 fp32 fit is therefore DOUBLY refuted: +9 depth AND the
  orthogonal 32-bit plumbing wall.

## THE CONSOLIDATED FIT MATRIX  {precision} × {smallest real dense host}
| precision | whole-ISA depth | width req (hidden / inter) | smallest real dense-decoder host | params |
|---|---|---|---|---|
| **fp64** parallel | **24** | 2624 / 4320 | meta-llama/Llama-3.2-3B (H3072 L28) | **~3.6 B** |
| **fp32** parallel | **33** | 3008 / 7920 | zai-org/glm-edge-4b (H3072 L40) | **~5.0 B** |
| fp32 radix-16 lean (assembled) | 107 | 3008 / 7920 | meta-llama/Llama-3.1-405B (L126) | ~468 B |
| fp64 log-sink (serial, current) | 153 | 2624 / 4320 | none in top-10k | — |

FP64 fits a smaller host than FP32 by ~1.4 B params, driven entirely by the 24-vs-33
layer difference (both share hidden ≈ 3072-class hosts; FP32's wider inter=7920 also
excludes some).  Both dwarf 0.5 B because the ISA WIDTH (hidden ≥ 2624) is the binder.

## Bottom line
The shallow-wide parallel rewrite is REAL and large (log-sink serial 108 → fp64 24 ASAP,
a 4.5× depth collapse, verified byte-exact in the divmod core).  It closes the gap the
menu doc projected — but only in **FP64**, which hits 24 exactly.  **FP32 lands at 33**:
the chunking + refine the 2²⁴ ceiling forces cost +9 layers that no reorder removes.  So
the FP32 0.5B-budget (≤24) fit is a **construction-NO**, and FP64 is what buys the
shallower ≤24 build and the smaller (~3.6 B) host.
