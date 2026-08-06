# DOOM vs GENERAL boundary (`c4_min/` + `neural_vm/`)

Where does the DOOM-specific work end and the GENERAL c4-VM core (the one that runs ANY
program — Mandelbrot, the 1096 corpus, a quine, self-emulation) begin? This maps the
boundary at the FLAG level and the MODULE level and gives the honest verdict.

Generated 2026-08-06. Default golden `7d4afe61` (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON — the
general-correctness fix; `069cc32f` is the `C4_BP_RESTORE_HIBYTE=0` rollback), re-confirmed
intact (`python -m c4_min._fingerprint_build`). **Docs-only.**

Companion: [`DOOM_FLAG_REGISTRY.md`](DOOM_FLAG_REGISTRY.md) (per-flag table),
[`FLAG_REGISTRY.md`](FLAG_REGISTRY.md) (`neural_vm/` + `tools/` flags),
[`HF_MODEL_FIT.md`](HF_MODEL_FIT.md) (the general-core parameterization).

---

## TL;DR verdict

**Separation is BOTH by-module AND by-flag, and it is clean.**

1. **By-flag:** every doom / perf-stack lever is a **DEFAULT-OFF** `C4_*` env flag. With
   all doom flags OFF (the default), the build IS the general VM and the golden
   fingerprint is `7d4afe61` (verified) — this is the default build with the
   general-correctness fix `C4_BP_RESTORE_HIBYTE` DEFAULT-ON (`=0` rolls back to the
   pre-fix `069cc32f`). The two golden-MOVING doom flags (`C4_DOOM_DRAWSPAN`,
   `C4_INGEST_WIDE`) are also DEFAULT-OFF, so `7d4afe61` is the flag-OFF general-core golden.
2. **By-module:** the doom render macros live in **dedicated modules** (`doom_*.py`) and
   the general core (`isa.py`, `qwen_full_vm.py`, `qwen_vanilla_vm.py`,
   `nibble_pure_forward_complete.py`, the compiler/transpiler) **does not import a single
   doom module.** The dependency arrow points ONE way: `doom_* → core`
   (`doom_drawcol.py: from . import isa`), never `core → doom_*`.
3. **The doom opcodes sit ABOVE the golden opcode band.** `NUM_OPS=40` is the golden
   one-hot width; DRAWSPAN=47 / DRAWCOL=48 / DRAWSPANF=49 (and the float ops 40-43) all
   live above 40 and only widen the band when their gate is on. So the baseline OP_IS
   layout is byte-identical with doom off — the doom ops cannot even perturb the general
   ISA's dim layout.
4. **`neural_vm/` has ZERO doom content.** Every `doom*` filename is in `c4_min/`;
   `neural_vm/` is the pure declarative-compiler substrate.

The one honest nuance (not an entanglement): a handful of perf-stack levers are
DEFAULT-**ON** (`C4_SCHED_FAST_BUILD`, `C4_DIRECT_CAM_LIVE_LOCAL`,
`C4_BLOCK0_DROP_DEAD_KV`, `C4_MATERIALIZE_IDEMPOTENT`, `C4_OVERLAY_PRECOMPUTE`,
`C4_KV_STACK`, `C4_LEA_Q_SNAP`, `C4_LIVE_HEADS_MEMO`) — but they are all
registry-classified `byte-exact` execution/runtime knobs (they change HOW the forward is
scheduled, not WHAT weights exist), and the default golden `7d4afe61` is computed WITH them
at their defaults. They are general-VM perf infrastructure, not doom-specific.

---

## Layer 1 — DOOM-SPECIFIC (only meaningful for the doom program)

### Flags (all DEFAULT-OFF unless noted)

| flag | what it is | golden |
|---|---|---|
| `C4_DOOM_FAST` | ★ one-toggle composite for the whole byte-exact fast-doom stack (19 members) | byte-exact |
| `C4_DOOM_DRAWCOL` | native DRAWCOL / DRAWSPANF render-macro opcodes (48/49) — textured wall column / floor span | byte-exact (band widens only on) |
| `C4_DOOM_DRAWSPAN` | native DRAWSPAN render-macro opcode (47) — V_DrawPatch column copy | **golden-MOVING on** (ON `2f69350f…` measured on the old `069cc32f` base; shifts under the new default); OFF = the default `7d4afe61` |
| `C4_DOOM_BLIT` | doom framebuffer blit macro | byte-exact |
| `C4_DOOM_NAMEEQ` | fused 8-char WAD lump-name compare (#828) | byte-exact |
| `C4_DOOM_FIXEDPOINT` | native fixed-point MAC for the doom renderer | byte-exact |
| `C4_MEGABLOCK_DOOM_LEAN` | doom-lean variant of the megablock region | byte-exact |

### Modules (`c4_min/`, doom-only)

| module | role |
|---|---|
| `doom_fast.py` | the `C4_DOOM_FAST` composite expander (`expand_doom_fast()`) |
| `doom_drawcol.py` | DRAWCOL / DRAWSPANF opcode semantics (`R_DrawColumn` / `R_DrawSpan`) |
| `doom_drawspan.py` | DRAWSPAN opcode semantics (`V_DrawPatch`) |
| `doom_blit.py` | framebuffer blit macro |
| `doom_nameeq.py` | WAD lump-name compare intrinsic |
| `doom_fixedpoint.py` | doom fixed-point MAC |
| `verify_drawcol_onvm.py`, `verify_drawspan_onvm.py`, `measure_doom_fixedpoint.py` | doom-op verifiers / measurement |

WAD-hash / doom-tuned direct-CAM: the WAD name-hash cache lives in the separate
`c4_doom` repo (`wad-name-hash-cache` branch); the c4_min side is the `doom_nameeq.py`
intrinsic. The direct-CAM gathers themselves (below) are GENERAL mechanisms tuned by the
doom working-set, not doom-only code.

---

## Layer 2 — PERF STACK (general mechanism, doom-TUNED defaults)

These are byte-exact compute-fusion / schedule / attention / eviction levers. They are
**general** (any program with a big persistent KV benefits) but their winning parameter
choices (block_k=512, the megablock member set) were tuned on the doom frame. They are
NOT doom-only: the mechanism works for Mandelbrot, self-emulation, the 1096 corpus.
The `C4_DOOM_FAST` composite is just a convenience bundle of the subset that wins on
doom. Full per-flag table in [`DOOM_FLAG_REGISTRY.md`](DOOM_FLAG_REGISTRY.md).

Representative modules (`c4_min/`): `fused_megablock.py`, `fused_ffn_megakernel.py`,
`block0_fused_ffn.py`, `block0_graph.py`, `whole_step_graph.py`, `megastep_graph.py`,
`graphed_fused_forward.py`, `direct_cam_batched.py`, `direct_cam_read.py`,
`direct_local_cam.py`, `selfemu_direct_cam.py`, `flash_softmax1.py`,
`banded_local_attn.py`, `tight_attn_compose.py`, `precomputed_schedule.py`,
`pf_kbatch.py`, `pf_speculative.py`, `batched_speculative.py`, `nibble_speculative.py`,
`pos_sparse_fused.py`, `nibble_evict_schedule.py`, `qwen_lean_evict*.py`,
`multigpu_ksplit.py`.

**DEFAULT-ON members of this layer (byte-exact runtime knobs, part of golden):**
`C4_SCHED_FAST_BUILD`, `C4_DIRECT_CAM_LIVE_LOCAL`, `C4_BLOCK0_DROP_DEAD_KV`,
`C4_MATERIALIZE_IDEMPOTENT`, `C4_OVERLAY_PRECOMPUTE`, `C4_KV_STACK`, `C4_LEA_Q_SNAP`,
`C4_LIVE_HEADS_MEMO`. These are the "always-safe" fast paths; the golden fingerprint is
computed with them on, so they are baseline general-VM infra, not entanglement.

---

## Layer 3 — GENERAL CORE (runs ANY program — the default build)

### The ISA + the model build + the compiler/transpiler

| module (`c4_min/`) | role |
|---|---|
| `isa.py` | the C4 instruction set. `NUM_OPS=40` golden band; doom/float opcodes sit ABOVE it, gated |
| `qwen_full_vm.py` | **`build(code_size, subset, arch, efficient_alu, recurrent_divmod, pad_to_stock, …)`** — the genuine `Qwen2Model` whose layers ARE the fused VM step. The parameterized general build |
| `qwen_vanilla_vm.py` | **`build(subset, …)`** — stock `Qwen2ForCausalLM` with real embed/lm_head; runs the VM through the STANDARD autoregressive loop (discrete-token registers) |
| `nibble_pure_forward_complete.py` | the complete nibble pure-forward VM (recurrent core, `C4_LEA_Q_SNAP` etc.) |
| `nibble_pure_forward.py` / `_cached.py` / `_gpu.py` | pure-forward variants |
| `nibble_compiler.py`, `nibble_bake.py` | bytecode → weights nibble compiler/baker |
| `compile_ffn.py`, `compile_attn.py`, `loop_compiler.py` | FFN / attention / loop lowering |
| `blogspec_*.py` | the BLOG_SPEC-faithful compiler / layout / memory / model / vocab |
| `compact_alloc.py` | the LEAN streaming compact allocator (`build_compact_sparse_streaming` — the golden-fingerprint build) |
| `native_fp32_vm.py`, `native_fp32_baked.py` | fp32 baked VM |
| `export_onnx_compact.py` | ONNX export of the compact build |
| `qwen_fit_solver.py` | the HF-fit configurator (see `HF_MODEL_FIT.md`) |

### `neural_vm/` — the declarative-compiler substrate (ZERO doom)

`neural_vm/unified_compiler/` is the sole live weight-authoring path
(`compile_full_vm_dynamic`), the per-layer `ops/lN_ops.py`, the allocators, the
residual-band registry, the KV-eviction machinery. **No `doom*` file exists in
`neural_vm/`; no `neural_vm/` module reads a `C4_DOOM_*` flag.** It is the pure general
substrate the 1096 corpus and the `neural_vm/`-era campaign run on.

### Proof the general core is clean with doom OFF

- **Import direction (one-way):** `grep` for doom imports in `isa.py`,
  `qwen_full_vm.py`, `qwen_vanilla_vm.py`, `nibble_pure_forward_complete.py`,
  `nibble_compiler.py`, `nibble_bake.py`, `compact_alloc.py` → **no doom import** (the
  only doom strings in `isa.py` are COMMENTS + the gate-mirror helper that points AT
  `doom_drawspan`/`doom_drawcol`, importing nothing). The reverse
  (`doom_drawcol.py: from . import isa`) confirms doom builds ON the core.
- **Opcode band:** `isa.py` — `NUM_OPS = 40`; `DRAWSPAN = 47`, `DRAWCOL = 48`,
  `DRAWSPANF = 49` and the float ops (40-43) all sit above the golden 40-band and only
  widen it (`NUM_OPS_DRAWSPAN=48`, `NUM_OPS_DRAWCOL=50`, `NUM_OPS_FLOAT=44`) when their
  DEFAULT-OFF gate fires. So with doom off the ISA one-hot layout is byte-identical.
- **Golden gate:** `c4_min._fingerprint_build` = `7d4afe61` with everything at
  defaults (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON) — this IS the general VM, and it is what
  Mandelbrot runs on byte-exact (`=0` rolls back to the pre-fix `069cc32f`).
- **Mandelbrot uses the general build:** `mandelbrot_native.py` / `lean_mandelbrot.py` /
  `_mandel_run.py` bake through `qwen_full_vm.build` / the compact streaming build with
  NO doom flag — the general-program byte-exactness proof (per the consolidate commit
  log, Mandelbrot is fully byte-exact interior/escape/boundary against the oracle on the
  same hardened build).

---

## Is the doom stuff cleanly TOGGLEABLE / excludable?

**Yes — on both axes.**

- **Excludable by flag:** unset every `C4_DOOM_*` (the default) → the general VM, default
  golden `7d4afe61` (`=0` on `C4_BP_RESTORE_HIBYTE` rolls back to `069cc32f`).
  `C4_DOOM_FAST=1 C4_BLOCK0_DK=0` shows even the composite honors per-member
  opt-out (`setdefault` semantics).
- **Excludable by module:** you could delete the `doom_*.py` files and the general core
  still imports and builds (nothing imports them). The only residue would be the two
  gate-mirror helper functions + comments in `isa.py` that reference the doom module
  paths — cosmetic, not a hard dependency.

### The one genuine footgun (documented, not an entanglement)

`C4_WHOLE_STEP_GRAPH` vs `C4_WHOLESTEP_GRAPH` (differ by one underscore) are TWO
different perf flags in two different modules (`whole_step_graph.py` vs `pf_kbatch.py`)
meaning different things, with no alias. Both DEFAULT-OFF byte-exact. This is a naming
hazard flagged in `DOOM_FLAG_REGISTRY.md`, not a doom/general entanglement.

---

## Golden safety

Default golden `7d4afe61` (`C4_BP_RESTORE_HIBYTE` DEFAULT-ON) re-confirmed; the
`C4_BP_RESTORE_HIBYTE=0` rollback build is
`069cc32fa7cecfbceae448a7dbf6e2140b3db6cf6857c8accec5639b9c55c0ca`.
This doc adds no build/weight change.
