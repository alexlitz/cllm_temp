# Compacted / Qwen / lean bake: signed-ordering compare + countdown fixes (#691)

Two pre-existing model-correctness bugs in the **compacted** fused VM bake
(`qwen_full_vm` + the lean forwards `qwen_lean_forward` / `qwen_lean_cuda_graph` /
`qwen_lean_evict`), both precisely characterized by #691. Companion to
`SIGNED_COMPARE_2026_07_21.md` (#673, the DEEP `nibble_pure_forward_complete` path)
and `WIDTH32_FAMILY2.md` (#667/#671).

The compacted bake uses a **structurally different** compare than the deep 20-head
pure-forward path: the RoPE §Memory register CAM + a compacted single-frame register
layout (5 register CAM heads + 1 memory head; each register rides ONE frame token,
gathered by a content-match + recency). But its *compute* blocks are the SAME
`nibble_pure_forward` SwiGLU FFN gadgets ported onto Qwen's MLP — so the compare
weights are shared with the deep path; only the block LIST that `qwen_full_vm._block_specs`
assembles differs.

## BUG 1 — signed / ordering compare degenerated to a constant

### Symptom (measured, matches #691)
On the compacted bake `LT` and `GT` always returned **0**, `LE` and `GE` always **1**,
REGARDLESS of operands (passing only when that constant happened to be the correct
answer — 4/49 corpus failures). `EQ` and `NE` were correct.

### Root cause — the `cmp-finalize` block was missing from the compacted bake
`cmp_dispatch_rules` writes AX from the scratch lanes:
`EQ=CMP_EQ  NE=1-CMP_EQ  LT=CMP_LT  GT=CMP_GT  LE=1-CMP_GT  GE=1-CMP_LT`.

Those lanes are produced by TWO FFN blocks in the deep path (#673):

1. `compile_cmp_compute` — the UNGATED primitives: `CMP_EQ` (final), the raw unsigned
   ramps `MAG_GT`/`MAG_LT`, and the sign bits `SGN_STK`/`SGN_AX`.
2. `compile_cmp_signed_finalize` — combines them into the SIGNED verdict:
   `CMP_GT = clamp01(MAG_GT) + (SGN_AX - SGN_STK)`,
   `CMP_LT = clamp01(MAG_LT) + (SGN_STK - SGN_AX)`.

`build_pure_forward_model` and `build_pure_forward_complete_model` BOTH append
`cmp-finalize` right after `cmp-compute`. **`qwen_full_vm._block_specs` appended only
`cmp-compute`** (one line: `specs += [("cmp-compute", ...)]`). So `CMP_GT` / `CMP_LT`
were never written and stayed 0 — hence `LT`/`GT` ≡ 0 and `LE`/`GE` ≡ 1 for every
operand. `EQ`/`NE` read `CMP_EQ`, which `cmp-compute` DOES write, so they were fine —
exactly the #691 signature.

### Fix (block/rule/dim)
`c4_min/qwen_full_vm.py`, `_block_specs`, the `if subset.cmp:` branch:
```python
specs += [("cmp-compute", PF.compile_cmp_compute(L, dim)),
          ("cmp-finalize", PF.compile_cmp_signed_finalize(L, dim))]   # was: cmp-compute only
```
The lean forwards (`qwen_lean_forward` etc.) COPY the built `Qwen2Model` weights, so
the fix flows through them automatically — the finalize block becomes a new Qwen layer
(the compacted cmp model goes 10 → 11 layers).

The compare weights are the SAME as the deep path; their signed correctness is proven
by `test_pure_forward.test_cmp_signed_gadget_two_complement` (16 cross-sign pairs incl.
INT_MIN, both-negative, recompose-noisy ~2^32), which passes.

### Signedness scope on the compacted path
At the DEFAULT 8-bit fold (`C4_VM_WIDTH32` off — what the corpus/demos build) every
value is `< 256 < 2^31`, so nibble 7 = 0, `SGN_* = 0`, and `clamp01(MAG)+0` is the
plain-unsigned verdict — which is what `isa.interpret` also compares (unsigned byte).
So the compacted bake is now byte-exact vs `isa.interpret` for the full 0..255 byte
range (incl. the 128–255 "negative bytes"). This is the fix for the #691 bug.

TRUE 32-bit-signed operands (genuine two's-complement negatives, `C4_VM_WIDTH32=1`)
are correct in the compare GADGET (gadget test + the negative-on-STACK cases
`LT(-10,5)`, `LT(-100,50)` pass end-to-end HF and lean, byte-identical). The residual
`LT(-3,0)` / `GT(2,-3)` class is a PRE-EXISTING, separate limitation of the compacted
`run_program` DRIVER, not the compare: its `_snap` requant is a single flat argmax over
`VALVOCAB ≈ 0x10100`, which cannot represent a 32-bit negative — a computed negative
collapses to 0 at the AX fold BEFORE it reaches the compare. Full 32-bit-signed
end-to-end on the compacted path needs the deep path's per-byte requant
(`_snap_lane_bytes`) ported into the driver — the same "narrow AX sign-delivery
follow-up" #673 documents, orthogonal to this compare fix. A secondary correctness
tidy landed alongside: `_block_specs`'s `fold` block was hardcoded `modulus=256`; it is
now width-aware (`compile_fold(L.AX_VAL, L.ONE, dim)` → 256 at the 8-bit fold, a no-op
2^32 ramp under WIDTH32), BYTE-IDENTICAL when WIDTH32 is off, so it no longer
mod-clamps a wide/negative SUB result.

## BUG 2 — countdown ≥ 100 (really ≥ 64) diverged

### Symptom
A loop counting down from a large `n` diverged from `isa.interpret` (not hit by shallow
samples). #691 saw it "≥ 100"; the true boundary is `n ≥ 64`.

### Root cause — the REFERENCE oracle's 256-step cap, not the model
`isa.interpret(code, max_steps=256)` caps at 256 VM steps. A countdown from `n` runs
`4n+2` steps (each loop body is `PSH/IMM/SUB/BNZ` = 4 steps), so for `n ≥ 64` (258
steps) the reference is TRUNCATED to 256 emitted values — while the model driver
(`run_program`) runs the loop to completion at its own `max_steps` (callers pass
`n*4+20`). The two traces then have different LENGTHS and mismatch. The MODEL is
correct: it counts down byte-exact all the way to 0, across the nibble-carry
boundaries (100 = 0x64, 200 = 0xC8), for every n up to 255 — verified `model == the
uncapped isa.interpret` for n = 64/99/100/200/255. It is a pure test-harness artifact
(a truncated golden), NOT an 8-bit-fold / nibble-carry overflow.

### Fix
Run the reference oracle to the SAME step budget as the driver, in every driver:
```python
ref_trace = isa.interpret(code, max_steps=max_steps)   # was: isa.interpret(code)
```
Applied in `qwen_full_vm.run_program`, `qwen_lean_forward.run_program_lean` +
`draft_program_lean` (the speculative draft, `max_steps=4096`),
`qwen_lean_cuda_graph.run_program_lean_graphed`, and
`qwen_lean_evict.run_program_lean_evict`.

## Verification (GPU cuda:0)

- **CMP battery** (6 ops × 22–24 operand pairs — equal / a<b / a>b / 0-255 boundary /
  128-255 negative-bytes): **144/144 exact** on the genuine `transformers.Qwen2Model.forward`
  AND **144/144** on the lean forward (`qwen_lean_forward`), matching `isa.interpret`.
- **Countdown battery** (n = 2,3,50,63,64,99,100,101,150,200,255): **11/11 exact** on
  BOTH HF Qwen2Model and lean; also PASS through the speculative lean driver (n=100/200/255,
  402/802/1022 steps, ~58–64× speculation speedup). Full trace length `= 4n+2` (not the
  256 cap); final AX = 0.
- **Signed gadget** `test_pure_forward.test_cmp_signed_gadget_two_complement` — PASS
  (the compacted bake's compare weights, 16 signed pairs incl. INT_MIN).
- **Negative-on-STACK signed cases** (`C4_VM_WIDTH32=1`): HF == lean byte-identical for
  all 8; `LT(-10,5)`/`LT(-100,50)`/`GT(-5,10)`/`LE(-10,5)`/`GE(-5,10)` match the signed
  golden end-to-end (the `LT(-3,0)`/`GT(2,-3)` class is the driver value-width limitation
  noted above).
- **No regression:** `test_qwen_full_vm.py` + `test_qwen_lean_forward.py` — 171 passed,
  4 skipped, 1 PRE-EXISTING failure (`test_base_subset_fits_stock_0_5b`: the base subset
  hidden_size is 960 > 896, unrelated — driven by the #667/#673 layout-dim growth on the
  base commit, fails identically on clean HEAD). `test_muldiv_through_qwen` (MUL/DIV/MOD)
  and `test_corpus_memcmp_families_100pct` PASS — the compare fix is orthogonal to muldiv
  (#699). `test_pure_forward.py` cmp/bitwise PASS (deep-path cmp weights untouched).

## Regression tests added
- `test_qwen_full_vm.py::test_cmp_ordering_matrix_through_qwen` — BOTH orderings of every
  op across the byte range (the #691 BUG 1 tripwire; would catch a missing finalize).
- `test_qwen_full_vm.py::test_countdown_over_256_steps` — countdown n=50..255, asserts
  full trace length `4n+2` and final AX 0.
- `test_qwen_lean_forward.py::test_cmp_lean_eq_hf` (strengthened to assert `exact`, full
  ordering matrix) + `::test_countdown_over_256_steps_lean` — same on the lean forward.
