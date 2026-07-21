# Signed comparison on the production pure-forward path (#673 follow-up)

Companion to `WIDTH32_FAMILY2.md`. That fix (#667/#671, `C4_VM_WIDTH32`) made
compare/branch/loop **unsigned** 32-bit-exact and explicitly left signedness open
("Unsigned 32-bit ordering is exact", §Signedness). This fix makes the C4 ordering
comparisons (`LT/GT/LE/GE`) **SIGNED** two's-complement, matching classic C4 / gcc.

## Model-path reconciliation (de-confusing #671 vs #673)

There are three files, and only ONE is the production compute path:

- **`nibble_pure_forward_complete.py`** — the **PRODUCTION** model. `build_pure_forward_complete_model`
  / `run_pure_forward_complete` are what the 1096 corpus (`run_1096_pure_forward.py`),
  the demos (`demo_model_runs_c`, `chat_eliza`, `bundle_small`, `libprog_corpus`, …)
  and `compact_alloc.build_compact_sparse_streaming` (the memory-lean streaming build)
  all run. Its comparison is `compile_cmp_compute` + (new) `compile_cmp_signed_finalize`,
  **imported from `nibble_pure_forward`** — it does NOT use `nibble_unified`'s
  `compile_cmp_expert`.
- **`nibble_pure_forward.py`** — the pure-forward VM library the production build is
  layered on. Defines `compile_cmp_compute` (the shared CMP gadget) and the smaller
  `build_pure_forward_model` (cmp-only / lite builds used by unit tests). Same CMP
  weights as production.
- **`nibble_vm.py` / `nibble_unified.py`** — the **legacy** single-block step model
  (`build_step_model` / `build_unified_model`) and its `compile_cmp_expert`. This is
  the model **#671** widened + tested (`test_width32_family2.py`). The production path
  does NOT import `compile_cmp_expert`; #671's compare fix therefore never touched
  the production compare. (`nibble_vm`'s `S/RELU_S/_fold_modulus/_snap_lane`
  helpers ARE shared, but its cmp expert is not on the production path.)

So #671 fixed an **unsigned width** issue on the legacy `nibble_vm`/`nibble_unified`
path; #673 correctly observed the production path already carries wide unsigned
values (via `C4_VM_WIDTH32`) but its **signed** ordering was still wrong — at the
reference oracle too (`ref_interpret` / `isa.interpret` compared unsigned). Both are
addressed here.

## The bug

The register value lanes (`AX_VAL`, `STK_VAL`) recompose as `Σ 16^j · nibble_j ≥ 0`
— the **unsigned magnitude** of the two's-complement word. `compile_cmp_compute`
computed `d = STK_VAL - AX_VAL` and took `step(d ≥ 1)` / `step(-d ≥ 1)` as GT / LT.
That is UNSIGNED ordering: `-10` recomposes to `4294967286`, so `(-10) < 5` (LT) was
`false`, `(-10 < 0)` was `false`, `while (i > -3)` never ran. The reference
`ref_interpret` / `isa.interpret` did the same unsigned compare, so #673's oracle
reproduced the bug (no correct golden).

## The fix

`LT/GT/LE/GE` reduce to `sign(a-b)` combined with EQ. We correct the unsigned
magnitude order by the two operands' 32-bit sign bit (bit 31):

```
signed_GT = clamp01(MAG_GT) + (SGN_AX  - SGN_STK)
signed_LT = clamp01(MAG_LT) + (SGN_STK - SGN_AX)
```

Derivation: the naive `same_sign·mag ± cross_sign` products cancel to this linear
correction — it flips exactly the two cross-sign cases (positive-STK vs negative-AX
becomes GT; negative-STK vs positive-AX becomes LT) and leaves same-sign pairs
unchanged; the sum is always exactly 0 or 1. `EQ` (and `NE = 1-EQ`) are sign-agnostic
and unchanged; `LE = 1-GT`, `GE = 1-LT` still hold for the signed verdict.

Two SwiGLU blocks (both vanilla — softmax1/SwiGLU step gadgets, no masking):

1. **`compile_cmp_compute`** — the ungated primitives: `CMP_EQ` (final), the raw
   unsigned ramps `MAG_GT`/`MAG_LT`, and the sign bits `SGN_STK`/`SGN_AX`.
2. **`compile_cmp_signed_finalize`** — `CMP_GT = clamp01(MAG_GT) + SGN_AX - SGN_STK`,
   `CMP_LT = clamp01(MAG_LT) + SGN_STK - SGN_AX`.

### Two numerical subtleties (both handled)

- **`sign(v)` is read from the value's TOP NIBBLE** (`step(nib7 ≥ 8)`, threshold
  `7.5`), NOT the recomposed scalar. Reading the sign from the ~2^31 scalar needs a
  `-RELU_S·(2^31-0.5)` bias that is not fp32-representable (specs are built fp32 then
  cast fp64), which corrupts the threshold. The nibble is a small operand (0..15) so
  the step is fp-exact. Each sign lands in its OWN `SGN_*` dim.
- **`MAG_GT`/`MAG_LT` are `clamp01`'d in the finalize block.** For a cross-sign pair
  `|d| ≈ 2^32`, the raw ramp `relu(d) - relu(d-1)` cancels two ~2^32-scale relus
  imperfectly (the recompose's ~4-unit fp error) and can read e.g. `95` instead of
  `1`; `clamp01(m) = relu(m) - relu(m-1)` collapses it to the correct unsigned `1`
  before the sign correction. Same-sign differences are computed exactly in the
  linear layer (both operands < 2^53) so those stay clean.

### Oracles made signed (so there is a correct golden)

- `nibble_pure_forward_complete.ref_interpret` — `LT/GT/LE/GE` now compare the
  two's-complement SIGNED 32-bit value (sign bit 31), at the value width `mask`.
- `isa.interpret` — the 8-bit slice oracle stays a plain unsigned byte compare (every
  value is masked `<256 < 2^31`, so the signed order coincides with the unsigned byte
  order there — it matches the model at the 8-bit fold). 32-bit signed is exercised
  via `ref_interpret` / `C4_VM_WIDTH32`.

## No-regression by construction (8-bit fold)

Under the DEFAULT 8-bit fold (`C4_VM_WIDTH32` off, what the 1096 corpus/demos build)
the recompose/ingest carry only nibbles 0..4, so nibble 7 is always 0, `SGN_* = 0`,
`MAG_*` never leave the 8-bit range (no cross-sign noise), and
`clamp01(MAG) + 0` is the pre-fix unsigned `CMP_GT`/`CMP_LT` — the comparison VERDICT
is byte-identical to before. (The model gains one block + four scratch lanes, so the
raw weight structure differs, but there is no fixed c4_min golden-hash gate and the
behavioural corpus/demo verdicts are unchanged.)

## Verified

- **Gadget (deterministic, fp64):** `test_pure_forward.test_cmp_signed_gadget_two_complement`
  — 16 pairs incl. cross-sign, both-negative, `INT_MIN`, and the recompose-noisy
  ~2^32 scalar — `LT/GT/EQ` all match Python's signed order.
- **Production model (`build_compact_sparse_streaming`, `C4_VM_WIDTH32=1`):** the
  three #673 f3 programs via `model.forward` (`_verify_signed_cmp_production.py`):
  - `f3_signed_cmp` `(-10) < 5 == 1` — **PASS**
  - `f3_if_neg` `(-3) < 0 == 1` — **PASS**
  - `f3_count_down_neg` `2 > (-3) == 1` — **FAIL** (see limitation below)
- **No regression:** `test_pure_forward.py` 12/12 + `test_nibble_vm.py` 9/9 at the
  8-bit fold; the production `if_gt`/`if_lt`/`if_eq` corpus programs match gcc; #671's
  `test_width32_family2.py` 18/18 (unsigned wide path untouched).

## Known limitation — negative-in-AX at a GT (`f3_count_down_neg`)

`f3_signed_cmp` / `f3_if_neg` put the negative operand on the **stack** (its nibbles
are freshly written by the stack-pop KV head → `SGN_STK` fires), and both PASS.
`f3_count_down_neg` puts the negative in **AX** (computed by a prior `SUB`, then
compared with a positive that is popped from the stack) and FAILs.

The negative itself is delivered correctly: the `buried_SUB_neg` / `buried_then_ADD`
diagnostics (same buried-stack `SUB`) return `AX = -3` / `-1` byte-exact, so the ALU
consumes the computed negative. The gadget is byte-correct for this operand pair in
isolation (`2 > -3 → GT` in `test_cmp_signed_gadget_two_complement`). So the divergence
is in how the just-computed negative AX's **sign** reaches the compare block on this
specific program shape (a negative in AX + a value popped from a buried stack slot),
not in the signed-compare algebra. It is a narrow AX-ingest / sign-delivery follow-up;
the comparison-cluster corpus and the two stack-operand f3 programs are unaffected.

## Recommendation on `C4_VM_WIDTH32`

Signed comparison is only meaningful when values carry their full 32-bit two's
complement, i.e. under `C4_VM_WIDTH32=1`. At the 8-bit fold a "negative" is
indistinguishable from a positive byte, so the fix is (correctly) inert there. The
1096 runner already masks to 32-bit (`mask=0xFFFFFFFF`) but does NOT set
`C4_VM_WIDTH32`, so the production corpus/demo model runs 8-bit-folded. To get
signed (and wide) arithmetic end-to-end on the production path, `C4_VM_WIDTH32=1`
belongs on it — the same recommendation #671 implied for wide unsigned values. It is
default-OFF and byte-identical when off, so flipping it on is a safe, opt-in step
(guarded by fp64 exec cost); this fix is the signed half of that story.
