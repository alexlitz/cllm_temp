# Native FP32 (C4_FLOAT_OPS) bit-exact battery — task #800

Golden gate: **069cc32f UNCHANGED** (default build; all FP work is behind
`C4_FLOAT_OPS`, default OFF — verified via `_fingerprint_build`).

## The two F_DIV bugs (FIXED)

1. `3.0/2.0 -> 1.0000004` (want 1.5) — the quotient-significand-`>= 1` case.
   TWO root causes: (a) the scalar two-limb remainder was doubled 26x, amplifying the
   `~value*2^-24` gadget residue (`1/SILU_S = 1/60` not fp32-representable) past 1.0 and
   injecting spurious low quotient bits; (b) the normalize block read its own `hi`
   (`LZ == _DIV_SHIFT`) flag in the SAME block (stale block-input), so the quotient-`>=1`
   mantissa was truncated. Fix: bit-vector remainder + compute `hi` in the prior lead block.
2. `1.0/3.0 -> 0.33333328` (want 0.33333334) — last-2-bits off. Same residue amplification
   corrupted the low quotient bits feeding guard/round/sticky. Fix: same bit-vector
   remainder, with each new remainder bit half-snapped to a clean 0/1 each iteration so
   the residue can NEVER accumulate across the 26 restoring-division steps.

Both are byte-exact now (see `test_fp_div_megablock.py`, 23 cases incl. the two bugs).

## Full bit-exact battery vs the IEEE-754 oracle (SoftFloat refvectors_small.txt, RNE)

676 vectors per op (Berkeley SoftFloat, round-to-nearest-even), cross-checked by
`_fp_softfloat_crosscheck.py`. NaN treated as equivalent (SoftFloat emits sign-1 qNaN
`ffc00000`; x86/oracle canonicalise to `7fc00000`).

| op     | bit-exact | normal×normal | notes |
|--------|-----------|---------------|-------|
| f32add | 620 / 676 | (all normal-result normal×normal exact) | misses = ±0/subnormal specials + subnormal operands |
| f32sub | 625 / 676 | (all normal-result normal×normal exact) | misses = ±0/subnormal specials + subnormal operands |
| f32mul | 528 / 676 | 250 / 256 | 6 normal×normal misses = gradual-underflow (subnormal result); rest = subnormal operands |
| f32div | 548 / 676 | 238 / 256 | 18 normal×normal misses = gradual-underflow (subnormal result); rest = subnormal operands |

**F_DIV normal×normal with a NORMAL result: 238 / 238 = 100% bit-exact.** The 18
normal×normal misses are EXACTLY the 18 vectors whose correctly-rounded result is a
subnormal (gradual underflow) — verified by counting.

## HONEST gaps (all four ops, shared, pre-existing)

- **Subnormal OPERANDS**: the decode assumes a normalized significand (implicit leading 1
  at bit 23); a subnormal's significand has its leading 1 lower and biased exponent 0, so
  the field decode + exponent math are wrong. Shared by add/sub/mul/div.
- **Gradual underflow (subnormal RESULT)**: the encode FLUSHES to signed zero when the
  biased result exponent `<= 0` instead of denormalising the significand. Shared by all
  four (F_MUL/F_ADD documented deferral; F_DIV now matches that same behavior).
- Inf/NaN/±0 specials: bit-exact (the special-override blocks handle them).

These are the documented shared "subnormal is a GAP" limitation, NOT a division bug.

## Step savings vs soft-float

Per-op transformer block counts (one op = one megablock forward, dim=1008):

| op     | blocks | FFN units |
|--------|--------|-----------|
| f32add |   37   |  6,677    |
| f32sub |   38   |  6,678    |
| f32mul |   28   |  5,823    |
| f32div |  369   | 79,556    |

vs a Berkeley SoftFloat `f32_div` executed on the c4 VM: `softfloat_div` compiles to
~hundreds of C statements (leading-zero count loop, 30+-iteration long-division inner
loop with a 64-bit estimate/refine, rounding), each lowering to MANY VM *instructions*
each of which is itself a full transformer step. The native megablock replaces that
entire per-value software routine with ONE opcode dispatch whose fixed 369-block forward
computes the correctly-rounded quotient directly — i.e. the step count drops from
O(hundreds of VM instructions × steps-per-instruction) to a single F_DIV dispatch.
f32add/sub/mul are an even larger win (28-38 blocks vs the full soft-float add/mul routine).

## f64 (double)

Not implemented. Only f32 (single) F_ADD/F_SUB/F_MUL/F_DIV exist in the C4_FLOAT_OPS ISA
extension. The softfloat file has f64* vectors but there is no f64 opcode to test.
