# F_DIV bug root-cause (task #800)

## Symptom
- `3.0/2.0` -> 1.0000004 (want 1.5); `1.0/3.0` -> 0.33333328 (want 0.33333334)
- Full battery: F_DIV 4467/9036 (~49%), normal×normal 3149/7239 (~43%)

## Root cause (CONFIRMED)
The restoring-division core keeps the remainder `DR` as a **big scalar** split into
two limbs (`_DIV_LIMB=13`, each up to 2^13=8192) and DOUBLES it every iteration via
`_ident(scale=2.0)`.

- The neural gadgets (`_ident`, `_step_ge`) route values through `hidden * (scale/SILU_S)`
  where `1/SILU_S = 1/60` is NOT exactly representable in fp32. For a NON-power-of-2
  value this introduces a residue ~`value * 2^-24`.
- Powers of 2 are exact (`2*512=1024` exact) but arbitrary limb values are not.
- The `2*DR` doubling AMPLIFIES this residue: at init `DR1 = 1536.00012` (residue 2^-13);
  after ~12 doublings the residue crosses 1.0 and starts flipping the compare, producing
  SPURIOUS quotient bits below the true quotient. e.g. for 3.0/2.0 QINT gains bits at
  positions 1 and 4 that should be 0 -> wrong low mantissa -> 1.0000004.

## Why the snap-to-int patch is hard
`_step_ge` is only a CLEAN 0/1 when the form's residue keeps it OUT of the ramp
transition band `[k-0.75, k-0.5]`. A residue >= ~0.25 lands inside a ramp and leaks a
FRACTION. A `sum_{k=1..N}[src>=k]` round-staircase also accumulates ~0.02 of matmul
rounding across its N terms. So a scalar snap is finicky and not bit-exact.

## Correct fix (LANDED)
Align DIV with the ADD/MUL idiom: keep the remainder as an EXACT BIT-VECTOR (each bit
0/1) and do shift (bit re-index, exact) / compare / subtract in the bit/nibble domain
with carry rounds. No big-scalar doubling => no residue amplification.

Per iteration (bit-vector remainder DRB, divisor bits BBIT):
  A shift : S = (k==0)?R:2R      -- bit re-index into DRB2 (exact)
  B.0 sn  : columns = S + (~B) + 1 over 8 nibbles (S-B+2^32)
  B carry : 8 nibble carry-settle rounds (ping-pong DCOL/DCOL2)
  B.1a dd : explode settled nibbles -> DDIFF bits ; bit31 = [S < B]
  B.1b q  : q = 1 - DDIFF[31] = [S>=B] ; record QBIT[_DIV_SHIFT-k] (both read DDIFF[31])
  D  sel  : R[i] = [ (q?DDIFF[i]:S[i]) >= 0.5 ]  -- HALF-SNAP to clean 0/1 (kills residue)

THREE bugs found & fixed:
  1. scalar-remainder residue amplification -> bit-vector + per-iter half-snap.
  2. normalize read its own `hi` flag same-block (stale) -> compute `hi` in the lead block.
  3. QBIT recorded from a same-block-written `q` (stale) -> record QBIT from DDIFF[31].
Also a latent aliasing: old DR/BMV were _scalar()s but code used DR+1/BMV+1 which overlapped
AMV/QBIT[0]; the new DRB/DRB2/BBIT/DSN/DCOL/DCOL2/DDIFF/DCMP bands are properly sized.

RESULT: F_DIV normal×normal-with-normal-result = 100% bit-exact; overall 548/676 vs
SoftFloat (misses = subnormal operands + gradual-underflow, the documented shared gap).
Golden 069cc32f unchanged. See _fp_battery_results.md.

## Gadget facts learned
- MUST call `fp._set_one(L)` before building isolated blocks (sets `A._ONE`).
- `silu(S)/SILU_S = 1.0` exact; residue is from `1/SILU_S` W_down scaling, ~value*2^-24.
- `_step_ge` exact at ANY scale for near-integer forms (verified to 20000) — the COMPARES
  are fine; the DOUBLING is the problem.
