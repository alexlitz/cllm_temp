# SHIFT_TIGHT — the tightest fp32-exact SHL/SHR: a DIRECT 8×8 nibble select

`shift_tight_nibble.py` builds the **tightest** 32-bit SHL/SHR shifter by
replacing the two big overheads of the landed shifters with their minimal
forms:

* the **log-stage pipeline** (5 conditional `2**k` mux stages over 32 bit-planes
  in the bit-granular `nibble_bitwise` gadget, or 3 coarse log stages over 8
  nibble-planes in the `shifter_bakeoff` nibble mux-tree) → a **single DIRECT
  8×8 nibble select**, no log stages at all; and
* the **per-value amount decode** (the mux-tree scans all 64 shift counts to
  build its coarse-amount bits and fine multiplier) → the **equality-pulse
  gadget** on the *scalar* `c = n//4` and `r = n mod 4`, an 8- and 4-wide
  one-hot instead of a 64-wide scan.

Everything is a nibble (≤ 15), a one-hot (0/1), a small carry (≤ 7), or a
product `v·2^r ≤ 120` — so the whole shifter is **fp32-exact** (max relu argument
≈ 22 300 ≪ 2²⁴), with **no MAGIC constant and no fp64**.

## The design

A shift by `n` (0..31; `n ≥ 32 → 0`; unsigned/logical, both directions) is
`n = 4·c + r` with `c = n÷4` (coarse whole-nibble shift, 0..7) and `r = n mod 4`
(fine sub-nibble shift, 0..3).

1. **Amount decode** (2 tiny blocks, the equality-pulse gadget).
   Block 1: `c = floor(n/4)` and `r = n − 4c` share ONE floor staircase (kmax=7),
   plus `keep = [n < 32]`.  Block 2: the coarse one-hot `ceq[k] = [c==k]` (k=0..7,
   8 point pulses), the fine one-hot `feq[m] = [r==m]` (m=0..3, 4 pulses), the
   fine multiplier `POW`, and `RNZ = [r≠0]` (the SHR whole-nibble bypass) — all
   read from the *scalar* `c`/`r`, so 4/8-wide, not a 64-wide scan.

2. **Coarse = a DIRECT 8×8 nibble select** (the ~64-term core, NO log stages).
   For each output nibble `j`,
   `out_nib[j] = Σ_k in_nib[j−k]·ceq[k]` (SHL) / `Σ_k in_nib[j+k]·ceq[k]` (SHR),
   over the valid neighbours only (triangular: **36 terms per direction**).  Each
   term is ONE `_guard` unit (window `ceq[k]==1`, gate value `in_nib[j∓k]`) — the
   whole coarse shift is a single FFN block, no stage buffers.

3. **Fine = a minimal sub-nibble shift** (0..3 bits).  Per coarse nibble `v` form
   `p = v·POW` (SHL `POW = 2^r`, SHR `POW = 2^(4−r)` for r>0), then split `p ≤ 120`
   into `p mod 16` and the carry `floor(p/16) ≤ 7` with ONE shared `floor/16`
   staircase (via `_floor_div_pow2`), and merge the carry into the neighbour
   (UP for SHL, DOWN for SHR; SHR's r=0 is the pure whole-nibble bypass).

4. **Lean muxes.**  Every gated unit is the bare `_guard` AND primitive (1 window
   + 1 gate term → **4 nz**) or a bare copy/step — no guard-window lowering
   overhead.  Achieved **~4.1 nz/unit** (the mux-tree's guard-window lowering ran
   ~8.5 nz/unit).

5. **`n ≥ 32 → 0`.**  `keep = [n < 32]` gates the final recompose write, so any
   count ≥ 32 zeroes the result — matching `ref_interpret(mask=0xFFFFFFFF)`
   (`(pop <</>> n) & 0xFFFFFFFF`, `n` UNMASKED).

## Results

*(depth = SwiGLU blocks; weights = non-zero params across all six tensors of
every block; measured by `shift_tight_nibble.measure`.)*

| variant | op | depth | weights | coarse-select | fine | amount-decode | nz/unit | fp32 | byte-exact |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **tight-direct-8×8** | SHL | **6** | **1 394** | **168** | 788 | 438 | 4.09 | yes | 332/332 32-bit, 60/60 8-bit |
| **tight-direct-8×8** | SHR | **6** | **1 465** | **168** | 875 | 422 | 4.15 | yes | 332/332 32-bit, 60/60 8-bit |
| nibble mux-tree *(ref)* | SHL | 8 | 4 140 | *(3 log stages)* | *(peel)* | *(64-scan)* | ~8.5 | yes | — |
| bit log-shifter *(landed ref)* | SHL | 8 | 5 269 | *(5 log stages)* | — | — | — | yes | — |

Byte-exactness is the lean **arithmetic sim on the nibble/one-hot bands** (a tiny
residual dict, NOT a DIM-8192 forward): the edge grid `x ∈ {0x80000000,
0xFFFFFFFF, 0xDEADBEEF, 0x1}` × `n ∈ {0,1,7,15,16,31,32,40}` + 300 random `(x,n)`
against the 32-bit reference (`(x <</>> n) & 0xFFFFFFFF`, which the driver pins
equal to `nibble_pure_forward_complete.ref_interpret(mask=0xFFFFFFFF)`
**200/200**), plus 60 random byte cases against the 8-bit `isa.interpret`.  A
separate 5 000-sample stress with `n ∈ 0..255` is **5 000/5 000** for both
directions.

## Verdict — where the nz actually goes

**The tight shifter lands at 1 394 nz (SHL) / 1 465 nz (SHR)** — **~34–35 % of
the 4 140-nz nibble mux-tree** (2 746 / 2 675 nz UNDER it), at **6 blocks vs 8**,
fully byte-exact and fp32-exact.  That is roughly a **3× cut** over the mux-tree
and a **~3.6–3.8× cut** over the landed bit-log-shifter (5 269 nz).

The breakdown is the honest headline:

* **The coarse DIRECT 8×8 select IS "a few hundred": just 168 nz.**  It is the
  ~64-term core the design targets (36 gated terms × 4 nz + 8 SET-clears × 3 nz =
  168), and it replaces the mux-tree's entire coarse **log pipeline** (3 stages,
  ~1 000 nz across blocks 1–2 there) with ONE block.  This is the real win: the
  coarse shift, which the log-shifters spend most of their depth and a big chunk
  of their weight on, collapses to **168 nz / 1 block**.

* **The amount decode is now cheap: ~420–440 nz.**  The equality-pulse gadget on
  the scalar `c`/`r` (8- and 4-wide one-hots) replaces the mux-tree's ~1 900-nz
  64-value per-amount scan — a ~4× cut on this component alone.

* **The FINE sub-nibble shift DOMINATES the total: ~790–875 nz.**  This is the
  honest caveat.  The coarse select and amount decode are both firmly in the
  "few hundred" range, but the fine peel is not: splitting each of the 8 nibbles'
  `v·2^r ≤ 120` product into `mod 16` + `carry` costs a kmax=7 `floor/16`
  staircase per nibble (already halved by routing ONE staircase to both
  destinations via `_floor_div_pow2`, from 968 → ~450 nz for the peel), plus the
  8 six-weight multiplies.  That fine machinery — not the coarse select — is what
  keeps the **total** at ~1.4K rather than a few hundred.

So the answer to *"can the coarse select be a few hundred nz?"* is **yes,
emphatically — 168 nz**, and the log-stage overhead the mux-tree paid for the
coarse shift is entirely eliminated.  The total does **not** reach a few hundred,
and the reason is precise and worth stating plainly: **the per-nibble fine-shift
peel (the `mod 16` + carry split), not the coarse select or the amount decode,
is the dominant cost of an fp32-exact nibble shifter.**  Even so the tight
direct-8×8 is the **smallest** byte-exact fp32 shifter measured here — ~3× under
the mux-tree and ~3.7× under the landed bit-log-shifter, at shallower depth.

## Reuse / boundaries

Reuses only the shared primitives from `nibble_alu32` (`_empty_spec`,
`_floor_div_pow`, `_floor_div_pow2`, `_mul_gate`, `_guard`, `_step_ge`, `_ident`,
`RELU_S`) and the equality-pulse / one-hot construction pattern (a difference of
two `_step_ge` ramps).  Does **not** edit any existing file, and does not touch
`shifter_bakeoff.py` / `shift_attention_bench.py` / `mul_*` (other agents own
those).  Run the measurement with `python -m c4_min.shift_tight_nibble`.
