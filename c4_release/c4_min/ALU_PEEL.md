# ONE shared PEEL gadget across ADD / MUL / DIV (`alu_peel.py`)

Every recurrent ALU op has, per iteration, a **PEEL** (shift the value by one
digit + split off the next nibble) and an op-specific **COMBINE** (carry-add /
partial-product accumulate / quotient-select + `q*b`-subtract). The peel is
structurally the SAME gadget across ops. This module factors it into **ONE shared
stored weight block** that ADD / MUL / DIV all reference, keeping only the combine
distinct.

Code: [`alu_peel.py`](alu_peel.py) · tests: [`test_alu_peel.py`](test_alu_peel.py)
· builds on the per-op registry [`alu_units.py`](alu_units.py) /
[`ALU_UNITS.md`](ALU_UNITS.md).

---

## 1. What the peel IS (and where it lives today)

The peel is the **radix-16 nibble floor/mod split** — the sharp `_step_ge`
staircase `floor(form / 16)` that snaps a carry-save value into clean integer
nibbles + a carry. It is the same primitive everywhere:

| op | peel site (module) | lane / band | radix |
|----|--------------------|-------------|-------|
| **ADD/SUB** | `_byte_add_block` (`nibble_alu32`) | byte sum → `ADD_RES`/`SUB_RES` nibbles | 16 (kmax 32, sum < 512) |
| **MUL** | `_mul_split_block` + `_nibble_carry_round` (`mul-carry*`) | `MCOL` / `MC1` double buffer | 16 (kmax 15, col < 256) |
| **DIV** | `_nibble_carry_round` on `QB` (`qbc*`), `KB` (`kb-c*`); `div_radix16_lean` `split1`/`split2`/`shift` | `QB` / `KB` / `LR_R` | 16 (kmax 15) |

Two important facts:

* The peel **primitive** (`_floor_div_pow` / `_floor_div_pow2` /
  `_nibble_carry_round`) is **already ONE shared library function** in
  `nibble_alu32` — imported and reused by `div_radix16_lean`,
  `div_radix16_hardened`, MUL and ADD. There is no per-op copy of the *staircase
  logic*.
* What is **NOT** shared is the STORED weight BLOCK. Each op routes its peel
  through a **different residual band** (ADD: byte sum; MUL: `MCOL`/`MC1`; DIV:
  `QB`/`KB`), so the SAME gadget compiles to **distinct stored tensors**. The
  whole-tensor weight-tie ([`weight_dedup.py`](weight_dedup.py)) collapses copies
  **within** a band but cannot tie **across** bands.

## 2. The two conditions that make sharing work

Both are honoured (per the de-entanglement `register_residual_band` approach):

* **(a) canonical operand-digit lane** — a dedicated `ALU_PEEL_IN` / `ALU_PEEL_OUT`
  band (`extend_layout_for_peel`) that every op routes the peeled value THROUGH.
  One peel block reads/writes the SAME place, so it is byte-identical no matter
  which op called it → ties to ONE stored copy.
* **(b) common peel radix** — every op's peel is already **radix-16** (nibble),
  including ADD (which splits its *byte* sum into two *nibbles*). So ONE shared
  nibble-peel is enough; there is no byte-radix peel to reconcile. The two kmax
  regimes (`nibble15` for MUL/DIV carry-save columns < 256; `nibble32` for ADD's
  byte sum < 512) are exposed as named `SHARED_PEEL_WIDTHS` so each op references
  the matching one — we do **not** force one kmax (that would bake dead thresholds
  into the narrow ops or clip the wide one).

## 3. What it buys — measured, honest

`python -m c4_min.alu_peel` prints the before/after on the REAL DIV(recurrent) +
MUL stacks (`measure_peel_consolidation`):

```
=== shared PEEL consolidation (distinct STORED peel tensors) ===
  peel apply-sites (refs): 13
  distinct peel tensors  : 3 (private per-op bands) -> 1 (ONE canonical lane)
  private tie-groups (each a distinct stored tensor):
    - alu-div-qbc-0  (x6)      # DIV per-iteration QB carry round
    - alu-mul-carry0 (x4)      # MUL ripple carry round, MCOL buffer
    - alu-mul-carry1 (x3)      # MUL ripple carry round, MC1 buffer
  peel nnz (distinct)    : 3810 -> 1374  (saved 2436)
  net distinct blocks    : saved 2 (canonical-COMBINE, route-free)
                           / 0 (naive: +2 route block(s), +108 nnz)
```

* **13 carry-round peel apply-sites** across DIV+MUL are stored as only **3
  distinct tensors** today — because 3 different bands (`QB`, `MCOL`, `MC1`) carry
  the *same* radix-16 carry-round gadget. (The 6/4/3 copies within each band
  already tie.)
* The **canonical lane collapses those 3 → 1** shared stored peel: **net 2 distinct
  blocks + 2436 nz weights saved** when the op's COMBINE writes/reads the canonical
  lane directly (the route-free form).
* **Honest caveat:** the *naive* form that keeps each op's private band and adds a
  copy-in / copy-out block per lane **breaks even** (0 net blocks) — the route glue
  costs back the 2 blocks the share saved. The real win needs the COMBINE to target
  the canonical lane (no route glue). Both numbers are reported.
* **Scope honesty:** only the `_nibble_carry_round` sites are the SAME gadget that
  shares byte-identically. The MUL `mul-split` (a PP→column split, a different
  block shape) and the DIV `kb-raw` (a multiply fan-out, not a peel) are
  peel-*adjacent* and are **NOT** claimed as consolidated by this block.
* **This is a WEIGHT-COUNT / consolidation win, NOT a depth reduction.** The
  recurrence applies the peel the same number of times either way; only the count
  of DISTINCT stored blocks drops.

## 4. Byte-identical (it's a refactor)

The shared canonical-lane peel computes exactly the same floor/mod split as each
op's private peel. `verify_shared_peel_byte_exact` runs a battery (edge + random
carry-save column stacks) through (1) the op's private peel on its own band and
(2) `route-in → shared peel → route-out` on the canonical lane, at **fp64 AND
fp32**:

```
[fp64] snapped-nibble mismatches: 0/2412 | raw residue priv=5.4e-06 shared=2.0e-05; snap margin 0.500 -> BYTE-EXACT
[fp32] snapped-nibble mismatches: 0/2412 | raw residue priv=0.0e+00 shared=7.8e-03; snap margin 0.492 -> BYTE-EXACT
```

The byte-exactness criterion is the **snapped integer nibble** the downstream sharp
staircase reads (the peel output is always consumed by a half-integer-thresholded
`_step_ge` in the real ALU, which snaps sub-integer residue). The extra copy hops
add a tiny residue (fp32 ~8e-3) that stays **far below the 0.5 snap margin**
(margin 0.492), so every nibble snaps identically — byte-exact. The width-8 MUL
carry-round is the **8-column prefix** of the width-9 canonical peel, so ONE block
services both widths (`test_width8_mul_carry_equals_width9_canonical_prefix`).

## 5. Gating + registry wiring

* **`C4_SHARED_PEEL`** (env, default **OFF**). OFF == the current private per-op
  peels — **byte-identical to golden** (the canonical band is never allocated;
  `extend_layout_for_peel` is only called on the shared path, so a private build's
  `L.D` is unchanged, and `nibble_alu32` is untouched). ON == ops route their peel
  through the canonical lane + the ONE shared block. The default stays OFF until a
  build wires the canonical lane end-to-end; the byte-exact proof is the promotion
  gate.
* **Registry:** `register_peel_units()` registers the peel as its own op family
  (`op="peel"`) in `alu_units.py`, one unit per width (`nibble15` / `nibble32`).
  The unit's `wired` follows `C4_SHARED_PEEL` (a measure-registered but not-selected
  unit while OFF), mirroring the sibling divide-unit promotion path. The peel is
  thus selected *per radix* through the same registry the ADD/MUL/DIV units use.

## 6. Fit-to-model budget effect

The shared peel is a **stored-weight** consolidation; it does **not** change the
`select_units_for_model` **layer/depth** budget (the divide is still ~80 applied
blocks; the recurrent fold still stores its ~17 layers). What it changes is the
**distinct stored block / nz-weight** count inside those layers: the ~13
carry-round peel sites across ADD/MUL/DIV reference ONE shared peel tensor instead
of 3, so a size-constrained fit (weight budget, not layer budget) sees a smaller
unique-weight footprint. The fit selector's layer accounting is unchanged; the
weight accounting drops by the consolidation above.
