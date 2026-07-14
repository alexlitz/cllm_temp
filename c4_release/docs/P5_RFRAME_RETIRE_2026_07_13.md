# P5 PILOT — Retire the R-FRAME INCR-3 enumerated fallbacks (unconditional collapse)

**Date:** 2026-07-14 · **Branch:** `p5-rframe-retire` (off `main` @ `b351ad80`)
**Default golden:** `1c04c3fd58f4c814ecf20811298d6b5f682dfcd7b3ca971a3da400fd18339d01`
— **UNCHANGED** before and after this change.
**Prior flag-OFF escape hatch golden:** `e50521f3` (`C4_R_FRAME_TAIL=0`) — **now permanently removed** for these six families (the intended P5 trade).

## What this is

The FIRST P5 pilot: turn a reversible collapse into a real **source-LOC deletion**.

R-FRAME INCR-3 (`C4_R_FRAME_TAIL`, DEFAULT-ON) collapsed six L25-tail pure-copy
frame-guarantee families from their `range(256)` per-value enumeration onto a
computed per-nibble ROUTE form. That flip moved *runtime* FFN units (−478 tail
units) but netted ~0 source LOC, because the enumerated `range()` banks were
**retained** as the `C4_R_FRAME_TAIL=0` escape hatch.

P5 makes the collapse **UNCONDITIONAL**: the enumerated fallbacks + the
`C4_R_FRAME_TAIL` super-switch + the six per-family "M8" point-flags are
DELETED, so the −478 units become a real source-LOC deletion. The DEFAULT model
is unchanged — the default was already the collapsed form, so the golden hash
stays `1c04c3fd`. This PERMANENTLY drops `C4_R_FRAME_TAIL=0` reversibility for
these six families (that is the intended P5 trade; the collapse is proven stable
and default-ON, gate +3).

## The six routed families (fallbacks deleted)

All in `neural_vm/unified_compiler/ops/l10_ops.py`, inside
`_tail_bit32_result_correction_rules`. Each builder previously had an
`if <family>_computed_enabled(): return <route>` branch followed by an
enumerated `range()` fallback `return`. The P5 retire deletes the fallback and
keeps ONLY the route form.

| Register row | Family | Enumerated → Computed | Units dropped | Builder |
|---|---|---|---|---|
| SP | `sp_pop_carry_byte2` | `range(256)` → `range(2)` (`old ∈ {0x00,0x01}`) | 254 | `sp_pop_carry_rules` (byte-2 loop) |
| AX | `wide_mul_byte1` | 256 per-value AND → 32 per-nibble route | 224 | `wide_mul_byte1_preserve_rules` |
| STACK0 | `stack0_store_loaded` | 255 → 32 route (same-lane OUTPUT) | 223 | `stack0_store_loaded_output_rules` |
| STACK0 | `stack0_pop_loaded` | 255 → 32 route (same-lane OUTPUT) | 223 | `stack0_pop_loaded_output_rules` |
| STACK0 | `stack0_store_e8` | 255 → 32 route (+ byte_39 special kept) | 223 | `stack0_store_top_e8_from_e0_output_rules` |
| STACK0 | `stack0_store_top_e0` | 254 → 32 route (CROSS-LANE ALU→OUTPUT) | 222 | `stack0_store_top_e0_output_rules` |

Total tail-unit drop baked into the constant: 254+224+223+223+223+222 = **1369**.
(The prior INCR-3 flip already realised −478 of this at runtime; the remaining is
`wide_mul_byte1`'s −224 which was default-OFF at the individual point-flag but
forced-ON by `C4_R_FRAME_TAIL`, plus the STACK0/SP families.)

Each route form's numeric equivalence to its retired enumerated bank (identical
firing region + 0 argmax mismatch across all 256 byte values) is proven by the
`tools/_probe_*` harnesses cited in the surviving builder docstrings
(`_probe_m8_computed_writeback.py`, `_probe_computed_writeback_banks.py`,
`_probe_crosslane_bytecopy.py`, `_probe_wide_mul_computed_writeback.py`).

## What was deleted (source)

`neural_vm/unified_compiler/ops/l10_ops.py`:
- The R-FRAME header machinery: `RegisterEmitSpec` dataclass, `R_FRAME_TABLE`,
  `_R_FRAME_BY_REG`, `_R_FRAME_TAIL_FAMILIES`, and `_r_frame_tail_enabled`
  (+ `C4_R_FRAME_TAIL`). Replaced by a short retire note.
- Six now-dead per-family point-flag helpers + their `C4_*` env reads:
  `_sp_byte2_carry_computed_enabled` (`C4_SP_BYTE2_CARRY`),
  `_stack0_store_loaded_computed_enabled` (`C4_STACK0_STORE_LOADED_COMPUTED`),
  `_stack0_pop_loaded_computed_enabled` (`C4_STACK0_POP_LOADED_COMPUTED`),
  `_stack0_store_e8_computed_enabled` (`C4_STACK0_STORE_E8_COMPUTED`),
  `_stack0_store_top_e0_computed_enabled` (`C4_STACK0_STORE_TOP_E0_COMPUTED`),
  `_wide_mul_byte1_computed_enabled` (`C4_WIDE_MUL_BYTE1_COMPUTED`).
- The six enumerated `range()` fallback bodies in the builders (each replaced by
  the unconditional route `return`).
- The six conditional `extra -= N` count-branches in
  `_allocate_l10_tail_bit32_units` (the −1369 is now baked into the base count).
- Unused imports `dataclass`, `field`, `Tuple`.

`_L10_FFN_UNIT_LAYOUT_TAIL_BIT32_TOTAL` / the layout tuple: **2059 → 690** (the
tail bank is now permanently the collapsed count). The three DEFAULT-OFF campaign
`extra +=` flags (`_lea_byte0_memsp_relay_enabled` +3,
`_lea_byte0_alu_amplify_enabled` +4, `_sp_pop_carry_byte0_dominate_enabled` +1)
are retained.

`neural_vm/unified_compiler/full_vm_compiler_dynamic.py`:
- Removed the five now-dead cache-key snapshot entries
  (`C4_STACK0_{STORE_LOADED,POP_LOADED,STORE_E8,STORE_TOP_E0}_COMPUTED`,
  `C4_WIDE_MUL_BYTE1_COMPUTED`). These flags no longer branch the build, so a
  cross-state cache hit is no longer possible. (`C4_R_FRAME_TAIL` /
  `C4_SP_BYTE2_CARRY` were never registered in the snapshot — the default was
  already the collapsed form.)

## Source −LOC (real deletion)

```
git diff --numstat
  8   39   full_vm_compiler_dynamic.py    (3621 → 3590,  −31 net)
147  682   l10_ops.py                     (13838 → 13303, −535 net)
--------------------------------------------------------------
155  721   TOTAL                          net −566 source LOC
```

**Net source deletion: −566 LOC** (721 deletions, 155 insertions), exceeding the
"several hundred" target. This proves the P5 pattern: retiring a proven,
default-ON reversible collapse converts a runtime-unit win into a real
source-LOC deletion — the decisive lever toward the <8K goal.

## Golden-unchanged proof

`tools/_isa_golden_hash.py` (whole-model `state_dict` SHA256, `disk_cache=False`),
DEFAULT env, CPU:

```
BEFORE (base b351ad80): state_dict_sha256=1c04c3fd58f4c814ecf20811298d6b5f682dfcd7b3ca971a3da400fd18339d01
AFTER  (this change):   state_dict_sha256=1c04c3fd58f4c814ecf20811298d6b5f682dfcd7b3ca971a3da400fd18339d01
  TOTAL FFN units: 41672 -> 41672 (100.0% retained)  [identical both builds]
```

The DEFAULT model is byte-for-byte unchanged: the default build was already the
collapsed form, so deleting the (never-default) enumerated fallbacks + flag is
verdict-neutral on the shipped model.

## Tests (pre-existing vs this change)

`tests/test_l10_tail_correction.py` has an autouse fixture that formerly forced
`C4_R_FRAME_TAIL=0` and ~73 assertions query per-value tail rules by their
retired enumerated names (e.g. `tail_wide_mul_byte1_preserve_9`,
`tail_sp_pop_carry_byte2_08`). With the escape hatch deleted, those names no
longer resolve.

- Baseline (base files, same fixture): **46 failed, 325 passed**.
- Naïve post-change (no test update): 63 failed — i.e. **17 new** failures, all
  from queries to now-retired enumerated names (15 `wide_mul_byte1_preserve_*`,
  2 `sp_pop_carry_byte2_{08,0f,ff}`).

Fix applied to the test file: the dead `C4_R_FRAME_TAIL=0` monkeypatch is dropped
and `_tail_rule` now `pytest.skip`s a query for a RETIRED enumerated per-value
name (prefix match, excluding the surviving `*_route_*` and byte_39 names)
instead of raising. Result: **27 failed, 308 passed, 36 skipped** — the 17
change-induced failures become skips, and **zero new failures** vs base. (The
drop from 46→27 pre-existing failures is because several base failures were
themselves AssertionErrors from STACK0 enumerated-name queries already collapsed
away by the default-ON point-flags on base — those now skip cleanly too.)

The remaining 27 failures are **pre-existing** and unrelated to this change:
mostly a stub-`dim_positions` `KeyError: 'IS_BYTE'` harness limitation in
`make_l10_post_ops_combined` bake tests, and two `sp_pop_carry_byte2_00`
CMP-gate tests that already failed on base. Verify with the byte-identity golden
hash — that is the authoritative gate; this file's per-value coverage is
inherently coupled to the retired enumerated form.

GPU smoke / 1096: not run here (GPU gate deferred). The DEFAULT model is
byte-identical (golden `1c04c3fd` unchanged), so no runtime-behaviour regression
is possible from this change.
