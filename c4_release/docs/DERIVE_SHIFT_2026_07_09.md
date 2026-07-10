# Deriving the SHIFT family (SHL/SHR) from BLOG_SPEC

Task #448. Branch `derive-shift`. Flag `C4_DERIVE_SHIFT` (DEFAULT-OFF).
Golden `e50521f3` (flags-OFF `f725c06e`).

## Mission

Derive the SHIFT family (SHL opcode 23, SHR opcode 24) from `docs/BLOG_SPEC.md`
§597-599 building blocks — shifts as *multiply/divide by powers of two* with
overflow handled by the *mod-by-floor* primitive — replacing the hand-authored
per-`(value, shift)` bit-op enumeration, with ZERO per-op magic constants. Fix
the DSL/lowering, never a magic-number FFN rule. Follow the #391 JMP /
`C4_DERIVE_IMM` pilot.

## The BLOG_SPEC building blocks

Three passages, read literally, give the whole family:

* **§599 (Shifts):** "Left and right shifts could simply be handled by 32
  different multiplications by powers of two each, but it does not properly
  handle overflow, however overflow can be handled via the modulus by floor
  mentioned above."
* **§555 (MAGIC floor):** `floor(x)` without a floor op — at scale `MAGIC =
  2^23`, `(x + MAGIC) - MAGIC` rounds to an integer; for positive values
  subtract `0.5` first to get true floor.
* **§560 (Bit-range extraction):** "multiply by negative powers of two, do a
  floor, then a mod by a power of two corresponding to the length of the
  range."

So the two directions are, with NO bit-shift operator and NO `& 0xFF` mask:

```
SHL(v, s) = (v * 2**s) mod 2**8      # §599 multiply-by-2^s + §560 mod-by-floor overflow cap
SHR(v, s) = floor(v / 2**s)          # §599 divide-by-2^s + §555 floor
```

where the modulus is itself expressed through the floor primitive (the §555
"modulus by floor"):

```
x mod m = x - m * floor(x / m)
```

The powers of two come from ONE derived table (`2**s`); the byte cap is the
generic mod-by-floor; there is no per-shift hand-typed constant and no per-op
mask. This is the spec's "powers of two + mod-by-floor" recipe, verbatim.

## What was hand-authored (the enumeration being replaced)

The live golden path is `alu_mode='lookup'` (the default). The L13 shifts FFN
is a 4096-unit lookup table built by `ops/l13_ops._layer13_shifts_rules`:
`8 shift amounts × 16 a_hi × 16 a_lo × 2 directions`. Each unit is a 5-way AND
(`MARK_AX + ALU_LO[a_lo] + ALU_HI[a_hi] + AX_CARRY_LO[s] + AX_CARRY_HI[0]`,
gated on `OP_SHL` / `OP_SHR`) that writes the result nibble one-hots
`OUTPUT_LO[result_lo]` / `OUTPUT_HI[result_hi]`.

The ONLY place shift *semantics* were hand-authored was the per-`(value, shift)`
result each unit stores, computed by two magic Python bit-op lambdas:

```python
_layer13_shl_rules:  shift_fn = lambda v, s: (v << s) & 0xFF
_layer13_shr_rules:  shift_fn = lambda v, s: v >> s
```

These two lambdas — the bit-shift operators and the `& 0xFF` mask — are the
"hand-authored shift enumeration" the mission targets.

## The derivation

New module `neural_vm/unified_compiler/shift_semantics_dsl.py` (158 lines)
supplies the spec building blocks as integer primitives — **no `<<`, no `>>`,
no `& 0xFF`** anywhere in the derivation path:

* `power_of_two(s)` -> `2 ** s` — the §599 "multiplication by powers of two"
  (one derived table, no per-shift constant).
* `magic_floor_div(n, d)` -> `n // d` — the §555 floor (integer floor at
  derivation time; the MAGIC-floor trick / the fp32 nibble-window floor is what
  computes it in the running network).
* `mod_by_floor(x, m)` -> `x - m * magic_floor_div(x, m)` — the §560/§555
  "modulus by floor" (no `%`, no mask).
* `shift_result(value, shift, direction=...)`:
  * `left`  -> `mod_by_floor(value * power_of_two(shift), 256)`
  * `right` -> `magic_floor_div(value, power_of_two(shift))`

`ops/l13_ops._layer13_shl_rules` / `_layer13_shr_rules` swap the magic lambdas
for `shl_result` / `shr_result` when `shared.derive_shift_enabled()` is true.
The lookup-table STRUCTURE (rule count, 5-way AND, one-hot writes, unit walk
order) is untouched — only the stored result VALUE is now derived. The single
structural constant is the byte width (`2**8`), the modulus the whole
byte-addressed VM is built on — a spec-structural safety factor, not a per-op
magic number.

### Flag plumbing (mirrors `C4_DERIVE_IMM`)

* `ops/shared.derive_shift_enabled()` — reads `C4_DERIVE_SHIFT` (DEFAULT-OFF).
* `full_vm_compiler_dynamic.py` — `C4_DERIVE_SHIFT` registered in BOTH
  cache-key snapshots (the in-proc memo `_inproc_snapshot` and the disk-cache
  `kwargs_snapshot`) so a derived build never shares a memo/disk entry with a
  hand build.

## Verdict: correct

**Byte-identical, proven three ways:**

1. **Exhaustive formula proof** (import-time assert in the module):
   `shl_result(v,s) == (v<<s)&0xFF` and `shr_result(v,s) == v>>s` for every
   `v in 0..255`, `s in 0..7` (2048 pairs each). Any regression trips the
   import.
2. **Rule-level parity**: `_layer13_shifts_rules(100.0)` produces 4096 rules
   whose `repr` is identical between flag-OFF and flag-ON (0 / 4096 mismatches).
3. **Whole-model golden hash** (`tools/_isa_golden_hash.py`, CPU, disk-cache
   off):
   * flag-OFF: `e50521f32b0ed952…` (== golden, unchanged by this branch)
   * flag-ON (`C4_DERIVE_SHIFT=1`): `e50521f32b0ed952…` (**identical**)

   The two builds are the *same tensors*, so every downstream verdict (smoke,
   cpu_full_trace) is identical to the golden build by construction.

## Bonus fixes: none possible (and why)

The clean derivation is byte-identical to the hand path, so it does NOT change
any verdict — it cannot fix a currently-failing shift program because the model
weights are unchanged. The mission asked to check whether it fixes failing
shifts; the honest answer is **no**, and the reason is architectural, not a
derivation defect:

The failing shift smoke cases are the 8-bit-shift ones
(`test_shl_8bit` = `1 << 8 = 256`, `test_shr_8bit` = `0x100 >> 8 = 1`). These
fail because the L13 lookup table only enumerates shift amounts `s in 0..7`
(the `AX_CARRY_LO[s]` one-hot spans one byte's shift range) and operates on ONE
byte with no inter-byte carry. A shift of 8 crosses a byte boundary — it is a
multi-byte / whole-word operation the single-byte lookup structure does not
express, regardless of how the per-cell value is computed. The derived formula
handles `s >= 8` correctly in the abstract (`shl_result(1, 8) == 256`), but the
table it feeds has no `s == 8` unit and no byte-1 output lane, so the value
never reaches the model. Fixing 8-bit shifts requires *widening the lookup
structure* (a second byte lane + a `q = s // chunk_bits` byte-offset select,
the `alu/ops/shift.py` composite's job), which is out of scope for a
byte-identical value-derivation and would be a structural (weight-changing)
change, not a spec-formula swap.

The in-range shift cases (`TestSmokeShift::test_shl` = `21 << 1 = 42`,
`test_shr` = `84 >> 1 = 42`) pass on the golden build and continue to pass
byte-identically under `C4_DERIVE_SHIFT=1`.

## LOC + magic-constant collapse

The derivation is **magic-constant-collapsing, not line-collapsing**: the
4096-rule lookup TABLE is the load-bearing structure (fixed, width-locked count
— per memory `project_core_loc_reduction_reality`), so it is neither deleted
nor shrunk. What collapses is the *semantic authoring surface*:

| | hand path | derived path |
|---|---|---|
| shift-semantics authoring | 2 magic bit-op lambdas (`(v<<s)&0xFF`, `v>>s`) | 1 shared spec formula (`shift_result`, direction-parameterized) |
| per-op magic constants | `<<`, `>>`, `& 0xFF` mask, per-direction | 0 (powers of two from one table; byte cap = generic mod-by-floor) |
| powers of two | implicit in the `<<` operator, per case | 1 explicit derived table `power_of_two(s)` |
| new spec module | — | `shift_semantics_dsl.py`, 158 lines (reusable §555/§560/§599 primitives) |

Net: `+229` lines added (mostly the reusable spec-primitive module + its
docstrings + the exhaustive proof), `−2` magic bit-op lambdas swapped for a
spec formula. The value is the **magic-constant collapse** (bit-ops + mask ->
spec building blocks with zero per-op constants), matching the campaign's
"fix via compiler/DSL, zero per-op magic" methodology, not a raw LOC cut.

## Honest limits

1. **Byte-identical => verdict-neutral.** The derivation is a faithful
   re-authoring of an already-correct lookup, so it moves no failing program.
   The value is architectural (spec-grounded, magic-free authoring), not a
   pass-count gain.
2. **8-bit (and wider) shifts stay blocked** by the single-byte lookup
   structure, not by the value derivation. See the bonus-fix section — the fix
   is a structural widen (`alu/ops/shift.py` composite), out of scope here.
3. **Lookup mode only.** This derives the `alu_mode='lookup'` golden path (the
   4096-rule L13 table). The `alu_mode='efficient'` `ALUShiftComposite`
   (`efficient_alu_neural.py`, the 4-stage BD->GE precompute/select pipeline)
   already embodies the same "multiply/divide by powers of two + step-function
   floor/mod" spec idea in its `ShlPrecomputeFFN` / `ShrPrecomputeFFN`; routing
   it through this DSL (deleting `build_shl_layers`/`build_shr_layers`) is a
   natural follow-up but not needed for the default golden build.
4. **`wide_shift_rules` is a DIFFERENT enumeration** (`wide_alu_dsl.py`, a
   per-`(byte, shift, value)` lookup with no inter-byte carry) that is not on
   the live golden path; it was not touched.

## Files

* `neural_vm/unified_compiler/shift_semantics_dsl.py` (new) — the §555/§560/§599
  spec primitives + `shift_result` / `shl_result` / `shr_result` + exhaustive
  proof.
* `neural_vm/unified_compiler/ops/shared.py` — `derive_shift_enabled()`.
* `neural_vm/unified_compiler/ops/l13_ops.py` — `_layer13_shl_rules` /
  `_layer13_shr_rules` swap the magic lambda for the spec formula under the flag.
* `neural_vm/unified_compiler/full_vm_compiler_dynamic.py` — `C4_DERIVE_SHIFT`
  in both cache-key snapshots.

## Gates

| gate | result |
|---|---|
| golden flag-OFF | `e50521f3…` (== golden, every edit) |
| golden flag-ON (`C4_DERIVE_SHIFT=1`) | `e50521f3…` (**byte-identical**) |
| 4096-rule repr parity (OFF vs ON) | 0 mismatches |
| exhaustive formula proof (2048×2 pairs) | PASS |
| smoke (flag-ON) | byte-identical model => identical to golden smoke |
