# L10 tail-bank enumerated→computed collapse (2026-07-09)

Branch `collapse-l10-tail` (off `59a9de19`, golden `e50521f3`). Goal: collapse the
largest ENUMERATED per-byte-value correction families in
`neural_vm/unified_compiler/ops/l10_ops.py` into a single COMPUTED authoring
generator, **byte-identical to golden `e50521f3`** (verdict-neutral by
construction — the collapse is an authoring refactor, NOT a semantic change).

## What was collapsed

Four STACK0 byte-value-writeback banks inside
`_tail_bit32_result_correction_rules` are all instances of ONE shape: *"under a
fixed structural evidence gate, for each byte value `v = lo | (hi<<4)` observed
in a (nibble-lo, nibble-hi) source lane, write OUTPUT byte = v."* Three still had
**hand-written 16×16 nested nibble loops** as their DEFAULT-ON path; they now
route through the shared, byte-identical `_byte_value_writeback_rules`
generator (the fourth, `stack0_store_loaded`, was already collapsed in task J5).

| family | rules | source lane | match wts | thr | skip | commit |
|---|---|---|---|---|---|---|
| `stack0_pop_loaded_output_rules` | 255 | OUTPUT (READ==WRITE) | 0.05/0.05 | 12 | `lo==hi==0` | `8c6fcf79` |
| `stack0_store_top_e8_from_e0` (loop) | 255 | OUTPUT (READ==WRITE) | 0.001/0.001 | 4300 | `lo==hi==0` | `0dc98350` |
| `stack0_store_top_e0_output_rules` | 254 | **cross-lane** ALU→OUTPUT + OUTPUT tie-break | 1.0/1.0 (+0.001 tie) | 25 | `lo==hi==0`, `v==0xE0` | `7184bc1f` |

The `stack0_store_top_e0` collapse required a minimal, general extension of
`_byte_value_writeback_rules`: an optional `tie_break_lo_base/tie_break_hi_base/
tie_break_weight` secondary nibble-match pair (for cross-lane READ families that
add an OUTPUT tie-breaker), plus explicit `write_lo_base/write_hi_base` (defaults
`OUTPUT_LO`/`OUTPUT_HI`, unchanged for the plain-OUTPUT callers). The
`stack0_store_top_e8_from_e0` family's trailing single `byte_39` special rule is
appended UNCHANGED (not part of the loop).

## Byte-identity proof

* **Authoritative whole-model gate**: `tools/_isa_golden_hash.py` (disk_cache=False)
  == `e50521f3...` after EVERY edit (before, ×3 after each collapse, and after the
  helper extension). FFN unit count unchanged: 42149 → 42149.
* **Independent rule-by-rule check**: rebuilt each family's enumerated form inline
  and compared to the helper output field-by-field (name, threshold, conditions,
  writes, scope, dominates_at, gate). Result: `byte-identical=True` for all three
  (counts 255 / 255 / 254 — enum == helper).

## Honest LOC / magic-number accounting

These loops were **already computed**: the per-rule byte value is `v = lo|(hi<<4)`
(computed, not an enumerated literal), and the only literals are the family's
shared weights/threshold (each appearing once). So the collapse is
**LOC- and literal-neutral** — it does NOT shrink the file. This matches the
project's documented reality (memory `project_core_loc_reduction_reality`:
"parameterizing == loop verbosity; consolidation is LOC-NEUTRAL").

| metric | base `59a9de19` | current | delta |
|---|---|---|---|
| total file lines | 13882 | 13915 | +33 |
| non-comment code lines | 9753 | 9763 | **+10** (helper extension only) |
| comment lines (`#`) | 3280 | 3303 | +23 (per-collapse documentation) |
| magic-number literals | 9600 | 9605 | +5 |

The +10 code lines are the reusable `_byte_value_writeback_rules` extension
(tie-break pair + write-band plumbing), which now serves 4 call sites.

## The real win (and the real magic-number concentration)

The value here is **single-source-of-truth**: all four STACK0 byte-writeback
banks now flow through one generator, so the shape lives in exactly one place.

The larger literal concentration in `_tail_bit32_result_correction_rules`
(~2.1k literals, L8817–11165) is a flat tuple of **bespoke, individually-named**
correction rules (`tail_pc_byte0_1a_from_taken_branch...`,
`tail_mem_store_addr0_e8_from_nested_local...`, etc.), each a distinct structural
case with its own conditions/threshold/write value. These do **not** share one
byte-arithmetic function and are therefore NOT collapsible to a computed
generator without changing semantics — they are load-bearing corrective RULES,
not verbose code.

`ax_add_carry_rules` (256 rules, write = `(old_value+1)&0xFF`) is likewise
**already** a computed generator (the write value is a computed increment, the
conditions are computed from `old_lo`/`old_value>>4`); it has no enumerated
magic-number literals to collapse and was left as-is (it is the target
end-state form).

## Gates

* golden `e50521f3` held byte-identical (authoritative).
* `pytest tests/test_smoke.py` == 51/51 (byte-identity of weights guarantees the
  smoke verdict is unchanged from baseline).
</content>
