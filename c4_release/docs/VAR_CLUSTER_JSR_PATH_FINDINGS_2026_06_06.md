# Var-Cluster JSR-Path Fix — Findings (2026-06-06)

JSR-path sibling of commit `f4f9103d` (L3 mem_byte_0_default counter-write
at L14). Adds `layer14_jsr_mem_default_suppress` (phase=14.46) gated on
`MEM_STORE=1 AND MEM_ADDR_SRC=0` to cancel the L3 `MEM DEFAULT` +0.940
baseline at PSH/JSR/ENT store positions.

## What worked

- Compilation succeeds; L14.ffn reports 1890 live units (1891 baked, 1
  dead unit from `demo_phase6_wave7` no-op pruned by `_right_size_ffns`).
  The new 8 units bake correctly and the W_down writes survive the
  post-bake right-size pass at their shifted indices (1862-1869).
- Smoke regression: 45 passed + 6 failed = 45/52 (matches f4f9103d
  baseline exactly). Same 6 failures (test_lea_basic, test_si_li_*,
  test_sc_lc_roundtrip); no new failures.
- The `MEM_addr1` slot at JSR step 0 now predicts **0xff** correctly
  (was `0x00` pre-fix). This was the original divergence point flagged
  by the 1096 attribution diagnostic for var_simple_0, if_var_0, and
  var_three_0.

## Why all 3 1096 tests still fail

The first-token divergence has **shifted** from `MEM_addr1` (the slot
the brief targeted) to `MEM_addr2`, with a new error mode:

| Test          | Old divergence (pre-fix)            | New divergence (this fix)         |
|---------------|--------------------------------------|------------------------------------|
| var_simple_0  | `MEM_addr1`: 0xff vs 0x00            | `MEM_addr2`: **0x00 vs 0x11**      |
| if_var_0      | `MEM_addr1`: 0xff vs 0x00            | `MEM_addr2`: **0x00 vs 0x11**      |
| var_three_0   | `MEM_addr1`: 0xff vs 0x00            | `MEM_addr2`: **0x00 vs 0x11**      |

For JSR step 0 with SP=0xFFFC the return-address bytes are
`0xFC, 0xFF, 0x00, 0x00` (little-endian). The L3 baseline of `0x00` is:

- WRONG for `MEM_addr0` (true byte 0xFC)
- WRONG for `MEM_addr1` (true byte 0xff) — fixed by this op
- CORRECT for `MEM_addr2` (true byte 0x00) — this op over-cancels here
- CORRECT for `MEM_addr3` (true byte 0x00) — this op over-cancels here

This op uniformly cancels the L3 baseline at all 4 addr-byte positions
(matching the L3 rule shape), so for SP=0xFFFC it leaves `MEM_addr2` /
`MEM_addr3` without the +0.940 push toward 0. Some residual write at
`OUTPUT_LO[1]`/`OUTPUT_HI[1]` (value 0x11) then wins the argmax instead.

The candidate source of the 0x11 residue at `MEM_addr2` is unclear from
the attribution diagnostic — `attribute_1096_failure.py` reports
"_No op declares `produces` over the suspect dims_" for the new
divergence. The L14 mem_generation head ALREADY contains a `-1.0`
cancel at `OUTPUT_LO[0]`/`OUTPUT_HI[0]` (l14_ops.py:528-529, 622-623)
for the same purpose, so stacking this op's `-1.0` cancel on top of
the existing L14 attn cancel doubles the negative push at
`OUTPUT_LO[0]`/`OUTPUT_HI[0]` exactly where the byte's true value
needs to be 0x00.

## Proposed follow-up

Two paths forward, in order of risk:

1. **Narrow the gate to `MEM_addr1` only** (BYTE_INDEX_0). This keeps
   the fix for the originally-failing slot without touching `MEM_addr2`/
   `MEM_addr3` where the L3 baseline of 0x00 is correct for 16-bit
   addresses. Keep the marker (`addr_b0`) cancel because SP can be in
   any range. Smallest blast radius.

2. **Reuse the L14 mem_generation head's existing `-1.0` cancel** and
   instead investigate why `MEM_addr1` previously emitted `0x00` even
   though the head's `+1.0` V→O sum should have written `0xff` to
   `OUTPUT_LO[15]`/`OUTPUT_HI[15]`. The original failure may be a
   head-firing strength issue at the JSR step rather than an L3
   baseline competing too strongly. Tracking down the head firing on
   JSR step 0 with SP=0xFFFC may unblock the test without an extra
   cancel.

Both follow-ups are out of scope for this commit (which honours the
brief's "ONE compile + ONE smoke" budget cap).

## Files changed

- `c4_release/neural_vm/setup_helpers_l14.py` — add
  `_set_layer14_jsr_mem_default_suppress` helper (8 units, mirrors L3
  rule shape with the new gate).
- `c4_release/neural_vm/setup_helpers.py` — re-export the helper.
- `c4_release/neural_vm/vm_step.py` — re-export the helper.
- `c4_release/neural_vm/unified_compiler/ops/l14_ops.py`:
  - Add `make_layer14_jsr_mem_default_suppress_op` factory (phase=14.46).
  - Insert `layer14_jsr_mem_default_suppress: (None, 8)` into the L14
    cleanup-chain layout between `mem_addr_src_default_suppress` (14.45)
    and `addr_key_neural_decode` (14.5).
  - Shift `jsr_ax_bytes_zero` / `lc_ax_bytes_zero` /
    `alu_nocarry_ax_bytes_zero` claim indices by +8.
  - Update chain-tail `ffn_units_used` annotation 1883 → 1891.
- `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py` — register
  the new op between `mem_addr_src_default_suppress` (14.45) and
  `addr_key_neural_decode` (14.5).
