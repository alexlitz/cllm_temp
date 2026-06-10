# Runner override hard-removal + upstream fix queue (2026-06-09)

This doc records the smoke baseline AFTER hard-deleting every Python
"correction" from the pure_neural code path of both runners
(`AutoregressiveVMRunner` + `BatchedPureNeuralRunner`). The runners are
now PURE forward-pass wrappers: they observe model emission, do real
PRTF/OPEN/CLOS/READ I/O at the host boundary, and append PUTCHAR bytes
to the output stream. Nothing else.

See the `refactor(runners): force-remove all Python ALU/shadow
overrides` commit for the line-by-line removal log (-823 / +12 in the
two runner files).

## Smoke (CUDA_VISIBLE_DEVICES=1 tests/test_smoke.py)

| State                                        | Pass count |
|----------------------------------------------|-----------:|
| Pre-removal (with cheats)                    | 46 / 51    |
| Post-removal (this commit)                   | 29 / 51    |
| Regression                                   | 17         |

Pre-removal failures (`Memory` group, untouched by this work):

  * `TestSmokeMemory::test_si_li_roundtrip`
  * `TestSmokeMemory::test_sc_lc_roundtrip`
  * `TestSmokeMemory::test_si_li_multiple_stores`
  * `TestSmokeMemory::test_si_li_overwrite`
  * `TestSmokeMemory::test_si_li_16bit_value`

Newly-failing post removal (regression cluster):

  * **Cluster A — binary ALU pop (13 tests).** All ADD/SUB/DIV/MOD/
    OR/AND/XOR/SHL/SHR + 5 of the 32-bit cascade + MUL overflow.
    `test_add_basic` (`IMM 2; PSH; IMM 3; ADD; EXIT` → expect 5) fails
    — the most fundamental binary op. Root cause: the Python ALU
    rescue at `batched_pure_neural._compute_alu_legacy(...)` was
    masking the neural ALU defect at every binary-pop step.
  * **Cluster B — CMP (3 tests).** `eq_true`, `gt_true`, `ge_true`.
    Probed 2026-06-09 via `ece84b38`: L9 lo_eq rule does not amplify
    `CMP+2` at the EQ step's AX-marker row in Shape B; L10 cmp_combine
    default fires AX=0. The collapsed-step IMM+CMP recovery and the
    non-collapsed EQ/NE recovery were masking both Shape A and Shape B
    failures.
  * **Cluster C — integration (2 tests).** `test_simple_function`
    (JSR+LEV roundtrip with ENT/ADJ frame), `test_cmp_and_branch`
    (EQ(5,5) + BZ). Both depend on Cluster A or B.
  * **Cluster D — memory (1 test).** `test_si_li_16bit_value` flipped
    from a previously-tracked pre-existing failure into a different
    failure mode (likely the removed `_extract_mem_section` shadow-
    addr rescue at MEM-section addr==0 fallback to `stack0_shadow`).

## Why no upstream fix landed in this commit

Per memory note `feedback_single_rule_fixes_are_zero_sum.md`: 0 of 5
historical single-rule fix attempts have netted positive. The L9/L10
ALU surfaces driving Clusters A+B+C are cross-layer
(`L7 operand_gather head 0` → `L10 cmp_combine` /
`L10 bitwise_or/xor/and` / `L10 mul_lo`) with shared upstream
amplification at L9 (`lo_eq`, `STACK0 → ALU_LO/HI` relay). Touching a
single rule risks moving the failure rather than fixing it. The fix
queue below is ordered by attribution evidence + expected unlock
breadth so the next agent can pick the highest-impact entry without
re-probing.

## Prioritized upstream fix queue

Entries are ordered by `(estimated smoke recovery) / (declarative
surface size to touch)`. Each entry has a single anchor file +
verification command.

### F1 — L7 `operand_gather` head 0 STACK0 → ALU_LO/HI relay

* **Anchor:** `c4_release/neural_vm/unified_compiler/ops/l7_ops.py`
  → `_layer7_operand_gather_head_specs` head 0 (lines 240-262).
* **Symptom:** at the binary-pop step's AX marker, head 0 attends to
  the previous PSH-step's STACK0 byte 0, reads `CLEAN_EMBED_LO/HI`,
  and writes `ALU_LO/HI[a]` where `a` is the operand-A nibble. ece84b38
  probing showed `ALU_LO[0] = 1, ALU_LO[1..15] = 0` at the AX-marker
  row when operand A should be nonzero — i.e. the head reads operand A
  as 0 because it fails to attend to STACK0 with the right key.
* **Hypothesis:** the Q-side
  `AP(0, BD.MARK_AX, L), AP(33, BD.MARK_AX, L)` plus `OP_LEA/ADJ/ENT`
  blockers selects AX-marker rows correctly, but the K-side
  `AP(0, BD.STACK0_BYTE0, L)` matches the PRIOR step's STACK0 too
  weakly when the binary-pop step's emitted MEM section has not yet
  been flagged `MEM_STORE=1` — which is one of the runner overrides we
  just removed (`set_mem_store_positions`). The model currently relies
  on the runner to flag the PSH-step STACK0 marker post hoc.
* **Estimated unlock:** Cluster A (13 tests) + Cluster B partial
  (some CMP) — possibly +13 to +16 of the 17 regressions.
* **Verify (declarative):** byte-identity gate via
  `compare_symbolic_to_lowered_attn(...)` for the modified head, then
  `CUDA_VISIBLE_DEVICES=1 pytest c4_release/tests/test_smoke.py
  -k 'TestSmokeBasic or TestSmokeBitwise or TestSmoke32Bit' --tb=no -q`.

### F2 — L9 `lo_eq` amplification at MARK_AX in Shape B

* **Anchor:** `c4_release/neural_vm/unified_compiler/ops/l9_ops.py`
  → search for `lo_eq` rule emission. The probe (ece84b38) noted "the
  upstream L9 lo_eq rule does not amplify CMP+2 at the EQ step's
  AX-marker row in Shape B." Shape B = the non-collapsed
  `IMM 42; PSH; IMM 42; EQ; EXIT` decode where `EQ` gets its own
  emitted step.
* **Symptom:** `test_eq_true` AX = 0 (broken) instead of 1.
  `test_gt_true` / `test_ge_true` follow the same path (override3
  score below the +2.5 threshold because CMP+2 amplification is
  missing).
* **Estimated unlock:** Cluster B (3 tests).
* **Verify:** `decl_verifier.py` on the affected rule, then
  `CUDA_VISIBLE_DEVICES=1 pytest c4_release/tests/test_smoke.py::
  TestSmokeComparison --tb=no -q`.

### F3 — L10 multi-byte writeback chain (`OUTPUT_LO/HI` byte-1..3)

* **Anchor:** `c4_release/neural_vm/unified_compiler/ops/l10_ops.py`
  → ADD/SUB carry propagation. The 32-bit cascade tests
  (`test_add_16bit` / `test_sub_16bit` / `test_or_16bit` /
  `test_xor_16bit` / `test_add_carry_cascade`) hit the broken
  multi-byte writeback that the non-collapsed Python ALU recovery was
  fixing. From the deleted `_NON_COLLAPSED_RECOVERY_OPS` comment: "the
  multi-byte ADD/SUB/OR/XOR writeback chain (OUTPUT_LO/HI byte-1..3
  lanes) drops or inverts the high bytes — wrong whenever the
  architectural result differs from the broken default."
* **Estimated unlock:** 5 32-bit tests, partial overlap with F1.
* **Verify:** byte-identity gate then
  `CUDA_VISIBLE_DEVICES=1 pytest c4_release/tests/test_smoke.py::
  TestSmoke32Bit --tb=no -q`.

### F4 — L15/L16 memory_lookup at the current-step MEM marker

* **Anchor:** `c4_release/neural_vm/unified_compiler/ops/l15_ops.py`
  → memory_lookup K-side `MEM_STORE` predicate. The deleted
  `set_mem_store_positions` runner override (commit aaf399c9 found
  it) was injecting `MEM_STORE=1` at MEM markers BEFORE the model
  predicted `MEM_addr0`, masking the L15 lookup defect at the
  current-step marker (PSH/JSR/ENT push targets collapsing from
  `0xfff0` to `0xffe0`).
* **Estimated unlock:** part of Cluster C (`test_simple_function`)
  + may help Cluster D memory tests.
* **Verify:** `CUDA_VISIBLE_DEVICES=1 pytest
  c4_release/tests/test_smoke.py::TestSmokeFunctionCall::
  test_simple_function c4_release/tests/test_smoke.py::TestSmokeMemory
  --tb=no -q`.

### F5 — IMM AX byte-0 for `imm ∈ [0xE0, 0xFF]`

* **Anchor:** unclear (see deleted comment in
  `batched_pure_neural.py` removal-1 retry). 2026-06-06 probe could
  not localize: after L7 head 5 K-side OP_IMM blocker (commit
  ff4edb61) and L10 threshold refinement to 7 (commit 983b70c9), the
  L34 residual probe shows `tail_lea` rule scores sum=1.0 (MARK_AX
  only) at every IMM AX marker — well below threshold. Yet removing
  the IMM AX override STILL regressed `xor_basic`, `add_16bit`,
  `add_carry_cascade`. Conclusion (still open): "A separate model-
  side surface is emitting wrong AX bytes for IMM in [0xE0, 0xFF]."
* **Estimated unlock:** subset of Cluster A (the high-byte IMM
  operand tests).
* **Verify:** re-run `tools/removal_1_probe_and_smoke.py` (the L34
  probe + smoke combination from the 2026-06-06 retry).

## Confirmed dead code removed

The following can stay removed even after F1–F5 land (they are pure
runner cheats with no architectural justification):

  * `BatchedPureNeuralRunner._compute_alu_legacy` calls (both
    collapsed and non-collapsed).
  * `_BINARY_POP_OPS` import + recovery sets in
    `batched_pure_neural.py`.
  * `_alu_recovery_disabled_for` / `C4_DISABLE_BATCHED_ALU_RECOVERY[_OPS]`
    toggle framework.
  * `_ElementState.last_pushed_value` + `stack0_shadow` shadow fields.
  * `_ElementState.mem_store_positions` / `mem_addr_src_positions` +
    `_current_step_store_mem_marker` / `_add_current_step_marker` /
    `_addr_src_positions` helpers + `set_mem_store_positions` /
    `set_mem_addr_src_positions` embedding calls.
  * `AutoregressiveVMRunner._inject_getchar` def + dispatch call.
  * `_handle_skipped_io_op` + `_handle_pure_neural_stall` +
    `_tail_has_io_op` + `_IO_OPS_*` constants + `_stall_count_at_idx`.

The PRTF/OPEN/CLOS/READ `_neural_*_emit` shims stay — they cross the
VM/host boundary and have no neural counterpart by design.

## Acceptance gate (next-agent checklist)

  1. Run smoke baseline pre-fix: 29/51 expected (CI matches).
  2. Pick the single highest-priority unattempted entry from F1–F5.
  3. Apply the fix via declarative DSL only (no imperative
     `block.ffn.W_*[...] = X` writes). Lint with
     `tools/lint_raw_ffn_rule.py`.
  4. Byte-identity gate via the relevant
     `compare_symbolic_to_lowered_*` helper before commit.
  5. Re-run smoke; commit if pass count increased AND no previously-
     passing test regressed. If both conditions fail, REVERT — do not
     stack a corrective op (`feedback_single_rule_fixes_are_zero_sum`).
