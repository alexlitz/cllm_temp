# Override Removal Status — 2026-06-06

Live status matrix for the runner-override removal campaign. Tracks
every override in `c4_release/neural_vm/batched_pure_neural.py`, the
tests it currently keeps green under the present weights, and the
model-side fix that would let it be deleted.

Base: `main` HEAD `f4f9103d` ("fix(var-cluster): L3 mem_byte_0_default
counter-write at L14").

Parent plan: [`RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md`](RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md).
Prior attempt findings:
- [`REMOVAL_1_IMM_OVERRIDE_2026_06_05.md`](REMOVAL_1_IMM_OVERRIDE_2026_06_05.md)
- [`REMOVAL_2_FINDINGS_2026_06_05.md`](REMOVAL_2_FINDINGS_2026_06_05.md)
- [`REMOVAL_3_FINDINGS_2026_06_05.md`](REMOVAL_3_FINDINGS_2026_06_05.md)
- [`REMOVAL_3_DEEP_DIVE_2026_06_05.md`](REMOVAL_3_DEEP_DIVE_2026_06_05.md)
- [`REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md`](REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md)

## TL;DR

- Smoke baseline on `main` HEAD: **45/51** passing (`tests/test_smoke.py`,
  1 deselected `test_compile_and_run` / `slow` marker, 51 active).
- Smoke with the three tool-calling overrides disabled
  (IMM-AX + collapsed-step ALU synth + non-collapsed binary-ALU synth):
  **28/51** — drops 17 tests.
- BZ/BNZ runner-side PC carry-forward is also an override (technically),
  but it patches the L4 FFN redirect not a Python tool call. Disabling
  it alone costs 2 tests (`test_bz_branch`, `test_bnz_branch`).
- The other override-class call sites (LI/LC load, GETCHAR/PUTCHAR
  shims, PRTF/OPEN/CLOS/READ shims, MEM-section addr substitution) are
  runner-state infrastructure, not bug-mask overrides — they don't
  enter the "remove" surface.

## Override enumeration

All file:line references are against `main` HEAD `f4f9103d`,
`c4_release/neural_vm/batched_pure_neural.py` (2339 lines total).

The `_dispatch_pure_neural` method (line 1953) is where every per-op
override fires. The matrix below lists each conditional block by the
opcode it gates on, in source order.

### 1. IMM AX literal override

- **File:line**: `batched_pure_neural.py:1985-2011`
- **Commit**: `b5cf7099` "fix(xor-basic): override AX from bytecode imm in batched IMM dispatch"
- **Status**: ACTIVE — runner-side computation of `imm_val` from
  `s.bytecode[exec_idx] >> 8`, then `_override_register_in_last_step(s.context, Token.REG_AX, imm_val)`.
- **Tool-call?**: YES. Reads the bytecode operand and overwrites the
  emitted REG_AX bytes. Pure tool call.
- **Tests blocked** (lost when disabled vs baseline, individually): 3 —
  `test_xor_basic`, `test_add_16bit`, `test_add_carry_cascade`.
- **Cluster**: IMM/BITWISE/32-bit.
- **Root cause**: L10 `tail_lea_local_ax_marker_byte0_e8`
  (`l10_ops.py:6561-6591`) and L16 `l16_psh_mem_addr0_e0_from_sp_no_addr_src`
  (`l16_ops.py:768`) rules fire on IMM dispatch rows where the
  immediate has the high bit set (IMM 0xC8 = 200, IMM 0xFF, etc.),
  writing 0xE8 / 0xE0 into AX byte 0. The leak discriminator is
  `FETCH_LO+8` (low nibble == 8) and "OP_LEA leaking into MARK_AX on
  IMM rows" — per `REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` and
  `REMOVAL_3_FINDINGS_2026_06_05.md`.
- **Model-side fix that would remove this**:
  - Block OP_LEA from reaching MARK_AX during IMM steps via L6/L7
    routing change, OR add an `OP_LEA`-distinguishing positive evidence
    dim to the L10 rule (e.g. `BP_PRESENT`) that's absent on real IMM
    rows. The `("OP_IMM", -1e9)` blocker pattern from `EDGE_POW2_OP_IMM_LEAK.md`
    was tried and **disconfirmed** — see Removal-1 doc; the rule's
    load-bearing positive is OP_LEA leakage, not OP_IMM gating failure.
  - Partial structural fix (commit `0b957e03`) added a `MEM_ADDR_SRC`
    predicate to `tail_lea_local_ax_marker_byte0_e8`; that helped the
    `IMM 0xFF` family but not the `IMM 0xC8` family because `FETCH_LO+8`
    pushes 0xC8 score over threshold.
- **Confidence the override is needed today**: HIGH.

### 2. Collapsed-step binary-ALU synth (`_compute_alu_legacy(IMM-skipped-op)`)

- **File:line**: `batched_pure_neural.py:2116-2156`
- **Commit**: `f3342968` "fix(highbit-imm): synth skipped binary ALU in batched IMM dispatch"
- **Status**: ACTIVE — when an IMM step's emitted REG_PC has advanced
  by **two** instructions instead of one and the skipped opcode is in
  `_BINARY_POP_OPS` (MUL/DIV/MOD/SHL/SHR), the runner runs
  `self._serial._compute_alu_legacy(skipped_op, stack_val, ax_after_imm)`
  and overwrites REG_AX in the just-emitted step.
- **Tool-call?**: YES. Calls the legacy Python ALU on bytecode-derived
  operands.
- **Tests blocked** (lost when disabled alone vs baseline): 12 —
  `test_sub_basic`, `test_div_basic`, `test_mod_basic`,
  `test_xor_basic`, `test_eq_true`, `test_gt_true`, `test_ge_true`,
  `test_shl`, `test_shr`, `test_add_16bit`, `test_add_carry_cascade`,
  `test_mul_overflow`. (Note: some of these overlap with override 1;
  per-override individual disable measures double-count, see
  "Sum impact" below.)
- **Cluster**: ALU (binary-pop), CMP-true, SHIFT, 32-bit cascade.
- **Root cause**: For `IMM N; PSH; IMM M; <binop>; EXIT` the IMM-M
  step's REG_PC bytes encode `PC = idx of EXIT` (advance +2 instead
  of +1) and REG_AX bytes are zero. The `<binop>` step is **never
  emitted by the model** at all. Per `REMOVAL_2_FINDINGS_2026_06_05.md`
  byte-emit probe.
- **Model-side fix that would remove this**:
  - Suppress the +2 NEXT_PC lookahead inserted by L9/L10/L14 rules
    when the next-bytecode opcode is in `_BINARY_POP_OPS`. The IMM
    step's PC must advance by exactly +1 instruction so the binop
    step actually runs.
  - OR split the L7 operand_gather attention into "IMM-only" vs
    "binop-from-fused-IMM" variants and have the model emit a real
    binop AX inside the fused step (multi-layer refactor).
- **Confidence the override is needed today**: HIGH. The brief
  estimates "5-10 sessions" for a proper fix.

### 3. Non-collapsed binary-ALU synth (`_compute_alu_legacy(binop)`)

- **File:line**: `batched_pure_neural.py:2158-2192`
- **Commits**: `69f77682` "fix(cmp-false): synth CMP AX in non-collapsed binary-pop step"
  (initial CMP version) plus `ebb3f09a` "fix(cascade-32bit): extend
  non-collapsed binop recovery to 32-bit ALU ops" (32-bit cascade
  extension).
- **Status**: ACTIVE — when `exec_op` itself is in
  `_NON_COLLAPSED_RECOVERY_OPS` (EQ/NE/LT/GT/LE/GE + ADD/SUB/OR/XOR/AND),
  the runner runs `self._serial._compute_alu_legacy(exec_op, stack_val, ax_rhs)`
  and overwrites REG_AX in the just-emitted step.
- **Tool-call?**: YES.
- **Tests blocked** (lost when disabled alone vs baseline): 5 —
  `test_eq_false`, `test_ne_true`, `test_sub_16bit`, `test_or_16bit`,
  `test_xor_16bit`.
- **Cluster**: CMP-false/CMP-true, 32-bit cascade.
- **Root cause** (CMP cluster, per `REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md`
  and `CMP_POLARITY_INVESTIGATION_2026_06_03.md`):
  - Value-dependent failure mode: `EQ(0,0)`, `EQ(16,16)`, `EQ(128,128)`
    pass; `EQ(17,17)`, `EQ(42,42)` fail. PASS pattern is "either hi
    or lo nibble is 0"; FAIL pattern is "both nibbles nonzero".
  - The brief's L7 head 0 K-side gate hypothesis was **refuted**:
    L7 K-side is purely positional (`STACK0_BYTE0` is a positional
    flag), but the failure is value-dependent. The divergence is in
    L1-L6 value-dependent flag clobbering OR L8/L9 ALU. Per the EQ(17,17)
    byte-1 corruption memory note (`project_eq_byte1_l6_divergence.md`)
    the candidate ops are `layer6_attn` / `layer6_routing_ffn` /
    `layer6_relay_heads`.
- **Root cause** (32-bit cascade subset, per `REMOVAL_3_FINDINGS_2026_06_05.md`):
  - Same L10 / L16 tail leak as override 1; the IMM operand poisons
    `s.last_pushed_value`, then the multi-byte ADD/SUB/OR/XOR
    writeback chain (OUTPUT_LO/HI byte-1..byte-3 lanes) drops or
    inverts the high bytes.
- **Model-side fix that would remove this**:
  - CMP cluster: hooked-forward capture on `EQ(17, 17)` at L7 input to
    discriminate L1-L6 CLEAN_EMBED clobbering vs L8/L9 ALU bug; then
    fix the discriminated layer.
  - 32-bit cascade: fix the OUTPUT_LO/HI multi-byte writeback chain
    in the layer that emits the high bytes (likely L13/L14).
- **Confidence the override is needed today**: HIGH.

### 4. BZ/BNZ taken-path PC carry-forward

- **File:line**: `batched_pure_neural.py:2077-2104`
- **Commit**: `44709e13` "fix(run_vm): hoist BZ/BNZ override above pure_neural early-return"
  (the run_vm version; batched mirrors it inline).
- **Status**: ACTIVE — computes `target_pc` from bytecode imm field
  and overwrites REG_PC + `s.last_pc` in the just-emitted step.
- **Tool-call?**: PARTIAL. Reads bytecode to compute target. Documented
  in `docs/ABSDIFF_BZ_REDIRECT_BUG.md` as a fix for a real L4 FFN BZ
  redirect that doesn't propagate the taken-path PC.
- **Tests blocked** (lost when disabled alone vs baseline): 2 —
  `test_bz_branch`, `test_bnz_branch`.
- **Cluster**: CONTROL FLOW (branch).
- **Model-side fix that would remove this**: per
  `ABSDIFF_BZ_REDIRECT_BUG.md`, repair the L4 FFN BZ redirect block
  to propagate `target_pc` into the same step's PC bytes when the
  branch is taken.
- **Confidence the override is needed today**: HIGH for smoke pass.
  Lower priority for thesis-purity (it patches a known FFN bug, not
  a runner tool call).
- **Plan note**: parent plan section "Removal 6" lists this as "Keep"
  for now since the model fix is bounded but not yet attempted.

### 5. `_decode_bail_exit_code` (cap-hit / no-HALT recovery)

- **File:line**: `batched_pure_neural.py:1122` (run-level), `:1944`
  (per-step), `:2259-2291` (definition).
- **Commit**: `b6632137` "fix(batched): sign-extension divergence on IMM operands >= 0x80"
- **Status**: ACTIVE — when an element hits its expected-steps cap
  without emitting HALT, scan forward from the last STEP_END for the
  next REG_AX marker and return that as the exit code. Falls back to
  `last_step_end_ax`, then to a context-wide `_decode_exit_code`.
- **Tool-call?**: NO. Decoder only; reads tokens already emitted by
  the model. Does not compute or substitute values from bytecode.
  Listed in the parent plan as "Keep" / "mostly legit".
- **Tests blocked**: not measurable separately (the bail path triggers
  only on cap-hit, which itself depends on other overrides being
  active). Likely 0 tests when other overrides are active.
- **Cluster**: bail-path robustness (not opcode-specific).
- **Model-side fix that would remove this**: model emits HALT
  cleanly at end of every program execution, eliminating cap-hit
  bail entirely.
- **Plan note**: keep.

### 6. MEM-section addr substitution

- **File:line**: `batched_pure_neural.py:2061-2075`
- **Status**: ACTIVE — for MEM-store ops, if the model emitted addr=0
  inside the MEM section but `stack0_shadow` has a non-zero address,
  rewrite addr bytes from `stack0_shadow` and call
  `_override_mem_section_in_last_step`.
- **Tool-call?**: PARTIAL. Reads a shadow VM-side value to repair
  emitted bytes. But the substitution is bounded: only fires when
  the model emitted addr=0 and the runner has a known good source.
- **Tests blocked**: not measured (no env gate added this session;
  the memory cluster failures `test_si_li_*` are persistent baseline
  failures with this override active, so disabling it would likely
  not change them).
- **Cluster**: MEM (SI/SC).
- **Model-side fix that would remove this**: L10 PSH addr0_e0 bug
  (memory note `project_l10_psh_addr_ent_bug.md`) and related L15
  memory_lookup bakes — the model should emit the correct addr bytes
  inside the MEM section directly.
- **Plan note**: not in parent plan; treated as cache/window metadata,
  not opcode synth.

### 7. LI / LC memory-load (`_override_register_in_last_step(REG_AX, loaded)`)

- **File:line**: `batched_pure_neural.py:2029-2037`
- **Status**: ACTIVE — for LI/LC, look up `s.mem_history[prev_ax]`,
  decode 1 or 4 bytes, override REG_AX with the loaded value.
- **Tool-call?**: YES, but it reads runner-tracked memory (which
  itself comes from neural-emitted MEM sections), not the bytecode.
- **Tests blocked**: 5 of the 6 `test_si_li_*` tests fail at baseline
  WITH the override active, so disabling it likely doesn't change
  the failure count; the actual blocker is upstream in SI/SC emit
  (`project_l10_psh_addr_ent_bug.md`).
- **Cluster**: MEM (LI/LC).
- **Model-side fix that would remove this**: the model must emit
  the loaded value as REG_AX bytes 0-3 inside the LI/LC step.
  Today's failure is upstream (SI's emit lacks the correct address
  / value to be looked up), so this override is a downstream rescue.
- **Plan note**: keep until SI/SC clean.

### 8. GETCHAR stdin injection

- **File:line**: `batched_pure_neural.py:2017-2027`
- **Status**: ACTIVE — pop one byte from `s.stdin_buffer` and override
  REG_AX with it.
- **Tool-call?**: NECESSARY I/O SHIM. GETCHAR is a host-boundary
  opcode; the model can't read stdin.
- **Tests blocked**: 0 (no smoke test exercises GETCHAR today).
- **Cluster**: I/O.
- **Model-side fix**: NA — this is a permanent runner responsibility.

### 9. PRTF / OPEN / CLOS / READ shims

- **File:line**: `batched_pure_neural.py:2039-2059`
- **Status**: ACTIVE — borrows serial-runner state, calls
  `_serial._neural_prtf_emit` / `_neural_open_emit` / `_neural_clos_emit` /
  `_neural_read_emit`, then restores.
- **Tool-call?**: NECESSARY I/O SHIMS (host syscalls).
- **Tests blocked**: 0 in smoke (these are exercised in 1098 / 1096
  corpus tests, not smoke).
- **Cluster**: I/O.
- **Model-side fix**: NA — permanent runner responsibility.

### 10. EXIT early-halt + neural-authoritative early exit

- **File:line**: `batched_pure_neural.py:2106-2114` (EXIT halt),
  `:2194-2205` (next-op-is-EXIT early halt).
- **Status**: ACTIVE — when `exec_op == EXIT` or the next instruction
  is EXIT, mark `s.halted = True` and set exit code.
- **Tool-call?**: NO. Control-flow optimization; reads the emitted
  AX as-is.
- **Tests blocked**: 0 by itself (load-bearing for many tests because
  it terminates the bail loop).
- **Cluster**: termination.
- **Model-side fix**: NA — purely a runner-side stop condition.

## Per-override smoke impact (measured this session)

All runs against `main` HEAD `f4f9103d`, `tests/test_smoke.py`
(51 selected, 1 deselected). One env-var per override block was added
behind a sentinel to measure; the file was restored before the
findings doc commit.

| Override | Smoke (passing) | Delta vs baseline | Tests lost |
|---|---|---|---|
| Baseline (all overrides ON) | **45/51** | — | (1) lea_basic, 5 si_li_*/sc_lc_* MEM |
| Only `b5cf7099` IMM-AX off | 42/51 | −3 | xor_basic, add_16bit, add_carry_cascade |
| Only `f3342968` collapsed-ALU off | 33/51 | −12 | sub/div/mod_basic, xor_basic, eq_true/gt_true/ge_true, shl, shr, add_16bit, add_carry_cascade, mul_overflow |
| Only `69f77682+ebb3f09a` non-collapsed off | 40/51 | −5 | eq_false, ne_true, sub_16bit, or_16bit, xor_16bit |
| Only `44709e13` BZ/BNZ off | 43/51 | −2 | bz_branch, bnz_branch |
| All three tool-call overrides off (IMM + collapsed + non-collapsed) | **28/51** | **−17** | (sum of all three columns, deduplicated) |

Note on overlap: `test_xor_basic`, `test_add_16bit`,
`test_add_carry_cascade` are listed under both override 1 (IMM-AX) and
override 2 (collapsed-ALU). That's because they fail under either
disable in isolation — both overrides are necessary independently
(IMM-AX cleans the operand; collapsed-ALU synthesizes the binop
result). The combined-disable measurement counts each test once.

## Cluster impact summary (all three tool-call overrides off)

| Cluster | Total smoke tests | Failures w/ overrides ON | Failures w/ tool-call overrides OFF | Delta |
|---|---|---|---|---|
| BASIC (ADD/SUB/MUL/DIV/MOD basic) | 6 | 0 | 3 (sub/div/mod) | +3 |
| BITWISE (OR/AND/XOR basic) | 3 | 0 | 1 (xor) | +1 |
| COMPARISON (EQ/NE/LT/GT/LE/GE) | 7 | 0 | 5 (eq_true/eq_false/ne_true/gt_true/ge_true) | +5 |
| ADDRESS (LEA/ADJ) | 2 | 1 (lea_basic) | 1 | 0 |
| MEMORY (SI/SC/LI/LC) | 6 | 5 (si_li_* + sc_lc_*) | 5 | 0 |
| SHIFT (SHL/SHR) | 2 | 0 | 2 | +2 |
| 32-BIT (add/sub/or/and/xor cascade + mul_overflow + shl_8bit / shr_8bit) | 10 | 0 | 6 (add_16bit, add_carry_cascade, sub_16bit, or_16bit, xor_16bit, mul_overflow) | +6 |
| CONTROL FLOW (JMP/BZ/BNZ) | 3 | 0 | 0 (overrides 4 still on here) | 0 |
| FUNCTION CALL | 1 | 0 | 0 | 0 |
| HANDLER STATUS | 3 | 0 | 0 | 0 |
| Other (integration) | 1 | 0 | 0 | 0 |

The single biggest model-side wins by failure count:

1. **Override 2** (`f3342968` collapsed-step ALU): rescues **12 tests**.
   Root cause is the IMM step emitting `PC = exec_idx + 2` and skipping
   the binop emit entirely. Fix surface: L9/L10/L14 NEXT_PC rules that
   add the spurious +INSTR_WIDTH when the next bytecode opcode is in
   `_BINARY_POP_OPS`.
2. **Override 3** (`69f77682 + ebb3f09a`): rescues **5 tests**. Split
   between CMP cluster (value-dependent both-nibbles-nonzero bug, fix
   surface L6 attn / routing_ffn / relay_heads per memory note
   `project_eq_byte1_l6_divergence.md`) and 32-bit cascade (OUTPUT_LO/HI
   multi-byte writeback chain bug, fix surface L13/L14).
3. **Override 1** (`b5cf7099`): rescues **3 tests**. Root cause is L10
   `tail_lea_local_ax_marker_byte0_e8` + L16
   `l16_psh_mem_addr0_e0_from_sp_no_addr_src` firing on IMM rows with
   `FETCH_LO+8` set, writing 0xE8/0xE0 into AX byte 0. Fix surface: L7
   head 5 K-side gate (block OP_LEA broadcast on IMM rows) OR add a
   load-bearing positive on the L10 rule that real LEA but not real
   IMM has.
4. **Override 4** (`44709e13` BZ/BNZ PC carry-forward): rescues **2
   tests**. Fix surface: L4 FFN BZ redirect per
   `docs/ABSDIFF_BZ_REDIRECT_BUG.md`.

## Status legend

- **ACTIVE**: override present and load-bearing for the listed tests.
- **PARTIAL**: override present, but a partial structural fix (e.g.
  Removal-1's `MEM_ADDR_SRC` predicate) is in place too. Override
  still required for the residual cases.
- **REMOVED**: override deleted (none in this category as of 2026-06-06).

## Priority order for next-session fixes

Ranked by `(tests rescued / estimated fix complexity)`:

1. **BZ/BNZ override** (override 4) — 2 tests, fix is bounded per
   `ABSDIFF_BZ_REDIRECT_BUG.md`. Smallest scope; biggest
   tests-per-session ratio.
2. **CMP cluster of override 3** — 2-4 tests (eq_false, ne_true plus
   possible eq_true/gt_true/ge_true that overlap with override 2).
   Per `REMOVAL_4_L7_KSIDE_FINDINGS_2026_06_05.md`, run the EQ(17,17)
   hooked-forward at L7 input first; do NOT touch L7 K-side weights
   blindly (single-rule fix audit: 0/5 historical agents netted positive
   with speculative changes — `feedback_single_rule_fixes_are_zero_sum.md`).
3. **32-bit cascade subset of override 3** — 3 tests (sub_16bit,
   or_16bit, xor_16bit). Fix surface is multi-byte OUTPUT_LO/HI
   writeback chain in L13/L14.
4. **IMM override** (override 1) — 3 tests. Fix surface refuted in
   Removal-1 attempt; needs an L7 head 5 K-side investigation, not a
   blocker-strength tweak. Multi-session.
5. **Collapsed-step ALU** (override 2) — 12 tests but the highest
   risk. Per `REMOVAL_2_FINDINGS_2026_06_05.md`, the bug is the
   +INSTR_WIDTH lookahead that fuses IMM+binop into one step. Either
   a single L9/L10/L14 dim/rule has to be located and its OP_<binop>
   activation suppressed (bounded if found), OR a multi-layer split
   into "IMM-only" + "binop-from-fused" attention variants (multi-
   session). Per the brief estimate: 5-10 sessions.

## Acceptance criteria for "campaign complete"

(From `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` §"Acceptance criteria"):

- [ ] `_compute_alu_legacy` is no longer called from `_dispatch_pure_neural`.
- [ ] No `_override_register_in_last_step(... REG_AX ...)` calls for
  opcode-specific computation (LI/LC and GETCHAR are I/O shims and
  not in scope).
- [ ] `_BINARY_POP_OPS`, `_NON_COLLAPSED_RECOVERY_OPS` references are
  deleted from `batched_pure_neural.py`.
- [ ] All AX values flow naturally through `model.forward(x)`.
- [ ] Smoke baseline holds at ≥ 45/51 (or higher) WITHOUT the
  tool-call overrides.

Current progress: **0/5**. All four removals attempted to date
(Removal-1, Removal-2, Removal-3, Removal-4) ended in `findings-only`
doc commits; no override has been removed.

## Cross-references (current weights)

- `c4_release/neural_vm/batched_pure_neural.py:1985-2011` —
  override 1 (IMM AX).
- `c4_release/neural_vm/batched_pure_neural.py:2077-2104` —
  override 4 (BZ/BNZ PC carry-forward).
- `c4_release/neural_vm/batched_pure_neural.py:2116-2156` —
  override 2 (collapsed-step ALU synth).
- `c4_release/neural_vm/batched_pure_neural.py:2158-2192` —
  override 3 (non-collapsed binary-ALU synth).
- `c4_release/neural_vm/batched_pure_neural.py:2259-2291` —
  override 5 (`_decode_bail_exit_code`).
- `c4_release/neural_vm/run_vm.py:2310-2318` — sibling LEA override
  in the serial runner (referenced by override 1's comment).
- `c4_release/neural_vm/run_vm.py:2273-2275` — sibling
  `_compute_alu_legacy` call in handler-mode serial runner.
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6561-6591` —
  `tail_lea_local_ax_marker_byte0_e8`, the rule behind override 1.
- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:768` —
  `l16_psh_mem_addr0_e0_from_sp_no_addr_src`, the sibling rule.
- `c4_release/docs/CMP_POLARITY_INVESTIGATION_2026_06_03.md` —
  upstream brief for override 3 CMP cluster.
- `c4_release/docs/ABSDIFF_BZ_REDIRECT_BUG.md` — upstream brief for
  override 4 model fix.
- `c4_release/docs/OP_LEA_LEAK_INVESTIGATION_2026_06_05.md` — upstream
  brief for override 1 model fix.

## Measurement methodology

For each per-override impact row, an env-var gate
`C4_DISABLE_<NAME>_OVERRIDE` was added to the override's `if` condition
(temporary, not committed); the test suite was run via:

```
cd c4_release && C4_DISABLE_<NAME>_OVERRIDE=1 \
    python -m pytest tests/test_smoke.py --tb=no -q
```

After all measurements the file was restored via `git checkout
neural_vm/batched_pure_neural.py` — no override code changes are in
this commit. Only this doc.

The baseline "45/51" reading is one passing test below the
historically-quoted "46/52": the discrepancy is the deselected
`test_compile_and_run` (slow marker, requires `cc` compile) plus
`test_lea_basic` which is a persistent baseline failure independent of
the overrides studied here. The per-override-disabled rows preserve
the same 51-selected denominator for direct comparison.
