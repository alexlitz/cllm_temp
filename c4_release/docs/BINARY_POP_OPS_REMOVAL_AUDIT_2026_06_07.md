# `_BINARY_POP_OPS` removal audit (2026-06-07)

Wave 1 Cluster E1 audit per the C6c89d07 incident. **Audit-only — no
code removed.** Documents exactly what the runner-side
`_BINARY_POP_OPS` block intercepts so Wave 2 can remove it cleanly
after the joint L28+L34 fix
(`docs/COLLAPSED_STEP_JOINT_FIX_PLAN_2026_06_07.md`) lands.

Base: `9570e232` (`docs: 2-wave parallel-agent plan to 51/51 + clean
1096`).

## 1. Where the block lives

Two distinct sites both gated by membership in `_BINARY_POP_OPS`:

| Site | File | Lines | Purpose |
|---|---|---|---|
| Set definition | `c4_release/neural_vm/run_vm.py` | 123-140 | Canonical 16-opcode set; imported by both runners |
| Serial runner SP+=8 + ALU | `c4_release/neural_vm/run_vm.py` | 2334-2354 | `_BINARY_POP_OPS` branch in serial `_apply_overrides` |
| Batched **collapsed-step** synth (`f3342968`) | `c4_release/neural_vm/batched_pure_neural.py` | 2149-2189 | The override Wave 2 wants gone |
| Batched **non-collapsed** recovery | `c4_release/neural_vm/batched_pure_neural.py` | 2191-2256 | Sibling override using local `_NON_COLLAPSED_RECOVERY_OPS` subset |

## 2. Opcodes covered

`_BINARY_POP_OPS` (16 opcodes,
`run_vm.py:123-140`):

```
ADD, SUB, MUL, DIV, MOD,
EQ,  NE,  LT,  GT, LE, GE,
OR,  XOR, AND,
SHL, SHR
```

Derived sets used at the override sites:

- `_NEURAL_32BIT_OPS = {ADD, SUB, OR, XOR, AND}` (multi-byte carry, the
  *non-collapsed* recovery surface for the 32-bit cascade tests).
- `_RUNNER_ALU_OPS = _BINARY_POP_OPS - _NEURAL_32BIT_OPS - {SI, SC}` —
  used only by the serial-runner fallback ALU.
- `_NON_COLLAPSED_RECOVERY_OPS = (ADD, SUB, OR, XOR, AND, EQ, NE)`
  (`batched_pure_neural.py:2241-2244`) — non-collapsed sibling
  override; **not** the Wave 2 target but flagged here because it also
  reads `_BINARY_POP_OPS` semantics.

BZ/BNZ are no longer routed through this set after `44709e13`; the
sibling control-flow override handles them.

## 3. Trigger condition (the `f3342968` synth)

`batched_pure_neural.py:2166-2189`. All four predicates must hold:

1. `s.last_pc is not None` — model has emitted at least one REG_PC.
2. `exec_op == Opcode.IMM` — the *executed* op this step was IMM
   (always the `M` IMM in `IMM N; PSH; IMM M; <binop>; EXIT`).
3. `s.last_pushed_value is not None` — runner shadow snapshot of the
   pre-PSH AX exists (set at `batched_pure_neural.py:2013-2019`).
4. `post_idx == skipped_idx + 1` where `post_idx = s.last_pc /
   INSTR_WIDTH` and `skipped_idx = exec_idx + 1` — i.e. the model's
   emitted PC has advanced **two** instructions in a single STEP_END,
   so exactly one bytecode slot is unaccounted for.
5. `s.bytecode[skipped_idx] & 0xFF in _BINARY_POP_OPS` — the skipped
   slot is a binary-pop opcode.

This is the *collapsed-step* signature: model emits Block A (IMM) and
Block B (post-binop PC) under a single STEP_END, never emitting the
intermediate binop step's tokens at all.

## 4. What value gets written

```python
imm_val      = sign_extend((bytecode[exec_idx] >> 8) & 0xFFFFFF)
ax_after_imm = imm_val & 0xFFFFFFFF
stack_val    = s.last_pushed_value  # clean pre-PSH AX shadow
alu_result   = _compute_alu_legacy(skipped_op, stack_val, ax_after_imm)
s.last_ax    = alu_result
_override_register_in_last_step(s.context, Token.REG_AX, alu_result)
```

So the synth:
- Reads the IMM operand byte from bytecode (not from the noisy neural
  emit).
- Reads `last_pushed_value` (a runner shadow, the clean pre-PSH AX).
- Computes `legacy_alu(skipped_op, stack_val, ax_after_imm)`.
- Rewrites `REG_AX` in the last step in `s.context`.
- Updates `s.last_ax`.

It does **not** write a stale opcode byte or PC+1 byte. It overwrites
the model's broken `REG_AX=0` emit with the architecturally-correct
result.

## 5. Tests load-bearing on this synth

Per `docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` (Override surface
table, re-confirmed against current HEAD):

| Test | Opcode | Status today |
|---|---|---|
| `test_sub_basic` | SUB | passes via synth |
| `test_div_basic` | DIV | passes via synth |
| `test_mod_basic` | MOD | passes via synth |
| `test_eq_true` | EQ (Shape A) | passes via synth |
| `test_gt_true` | GT | passes via synth |
| `test_ge_true` | GE | passes via synth |
| `test_shl` | SHL | passes via synth |
| `test_shr` | SHR | passes via synth |
| `test_mul_overflow` | MUL | passes via synth |

9 unique tests. Three more (`test_xor_basic`, `test_add_16bit`,
`test_add_carry_cascade`) overlap the *non-collapsed* recovery
(`batched_pure_neural.py:2241-2256`) and are NOT covered by the
collapsed-step block — they need both surviving.

Shape-B EQ/NE (`test_eq_false`, `test_ne_true`, `test_cmp_and_branch`)
ride on the non-collapsed recovery's EQ/NE entries (re-added per
removal-4 follow-up; see comment at lines 2220-2240).

## 6. Conditions under which removal is safe

Removal is safe iff **both** model-side fixes in
`COLLAPSED_STEP_JOINT_FIX_PLAN_2026_06_07.md` land:

- **Edit A (L28)** — tighten
  `l16_lev_stack0_byte0_preserve_*` so the ×97 gate self-feedback
  cannot fire at the IMM step. Reverted attempts (`1ed6b5b6`,
  `cd53c076`, `b91e3cd2`) used insufficient OP_LEV weights; the plan
  uses `OP_LEV +5.0` with explicit `OP_IMM`/`OP_PSH -10.0` blockers
  and threshold 7.5.
- **Edit B (L34)** — `stack0_pop_loaded_output_rules`: drop the
  OUTPUT_LO/HI condition weight 0.1→0.05 and raise threshold
  10.5→12.0 so a +40 residual (post-A) no longer ignites the 256-rule
  family.
- **Edit C (L34, optional)** — re-key the 256-rule family onto
  `ALU_LO+lo`/`ALU_HI+hi` primary conditions (weight 1.0) with
  OUTPUT_* relegated to 0.001 confirmation. Matches the
  `stack0_store_top_e0_output_rules` pattern.

Per-step verification gates (mandatory before removal per the plan):

1. `compare_symbolic_to_lowered_ffn(ir, dim_positions, S=100.0)`
   byte-identical after each edit.
2. `decl_verifier.py verify_rule_strength` confirms new L34 threshold
   dominates only at the (lo,hi) ALU-band hit.
3. `/tmp/probe_block_chain.py` (recreate per
   `REMOVAL_2_DEEP_DIVE_2026_06_06.md`) — per-block residual at token
   132 stays within ±100 across all 36 blocks during IMM-M step.
4. `tests/test_smoke_*.py` holds ≥45/51 baseline with override
   *still in place*.
5. Only then: delete `batched_pure_neural.py:2149-2189` and re-run
   smoke. Target 45/51 → 45/51 with the 9 binop tests passing
   natively.

## 7. What NOT to touch in Wave 2

- Do **not** delete the `_BINARY_POP_OPS` set definition in
  `run_vm.py:123-140` — the serial runner still uses it
  (`run_vm.py:2334`), the non-collapsed recovery still uses it
  (`batched_pure_neural.py:2174` via the same import), and
  `_RUNNER_ALU_OPS` derives from it.
- Do **not** delete the non-collapsed recovery block
  (`batched_pure_neural.py:2191-2256`). It is the surviving Removal-4
  surface and covers a different model bug (32-bit cascade writeback +
  Shape-B EQ/NE), with its own removal plan in
  `32BIT_CASCADE_REAL_SURFACE_2026_06_07.md`.
- Do **not** delete `last_pushed_value` plumbing
  (`batched_pure_neural.py:189, 2013-2019`) — the non-collapsed
  recovery still reads it.

The Wave 2 cut is exclusively the 41-line block
`batched_pure_neural.py:2149-2189` plus its comment header (line 2126
onward already describes the BZ/BNZ removal pattern Wave 2 mirrors).

## 8. Expected post-removal smoke

Assuming Edits A+B+C land cleanly and pre-removal verifier gates pass:

- Pre-removal: 45/51 (current baseline, override in place).
- Post-removal: 45/51 (the 9 binop tests transition from
  override-passing to native-passing; the non-collapsed-recovery and
  control-flow tests are independent).

A regression below 45/51 indicates either (a) one of the joint edits
did not fully starve the cascade (raise Stage-0 search for a 4th
upstream writer per the plan's confidence section) or (b) a previously
masked failure surfaced in a non-binop test (rare; investigate the
specific test).

## 9. References

- Override block: `c4_release/neural_vm/batched_pure_neural.py:2149-2189`
- Set definition: `c4_release/neural_vm/run_vm.py:123-140`
- Serial sibling: `c4_release/neural_vm/run_vm.py:2334-2354`
- Non-collapsed sibling: `c4_release/neural_vm/batched_pure_neural.py:2191-2256`
- Joint fix plan: `c4_release/docs/COLLAPSED_STEP_JOINT_FIX_PLAN_2026_06_07.md`
- Attribution: `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`
- Per-block probe: `c4_release/docs/REMOVAL_2_DEEP_DIVE_2026_06_06.md`
- Override status table: `c4_release/docs/OVERRIDE_REMOVAL_STATUS_2026_06_06.md`
- Wave plan: `c4_release/docs/WAVE_PLAN_2026_06_07.md`
- Reverted attempts: `1ed6b5b6` (reverted by `4f627491`), `cd53c076`,
  `b91e3cd2`.
