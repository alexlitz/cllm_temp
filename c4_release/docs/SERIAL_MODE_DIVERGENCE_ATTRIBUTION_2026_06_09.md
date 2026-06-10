# Serial-mode smoke divergence attribution — 2026-06-09

_Status: DIAGNOSTIC. Read-only attribution; no code fixes. Root caused at
`batched_pure_neural.py:2149-2256` (Python ALU recovery overrides) — the
hypothesis that this was a cross-step KV propagation / `OUTPUT_HI_PREV_STEP`
bug is REJECTED._

## 1. The 17-test failure set (serial vs batched)

`C4_SMOKE_EXECUTOR=serial` adds 17 failures on top of the 5 already failing
under the default batched executor:

| Cluster | Tests | Opcodes |
|---|---|---|
| Comparisons | `test_eq_true`, `test_gt_true`, `test_ge_true` | `EQ`, `GT`, `GE` |
| Shifts | `test_shl`, `test_shr` | `SHL`, `SHR` |
| 32-bit arithmetic | `test_add_16bit`, `test_add_carry_cascade`, `test_sub_16bit`, `test_or_16bit`, `test_and_16bit`, `test_xor_16bit` | `ADD`, `SUB`, `OR`, `AND`, `XOR` |
| Multiplication | `test_mul_overflow` | `MUL` |
| Integration | `test_cmp_and_branch` | `EQ` + `BZ` |
| Memory | one 16-bit store-load test | `SI`/`LI` |

All 16 ALU/CMP failures share the structure `IMM X; PSH; IMM Y; <BINOP>; EXIT`
(or its 32-bit variant). The single memory test is the only outlier and
falls outside this class.

## 2. Hypothesis tested — REJECTED

The brief proposed: the batched runner has a KV-cache leak between
concurrent programs, and that leak supplies cross-step state via
`OUTPUT_HI_PREV_STEP` (commits 1eff091, 7afb953, 7291034). Serial loses
that "free" state and fails.

Direct measurements rule this out:

1. **Batched mode does not enable KV cache by default** — `use_kv_cache`
   defaults to `False` (`batched_pure_neural.py:295-296`, gated by
   `C4_BATCH_USE_KV_CACHE`). There is no shared KV state across batched
   programs to leak.
2. **`OUTPUT_HI_PREV_STEP` shares the same numeric slot (190) as
   `OUTPUT_HI`** — `vm_step.py:2316`. It is rename-only (B9 Q3 Option B,
   docs/B9_OUTPUT_HI_SPLIT_SPEC.md §4). No separate band exists that
   could be selectively populated by sibling programs.
3. **`replay_expected_diff.py` shows the same `layer3_carry_forward_attn`
   block-3 divergence on `OUTPUT_HI` / `OUTPUT_LO` / `AX_CARRY_LO` /
   `AX_FULL_LO` for EVERY program** — including passing tests
   (`IMM 42; EXIT`). The oracle gap is chronic and not serial-specific.
4. **Direct serial-runner execution fails identically when shl is the
   first program in a fresh runner** — no prior-program state needed.
   (`AutoregressiveVMRunner(pure_neural=True, ...).run(shl_bytecode)` →
   `('', 0)`, expected `('', 42)`.) Cross-program state propagation is
   not the mechanism.

## 3. Actual root cause — Python ALU recovery is batched-only

The batched runner contains TWO Python-side ALU overrides that the serial
`AutoregressiveVMRunner` (`pure_neural=True`) does NOT have:

### 3.1 Collapsed-step binary-ALU recovery
`batched_pure_neural.py:2149-2189`

When `IMM` is followed by a binary-pop op (anything in `_BINARY_POP_OPS`
= `{ADD, SUB, MUL, DIV, MOD, EQ, NE, LT, GT, LE, GE, OR, XOR, AND,
SHL, SHR}`) and the model emits both register blocks under a single
`STEP_END`, the runner detects `last_pc == exec_idx + 2` and invokes
`self._serial._compute_alu_legacy(skipped_op, stack_val, ax_after_imm)`,
then overrides `s.last_ax` and the `REG_AX` token bytes in `s.context`.

### 3.2 Non-collapsed binary-ALU recovery
`batched_pure_neural.py:2191-2256`

For ADD/SUB/OR/XOR/AND/EQ/NE executed as their own step (32-bit cascade
tests), the runner re-runs the legacy Python ALU using
`s.last_pushed_value` and `prev_ax`, then overrides `s.last_ax`.

### 3.3 Why serial does not have these
`AutoregressiveVMRunner._dispatch_step` (`run_vm.py:2066-2201`) takes
the early-return at line 2200 when `self.pure_neural == True` — the
binary-pop path at line 2334 (`elif exec_op in _BINARY_POP_OPS`) and
its `_compute_alu_legacy` call are unreachable. The serial pure-neural
runner trusts the model's emitted AX unconditionally.

The model itself emits AX=0 for SHL of 21 (and analogous wrong values
for the other 15 ops). The batched runner masks this with Python ALU;
the serial runner exposes it.

## 4. First divergent (layer, op, dim) per cluster

The clusters do not have distinct first-divergence points because the
divergence is in the runner Python harness, not in the model graph.
The model itself produces incorrect AX for every member of
`_BINARY_POP_OPS`; the surfacing only differs by which runner observes
the result.

Static oracle attribution (independent of serial vs batched) — same for
EVERY tested cluster including IMM_EXIT (a passing test):
- **L3, `layer3_carry_forward_attn`, `OUTPUT_HI+0` at block 3.**

This is a chronic gap, not the serial-failure cause. It would surface
in any executor mode.

## 5. Conclusion — single root, not multiple

Single root cause: **the batched runner's Python ALU recovery overrides
(`_compute_alu_legacy` calls in `batched_pure_neural.py:2149-2256`) mask
broken neural binary-pop emission for every op in `_BINARY_POP_OPS`**.
Removing the masks (or running through `pure_neural` serial) exposes
the underlying neural ALU bug for ~17 smoke tests.

The hypothesised cross-step `OUTPUT_HI_PREV_STEP` failure is NOT the
mechanism. Fixing the cross-step propagation would not change serial
mode pass/fail for these tests.

The one outlier — the 16-bit memory store-load test — does not fit the
`_BINARY_POP_OPS` recovery pattern and likely has an independent root
(probably `SI`/`LI` neural emit on multi-byte addresses); deferred to
a separate attribution.

## 6. Pointers for the 5 parallel fix agents

The fix is upstream of the recovery: the neural binary-pop emission
must produce the correct AX without Python rescue. Candidates to
investigate per opcode family:

- **SHL/SHR** — `L13.ffn.precompute_stage.shl/shr_precompute` +
  `L13.ffn.select_stage.shl/shr_select` (4096 + 528 units each, see
  `replay_expected_diff` retention listing).
- **MUL** — `layer12_mul_combine` (already declares
  `produces` `OUTPUT_HI@AX_byte0`).
- **ADD/SUB/OR/AND/XOR (16/32-bit)** — `layer9_alu` + `layer10_alu`
  + L17 `tail_bit32_result_correction` (per BLOG_SPEC, multi-byte
  carry/borrow chain).
- **EQ/GT/GE** — `layer10_cmp_combine` (per memory note
  `project_eq_byte1_l6_divergence.md` — EQ Shape-B fix landed there).
- **CMP+branch** — derivative of EQ; the BZ taken-path is sound (it
  reads the EQ result).

Removing the Python recovery before the neural fix lands will re-fail
the batched smoke gate for these 16 tests.

## 7. Incremental-removal toggle (task #133, 2026-06-09)

Removing the recovery code in one commit would regress smoke from
"46/51 with cheats" to fewer-pass-without-cheats and force every fix
agent to land simultaneously. Instead the recovery now consults two
env vars so each cluster can be retired independently as its upstream
neural fix lands.

### 7.1 Env vars

| Env var | Type | Default | Effect |
|---|---|---|---|
| `C4_DISABLE_BATCHED_ALU_RECOVERY` | bool (`1`/`true`/`yes`/`on`) | `0` | When set, BOTH recovery branches (collapsed + non-collapsed) skip `_compute_alu_legacy` for every op. Raw neural AX flows through. |
| `C4_DISABLE_BATCHED_ALU_RECOVERY_OPS` | comma-separated opcode names | _empty_ | When non-empty, ONLY these ops skip recovery. Every other op in `_BINARY_POP_OPS` continues to receive the cheat. Unknown names are silently skipped. Case-insensitive; whitespace tolerated. |

Both vars are checked at every dispatch call (re-read from
`os.environ`), so tests can flip them at runtime via
`monkeypatch.setenv` without re-importing the module.

Defaults preserve the 2026-06-09 smoke ~46/51 baseline — the recovery
code is NOT removed, only gated.

### 7.2 Rollout protocol

For each cluster owner in §6:

1. Land the upstream neural fix (e.g. ADD/SUB `layer9_alu` + L17
   correction). Confirm `replay_expected_diff` no longer flags the
   relevant `OUTPUT_LO`/`OUTPUT_HI` divergence at the targeted layer.
2. Run smoke locally with the per-op selector enabled for the
   cluster's opcodes:

   ```
   C4_DISABLE_BATCHED_ALU_RECOVERY_OPS=ADD,SUB \
       pytest tests/test_smoke.py -v
   ```

   Pass criteria: cluster tests pass WITHOUT the recovery; every other
   smoke test still passes (the other ops still get the cheat).

3. Once all 5 clusters are toggled off independently, flip the global
   `C4_DISABLE_BATCHED_ALU_RECOVERY=1` for a final smoke run. When
   that is green, a separate cleanup commit can delete the recovery
   code outright.

### 7.3 Test coverage

`tests/test_batched_alu_recovery_toggle.py` verifies:

- Env-var parsing (empty, well-formed, whitespace/case, unknown names).
- `_alu_recovery_disabled_for` honors both the global flag and the
  per-op selector.
- Collapsed-step recovery path (`batched_pure_neural.py:2218-2264`)
  actually consults the toggle — when disabled, `_compute_alu_legacy`
  is NOT invoked and AX is NOT rewritten with the sentinel.
- Non-collapsed recovery path (`batched_pure_neural.py:2266-2335`)
  consults the toggle the same way.

The dispatch is exercised with a stubbed `_serial` so no GPU / model
build is required.

