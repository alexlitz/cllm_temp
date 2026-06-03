# Cluster D attempt 3 — findings (revert)

Date: 2026-06-03
Branch: speedup-cache-and-buckets (worktree `/tmp/c4-cluster-d-l10` at HEAD `cca25226`)
Author: agent (Claude Opus 4.7)

## TL;DR

Cherry-picked the prior cluster D fix (`3f14a375`: physical L6→L25 relocation of BZ/BNZ
PC-override) + made it strict-mode safe (changed cancel-band gate from `OUTPUT_LO.*.-1`
to same-step `OUTPUT_LO`). Compiled cleanly, op landed at physical L25 (well past
`layer10_alu` at L13), smoke = **26 pass / 26 fail** (vs prior commit's reported 25/26).
**The 6 CMP cluster tests (`test_eq_true`, `test_eq_false`, `test_lt_true`,
`test_ne_true`, `test_gt_true`, `test_ge_true`) did NOT improve.** Reverted.

## What the change did

- Cherry-pick of `3f14a375` (post-L9 BZ/BNZ PC override op with
  `requires={"after": "layer10_alu"}`, `kind="ffn"`).
- Cancel-band gate switched from cross-step `OUTPUT_LO.*.-1` to same-step `OUTPUT_LO`
  in `_post_l9_bz_pc_override_rules` and `_post_l9_bnz_pc_override_rules`
  (`l6_ops.py` ~lines 4408-4480). Rationale: at the L25 placement, OUTPUT_LO has
  been freshly written this step by upstream layers (L6 routing FFN, L8 ALU, ...).
  Subtracting `OUTPUT_LO.*.-1` would subtract the prior step's value, leaving the
  current step's residual intact and causing double-write. Subtracting same-step
  `OUTPUT_LO` correctly cancels the residual. This also makes the strict-mode
  `CrossStepReadError` go away (the `OUTPUT_LO.*.-1` read was the one new
  finding flagged by the Step 4 baseline allowlist).
- Verified op declaration reads = `{MARK_PC, MARK_STACK0, OP_BZ, OP_BNZ, CMP,
  IS_BYTE, FETCH_LO, FETCH_HI, OUTPUT_LO, OUTPUT_HI_THIS_STEP}` (same-step CMP,
  same-step OUTPUT_LO). No `.*.-1` aliases.

## Physical placement (verified post-compile)

  | Op                              | Layer | Notes                              |
  |---------------------------------|-------|------------------------------------|
  | layer6_attn (dep anchor)        | L6    | unchanged                          |
  | layer6_routing_ffn              | L6    | BZ/BNZ bands zero-cleared          |
  | layer9_alu                      | L10   | authoritative CMP writer           |
  | layer10_alu                     | L13   | cmp_combine writes OUTPUT_LO       |
  | post_l9_bz_bnz_pc_override      | **L25** | new op — well past L9 ALU+L10 ALU |

Confirmed via `layout.ops_per_layer[25]` and `layout.resolve_block_op_layer(op)`.

## Smoke result (alu_mode='efficient', batched pure-neural)

  - **26 pass / 26 fail / 29 xfailed / 11 xpassed** (`tests/test_smoke.py` +
    `tests/test_smoke_pure_neural.py`)
  - Prior commit `3f14a375` message reported 25/26 — my +1 pass delta is
    likely noise from other intervening commits, not from this fix.
  - **None of the 6 D-CMP tests changed**: still
    `test_eq_true` (expected 1, got 0), `test_eq_false` (expected 0, got 1),
    `test_lt_true` / `test_ne_true` / `test_gt_true` / `test_ge_true` (all
    expected 1, got 0).

## Why the cluster D hypothesis was wrong for these tests

The D-CMP failures share an **inverted output**:

  - EQ(42, 42) → 0 instead of 1 (zero output, expected the eq flag)
  - EQ(10, 20) → 1 instead of 0 (flipped)
  - NE(10, 20) → 0 instead of 1
  - LT/GT/GE_true (10 vs 20) → 0 instead of 1

These tests are **pure ALU comparison opcodes**, not branches. The bytecode is
`(IMM, x), PSH, (IMM, y), <OP>, EXIT`. The EXIT exit code is the AX value of
the CMP opcode itself — set by the L9 ALU's `cmp` substage feeding L10's
`cmp_combine` / `comparison_combine` rules (`l10_ops.py:490, 652, 1017`).

The BZ/BNZ PC override has **no role in the AX value** of EQ/NE/LT/GT/LE/GE.
Moving the override from L6 to L25 cannot change those tests.

The triage doc `c4_release/docs/SMOKE_TRIAGE_POST_L5.md` lines 6-11 attribute
the failures to "L6 cross-step CMP.*.-1 read on BZ override" — that
attribution is incorrect. The proximate cause sits in the **L10 cmp_combine /
comparison_combine path**: the rules either produce the inverted polarity or
the OUTPUT_LO write is being clobbered downstream.

## Next-level blocker (where to look next)

1. **L10 cmp_combine sign / polarity audit.** `_l10_comparison_combine_rules`
   (`l10_ops.py:490-650`) uses `cmp_override_2way("OP_EQ", "CMP+0", 1, 0, ...)`
   semantics: default 0, override CMP+0 → 1 (`grep -n "OP_EQ" c4_release/neural_vm/unified_compiler/ops/l10_ops.py`).
   Confirm CMP+0 is the EQ flag (a==b → CMP+0=1) and that the default-0/override-1
   semantics writes to OUTPUT_LO+0 of the AX row, not a different output dim.

2. **L9 ALU CMP write — verify CMP+0 polarity for the 8 cases.**
   `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:1062-1185` (cmp + hi_lt
   substages, 120+18 units). Check that the ALU writes CMP+0=1 on `a==b` and
   CMP+0=0 on `a!=b`. If the polarity inverted in a recent refactor (efficient
   mode AddSub5StageBlock → wide_addsub_rules — see commit `69b46486` dsl-w3),
   the override default-0/override-1 in L10 will INVERT the output as observed.

3. **Verify on a pure-neural run with NO Python overrides.** The `pure_neural`
   suite (`tests/test_smoke_pure_neural.py::TestSmokePureNeuralComparison`)
   marks EQ/NE/LT/GT/LE/GE as xfail. If those xpass with the same flipped
   values, the bug is in the neural CMP path (most likely L9 cmp substage or
   L10 cmp_combine). If they xpass with correct values, the bug is in the
   handler-mode CMP path or the PSH→stack-byte forwarding before the CMP.

4. **Decl-verifier on `layer10_alu` + `l10_post_ops_combined`.**
   `c4_release/neural_vm/unified_compiler/decl_verifier.py` — check that the
   declared rules' written values match what the bake produces (rule polarity
   sanity check).

## Why I reverted

Per task brief: "If negative or zero: revert and document the next-level
blocker." The 6 CMP tests targeted by the brief did not change. The architecture
of the cluster D fix was sound (op lands at L25, strict-mode clean, BZ/BNZ
pure-neural test family still works at the same op), but the **observed CMP
test failures are not caused by the BZ override path** — moving it does not
move the needle.

The cluster D physical-relocation pattern remains a useful tool for any
future BZ-on-step-1 work, but it cannot fix CMP polarity issues. The next
agent should target the L9 cmp substage and L10 cmp_combine polarity, NOT
the BZ override.

## File touchpoints (if the next agent re-attempts)

- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4390-4554` — the
  prior cluster D code (`_post_l9_bz_pc_override_rules` + `make_post_l9_bz_bnz_pc_override_op`),
  with cancel-gate corrected to same-step OUTPUT_LO. Use `git show 3f14a375` +
  the same-step `OUTPUT_LO` patch documented above to reproduce.
- `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py:238-246` — wiring
  point for `make_post_l9_bz_bnz_pc_override_op()` into `all_core_ops`.
- `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:1062, 1185` — L9 ALU
  cmp / hi_lt substages (CMP write).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:490-650, 652-1017` —
  L10 cmp_combine / comparison_combine (CMP → OUTPUT_LO).
- `c4_release/docs/SMOKE_TRIAGE_POST_L5.md` — triage entries 6-11 (D-CMP).
- `c4_release/docs/CMP_PATH_AUDIT.md` — write/read chain audit (referenced
  but its absdiff conclusion does not apply to these AX-output tests).
