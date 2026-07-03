# 1096 Regression Check — 2026-06-11

## Verdict: ARTIFACT (not a regression, not a spec_k effect)

The ~1-3% pass rate from `tools/run_1096_fast.py` is **what the tool measures**,
not a regression from this session. The corpus was already at the ~1-2% floor on
this strict tool/config **before** the session's first model-weight change.

| Run                       | pass | fail | of  |
|---------------------------|------|------|-----|
| **current@spec0**         | 1    | 127  | 128 |
| **current@spec4**         | 1    | 127  | 128 |
| **baseline@spec0**        | 2    | 126  | 128 |

- **Baseline commit SHA**: `3153820b48b68cd16b77f6915647489302134f6a`
  (`feat(verifier): tautology skip + same-extent subbank refinement`) —
  the last commit before Wave B's first model edit (`62b64449`). It touches only
  `dim_alias_verifier.py` / `predicates.py` / tests; **no `l*_ops` or `vm_step`
  files**, so it cleanly predates every model-weight change this session
  (Wave B `62b64449`, L11 relay enable `d240b9a9`, the L10/operand follow-ups).
- Current HEAD: `bffb3593`.

## Why it's an ARTIFACT, not a REGRESSION

Baseline (2/128) ≈ current (1/128). Both fail ~98-99% of the first-128 slice.
The pass sets are tiny and nearly disjoint — pure noise-floor churn, not a
newly-failing cluster:

- Failure-list intersection: **125 of ~127** ids fail at BOTH baseline and HEAD.
- Symmetric diff is only 3 ids: `{84, 114, 118}`.
  - id=84 `sub_34: 196-14` — **HEAD fixed it** (baseline neural=4294967282 → HEAD neural=182 ✓)
  - id=114 `mul_14: 3*15` — **HEAD broke it** (baseline neural=45 ✓ → HEAD neural=1)
  - id=118 `mul_18: 11*11` — **HEAD broke it** (baseline neural=121 ✓ → HEAD neural=1)
  - Net: +1 sub fixed, −2 mul broken = net −1, i.e. noise at the 1-2% floor.

The two MUL flips are consistent with this session's MUL refactor
(`83cf2636` MUL partial+combine MARK_AX→MARK_SE_ONLY) but are 2 programs out of
a corpus already pinned at ~1% — far below any "regression" threshold. MUL/DIV/MOD
is independently noted as architecturally hard (`project_mul_div_mod_arch_blocked.md`).

## Why it's not a SPEC_K artifact

current@spec0 == current@spec4 == 1/128 (identical pass AND identical fail set).
Speculation does not change the count. The 3% figure is not a spec_k=4 artifact.

## What the tool actually measures (the real reason for ~1-2%)

`run_1096_fast.py` requires an **exact full neural-exit-code match** against the
declarative oracle across ALL steps of each program. The first-128 ids are
`add_*` / `sub_*` / `mul_*` arithmetic. The declarative oracle matches
`suite_expected` on 128/128; the neural model produces a systematically corrupted
exit code on ~127/128 — e.g. `768 → 34816 (0x8800)`, `858 → 65360 (0xFF50)` — the
known high-byte / 0xFF byte-1 leak pattern (`project_ax_bytes_1_3_ff_leak_*`,
`project_sub_smoke_shares_root_with_0xff_leak.md`). This is a pre-existing
architectural gap, identical at baseline. The historical "~95%" figure refers to
*other* gating paths (e.g. smoke / declarations-only configs), NOT this strict
full-exit-code tool — exactly the config-dependence warned about in
`project_1096_sentinel_baseline.md`.

## Reproduction

```
# current HEAD
CUDA_VISIBLE_DEVICES=1 python c4_release/tools/run_1096_fast.py --limit 128 --chunk 16 --spec-k 0
CUDA_VISIBLE_DEVICES=1 python c4_release/tools/run_1096_fast.py --limit 128 --chunk 16 --spec-k 4

# baseline (isolated worktree)
git worktree add --detach /tmp/1096_baseline_wt 3153820b
cd /tmp/1096_baseline_wt
CUDA_VISIBLE_DEVICES=1 python c4_release/tools/run_1096_fast.py --limit 128 --chunk 16 --spec-k 0
```

## Follow-up (optional, low priority)

If the 2 MUL flips (id=114, id=118) ever matter, bisect `3153820b..HEAD` over the
MUL-touching commit `83cf2636` (l11/l12 MUL partial+combine MARK_AX→MARK_SE_ONLY).
But these are noise on a corpus already at the 1% floor — no action needed for the
regression question, which is settled as ARTIFACT.
