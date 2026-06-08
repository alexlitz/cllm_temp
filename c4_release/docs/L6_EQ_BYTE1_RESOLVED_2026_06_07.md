# L6 EQ(17,17) byte-1 divergence — RESOLVED upstream at L10 cmp_combine (2026-06-07)

Attribution closure for the Shape-B `EQ_FALSE / NE_TRUE` 1096-subset
investigation cited in
[`32BIT_CASCADE_REAL_SURFACE_2026_06_07.md`](32BIT_CASCADE_REAL_SURFACE_2026_06_07.md)
section "Shape-B EQ_FALSE/NE_TRUE subset" and memory note
`project_eq_byte1_l6_divergence.md`.

## TL;DR

The L6 byte-1 divergence is **no longer reproducible** as of HEAD
`68025246` on `speedup-cache-and-buckets`. Commits `e70ddc6d`
("fix(removal-4): Shape B CMP combine + remove 69f77682") and
`4273e968` ("fix(removal-4): restore EQ/NE non-collapsed recovery to
fix integration") landed a structural fix at the L10 `cmp_combine`
rule (the CMP+0 hi_lt blocker), and that fix resolves the Shape-B
EQ_FALSE / NE_TRUE failure mode the L6 brief targeted.

**Resolution layer**: L10 `cmp_combine` `_cmp_override_3way` rule —
NOT L6 `layer6_attn / layer6_routing_ffn / layer6_relay_heads`.

**Verifier-blind status of L6** (per
`L6_EQ_VERIFIER_BLIND_2026_06_04.md`) is therefore moot for this
failure mode: the EQ corruption at block 6 the memory note
observed is real, but it is *not load-bearing* on the Shape-B
outcome because the downstream L10 cmp_combine fix re-anchors AX
correctly for EQ/NE regardless of the L6 byte-1 residual.

## Live state (2026-06-07, HEAD `68025246`)

### Smoke (tests/test_smoke.py::TestSmokeComparison)

| Test | Result |
|---|---|
| test_eq_true | PASS |
| test_eq_false | PASS |
| test_lt_true | PASS |
| test_ne_true | PASS |
| test_gt_true | PASS |
| test_le_true | PASS |
| test_ge_true | PASS |

7/7 passing. Smoke baseline 45/51 (6 unrelated SI/LI/LEA memory
failures, none touching the CMP family).

### 1096 corpus `if_eq_*` (25 tests, Shape-B EQ surface)

```
pytest tests/test_suite_1096_pure_neural_pytest.py -k if_eq --tb=no -q
... 25 xpassed in 27s
```

25/25 XPASS. Every `if_eq` program (operand pairs ranging across the
1..50 value space, both equal and unequal cases) decodes correctly.

### 1096 corpus broader CMP sweep

| Subset | xpass | xfail | notes |
|---|---:|---:|---|
| `if_eq_*` | 25 | 0 | Shape-B EQ_FALSE+EQ_TRUE — fully recovered |
| `if_gt_*` | 25 | 0 | GT recovery clean |
| `if_lt_*` | 24 | 1 | `0395_if_lt_20_96_<_70` — one isolated failure, sample-size class not L6 |
| `if_var_*` | 12 | 13 | L3 SP_byte2 surface per `project_var_failure_mode_shifted.md`; not the L6 EQ surface |

The `if_var_*` cluster is the L3 `SP_byte2 OUTPUT_LO[1]-vs-[0]`
failure mode tracked separately in memory; it is structurally
unrelated to EQ byte-1.

## Why the L6 surface is no longer load-bearing

The original block-by-block diff (`EQ_BLOCK_DIFF_2026_06_04.md`)
showed a +339 prediction-row residual norm jump at block 6 for
EQ(17,17) vs EQ(5,5). That observation was correct, but the fix-by-
override at L10 cmp_combine (`e70ddc6d`) intercepts the CMP +0/+1
hi_lt/hi_eq signals at the result-combine layer, well after the
attention-residual divergence at L6. The CMP combine rule writes AX
directly from CMP+0/+1 with a properly-blocked threshold, so any
upstream byte-1 noise in OUTPUT_HI is cancelled.

The verifier-blind status of `layer6_routing_ffn`'s 1315 rules
(0/0 with scope/dominates_at, per
`L6_EQ_VERIFIER_BLIND_2026_06_04.md`) remains an Phase 7.E annotation
debt, but it no longer blocks Shape-B CMP correctness.

## Fix to brief tracker

Brief: "L6 byte-1 divergence in EQ(17,17) affects Shape-B
EQ_FALSE/NE_TRUE 1096 tests."

- **Was**: open investigation across `layer6_attn`,
  `layer6_routing_ffn`, `layer6_relay_heads`.
- **Now**: RESOLVED upstream at L10 `cmp_combine` (commits
  `e70ddc6d` and `4273e968`). No L6 patch warranted.

## Recommended memory update

Mark `project_eq_byte1_l6_divergence.md` as **superseded**:

```
SUPERSEDED 2026-06-07 by L10 cmp_combine fix (commits e70ddc6d +
4273e968). The L6 attribution was anatomically correct but the
load-bearing surface for Shape-B EQ/NE was at L10 cmp_combine, not
L6. See docs/L6_EQ_BYTE1_RESOLVED_2026_06_07.md.
```

## Cross-references

- Memory note: `project_eq_byte1_l6_divergence.md` (2026-06-04 — now
  superseded).
- `docs/L6_EQ_VERIFIER_BLIND_2026_06_04.md` — verifier-blind status of
  `layer6_routing_ffn` (still accurate as a Phase 7.E annotation debt
  note; no longer a CMP blocker).
- `docs/EQ_BLOCK_DIFF_2026_06_04.md` — original block-by-block diff
  (measurement remains valid; fix surface attribution refuted).
- `docs/32BIT_CASCADE_REAL_SURFACE_2026_06_07.md` §"Shape-B
  EQ_FALSE/NE_TRUE subset" — the cross-doc citation that pointed
  here. The "independent L6 fix" claim in that section can now be
  marked resolved.
- `docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` — Shape-A
  collapsed-step surface; separate fix track (L27→L28→L34 cascade).
- Resolving commits: `e70ddc6d`, `4273e968` (both 2026-06-07).

## Confidence

- **High** that the Shape-B EQ_FALSE / NE_TRUE 1096 subset passes
  today (25/25 if_eq pass + all smoke CMP pass, direct test).
- **High** that commits `e70ddc6d` + `4273e968` are the resolving
  changes (commit bodies explicitly cite the Shape B CMP fix at L10
  cmp_combine).
- **High** that the L6 surface is not a blocker (no L6 patch landed
  between the failing-state memory note and the current passing
  state; the only intervening change touching the CMP path is the L10
  cmp_combine work).
- **Medium** that no L6 byte-1 residual still leaks into other
  downstream consumers — `layer6_routing_ffn`'s 0/0 annotation status
  means we cannot formally prove this via the verifier. Future
  Phase 7.E annotation work would close this gap.
