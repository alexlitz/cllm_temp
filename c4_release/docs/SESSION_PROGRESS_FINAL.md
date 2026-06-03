# Session progress — cumulative measurement at HEAD `5e9ff9be`

Date: 2026-06-03
Worktree: `/tmp/c4-full-measurement` (HEAD-based, isolated)
HEAD: `5e9ff9be05c056d64cf997226fe9362ed62756f5`
  (`dsl: wide_alu_dsl.py scaffolding + bitwise_rules POC`)
Branch context: `speedup-cache-and-buckets`

Two measurements at the current tip:

1. **Smoke** — `c4_release/tests/test_smoke.py` via
   `tools/smoke_track.py` (51 collected, 1 deselected).
2. **1096 lookup-mode cluster sample** — 135 selected indices across
   five clusters, run via `run_1096_clusters.py` (in worktree root)
   which wraps `run_1096_neural_declarative_diagnostic` with
   `alu_mode='lookup'`, `spec_k=0`, declarations-only bake on,
   `comparison_mode='final-output'`.

The sample composition (135 tests) is below the brief's nominal
200-test target because the named clusters total 135. No extra
indices were drawn; the brief's enumerated ranges were the contract.

## Smoke — current

```
passed=25 failed=26 errors=0 skipped=0 xfail=0 xpass=0 duration=93.15s
snapshot: c4_release/.smoke-snapshots/smoke_5e9ff9be_20260603T165613Z.json
```

Per-class pass/fail (pytest -v output):

| Class                  | Pass | Fail |
|------------------------|-----:|-----:|
| TestSmokeBasic         |    1 |    3 (sub, div, mod) |
| TestSmokeBitwise       |    1 |    2 (and, xor)      |
| TestSmokeComparison    |    0 |    6 (eq_true, eq_false, lt_true, ne_true, gt_true, ge_true) |
| TestSmokeAddress       |    n |    1 (adj_sp)        |
| TestSmokeMemory        |    n |    5 (si_li_roundtrip, sc_lc_roundtrip, si_li_multiple_stores, si_li_overwrite, si_li_16bit_value) |
| TestSmokeShift         |    0 |    2 (shl, shr)      |
| TestSmoke32Bit         |    n |    7 (add_16bit, add_carry_cascade, sub_16bit, or_16bit, and_16bit, xor_16bit, mul_overflow) |
| (other classes)        |   passes round out 25 |     |

(Class "n" counts above are the residual after subtracting the listed
failures from the 25 total passes; the snapshot JSON has the per-test
granularity if needed.)

## 1096 — cluster sample at HEAD

```
Overall pass: 19/135
  add        idx[  0.. 49] pass= 0/50 errors=0 wall= 76.0s
  var_three  idx[300..324] pass= 0/25 errors=0 wall=629.6s
  if_eq      idx[400..424] pass=10/25 errors=0 wall= 43.3s
  edge_pow2  idx[1012..1021] pass= 9/10 errors=0 wall=  2.8s
  absdiff    idx[1046..1070] pass= 0/25 errors=0 wall=521.8s
```

All failures are `neural-divergence` (no `suite/declarative-mismatch`,
no compile errors). i.e. the declarative interpreter matches the suite
expectation on every selected program; the neural VM's exit code
diverges.

Cluster reads, lining up with existing smoke triage:

- **add 0/50** — corresponds to the smoke `A-AXZERO` cluster
  (AX → out-byte residual chain). `add_42` was the canonical failing
  case in `SMOKE_MEMORY_TRACE_20260603.md`.
- **var_three 0/25** — `var_*` cluster. Per
  `~/.claude/.../project_var_failure_mode_shifted.md`, current
  failure mode is L3 block=3 gen=13 SP_byte2 OUTPUT_LO[1]-vs-[0]
  (shifted from the older L15 gen=0 REG_PC root cause). Not yet
  fixed at this HEAD.
- **if_eq 10/25** — partial; matches the smoke comparison cluster
  (L6 cross-step `CMP.*.-1` read on BZ/BNZ override) being partially
  active. Passing ones likely route through paths the override
  doesn't catch.
- **edge_pow2 9/10** — only `edge_pow2_3 (2^3)` fails; the rest pass.
  Suggests the SHL fast path mostly works but breaks on some shift
  count or operand alignment.
- **absdiff 0/25** — `absdiff_*` is the BZ-redirect cluster
  (`ABSDIFF_BZ_REDIRECT_BUG.md`); unchanged at this HEAD.

## Cumulative comparison table

Smoke (51 collected, 1 deselected → 50 active):

| Stage              | Commit     | Pass | Fail | Errors | Delta vs baseline |
|--------------------|------------|-----:|-----:|-------:|-------------------|
| Baseline           | `2ed33168` |    6 |    4 |     41 | —                 |
| After L5 fix       | `43dc13a2` |   25 |   26 |      0 | +19P / +22F / -41E |
| Current HEAD       | `5e9ff9be` |   25 |   26 |      0 | +19P / +22F / -41E |

Smoke delta vs `43dc13a2` (post-L5): **0 / 0 / 0** — no smoke
regression and no smoke recovery since the L5 fix landed.

1096 (sample composition note in caveat below):

| Stage              | Commit     | Pass | Total | Delta vs baseline |
|--------------------|------------|-----:|------:|-------------------|
| Baseline           | `b1dd9a19` |   16 |   142 | —                 |
| After L5 fix       | `43dc13a2` |   12 |   135 | -4P (and -7 total population) |
| Current HEAD       | `5e9ff9be` |   19 |   135 | +3P vs baseline; +7P vs post-L5 |

1096 delta vs `43dc13a2`: **+7 pass** on the 135-test cluster sample.
The gain is concentrated in:
- `if_eq` cluster: 10/25 now passing.
- `edge_pow2`: 9/10 (essentially solved at this slice).

The `add`, `var_three`, and `absdiff` clusters remain at 0 passes —
their root causes (AX-zero residual, var_* L3 SP_byte2, BZ redirect)
are tracked in their existing root-cause docs and have not been
addressed by the L5 fix or the post-L5 commits.

### Caveat: 1096 sample composition

The historical 1096 figures (`16/142`, `12/135`) cited in the brief
are population counts whose exact composition is not 1:1 with the
present sample. The "post-L5 12/135" line and the current "19/135"
were collected with the same five-cluster slice (add/var_three/if_eq/
edge_pow2/absdiff, totals 50/25/25/10/25 = 135), so the +7 delta on
this slice is apples-to-apples. The baseline `16/142` figure is from
a different slice and is included only as a reference point — the
+3 vs baseline number should be read as "current sample passes 19,
baseline-era sample passed 16," not as a strict cluster-level delta.

## Sources

- Smoke snapshot: `c4_release/.smoke-snapshots/smoke_5e9ff9be_20260603T165613Z.json`
- 1096 raw log: `/tmp/c4-full-measurement/.1096_clusters.log` (worktree-local; not committed)
- Cluster shim: `/tmp/c4-full-measurement/run_1096_clusters.py` (worktree-local; not committed)
- Existing triage docs referenced:
  `SMOKE_TRIAGE_POST_L5.md`,
  `SMOKE_MEMORY_TRACE_20260603.md`,
  `ABSDIFF_BZ_REDIRECT_BUG.md`,
  `CMP_PATH_AUDIT.md`,
  `SMOKE_COMPARISON_OP_DECODE_MISSING.md`,
  `SMOKE_ROOT_CAUSE_BAKE_COLLISION.md`.
