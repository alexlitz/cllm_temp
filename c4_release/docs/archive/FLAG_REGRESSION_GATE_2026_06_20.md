# Flag-ON Regression Gate (`tools/flag_regression_gate.py`) — design + demo

2026-06-20. The fast CPU cross-cluster gate for the **campaign** config
(`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`). Tooling only — the golden
flag-OFF model is byte-identical (`4958b35b`, re-verified with this change in
the tree).

## The blind spot this closes

The byte-identity gates (`compare_symbolic_to_lowered_ffn`,
`tools/_isa_golden_hash.py`) verify only the **flag-OFF** golden (35-token)
model. A change can be byte-identical OFF yet silently **regress a whole
cluster FLAG-ON** in the 30-token campaign config. This exact blind spot let a
mul `l14` fix pass golden byte-identity while crushing add/sub/div ~−60 in the
campaign config. There was NO fast structural gate for "does turning flag X on
(or applying branch Y) regress any cluster in the campaign config." This gate
is the automated form of the by-hand "cross-cluster verify."

## Design

- **Input.** `--flag C4_MY_FIX` (the fix's kill-switch: OFF leaves it unset, ON
  sets it, BOTH inside the campaign env) **or** `--base <commit>` (HEAD's
  working tree vs the base commit's source, checked out into a throwaway
  `git worktree` — **never** stashes / touches the live tree).
- **Sample.** `tools/flag_regression_sample.json`: 72 ids across ALL 56
  clusters — one cheapest representative per cluster, **4** each for the arith
  demo clusters (add/sub/mul/div/mod), **2** for `var_simple`. The four
  MEM-SMOKE `var_*` clusters (`var_simple`/`var_mul`/`var_three`/`var_update`)
  are ALWAYS present so a campaign fix can never silently break the SI/LI
  store-load path; they get an explicit `[MEM-SMOKE]` tag in the report. Ids
  prefer `tripwire_baseline.json` members (they span pass+fail). Regenerate
  with `--regen-sample`.
- **Verdict.** Both states are scored with the VALIDATED bit-exact
  `cpu_full_trace` (`--spec-k 0`, `--workers 2`, `--max-steps-cap 18`) — the
  same per-program full_trace verdict as `run_1096_canonical`, on CPU, no GPU.
- **Report.** Per-cluster `ok -> fail` (REGRESSION — **non-zero exit, BLOCK**)
  and `fail -> ok` (flip — gain), an always-printed MEM-SMOKE line, and a net.
- **Memory discipline.** `--workers 2` max, dedicated
  `C4_VM_CACHE_DIR=/tmp/c4cache_reggate` with per-state `off`/`on` subdirs (the
  two models never collide on disk), each state baked at most once (warmed in
  the parent → workers `torch.load`). The `cpu_full_trace` subprocess is the
  only child and is reaped before the gate returns.

## Demo — PROVEN both directions

The demo reuses commit `ba06deaa` ("mul byte-0/byte-1 result delivery,
SALVAGED + UNVERIFIED"), re-applied to `_layer14_alu_high_byte_relay_spec`
behind its own kill-switch `C4_MUL_BLK33_CLAWBACK` (gated on
`no_stack0_emit_enabled()` so it only exists in the campaign config, and only
when the flag is set → golden byte-identical OFF). It hard-excludes the MARK_AX
marker row from the high-byte relay's Q rows; that starves the byte-1 relay for
add/sub/div in the 30-token frame.

| Demo | command (`--clusters sub`, warm caches) | OFF | ON | gate verdict | exit |
|------|------------------------------------------|-----|-----|--------------|------|
| **A — catches the regression** | `--flag C4_MUL_BLK33_CLAWBACK` | sub **4/4 PASS** | sub **0/4 PASS** | REGRESSIONS (ok→fail) [4] `sub: ok->fail`; MEM-SMOKE clean; net **−4** "do NOT land" | **1** |
| **B — passes a clean change** | `--flag C4_REGGATE_BENIGN_NOOP` (no-op flag) | sub **4/4 PASS** | sub **4/4 PASS** | REGRESSIONS [0]; FLIPS [0]; MEM-SMOKE clean; net **+0** "CLEAN" | **0** |

Demo A on the broader arith set (`--clusters add,sub,div,mul`, 16 ids) shows the
OFF baseline add 2/4, sub 4/4, mul 2/4, div 1/4 — the clawback regresses the
sub (and add/div) byte-1 relay exactly as the original mul-`l14` fix did to the
full campaign run.

## Runtime

CPU full_trace is `O(steps^2)` per program and `--workers 2` is the memory-safe
cap, so wall time scales with sample size/depth:

- `--clusters sub` (4 ids), warm caches: **~355 s** both-states (Demo A),
  **~417 s** with cold bakes (Demo B).
- `--clusters add,sub,div,mul` (16 ids): ~25 min.
- Full 56-cluster default sample (72 ids, `--max-steps-cap 18` skips the deep
  loop/gcd/rec band): ~30–40 min — a CI / pre-merge gate, not an inner-loop
  tool.

While iterating on one fix, pass `--clusters <the few you touched>` for the
few-minute path; run the full sample once before landing. Cold bake per state
~40 s; warm re-run `torch.load` ~2–3 s.

## Status

`tools/flag_regression_gate.py` + `tools/flag_regression_sample.json` are the
gate; the `C4_MUL_BLK33_CLAWBACK` block in `l14_ops.py` is the demo regression
(OFF by default, golden-safe — golden `4958b35b` re-verified). Documented in
`CLAUDE.md` as the mandatory flag-ON cross-cluster gate.
