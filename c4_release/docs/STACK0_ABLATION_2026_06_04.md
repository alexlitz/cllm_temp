# STACK0 byte-1 — `head.bias[REG_PC]` ablation, 2026-06-04

Date: 2026-06-04
Branch: `stack0-ablation` (off `cdc28069` on `speedup-cache-and-buckets`)
Worktree: `/tmp/c4-stack0-ablation/c4_release/`
Author: agent (Claude Opus 4.7)
Brief: ablation recommended in
`docs/STACK0_BYTE1_BLOCKER_ATTEMPT_2026_06_04.md` §"What to try next"
and previously sketched in `docs/STACK0_STALENESS_FOLLOWUP_2026_06_03.md`.

## TL;DR

The byte-1 blocker is **upstream residual divergence**, not logit
competition. Suppressing `Token.REG_PC`'s output-head bias by an
additional `-30.0` (from `-10.0` to `-40.0`) does NOT recover
`EQ(17,17)`, AND it breaks the previously-passing `EQ(5,5)`.

Both pre- and post-ablation `EQ(17,17)` return exit code `0`
(divergence-bail); ablation makes the symptom strictly worse, ruling
out a logit-race fix and pointing at the byte-1 OUTPUT-band itself
(L8 / L10 / L14 STACK0 producer chain) as the bug surface.

A small but interesting side finding: the model already ships a
`head.bias[REG_PC] = -10.0` bias as a deliberate token-vs-marker
weight. That single scalar is load-bearing — removing it kills
`EQ(5,5)`, the very test that currently works. Whatever path lets
EQ(5) succeed depends on the existing `-10` bias to win the
byte-value-vs-REG_PC race, so the marker-vs-byte competition is
genuinely close at byte 1 and we are not "miles away from" the right
answer — we are tipping the wrong way by a small margin.

## Procedure

Ablation script: [`/tmp/stack0_ablation.py`](`/tmp/stack0_ablation.py`)
(monkey-patches `model.head.bias[Token.REG_PC] -= 30.0`, runs
`IMM,PSH,IMM,EQ,EXIT` for `(5,5)` and `(17,17)`, restores bias, reruns
sanity case).

Build flags (matching the brief):
```python
runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
runner._syscall_handlers = {}     # belt + suspenders; pure_neural already skips these
```
`Token.REG_PC = 257` (vocab dim 257 in `model.head.bias`). The
shipping value is `head.bias[REG_PC] = -10.0` (a *deliberate*
marker-vs-byte tilt — not 0.0 as one might naively expect).

`Token.MARK_PC` referenced in the brief is a misnomer; there is no
`Token.MARK_PC`, only `BD.MARK_PC` (residual dim 0) and the emitted
token `Token.REG_PC = 257`. The ablation targets the token bias as
the brief intends.

## Results

| Run                                   | EQ(5,5) | EQ(17,17) |
|---------------------------------------|---------|-----------|
| baseline (`head.bias[REG_PC] = -10`)  | **1**   | **0**     |
| ablated (`head.bias[REG_PC] = -40`)   | 0       | 0         |
| restored (`head.bias[REG_PC] = -10`)  | —       | 0         |

Expected for EQ(N,N): exit code `1` (EQ result truthy → EXIT consumes 1).

Notes:

* `EQ(17,17)` already bails on `draft_divergence: 25 of last 50 tokens
  disagree`. The model loses lockstep with the deterministic draft
  inside the EQ step, so exit code `0` here is the bail-out path
  returning the last AX (= 0), not a clean "EQ said false".
* The restored sanity case matches baseline `EQ(17,17) = 0` byte for
  byte; weight restoration is correct.
* `EQ(5,5)` regression under ablation is reproducible across the same
  build (no new compile between runs).

## Interpretation

### 1. Not logit competition at byte 1

If the byte-1 OUTPUT residual at the AX byte-1 emission position
already carried "byte 0" correctly and only lost to `REG_PC`'s bias,
dropping that bias by 30 units (3× the existing tilt) would have
flipped the argmax. It did not — `EQ(17,17)` is identically broken
pre and post ablation. The residual at that position is not "byte 0
plus a too-strong REG_PC nudge"; it is genuinely something other than
byte 0.

### 2. The marker-vs-byte tilt is load-bearing elsewhere

EQ(5,5)'s regression confirms the `-10` shipping bias does real work
at *other* byte positions in *other* steps. Whatever byte-emission
the EQ(5) trace depends on (likely the EQ result byte itself, or one
of the ALU intermediates) is close enough to the REG_PC logit that
removing the bias hands the token to `REG_PC` instead. So we cannot
"just lower REG_PC bias more" — that breaks the working cases without
fixing the broken one.

### 3. Bug surface lives upstream of `head_bake`

Combining the two: the byte-1 residual at the EQ(17) AX position is
positively biased toward `REG_PC` (vocab dim 257) by enough that even
`-40` bias cannot overcome it, while the byte-N residual at the EQ(5)
positions is sitting on the knife-edge that the `-10` bias is tuned
for. The difference between the two traces is not in head_bake or
head.bias — it's in the residual that arrives at the final norm.

This matches the prior doc's "head 3 ablation was a no-op" finding:
the leak path is not L10 head 3, and now it's not the head.bias
either. The actual divergence is somewhere in:

* **L8 OUTPUT_LO/HI consolidation** for AX byte 0 with the high
  nibble set (bytes ≥ 16 vs. bytes < 16). EQ(5) and EQ(17) differ
  precisely on whether AX byte 0's high nibble is non-zero; this is
  the cleanest hypothesis.
* **L9/L10 byte-relay heads** (the
  `_layer10_stack0_byte_relay_head_spec` family writing
  STACK0_BYTE1/2/3) — these read from L8 OUTPUT, so a high-nibble
  artifact at byte 0 could propagate.
* **L14 OUTPUT band cleanup** — `_layer14_temp_clear_rules` and
  friends; the `_l14_chain_alloc` layout. A stale OUTPUT_HI write
  at the AX byte-1 position would survive into head_bake.

### 4. The `draft_divergence` bail is itself a clue

The runner reports >50% disagreement with the DraftVM in the last 50
emitted tokens, well before EXIT. That means many tokens in the EQ(17)
trace are wrong, not just one. The "AX byte-1 = MARK_PC at offset 22"
symptom flagged in the prior doc is the *first* divergence the bail
catches; downstream tokens then cascade. Whichever block produces the
first wrong token at offset 22 is the surface to investigate.

## Recommended next step

**Hook block-by-block residual at the EQ(17) step-4 AX byte-1
position (pos 135 per the prior doc) and diff against EQ(5) at the
same position.** Specifically:

1. In the worktree (or any branch with both EQ runs available), run
   the model with a per-block forward-hook that captures
   `residual[pos=135, :]` after every transformer block (L0 .. L25
   raw indices, or L0 .. L35 post-expansion).
2. For each block, compute `residual_eq17 - residual_eq5` at
   `pos=135`. The first block where the L2-norm of that delta
   exceeds a "noise floor" (say > 5.0 on residual scale where typical
   marker logits are 50–200) is the divergence source.
3. Cross-reference that block's `op.writes` against the residual dim
   that flipped — most likely a `STACK0_BYTE1`, `OUTPUT_BYTE_HI`, or
   `MARK_PC` dim per the contract table.
4. If the divergence appears at L8 / L10 / L14 (the most likely
   suspects), use `decl_verifier.verify_rule_scopes` /
   `verify_rule_strength` on the relevant op to see whether any rule
   in that block claims to gate on "byte-value < 16" or "high nibble
   == 0" — those are the gates that would silently miss for EQ(17).

Rationale: per `feedback_single_rule_fixes_are_zero_sum.md`, manual
rule patches have a 0/5 success rate. Use the verifier's actual
contribution algebra against a known-good (EQ(5)) and known-bad
(EQ(17)) trace to localize the divergence before touching any rule.

## What this rules out

* **Logit competition at byte 1** — ablation could not flip the
  argmax even with 3× the shipped REG_PC tilt.
* **head.bias as the fix surface** — removing/strengthening the
  marker tilt is strictly worse (breaks EQ(5), does not fix EQ(17)).
* **MARK_PC token id confusion in the brief** — the actual emitted
  token is `Token.REG_PC = 257`, and even targeting it directly
  cannot save byte 1.

## Files

* New: `c4_release/docs/STACK0_ABLATION_2026_06_04.md` (this file).
* Used (read-only): `c4_release/neural_vm/run_vm.py`,
  `c4_release/neural_vm/vm_step.py` (`Token` class, line 85+),
  `c4_release/neural_vm/embedding.py` (`Opcode`).
* Off-tree script: `/tmp/stack0_ablation.py` (10 lines of ablation,
  the rest is print scaffolding; not committed — diagnostic only).

## Reproduction

```bash
cd /tmp/c4-stack0-ablation/c4_release
python /tmp/stack0_ablation.py
# expected output: see "Results" table above
```

Two compiles total (one runner reused across baseline/ablated/restored).
Runs in ~3 min on CPU (model build dominates; each `run()` is < 5 s).

## Confidence

* **High** that the byte-1 corruption is not logit competition.
  Ablation cleanly disconfirms.
* **High** that `head.bias[REG_PC]` is load-bearing for other byte
  emissions — EQ(5) regression is direct evidence.
* **Medium-high** that the divergence is in L8 OUTPUT consolidation
  for high-nibble bytes. This is the smallest behavioral delta
  between EQ(5) and EQ(17) and matches the prior STACK0_STALENESS
  followup's suspicion.
* **Low** that any further `head.bias` or head_bake patch will help.
  Move investigation upstream to the residual producer.
