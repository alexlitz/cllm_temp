# Phase 5 JSR/ENT/LEV Follow-up: Baseline Still Broken

**Date:** 2026-05-11
**Branch:** `p5-jsr-ent-lev`
**Tests targeted:** 5 remaining xfails in `tests/test_pure_neural_jsr_ent_lev.py`

## TL;DR

The premise "`test_jsr_then_lev_simple` + `test_lev_returns_to_caller`
already XPASS, the remaining 5 just need targeted fixes" is **incorrect**
on the current `main` HEAD (commit `b15e428`). Both supposedly-passing
tests still fail in pure_neural mode:

- `test_jsr_then_lev_simple`: returns `4160946183` (`0xF7F7F7F7`),
  expected `7`.
- `test_lev_returns_to_caller`: program never reaches the caller's
  `EXIT`; the test hits its 200s pytest-timeout instead of returning `0`.

This means **all 7 tests in `TestPureNeuralJSRLEVSimple` /
`TestPureNeuralJSRSemantics` / `TestPureNeuralENTSemantics` /
`TestPureNeuralLEVCornerCases` are still in the broken cascade**, not
just the 5 marked `@pytest.mark.xfail`. The xfail removal in
commit `cad2a8f` ("Phase 5: remove xfail from 2 confirmed JSR/LEV
xpasses") was performed against a stale state or with a different
fixture cache; the regression is reproducible on a fresh worktree of
HEAD.

## How I confirmed

```bash
git -C /home/alexlitz/Documents/misc/c4_release worktree add /tmp/p5-jsr-ent-lev -b p5-jsr-ent-lev main
cd /tmp/p5-jsr-ent-lev
CUDA_VISIBLE_DEVICES=0 timeout 180 python -m pytest \
    c4_release/tests/test_pure_neural_jsr_ent_lev.py::TestPureNeuralJSRLEVSimple::test_jsr_then_lev_simple \
    -v --tb=short --timeout=150
```

Result: `FAILED ... assert 4160946183 == 7`.

Same failure reproduces at `cad2a8f` (the commit that removed the
xfail) and at the more recent `6578c20`, `593c182~1`, etc. — i.e. the
test has been silently regressed for multiple commits.

## Probing the failure with a one-shot harness

A direct script (`AutoregressiveVMRunner(trust_neural_alu=True,
pure_neural=True)`) shows the failure shape clearly:

```
Test IMM 7 EXIT:            result=0x7         (correct baseline)
Test JSR 2 EXIT (skip ENT/LEV)   result=0xf8030063  (byte 0 correct = 99; bytes 1-3 garbage)
Test JSR 2; NOP; ENT 0; EXIT     hangs (model never emits EXIT-terminating output)
```

Interpretation:
- IMM→EXIT roundtrip works (Phase 1 baseline OK).
- JSR-then-EXIT preserves AX byte 0 (`0x63 = 99`) but bytes 1–3 are
  contaminated with what look like SP and return-address bytes (`0xF8 =
  byte 0 of `0xFFF8 = SP - 8`, `0x03` from the JSR target PC).
- Adding an ENT after JSR breaks termination entirely.

So the failure on `test_jsr_then_lev_simple` is not "JSR is fine; LEV
breaks" — it's that JSR itself **clobbers AX bytes 1–3** with
return-address arithmetic, and ENT then derails the autoregressive
loop. The full LEV path is downstream of both.

## Where the contamination comes in

`vm_step.py` lines 3445–3487 (`_set_layer9_alu`, JSR section) explicitly
adds 32 units to preserve AX through JSR by routing `AX_CARRY → OUTPUT`
at the AX marker:

```python
ffn.W_up[unit, BD.OP_JSR] = S
ffn.W_up[unit, BD.MARK_AX] = S
ffn.W_up[unit, BD.MARK_SP] = -S * 8  # block at SP marker
...
ffn.W_gate[unit, BD.AX_CARRY_LO + k] = 1.0
ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
```

Empirically this routing fires for byte 0 but not for bytes 1–3 of AX
— evidence: the observed AX value at EXIT is `0xf8030063`, low byte
matches, others get SP/PC bytes from neighboring marker positions.

The most plausible root cause is that `AX_CARRY_HI` at the AX marker
during a JSR step does not actually contain the previous step's AX byte
0 / high nibbles — it's getting clobbered by the L6 head 7 routing
(which writes PC-OUTPUT into AX_CARRY at the STACK0 marker for return-
addr push) bleeding via softmax leakage at higher byte indices, or by
L7 SP-gather firing on `OP_JSR` (head 1 in `setup_helpers.py`
~L679–713) without the MARK_AX gate isolating it.

## Recommended path forward

Spending more time on the 5 specifically-tagged xfails is not the right
investment until the basic JSR roundtrip is fixed. Concretely:

1. **Re-xfail the 2 supposedly-passing tests.** They are failing
   today. Reverting commit `cad2a8f` (or re-adding the xfail markers
   with an updated reason) avoids misleading future contributors.
2. **Trace AX_CARRY at the AX marker through a JSR step.** The trace
   script `scripts/debug/trace_jsr_lev_simple.py` already wires up the
   teacher-forced 5-marker dump used by D4 — extend it to also print
   `AX_CARRY_LO/HI` at the AX marker at layers L5..L9 to identify which
   layer overwrites the carry.
3. **If AX_CARRY at AX marker is the leak,** the candidate fixes are
   either (a) tighten the L6 head 7 anti-leakage gate (it currently
   uses `Q[MARK_STACK0]=L6` with a `GATE=33` gate — but no negative
   gate against `MARK_AX`), or (b) introduce a per-byte block of L6
   head 7's V→`AX_CARRY_HI` write at the AX marker.
4. **Then attempt ENT.** The ENT first-step SP byte 0/1-3 units in
   `vm_step.py` ~L3994–4080 are not the issue — they fire only when
   `HAS_SE` is gated correctly. The likely ENT regression is upstream:
   ENT step's PC marker prediction itself goes wrong because the prior
   JSR step left the residual stream malformed.

## What was tried in this session

- Bisected `test_jsr_then_lev_simple` across `b15e428` → `593c182~1` →
  `28978ee~1` → `cad2a8f`. All produce the same garbage output. The
  regression predates `cad2a8f`.
- Verified IMM/EXIT works (`0x7`) — Phase 1 is green.
- Verified the L9 ALU JSR-preserve routing exists at the source level
  (`vm_step.py:3445–3487`) and reads `AX_CARRY`; it is not missing,
  just not effective at higher byte indices.
- The `compile_full_vm` bake reports `Total blocks: 17 -> 26` and 30.5%
  FFN units retained — the model is built successfully, so the failure
  is in the trained weight values / dim routing, not a build error.

## Files relevant to the fix (when someone takes it up)

- `c4_release/neural_vm/vm_step.py` (`_set_layer9_alu` JSR section,
  L3445-3487; `_set_layer8_alu` ENT SP-decrement L3986-4080)
- `c4_release/neural_vm/weight_modules/function_calls.py`
  (`_set_jsr_weights`, `_set_ent_weights`, `_set_lev_weights`)
- `c4_release/neural_vm/setup_helpers.py` (L7 head 1 SP gather at AX
  marker, L679-713)
- `c4_release/scripts/debug/trace_jsr_lev_simple.py` (existing trace
  harness, extend it)
