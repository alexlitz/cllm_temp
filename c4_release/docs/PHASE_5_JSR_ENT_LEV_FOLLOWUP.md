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

---

## 2026-06-12 spec_k=0 re-probe (agent adfb7a9) — root PINNED to L7 operand_gather head 1 (BP/SP→ALU on spurious OP_ENT); no in-scope or discrete-gate fix

A fresh, hook-free, spec_k=0 ground-truth investigation (`tools/probe_groundtruth.py`
`residual_at` / `_final_context`, one model build, ~6 throwaway traces, all removed,
NO production weight change) pins the IMM-in-callee corruption to a **single
mechanism** and proves there is **no declarative gate to fix it** in or out of
the briefed JSR scope.

### Method: three-way isolation
- `CONTROL = IMM 42; EXIT` — first-step IMM, PASSES (AX=42).
- `NONFIRST = IMM 1; PSH; IMM 42; EXIT` — **non-first-step IMM, no call frame**,
  PASSES (AX=42). The critical control the prior agents lacked.
- `SIMPLE = JSR 3; EXIT; NOP; ENT 0; IMM 42; LEV` — callee IMM, FAILS (AX=8 →
  carried → 208/17).

The IMM step in all three is probed at its REG_AX marker across the 37 physical
blocks (logical map: blk6=L6, blk7=L7, blk8/9=L8 AddSub, blk10=L9, blk11=L10).

### What diverges (and what does NOT)
- **The operand IS gathered correctly through L7 (blk7):** all three reach
  `ALU_LO=12(nibble for value-4-lo), ALU_HI=4`. SIMPLE's blk7 operand bands are
  **byte-identical** to the passing NONFIRST.
- **The DISCRETE gates at the IMM-step AX marker are byte-identical** across
  CONTROL / NONFIRST / SIMPLE: `OP_IMM=0, OP_ENT=1, OP_PSH=1, OP_LEA/ADJ/LEV=0,
  MARK_AX=1, HAS_SE=0` after both L6 and L7. So `OP_ENT=1 at the IMM AX marker
  is NORMAL` (confirms the af3960a/a6e7d80 notes) — it is **not** a mis-decode,
  and there is no opcode/marker flag that separates the failing callee IMM from
  the passing non-call IMM.
- **The ONLY input difference is value-level:** `CLEAN_EMBED_HI` = `8(mag 40)`
  in CONTROL (the L5 head-3 **first-step 40× FETCH amplification**, gated
  `HAS_SE=0` first-step only) vs `0(mag 1)` in BOTH NONFIRST and SIMPLE.

### The corruptor (PINNED)
At **L7 (blk7) `_layer7_operand_gather_head_specs` head 1**
(`l7_ops.py:263-287`): Q is gated **positively on OP_ENT/OP_LEA/OP_ADJ**
(`AP(0, BD.OP_ENT, L)`), K attends `MARK_BP`/`MARK_SP`, V copies OUTPUT, O writes
`ALU_LO/HI` at **scale 6.0**. Head 0 (the legit STACK0-byte0→ALU operand gather)
is **SUPPRESSED** by `AP(0, BD.OP_ENT, -L)` whenever OP_ENT is active.

So on the callee IMM step (OP_ENT spuriously 1): head 0 is OFF (the real operand
does not reach ALU at full strength), and **head 1 FIRES and pulls the call
frame's BP/SP register value into ALU** (`ALU_LO=1` at **magnitude 81**, vs the
clean operand's magnitude 1). L8 (blk8) then computes against the polluted ALU,
writing `OUTPUT_HI[1]=+5` and `AX_CARRY_LO[2]=+90`; the emitted AX byte0 = **8**
(the ENT frame constant). That 8 carries forward (L3 head 1) into the LEV-step
AX_CARRY; **L16 `l16_lev_ax_carry_*` (l16_ops.py:256-322) faithfully routes the
carried 8 → OUTPUT** → final 208/17. (L16 LEV routing is CORRECT — it restores
whatever AX it is handed; the bug is purely the upstream callee-IMM corruption.)

### Why CONTROL passes but NONFIRST/SIMPLE differ
- CONTROL: first-step 40× amplification makes the operand magnitude 40, which
  **overpowers** head 1's corruption.
- NONFIRST: no 40× (non-first step), operand magnitude 1 — but there is **no
  frame data** for head 1's BP/SP gather to find, so head 1 is harmless → passes.
- SIMPLE: no 40× AND head 1 finds the live JSR/ENT frame → corruption magnitude
  81 buries the magnitude-1 operand → fails.

### Why NO fix lands (in OR out of the briefed scope)
1. **The corruptor is L7 operand_gather head 1 — explicitly OUT OF SCOPE** (the
   brief reserves L7 heads 0/1 for a parallel agent; this session must not edit
   them).
2. **Even with scope, there is no discrete gate to hang a declarative rule on:**
   the failing callee IMM and the passing non-call IMM have byte-identical
   opcode/marker flags. The only separator is the value-level frame-memory
   content head 1 attends to — not expressible as an FFN/attention gate without a
   new "actual-opcode == IMM (not ENT)" signal the AX marker does not carry.
3. The "make the callee tag OP_IMM not OP_ENT" path is upstream opcode-flag
   decode (L4/L5/L6), also out of scope, and risky: OP_ENT=1 at the AX marker is
   the *shared normal state* the 46 passing tests rely on.

### Candidate fix directions (for a future, broad-tree, in-scope owner)
- **(A) Amplify the legit operand on subsequent steps** so head 0's STACK0→ALU
  (or the L6 IMM-carry-refresh) beats head 1's frame gather — e.g. extend the L5
  head-3 40× amplification (or an L6/L7 equivalent) to non-first IMM steps. HIGH
  RISK: touches the shared operand path used by all non-first-step IMM (memory /
  var / expr clusters). Must be byte-identity-gated against those.
- **(B) Make head 1 require a *real* ENT/LEA/ADJ opcode** (a clean
  actual-opcode-at-AX flag) so it stops firing on the spurious OP_ENT at IMM
  steps. Needs a new opcode-flag dim/decode (L4/L5/L6 surface).
- **(C) Per-step frame-context clear** of STACK0/BP/SP residue before the callee
  body runs, so head 1's gather finds nothing (mirrors NONFIRST's passing case).
  This is the "frame-boundary clear" the brief envisioned; it must be gated on a
  callee-entry context the model does not currently expose discretely.

func_* baseline (canonical runner, spec_k=0): **0 pass** (confirmed for the 48
func_max/min ids reached before the 1200s OOM-split timeout; func_* is 0/150
per all prior runs). simple_function pytest: **expected 0, got 17** (FAIL).
Smoke baseline this session (authoritative `pytest tests/test_smoke.py`):
**46 pass / 5 fail** {mul_basic, simple_function, eq_true, eq_false,
mul_overflow}, 0 xfail — unchanged (no weight edit). Worktree left at the green
baseline; throwaway probes removed.

## 2026-06-12 spec_k=0 — in-scope amplification fix EMPIRICALLY TESTED and FAILS; corruption is in ALU→OUTPUT, not AX_CARRY

Re-probed adfb7a9's root with fresh hook-free spec_k=0 traces (`probe_groundtruth.residual_at`, one build, throwaway probes removed) AND empirically tested the briefed direction-(A) amplification. Findings sharpen and CONFIRM the "no in-scope fix" verdict, and PIN exactly why the amplification cannot land outside `operand_gather`.

### Fresh measurements (CONTROL=`IMM 42;EXIT` pass, NONFIRST=`IMM 1;PSH;IMM 42;EXIT` pass, SIMPLE=`JSR 3;EXIT;NOP;ENT 0;IMM 42;LEV` fail)
- **func_* baseline (canonical, spec_k=0, ids 550-559/575-579/600-604/625-629/650-654/675-679): 1/35 pass.** The ONLY pass is `func_identity_7` (`identity(8)`) — corruption emits the ENT frame constant 8, which *coincidentally equals* the operand. Every other func_* emits a small wrong value (identity→8, add/mul/square/max/min cascade). This is a crisp confirmation: the callee IMM operand is universally replaced by 8.
- **The operand divergence is at physical block 8 (L8), NOT block 7.** At the IMM-step AX marker: blk5/6/7 are byte-identical across NONFIRST/SIMPLE (`ALU_LO[12]=1` = value 42 at magnitude 1). At **blk8** SIMPLE jumps to `ALU_LO[1]=81.1` (the live frame value), NONFIRST stays `ALU_LO[12]=1`. CONTROL holds `ALU_LO[12]=40, ALU_HI[4]=40` all the way through blk8/9/11 — the L5-head-3 first-step 40× amplification survives the L8 ENT machinery and the operand wins. So the deciding quantity is the **ALU magnitude at the AX marker going INTO L8**.
- **NO discrete separator exists** (independently re-confirmed, including the multi-wide opcode-byte bands): diffing EVERY single-flag residual dim (NONFIRST vs SIMPLE) at the IMM-step **PC marker** (blocks 0-5) = **0 diffs**; at the IMM-step **AX marker** (blocks 6-7) = **0 diffs** (block 8's only diff is `MARK_THINKING_END`, a downstream IO artifact). Additionally the 16-wide `OPCODE_BYTE_LO/HI` bands at the IMM-step PC AND AX markers are **identical** between the failing callee IMM and the passing non-call IMM (`OPCODE_BYTE_LO[4]`, `OPCODE_BYTE_HI[11]` for BOTH — itself a JSR-prologue contamination artifact, not a clean IMM=0x01 decode), so even the raw opcode byte cannot key the fix. The failing callee IMM and the passing non-call IMM are byte-identical in every opcode/marker flag and every opcode-byte band. Confirms there is no PC- or AX-marker gate to hang an amplification or a blocker on.

### Empirical in-scope fix attempt (TESTED, reverted) — `_layer6_imm_carry_refresh` extended to OP_ENT
Direction (A) was tested at its lowest-risk, most-benign incarnation: the L6 IMM AX_CARRY refresh (`_layer6_imm_carry_refresh_rules`, l6_ops.py:599) is gated `OP_IMM` and never fires on the callee IMM (OP_IMM=0 there). Adding `("OP_ENT", 0.2)` as an additive opcode tilt (byte-identity-verified against the updated legacy `_set_layer6_routing_ffn`; the existing OP_IMM path is unchanged because OP_ENT=0 on every pure-IMM step) makes the refresh fire on the callee IMM AX row and restores the genuine immediate into AX_CARRY at L6. **Result: simple_function STILL `got 8` (no change); the other 6 ENT/ADJ/LEA/LEV/IMM/PSH smoke tests still pass.**
- **Why it fails:** AX_CARRY is the wrong dim. The emitted AX byte is decoded from OUTPUT, which L8 writes from **ALU** at the AX marker. The L7 operand_gather head-1 BP/SP frame leak pollutes **ALU** (→81), and L8's OP_ENT-gated `ent_lo`/`ent_borrow` machinery computes OUTPUT from that polluted ALU. Refreshing AX_CARRY at L6 (before L8) does not touch the ALU→OUTPUT path that L8 owns. Change reverted (zero positive effect, non-zero maintenance/risk).

### Why NO in-scope amplification can win (the precise wall)
The deciding quantity is **ALU at the AX marker**, and ALU-at-AX is written only by (1) L7 `operand_gather` heads 0/1 (FORBIDDEN) and (2) L8. Any amplification must land in ALU at magnitude >81 **after** L7 fires (L7 > L6, so L6/L5 writes are overwritten by head 1). The only in-scope place left is an L8 FFN write of ALU on `MARK_AX + OP_ENT + FETCH`, but that fires identically on a **real ENT step** (`test_lea_basic` = `ENT; IMM 0; LEA 2; EXIT`), whose AX marker is byte-identical (OP_ENT=1, MARK_AX=1, OP_IMM=0) and whose ALU legitimately holds SP for the ENT SP-decrement — injecting FETCH there corrupts ENT. There is no discrete dim to separate the two. CONTROL only wins because its operand is amplified to 40 by the **first-step-only** L5 head-3 path (`HAS_SE=0`), which provably cannot be extended to a non-first step without a callee-context signal the model does not expose.

### Minimal change that WOULD fix it (requires the forbidden surface)
Inside `_layer7_operand_gather_head_specs` (l7_ops.py:263-287), head 1 attends MARK_BP/MARK_SP and copies their **OUTPUT** into ALU at scale 6.0, gated `AP(0, BD.OP_ENT, L)`. The minimal fix is to make head 1 (and the head-0 un-suppression) key on a **real-opcode-at-AX == ENT/LEA/ADJ** signal instead of the marker-scheme `OP_ENT` (which a callee IMM also carries). That requires a new clean opcode-flag dim decoded at L4/L5/L6 (`actual_opcode_at_AX`) — a multi-commit opcode-decode change — and then head 1's Q gate `AP(0, BD.OP_ENT, L) → AP(0, BD.ACTUAL_OP_ENT, L)` plus head 0's suppression `AP(*, BD.OP_ENT, -L) → AP(*, BD.ACTUAL_OP_ENT, -L)`. With that, on the callee IMM (ACTUAL_OP_ENT=0) head 0 fires (real operand → ALU) and head 1 stays off (mirrors NONFIRST's passing case); on a real ENT it behaves as today. This is THE single fix and it is squarely inside the reserved operand_gather + opcode-decode surface.

func_* after (canonical, spec_k=0, same 35-id slice): **1/35 (unchanged).** simple_function pytest: **expected 42, got 8 (FAIL, unchanged).** Smoke subset {simple_function,imm,add_basic,sub_basic,lea,ent,adj,lev,psh}: 6 pass / 1 fail (only simple_function) — NO regression. Worktree left at the green baseline; NO production weight change; throwaway probes removed.
