# Faithful decode for the interp-oracle gate — AX-high-byte + framing-drift / cross-step

`tools/interp_oracle_gate.py` is the CPU-only interpreter-vs-oracle attribution
gate. Before this work it was HIGH-confidence only for **PC** and **AX byte 0**;
AX bytes 1..3 and any multi-byte / cross-step case returned a LOW-confidence
`FAIL?` "flat-decode" verdict (often a teacher-forcing artifact). This change
makes the gate's decode FAITHFUL — and HIGH-confidence (rule-attributed) — for
the **AX high bytes (1..3)** wherever the single forward provably matches the
neural `full_trace` runner, and adds a SOUND **CROSS-STEP** class that flags
(never over-claims) the autoregressive framing-drift cases the single forward
cannot reproduce, pointing them at the GPU `--faithfulness-check` to resolve.

The interpreter's FORWARD MATH was already byte-for-byte faithful (validated by
`tools/faithful_interpreter_validate.py`). The gap this closes is in the
gate's **DECODE** (turning per-position argmax into a per-step (PC, AX) verdict
that matches production), not the forward.

## The decode model (how production decodes, and how the gate now mirrors it)

A VM step is 35 tokens (`DraftVM.draft_tokens`):
`REG_PC marker + pc[0..3], REG_AX + ax[0..3], REG_SP + sp[0..3], REG_BP + bp[0..3],
STACK0 + st0[0..3], MEM + addr[0..3] + val[0..3], STEP_END/HALT`. The full_trace
criterion (`run_1096_canonical.py --criterion full_trace`, the authority) checks
only the decoded `(PC, AX)` per step.

Three facts of the production decode (`neural_vm/batched_pure_neural.py`) drive
the gate's model:

1. **Markers are re-anchored, not read.** Production's `_ff_check_new_steps`
   decodes a step's register from its slice, and the step boundary itself is
   re-anchored via the Python STEP_END dispatch (`_step_one`). So the gate
   decodes the four register VALUE bytes at the FIXED step offsets
   (PC@1..4, AX@6..9) — `_decode_reg_fixed` — and ignores the model's
   prediction of the marker TOKEN itself. **This is what makes the AX high
   bytes (1..3) decodable from the single forward**: their value tokens sit at
   fixed offsets 7/8/9 and the model's argmax there over the teacher-forced
   tape is its own emission for any step not yet poisoned.

2. **`_UNSAFE_OFFSETS` (26..33, the MEM addr/val bytes) are DraftVM-trusted.**
   The embedding's MEM-metadata injection makes the flat argmax disagree with
   the DraftVM there for EVERY program, so production never reads a model
   correction at those offsets. The gate mirrors this (these offsets are not a
   poisoning correction).

3. **Production is AUTOREGRESSIVE.** When the model emits a register-VALUE byte
   that differs from the oracle tape (a "value correction"), production appends
   the WRONG byte and re-syncs its DraftVM from the model's now-wrong register
   state (`_sync_draft_vm`). EVERY later step then reads a poisoned context the
   single teacher-forced forward never sees — under teacher forcing the model is
   fed the CORRECT tape and "recovers". This is the irreducible framing-drift /
   cross-step gap.

### The soundness guard (`_value_correction_step`)

From fact 3: the single-forward per-step `(PC, AX)` decode is FAITHFUL to
production for every step **at or before the first register-VALUE-byte
correction**, and UNRELIABLE for any step strictly after it. So the gate
computes the first step at which the model's emitted value byte (any of
PC/AX/SP/BP/STACK0, offsets 1-4/6-9/11-14/16-19/21-24) diverges from the oracle
tape, and:

* a **FAIL** at step `s` is **HIGH-confidence** iff `s <= vcorr` (or `vcorr` is
  `None`) — attributed (incl. AX bytes 1..3);
* a FAIL at step `s > vcorr` is **CROSS-STEP** — flagged, not attributed;
* a clean single-forward "pass" with `vcorr` before the last step is also
  **CROSS-STEP** (production could fail a later poisoned step the forward can't
  see — the var/func cluster: neural fails @ step 4, the single forward
  "passes"). A correction at the LAST step (`vcorr >= n_steps-1`) cannot poison
  any later `(PC, AX)` check, so it stays a faithful PASS (the edge_literal
  single/last-step cluster).

Marker-token mispredictions (offsets 0/5/10/15/20/25/34) and the MEM bytes do
NOT count — production re-anchors / trusts them. **Only register VALUE bytes
poison.**

**Soundness contract — measured 0 false-trusts.** Over the var/func/expr/edge
validation sample, "guard says faithful" ⟹ "the gate's verdict matches the
neural full_trace runner" with **0 violations** (the guard never over-trusts).
This requires ALL register-value offsets (PC/AX/SP/BP/STACK0): a PC+AX-only
guard gives **34+ false-trusts** because the var/func PC at step 4 is poisoned
by a BP/SP/STACK0 correction at step 1 — and a PC+AX-only PASS-path relaxation
was measured to FALSELY PASS 5 var programs that neural fails @ step 4, so it
was rejected. The price of soundness is conservative under-claiming: some genuine
passes whose only corrections are SP/BP/STACK0 (e.g. the eq/lt/.../boolean smoke)
are flagged CROSS-STEP rather than PASSed. That is the safe direction — the gate
never claims a PASS (or an attributed FAIL) it cannot back; resolve the flagged
ones with the GPU `--faithfulness-check`.

### Cached faithful forward (`FaithfulForwardCache`)

`faithful_full_forward` re-extracted the per-block head specs on every call
(~2.2M `nonzero()` calls, ~4.5 s/forward, >90 % of its runtime, independent of
context length). `FaithfulForwardCache` extracts the head specs + dense FFN
weights ONCE and reuses them — **byte-identical argmax** to
`faithful_full_forward` (validated at S=70/175/350), at **0.4-1.1 s/forward
(5-13x faster)**. This keeps the per-program decode + attribution cheap.

## Before / after gate-vs-neural match rate (per cluster)

The authority is `tools/run_1096_canonical.py --criterion full_trace` (GPU,
spec_k forced to the production decode). Match = same verdict (PASS/FAIL) AND
same first-divergence step (and, for the AX-only cluster, same decoded AX).

Authority for these counts: `neural_authority.json` from
`tools/run_1096_canonical.py --ids 250-274,850-874,1031-1045,550-553
--criterion full_trace` (GPU). The gate's HIGH-confidence verdicts are validated
against it with **0 soundness violations** (a HIGH-confidence PASS/FAIL never
disagrees with neural).

| Cluster (ids) | n | neural | BEFORE gate (LOW-conf `FAIL?` / fixed-35) | AFTER gate (sound full-offset guard) |
|---|---|---|---|---|
| **edge_literal** (1031-1045) | 15 | 5 PASS, 10 FAIL@0 (AX byte 1) | all decoded but the 10 AX-byte-1 fails were **LOW-confidence `FAIL?`** (not trusted/attributed) | **10 FAIL@0 AX[1] now HIGH-confidence + attributed** (match neural exactly, 10/10). Of the 5 neural PASSes: 2 stay PASS, 3 (2563/2417/2212 — a STACK0 correction at a non-last step) are conservatively flagged CROSS-STEP for GPU confirm (sound under-claim, not a false PASS). |
| **expr_mul_div** (850-874) AX-only | 14 | AX byte 1-2 fails @ steps 3-6 | LOW-conf `FAIL?` or ALU-OPAQUE; **0/14 step-matched** the neural divergence (fixed-35 desynced to step 1) | the re-anchored decode would step-match **9/14**, but the sound full-offset guard correctly flags all 14 as **CROSS-STEP** (each has a register-VALUE correction at step 1 — the decode is downstream of poisoning). The re-anchored value IS faithful for the 9 (it agrees with neural), but the gate won't HIGH-claim a verdict it can't certify is poison-free. |
| **var_simple** (250-274) | 25 | all neural FAIL@4 (PC) | fixed-35 gave step 7/8 or spurious PASS — **0/25 matched** | **all 25 correctly flagged CROSS-STEP** (var PC framing-drift is irreducibly autoregressive — the single forward "recovers" under teacher forcing; proven 0/14 even with re-anchoring). The old gate falsely PASSed 5 of these (262/263/267/271/272); the guard now flags them. |
| **func_identity** (550-553) | 4 | all neural FAIL@4 (PC) | fixed-35 gave FAIL@8 — **0/4 matched** | **all 4 correctly flagged CROSS-STEP** |

Summary of the AX-HIGH-BYTE target (the 24 AX-only neural fails across
edge_literal + expr_mul_div):

* **BEFORE:** single-forward fixed-35 step-match **10/24**, AX-value-match
  **10/24** (the 10 edge_literal; expr desynced to step 1). All 24 were emitted
  as LOW-confidence `FAIL?` — NOT attributed, NOT trusted.
* **AFTER (re-anchored decode + sound guard):** the **10 edge_literal AX-byte-1
  fails are now HIGH-confidence + attributed and match neural exactly (10/10)** —
  the headline win, since these are the single-step AX-high-byte bugs the task
  targeted. The re-anchored decode is ALSO faithful for the 9 step-matching
  expr_mul_div cases (it agrees with neural), but the sound guard flags them
  CROSS-STEP because each is downstream of a step-1 value correction — the gate
  will not HIGH-claim a multi-step AX value it cannot certify is poison-free.
* **The headline soundness result:** over the full target slice, the gate's
  HIGH-confidence verdicts (PASS + attributed FAIL) match the neural full_trace
  runner with **0 violations**. The old gate's `FAIL?` class mixed real bugs with
  teacher-forcing artifacts AND (separately) the old PASS class falsely passed 5
  var programs; both are now sound.

The PC framing-drift cluster (var/func, 29 neural fails) is **not** single-forward
faithful (proven: 0/14 even with re-anchoring — these need autoregression), and
the gate now FLAGS all of them CROSS-STEP rather than emitting the old misleading
step-7/8 verdict or a false PASS.

### Net effect on the gate's confidence classes (target slice 250-274,
850-874, 1031-1045, 550-553)

* **BEFORE:** 10 PASS, 4 HIGH-conf FAIL, **38 LOW-conf `FAIL?`** (the unusable
  class), 17 ALU-OPAQUE → gate authoritative on 14/69, 38/69 untrusted.
* **AFTER:** the AX-high-byte cases that ARE single-forward faithful become
  HIGH-confidence attributed FAILs; the rest become a clearly-labelled,
  GPU-resolvable CROSS-STEP class. No verdict the gate marks HIGH-confidence
  disagrees with neural (the soundness guard).

## Can we now debug var / expr / framing-drift entirely on CPU via the gate?

**Partly, and now HONESTLY (the gate is SOUND about exactly where):**

* **Single-step AX-high-byte bugs (edge_literal AX byte-1): YES, fully on CPU.**
  The gate decodes the AX high bytes faithfully, attributes the owning rule, and
  provably matches the neural full_trace (10/10 edge_literal AX-byte-1 fails).
  These were the LOW-confidence `FAIL?` "untrusted" class before; they are now
  CPU-authoritative + rule-attributed. This is the clean, sound AX-high-byte win.

* **Multi-step AX-high-byte (expr AX byte 1-2) and PC framing-drift (var/func):
  flagged precisely, not HIGH-claimed, on CPU.** Every one of these has a
  register-VALUE correction at an early step, so the decode is downstream of
  production's autoregressive context poisoning, which a single teacher-forced
  forward provably cannot reproduce (the model "recovers" under teacher forcing;
  measured 0/14 for var/func even with re-anchoring). The re-anchored AX value
  the gate computes for the expr cases DOES agree with neural for 9/14, but the
  gate will not HIGH-claim a verdict it cannot certify is poison-free — soundness
  over recall. It instead FLAGS them CROSS-STEP (with the exact poisoning step)
  and routes them to the GPU `--faithfulness-check` (the real production decode).

So the gate becomes the trustworthy CPU authority for the **single-step
AX-high-byte cluster** (the headline win) and a **sound triage** for the
multi-step / framing-drift clusters: it tells you precisely which programs are
single-forward-faithful (HIGH-confidence, attributed) vs which are downstream of
cross-step poisoning (CROSS-STEP, GPU-resolve) — with **0 false-trusts**. This
replaces the old all-LOW-confidence `FAIL?` that mixed real bugs with
teacher-forcing artifacts AND the old PASS class that falsely passed 5 var
programs. The "bulk" the gate can now own on CPU is the single-step AX-high-byte
+ PC-byte-0 + AX-byte-0 set; the genuinely autoregressive framing-drift remains a
GPU-confirm class, but is now correctly identified instead of mis-verdicted.

## Residual coverage gaps (separate from the decode gap)

1. **The 4 imperative ALU blocks** (AddSub5StageBlock, FlattenedALUMul,
   FlattenedDivMod, ALUShiftComposite) have no IR rule form — the interpreter
   runs the REAL composite block, which is not bit-certified vs production's
   imperative ALU. Any program whose path touches ADD/SUB/MUL/DIV/MOD/SHL/SHR is
   **ALU-OPAQUE** (the gate declines to judge). This is an INTERPRETER COVERAGE
   gap (task #230), not a decode gap.

2. **The PC framing-drift / cross-step class is a true production-decode
   property**, not an interpreter math bug. The interpreter's FORWARD is faithful
   there (the math is right); the gate simply cannot drive the AUTOREGRESSIVE
   poisoning from a single forward cheaply on CPU. A from-scratch CPU
   autoregressive replica was prototyped but is (a) prohibitively slow on CPU
   (~minutes/program, the per-token re-decode after a correction is the cost) and
   (b) not byte-faithful to production's full spec-loop (KV-cache / windowing /
   accept-correct-resync interplay) — so it was NOT shipped to avoid a tool that
   gives wrong answers. The authoritative resolver for this class is the GPU
   `--faithfulness-check`, which runs the real production decode.

## Validation tools

* `tools/run_1096_canonical.py --ids <...> --criterion full_trace` — the GPU
  authority the gate is measured against.
* `tools/interp_oracle_gate.py --ids <...>` — the extended gate (CPU).
* `tools/interp_oracle_gate.py --faithfulness-check N` — interp-vs-real-neural
  confirm on a free GPU; the cross-step resolver.
* `tools/faithful_interpreter_validate.py` — proves the forward math (the
  dependency this gate trusts) is byte-for-byte == the model.
