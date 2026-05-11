# Phase 4 BZ/BNZ Instrumentation Findings (2026-05-11)

Deep instrumentation per Wave A.4's recommended next step. Runs the pure_neural
model on the same bytecode used by `test_bz_not_taken[1]`, with step 0
teacher-forced to the "correct" oracle state (PC=10, AX=1 after IMM 1) so
that step-1 PC marker activation can be inspected independently of step 0
correctness.

## TL;DR

* The user-supplied hint that **OP_BZ activation reaches 15 instead of 5 at
  step 1 PC marker is WRONG.** Measured value at step-1 PC marker:
  `OP_BZ = 5.000`, `OP_BNZ = 5.000`, `OP_JMP = 5.000` (all three branch-class
  opcode flags are decoded to ~5 by the L5 all-step opcode decode at PC
  marker — vm_step.py:4146-4163).
* The L6 BZ AX-passthrough units (vm_step.py:4917-4933) are **not** the
  corruption path. With OP_BZ=5 and the existing -S MARK_PC blocker, those
  units do **not** fire at the PC marker (up = 500+0-100-400 = 0 → silu≈0).
* The actual corruption is **upstream of L6**, in **L3 PC carry-forward**.
  After L3 attention at step-1 PC marker:
  `EMBED_LO[0] = 1.0` (one-hot index 0)
  even though the source position (step-0 PC byte 0, position 37) correctly
  carries `EMBED_LO[10] = 1.0` (matching the teacher-forced PC=10 in step 0).
* Consequence: the L3 PC-increment FFN unit fires at `(EMBED_LO[k=0], +8)`
  and writes `OUTPUT_LO[8] = +1.0` (predicting PC byte 0 = 8). Visible in
  the per-layer trace as `L3 delta: dOUT_LO[8]=+1.000 dOUT_HI[0]=+1.000`.
* That `OUTPUT_LO[8]=1, OUTPUT_HI[0]=1` is the **fall-through PC the model
  predicts after BZ at step 1** — and equals 8, not 18 (the correct
  fall-through `PC=10 + INSTR_WIDTH=8 = 18 = 0x12`). Hence the
  not-taken case "hangs at BZ" or jumps to the wrong PC; the test fails.

## Root cause: L3 PC carry-forward attention favors self-attention over source

L3 attention head 0 (`Primitives.carry_forward_attention`) is supposed to
copy the previous step's PC byte 0 EMBED_LO/HI to the current step's PC
marker. Q[base, MARK_PC] = 15, K[base, L1H1[PC]] = 15, K[base, L1H0[PC]] = -15.

* Source position (step-0 PC byte 0, dist 34 from step-1 PC marker):
  `L1H1[PC]=1, L1H0[PC]=0, EMBED_LO[10]=1`. Score contribution
  `Q·K = (15·1) - (15·0) = 15`; with HD=64, scaled score
  `15²/8 ≈ 28.1`. ALiBi penalty `slope=0.5 × dist=34 = 17`. Plus the
  anti-leak gate term (Q[gate, MARK_PC] + Q[gate, CONST]=-L/2; K[gate, CONST]
  = L): at this position, MARK_PC=0 (this is a byte token, not a marker), so
  Q[gate] = `-L/2`, K[gate] = `L` → contribution `-L²/(2·8) ≈ -14`. Total
  ≈ 28.1 − 17 − 14 ≈ **−2.9**.
* Self position (step-1 PC marker itself, dist 0): `MARK_PC=1, L1H1[PC]=1,
  L1H0[PC]=1, EMBED_LO=zeros`. Main head score = 0 (because L1H1 and L1H0
  cancel). Anti-leak gate at this position: MARK_PC=1, so Q[gate] = L − L/2
  = L/2 → contribution `+L²/(2·8) ≈ +14`. ALiBi: 0. Total ≈ **+14**.

Softmax thus puts most weight on the PC-marker itself (which has EMBED=0)
and ignores the source position. Empirical attention weights from the
instrumented trace:

```
L3 attn head 0 (PC carry-fwd) at pos 71 - top 5 attended positions:
  pos=71 tok=257 (PC marker)  weight=0.393  EMBED_LO[0]=1.00  L1H1[PC]=1.00 L1H0[PC]=1.00
  pos=70 tok=262 (MEM marker) weight=0.239  EMBED_LO[0]=0.00  L1H1[PC]=0.00 L1H0[PC]=0.00
  pos=69 tok=0   (mem byte)   weight=0.145  EMBED_LO[0]=1.00  ...
  pos=68 tok=0                weight=0.088  EMBED_LO[0]=1.00  ...
  pos=67 tok=0                weight=0.053  EMBED_LO[0]=1.00  ...
```

All top-attended positions are near the query (pos 71) and have either zero
EMBED or EMBED_LO[0]=1 (because the value byte is 0). The intended source
(pos 37, token=10, EMBED_LO[10]=1) does not appear in the top 5.

## Why the diagnosis hint missed this

The Wave A.4 hint said "OP_BZ reaches 15 at PC marker". Measurement here
shows OP_BZ = 5, exactly as the L5 all-step PC decode is configured to
write. The proposed Options A/B in `PHASE_4_BZ_BNZ_DIAGNOSIS.md` both
target the L6 BZ AX-passthrough threshold, but that's a downstream
symptom, not the cause — and at OP_BZ=5 the L6 BZ AX-passthrough
correctly does **not** fire at the PC marker.

## Layer-by-layer summary (step-1 PC marker, teacher-forced step 0)

```
[after embed] OP_BZ=0.000   MARK_PC=1.00 EMBED_LO=∅
[L0 after ffn] OP_BZ=0.000  MARK_PC=1.00  HAS_SE=0       (L0 attn writes H0..H7)
[L1 after ffn] OP_BZ=0.000  MARK_PC=1.00  HAS_SE=+1.00   (L1 attn writes L1H0..2 + HAS_SE)
[L2 after ffn] OP_BZ=0.000  MARK_PC=1.00  HAS_SE=+1.00
[L3 after attn] EMBED_LO[0]=1.0          ← BUG: should be EMBED_LO[10]=1.0
[L3 after ffn ] dOUT_LO[8]=+1.000        ← incorrect PC prediction (8 instead of 18)
[L5 after ffn ] OP_BZ=+5.000  OP_BNZ=+5.000  OP_JMP=+5.000   (all three decoded)
[L6 after attn] CMP[4]=+0.96  CMP[5]=+1.14  (AX_*_IS_ZERO; firing as if AX=0)
[L6 after ffn ] no L6 unit fires at PC marker writing OUTPUT (BZ AX-passthrough is OK)
[L10 after ffn] dOUT_HI[0]=+387.115      ← large write, but to index 0 (same as default)
[end of forward] OUTPUT_LO[8]=1, OUTPUT_HI[0]=1 — predicting PC byte 0 = 8
```

The L6 BZ-PC-override units (vm_step.py:5136-5195) require
`MARK_PC + CMP[2] + CMP[4] + CMP[5] > 3.5`. At PC marker the values are
`MARK_PC=1, CMP[2]≈0, CMP[4]≈1.4, CMP[5]≈1.1` → sum ≈ 3.5, right at the
threshold. With CMP[2] ≈ 0 (OP_BZ relay via L6 head 4 evidently not
firing strongly), these units do **not** fire and do **not** redirect PC.

## CMP[4]/CMP[5] firing erroneously

Even though AX=1 (teacher-forced), `CMP[4]=AX_LO_IS_ZERO=+1.44` and
`CMP[5]=AX_HI_IS_ZERO=+1.14` at step-1 PC marker. This says the model
thinks AX=0 at this position. This is **another upstream bug** in the
AX-zero-detection or AX carry-forward path at the PC marker; it would
also cause `test_bz_taken` failures.

## Recommended next steps (not attempted in this pass)

1. **Fix L3 PC carry-forward attention**: increase the gain on `L1H1[PC]`
   in the key projection, OR reduce ALiBi slope for head 0 specifically,
   OR strengthen the self-suppression so the query position never wins
   the softmax. Concrete options:
   * Bump `Primitives.carry_forward_attention`'s `L` from 15 to ~30 for
     the main head so the source-side score (≈ L²/8 = 112) dominates over
     the gate term (still ≈ L²/(2·8) = 56) and ALiBi (1× dist).
   * Add a **stronger self-position negative**: at the PC marker itself
     where `MARK_PC=1 AND L1H0[PC]=1`, force K to be very negative — e.g.
     reuse a dedicated dim like `IS_MARK` weighted at `-L·5`.
   * Lower the ALiBi slope from 0.5 to 0.0 for head 0 (PC carry needs to
     attend across ~36-token step boundaries).

2. **Fix CMP[4]/CMP[5] at PC marker**: trace L6 attention head 4
   (`_set_bz_bnz_relay`) to see why AX_*_IS_ZERO is firing when AX=1.
   Likely the relay attends from PC marker to AX marker but reads the
   wrong AX value, or the comparison is reading AX_CARRY which is zero
   on first step.

3. **Note on Phase 1 tests**: many Phase 1 tests in
   `test_pure_neural_pc.py` are **not** xfail-marked but currently
   **fail** in pure_neural mode. Even the trivial
   `TestPureNeuralSingleInstruction::test_imm_then_exit` (single IMM 5
   + EXIT) returns 67372036 = 0x04040404 instead of 5 — the model
   emits all-0x04 byte tokens regardless of program. Phase 1 itself
   is currently broken at a more fundamental level than the Phase 4
   BZ/BNZ gap implies; the L3 PC carry-forward bug surfaced here is
   one of (at least) several pure_neural-mode regressions.

   Recommended: triage `test_imm_then_exit` (single-step IMM) first.
   If even that doesn't work in pure_neural mode, BZ/BNZ tests can
   only be diagnosed once the underlying generation degeneracy is
   fixed.

## Instrumentation script

The instrumentation lives at
`c4_release/debug_archive/trace_bz_not_taken.py`. It:

1. Builds a `pure_neural` runner (same as the test fixture).
2. Teacher-forces step 0's output (PC=10, AX=1) so step-1 PC marker
   activation can be inspected without depending on step 0 being
   correctly executed by the network.
3. Runs the model layer-by-layer and prints, at the step-1 PC marker
   position: OP_BZ/OP_BNZ/OP_JMP, MARK_PC/AX, HAS_SE, CMP[2..5],
   OUTPUT_LO/HI argmax+value, plus per-layer deltas to these dims.
4. Includes focused L3 FFN unit-level trace (top units by |OUTPUT
   contribution|, with W_up/W_gate inputs printed) and a manual
   attention-score computation for L3 head 0.

Cost: ~70s per run (single forward pass on a ~110-token context).

## Files referenced

* `c4_release/neural_vm/vm_step.py:3005-3294` — `_set_layer3_ffn` (PC
  default + increment + carry correction)
* `c4_release/neural_vm/unified_compiler/migrated_ops.py:2013-2081` —
  `make_layer3_carry_forward_attn_op` (heads 0-6)
* `c4_release/neural_vm/unified_compiler/primitives.py:75-141` —
  `Primitives.carry_forward_attention` (the broken head 0)
* `c4_release/debug_archive/trace_bz_not_taken.py` — instrumentation
