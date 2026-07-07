# PROJECT: func-cluster step-0 JSR BP byte-3 leak (`C4_JSR_BP_BYTE3_CLEAR`, DEFAULT-OFF)

Task #428. Base golden `f725c06e` (campaign default; STEP_TOKENS=30). Flag
DEFAULT-OFF, flag-OFF byte-identical to `f725c06e` (verified twice via
`tools/_isa_golden_hash.py`, 42117 FFN units unchanged).

## 0. TL;DR

Every program whose `main()` calls a function starts with a **JSR** at step 0
(`step0_op=0x03`): func_min/max/identity/add, var_three, loop_sum, gcd,
nested_quad, rec_sum — essentially the whole non-trivial corpus. On that step-0
JSR the caller's BP is `0x00010000` (byte-2=0x01, byte-3=0x00) but the model
emits **BP byte-3 = 0x01** → BP decodes `0x01010000`. This is the **FIRST
value-byte correction** (the cross-step autoregressive poisoning point,
`interp_oracle_gate._value_correction_step`, `vcorr=0`) for the entire func
cluster, and it poisons every downstream frame-relative read
(`mem[BP-8]`/ENT/LEV/LI). The func cluster's later step-11/12/13 divergences
that earlier docs chased are all DOWNSTREAM of this step-0 poisoning
(CROSS-STEP, teacher-forcing artifacts).

The just-landed `C4_LOOP_LEA_OPLEA_GATE` (the +44 0xE8-slam fix) is a genuine
no-op here — a different mechanism, a different step.

## 1. Attribution (BUILT dims, campaign config, spec_k=0)

Tools: `tools/_probe_funcmin_attr.py` (gate classify → all func FAIL with
`vcorr=0`), `tools/_probe_funcmin_vcorr.py` (per-register per-byte decode → the
vcorr byte is **BP byte-3**, `got=1 want=0` at step 0),
`tools/_probe_bp_b3_logit_attr.py` (LM-head logit attribution),
`tools/_probe_bp_byte3_writers.py` (block-input trace),
`tools/_probe_funcmin_inject.py` (hook-inject confirmation).

* **The leaked byte**: at step 0 (JSR) BP byte-3 = `got=0x01 want=0x00`.
* **LM-head logit** (at the BP byte-3 predictor position): `logit[1]=78.24`
  vs `logit[0]=73.94`, margin `+4.3`, driven **100 %** by `OUTPUT_LO+1`
  (`W_head[1,OUTPUT_LO+1]-W_head[0,·] = +5`; residual `OUTPUT_LO+1 = +8.0`).
* **The row**: the BP byte-3 PREDICTOR row is the BP byte-2 value row
  (`IS_BYTE`, `BYTE_INDEX_2`, `H1+3`, `OP_JSR≈6.85`, `HAS_SE=0`). The OUTPUT
  band there is a near-TIE the WRONG way: `OUTPUT_LO+0 = +6.94` (nibble 0,
  correct 0x?0) vs `OUTPUT_LO+1 = +8.00` (nibble 1) → decodes low nibble 1.
* **Block trace**: a weak `+2.0` enters `OUTPUT_LO+1` at physical block ~18;
  the block-58 amplifier (`tail_lea_local_ax_byte0_amplify` region) scales the
  whole band ~3.4× (→ +6.94 / +8.00), preserving the wrong winner.
* **Why the leak persists**: the ENT step (step 1) HAS a byte-3 clear
  (`l6_ent_after_jsr_bp_byte3_00`, gated `OP_ENT`+`HAS_SE`). The **JSR step had
  no byte-3 clear**.
* **Root confirmed** (`_probe_funcmin_inject.py`): hook-inject clearing BP
  byte-3 → advances the vcorr from **step 0 → step 2** on func_min (675),
  func_max (650), func_identity (550), func_add (575) — the whole cluster.

## 2. writer_index / competition

The wrong cell (`OUTPUT_LO+1` at the BP byte-3 predictor row) is owned in the
tail by the OUTPUT-amplifier chain (`tail_lea_local_ax_byte0_amplify` /
`tail_bit32_result_correction`); the correct writer (the byte-3=0x00 default
present at block 4, `OUTPUT_LO+0=+0.94`) is out-scaled. A tail post_op placed
AFTER `tail_bit32_result_correction` (so it DOMINATES the block-58 amplifier)
hands the cell back to nibble 0.

## 3. Fix (`make_l10_jsr_bp_byte3_clear_op`, `l10_ops.py`)

A flag-gated `PureFFN` tail post_op (`requires after
tail_bit32_result_correction`) — 32 winner-take-all units forcing `OUTPUT_LO`
and `OUTPUT_HI_THIS_STEP` → nibble 0 (byte-3 = 0x00) on the JSR-step BP byte-3
predictor row. Discriminator (all BUILT-dim-verified, clean/always-present):

* `CONST -12` bias so **`OP_JSR·2` is a HARD requirement** — a non-JSR row
  (`OP_JSR==0`) falls sub-threshold → add/sub (no step-0 JSR) BYTE-IDENTICAL.
* `H1+3` (BP-register byte-row staging one-hot; PC=+0/AX=+1/SP=+2/BP=+3) — the
  clean BP discriminator on value rows.
* `IS_BYTE` + `BYTE_INDEX_2·10` as the **LOAD-BEARING row selector**;
  `BYTE_INDEX_0/1/3·(-10)` so the byte-0/1 predictor rows go deep-negative
  (NOT hard-blocked — the ~0.01 adjacent-index residue would false-veto, the
  documented `l6_ent_after_jsr_bp_byte*` trap).
* `HAS_SE` (ENT rows, already handled) + non-BP `H1+0/+1/+2` + every register
  MARKER hard-blocked (`-1e6`).

Threshold 6.0: JSR byte-3 predictor scores ~13.3; JSR byte-0/1 predictors ~-5.9;
non-JSR byte-3 predictor ~-0.4 → only the target row fires.

## 4. Verification

* **Golden**: `f725c06e` byte-identical flag-OFF (twice). 42117 units unchanged.
* **Isolated FFN**: on the synthetic target row → ΔOUTPUT_LO+0 = +1258,
  ΔOUTPUT_LO+1 = -1243 (byte-3 → 0x00).
* **Live model flag-ON**: BP byte-3 → 0x00 on every step-0 JSR; BP byte-2 stays
  0x01; BP byte-0/1 and SP byte-3 untouched; add/sub byte-identical ON/OFF.
* **On-target AR verdict** (`cpu_full_trace --spec-k 0` ON vs OFF on
  func_min/max + var_three/loop_sum + add/sub) + broad regression: launched,
  extremely slow under the multi-agent CPU contention — DEFERRED to the main
  thread (per brief). The teacher-forced + hook-inject evidence establishes the
  root and the byte flip; the AR PASS/advance verdict is the remaining
  confirmation.

## 5. Next wall (after this fix)

Hook-inject shows the NEXT vcorr after BP byte-3 is **step 2, SP byte-2** (the
SP re-anchoring on the callee-frame setup) — a separate follow-up. Whether it is
a genuine value leak or a benign re-anchoring the AR decode recovers from is TBD
(the teacher-forced tape cannot model the now-correct BP downstream).
