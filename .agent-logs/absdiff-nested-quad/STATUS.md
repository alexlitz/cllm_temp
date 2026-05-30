# B3-η STATUS: absdiff (0/25) and nested_quad (0/16) investigation

## Outcome: CROSS-REFERENCED (no new fix landed; B2-H cherry-picked)

Both dead categories are blocked at the same step1:MEM_addr0 0xF0→0xE0
bug already identified in B2-H (Recovery: current-step MEM marker fix in
batched runner). After applying B2-H, both categories advance past
step1 but still fail to pass; the downstream blocker is the SAME L10
authority rule `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority`
documented as the B3-α issue.

## Test sources

`c4_release/tests/test_suite_1000.py`:
- `nested_quad` (Category 10 first half, lines 461-469, IDs 950-974, 25
  tests; first-ever diag captured only IDs 959-974 = 16 tests):
  ```c
  int double_it(int x) { return x * 2; }
  int quad(int x)      { return double_it(double_it(x)); }
  int main()           { return quad(N); }
  ```
- `absdiff` (Category 12, lines 553-565, IDs 1046-1070, 25 tests):
  ```c
  int abs_diff(int a, int b) {
      if (a > b) return a - b;
      return b - a;
  }
  int main() { return abs_diff(A, B); }
  ```

## Baseline (base 4d069f7) divergences

### absdiff first_token_divergence (autoregressive, C4_BATCH_USE_KV_CACHE=0)
```
id=1046  step1:MEM_addr0 abs=329 gen=61 expected=0xf0 neural=0xe0
id=1047  step1:MEM_addr0 abs=329 gen=61 expected=0xf0 neural=0xe0
id=1048  step1:MEM_addr0 abs=329 gen=61 expected=0xf0 neural=0xe0
```
All 3/3 fail at the **exact same step:slot** with identical logits:
- expected_logit = +4.4e14, argmax_logit = +1.05e16, margin = -1.01e16
- OUTPUT_HI[14] (0xe high nibble) wins over OUTPUT_HI[15] (0xf high nibble)

### nested_quad first_token_divergence (autoregressive)
```
id=0959  step1:MEM_addr0 abs=273 gen=61 expected=0xf0 neural=0xe0
id=0960  step1:MEM_addr0 abs=273 gen=61 expected=0xf0 neural=0xe0
id=0961  step1:MEM_addr0 abs=273 gen=61 expected=0xf0 neural=0xe0
```
Same first-fatal step and slot as absdiff (expected=0xf0 neural=0xe0,
argmax_logit = +1.05e16). All nested_quad rows produce neural=244 (0xF4)
as the constant final result.

## Cross-reference: this IS the B2-H bug

The step1:MEM_addr0 0xF0→0xE0 collapse at ENT push-BP is exactly what
B2-H (origin/fix/jsr-ent-cascade-step1, commit 40014a9, also reachable
via 0ce603 chain) addressed at the runner level by tracking the
current-step MEM marker as a store before MEM_addr0 prediction.

## Behavior with B2-H cherry-picked (2663ca2 in this branch)

### absdiff post-B2-H
```
id=1046  trace_error=CUDA OOM (GPU contention; not the bug)
id=1047  step3:STACK0_byte0 expected=0x59 argmax=0x39 (HIGH nibble drift 5→3)
id=1048  step3:MEM_addr0 expected=0xe8 argmax=0x57 (cascade noise)
```

### nested_quad post-B2-H
```
id=0959  step3:MEM_addr0 abs=343 gen=131 expected=0xe8 neural=0x08
         argmax=0xe0  argmax_logit=+9.35e13  margin=-9.35e13
id=0960  step3:MEM_addr0 abs=343 gen=131 expected=0xe8 neural=0x08
         argmax=0xe0  argmax_logit=+9.35e13  margin=-9.35e13
```

The B2-H fix successfully unblocks step1. The new failure at step3 is
the **PSH-after-ENT-0 0xE8 → 0xE0** case, which is exactly the
mechanism documented in `.agent-logs/step0-reg-pc/STATUS.md` (B3-α) on
`integration/batch-merge-lean`: the FFN rule
`tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority`
(l10_ops.py:3888-3922) with `strength=5_000_000_000.0` and
`threshold=40` pins 0xE0 over the correct 0xE8 PSH target because
its `byte_writes(0xE0, strength=5e9)` overwhelms sibling rules
whenever its gates clear.

## Teacher-forced lowering audit (provides cleaner per-step picture)

### nested_quad id=0959 (lowering audit, base 4d069f7)
```
first_fatal: step6:AX_byte0  abs=428 expected=0xe8 argmax=0x00 margin=-979
             (register-result row feeds runner architectural state)
first_info:  step3:MEM_addr0 abs=343 expected=0xe8 argmax=0x00 margin=-299
             (support-drift outside fatal register gate)
```

### absdiff id=1046 (lowering audit)
```
first_fatal: step7:SP_byte0     abs=524 expected=0xd0 argmax=0xd8 margin=-1.8e4
first_info:  step3:MEM_addr0    abs=399 expected=0xe8 argmax=0x00 margin=-303
```

Teacher-forced correctly emits step1 (no autoregressive cascade), so the
audit sees later failures: step3 MEM_addr0 drift is the same cascade
the autoregressive runner hits once step1 is corrected by B2-H.

## Attempts at downstream fix (NOT landed — both regress)

### Attempt A: cherry-pick 0ce6030 ("Guard L10/L16 PSH MEM addr0 0xE0
rules against ENT context"), which adds `OP_ENT=-1e6` to the dominant
rule.

Result: tests regress to **step0:REG_PC** failure (model picks byte
0xE0 over the REG_PC sentinel at the very first marker prediction):
```
absdiff/nested_quad  step0:REG_PC abs=268/212
                     expected=REG_PC argmax=0xe0
                     argmax_logit=+4.36e11  margin=-4.36e11
```
This matches the original B3-α brief's symptom exactly. The OP_ENT
discriminator change apparently shifts where the 5e9 rule fires (out of
ENT contexts and into the JSR-step0 context with REG_PC slot
competition). Reverted.

### Attempt B: reduce `strength=5_000_000_000.0` to `1_000_000.0` per
the doc's recommendation in `.agent-logs/step0-reg-pc/STATUS.md` step
2.

Result: smoke test of basic `add` (IDs 0-7) regresses globally:
```
id=0002 step0:SP_byte0 abs=55 expected=0x00 neural=0xf8 margin=-1089
id=0003 step0:AX_byte0 abs=50 expected=0xe4 neural=0x01 margin=-3.3e5
id=0004 step0:AX_byte0 abs=50 expected=0xf2 neural=0xe8 margin=-5.6e8
...
```
The 5e9 rule was apparently keeping some unrelated SP/AX heads
calibrated by side effect; reducing it 5000× lets other rules
dominate at the wrong slots. Reverted.

Note: nested_quad with the strength fix still fails at step3:MEM_addr0
0xe8→0xe0, just with scaled-down logits (~1.9e10 instead of 9.35e13);
the cascade rule still wins after the reduction. So even a clean
strength reduction would not unblock nested_quad on its own — a
discriminator-based fix (per B3-α step 1: `OUTPUT_HI+15 = -1.0`) is
needed to suppress 0xE0 when 0xF0/0xE8 evidence is present, and a
coordinated re-tuning of the four sibling `tail_mem_store_addr0_*`
rules is required to avoid breaking the existing local-store contexts.

## Recommended follow-up

Both categories will likely unblock together with one coordinated fix
to the L10 0xE0 emitter family:

1. Implement the discriminator-style fix recommended in
   `.agent-logs/step0-reg-pc/STATUS.md` (add `OUTPUT_HI+15 = -1.0` or
   similar evidence guards to the 5e9 authority rule so it
   self-suppresses once the 0xF0 / 0xE8 sibling rules emit positive
   evidence) — but pair it with a regression sweep over IDs 0-7
   (basic_add) to catch the SP/AX collateral damage that bare
   strength-reduction triggers.
2. Add a focused L10 op-level test pinning the rule's gates for
   `if_var_0` SI (id 425 step1), `quad(8)` ENT (id 959 step1), and
   the PSH-after-ENT-0 row (id 959 step3) to lock-in the discrimination
   before downstream re-baking.
3. Re-run the full IDs 959-1095 slice to confirm absdiff (25 tests)
   and nested_quad (16 tests in the captured window) flip green
   together; both should ride the same fix because their failures share
   the dominant emitter.

## Files / commits referenced

- Test sources: `c4_release/tests/test_suite_1000.py:461-469,553-565`
- Bug-bearing rule: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3888-3922`
- B2-H fix (cherry-picked here as 2663ca2): origin/fix/jsr-ent-cascade-step1, commit 40014a9
- Failed downstream attempts: 0ce6030 (regressed step0:REG_PC),
  strength=1e6 patch (regressed basic_add step0:SP_byte0/AX_byte0)
- Cross-reference docs (on integration/batch-merge-lean only):
  `.agent-logs/step0-reg-pc/STATUS.md` (B3-α),
  `.agent-logs/batch-959_1095/divergence_rows.txt` (first capture).

## Logs (this run)

- `/tmp/absdiff_trace.log` — baseline absdiff IDs 1046-1048 trace
- `/tmp/nested_quad_trace.log` — baseline nested_quad IDs 959-961
- `/tmp/absdiff_lowering.log`, `/tmp/nested_quad_lowering.log` —
  teacher-forced lowering audit (B2-H not required for teacher-forced)
- `/tmp/absdiff_postfix.log`, `/tmp/nested_quad_postfix.log` —
  with B2-H only
- `/tmp/absdiff_postboth.log`, `/tmp/nested_quad_postboth.log` —
  with B2-H + 0ce6030 (regressed)
- `/tmp/smoke_strength.log` — basic_add regression under strength=1e6
