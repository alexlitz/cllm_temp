# Phase 4 BZ/BNZ Diagnosis (2026-05-10)

## Status snapshot

Running `c4_release/tests/test_pure_neural_jmp_bz.py` on `main` (after commit
`241bbdd Phase 4 BZ/BNZ: suppress FETCH->PC at IMM step after taken branch`):

```
======================= 12 xfailed in 1591.24s (0:26:31) =======================
```

All 12 tests xfailed (none xpassed). This is a regression from the previously
recorded baseline of 11 xfail + 1 xpass (`test_countdown_loop` xpass).

Single test runtime: ~130s/test (each test runs the full autoregressive
pure_neural VM with no Python override, ~280 model forward passes per test).

## Earlier-agent diagnosis (preserved from commit `8ddbbd3`/`241bbdd` message)

> remaining failures (BZ/BNZ not_taken, countdown_loop) are due to a separate
> root cause (leaked OP_BZ at PC marker triggering BZ AX-passthrough unit) —
> not yet addressed.

User-supplied hint repeated:

> Phase 4 BZ/BNZ root cause: OP_BZ activation reaches 15 instead of 5 at step 1
> PC marker.

## Code paths verified relevant

1. `_inject_active_opcode` (neural_embedding.py:202) injects `OP_BZ = 5.0` at
   *all* positions when `active_opcode == 4` (BZ). This is the legacy "global
   opcode hint" mechanism. In **pure_neural mode** the runner skips
   `set_active_opcode`, so this injection never happens (run_vm.py:851-857).

2. `_set_layer5_op_decode` (vm_step.py:3840-3922) — first-step decode at
   PC marker via `OPCODE_BYTE_LO/HI`. Writes `OP_BZ` with W_down `10/S` only at
   PC marker, gated by `HAS_SE = -S` (only fires on the very first step).

3. **All-step opcode decode at PC marker** (vm_step.py:4105-4122) — writes
   `OP_BZ` with `W_down = 10/S` at every PC marker when the opcode byte is BZ.
   This is the line said to enable "BZ/BNZ/LEV/EXIT routing at the PC marker
   without relying on `_inject_active_opcode`":
   ```python
   for op_val, lo, hi in [
       (Opcode.BZ, 4, 0),
       (Opcode.BNZ, 5, 0),
       (Opcode.LEV, 8, 0),
       (Opcode.EXIT, 6, 2),
       (Opcode.JMP, 2, 0),
   ]:
       op_dim = BD.opcode_dim(op_val)
       ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
       ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
       ffn.W_up[unit, BD.MARK_PC] = S
       ffn.b_up[unit] = -S * 2.5
       ffn.b_gate[unit] = 1.0
       ffn.W_down[op_dim, unit] = 10.0 / S
   ```
   This writes `OP_BZ ≈ 5` at the PC marker (per unit), but the AX-marker
   variant (gated by `W_gate=MARK_AX`) also writes 5 at AX marker. Their
   residual sums into downstream layers.

4. `_set_layer6_routing_ffn` BZ AX-passthrough (vm_step.py:4876-4892) intends
   to fire ONLY at MARK_AX during BZ:
   ```python
   ffn.W_up[unit, BD.OP_BZ] = S          # +500 when OP_BZ=5
   ffn.W_up[unit, BD.MARK_AX] = S        # +100 when at AX
   ffn.W_up[unit, BD.MARK_PC] = -S       # -100 when at PC (blocker)
   ffn.b_up[unit] = -S * T               # T=4.0 → -400
   ffn.W_gate[unit, BD.AX_CARRY_LO+k] = 1.0
   ffn.W_down[BD.OUTPUT_LO+k, unit] = 2.0/S
   ```
   * At AX marker with OP_BZ=5: `up = 500 + 100 - 0 - 400 = 200` (fires, gates AX_CARRY → OUTPUT, OK)
   * At PC marker with OP_BZ=5: `up = 500 + 0 - 100 - 400 = 0` (silu≈0, doesn't fire — OK)
   * At PC marker with OP_BZ=15 (leaked): `up = 1500 + 0 - 100 - 400 = 1000` (FIRES → writes AX into PC OUTPUT — BUG)

5. `_set_bz_bnz_relay` (vm_step.py:8576) — L6 attention head 4. The Q-side
   uses `W_q[base, BD.OP_BZ] = L/5.0` (assumes OP_BZ ~= 5 → contributes L). If
   OP_BZ leaks above 5, the head's Q score scales up linearly; with OP_BZ=15
   the contribution is `(15)*(L/5)=3L` instead of L. This may invert the
   designed score budget at non-PC positions, but the head only writes
   `CMP[2..5]` so it's not the direct trigger of the regression — it just
   makes the relay attend "too strongly" and possibly over-write CMP[4]/[5].

## Hypothesised origin of "OP_BZ = 15" (not yet verified at activation level)

The all-step PC decode (path 3 above) writes OP_BZ at PC marker.
The AX-marker decode (path inside _set_layer5_op_decode, gated by `MARK_AX`)
writes OP_BZ at AX marker.

In pure_neural mode neither matches "all positions" — yet the user's diagnosis
text says OP_BZ reaches 15 (= 3 × 5) at step 1 PC marker. Three plausible
mechanisms:
* L4 (or earlier) FFN units that *re-derive* the opcode flag (`opcode_dim`
  writes happen in `_set_layer5_op_decode` and the all-step decode in
  `_set_layer5_op_decode_at_pc`) may both fire at the same MARK_PC position.
  If both an "AX-marker" unit and a "PC-marker" unit happen to share residual
  positions on a particular step, the value can add.
* L6 attention head 4's V output goes to `CMP+2`, not OP_BZ; but residual-stream
  leakage from the relay-head FFN that writes OP_BZ at AX could spill back to
  the same dim at PC marker after attention mixing.
* `_inject_active_opcode` is NOT supposed to run in pure_neural — but the
  call-site (`run_vm.py:328-331`) guards with `if not self.pure_neural`. Worth
  verifying that the model's `embed.forward(token_ids, active_opcode=None)`
  truly receives `None` and that no path silently passes the prior opcode.

## What would unblock the not-taken tests

Given the existing fix in `_set_function_call_weights` blocks spurious
OP_BZ-driven writes to PC OUTPUT *at MARK_STACK0* positions, the analogous
fix for the not-taken case is to add a stronger MARK_PC blocker (or an
OP_BZ-cap) to the BZ AX-passthrough units in `_set_layer6_routing_ffn`
(lines 4876-4910). Two options:

(A) Raise the MARK_PC blocker strength so it dominates even when OP_BZ leaks
    to 15. Currently `W_up[MARK_PC] = -S` (-100). With OP_BZ=15 the up-score
    at PC is +1000, so the blocker would need to be at least `-S * 11`
    (~-1100). Example: `ffn.W_up[unit, BD.MARK_PC] = -S * 12` for both BZ and
    BNZ passthrough units (lines 4880, 4888, 4898, 4906).

(B) Bound OP_BZ contribution: instead of `W_up[OP_BZ] = S`, use
    `W_up[OP_BZ] = S/3` (so OP_BZ=15 contributes the intended +500). This
    preserves the threshold math for the AX-marker case (3× OP_BZ at AX
    marker would also be 15, contribution = 500 = correct).

Option (A) is the minimum-blast-radius fix. Option (B) would also need
verification that AX-marker OP_BZ truly reaches 15 in pure_neural mode.

## What was NOT done in this diagnosis pass

Because each test run takes ~130s on shared GPU (the suite ran in 26.5 min)
and a focused activation trace would also need a fresh model build (~7s
cold-start) plus token-level instrumentation:

* No instrumented run was performed to verify the OP_BZ=15 claim numerically
  at the actual PC marker. The user's prior diagnosis is taken as given.
* No fix was attempted because the user-supplied numerical hint is just an
  *intermediate* signal; the actual blast radius (which downstream FFN unit
  fires erroneously) needs verification before patching, and Option (A)/(B)
  above are speculative until that's checked.

## Recommended next step (cheap)

Add a one-shot instrumentation hook in `AutoregressiveVMRunner` (or directly
in `_set_layer6_routing_ffn`'s call site) that, on every forward pass with
pure_neural=True and step==1, prints `OP_BZ` activation at the current PC
marker position. Run `test_bz_not_taken[1]` once. If OP_BZ at PC marker is
indeed 15, Option (A) is safe to try. If it's 5, the AX-passthrough unit is
not the failing unit and a different FFN unit must be located.

Once the failing unit is identified, the patch is ~3 lines (one MARK_PC
blocker bump) and the fix can be validated by a single 130s test run.

## Files relevant

* `c4_release/tests/test_pure_neural_jmp_bz.py` — the test gate.
* `c4_release/neural_vm/run_vm.py:324-355` — pure_neural mode entry skips
  active-opcode injection and forced STEP_END.
* `c4_release/neural_vm/neural_embedding.py:202-224` — `_inject_active_opcode`.
* `c4_release/neural_vm/vm_step.py:3840-4122` — L5 opcode decode (AX + PC).
* `c4_release/neural_vm/vm_step.py:4307-4910` — `_set_layer6_routing_ffn`,
  including BZ/BNZ AX-passthrough units at 4876-4910.
* `c4_release/neural_vm/vm_step.py:5095-5199` — `_set_layer6_routing_ffn`
  BZ/BNZ PC-override units (taken-path).
* `c4_release/neural_vm/vm_step.py:8576-8636` — `_set_bz_bnz_relay` (L6 attn
  head 4, BZ/BNZ relay to CMP+2..5).
* `c4_release/neural_vm/vm_step.py:9673-9728` — JSR STACK0 marker fix
  (commit 241bbdd's blockers, useful pattern to follow).
* `c4_release/neural_vm/unified_compiler/migrated_ops.py:577-608` —
  `make_layer6_bz_bnz_relay_bake_op` (compiler integration).
