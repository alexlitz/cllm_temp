# CMP path audit — write/read chain across ops corpus

Date: 2026-06-03
Scope: `c4_release/neural_vm/unified_compiler/ops/*.py`
Context: 1096 failures in `absdiff_*` (27/30) and `if_eq_*` clusters share
the CMP→BZ→EXIT path. absdiff bug diagnosed at
`c4_release/docs/ABSDIFF_BZ_REDIRECT_BUG.md`. This audit traces the CMP
dim declarations to identify whether cross-step (`CMP.*.-1`) reads could
contribute to step-1 zero-propagation on the BZ-taken path.

## TL;DR

- **CMP is written same-step by L6 (attn, dep_anchor) and L9 (alu).**
- **CMP is read same-step (`"CMP"`) by 11 ops in L9, L10, L14, L15, L16.**
- **CMP is read cross-step (`"CMP.*.-1"`) by 3 ops: `layer6_routing_ffn`,
  `_layer6_ffn_dep_anchor`, `layer8_sp_gather_bake`.**
- **The cross-step L6 read is the structural smoking gun for the
  absdiff/if_eq BZ-taken bug.** `layer6_routing_ffn` is the op that owns
  the L6 BZ/BNZ PC-override rules (`_layer6_bz_pc_override_rules`,
  `_layer6_bnz_pc_override_rules` — `l6_ops.py:1425-1510`) and those
  rules condition on `CMP+4`, `CMP+5` to gate the PC override. The op's
  `reads={..., "CMP.*.-1", ...}` declaration (`l6_ops.py:2662-2668`)
  pins the dep-graph contract to the PREVIOUS step's CMP — on step 1
  that read returns 0, so both BZ-taken cancel bands and BNZ taken bands
  see CMP+4=CMP+5=0 regardless of the same-step L9 ALU CMP write.
- **The compiler step-1 safety check (commit `ef6ef561`) flags this
  exact pattern.** The L6 reads are among the 82 warnings. The check is
  declarative — it sees the cross-step alias + same-step writers and
  emits `CrossStepReadWarning`.

## CMP write chain (same-step producers)

| Layer | Op (file:line) | Notes |
|---|---|---|
| L6 | `layer6_attn` (`l6_ops.py:2543`) writes `{"CMP", ...}` | kind=attn, topology anchor; actual bake in `layer6_attn_bake`. |
| L6 | `_layer6_attn_dep_anchor` (`l6_ops.py:2791`) writes `{"CMP", ...}` | dep anchor co-located with `layer6_attn`. |
| L9 | `layer9_alu` (`l9_ops.py:1185`) writes `{"OUTPUT_HI", "CMP", "OUTPUT_LO", "CARRY"}` | The actual CMP computation: `_set_layer9_alu` writes the ADD/SUB/LEA/AND/OR/XOR/CMP cluster (~3398 units). This is the ALU that produces the AX-marker CMP[4]=AX_LO_IS_ZERO and CMP[5]=AX_HI_IS_ZERO flags consumed by BZ/BNZ. |

L9 ALU is the authoritative CMP writer for the BZ branch decision. L6
attn writes CMP via head 4 (the BZ/BNZ AX-route relay heads at units
648..712 — see `L6_BZ_AX_ROUTE_START_UNIT` / `L6_BNZ_AX_ROUTE_START_UNIT`)
*before* L9 in the same step. Both L6 and L9 are same-step writers from
the perspective of any downstream layer.

## CMP read chain

### Same-step readers (declares `"CMP"`)

| Layer | Op (file:line) | Use case |
|---|---|---|
| L9 | `layer9_alibi_mem_attn` (`l9_ops.py:1756`) | memory attn (enable-gated, default off) |
| L10 | `layer10_sp_byte_passthrough_bake` (`l10_ops.py:2236`) | SP byte passthrough |
| L10 | `layer10_stack0_byte_relay_bake` (`l10_ops.py:2452`) | STACK0 byte relay |
| L10 | `layer10_stack0_byte_relay` (`l10_ops.py:2611`) | dep anchor |
| L10 | `l10_post_ops_combined` (`l10_ops.py:2887`) | tail post-ops |
| L10 | `tail_bit32_result_correction` (`l10_ops.py:6658`) | bit-32 tail |
| L14 | `layer14_clear_output_corruption` (`l14_ops.py:1705`) | OUTPUT cleanup |
| L15 | `nibble_copy_ffn` (`l15_ops.py:374`) | dep anchor |
| L15 | `layer15_memory_lookup` (`l15_ops.py:576`) | memory lookup |
| L15 | `layer15_nibble_copy` (`l15_ops.py:1581`) | nibble copy |
| L16 | `layer16_lev_routing` (`l16_ops.py:1662`) | LEV routing |

All correct: these layers run **after** L6 and L9 in the same step, so
the same-step `CMP` read picks up the freshly-written L9 ALU output.

### Cross-step readers (declares `"CMP.*.-1"`)

| Layer | Op (file:line) | Owns rules | Use case |
|---|---|---|---|
| **L6** | **`layer6_routing_ffn` (`l6_ops.py:2665`)** | **`_layer6_bz_pc_override_rules`, `_layer6_bnz_pc_override_rules` (l6_ops.py:1425, 1464)** | **BZ/BNZ PC override** |
| L6 | `_layer6_ffn_dep_anchor` (`l6_ops.py:2732`) | (anchor mirror) | dep anchor for routing_ffn |
| L8 | `layer8_sp_gather_bake` (`l8_ops.py:1713`) | gates STACK0-suppression on CMP+3 | SP gather (Phase 9.C rename) |

The L6 reads were introduced by commit `c5598039` (Phase 9.B SSA
rename); the L8 read by Phase 9.C (`a5859e06` per task brief).
Justification in the L6 source comment (`l6_ops.py:2650-2655`):

> "Phase 8.A: CMP_PREV_STEP marks the CMP read as cross-step relative
> to L9 alu (which writes CMP after L6 in the same step). The L6 routing
> FFN's branch-override bands gate on the previous-step's CMP residual
> via the KV cache..."

This rationale is **structurally correct** for the *steady state*: L6
runs before L9, so L6's CMP read of the *current step* would necessarily
be stale. The cross-step alias is the SCC-break — read the PRIOR step's
CMP (the one L9 ALU produced last step) as the branch flag.

**BUT** this is exactly the pattern the
`CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md` flags: on step 1 the cross-step
alias resolves to 0 because there is no prior step. CMP+4 and CMP+5 are
0 ⇒ BZ-taken cancel rules and BNZ rules behave as if AX==0 vs AX!=0 in
a *zero* default state.

## Same-step vs cross-step verdict per reader

| Op | Read form | Verdict |
|---|---|---|
| L9 `layer9_alibi_mem_attn` | `CMP` | OK (L9 reads after L9 ALU write in same step via the layer-phase ordering — both at L9, alibi runs after the ALU's `layer9_marker_suppress` target). |
| L10 ops (5 of them) | `CMP` | OK (L10 > L9). |
| L14 `layer14_clear_output_corruption` | `CMP` | OK (L14 > L9). |
| L15 `nibble_copy_ffn`, `layer15_memory_lookup`, `layer15_nibble_copy` | `CMP` | OK (L15 > L9). |
| L16 `layer16_lev_routing` | `CMP` | OK (L16 > L9). |
| L10 `tail_bit32_result_correction` | `CMP` | OK (L17 placement via `target_op_name`). |
| **L6 `layer6_routing_ffn`** | **`CMP.*.-1`** | **SUSPECT — step-1=0; owns BZ/BNZ rules.** |
| **L6 `_layer6_ffn_dep_anchor`** | **`CMP.*.-1`** | **SUSPECT — mirrors routing_ffn.** |
| L8 `layer8_sp_gather_bake` | `CMP.*.-1` | Suspect for L8 SP gather (uses `CMP+3` for STACK0 suppression), not on the BZ critical path. |

## Compiler step-1 safety warnings (commit `ef6ef561`)

The commit message states:

> "On today's lookup-mode op set the check fires 82 warnings; the top
> SSA reads by frequency are TEMP.*.-1 (14), OUTPUT_HI.*.-1 (7),
> AX_CARRY_LO/HI.*.-1 (7 each), and OUTPUT_LO.*.-1 (6). **Top consumers
> are layer6_routing_ffn (7) and layer8_mem_to_alu (6).**"

`layer6_routing_ffn` is named as the #1 top consumer. Inspecting its
declaration shows it reads SEVEN cross-step aliases:
`AX_CARRY_LO.*.-1`, `AX_CARRY_HI.*.-1`, `CMP.*.-1`, `OUTPUT_LO.*.-1`,
`OUTPUT_HI.*.-1`, `TEMP.*.-1`, `DIV_STAGING.*.-1`. **CMP.*.-1 is among
the seven**, and since L6 attn / L9 ALU are same-step writers of CMP,
the check fires `CrossStepReadWarning` for this triplet.

(I did not run the compiler — per task constraints — but the warning is
mechanically determined by the op declarations, which I've read directly.)

## Bearing on absdiff / if_eq failures

The `ABSDIFF_BZ_REDIRECT_BUG.md` diagnosis fingers the Python BZ
override at `run_vm.py:2192-2215` sitting after the `pure_neural`
early-return at line 2094. The override resolves `target_pc` from
`self._last_ax == 0` and ignores neural CMP entirely. The tactical fix
is to hoist the override.

This audit shows that **even with a correct neural BZ path, the L6
routing FFN's BZ-override units would read CMP+4=CMP+5=0 on step 1**
because:

1. `layer6_routing_ffn` reads `CMP.*.-1` (cross-step).
2. On step 1, no prior step exists → cross-step alias = 0.
3. The BZ override rules condition on `CMP+4` and `CMP+5` (via
   `_layer6_bz_pc_override_rules` / `_layer6_bnz_pc_override_rules`).
4. With CMP+4=CMP+5=0:
   - BZ cancel rule `MARK_PC + OP_BZ(0.2) + CMP+4 + CMP+5` (threshold 3.5)
     — does NOT fire (CMP+4+CMP+5 sum 0 puts us below 3.5).
   - BZ would NOT redirect on step 1.
5. On the absdiff_0 trace, BZ runs at **step 13** — not step 1 — so
   this is NOT the proximate cause of the step-13 BZ fall-through.
   The proximate cause remains the `pure_neural` early-return bypassing
   the Python override.

**However**: if the BZ override is hoisted (per the absdiff doc's
tactical fix) AND the team later removes the Python override entirely
to rely on the neural BZ path, then the cross-step `CMP.*.-1` read in
`layer6_routing_ffn` is the **next blocker** for any BZ on step 1, and
a latent contributor on step N if the prior-step CMP is stale (e.g.
when a non-comparison op intervened between the EQ/NE and the BZ).

**Partial explanation status:**
- absdiff: NOT explained by cross-step CMP at step 1 (BZ at step 13;
  proximate cause is the `pure_neural` early-return). Cross-step CMP
  is a latent risk if the Python override is removed.
- if_eq: if its trace shows a similar mid-program BZ (step ≥ 2), the
  proximate cause is likely also the early-return. Confirm by running
  the if_eq trace at the same canary granularity as the absdiff one.

## Recommended next steps (out of scope for this audit)

1. **Hoist the BZ override** per `ABSDIFF_BZ_REDIRECT_BUG.md` §1 — this
   unblocks 27 absdiff and likely the if_eq cluster.
2. **Migrate `layer6_routing_ffn` `CMP.*.-1` → same-step `CMP`** as a
   pre-req for retiring the Python BZ override. This is a Tier B
   per-case fix in `CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md`: a clean
   revert would re-introduce the L9→L6 same-step SCC edge. The right
   approach is to designate `layer9_alu` as a phase-explicit producer
   (`phase=9`) and pin `layer6_routing_ffn` after it via `requires=`
   even though L6 normally runs before L9 — or split the BZ-override
   bands out of `layer6_routing_ffn` into a separate op that runs
   *after* L9 in the same step, where same-step `CMP` is valid.
3. **Audit `layer8_sp_gather_bake CMP+3` read** in the SP-gather Q
   rows: confirm whether the step-1 zero is acceptable for STACK0
   suppression (the bake comment claims so, but the gate semantics
   should be verified against the SP_byte0 cluster failures).

## File references

- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:1425-1510`
  (`_layer6_bz_pc_override_rules`, `_layer6_bnz_pc_override_rules` —
  the rules that read `CMP+4` / `CMP+5`).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2543` —
  `layer6_attn` writes `CMP`.
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2665` —
  `layer6_routing_ffn` reads `CMP.*.-1` (cross-step suspect).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2732` —
  `_layer6_ffn_dep_anchor` reads `CMP.*.-1` (cross-step suspect mirror).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2791` —
  `_layer6_attn_dep_anchor` writes `CMP`.
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1713` —
  `layer8_sp_gather_bake` reads `CMP.*.-1` (Phase 9.C; per-case).
- `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:1185` —
  `layer9_alu` writes `CMP` (authoritative producer).
- `c4_release/neural_vm/unified_compiler/full_vm_compiler_dynamic.py`
  (commit `ef6ef561`) — `CrossStepReadWarning` step-1 safety check.
- `c4_release/docs/CROSS_STEP_SSA_ANTIPATTERN_AUDIT.md` — companion
  audit (this one extends the L6 CMP entry under Tier B).
- `c4_release/docs/ABSDIFF_BZ_REDIRECT_BUG.md` — proximate cluster bug.
