# Wall-2 SE-relay transmission — findings (2026-06-11)

Status: **transmission solvable; downstream CMP cascade architecturally
blocks lt/le.** Stopped at the last green baseline (HEAD `4c82768e`) per
the task STOP condition. No code change landed.

## Topology (empirical, spec_k=0 cold build)

The block numbering in prior notes was imprecise. Verified per-block
probe of the runtime (post-expansion) model:

- **Runtime block 11** = the **SE-consumer** block. Its FFN reads
  `SE_ALU_LO` (`FFN_reads_SE_ALU_LO≈13600`). Its attention has **8
  heads**: head 0 = MARK_SP byte-passthrough (slope 5.0), head 1 =
  MARK_PC (slope 1.0), **heads 2/3/4 EMPTY weights** but sloped
  1.0/0.5/1.0 by the phase-999 `layer10_residual_alibi_slopes` op
  (`alu_ops.py:1873-1883`, which writes source `model.blocks[10]` =
  runtime 11), **heads 5/6/7 EMPTY** with the untouched decay slope
  (0.016/0.008/0.004).
- The L9 `step_end_operand_relay` lands on **runtime block 11 at heads
  3/4** (via `target_op_name="layer9_marker_suppress"`).
- **Runtime block 12** = the resized L10 attention (`_layer10_attn_anchor`,
  13 heads). This is where PSH-STACK0-passthrough (head 3) and
  STACK0-byte-relay (head 4) ACTUALLY live — NOT block 11. The
  `residual_alibi` op's `[3]=0.5 / [4]=1.0` writes on block 11 are
  therefore aimed at the wrong block; on block 11 they only clobber the
  relay.

So the original "slots 3/4 are contested with STACK0-passthrough" framing
was wrong: on the SE-consumer block (11) heads 3/4 host ONLY the relay.
The clobber is `residual_alibi` (a phase-999 model op that runs AFTER the
relay's block-op bake) overwriting the relay's slope 0.2 → 0.5/1.0.

## Wall 2 transmission IS fixable (verified)

One-line change in `make_layer10_residual_alibi_slopes_op._bake`
(`alu_ops.py`): set block-11 `alibi_slopes[3]=alibi_slopes[4]=0.15`
(the relay's transmit slope) instead of 0.5/1.0. Relay stays at 3/4
(V/O unchanged from baseline). Probe (`probe_se_relay.py`, L9_BLOCK=11):

| dim | baseline | after slope fix |
|-----|---------:|----------------:|
| SE_ALU_LO     | 1.41  | **5.57** |
| SE_AX_CARRY_LO| 0.001 | **0.96** |
| SE_OP_EQ      | 0.01  | **4.60** |
| SE_CMP_GROUP  | 0.002 | **0.92** |

All targets ~1.0+ — clean AX→SE transmission, matching the brief's
acceptance for the SE_* dims. No relay move, no resize needed.

(Re-homing the relay to free slots 5/6, or to block-12 slots 13/14 via a
15-head resize, were also tried; 5/6 transmits identically, 13/14 lands
on the WRONG block — block 12, whose FFN is not the SE consumer — and
reads back 0.)

## Why it can't land: downstream CMP cascade blocks lt/le

Both transmission variants (relay at 3/4 with slope-fix, and relay at
5/6) produce the SAME smoke result — a **net regression**:

```
TARGETS passing (1/6): [eq_false]            (eq_true/mul/and/and_16bit/add_carry_cascade still fail)
GUARDRAIL REGRESSIONS: [lt_true, le_true, cmp_and_branch]
```

Root: once the relay transmits, `SE_CMP_GROUP` rises 0.002→0.92, which
ACTIVATES the migrated L9 CMP cascade (`_layer9_cmp_rules`,
`l9_ops.py:548`, hi_eq/lo_eq/hi_lt/lo_lt gated on `SE_CMP_GROUP`, reading
`SE_ALU_*`/`SE_AX_CARRY_*`). Active, it writes raw `CMP+i` at the SE row
with values that CORRUPT the lt/le path (which at baseline resolves via
the dormant-cascade default path in L10 `cmp_combine`). This is the
"SE comparison cascade is architecturally incompatible with the relayed
2-cell operand encoding" hazard already on record in
`project_operand_gather_hybrid_encoding_is_cmp_alu_root` and
`project_probe_path_spec_k_not_hooks`.

Flipping the L10 `cmp_combine` opcode gates from raw `OP_<NAME>` →
`SE_OP_<NAME>` (`_l10_comparison_combine_rules` + `_layer10_alu_cmp_combine_rules`,
`l10_ops.py:535/731`; ded5e4cd's idea) had **zero observable effect** on
the smoke result — neither fixed EQ nor changed lt/le — so the EQ failure
is NOT merely a gate-dim mismatch; the EQ result resolves on a path the
flip doesn't reach, and the lt/le regression is upstream of cmp_combine
(in the L9 cascade's raw `CMP+i` writes).

## Conclusion / next step

Wall 2 (transmission) is a solved, one-line slope fix. The remaining
blocker is NOT transmission or slot allocation — it is that the L9 SE CMP
cascade, once fed, computes `CMP+i` flags that are wrong for ordered
comparisons under the 2-cell `SE_ALU`/`SE_AX_CARRY` operand layout. The
next agent should attack the **L9 CMP cascade semantics under the relayed
encoding** (e.g. make hi_eq/lo_eq/hi_lt/lo_lt robust to the nibble-pair
operand representation, then re-point cmp_combine at the corrected
flags), NOT the relay placement or ALiBi slope.

Tools used: `tools/probe_se_relay.py`, `tools/run_full_smoke.py`
(focused groups: comparison bitwise basic bit32 shift integration),
the ALiBi collision map, and per-block weight probes.
