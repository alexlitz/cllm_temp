# L29 Amplifier Attribution — XOR_BASIC OUTPUT_LO probe (2026-06-09)

Re-localization attempt for the collapsed-step cascade amplifier under
the new `dim_oracle` infra. Per the B2.2 brief, the +3,868 amplification
was hypothesized to live at L29 (NOT L28 as documented in
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`, since Edit A capped L28
at +40).

**Finding: L29 hypothesis does not hold for `XOR_BASIC`. L29 is the
SHL/SHR composite (opcode-gated to OP_SHL/OP_SHR) and does not fire on
the XOR path.**

## Layout topology

`compile_full_vm_dynamic(declarations_only=True)` (the layout the
replay tool actually drives) has **23 IR layers** (L00..L22). The
"L29" label seen in unit-shrink reporting (`L29.ffn.precompute_stage...`)
belongs to the per-block PyTorch-model layer index after composite
expansion (36 blocks total) — that index is NOT what
`SymbolicForwardRunner.block_idx` ranges over. The runner walks
`ops_per_layer` (length 23).

The 36-block-index L29 corresponds to the L13 ALU shift composite:

| L29 sub-stage                            | source                                            |
|------------------------------------------|---------------------------------------------------|
| `L29.ffn.precompute_stage.shl_precompute`| `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:70` (`make_precompute`)  |
| `L29.ffn.precompute_stage.shr_precompute`| same op                                           |
| `L29.ffn.select_stage.shl_select`        | `alu_ops.py:105` (`make_select`)                  |
| `L29.ffn.select_stage.shr_select`        | same op                                           |
| `L29.ffn.getobd_stage.ge_to_bd`          | `alu_ops.py:140` (`make_getobd`)                  |

All five ops carry `opcodes={"OP_SHL", "OP_SHR"}` — the Tier A
opcode-mask gate zeros every residual write outside SHL/SHR
(`alu_ops.py:67`). Under XOR_BASIC (`IMM 0xFF; PSH; IMM 0xD5; XOR; EXIT`),
neither OP_SHL nor OP_SHR ever fires, so the L29 composite's contribution
to OUTPUT_LO is structurally zero.

## Oracle replay result

`replay_expected_diff.py` does not yet support XOR (parser table is
ADD/SUB/MUL only at `replay_expected_diff.py:72-77`; `symbolic_forward.py`
only declares `OP_ADD/SUB/MUL`). Used `ADD` as a proxy ALU step for
the cascade probe (same residual amplification class per
`COLLAPSED_STEP_REAL_SURFACE`):

```
python c4_release/tools/replay_expected_diff.py \
    --program "IMM 0xFF; PSH; IMM 0xD5; ADD; EXIT" \
    --dim OUTPUT_LO --declarations-only --atol 0.5
First divergence at block 0 (step=0 position=0 dim=OUTPUT_LO+0)
  — expected 1, got 0; suggested op: _layer0_threshold_attn_dep_anchor
```

With atol=1.0 the diff shifts to block 6 (`nibble_copy_ffn`,
`layer8_multibyte_fetch`). With atol=0.5 it pins to block 0 because the
oracle's `expected_trace` is **block-invariant**: it expects OUTPUT_LO
to be the AX byte0 nibble at every block, but the actual writers don't
fire until L9+. This is a documented oracle behaviour
(`dim_oracle.py:746-748`: "block-invariant"). The diff does NOT localize
amplifiers — it localizes "writer never fires / wrong cell" only.

## Per-rule contribution table — L29 (full-model index)

| sub-stage              | rules                  | fires on XOR? | OUTPUT_LO write magnitude          |
|------------------------|------------------------|---------------|------------------------------------|
| `shl_precompute`       | 28 units               | **No** — OP_SHL/SHR mask=0 | 0 (mask-zeroed)         |
| `shr_precompute`       | 28 units               | No            | 0                                  |
| `shl_select.flat_ffn`  | 528 units              | No            | 0                                  |
| `shr_select.flat_ffn`  | 528 units              | No            | 0                                  |
| `getobd.ge_to_bd.ffn`  | 64 units               | No            | 0 (`opcode_mask` zeros writes)     |

Total L29 contribution under XOR_BASIC: **0** (opcode-gated off).

## Conclusion

The "+3,868 at L29" hypothesis from the B2.2 brief does not apply to
`XOR_BASIC`. The original cascade documented in
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` was for the binary-pop ALU
class (SUB/DIV/MOD/SHL/SHR + MUL_overflow + Shape-A CMP), which exercises
the L20 (`layer16_lev_routing` → originally "L28") and L21
(`l10_post_ops_combined` → originally "L34") rules.

For XOR specifically:

1. L29 (SHL/SHR composite) is opcode-mask gated off.
2. `dim_oracle` does not yet model XOR semantics
   (`dim_oracle.py:405-422` covers ADD/SUB/MUL/EXIT; unknown opcodes are
   treated as no-op).
3. `replay_expected_diff` parser does not accept XOR
   (`replay_expected_diff.py:72`).

To re-localize the XOR-class amplifier, the next step is:

1. Extend `_OPCODE_TABLE` in `replay_expected_diff.py` and
   `_OPCODE_NAMES` in `symbolic_forward.py` to carry OP_XOR=15 / OP_OR=14
   / OP_AND=16 / OP_EQ..OP_GE=17..22 / OP_SHL=23 / OP_SHR=24.
2. Extend the `ReferenceOracle.all_states()` dispatcher in
   `dim_oracle.py:405-422` with the bitwise/CMP/shift cases (mirroring
   `soft_vm.py:OP_XOR=15` semantics).
3. Re-run the replay against OUTPUT_LO with XOR_BASIC and look for the
   first amplifying block — almost certainly L20 or L21 (the doc's
   "L28"/"L34"), NOT the L29 SHL/SHR composite.

## Fix attempted

None. The L29 hypothesis is structurally inconsistent with the XOR
opcode-gate at L29, so no single-rule guard at L29 would change XOR_BASIC
behaviour. Per `feedback_single_rule_fixes_are_zero_sum.md`, a
speculative L29 guard would have been zero-sum.

## Cross-references

- Replay tool: `c4_release/tools/replay_expected_diff.py`
- Dim oracle: `c4_release/neural_vm/unified_compiler/dim_oracle.py`
- L29 ops (ALU shift composite): `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-220`
- Prior attribution (L27/L28/L34, full-model indices):
  `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`
- Joint fix recipe: `c4_release/docs/COLLAPSED_STEP_JOINT_FIX_PLAN_2026_06_07.md`
