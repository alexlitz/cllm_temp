# Collapsed-step joint L28+L34 fix plan (2026-06-07)

Multi-rule structural fix recipe to remove the `f3342968` collapsed-step
binary-ALU override in `batched_pure_neural.py:2147-2187`. Companion to
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` (attribution doc).

The override is load-bearing for 9 binary-pop opcodes
(SUB/DIV/MOD/SHL/SHR + MUL_overflow + Shape-A CMPs EQ/GT/GE true). Three
single-rule L28 attempts (`1ed6b5b6`, `cd53c076`, `b91e3cd2`) were all
reverted. Per `feedback_single_rule_fixes_are_zero_sum.md`, the fix must
be joint.

## Cascade quantification (closed-form)

At IMM-M step 2 of `IMM 50; PSH; IMM 8; <binop>; EXIT`, position 132
(STACK0 marker), per `/tmp/probe_block_chain.py`:

### Stage 1 — L27 `layer15_nibble_copy` (initial spike)

File: `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1746-1782`.
Lowerer: `lower_l15_nibble_copy_ir` — 16 LO + 16 HI nibble copies from
the upstream PSH'd byte band. At STACK0 of the prior step, PSH wrote
byte `0x32 = 50`, so this step's L27 copy fires with:

- `OUTPUT_LO+2` += 40 (low nibble of 0x32 = 2)
- `OUTPUT_HI_THIS_STEP+3` += 40 (high nibble = 3)

Strength `40` is the lowerer's nudge magnitude (see
`_set_nibble_copy_ffn` heritage). Semantic role: legitimate for
LI/store-relay paths. **Cannot be lowered without breaking those.**

### Stage 2 — L28 `lev_stack0_byte0_preserve_*` (×97 amplifier)

File: `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:267-300`.
32 rules: `l16_lev_stack0_byte0_preserve_{lo,hi}_{k}` for k=0..15.

Conditions (`lev_stack0_preserve_conditions`, lines 267-279):

| dim | weight |
|---|---|
| `OP_LEV` | +1.0 |
| `MARK_STACK0` | +1.0 |
| `HAS_SE` | +1.0 |
| `BYTE_INDEX_0` | +1.0 |
| `MARK_PC/AX/SP/BP/MEM` | -8.0 each |
| `IS_BYTE`, `MEM_STORE` | -10.0 each |

`threshold=4.5`, `gate=OUTPUT_LO+k` (or HI), `write=OUTPUT_LO+k *= 0.5`
(i.e. `lev_stack0_preserve_strength = 50.0 / S` where `S=100`).

**Self-feedback math at IMM step (not LEV):**
- `OP_LEV ≈ 0` (we're at IMM), but `MARK_STACK0 + HAS_SE + BYTE_INDEX_0
  = 3`, all `MARK_*` negative dims are 0 at STACK0 marker, so the
  pre-gate activation `= 3` — **below threshold 4.5**.
- The cascade observation (×97 → +3,868) implies the rule is NOT
  meant to fire at IMM but does anyway through some softmax/relu
  margin on the gate side: `gate = OUTPUT_LO+2 = +40` rectifies and
  feeds into the FFN's down-projection. With 32 rules sharing the
  same condition gate, cross-block recurrence accumulates the +40
  spike multiplicatively per block (L28's output += 0.5×gate; gate
  reads from the running residual including L28's own write because
  the residual stream is in-place per-block).
- Closed form per block: `out_n = 1.5 × out_{n-1}` ignoring threshold
  rectification; 40 × 1.5^11 ≈ 40 × 86 ≈ 3,440. Empirical 3,868
  matches at L28 (single block, but the rule family of 32 dims
  multi-applies via gate self-reference inside one FFN pass).

### Stage 3 — L34 `stack0_pop_loaded_output_rules` (final crusher)

File: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:4007-4052`.
256 rules: `tail_stack0_pop_loaded_byte_{lo|hi<<4:02x}` for all (lo,hi)
≠ (0,0). Per-rule:

- `conditions = base_conditions + ((OUTPUT_LO+lo, 0.1),
  (OUTPUT_HI_THIS_STEP+hi, 0.1))`
- `base_conditions` sum at IMM-STACK0: `MARK_STACK0 + HAS_SE + CMP+3·0.5
  ≈ 2.5` (others zero or strongly negative on guards).
- `threshold = 10.5`, `gate = MARK_STACK0` (always 1 here),
  `writes = byte_writes(value, strength=500.0)`.

**Over-fire math with carried-in OUTPUT_LO+2 = +3868:**

- The (lo=2, hi=*) sub-family: `(OUTPUT_LO+2, 0.1) × 3868 = +386.8`.
- Pre-gate activation = 2.5 + 386.8 = +389.3 → crosses 10.5 by 378.
- 16 (lo=2, hi=0..15) rules fire at full strength: each writes
  `value | (hi<<4)` to OUTPUT_LO/HI with strength 500.
- Similarly (hi=3, lo=*): another 16 rules fire from OUTPUT_HI+3=+40.
- Sum across 32 firing rules × 500 strength × 16 byte-write dims ×
  large multiplier through scale factor → observed -6.29e8 dominant
  suppressor (the negative comes from the byte-writes being mis-keyed:
  the broad rule family was designed to fire only for the **single
  loaded byte** but with OUTPUT_LO+2 swamped the family fires for ~256
  values simultaneously, summing destructively).

## Joint fix recipe

Three coordinated edits. **Each must be byte-identity-gated via
`compare_symbolic_to_lowered_ffn` before commit.** Verify with
`/tmp/probe_block_chain.py` re-created from `REMOVAL_2_DEEP_DIVE_2026_06_06.md`.

### Edit A — L28: starve the gate self-feedback

`c4_release/neural_vm/unified_compiler/ops/l16_ops.py:267-300`.

Add a strong positive `OP_LEV` requirement so the rule cannot fire at
IMM step. Current conditions include `OP_LEV +1.0` but with
threshold 4.5 and other positive dims summing to ≥3 the marginal
`OP_LEV ≈ 0.5` (carry-over neural emit) suffices. Tighten:

```python
lev_stack0_preserve_conditions = (
    ("OP_LEV", 5.0),          # was 1.0; force LEV requirement
    ("MARK_STACK0", 1.0),
    ("HAS_SE", 1.0),
    ("BYTE_INDEX_0", 1.0),
    ("OP_IMM", -10.0),        # NEW: explicit IMM blocker
    ("OP_PSH", -10.0),        # NEW: explicit PSH blocker
    ("MARK_PC", -8.0),
    ("MARK_AX", -8.0),
    ("MARK_SP", -8.0),
    ("MARK_BP", -8.0),
    ("MARK_MEM", -8.0),
    ("IS_BYTE", -10.0),
    ("MEM_STORE", -10.0),
)
# Threshold accommodates the new OP_LEV·5 weight at the LEV step:
# at real LEV: 5 + 1 + 1 + 1 = 8 ≥ 7.5 ✓
# at IMM:     0 + 1 + 1 + 1 = 3 < 7.5 ✗ (cuts the amplifier)
```

Bump threshold from 4.5 → 7.5. This is the safer variant of the
reverted `1ed6b5b6` (which used weight 2.0 / threshold 5.5; per the
attribution doc, that was empirically insufficient because L34's
0.1-weight condition still over-fired on L27's raw +40).

### Edit B — L34: raise threshold to swallow the residual spike

`c4_release/neural_vm/unified_compiler/ops/l10_ops.py:4007-4052`.

The 0.1-weight on `OUTPUT_LO+lo` and `OUTPUT_HI_THIS_STEP+hi` was
calibrated assuming OUTPUT_LO peaks at ~50 (legitimate L15 memory load
into STACK0). With Edit A killing the ×97 amplifier, the residual at
L34 falls back to L27's raw +40. Adjust:

```python
# Was 0.1 each; threshold 10.5. Raise condition weights so a real
# +50 load still fires (5.0) but +40 residual does not (4.0).
conditions=base_conditions + (
    (f"OUTPUT_LO+{lo}", 0.05),               # was 0.1
    (f"OUTPUT_HI_THIS_STEP+{hi}", 0.05),     # was 0.1
),
threshold=12.0,                              # was 10.5
```

New activation gates:
- legit L15 load (+50, +50): `2.5 + 5.0 + 5.0 = 12.5` ≥ 12.0 ✓
- IMM residual after Edit A (+40, +40): `2.5 + 4.0 + 4.0 = 10.5` < 12.0 ✗
- Pre-Edit-A amplified +3868: still fires (we WANT belt-and-braces;
  Edit B alone is insufficient — confirmed by the 1ed6b5b6 revert
  history where dropping L28 alone didn't help).

### Edit C — L34: cap the broad-family runaway

Even with Edit B, an unexpected residual could still summon all 256
rules. Add a per-rule mutual exclusion using `OUTPUT_LO+lo` as a
hard requirement (already there at 0.05) AND require the SUM of the
two condition dims be in tight band by adding a small negative on
the off-lo and off-hi positions is structurally impossible in
`multi_way_and_rule` — instead, lift the rule family into the
`stack0_store_top_e0_output_rules` pattern (line 4054) which uses
`ALU_LO+lo`/`ALU_HI+hi` (the cleaner band) plus a much smaller
`OUTPUT_LO+lo · 0.001` confirmation. Migration:

```python
conditions=base_conditions + (
    (f"ALU_LO+{lo}", 1.0),                   # NEW primary key
    (f"ALU_HI+{hi}", 1.0),                   # NEW primary key
    (f"OUTPUT_LO+{lo}", 0.001),              # confirmation only
    (f"OUTPUT_HI_THIS_STEP+{hi}", 0.001),    # confirmation only
),
threshold=4.5,                               # 2.5 + 1 + 1 = 4.5 on the right (lo,hi)
```

This makes the rule fire only when ALU bands carry the intended byte
(L15-load semantics), matching `stack0_store_top_e0_output_rules`.
The current 0.1-weight OUTPUT condition was a legacy shortcut that
the cascade exploits.

## Verification protocol

Per `feedback_single_rule_fixes_are_zero_sum.md`, run the verifier
BEFORE manually diagnosing.

1. `compare_symbolic_to_lowered_ffn(ir, dim_positions, S=100.0)` after
   each edit — must remain byte-identical on the non-cascade corpus.
2. `decl_verifier.py verify_rule_strength` on the L34 rule family —
   confirm new threshold dominates at the (lo,hi) ALU band hit.
3. `/tmp/probe_block_chain.py` (re-create per
   `REMOVAL_2_DEEP_DIVE_2026_06_06.md`) — assert per-block residual at
   token 132 stays within ±100 across all 36 blocks during IMM-M step.
4. Run `tests/test_smoke_*.py` — must hold ≥45/51 baseline.
5. Only after (1-4) green: remove `batched_pure_neural.py:2147-2187`
   override block and re-run smoke. Target: 45/51 → 45/51 (no regression)
   with 9 binop tests now passing without the override.

## Risk and session budget

- Edit A alone: HIGH risk (3 prior reverts).
- Edits A+B: MEDIUM risk; B catches the residual that broke A-only.
- Edits A+B+C: LOWER risk; C structurally re-keys L34 to ALU bands.
- Cumulative weight-identity surface: 32 L28 rules + 256 L34 rules =
  288 rules touched. Each must verify byte-identical against the
  non-cascade corpus.
- Estimated 3-5 sessions:
  1. Land Edit A standalone; verify L28 residual capped at +40 at L28
     output (probe). Expect smoke unchanged (override still load-bearing).
  2. Land Edit B; verify L34 residual stays sub-1e6 with override
     in place. Expect smoke unchanged.
  3. Land Edit C; verify ALU-band-keyed rules fire identically on
     non-IMM corpus.
  4. Remove `f3342968` override; verify smoke 45/51 → ≥45/51 (9
     binop tests now passing native).
  5. Buffer: regression investigation if any of (1-4) drops smoke.

## What NOT to do

- Do not retry the `1ed6b5b6` pattern (L28 OP_LEV 1.0→2.0,
  threshold 4.5→5.5) — already reverted.
- Do not lower L27's nibble-copy nudge magnitude — that breaks
  LI/store relays (the original semantic role of `layer15_nibble_copy`).
- Do not raise L28's `write_scale` to "preserve more aggressively"
  — the bug is the gate self-feedback, not the write magnitude.
- Do not stash the override edits while iterating; each edit must
  hold byte-identity under the override before removal.

## Cross-references

- Attribution doc: `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`
- Per-block probe details: `c4_release/docs/REMOVAL_2_DEEP_DIVE_2026_06_06.md`
- Override status table: `c4_release/docs/OVERRIDE_REMOVAL_STATUS_2026_06_06.md`
- L28 amplifier: `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:267-300`
- L34 suppressor: `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:4007-4052`
- L27 initial spike: `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1746-1782`
- Override block: `c4_release/neural_vm/batched_pure_neural.py:2147-2187`
- Reverted attempts: `1ed6b5b6` (reverted by `4f627491`), `cd53c076`,
  `b91e3cd2`.

## Confidence

- **High** that A+B+C jointly weaken the cascade below the override's
  threshold (per closed-form math above).
- **Medium** that no byte-identity regression appears on non-cascade
  corpus; the 0.05/0.001 weights are conservative versus current 0.1.
- **Medium** that L27's +40 spike is the only upstream input — if a
  4th upstream writer exists at the same dim, the recipe needs a
  Stage-0 edit.
- **Low-Medium** that the override can be removed in one session
  after A+B+C; staged removal recommended.
