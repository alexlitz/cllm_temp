# L5 byte-decode model-side fix attempt — REVERTED

Date: 2026-06-05
Worktree: `/tmp/c4-l5-bytedecode-fix`
Branch: `l5-bytedecode-fix` (off `01e63287` on `speedup-cache-and-buckets`)
Brief: replace the `b5cf7099` runner-side IMM AX override with a model-side
fix to the L5 byte-decode bug for IMM bytes in `[0xE0, 0xFF]`.

## TL;DR

**No model-side patch landed.** The brief's hypothesis (bug is in L0..L5
byte decode) is contradicted by the raw-model probe: the L0..L5 op chain
does not write to `OUTPUT_LO/HI` at all for the IMM AX-byte0 prediction
position — that band is owned by L6+. A symbolic `DSLInterpreter` walk
over the L0..L5 ops (declarations-only compile) confirms zero
`OUTPUT_LO/HI` writes through layer 5.

The bug **does** exist (probe confirms), but lives downstream of L5 in
the L10/L16 tail rules that bake `byte_value_writes(0xE0, strength=
200_000)` / `byte_writes(0xE8, strength=1_000_000)` etc. These tail
rules carry `MARK_AX:-1e6` or `OP_IMM:-1e9` blockers that should keep
them off the AX-marker IMM rows, but the upstream-broadcast attenuation
("MARK_AX attenuates to ~1e-3 via upstream broadcast", per
`EDGE_POW2_OP_IMM_LEAK.md`) means the blockers are partially crossable
in practice.

An exploratory L16 IMM AX OUTPUT-authority rule family (32 rules,
strength 1e9/S) was added and compiled to confirm whether a late-layer
authority can dominate the leaks. Result: it made things WORSE (broke
the previously-passing `test_imm_exit`, IMM 42 → 20). Reverted.

Per the zero-sum memory note (`feedback_single_rule_fixes_are_zero_sum
.md`, 0/5 historical) and the brief's one-attempt rule, the correct
action is to **document the disconfirmation of the brief's hypothesis**
and decline a further blind patch.

## Probe: raw-model AX byte 0 for all IMM values

Script: `/tmp/probe_imm_raw.py` (monkey-patches
`BatchedPureNeuralRunner._override_register_in_last_step` to no-op for
REG_AX during IMM probes, so the runner reads the RAW model AX byte 0
emission instead of the b5cf7099 override). One model compile.

```text
Total: 256, OK: 224, BAD: 32

Bad bytes (expected -> got):
  0x08 -> 0xE8 (mismatch)
  0xE0 -> 0x01 (mismatch)
  0xE1 -> 0x01 (mismatch)
  ... (15 cases, 0xE0..0xE7 + 0xE9..0xEF -> 0x01)
  0xE8 -> 0xE8 (correct -- 0xE8 happens to match the leak target)
  0xF0 -> 0xE8 (mismatch)
  0xF1 -> 0xE8 (mismatch)
  ... (16 cases, 0xF0..0xFF -> 0xE8)

Failures in [0xE0, 0xFF]: 31/32
  Got values: {0x01: 15, 0xE8: 16}
```

Pattern:

* `0xF_` (16 bytes, hi nibble = 15) → all map to `0xE8` (lo nibble 8,
  hi nibble 14). Hi flipped from 15 → 14; lo flipped (for all but
  0xF8) from {0..7, 9..15} → 8.
* `0xE_` (15 bytes, hi nibble = 14, excluding `0xE8`) → all map to
  `0x01` (lo nibble 1, hi nibble 0). Both nibbles flipped.
* `0x08` → `0xE8` (the EDGE_POW2 known case).

The pattern is two-rule: one rule that produces 0xE8 (LO+8, HI+14) and
fires when FETCH_HI+15 is the imm-hi-nibble signature, plus another
rule that produces 0x01 (LO+1, HI+0) and fires when FETCH_HI+14 is the
imm-hi-nibble signature. Both write with strength large enough to
overwhelm the legitimate L6 IMM FETCH→OUTPUT route (strength 2/S =
0.02).

## L0..L5 disconfirmation via DSLInterpreter

Script: `/tmp/probe_l5_imm.py`. Uses
`compile_full_vm_dynamic(strict=False, declarations_only=True)` to get
the op layout without baking, then walks L0..L5 symbolically for 10
representative IMM byte values (0x00, 0x0A, 0x55, 0xD5, 0xDF, 0xE0,
0xE7, 0xE8, 0xF0, 0xFF).

Result: every byte produces `OUTPUT_LO/HI/OUTPUT_HI_THIS_STEP = {}` (no
writes) through L0..L5. The only ops writing OUTPUT_LO/HI in the full
declared op chain are L19 (`layer16_lev_routing`, the one carrying the
bug rules) and L21 (`post_l9_bz_bnz_pc_override`). The brief's
hypothesis that L5 is the byte-decode layer is empirically false.

This matches the earlier
[`PSH_STACK0_CHAIN_L0_L5_CLEARED_2026_06_04.md`](PSH_STACK0_CHAIN_L0_L5_CLEARED_2026_06_04.md)
finding for a different cluster (CMP): L0..L5 produce byte-identical
OUTPUT/STACK0/ALU/AX_CARRY state. The bug always lives in L6+ routing
or L10/L16 tail rules.

## What I tried: L16 IMM AX OUTPUT authority

Added 32 rules to `_layer16_lev_routing_rules` (16 LO + 16 HI),
modelled on the existing `l16_psh_mem_addr0_restore_*` pattern:

```python
imm_ax_output_conditions = (
    ("OP_IMM", 1.0),
    ("MARK_AX", 1.0),
    ("MARK_PC", -1_000_000.0),
    ("MARK_SP", -1_000_000.0),
    ("MARK_BP", -1_000_000.0),
    ("MARK_STACK0", -1_000_000.0),
    ("MARK_MEM", -1_000_000.0),
    ("IS_BYTE", -1_000_000.0),
    ("OP_EXIT", -1000.0),
    ("OP_JMP", -1000.0),
)
authority_strength = 1_000_000_000.0 / S  # 1e7 per-cell
for k in range(16):
    rules.append(multi_way_and_rule(
        name=f"l16_imm_ax_output_authority_lo_{k}",
        conditions=imm_ax_output_conditions,
        threshold=2.5,
        gate=f"FETCH_LO+{k}",
        writes=(
            (f"OUTPUT_LO+{k}", authority_strength),
        ) + tuple(
            (f"OUTPUT_LO+{other}", -authority_strength)
            for other in range(16) if other != k
        ),
    ))
# (identical pattern for HI)
```

Updated `_L16_FFN_UNIT_LAYOUT` (792 → 824) and `ffn_units_used` on the
op (792 → 824).

### Why it failed

`test_imm_exit` (`IMM 42; EXIT`, expected 42) returned **20** with the
authority active — even though 42 = 0x2A was a previously-PASSING IMM
case. So the new rules broke a passing case rather than fixing the
broken ones.

Most likely cause: the `MARK_PC/SP/BP/STACK0/MEM/IS_BYTE: -1e6`
blockers I copied from the existing PSH MEM addr0 family are
**crossable** at MARK_AX because upstream broadcast attenuates
MARK_AX/OP_IMM to ~1e-3 (the same attenuation phenomenon noted in
`EDGE_POW2_OP_IMM_LEAK.md`). With `OP_IMM ≈ 5` from L5 decode at the
clean AX marker, the positive sum is +6; with even tiny MARK_PC
residual at the AX marker (say 1e-3), the negative contribution is
-1000, dropping the score well below the threshold=2.5. So the rule
fires conditionally on residual noise and the gate-times-write
arithmetic produced unintended OUTPUT_LO/HI tilts.

A correct authority rule needs sign-aware competition aware of the
upstream broadcast attenuation; or, equivalently, needs to read the
high-magnitude `OUTPUT_LO+8` / `OUTPUT_HI_THIS_STEP+14` signals
*directly* (mirroring how `l16_psh_mem_addr0_restore_lo_8` reads
`OUTPUT_LO+8 == 1` as evidence the MEM addr0 byte was already
materialised). Without that signal-driven gating, a blind authority
rule can only hurt.

Reverted via `git checkout
c4_release/neural_vm/unified_compiler/ops/l16_ops.py`.

## Smoke before/after

Smoke baseline (no fix, override active, `b5cf7099` parent state on
this branch HEAD `01e63287`): **40/52 passed** (per the b5cf7099 commit
message; the IMM AX override recovered `test_xor_basic`,
`test_add_16bit`, `test_add_carry_cascade`).

After my failed authority attempt: `test_imm_exit` REGRESSED (was
passing, now fails IMM 42 → 20). Not run to completion. **Reverted.**

After revert: smoke returns to 40/52 baseline.

## What would actually fix this

Per the probe + EDGE_POW2 cross-reference, the model-side fix needs to:

1. **Identify the exact rules** producing 0xE8 (for 0xF_ inputs) and
   0x01 (for 0xE_ inputs) at the AX marker. Candidates from the grep
   are `tail_lea_local_ax_marker_byte0_e8`
   (l10_ops.py:6400, `byte_writes(0xE8, strength=1_000_000)`) and
   `l16_psh_mem_addr0_e0_from_sp_no_addr_src` (l16_ops.py:768,
   `byte_value_writes(0xE0, strength=200_000)`), but their gating math
   makes them improbable firers at MARK_AX in clean inputs. The actual
   firing rule(s) for the observed 0xF_→0xE8 and 0xE_→0x01 pattern
   need to be pinpointed via residual-stream introspection (e.g. an
   activations-dump probe at the AX-marker position, comparing rule
   firings between IMM 0x55 (passing) and IMM 0xFF (failing)).

2. **Strengthen the blockers on those rules** with the same
   attenuation-aware constant as `EDGE_POW2_OP_IMM_LEAK.md` used
   (`OP_IMM: -1e9` instead of `-1e6`). Or add an `OUTPUT_LO+8` /
   `OUTPUT_HI+14` evidence-positive on the rule (i.e. fire only when
   those output dims are ALREADY large, so the rule reinforces an
   intended-emission shape rather than constructing one out of clean
   FETCH state).

3. **Validate via a per-byte probe** before running full smoke. The
   probe at `/tmp/probe_imm_raw.py` is the right harness: monkey-patch
   the override off, sweep 256 bytes, count failures in `[0xE0,
   0xFF]`. The byte-identity baseline target is 0 failures there.

This requires another compile budget to localise the firing rule, then
another to verify the fix. I'm out of budget for this session.

## Files touched (during the failed attempt, all reverted)

* `c4_release/neural_vm/unified_compiler/ops/l16_ops.py` —
  `_L16_FFN_UNIT_LAYOUT` (792→824), `ffn_units_used` (792→824), added
  32 `l16_imm_ax_output_authority_{lo,hi}_{k}` rules. **REVERTED.**

## Files added (this doc only)

* `c4_release/docs/L5_BYTEDECODE_FIX_ATTEMPT_2026_06_05.md` (this
  file).

## Off-tree artifacts (not committed)

* `/tmp/probe_imm_raw.py` — the 256-byte raw-model IMM AX probe.
* `/tmp/probe_imm_results.json` — the probe output (32 failures
  listed).
* `/tmp/probe_l5_imm.py` — the L0..L5 DSLInterpreter symbolic walk.

## Cross-references

* `EDGE_POW2_OP_IMM_LEAK.md` — earlier identification of `0xE8` leak
  pattern for IMM=8 via `l16_psh_mem_addr0_e0_from_sp_no_addr_src`; that
  fix is in place (`OP_IMM: -1e9`) but does not catch the
  0xF_→0xE8 / 0xE_→0x01 pattern observed here.
* `PSH_STACK0_CHAIN_L0_L5_CLEARED_2026_06_04.md` — earlier
  disconfirmation of L0..L5 hypothesis for a different cluster
  (matches the L0..L5 trace here).
* `b5cf7099` — the runner-side AX override that masks this bug.

## Confidence

* **High** that the bug exists and the pattern is `0xF_→0xE8` /
  `0xE_→0x01` for the 31 affected bytes in `[0xE0, 0xFF]` (direct
  raw-model probe).
* **High** that the bug is NOT in L0..L5 (DSLInterpreter symbolic
  trace + the architectural fact that L0..L5 do not write
  OUTPUT_LO/HI).
* **High** that the brief's "L5 nibble decode wraps at 14"
  hypothesis is empirically false.
* **Medium** that the bug is in the L10/L16 tail-rule family
  (gating evidence + the EDGE_POW2 cross-reference), but the
  specific firing rules for the 0xF_→0xE8 and 0xE_→0x01 patterns
  were not localised in this session.
* **High** that a blind authority rule in L16 does not fix this
  (the failed attempt regressed `test_imm_exit`).
