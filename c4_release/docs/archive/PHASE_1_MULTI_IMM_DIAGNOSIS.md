# Phase 1 Multi-IMM Diagnosis (2026-05-11)

## Status

Task: investigate the remaining 4 Phase 1 failures in `test_pure_neural_pc.py`
under the assumption they were "multi-IMM AX carry-forward" bugs.

Reality on `main` (commit `fd45c7b`): **13/13 Phase 1 tests fail**, not 4.
The runner-side workaround referenced in the task brief is not present in the
current tree, so the picture has regressed materially.

After the partial fix in this branch (`d3-phase1-multi-imm`):
- 1/13 passes (`test_imm_zero_then_exit`).
- The remaining 12 fail with the wrong value, but the failure mode is now a
  uniform byte broadcast (e.g. `IMM 5 -> 0x05050505 = 84215045`) instead of
  multi-million garbage values (e.g. `IMM 5 -> 67372036 = 0x04040404`).
- The IMM byte-0 emission is now correct (was 0x04 = `val - 1` or 0x101 = a
  marker token shoved into the byte slot).

## Root Causes Identified

### 1. CarryPropagationPostOp fires spuriously at REG_AX_mark (FIXED)

**Symptom:** for IMM 5 EXIT, the residual stream OUTPUT_LO at REG_AX_mark went
from clean `+160` at OLO[5] (correct, L6 IMM bake) to `-12.6 million` at
OLO[5] after L13 in the expanded model.

**Location:** `c4_release/neural_vm/vm_step.py` `CarryPropagationPostOp`
(block 13 in the expanded model, `byte_idx=0`).

**Mechanics:**
- The L10 ALU attention (`block 11` in the expanded model) writes a
  uniform `-217` offset to OUTPUT_LO[0..15], OUTPUT_HI[0..15], CARRY+1, and
  CARRY+2 at every MARK_AX position, even when no ALU op is active. (Source
  is several heads in `_compile_l10_attention` whose gates do not block
  cleanly at REG_AX_mark; see "Open Questions" below.)
- `CarryPropagationPostOp` ADD-unit weights include `MARK_AX = -S*1000`
  (-100000) as the marker suppression, plus `sub_carry_in = CARRY+2` with
  weight `-S*10` (-1000) for mutual exclusion with SUB.
- When CARRY+2 holds the negative residual `-217.11`, the
  `-1000 * -217 = +217000` contribution overwhelms the `-100000` MARK_AX
  suppression, and the unit fires with `up ≈ 51000`, silu = 51000, writing
  `-2/S * silu = -1000` to OLO[lo] and `+1000` to OLO[new_lo]. Across 256
  (lo, hi) pairs, this destroys the L6 IMM signal and produces multi-million
  noise.

**Fix:** in `CarryPropagationPostOp.forward()`, clamp CARRY+0..4 and
OUTPUT_LO/HI to non-negative values before the linear projection (mirrors the
FIX 2026-05-06 pattern in `BDToGEConverter`). The cleaner OUTPUT residual is
also used so subsequent post-ops never see the negative values flipped by
mutual-exclusion weights.

```python
def forward(self, x):
    x_clamped = x.clone()
    x_clamped[..., BD.CARRY + 0:BD.CARRY + 5] = torch.clamp(
        x[..., BD.CARRY + 0:BD.CARRY + 5], min=0
    )
    x_clamped[..., BD.OUTPUT_LO:BD.OUTPUT_LO + 16] = torch.clamp(...)
    x_clamped[..., BD.OUTPUT_HI:BD.OUTPUT_HI + 16] = torch.clamp(...)
    up   = F.linear(x_clamped, self.W_up)   + self.b_up
    gate = F.linear(x_clamped, self.W_gate) + self.b_gate
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, self.W_down, self.b_down)
```

### 2. CarryPropagationPostOp also fires spuriously at AX byte 1+ (FIXED)

**Symptom:** after fixing (1), `IMM 5 -> 0x04040405` — byte 0 correct, bytes
1-3 = `4` (the IMM value minus one). The trace at AX_b1 (pos 7) shows OLO[5]
goes from `+160` (L6 IMM bake leak — see issue 4) to `-3443` after L13. The
SUB-branch carry-prop unit (which decrements: `new_val = (lo + hi*16 - 1)`)
shifts the nibble argmax from 5 to 4.

**Mechanics:**
- At AX byte 1 the unit's W_up sees OLO[5] ≈ 160 (the leaked IMM value, see
  issue 4) and OHI[0] ≈ 175. With weight `+S = +100` on each, that's a
  contribution of +33500, again overwhelming the -950 bias.
- The unit fires with `up ≈ 22000`, silu = 22000, and writes
  `-2/S * 22000 = -440` to OLO[5] and `+440` to OLO[4]. Argmax flips from 5
  to 4.

**Fix:** reduce the W_up weight on OUTPUT_LO/HI from `S = 100` to `S*0.05 = 5`
in `CarryPropagationPostOp._bake_weights()`. This makes the OUTPUT_LO/HI a
weak "tag match" rather than a dominant trigger; only the `add_carry_in` /
`sub_carry_in` signals (still at `+S*2 = +200`) can fire the unit.

### 3. L10 attn OUTPUT pollution at REG_AX_mark — partial fix

**Symptom:** the uniform `-217` offset on OLO/OHI at REG_AX_mark survives all
downstream layers and buries the L6 IMM bake's `+160` at OLO[imm_lo]. The LM
head then prefers marker tokens (logit -10) over the byte token for the IMM
value (logit -541 after suppression).

**Partial fix (in this branch):** added 32 cleanup units to
`_set_layer14_clear_output_corruption` in `vm_step.py` that ADD `+217` to
every OLO/OHI dim at MARK_AX positions when no arithmetic / bitwise /
comparison / shift op is active. After this cleanup OLO[5] returns to ~160 at
REG_AX_mark and byte token 5 wins the LM head argmax (verified for IMM 5,
which now emits the correct byte 0 = 5).

**Open follow-up:** the cleaner fix is to find the L10 attn head that is
actually polluting and fix its gate so it does not fire at REG_AX_mark. The
trace at the L10 attn output shows the same `-217` written to all 16
OLO dims AND all 16 OHI dims AND CARRY+1, CARRY+2 — this looks like a
softmax1 averaging artifact from a head whose gate is not cleanly negative at
REG_AX_mark. Heads 5/6/7 of `model.blocks[10].attn` have non-empty W_v/W_o
writing to OLO[0..15]/OHI[0..15] (script
`c4_release/scripts/debug/check_l10_attn_weights.py`). Their `slot 33` gate
columns evaluate to ~-2500 at REG_AX_mark, which should suppress the head
under softmax1, so something else is keeping them alive — possibly the slot 1
or slot 2 content scores cancelling the gate. A focused diagnosis of heads 5-7
is the next step.

### 4. L6 IMM bake leaks to AX byte 1-3 (NOT FIXED)

**Symptom:** after issues 1+2+3 are fixed, the trace shows OLO[5] ≈ 160 at
**every** AX byte position (positions 5..9 of step 0), not just at
REG_AX_mark. The LM head therefore emits byte 5 at every position. Result:
`IMM 5 -> 0x05050505 = 84215045` instead of 5.

**Likely culprit:** `_set_layer6_routing_ffn` ADD/IMM units have
`IS_BYTE: -S * 10 = -1000` suppression, but the FETCH_LO/HI values at byte
positions (carried by L5 fetch head 0 attention from the bytecode position
into all step 0 positions) are large enough that
`hidden = silu(up + gate_contribution)` still fires. The 2026-05-09 fix note
in that function explicitly mentioned this leak ("even silu(0)≈0.27 at byte
positions started broadcasting the IMM value into AX bytes 1-3"); the fix
was `IS_BYTE = -S * 10`, but that alone is insufficient when OUTPUT pollution
elsewhere already needs to be overcome by amplifying the IMM signal.

Previously (before this branch) the spurious CarryProp firing in issues 1+2
was, by coincidence, masking this leak — it turned the broadcast `+160` at
byte 1+ into a different garbage value (`-217`-ish), which the LM head
treated as marker-fallback. With issues 1+2 fixed, the leak is unmasked.

**Suggested fix:** strengthen the IS_BYTE suppression in
`_set_layer6_routing_ffn` IMM units to `-S * 20` or higher, AND/or add a
companion cleanup unit at L14 that zeroes OUTPUT_LO/HI at AX byte positions
when only IMM is the active op (since IMM writes its result at the AX marker
and byte positions should default to 0 for `val < 256`).

This was not attempted in this branch due to time budget.

## Files Changed

- `c4_release/neural_vm/vm_step.py`
  - `CarryPropagationPostOp.forward`: new override that clamps CARRY/OUTPUT
    inputs to non-negative before the linear projection.
  - `CarryPropagationPostOp._bake_weights`: W_up weight on OUTPUT_LO/HI
    reduced from `S` to `S * 0.05` so OLO/OHI residuals do not dominate the
    activation.
  - `_set_layer14_clear_output_corruption`: added 32 cleanup units (16 OLO +
    16 OHI) that boost `+217` per dim at MARK_AX positions when no ALU op is
    active.

- `c4_release/scripts/debug/trace_phase1_multi_imm.py` (new),
  `c4_release/scripts/debug/trace_phase1_emit.py` (new),
  `c4_release/scripts/debug/trace_carryprop_units.py` (new),
  `c4_release/scripts/debug/trace_unit80.py` (new),
  `c4_release/scripts/debug/trace_block11_delta.py` (new),
  `c4_release/scripts/debug/check_l10_attn_weights.py` (new),
  `c4_release/scripts/debug/check_l10_head_wo.py` (new),
  `c4_release/scripts/debug/check_next_ax.py` (new),
  `c4_release/scripts/debug/trace_h1_at_ax_mark.py` (new),
  `c4_release/scripts/debug/trace_olo5_per_layer.py` (new).

## Test Status

Phase 1: 1/13 pass on this branch (was 0/13 on main).

The single pass is `test_imm_zero_then_exit` because IMM 0 produces a uniform
byte-0 broadcast (`0x00000000 == 0`) which is accidentally correct.

All other tests fail because of issue 4 (L6 IMM bake leak to byte positions).
Once issue 4 is addressed, the remaining 12 should pass.
