# edge_literal cluster — AX byte-1 HIGH-NIBBLE emission wall (2026-06-15)

**TL;DR.** The `edge_literal` cluster (corpus ids 1031-1045, `return <1435..9849>`)
fails full_trace because the model's AX **byte-1 emission is `byte1 mod 16`** — the
**high nibble of byte 1 (value ≥ 4096) has no emission path at all**, even on a
FRESH IMM step where the value is freshly written. This is the SAME architectural
wall as the documented AX byte-1 dump
([`AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md`](AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md),
memory `project_ax_byte1_dump_is_h1_onehot_wall`), manifested on fresh steps with
byte1 ≥ 16 instead of the carried-step variant. THREE prior agents bounced on the
adjacent byte-1 carry; this is a deliberate multi-session two-part build, NOT a
single declarative-rule fix. No model edit was made.

## How it was found — interpreter-first (the lane's whole point)

`CUDA_VISIBLE_DEVICES="" python tools/interp_oracle_gate.py --ids 1031-1045`
(CPU, ~10 s build) classified all 10 fails IDENTICALLY:

```
FAIL?  id1032:edge_literal_1: 5561  step=0 AX[1] exp=0x15 got=0x05  [LOW-CONF cross-step]
FAIL?  id1034:edge_literal_3: 9647  step=0 AX[1] exp=0x25 got=0x05
FAIL?  id1037:edge_literal_6: 5182  step=0 AX[1] exp=0x14 got=0x04
... (all 10 identical pattern)
PASS   id1031,1033,1035,1039,1041  (byte1 < 16 — emits correctly)
```

The gate marked them **LOW-CONFIDENCE / not attributed** (AX byte ≥ 1, no FFN
writer named) — correctly pointing at a non-FFN-rule (architectural) root rather
than mis-attributing a rule. The CPU pattern is exact: **`got == exp & 0x0F`**
(byte1 mod 16). PASS cases are exactly those whose byte1 < 16.

**Workflow validation (the deliverable the lane was asked for).** A neural argmax
probe on GPU confirmed the interpreter's CPU verdict byte-for-byte:

```
 val    byte1_exp   byte1_neural   matches mod16
 5561    0x15=21     0x05= 5        True
 9647    0x25=37     0x05= 5        True
 9257    0x24=36     0x04= 4        True
 2563    0x0a=10     0x0a=10        True   (PASS: byte1<16)
```

The interpreter-first gate localized the divergence (step 0, AX byte 1, exact
wrong value) on CPU in seconds with ZERO GPU probing for the diagnosis. GPU was
used only to (a) confirm interp == neural and (b) prove the fix is structural —
both confirmations, not diagnosis. **The CPU gate accelerated this materially:
the wrong byte + exact value were known before any model was placed on a GPU.**

## Root cause (GPU-confirmed, spec_k=0)

The AX byte-1 predictor row reads a **marker-distance positional one-hot across
the L0 H1/H2/H3 bands** (born at block ~12, persists byte-identically to the LM
head). `tools/probe_hband_byte1_map.py` shows the LM head's H-band emission
columns are **aliased mod 16**:

```
head.weight[v, H-cell] == head.weight[v+16, H-cell] == head.weight[v+32, H-cell]
  v 0..4  -> H1+(v+2)
  v 5..11 -> H2+(v-5)
  v 12..15-> H3+(v-12)
  v 16..  -> wraps to v mod 16   <-- the high nibble has NO distinct column
```

So the byte-1 emitter has only 16 distinct output cells (`byte1 mod 16`). This
is the L0 "H1 one-hot" the original wall write-up identified.

### The high nibble is ABSENT at the predictor row (not just unemitted)

A lo-nibble-matched diff (byte1=0x24 hi=2 vs byte1=0x04 hi=0, both lo=4) over
EVERY registry band at the byte-1 predictor row, at the final block, shows the
high nibble distinguishes ONLY:

```
byte-1 predictor row (relpos 2):
  SE_OP_NE+0:  hi2=5.6  hi0=17.6     <- comparison side-effect (coarse range flag)
  SE_OP_GT+0:  hi2=12.0 hi0=0.0      <- comparison side-effect (byte1 in [0x20,0x2F])
```

`OUTPUT_HI`, `AX_FULL_HI`, `AX_CARRY_HI` are **empty / byte-identical across all
byte1 high nibbles at this row, at every one of the 49 blocks**. There is **no
value-faithful band carrying byte1's high nibble at the predictor row**. The only
leak is the incidental `SE_OP_LT/GT/NE` comparison flags, which (a) encode a
coarse 16-wide range not the exact nibble and (b) are load-bearing for CMP/branch
on the shared path — repurposing them would corrupt comparisons.

## Why this is NOT a single declarative-rule fix

The existing `make_ax_byte1_dump_head_bake_op` + H1/H2/H3 `_DUMP_OUT` machinery
(commit 28fe39aa) covers byte1 **0..15** — it MIRRORS the same 16-cell H-band
layout, so it cannot represent byte1 ≥ 16 either. Extending it requires a value
SOURCE for the high nibble that does not exist at the predictor row.

A real fix is the documented two coordinated halves (same as the byte-1 carry
wall), on the **shared IMM path** the brief flags as highest-risk:

1. **New over-width band** carrying `byte1 >> 4` (4-bit high nibble, value/4096),
   materialized at the IMM value-decode point where the full operand is present
   (NOT the predictor row — the value is decoded upstream of where it is emitted).
   Declare op-locally via `register_residual_band(...)` (head-dim-preserving
   auto-widen; flag-gated so flag-off is byte-identical).
2. **Cross-position relay** to the byte-1 predictor row (an attention head —
   block budgets are tight per the carry-wall third bounce; the L0 H-band is born
   downstream of any L3/L4 carry, so a LATE-layer back-attending head is required,
   the same host-entanglement constraint that blocked the carry build).
3. **New LM-head columns** for byte tokens 16..255 keyed on the new high-nibble
   band's cells (the current columns alias mod 16).

All three byte-identity-gated together, flag default-OFF first. This is a
deliberate design+build, not a corrective rule stacked on the existing emitter
(which `feedback_single_rule_fixes_are_zero_sum` + the 3 prior bounces show nets
zero). The brief's gate — "every program loads literals, verify add/sub/mul/div/
if/bool no regression after EACH fix" — makes a brittle partial actively
dangerous here.

## Status

- No model edit. Working tree byte-identical to HEAD `233b77c4`; flags-off ==
  HEAD trivially (no flags added).
- edge_literal full_trace PASS unchanged: 5/15 (the byte1 < 16 programs); the 10
  byte1 ≥ 16 programs are blocked on this wall.
- `edge` (id 1020-1025 boundary fails, e.g. edge_256/edge_1000/edge_9999) share
  the identical root — any value ≥ 4096 hits the byte1 ≥ 16 wall; values 256..4095
  have byte1 1..15 and emit correctly.

## Probes

- `tools/interp_oracle_gate.py --ids 1031-1045` — CPU diagnosis (the lane gate).
- `tools/probe_hband_byte1_map.py` — LM-head H-band emission map (the mod-16 alias).
- ad-hoc lo-matched band diff at the byte-1 predictor row (high nibble absent).
