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

---

## RESOLVED (2026-06-16) — `full_width_byte_emission` + AX_CARRY value re-point

**edge_literal 1031-1045 now PASS 15/15 with `C4_AX_BYTE1_FULL_WIDTH=1`.** The
two-part build landed; flag default-OFF (the shipping build is byte-identical,
hash `7474f269…`, smoke 51/0).

### The "no value-faithful source" verdict was a STATIC-REGISTRY MISREAD
The original "`OUTPUT_HI` / `AX_FULL_HI` / `AX_CARRY_HI` empty at every block"
diff read the **static dim_registry positions** (`AX_CARRY_LO`=328). The
flag-OFF build already repacks dims (the widen moves `AX_CARRY_LO` 328→362), so
the static-328 probe read a **dead cell**. At the BUILT layout position
(`compile_full_vm_dynamic()[1].dim_positions`, the memory note
`feedback_probe_dims_use_built_layout_not_static_registry`) the source IS
present: **`AX_CARRY_LO+lo` / `AX_CARRY_HI+hi` carry byte-1's nibbles
value-faithfully (cell == nibble, NO offset)** at the byte-1 predictor row, on
an IMM step. A clean nibble sweep 0..F confirms `AC_LO am == lo`, `AC_HI am ==
hi` (`tools/probe_axcarry_byte1.py`, `tools/probe_byte1_builtscan.py`).

### The fix (two coordinated halves, both in `full_width_byte_emission`)
1. **Emission half** (the prior agent's `make_ax_byte1_full_width_emission_op`):
   the 256-cell `AX_BYTE1_FULL_WIDE` band + un-aliased LM-head columns 16..255.
2. **Value re-point** (this lane): `make_ax_byte1_full_width_fill_op` — an L25-
   tail FFN whose 240 rules reconstruct byte-1 into the wide band by AND-ing the
   `AX_CARRY_LO[lo]` + `AX_CARRY_HI[hi]` nibble pair, gated `OP_IMM` (IMM scope —
   ADD/SUB hold the arithmetic CARRY here, NOT byte-1) + `IS_BYTE` + a hard
   `NOT MARK_AX` blocker (marker+0 holds byte-0 at ~40). Generator extended with
   a `dump_nibble_lo/hi(+offset/weight)` + `dump_nibble_max` nibble-pair source.

### Gates (flag ON unless noted)
- edge_literal 1031-1045 full_trace: **5/15 → 15/15**.
- 1096 arith control ids 0-60: **54/61 unchanged** (no regression).
- byte-identity flag-OFF == golden (whole-model hash test passes); smoke 51/0.

### Remaining blocker for DEFAULT-ON: head-count width sensitivity
The 256-cell band pushes d_model 1080→1336, and the head-dim-preserving
auto-widen jumps **n_heads 10 → 13**. That head-count change perturbs the
documented width-sensitive tail ops → **2 smoke fails flag-ON**
(`test_mul_overflow` 500, `test_shl_8bit` 256 — both emit 0). This is the same
MUL/L10-tail width wall (`project_l10_tail_bank_width_sensitive`,
`project_mul_div_mod_arch_blocked`: "the 32-dim MUL_RESULT_HI family forces
d_model up which ALONE regresses bnz"), NOT a fill bug (the fill is OP_IMM-gated
and never fires on MUL/SHL). Making the feature default-ON needs a
**head-count-stable band packing** (a narrower band, or packing into existing
free dims without crossing the head-multiple) — a separate width-management
effort. Until then the feature ships flag-OFF.

---

## NARROW REBUILD (2026-06-17) — `AX_BYTE1_HINIB` 16-cell high-nibble band + carried-step source (`C4_AX_BYTE1_HINIB`)

The 256-cell `C4_AX_BYTE1_FULL_WIDTH` build above was NOT robust on the
**faithful CPU interpreter** (`tools/interp_oracle_gate.py`, byte-for-byte ==
neural): with the flag ON, edge_literal still **FAILed at step=1** (the
persisting-AX EXIT step), not step=0. The "15/15 GPU PASS" was measured on a
decode path that only checked the fresh IMM step. Two independent gaps in the
256-cell build: (a) the fill is `OP_IMM`-gated so it fixes ONLY the fresh IMM
step — on the carried step `return <v>` re-dumps AX, and `OP_IMM=0` there, so the
high nibble reverts to mod-16; (b) the 256-cell band needs +3 heads (default-ON
blocker above).

### What the byte-1 emission actually is (corrects the stale H1/H2/H3 story)
At the BUILT layout the LM head emits the byte-1 token as a **FACTORED nibble
pair** (probed spec_k=0, `tools/probe_ax_b1_hinib_cpu.py`):
`head.weight[v, OUTPUT_LO.*.-1+(v&0xF)] = 5.0` (LOW nibble) +
`head.weight[v, OUTPUT_HI.*.-1+(v>>4)] = 5.0` (HIGH nibble) + a
`STACK0_B0_DUMP_H3+(hi+4)` carry column. The LOW nibble emits correctly on
every step (carried via `H{1,2,3}_DUMP_OUT`); the HIGH nibble is the ONLY thing
lost, because an L14 default rule (`l14_ops.py:1852`,
`OUTPUT_HI_THIS_STEP+0=+50, +{nonzero}=-5000/S`) FORCES the byte-1 high nibble to
zero → every byte1≥16 collapses to `byte1 & 0x0F`.

### The narrow fix (two coordinated halves, both flag-gated default-OFF)
Only a **16-cell** high-nibble emission axis is needed, not a 256-cell value
band → d_model 1090→**1199**, n_heads 10→**11** (+1 head, not +3).
1. **Emission** (`make_ax_byte1_hinib_emission_op`): for v=16..255, add
   `head.weight[v, AX_BYTE1_HINIB+(v>>4)] = 5.0` (the un-aliased high nibble)
   AND a MIRROR of the carried LOW-nibble DUMP cell that token `v&0xF` already
   reads (`_ax_byte1_lo_dump_cell`: H1/H2/H3_DUMP_OUT). The low-nibble mirror is
   load-bearing — without it, on a carried step the within-high-nibble-group
   low-nibble tie-break falls back to the unreliable fresh `OUTPUT_LO` band and
   a same-high-nibble neighbour (e.g. 0x20 vs 0x25) wins.
2. **Fill** (`make_ax_byte1_hinib_fill_op`, L25 tail after the tail correctors):
   lights `AX_BYTE1_HINIB+hi` from a value-faithful high-nibble SOURCE present on
   BOTH steps — `H3_PREV_STEP+(4+hi)` (the existing AX byte-1 carry head's H3
   band, which already transports the byte-value H3 high-nibble one-hot
   cross-step; am==4+hi, v~12 on step0 AND step1). The carried-step source is
   exactly what the `OP_IMM`-gated 256-cell fill missed.

### The fill's row gate (THREE discriminators — the carry broadcasts widely)
`H3_PREV_STEP+(4+hi)` is broadcast by the carry head to the byte-1 row of EVERY
register (PC/SP/BP/STACK0) AND to byte2/3 of AX. A naive source-only gate
corrupts all of them (observed: byte2→0x10, PC byte-1→0x10 breaking the EXIT
PC). The gate needs all three:
  * `BYTE_INDEX_0+0` as the **multiplicative SwiGLU gate** (byte-1-only: ~1 at
    byte1, ~0 at byte2/3) — must be multiplicative, an additive byte-index term
    is swamped by the source's ~12 magnitude;
  * `SE_REG_AX_PRESENT` (additive, weight 4) as the **AX-register discriminator**
    (~4 at AX byte rows, ~1.48 SP, ~0.54 BP, ~0 PC);
  * the `H3_PREV_STEP+(4+hi)` source cell (additive) as the **value selector**.
  Threshold 24 passes AX-with-matching-hi and sinks PC/SP/BP and the hi-mismatch.
The HINIB write (`C4_AX_B1_HINIB_WRITE`, default 6 → HINIB cell ~6.9k →
contribution ~34k) is big enough to beat the L14 `-5000/S` OUTPUT_HI zero-default
yet small enough that the ~5-logit OUTPUT_LO low-nibble resolution survives.

### Gates (CPU-only, deferred-verify; parent runs the GPU verdict)
- edge_literal 1031-1045 (`tools/interp_oracle_gate.py`, faithful CPU):
  **2 PASS / 10 FAIL → 11 PASS / 0 FAIL** (the rest CROSS-STEP, the gate's
  conservative autoregressive deferral — `tools/probe_edge_fulltrace_cpu.py`
  decodes their per-step (PC,AX) and confirms **13/13 full_trace PASS**).
- Source range: H3_PREV_STEP reaches hi 1..2 (cells 5,6), covering edge_literal
  (byte1 max 0x26). Higher high nibbles (value≥0x3000) need a wider source carry
  (deferred); the columns/band already cover all 16 high nibbles.
- byte-identity flag-OFF: **whole-model state_dict SHA256 IDENTICAL** to pristine
  HEAD (`dfb7405495001aed…`, verified via a throwaway HEAD worktree). Flag-off
  omits the band (d_model 1090, n_heads 10) and bakes zero columns/rules.
- Default-OFF (`C4_AX_BYTE1_HINIB` opt-in). Probes: `tools/probe_ax_b1_hinib_cpu.py`,
  `tools/probe_b1_hinib_carry_source.py`, `tools/probe_edge_fulltrace_cpu.py`.
