# Wave B Phase 1 — Audit + relay-transmission blocker + Phase 2-3 plan

Date: 2026-06-12. HEAD: `3db94ff1`. Smoke baseline (spec_k=0, GPU0, from
worktree): **48 passed / 3 failed {mul_basic, mul_overflow,
simple_function} / 0 xfail** (`tests/test_smoke.py`, 140s). This is the
gate I must not regress.

This doc supersedes the *premise* of the Phase-1 brief and refines
`WALL4_SE_CMP_DECODE_ROW_FRONTIER_2026_06_11_SESSION2.md` (which stopped
at HEAD `8ad47bf4`, smoke 31p/10f, before the EQ + reconcile commits
landed the MARK_AX ordering engine).

## 0. The brief's premise is STALE — the relays are already enabled

- `make_layer11_step_end_operand_relay_op` is **`enable=True`** at this
  HEAD (`all_core_ops.py:301`), scoped to `include_op_name=True` only
  (operand bands `include_ax_carry/alu/cmp/stack0_byte=False`). It relays
  raw `OP_<NAME> -> OP_<NAME>` and lands on **physical block 12 (logical
  L11, 13 heads — has headroom)**.
- A dedicated **SE-tagged** relay, `make_layer9_step_end_operand_relay_op`
  (`l9_ops.py:2146`), is **registered and bakes unconditionally**
  (`all_core_ops.py:236`). It mirrors raw `ALU_LO/HI`, `AX_CARRY_LO/HI`,
  `CMP`, `CMP_GROUP`, `OP_<cmp>` into the `SE_*`-tagged sister dims
  (registry 837..911) at `MARK_SE_ONLY`. This is the "Wave A v2 / L9
  internal relay" prior memory recommended — it already exists.
- The live, working CMP compute is `_layer10_alu_ordering_engine_rules`
  (`l10_ops.py:1486`, 272 units) at **MARK_AX**, the SOLE CMP-flag writer
  for all six cmp opcodes since reconcile commit `11e1c749`. It feeds the
  legacy `ComparisonCombine` post-op (logical L14 / physical block 22)
  which decodes the RAW `CMP[0..3]` at the **MARK_AX row**.

So Phase-1 "enable the relay" is already done in code. The real Phase-1
finding is **the SE-tagged relay transmits NOTHING** (measured below),
and **why**.

## 1. AUDIT — what is at MARK_SE_ONLY vs MARK_AX (CMP/ALU, l9/l10/l11/l12)

| Rule family | File | Gate row | Operand dims it READS | Live? |
|---|---|---|---|---|
| `_layer9_cmp_rules` (272u: hi/lo eq/lt) | l9 | MARK_SE_ONLY | **SE_ALU_LO/HI, SE_AX_CARRY_LO/HI**, gate SE_CMP_GROUP | DEAD (SE bands empty) |
| `_layer10_alu_cmp_combine_rules` (18u override/default) | l10 | MARK_SE_ONLY | **raw CMP+i** (NOT SE_CMP) + OP_<cmp> | DEAD (raw CMP=0 at SE) |
| `_layer10_alu_ordering_engine_rules` (272u) | l10 | **MARK_AX** | raw ALU_LO/HI (A) + AX_CARRY_LO/HI (B) | **LIVE — drives decode** |
| `_layer10_alu_eq_engine_rules` (decode-margin) | l10 | MARK_AX | raw ALU/AX_CARRY | LIVE (margin push) |
| `_layer11_mul_partial_rules` | l11 | MARK_SE_ONLY | **raw ALU_LO, AX_CARRY_LO/HI** (NOT SE_) | STARVED |
| `_layer12_mul_combine_rules` | l12 | MARK_SE_ONLY | **raw ALU_HI, AX_CARRY_LO, TEMP** (NOT SE_) | STARVED |

Two distinct starvation modes:
1. **Transmission failure** — the SE-tagged relay does not populate
   `SE_*` (so the L9 CMP rules, which DO read `SE_*`, never fire).
2. **Wrong-dim readers** — the l10 cmp_combine overrides and the l11/l12
   MUL rules gate on `MARK_SE_ONLY` but read **raw** ALU/CARRY/CMP, which
   live at MARK_AX and are 0 at the SE row. Even a perfect transmission
   fix leaves these inert until they are switched to read `SE_*`.

## 2. ROOT of the transmission failure (decisive, spec_k=0)

Three CPU probes (kept in `tools/`):
- `probe_l9_se_relay.py` — at the LT step's SE row, `SE_ALU_LO` is a flat
  `-1.32..-1.41` band (NO operand one-hot), `SE_AX_CARRY_LO/SE_CMP/
  SE_CMP_GROUP/SE_OP_LT` all cold. The relay's V→O copy contributes only
  its bias floor; **zero attention-weighted operand content**.
- `probe_relay_landing.py` — the SE_ALU relay (head A of
  `layer9_step_end_operand_relay`) physically lands on **block 11
  (logical L10) head 3**, Q[MARK_SE]=10, K[MARK_AX]=10, **slope=0.5**,
  V.read(ALU_LO)=1.0, O.write(SE_ALU_LO)=1.0. Head B lands on block 11
  head 4, slope=1.0. Block 10 (logical L9) carries NO relay head.
- `probe_l9_relay_weights.py` — confirms blk11 slopes
  `[5.0, 1.0, 1.0, 0.5, 1.0, 0.016, 0.008, 0.004]`; head 3 = 0.5.

**Root**: the "L9" SE-tagged relay is mis-placed onto **physical block 11
= logical L10**, landing on heads 3/4 — head 3 is the **PSH STACK0
passthrough** head (`_L10_HEAD_LAYOUT` row 3) whose slope is set to 0.5
by `make_layer10_residual_alibi_slopes_op` (alu_ops.py:2068). Over the
d=29 MARK_AX→MARK_SE gap, slope 0.5 imposes a −14.5 ALiBi penalty that
swamps the L=10 QK match → the relay attends to nothing → only the bias
floor survives. This is the documented **Wall 2** (operand-relay slot
clobber), here on block 11 head 3.

**Why it can't simply be re-homed (in the smoke build):** the smoke gate
builds with **`num_heads=8`** (probe: block 11 has 8 heads, HD=109). All
8 (`_L10_HEAD_LAYOUT` 0..7) are owned (carry/byte-passthrough/PSH-
passthrough/stack0-relay). Block 11 has **no free head**. Head 3 is
load-bearing for PSH STACK0 passthrough = lt/le (memory
`project_operand_gather_hybrid_encoding_is_cmp_alu_root` Wall 2:
re-using/retuning head 3 regresses lt/le). So the relay cannot be
transmitted on block 11 without either widening block-11 head budget or
evicting a load-bearing head. (Block 12 / logical L11 DOES have 13 heads
and headroom — a candidate re-home target, see Phase 2.)

## 3. Even a transmitting relay is inert-or-regressive at the current decode (Wall 4)

The live decode reads **raw CMP at the MARK_AX row** (ComparisonCombine,
block 22; via L3 head-5 `OUTPUT_LO@MARK_AX -> AX_FULL` relay). The
SE-tagged relay + L9 CMP rules write the cascade at the **SE row** — a
different sequence row the decode never reads. So:
- transmission fix alone → SE_CMP populated, L9 CMP fires at SE, but the
  result is on a row the decode ignores → **additive, no gain** (best
  case); or
- if the SE cascade leaks cross-step into the next step's MARK_AX row
  (the documented `lt/le` regression, agents a40f8a22 / B2), → **−2**.

This is exactly why the MARK_AX ordering engine was authored
(`11e1c749`) and why the SE path remains dormant. **Wave B's SE
migration is correct in intent but blocked on a co-designed decode-row
move, not on the relay alone.**

## 4. Phase-1 verdict

- Relay state: **enabled but starved by a block-11-head-3 slope clobber
  (Wall 2); no free head on the 8-head smoke build to re-home it.**
- No code change landed: every available transmission fix is either
  inert at the current decode (Wall 4) or regresses lt/le (load-bearing
  head 3). Per the brief's fallback path, the blocker + the slot/slope
  fix are documented here. Baseline preserved at 48/3/0 (zero edits to
  weight-authoring code; only read-only probe tools added).

## 5. Phase 2-3 roadmap (concrete, multi-session)

**Phase 2 — make the SE row carry clean operands AND consume them.**
Two independent edits, each smoke-checkpointed:
1. **Re-home the SE relay off the contested head.** Move
   `layer9_step_end_operand_relay` heads A/B from block 11 heads 3/4 to a
   genuinely-free, authoritatively-sloped head. Target options, in order
   of risk:
   (a) **block 12 / logical L11** (13 heads, headroom) — but L11 runs
       AFTER L10, so the SE bands would arrive one block too late for the
       L9 CMP rules (those fire in the L9 FFN). Acceptable for the MUL
       readers (l11/l12 run at/after block 12) but NOT for L9 CMP.
   (b) **widen block-11 head budget** (`layer_max_heads` on the L10 attn
       op) to add head 8 as the relay, with an owned shallow slope
       (~0.15) set LAST (after `layer10_residual_alibi_slopes`), per
       `project_attention_dsl_alibi_slope_gap`. This is the clean fix but
       expands the attn head axis (byte-identity + d_model/n_heads
       divisibility checks required; see the d_model-expansion bnz
       regression noted in `project_mul_div_mod_arch_blocked`).
   Checkpoint: `probe_l9_se_relay.py` shows SE_ALU_LO+a / SE_AX_CARRY_LO+b
   one-hot ≈ source; smoke stays 48/3/0.
2. **Switch the wrong-dim SE readers to the SE_ mirrors.** In
   `_layer10_alu_cmp_combine_rules` change `CMP+i` → `SE_CMP+i`; in
   `_layer11_mul_partial_rules` / `_layer12_mul_combine_rules` change raw
   `ALU_LO/HI`, `AX_CARRY_LO/HI` → `SE_ALU_LO/HI`, `SE_AX_CARRY_LO/HI`.
   Byte-identity gate each via `compare_symbolic_to_lowered_ffn`.
   Checkpoint: smoke 48/3/0 (these are currently inert, so the switch is
   additive until transmission lands).

**Phase 2.5 — the decode-row move (Wall 4, the real unlock for CMP).**
Either (a) route the SE-row cmp result into the SAME step's MARK_AX-row
OUTPUT_LO via a within-step SE→AX relay head, OR (b) re-point L3 head 5
(`l3_ops.py` `_ax_full_relay_head_spec`) to read the SE row — WITHOUT
disturbing the AX-row default path lt/le/gt/ge currently rely on. This is
the documented frontier; co-design with the ordering engine so exactly
one writer drives the decode. Checkpoint: lt/le/gt/ge/eq/ne all green at
each sub-step (per-cmp smoke).

**Phase 3 — retire the MARK_AX engines + verify MUL falls out.**
Once the SE path drives the decode (Phase 2.5), retire
`_layer10_alu_ordering_engine_rules` + `_layer10_alu_eq_engine_rules`
flag writes (keep only the eq decode-margin if still needed). With the SE
operands transmitting (Phase 2.1) and the MUL readers on SE_ (Phase 2.2),
`_layer11_mul_partial` / `_layer12_mul_combine` finally see operands at
the SE row → **mul_basic / mul_overflow** should fall out (subject to the
separate L15/L25 OP_MUL OUTPUT-materialization corruptor documented in
`project_mul_div_mod_arch_blocked` — that is an L15/L20 surface, out of
the l9-l12 ALU scope, and is the remaining MUL blocker even after the
relay). Checkpoint: full smoke; targeted mul/cmp clusters in 1096.

## Tools added (read-only, spec_k=0, CPU to dodge shared-GPU OOM)
- `tools/probe_l9_se_relay.py`
- `tools/probe_relay_landing.py`
- `tools/probe_l9_relay_weights.py`
