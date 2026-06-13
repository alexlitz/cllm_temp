# AX byte-1 register-dump cross-step carry — LANDED (DEFAULT-ON) (2026-06-13)

The dedicated-band re-architecture of the AX byte-1 register-dump carry — the
campaign's #1 fix — is fully landed and **production-default** (integrated onto
main with mul width=2 default-on). The emission is behind `C4_AX_BYTE1_DUMP`,
now **default ON**; opt out with `C4_AX_BYTE1_DUMP=0` to restore the
byte-identical pre-carry behaviour. The last-mile gate over-fire (SHL/JMP) is
RESOLVED by a two-sided Σ AX_CARRY band-pass (see "Gate" below).

## Results (spec_k=0, GPU 0, canonical runner = efficient ALU)

| build | add 0-49 full_trace | sub 50-99 full_trace | smoke |
|-------|---------------------|----------------------|-------|
| baseline (pre-carry) | 12/50 | 5/50 | 50/1 |
| `C4_AX_BYTE1_DUMP=0` (opt-out) | 12/50 (byte-identical) | 5/50 | **50/1** |
| `C4_AX_BYTE1_DUMP=1` (**default**) | **40/50** (+28) | **27/50** (+22) | **50/1** |

(The single remaining smoke failure in every column is the pre-existing
`simple_function` JSR/ENT/LEV arch-block — identical to clean main; NOT a
carry regression.) id-0 (`654+114`) passes at the default: the PSH-step byte-1
dump re-emits `0x02` (654) instead of dropping it (142).

## Integration onto main (mul width=2 default-on)

Both the AX carry and the mul width=2 fix independently added the SAME compiler
band-threading machinery (`_PRODUCTION_EXTRA_RESIDUAL_DIMS` + the
head-dim-preserving widen + cache-key threading + `_LIVENESS_NEVER_SHARE`). They
were UNIFIED into one path carrying BOTH band sets:
`{H1_PREV_STEP:7, H1_DUMP_OUT:7, AX_CARRY_OVERFLOW:1}` (AX, always) +
`{MUL_RESULT_HI_LO:16, MUL_RESULT_HI_HI:16}` (mul, when `C4_MUL_WIDTH2`). The
combined widen is still head-dim-preserving `872 -> 981` (`n_heads 8 -> 9`, base
head_dim 109). The forward-declare-before-`add_op` (`compiler.pending_extra_dims`)
+ tail-allocate-after path is shared by both families.

The integration surfaced a (layer, head) COLLISION: the mul relay
(`layer13_mul_result_hi_relay`) claims L13 (block 16) head 6 when mul is
default-on, but the AX carry head ALSO pinned head 6 (it was free on the
mul-OFF base). Two ops baking the same head silently clobber the L13 attention
block -> smoke 14/51. Fixed by pinning the carry head to slot 7 (free in BOTH
mul configs). `mul_overflow` stays green (50/1) — the two band sets + the widen
compose.

## Architecture (all landed)

1. **Dedicated bands** `H1_PREV_STEP` (7) + `H1_DUMP_OUT` (7), in the
   production-default `extra_residual_dims` of `compile_full_vm_dynamic`. The
   14-dim widen rounds head-dim-preservingly `d_model 872 -> 981` (`n_heads
   8 -> 9`). Threaded by NAME via `layout.dim_positions`; forward-declared as
   `compiler.pending_extra_dims` before the `add_op` loop (so the carry ops
   validate) and bump-allocated at the tail AFTER it (so the base head_dim 109
   is captured clean). Both bands are in the dim-liveness `_LIVENESS_NEVER_SHARE`
   set — a multi-cell same-width liveness merge left donor residue in
   `H1_DUMP_OUT` (read a stale 1.0 instead of the carried one-hot).

2. **Carry head** (`layer13_ax_byte1_dump_carry`, L13 block-16 head **7**,
   `enable=True`): reads the PREVIOUS step's `H1` one-hot via `H1.*.-1` (SSA
   cross-step, allowlisted in `CROSS_STEP_DOCUMENTED_SAFE`) and writes the
   distinct `H1_PREV_STEP` band UNCONDITIONALLY -> breaks the H1-write 2-cycle.
   K-selection (the key reliability fix): a SHARP signature on slot 0
   (`ADDR_B0_LO+5`, weight 60) confines attention to byte-1 predictor rows
   among the prev step; `ADDR_B1_HI+8` on slot 1 (AX-register, ~4.0 AX / ~0
   PC/SP/BP, program-stable) rejects other registers; the AX_CARRY differential
   on slot 2 (its OWN slot — NOT mixed into the signature slot, which spread
   attention across prev byte 0..3) picks the FRESH prev row over the current
   carried one. Verified: H1_PREV_STEP holds the correct `H1+(v+2)` one-hot for
   v in 0..4 across 654/754/913/432/228.

3. **Dump FFN** (`ax_byte1_dump_repopulate`, L25 tail post_op after
   `tail_bit32_result_correction` AND after the overflow-flag precursor):
   gate-copies `H1_PREV_STEP -> H1_DUMP_OUT` on a carried byte-1 row. MIXED
   dim_map — the legacy gate dims (`ADDR_B0_LO`, `ADDR_B1_HI`, `AX_CARRY`,
   `MARK_*`) resolve from the dynamic *registry*
   (`build_default_registry_dynamic()`) because the model residual carries them
   at registry positions (the declarative `dim_positions` layout differs:
   `ADDR_B0_LO=506` vs registry `12`); the NEW bands (`H1_PREV_STEP`,
   `H1_DUMP_OUT`, `AX_CARRY_OVERFLOW`) resolve from `dim_positions`. Gate
   (balanced AND): `ADDR_B1_HI+8*0.25 + ADDR_B0_LO+5*2 + AX_CARRY -
   1000*AX_CARRY_OVERFLOW` over threshold 4.5 — fires on a genuine CARRY
   (AX_CARRY ~+2.7), dark on fresh IMM/ADD (~-988), on PC/SP/BP (no AX-register
   signal), on AX byte-2/3 (no signature), on memory LOADs (AX_CARRY ~+0.8),
   AND (the gate fix) on SHL/JMP (`AX_CARRY_OVERFLOW` kill — see "Gate").

4. **Head bake** (`ax_byte1_dump_head_bake`, phase 1002): mirrors the H1
   high-byte emission columns onto `H1_DUMP_OUT`
   (`head.weight[v, H1_DUMP_OUT+(v+2)]=5.0`, v in 0..4). Additive; gated by
   `C4_AX_BYTE1_DUMP` (DEFAULT-ON; in both disk- and in-proc cache keys).
   `=0` -> no columns -> byte-identical.

## Gate — the two-sided Σ AX_CARRY band-pass (the resolved last-mile)

The dump-read gate originally over-fired on TWO step classes whose Σ AX_CARRY
sits ABOVE the carry band: SHL/SHR-result (~ +12.85) and JMP (~ +47.86). A
SINGLE linear AND can only LOWER-bound Σ AX_CARRY, so a `>= 4.5` threshold also
admits them. (A naive difference-of-SiLU-steps ceiling was tried and REJECTED:
two unbounded ReLU tails never cancel, so it drove `H1_DUMP_OUT` NEGATIVE on SHR
and flipped the byte-1 argmax to a wrong token — `0x2A -> 0x102A`.)

**Fix = a precursor + a kill condition** (a §524 range check applied to a
CONTINUOUS band). A new 1-dim band `AX_CARRY_OVERFLOW` and a precursor FFN
`ax_byte1_carry_overflow_flag` (same L25 tail block, baked BEFORE the dump)
write `AX_CARRY_OVERFLOW = step(AX_CARRY_HI+2 >= 3.0)`. `AX_CARRY_HI+2` is the
clean discriminator: BOUNDED in the carry band (~0.63-1.31) but large for the
over-fire classes (SHL +6.41, JMP +23.93; `tools/probe_ax_carry_cells.py`). The
dump reads it as a `-1000` AND condition: out of band the AND sum is driven far
below threshold (silu ~= 0 -> `H1_DUMP_OUT` EXACTLY 0, NOT negative -> the real
SHL/JMP byte-1 stays on the normal `H1` path); in band the flag is 0, so the
dump fires byte-identically. The fire cluster (Σ ~2.65-2.70) and the over-fire
classes (>= 12.85) have a wide clean gap, so the cut is robust.

This delivers flag-ON smoke **50/1** (only the pre-existing `simple_function`
fails) and, as a bonus, lifts sub 50-99 to **27/50** (the prior gate's SHL/JMP
over-fire had been corrupting sub cases too). `C4_AX_BYTE1_DUMP` is now
DEFAULT-ON.

Tools: `tools/probe_h1prev_carry.py` (carry-band + AX_CARRY gate at the final
block), `tools/probe_ax_carry_gate_band.py` (Σ AX_CARRY per-program at the dump
read point), `tools/probe_ax_carry_cells.py` (the per-cell `AX_CARRY_HI+2`
discriminator). All must build with `alu_mode='efficient'` to match the
canonical runner.
