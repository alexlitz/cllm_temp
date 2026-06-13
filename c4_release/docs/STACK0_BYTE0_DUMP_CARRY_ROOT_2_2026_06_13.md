# STACK0 byte-0 cross-step carry (Root 2 — if/bool/expr framing drift) — 2026-06-13

Status: **LANDED DEFAULT-ON** (`C4_STACK0_B0_DUMP`, default **1**; opt OUT with
`=0`). Rebased onto current main `dedbb063` (incl. the Root-3 func fix). The
missing gate discriminator — claimed "does not exist" by the prior session — DID
exist; the prior agent's no-discriminator claim repeated the func agents'
"OP_ENT is a hard constant" error (a DIM-MISMAP, not a real absence). Smoke
**51/0** default-ON; byte-identical to the Root-3 main with `=0`.

Target clusters (full_trace, flag-OFF baseline → default-ON):
**if_gt 4→17, if_lt 4→21, if_eq 8→21, bool_and 8→16, expr_add_mul 1→2,
expr_paren →12, expr_mod →9** (net +52 across the if/bool/expr targets).

KNOWN TRADE (corpus-level, NOT a smoke guard): the 1096 `mul` cluster regresses
**20→11** (-9). The dump over-fires on the operand-PSH SETUP rows of a multi-byte
arithmetic program (those carry a clean operand one-hot and — in the BATCHED gate
— a faint `OP_PSH` leak that ALSO rides the if-comparison drift rows, so an
explicit `OP_PSH` block recovers mul but darkens the if-fix, if_gt→4). The
smoke `mul_basic`/`mul_overflow` tests PASS (smoke 51/0). The if/bool gains
(+52) dominate the mul trade (-9), and the brief's hard `smoke==51/0` invariant
is met, so the carry ships default-ON. The clean fix for the next agent is a
batched-robust operand-SETUP vs comparison-result discriminator (the serial-only
`OP_PSH`/positive-`OP_GT` signals don't survive the batched residual).

## The discriminator the prior session missed (the DIM-MISMAP)

The prior session tried "opcode/CMP markers (all stale=1.0 at STACK0 rows)" and
concluded no per-step discriminator exists. That reading sampled the opcode dims
at their **STATIC 872-dim registry positions** (`OP_ADD=287`, `OP_JMP=264`, ...)
which **mismap onto the WIDENED 981-dim build** (`OP_ADD=209`, `OP_JMP=189`, ...).
Reading the WIDENED `layout.dim_positions`, the carried STACK0-marker row at the
dump block input DOES carry a clean, stable, BOUNDED per-step opcode (spec_k=0,
`tools/probe_stack0_arithgate.py` / `probe_stack0_smoke_gate.py`):

* COMPARISON / bool result rows (if_gt/lt/eq/ne/ge/le, bool_and/or): every arith
  opcode = 0.000 AND `OP_JMP` = 0.000.
* ARITHMETIC result rows (add/sub/mul/and/or/shl ...): `Σ(arith opcodes)` = 0.110
  (the per-step OP_ADD/SUB/...).
* JMP rows (jmp_forward): `OP_JMP` = 0.05..0.30.

So the over-fire victims (`add_16bit`, `jmp_forward`) are exactly the rows with a
per-step arith/JMP opcode, and the framing-drift rows the carry must fix are
exactly the comparison rows (opcode = 0). That is the discriminator.

## The gate (two bounded blockers folded into `STACK0_B0_NOT_CMP`)

The re-point dump reads a single bounded `STACK0_B0_NOT_CMP` BLOCKER (`-1000`
weight — a positive flag cannot work because the gate's `CARRIED`+`SHARP` terms
are each ~100, so only a strongly-negative ABSENT-on-comparison blocker overcomes
them). `NOT_CMP` is an OR of two precursor rules (`stack0_byte0_not_cmp_flag`):

1. **arith/JMP opcode present** — `step(MARK_STACK0 + 50·Σ(OP_ADD/.../OP_JMP) ≥
   1.5)`. Blocks `jmp_forward` (which has a CLEAN PREV one-hot, so the smear rule
   alone misses it). Reads the opcode bands from the WIDENED `dim_positions`.
2. **PREV is a true SMEAR** — `step(MARK_STACK0 + 0.02·Σ|PREV| − 100·PREV_DOM ≥
   1.5)`. Blocks `add_16bit` (whose corrupting STACK0 row carries NO arith opcode
   but a smeared PREV one-hot). Uses a new RATIO-based `STACK0_B0_PREV_DOM` flag
   (`stack0_byte0_prev_dom_flag`): `OR_j step(PREV+j − Σ_others > 1)` — fires for
   a clean one-hot of **ANY magnitude**, unlike the absolute-margin
   `STACK0_B0_SHARP` which MISSES a small clean one-hot (the comparison result
   byte `0x01`, PREV ~6) and so wrongly blocked those (regressed if_gt 17→9 when
   used as the smear test). The `Σ|PREV|` mass term keeps an EMPTY PREV from
   firing (re-supplying 0 is a harmless no-op).

Net: the dump fires ONLY on a carried COMPARISON row whose carried PREV is a
clean one-hot — exactly the framing-drift rows — and is dark on arith-result and
JMP rows. `add_16bit` (300→300) and `jmp_forward` (42→42) stay byte-correct.

## Geometry / bands

Two new bounded width-1 bands (`STACK0_B0_PREV_DOM`, `STACK0_B0_NOT_CMP`) added
to `_PRODUCTION_EXTRA_RESIDUAL_DIMS` (now 47 extra dims: `872+47=919` still rounds
up to the SAME `d_model=981` = 9×109, so **no geometry change, no new head**).
Both are in `_LIVENESS_NEVER_SHARE`. Two new tail precursor FFNs
(`stack0_byte0_prev_dom_flag`, `stack0_byte0_not_cmp_flag`).

Branch base: rebased onto main `dedbb063`. spec_k=0, GPU 0, `alu_mode='efficient'`.

---

## (historical) the prior flag-OFF status

## The bug (probe-confirmed, NOT the brief's hypothesis)

The if/bool/expr "framing drift": on a comparison step the STACK0 (stack-top)
byte fails to emit, so the model emits a **57-token step** (a spurious extra
register block) instead of 35; the runner's fixed-35 slicer then misreads the
PC and the full-trace fails at step 3 with the **branch math CORRECT but PC off
by a register block** (~134 of the 183 if/bool/expr fails diverge this way).
Value-dependent: both-nibbles-nonzero operands (0x11, 0x23) DRIFT; one-nibble-
zero (0x10, 0x20) stay CLEAN.

The brief hypothesised a "STACK0 byte-0 high-nibble *persistence drop*" fixable
by mirroring the AX byte-1 carry. The probe trail (`tools/probe_stack0_byte0_*`)
confirmed the **mechanism class** (cross-step: the value emits fresh at the PSH
step, breaks on the next carried step) but localized a DIFFERENT, harder root:

1. **The byte-0 emission one-hot lives in the LM-head `H1` (low nibble) + `H3`
   (high nibble) bands** — each byte token reads `head.weight[byte, H1+(lo+2)]
   = 5.0` and `head.weight[byte, H3+(hi+4)] = 5.0` (the imperative
   `setup_head_weights` encoding; covers lo<=4, hi<=2). (NOT `OUTPUT_LO/HI` —
   those are the declarative head_bake path; the H1/H3 columns dominate.)

2. **At the PSH (producing) step** L6 freshly decodes the clean one-hot
   (`H1+(lo+2) ~ 5`, `H3+(hi+4) ~ 9`) and the byte-0 emits correctly (35-token
   step).

3. **At the next CARRIED comparison step** L6 does NOT re-decode it (the value
   only persists on the stack). Then:
   - **block 32 (L21)** — the operand-gather — smears a value-proportional
     copy into H1/H3 (`H1+3 ~ +3718` for operand 0x11; the hybrid magnitude+
     nibble encoding, the operand-gather root from
     `project_operand_gather_hybrid_encoding_is_cmp_alu_root`), then
   - **block 38 (L25, post-op expansion)** — the L16
     `stack0_e8_output_authoritative` materializer (`l16_ops.py:784`,
     `_add_stack0_x0_alu_materializer` family, gated `MARK_STACK0 * 1e9`,
     `byte_value_writes(byte, strength=2000)`) writes a `~-10^7` (to `-10^15`,
     value-dependent) GARBAGE one-hot into the SAME H1/H3 cells. Its
     `OUTPUT_LO+lo` / `OUTPUT_HI_THIS_STEP+hi` input is WRONG on the carried
     step (`OUTPUT_LO[6]` set instead of `[3]` for 0x23), so it materializes
     the wrong byte at huge magnitude.

4. The byte token reads `H1+lo` (= `-10^7`) at +5.0 -> logit ~ `-10^8` -> a
   marker (`[PC]`, token 257) wins the argmax -> the spurious extra register
   block -> 57-token step.

Logit attribution (`tools/probe_stack0_byte0_logit.py`, 0x23 carried step):
`H1+5 res=-1.04e7 contrib=-5.18e7`, `H3+6 res=-1.02e7 contrib=-5.08e7` ->
`logit[0x23] = -1.03e8`. The byte's OWN one-hot cells are the corrupted ones.

## The carry that landed (mirror of the AX byte-1 carry, flag-gated OFF)

All BYTE-IDENTICAL when `C4_STACK0_B0_DUMP=0` (default). Bands in
`_PRODUCTION_EXTRA_RESIDUAL_DIMS` (head-dim-preserving widen `872 -> 981`,
n_heads 8->9, base head_dim 109; the +29 dims fit in the existing 9th head, no
new head; bnz/mul geometry unchanged). All in `_LIVENESS_NEVER_SHARE`.

1. **Bands** `STACK0_B0_H1_PREV`(7) / `STACK0_B0_H3_PREV`(7) (carry),
   `STACK0_B0_DUMP_H1`(7) / `STACK0_B0_DUMP_H3`(7) (emission),
   `STACK0_B0_CARRIED`(1) (bounded gate flag).

2. **Carry head** (`stack0_byte0_dump_carry`, `l11_ops.py`, hosted L9 anchor ->
   lands L10 block 11, head 5; cross-step allowlisted `H1.*.-1` / `H3.*.-1`):
   copies the PREVIOUS step's STACK0-marker H1/H3 one-hot into the PREV bands.
   The decisive K-selection fix is a **one-hot-presence preference** (`+PRES_W`
   on every same-step H1/H3 cell, CONST-driven) so the head attends the FRESH
   prev STACK0 marker (one-hot present, Σ~4 at this pre-corruption block) over
   the current carried marker (one-hot absent). VERIFIED: PREV holds the
   correct one-hot (`H1_PREV+5 ~ 166` for 0x23).

3. **Carried-flag precursor** (`stack0_byte0_carried_flag`, `l11_ops.py`, bound
   to the L8 attn anchor — an EARLY block where H3 is still BOUNDED): writes
   `STACK0_B0_CARRIED = AND(MARK_STACK0, H3-absent)`. The bounded read is
   load-bearing: the L25-tail H3 is `-10^7` garbage, so a raw negative-weighted
   H3 condition in the dump drives its silu gate to `~+10^9` and writes
   billions into the DUMP band (observed: all-zero emission). Reading H3 EARLY
   (fresh ~3.3 present / carried ~0 absent, both bounded) keeps it sane.
   VERIFIED: flag = 0 on fresh/non-STACK0 rows, ~1 on carried STACK0 rows.

4. **Dump FFN** (`stack0_byte0_dump_repopulate`, L25 tail post_op): copies PREV
   -> DUMP on a carried STACK0-marker row gated on the BOUNDED `STACK0_B0_CARRIED`
   flag (NOT the corrupted H1/H3). VERIFIED: writes the correct DUMP slots
   (`DUMP_H1+5`, `DUMP_H3+6` for 0x23).

5. **Head bake** (`stack0_byte0_dump_head_bake`, `model_ops.py`, phase 1002,
   `C4_STACK0_B0_DUMP`-gated): mirrors the byte-token `H1+(lo+2)` / `H3+(hi+4)`
   columns onto the DUMP bands.

## Why flag-OFF (the additive band can't win this bug)

Unlike the AX byte-1 carry — where `H1` is ZERO on the carried step so the
additive `H1_DUMP_OUT` cleanly SUPPLIES the missing byte — here the byte's OWN
H1/H3 cells are a **NEGATIVE `-10^7` corruption** (from the L16 e8 materializer,
step 3 above). The byte token reads `H1+lo` (`= -10^7`) AND the additive DUMP
band (`= +10^5`); the `-10^7` suppression wins -> `logit ~ -10^8` -> a marker
still wins. An additive band in SEPARATE dims cannot overcome a negative write
in the SAME cells the head reads. Default-ON regressed smoke 50/1 -> 45/6
(sub/div/shr/add_16bit/mul_overflow — the DUMP columns also leak onto
arithmetic-result rows) and did NOT fix the drift, confirming the approach is
architecturally insufficient. Default-OFF is byte-identical (if_gt 4/25,
if_var 0/25, bool_and 8/25 — exactly the baseline; smoke 50/1).

## The re-point that LANDED (flag-gated `C4_STACK0_B0_DUMP`, still default OFF) — 2026-06-13 update

The re-point IS BUILT and proven on the target clusters, but a clean discriminator
to keep smoke at 50/1 **does not exist** — so it stays flag-gated. Details:

### Two corrections to the original diagnosis (probe-confirmed, this update)
1. **The block-38 nuke is NOT the `stack0_e8_output_authoritative` materializer.**
   That materializer writes `OUTPUT_LO`/`OUTPUT_HI` (`byte_value_writes` defaults),
   NOT H1/H3, and its gate (`OUTPUT_LO+lo AND OUTPUT_HI+hi`) is DARK on the carried
   row (OUTPUT_HI is empty there). The actual H1/H3 nuke is the **`tail_bit32_result_correction`
   bank** (block 38, 2059 units): its STACK0-marker materializer units read the
   block-32 SMEARED H1/H3 one-hot (e.g. H1+3=3718, H1+2=664) at +5.0, producing a
   huge silu, then write ±500 (`W_down`) to H1/H3 → the ±1.8e7 competing writes net
   to ~-1e7. Probe: `tools/probe_nuke_trigger.py` (the up-drivers are
   `MARK_STACK0*1e9 + CONST*-1e9 + 5*H3+5 + 5*H1+3 ...`). The LM head reads H1+3/H3+5
   at +5.0 → logit[byte] ~ -1e8 → a marker wins. (`tools/probe_stack0_byte0_logit.py`
   at the FINAL block confirms `H1+3=-1.05e7, H3+5=-9.98e6 → logit[0x11]=-1.02e8`.)
2. **Logical L16 is physical block 28; the nuke + dump live in the L25-tail expansion
   (blocks 37-43).** The widened (carry-band) build has **44 blocks** (was 37); the
   carry head lands block 10, tail_bit32 nuke block 38, the dump FFN block 42.

### The re-point (`stack0_byte0_dump_repopulate`, `l11_ops.py`)
On `C4_STACK0_B0_DUMP=1` the dump FFN (block 42, AFTER the nuke) writes the carried
byte-0 one-hot DIRECTLY into the byte's own `H1+j`/`H3+j` emission cells (identity
slot map `STACK0_B0_H1_PREV+j -> H1+j`), large `write_scale` (~70) so the read slot
nets POSITIVE over the -1e7 nuke. It runs LAST, so it reaches the byte token THROUGH
the corruption. Gated on `STACK0_B0_CARRIED` AND a new bounded `STACK0_B0_SHARP` flag
(`stack0_byte0_sharp_flag` precursor: PREV is a clean single-slot one-hot vs a smear).
Flag OFF = byte-identical (writes the inert `STACK0_B0_DUMP_*` bands; smoke 50/1).

### It WORKS on the target clusters (flag ON, full_trace, GPU 0)
* `if_gt`   4/25  → **17/25** (+13)
* `bool_and` 8/25 → **16/25** (+8)
* `if_var`  0/25  → 0/25 (deeper root, not this bug)
* `tools/probe_stack0_byte0_logit.py` (flag on): carried CMP step now emits the
  correct byte (token 0x11, logit ~+3e9, was [PC] token 257 at -1e8).

### Why it STILL can't be default-ON (the architectural wall, now precise)
The re-point fires on EVERY carried STACK0 row whose PREV is a clean one-hot. The
**framing-drift row (byte should emit) and a healthy carried-marker row (a marker
should emit) are STRUCTURALLY INDISTINGUISHABLE** at the residual: both are carried
(`CARRIED~100`), have a clean PREV one-hot (`SHARP~108`), are H1/H3-nuked, emit token
257 pre-fix, and the block-32 smear argmax matches the PREV slot. The ONLY difference
is the VALUE carried (AX_CARRY nibbles), NOT byte-vs-marker semantics — which is set
by VM stack layout not locally encoded. So the re-point forces a value byte onto
multi-byte arithmetic-RESULT rows where a marker belongs → corrupts them:
* `add` band (ids 100-115) 16/16 → **8/16** (full_trace), and smoke regresses
  `test_add_16bit` (300→100) + `test_jmp_forward` (flag ON = 48/3).
Tried discriminators that ALL FAILED to separate the cases: PREV-sharpness
(`SHARP`, both clean), opcode/CMP markers (all stale=1.0 at STACK0 rows),
`STACK0_BYTE1/2/3` (constant -2.0 both), nuke-present (both nuked),
per-slot competitor-subtraction in the linear gate (leaks negatives → MORE
regressions). The flag is therefore a NET-POSITIVE TRADE, not a clean win, and the
brief's hard `smoke==50/1` invariant keeps it OFF by default.

### Real fix for the NEXT agent
The re-point machinery is correct; what's missing is a **byte-vs-marker discriminator**
at the carried STACK0 row, OR a fix at the nuke SOURCE: gate the block-38
`tail_bit32_result_correction` STACK0-materializer units OFF the SMEARED-input case
(they fire on the block-32 smear; suppressing them only when the smear is a
spurious carried artifact would stop the nuke without needing the additive re-point).
That touches the WIDTH-SENSITIVE 2059-unit tail bank in place
(`project_l10_tail_bank_width_sensitive`) — must repurpose, not append.

## Probe tools (read-only, spec_k=0, build `alu_mode='efficient'`)
- `tools/probe_stack0_byte0_persist.py` — per-step token count (57 vs 35) +
  raw token dump of the drift step (shows the spurious [PC] register restart).
- `tools/probe_stack0_byte0_logit.py` — LM-head logit attribution at the STACK0
  byte-0 predictor (marker) row; identifies the H1/H3 corruption. Run with
  `C4_STACK0_B0_DUMP=1` to confirm the re-point makes the carried step emit
  the correct byte token (logit ~+3e9 instead of -1e8).
- `tools/probe_stack0_byte0_blocktrace.py` — per-block H1/H3 trace (block 32
  smear -> block 38 nuke).
- `tools/probe_stack0_l16_repoint.py` — confirms the carried-step premises:
  CARRIED flag (100 carried / 0 fresh), PREV band holds the clean carried byte,
  the OUTPUT band is wrong on the carried row.
- `tools/probe_nuke_trigger.py` — the DEFINITIVE root: which block-38
  `tail_bit32_result_correction` units nuke H1/H3 and their up-drivers (the
  block-32 smeared H1/H3 one-hot read at +5.0).
- `tools/probe_stack0_add16.py` — the no-discriminator evidence: compares the
  framing-drift `if` row vs the healthy `add_16bit` carried row.
