# STACK0 byte-0 cross-step carry (Root 2 — if/bool/expr framing drift) — 2026-06-13

Status: **carry infrastructure LANDED, flag-gated OFF** (`C4_STACK0_B0_DUMP`,
default 0). Byte-identical to clean main; smoke **50/1** (only the pre-existing
`simple_function` arch-block). The carry mechanism is verified end-to-end but
the additive-band approach is **architecturally insufficient** for this bug
(see "Why flag-off" below) — the real fix is localized but touches the
load-bearing, width-sensitive L16/L25 STACK0-marker bank.

Branch base: main HEAD `2479bb06`. spec_k=0, GPU 0, `alu_mode='efficient'`.

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

## The real fix (next agent)

The carry CONCEPT is right (the clean value only exists at the prev step), but
the re-emission must reach the byte token THROUGH the corruption, not around it.
Two candidate paths, both touching the load-bearing, WIDTH-SENSITIVE L16/L25
STACK0-marker bank (`project_l10_tail_bank_width_sensitive` — appending breaks
byte-identity; must REPURPOSE in place):

* **(A) Fix the L16 e8 materializer's input.** `stack0_e8_output_authoritative`
  (`l16_ops.py:784`) reads `OUTPUT_LO+lo` / `OUTPUT_HI_THIS_STEP+hi` to emit the
  STACK0 byte-0, but on a carried step OUTPUT is WRONG (`OUTPUT_LO[6]` not `[3]`
  for 0x23). Re-point it to read the carried byte-0 (from the `STACK0_B0_*_PREV`
  band this carry already provides, gated on `STACK0_B0_CARRIED`) instead of the
  broken OUTPUT band. This is the cleanest fix: the materializer already OWNS the
  H1/H3 write at the STACK0 marker; just feed it the right value.

* **(B) Gate the block-38 nuke OFF carried STACK0 rows + write the carried
  one-hot into the same H1/H3 cells.** Higher risk (the 1e9 MARK_STACK0 gate is
  load-bearing for the fresh-step materialization).

Path (A) is recommended — it reuses the existing STACK0-marker H1/H3 writer
rather than fighting it, and the carry bands already deliver the correct value
to that block.

## Probe tools (read-only, spec_k=0, build `alu_mode='efficient'`)
- `tools/probe_stack0_byte0_persist.py` — per-step token count (57 vs 35) +
  raw token dump of the drift step (shows the spurious [PC] register restart).
- `tools/probe_stack0_byte0_logit.py` — LM-head logit attribution at the STACK0
  byte-0 predictor (marker) row; identifies the H1/H3 corruption.
- `tools/probe_stack0_byte0_blocktrace.py` — per-block H1/H3 trace (block 32
  smear -> block 38 nuke).
