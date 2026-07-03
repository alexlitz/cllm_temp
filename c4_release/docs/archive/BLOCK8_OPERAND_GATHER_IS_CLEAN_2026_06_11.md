# Block-8 operand gather is CLEAN — the AND/MUL blocker is NOT here (2026-06-11)

Status: **The brief's premise is disproven.** The block-8 head-0 operand
gather is a faithful per-nibble identity copy for ALL nibble values 0..15
in BOTH nibble positions. There is **no high-nibble mis-gather**. The
"0xFF → ALU_LO cell 8" symptom is the gather *correctly* copying an
operand that was **already corrupted to 0xE8 upstream** by the IMM
immediate-value decode. No block-8 / l7 / model-ops-slope / alu-l10 change
can recover it. Smoke held at the 26-pass / 8-of-8-guardrail baseline (no
weight edits landed — only probes). Single proof tool:
`tools/probe_operand_gather_is_clean.py` (spec_k=0, hook-free).

> Note: the brief quotes a "36-pass baseline". The actual current baseline
> is **26 pass / 15 fail / 8-of-8 guardrails** (`tools/run_full_smoke.py`),
> matching `AND_MUL_MARK_AX_ENDRUN_2026_06_11.md`. 36 is stale.

## 1. What the brief claimed vs what is true

Claim: *"operand A at the AX row has a HIGH-NIBBLE mis-gather (0xFF → ALU_LO
cell 8, byte-identical to 0x08). The head-0 attention or the CLEAN_EMBED
encoding mis-maps high nibbles."*

Truth, proven by `probe_operand_gather_is_clean.py`:

* **[A] Token-level CLEAN_EMBED is a correct one-hot for ALL 256 values**
  (`0/256` mismatches). The gather's V source is faithful.
* **[B] The gather output is a correct per-nibble one-hot for every
  IMM-clean operand**, every low nibble 0..13 (skipping lo=8 — see below)
  and the high nibble, e.g. `A=0x2d → ALU_LO answer cell 13`, `A=0x29 →
  cell 9`, `A=0x27 → cell 7`. The V/O projection is a pure cell-k→cell-k
  identity (`_band_projection_writes` + `_band_output_writes`,
  `l7_ops.py:432-437`); it is structurally incapable of moving nibble 15
  to cell 8.
* The head-0 attention (slope 0.1, the prior Wall-1 fix at
  `model_ops.py:1229`) correctly argmaxes the operand-A STACK0_BYTE0 row
  with w≈0.97. When that row holds the right token it copies it perfectly
  (`and_70`: attends row135 tok=0x70 → ALU_LO cell0/ALU_HI cell7 = a clean
  one-hot of 0x70).

## 2. The actual symptom and its real cause

`probe_operand_gather_is_clean.py` section [C] and `tools/probe_stack0_value.py`:

For `IMM 0xFF; PSH; IMM 0x2A; AND`, the spec_k=0 replay's own STACK0_BYTE0
carrier holds token **0xE8 (232)**, not 0xFF. The gather faithfully copies
0xE8 (whose low nibble *is* 8 → "cell 8"). So the "cell-8 mis-gather" is
the gather doing its job on a poisoned input.

Where 0xE8 comes from — **the IMM immediate-value decode**, with NO gather,
NO stack, NO binop involved:

```
IMM v; EXIT        (tools: section [C] + a full 0..255 sweep)
  0x0F→0x0F  0x2A→0x2A  0x70→0x70  0x80→0x80  0xAB→0xAB   (OK)
  0x08→0xFFE8  0x88→0xFFE8  0xE0→0x1  0xF0→0xFFE8  0xFF→0xFFE8  (CORRUPT)
```

46/256 immediates mis-decode, in two crisp families:

1. **lo-nibble(v) == 8** → emits 0xE8 (0x08,0x18,0x28,…,0xD8 all fail).
2. **hi-nibble(v) ∈ {0xE, 0xF}** → emits 0x1 or 0xFFE8 (0xE0–0xEF, 0xF0–0xFF).

Block-level trace (`IMM 0xFF; EXIT`, ax row): OUTPUT_LO already carries the
correct cell 15 = 160 at block 8, but with spurious cells 3/11; **block 30
(logical L20) spikes** the spread and the **block-36 final decode argmaxes
OUTPUT_LO cell 8 / OUTPUT_HI cell 14 → 0xE8**. This is the *same* L20
block-30 cell-8 corruptor `AND_MUL_MARK_AX_ENDRUN_2026_06_11.md` named —
here it corrupts the bare IMM decode too.

## 3. The second, independent blocker: AND/MUL/XOR compute

Even with BOTH operands IMM-clean, AND mis-decodes (section [C]):

```
0x70 AND 0x2A → 0xFFF0  (expected 0x20)
0x2A AND 0x70 → 0xFFF0  (expected 0x20)
0x0F OR  0x30 → 0x3F    (OK — OR passes with the SAME gather + SAME
                          cell-0/cell-8 residue)
```

OR passing while AND fails on identical clean operands proves the residual
cell-0 (+5.56) and cell-8 (+0.45) gather artifacts are **tolerated** (not
the blocker). The AND/MUL/XOR result is corrupted by the L15 (block 25)
materialise + L20 (block 30) cell-8 spike documented in
`AND_MUL_MARK_AX_ENDRUN_2026_06_11.md` §2 — confirmed still live here.

## 4. Why no fix lands in the allowed surface

The allowed surface is **l7 head-0 gather + model_ops block-8 slope +
alu/l10 per-op compute**. All three are clean or downstream-blocked:

* head-0 gather: already a faithful identity (§1) — nothing to fix.
* block-8 slope: already 0.1 (prior Wall-1 fix); sharpening it further
  changes nothing because the input row is poisoned, not mis-attended.
* the cell-0 / cell-8 residue cleanup: OR tolerates it, so cleaning it
  unblocks nothing (and is the documented zero-sum single-rule trap —
  see `feedback_single_rule_fixes_are_zero_sum.md`).

The real blockers both live **outside** this surface:

1. **IMM immediate-value decode** (lo-nibble-8 + hi-nibble-E/F families).
   This is the OUTPUT-decode / value-emit path (L5 IMM dispatch → OUTPUT →
   L20 block-30 spike → block-36 decode), explicitly the "AVOID l6/l8/l9
   register-emit" + the L20 surface.
2. **AND/MUL/XOR compute** L15 cell-0 materialise + L20 cell-8 spike
   (already documented; the L15 writer is `l15_ops.py` / mul-combine
   GEToBD, the L20 spike has no op file in the allowed surface).

## 5. Bearing on the 12 smoke tests + 400 1096 programs

* `or_basic` passes (clean operands 0x0F/0x30, both IMM-clean, OR compute
  clean). `and_basic`/`xor_basic` use operand 0xFF — hit BOTH the IMM-decode
  bug AND the AND/XOR compute bug.
* `add_basic`/`sub_basic`/`eq_*`/`mul_basic` need the AND/MUL/ADD compute +
  decode (L15/L20), unaffected by any block-8 gather change.
* The ~400 if/expr/gcd/bool_and 1096 programs are gated by the same
  AND/MUL compute + (for any ≥0xE0 / lo-nibble-8 literal) the IMM-decode
  bug — neither is in the operand-gather.

## 6. The next sub-links (in priority order)

1. **IMM immediate-value decode** — fix the lo-nibble-8 and hi-nibble-{E,F}
   mis-decode in the OUTPUT/value-emit path. This is a NEW, sharply
   characterised bug (46/256 immediates, two clean families) that blocks
   any program with a high or lo-nibble-8 literal — strictly upstream of
   and independent from the AND/MUL compute. Highest leverage, smallest
   blast radius (the decode is per-value, not per-op).
2. **AND/MUL/XOR L15-materialise + L20 cell-8 spike** — as
   `AND_MUL_MARK_AX_ENDRUN_2026_06_11.md` §3: stop the L15 noisy OUTPUT
   materialise for OP_AND/OP_MUL and neutralise the L20 block-30 cell-8
   ±150M spike. Multi-op co-design across l15_ops + the L20 surface.

## Artifacts

* `tools/probe_operand_gather_is_clean.py` — the single authoritative
  proof (sections A/B/C above), spec_k=0, hook-free.
* `tools/probe_operand_highnibble.py` — per-value CLEAN_EMBED at the
  attended STACK0 row (shows the carrier holds the corrupt 0xE8).
* `tools/probe_operand_locate.py` — reconstructs head-0 attention at the
  AX row (argmax row + token + raw score), proving correct attention.
* `tools/probe_psh_stack0_carrier.py` — passing-vs-failing carrier token
  comparison (or_basic row135=0x0F vs and_basic row100=0xE8).
* `tools/probe_clean_embed_trace.py` — block-by-block CLEAN_EMBED at the
  carrier (constant from embed → block 8; the gather adds no corruption).
* `tools/probe_token_clean_embed.py` — token-embed CLEAN_EMBED 0/256 bad.
* `tools/probe_stack0_value.py` — STACK0 carrier value vs expected operand.
