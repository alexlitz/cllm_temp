# var_simple_12 (id 262) — links 5+6 diagnosis (2026-06-11)

HEAD `e457ba31`. Path: **spec_k=0, hook-free** (`tools/probe_groundtruth.py`,
`tools/probe_var_full_chain.py`). Dims via `probe.model.dim_positions`.
Build: 39 physical blocks / 27 logical layers (`block_layer_map()`).

## TL;DR — the two "remaining links" are ONE root, and it is NOT a free-slot blocker

After the four landed links (e10bbc79, 4c3191ef, 8ad47bf4, e457ba31), `probe_var_full_chain.py 262`
reports the genuine clean-state divergences:

| step | op  | reg.byte | oracle | neural | nature |
|------|-----|----------|--------|--------|--------|
| 3    | PSH | SP.b3    | 0x00   | 0x04   | **ROOT** (premature marker, see below) |
| 4    | IMM | PC.b0    | 0x3a   | 0x42   | **downstream debris** of step-3 |
| 5    | SI  | PC/AX/SP | …      | …      | downstream debris |

AX is now correct (`AX=0xffe8` through step 2, `0x1c` at step 4). The remaining
desync is the step-3 PSH SP register.

## Link A (step-3 PSH SP byte3 = 0x04) — the actual mechanism

The "0x04" is **not an OUTPUT byte leak**. The raw token stream
(`tools/probe_var_link5_raw.py`) shows the step-3 SP register emits only **3
bytes** then jumps straight to the BP marker:

```
step3 pos=209 REG_SP: 224, 255, 0, <BP marker 260 here>   ← SP truncated to 3 bytes
```

The probe's register decoder reads `ctx[mpos+4] = 260` (REG_BP) and masks it to
`260 & 0xFF = 0x04`. So **byte3 "0x04" = the BP marker token mod 256.** The SP
byte3 *prediction row* (pred_row=212) emits the BP marker (260) instead of byte
`0x00`.

### Genesis: block 12 / logical L11 attention crushes OUTPUT to −569.7

`tools/probe_var_link5_good.py` / `probe_var_link5_cmp.py` walk pred_row 212
(BUG, step-3 PSH) against pred_row 245 (GOOD, step-4 IMM SP byte3 → correctly
emits `0x00`) per block:

- OUTPUT_LO/HI sit at +0.94 through block 11 (L10) on both rows.
- At **block 12 (logical L11)** OUTPUT_LO **and** OUTPUT_HI (dims 69–100, the
  whole 32-nibble band) crash to **−569.7 on the BUG row only**; the GOOD row
  stays +2.9 and gets its winning `0x00` at block 37 (L25 tail, +1.08e7).
- With OUTPUT crushed, the LM-head byte-0 logit dies (block 11: +3.9 → block 12:
  **−5693**) while the BP marker logit (260) stays at −9.6, so the marker wins.

The block-12 FFN is an all-zero dep-anchor (1 hidden unit, all weights 0); the
−569.7 comes **entirely from the block-12 ATTENTION** (13 heads, head_dim 109 —
the resized L11/MUL-pipeline attention, NOT the 2 declared
`step_end_operand_relay` heads in `l11_ops.py`). On the BUG row that attention
attends a STEP_END/marker position (whose OUTPUT band carries the −240 marker-
clear signal, scaled ~×2.4 → −569.7) and copies it into OUTPUT.

### Discriminator = the PSH-store / CMP marker cluster

BUG row (step-3 PSH) carries **PSH_AT_SP=2, MEM_STORE=2, CMP=4** from block 6
onward; GOOD row (step-4 IMM) carries 0/0/0. The mis-attention fires only when
these PSH-store/CMP markers are present on the SP byte3 row.

## Link B (step-4 IMM PC byte0 = 0x42 want 0x3a) — downstream of Link A

`probe_var_link5_raw.py` marker map: step-4 emits a **DOUBLED register block**:

```
step4: PC=58(0x3a✓) AX=28 SP BP 268 | PC=66(0x42) AX=28 SP BP 268 261 STEP_END
```

The **first** PC dump (pos 232) is the correct `0x3a` (58). The decoder keeps the
**last** PC (pos 253, `0x42` = 66). The doubled emission is caused by the step-3
SP-byte3 truncation desyncing the per-step register-count / position machinery.
**Fix Link A and the doubled step-4 block disappears → Link B is resolved.** It is
not an independent PC-decode bug (PC is decoded cleanly as 0x42 from block 3; the
problem is that this whole register block should not have been emitted).

## Why this is STOPPED at a clean partial (zero edits)

Per the brief's STOP conditions:

1. **Not a single free-slot hard-NOT-blocker.** The writer is a 13-head L11/MUL-
   pipeline attention copy whose legitimate job is CLEAN_EMBED→OUTPUT relay for
   the multiplier. Darkening it on the PSH-step SP byte3 row needs the head's
   real Q/K spec (resized attention, not the declared `l11_ops.py` heads) and
   risks the MUL/ALU path. This is a multi-op change, not a free-slot blocker.
2. **Entangled with the CMP path owned by the parallel agent.** The discriminator
   that fires the misfire is `PSH_AT_SP=2 / MEM_STORE=2 / CMP=4`. The CMP marker
   on a PSH-step SP byte3 row is itself suspect upstream pollution; the brief
   reserves the CMP path (L3 head 5, L9 `_layer9_cmp_rules`, l10 cmp_combine) for
   another agent.
3. **The brief's premise was wrong about Link A.** It expected "a separate
   SP-byte-3 LM-head family … residual decodes 0x00, marker-emit position,"
   fixable with a hard NOT-blocker. In reality the residual OUTPUT *decodes 0x00
   until block 12 then is crushed to −569*; the leak is a premature MARKER, and
   the writer is an L11 attention mis-attention, not an LM-head family.

## Minimal remaining change (for the next agent / owner of the L11+MUL surface)

Single root to close BOTH links:

> On the **PSH-step SP byte3 prediction row** (IS_BYTE=1, MARK_SP-register byte3,
> PSH_AT_SP/MEM_STORE present), **darken the block-12 (logical L11) OUTPUT-writing
> attention head** so it does not attend the STEP_END marker position and copy the
> −240 marker-clear band into OUTPUT. Pattern: a hard subtractive Q NOT-blocker on
> the resized L11 attention head that writes OUTPUT_LO/HI (W_o cols ≈327–334, the
> CLEAN_EMBED→OUTPUT slots), keyed so it is byte-identical on the legit MUL /
> non-PSH-SP rows. This requires the resized-attention head spec (NOT the
> `l11_ops.py` declared relay heads) and a byte-identity gate against the MUL smoke
> tests (`test_mul_overflow`, etc.).
>
> Alternative (upstream): stop `PSH_AT_SP / MEM_STORE / CMP` from polluting the SP
> register byte3 row on a PSH step — but `CMP` belongs to the parallel CMP agent.

## Repro / gate tools added (read-only, spec_k=0)

- `tools/probe_var_link5_focus.py` — marker map + per-block top-|dim| walk for the
  two pred rows.
- `tools/probe_var_link5_raw.py` — raw token stream + per-position LM argmax for
  steps 2–5 (shows the SP truncation + doubled step-4 block).
- `tools/probe_var_link5_cmp.py` — BUG (212) vs GOOD (245) SP-byte3 row diff,
  per-block IS_BYTE/marker/byte-index, and the block-where-marker-wins sweep.
