# var_simple_12 (id 262) — block-12 OUTPUT-crush FIXED; next link = byte2/L15 (2026-06-12)

HEAD `e103c32b` + this commit. Path: **spec_k=0, hook-free**
(`tools/probe_var_full_chain.py`, `tools/probe_var_block12_head*.py`,
`tools/probe_var_psh_head_fire.py`, `tools/probe_var_l15_byte2.py`). Dims via
`probe.model.dim_positions`. Build: 39 physical blocks / 27 logical layers.

## What landed — the block-12 OUTPUT crush (LINK5/6 root)

The `VAR_SIMPLE_12_LINK5_6_DIAGNOSIS` "block-12 L11 attention crushes OUTPUT to
−569.7 on the var PSH SP byte3 row" is now **localized to a single declarative
head and FIXED**:

- Writer = **`_layer10_psh_stack0_passthrough_head_spec`** (l10_ops.py:1970),
  the L10 head-3 PSH STACK0-store passthrough. The dep-graph places this L10 op
  at **physical block 12 / logical L11** (the "resized L11/MUL-pipeline
  attention, 13 heads, head_dim 109" the prior diagnosis pointed at — it is NOT
  the MUL FFN nor the `step_end_operand_relay` heads). Head 3 has the largest
  OUTPUT-band W_o mass (192) of any block-12 head and writes OUTPUT_LO/HI at
  scale 3.0.
- Mechanism: the head's active gate (Q slot 33) keys on **PSH_AT_SP + IS_BYTE**
  without excluding register byte rows. On the var step-3 PSH, it fires
  (softmax1 wsum→1.0) not only on the legit STACK0-frame byte rows but ALSO on
  the SP/BP/AX **register** byte rows (`probe_var_psh_head_fire.py 262`: rows
  210/212/214/216/217 OUTsig −1000…−20000). There it averages CLEAN_EMBED /
  (OUTPUT−CLEAN_EMBED) debris into OUTPUT at scale 3.0 → OUTPUT crushed to
  −569 → SP-byte3 LM logit dies → BP marker (260) wins → SP truncates → desync.
- Discriminator (clean, NOT CMP): legit STACK0-frame byte rows have
  `STACK0_BYTE_h≈0.97` and the **H1 register-marker band (H1+0..H1+4 =
  PC/AX/SP/BP/MEM) ALL-ZERO**; the buggy register byte rows have all
  STACK0_BYTE=0 and exactly one H1+idx=1.
- Fix = hard subtractive NOT-blocker on free Q/K slot 7, keyed on the H1
  register-marker band (`Q[7]=−2e9·ΣH1+idx`, `K[7]=CONST·1`). On a register
  byte row one H1+idx=1 → slot scores −2e9 everywhere → softmax1 darkens the
  head → residual OUTPUT survives. On STACK0-frame byte rows H1=0 →
  byte-identical. Mirrors e457ba31 (l15 slot-64) / 8ad47bf4 (l18 slot-44).
  **Touches NEITHER CMP, PSH_AT_SP nor MEM_STORE** — only the register-marker
  band the head already (weakly) discriminates at slots 1/33.

### Verification (spec_k=0, GPU)
- `probe_var_full_chain.py 262`: step-3 PSH SP **byte3 0x04→0x00** — the BP-
  marker truncation is ELIMINATED; SP no longer truncated to 3 bytes
  (`0x400ffe0 → 0xfffe0`). OUTPUT_LO/HI at block 12 stays clean (no −569 crush).
- `pytest tests/test_smoke.py`: **46 pass / 5 fail / 0 xfail — ZERO regression**
  (the 5 fails are the pre-existing MUL/EQ/JSR arch-blocked set; no memory /
  SI / LI / PSH / ENT / LEA test moved).
- `test_l10_per_op.py` static-claims drift on this head is **pre-existing**
  (verified by reverting the edit — same failures on clean HEAD; the claims
  registry is stale for the bug-#33 differential V/O slots, unrelated to Q/K).

## The NOW-EXPOSED next link — step-3 PSH SP **byte2 = 0x0f** (STOPPED here)

With byte3 closed, `probe_var_full_chain.py 262` exposes a new first
divergence: **step-3 PSH SP byte2 exp=0x00 neu=0x0f**, genesis **physical block
30 / logical L19**. Precise attribution (`probe_var_full_chain` genesis walk +
per-block OUTPUT-decode at pred_row=211 + attn-vs-FFN split):

- OUTPUT band is clean ~0 through block 29 (cleared at L18); at **block 30 the
  *FFN* (not the attention)** writes `OUTPUT_LO[15]=+2.00` and
  `OUTPUT_HI[0]=+3.85` → decodes **0x0f**. Verified by the attn/FFN split:
  `pre=0.00 afterAttn=0.00 afterFFN=2.00` — the **block-30 PureFFN (logical
  L19)** is the writer, NOT the L15 `memory_lookup` attention heads 1-3 (those
  contribute <0.3 at this row; the MEM_STORE slot-65 darkening hypothesis was
  tried and **did not move the leak** — reverted).
- The leak row is a *register* byte row of a **store** step
  (`MEM_STORE≈1.5, OP_LI_RELAY=OP_LC_RELAY=0, MARK_AX=0`); this is the classic
  "0x0f CLEAN_EMBED-nibble / OUTPUT-default" FFN-debris pattern but from an FFN
  rule, not an attention head.

**Why this is STOPPED (separate writer class, not the block-12 attention
chain):** the byte2 link is a **block-30 / logical-L19 FFN rule** writing an
OUTPUT default on a store-step register byte row — a different op family and
layer than the block-12 attention fix, requiring its own FFN-rule attribution
(grep the L14/L15/L19 OUTPUT-default / `OUTPUT_LO+15` writers) and a
byte-identity gate against the memory smoke suite. It is NOT a continuation of
the psh_stack0_passthrough attention head and was not in the brief's localized
target. Reserved per the STOP-on-separate-writer rule and the
zero-sum-single-fix memory note. `probe_var_l15_byte2.py` (kept) is the
starting probe for the L19 FFN owner.

### Repro / gate tools added (read-only, spec_k=0)
- `tools/probe_var_block12_head.py [id] [blk]` — per-head W_o OUTPUT mass +
  softmax1 attention + OUTPUT contribution on the BUG/GOOD rows.
- `tools/probe_var_block12_head3.py [blk] [head]` — full Q/K/V/O nonzero dump
  for one head (identifies the owning op + free slots).
- `tools/probe_var_psh_head_fire.py [id]` — per-row wsum + OUTPUT contribution
  for the block-12 head-3 passthrough (legit vs crush rows).
- `tools/probe_var_psh_passthrough_disc.py [id]` — Q-gate discriminator dump
  (PSH_AT_SP / STACK0_BYTE / H1 / H4) for the head.
- `tools/probe_var_l15_byte2.py [id] [blk]` — block-30 L15 heads 1-3 per-row
  firing + H1 band (shows the byte2 link is load-bearing).
