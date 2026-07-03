# Wave 1 Cluster B3 — L5 head 0 TEMP staging probe (2026-06-07)

## TL;DR

Probed TEMP propagation from MARK_PC → MARK_AX across L0..L6 for IMM 0xFF
using `tools/imm_temp_staging_probe.py` against the
`test_xor_basic` bytecode (`IMM 0xFF / PSH / IMM 0xD5 / XOR / EXIT`,
override-on, expected exit code `0x2A`).

**Divergence layer: L5 attention.** TEMP at MARK_AX is identically zero
through every captured residual L0..L4 (pre_attn / pre_ffn / post_ffn at
each block) and remains zero entering L5. The first non-zero TEMP value
at MARK_AX appears at **L5_pre_ffn** (i.e. produced by L5 attention) with
the value `TEMP+10 = 1.0`, which is then doubled to `TEMP+10 = 2.0` by
L5/L6 FFN at MARK_AX.

**Expected vs actual TEMP at MARK_AX (post-L5-attn) for IMM 0xFF:**

| signal              | expected (nibble F=15)   | actual (post-L5-attn)        |
|---------------------|--------------------------|------------------------------|
| TEMP lo peak slot   | 15 (low nibble = F)      | 10 (`OUTPUT_BYTE_LO+10` = A) |
| TEMP lo peak value  | ~1.0 (one-hot)           | 1.0                          |
| TEMP hi peak slot   | 15 (high nibble = F)     | -- (all zero)                |
| TEMP hi peak value  | ~1.0 (one-hot)           | 0.0                          |
| FETCH_LO peak slot  | 15 at MARK_AX            | 5 (val 40.0) at MARK_AX (L6) |

**Candidate corrective layer/op: L4 `layer4_pc_relay` head 0 + L4 FFN
`_layer4_pc_plus1_ax_rules` chain.** Both are silent at MARK_AX because
their SOURCE (`EMBED_LO/HI` at MARK_PC row, position 44) is identically
zero in the captured residual. Without the upstream EMBED staging, the
PC+1@AX rotation has nothing to rotate, so TEMP stays empty entering
L5 attention. The actual TEMP+10 write seen at L5_pre_ffn comes from an
L5 attention head (head 0 or another head) reading whatever pattern is
in TEMP_PREV_STEP / cross-step state, NOT from the intended L4 staging.

The override at `batched_pure_neural.py:2050-2058` compensates by
writing the correct REG_AX bytes directly, masking the entire broken
upstream chain.

## Probe details

- Tool: `c4_release/tools/imm_temp_staging_probe.py` (parameterized via
  `PROBE_IMM=<int>` env var; defaults to 0xFF).
- Captures three residual snapshots per block (`pre_attn`, `pre_ffn`,
  `post_ffn`) via `forward_pre_hook` on attn/ffn and `forward_hook` on
  the block module itself.
- For each MARK_AX-bearing forward (`mark_ax_fwds = [6, 41, 74, 95]` for
  the XOR_BASIC bytecode), the IMM 0xFF AX-emit forward is `fwd=6`
  (seq_len=50, mark_pc_pos=44, mark_ax_pos=49).
- Per layer, dumps TEMP/FETCH/EMBED/CLEAN_EMBED nibble peaks at BOTH
  the MARK_PC row and the MARK_AX row.
- JSON dump written to
  `c4_release/tools/imm_temp_staging_probe_findings.json`.

## Layer-by-layer divergence walk (MARK_AX row)

```
L0_pre_attn .. L4_post_ffn:   TEMP=0   EMBED=0   FETCH=0   (CEMBD lo_peak=4 from L1+)
L5_pre_attn:                  TEMP=0   EMBED=0   FETCH=0   (still empty)
L5_pre_ffn (post-L5-attn):    TEMP+10=1.0   EMBED=0   FETCH=0
L5_post_ffn:                  TEMP+10=1.0   FETCH=0
L6_pre_ffn (post-L6-attn):    TEMP+10=2.0   FETCH_LO+5=40.0
L6_post_ffn:                  TEMP+10=2.0   TEMP+(16+11)=-2.0   FETCH_LO+5=40.0
```

Compare MARK_PC row at L5_pre_ffn:
```
MARK_PC  TEMP lo_peak=10 (1.0)   FETCH_LO+5=40.0   CEMBD lo_peak=3 (1.0)
MARK_AX  TEMP lo_peak=10 (1.0)   FETCH=empty       CEMBD lo_peak=4 (1.0)
```

TEMP at MARK_PC == TEMP at MARK_AX at L5_pre_ffn — the L5 attention is
writing the same TEMP pattern at both rows, NOT a row-differentiated
staging. This is consistent with the broken row gating described in
`IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md`.

CLEAN_EMBED at MARK_PC encodes PC counter `0x03` (lo_peak=3) and at
MARK_AX encodes PC+1 = `0x04` (lo_peak=4). These are positional / PC
counters, not the IMM byte content. The byte content (low nibble F,
high nibble F for IMM 0xFF) is absent everywhere in the captured
residual at marker positions.

## K-side reference (ADDR_KEY+32..47 at L5_pre_attn)

```
MARK_PC ADDR_KEY+32..47: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
MARK_AX ADDR_KEY+32..47: [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

At MARK_AX the ADDR_KEY+32..47 K-side has hot keys at slots 2 and 9.
L5 head 0's Q-side `TEMP+k` slots are all zero at MARK_AX, so the Q-K
dot product is dominated by ADDR_KEY similarity to whatever row's
ADDR_KEY contains the matching pattern — not by content match. With
softmax1's zero anchor, the head likely lands on the byte row whose
ADDR_KEY happens to align with slots 2/9 (an OPCODE-byte row), which
matches the observation in v3 that L5 head 0 picks up the OPCODE byte
nibble for high-bit IMMs.

## Candidate corrective op

**Primary suspect: `layer4_pc_relay.head_0` and `_layer4_pc_plus1_ax_rules`.**

The L4 attention pc_relay head 0 spec
(`l4_ops.py:336-355`) reads `V = EMBED_LO/HI+k` at `K=MARK_PC` row and
writes to MARK_AX. Source `EMBED_LO/HI` at MARK_PC is empty (PC marker
rows have no byte content), so the relay contributes 0 to MARK_AX
EMBED. The L4 FFN's PC+1@AX nibble-rotation chain
(`_layer4_pc_plus1_ax_rules`, `l4_ops.py:655-666`) consequently has
nothing to rotate, so TEMP at MARK_AX stays 0 entering L5.

The fix-direction (out of scope for Wave 1): either

  (a) route L4 pc_relay head 0 to source EMBED from the IS_BYTE row at
      ADDR_KEY = PC+1 (not from MARK_PC), or
  (b) add an earlier op (L2 or L3) that stages PC+1 byte content into
      MARK_PC EMBED before L4 reads it.

Both are multi-rule structural changes outside single-rule fix scope,
matching the conclusion in `IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md`.

## Files referenced

- `c4_release/tools/imm_temp_staging_probe.py` — the probe (this wave).
- `c4_release/tools/imm_temp_staging_probe_findings.json` — per-layer
  TEMP/EMBED/FETCH peaks at MARK_PC and MARK_AX (machine-readable).
- `c4_release/docs/IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md` — v3
  upstream context.
- `c4_release/neural_vm/unified_compiler/ops/l4_ops.py:336-381` —
  L4 pc_relay head specs (head 0 = MARK_AX EMBED relay, head 1 = AX
  byte TEMP relay).
- `c4_release/neural_vm/unified_compiler/ops/l4_ops.py:655-666` —
  `_layer4_pc_plus1_ax_rules` chain.
- `c4_release/neural_vm/unified_compiler/ops/l5_ops.py:309-327` —
  L5 head 0 spec (the bug surface — consumes TEMP at MARK_AX).
- `c4_release/neural_vm/batched_pure_neural.py:2050-2058` — the
  load-bearing override.

## Wave 2 input

This document is the diagnostic input for Wave 2. The candidate
corrective surface is **L4 pc_relay / PC+1@AX staging**, not L5 head 0
itself. Wave 2 should investigate whether the EMBED-at-MARK_PC staging
gap is a recent regression (compare to a known-good commit) before
attempting a structural fix.

## Smoke baseline

Wave 1 is probe + doc only — no model code changed. Smoke is expected
to remain at the pre-wave baseline. (Run output is logged separately;
the only files added are `tools/imm_temp_staging_probe.py` and this
doc.)
