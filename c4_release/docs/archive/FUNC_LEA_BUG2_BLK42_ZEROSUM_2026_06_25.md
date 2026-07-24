# func re-read LEA Bug #2 (block-42 byte-0 LO-nibble default): MACHINERY LANDED, ZERO-SUM vs func_identity — DEFAULT-OFF (2026-06-25)

Base: main `de1cccb0`. Bug #1 (`C4_FUNC_LEA_REREAD_BP_RESHARPEN`, the L7 head-1
re-read re-sharpen) re-applied here DEFAULT-ON with the post-flip cache-key fix
(`operand_from_memsp_enabled()` not raw `C4_OPERAND_FROM_MEMSP=="1"`). Bug #2
(`C4_FUNC_LEA_B0_RESTORE`, the byte-0 LO-nibble CAPTURE+RESTORE) is BUILT,
WORKING (flips func_add 575) and byte-identity-safe, but **DEFAULT-OFF** because
it is zero-sum vs the func_identity HOLD gate. Campaign config is the default
(`C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1`), spec_k=0, GPU0.

## What landed (committed)

* **Bug #1 re-applied** on the `de1cccb0` base, DEFAULT-ON in campaign,
  `operand_from_memsp_enabled()`-gated own kill-switch
  (`C4_FUNC_LEA_REREAD_BP_RESHARPEN=0`). **Cache-key fix**: both compiler
  cache-key snapshots gate on `operand_from_memsp_enabled()` (DEFAULT-ON
  post-flip) instead of the banked raw `C4_OPERAND_FROM_MEMSP=="1"` (which
  defaulted OFF → would let the campaign ON bake collide with a non-campaign
  cache entry).
* **Bug #2 machinery** (`C4_FUNC_LEA_B0_RESTORE`, DEFAULT-OFF): the
  `LEA_REREAD_B0` private band + `make_func_lea_b0_capture_op` (block-16 anchor)
  + `make_func_lea_b0_restore_op` (L25 tail, after `tail_bit32_result_correction`),
  a SILI/ENT-AXCARRY-precedent CAPTURE (clean one-hot) + winner-take-all RESTORE.
* Gates: flag-OFF golden byte-identical `cd54bfc0...3ad0b9be`
  (`C4_FUNC_LEA_REREAD_BP_RESHARPEN=0 C4_FUNC_LEA_B0_RESTORE=0`,
  `tools/_isa_golden_hash.py`, CPU) — IDENTICAL to the clean `de1cccb0` base.

## Bug #2 ON flips func_add 575 (GPU full_trace 1/1 PASS)

With Bug #1 + Bug #2 BOTH ON, `run_1096_canonical --ids 575 --spec-k 0
--criterion full_trace` → **1/1 PASS** (was step-11 fail `got ax=0xFFE8`,
`expected 0xFFE0`). The capture/restore re-establishes the pre-slam byte-0 LO
nibble (cell 0 = 0xE0) over the block-42 cell-8 (0xE8) stamp.

## WHY Bug #2 is ZERO-SUM (the blueprint's premise was incomplete)

The blueprint assumed block-42 is a pure cross-step corruptor stamping the
prior LEA &a's lo nibble (0xE8) over the fresh 0xE0. **It is actually a
LOAD-BEARING byte-0 default** that is CORRECT for some LEAs and WRONG for
others, and the cases are NOT locally separable.

Measured per-LEA (Bug-#1-only build, probe at block 20 / pre-slam block 38):

| LEA                              | pre-slam OUTPUT_LO argmax | block-42 stamp | wanted | verdict       |
|----------------------------------|---------------------------|----------------|--------|---------------|
| func_add &a  (FIRST, step 8)     | cell 8 (0xE8)             | cell 8         | 0xFFE8 | stamp = no-op |
| func_add &b  (RE-READ, step 11)  | cell 0 (0xE0)             | cell 8         | 0xFFE0 | stamp = WRONG |
| func_identity (only LEA, step 6) | cell 0 (0xE0)             | cell 8         | 0xFFE8 | stamp = RIGHT |

So **func_identity RELIES on the block-42 stamp** — its genuine pre-slam LEA
byte-0 is ITSELF wrong (cell-0 = 0xE0) and the stamp lifts it to the correct
0xE8. func_add &b is the only case the stamp breaks.

At the LEA AX row, func_add's re-read LEA (row 491) and func_identity's LEA
(row 293) are BIT-IDENTICAL in every probed dim (only `OP_LEA=5` + `MARK_AX=1`;
`OP_ENT/OP_JSR/OP_ADJ/MEM_ADDR_SRC/...` all 0). The desired addresses 0xFFE0
vs 0xFFE8 differ ONLY in the byte-0 LO nibble — byte-1 (0xFF) and the byte-0 HI
nibble (0xE) are identical — so there is NO local discriminator and no other
address bit to key on. Capturing+restoring the pre-slam value therefore FLIPS
func_add (575 → pass) but REGRESSES func_identity (550 step 6: 0xE8 → 0xE0,
GPU-confirmed `run_1096_canonical --ids 550` ok→fail).

## The TRUE fix (upstream, deeper than a tail capture/restore)

The block-42 byte-0 default is load-bearing because the GENUINE BP-frame LEA
byte-0 computation is wrong for func_identity (computes 0xE0, wants 0xE8). The
real fix is upstream where the LEA address byte-0 is computed from the BP frame
offset — so func_identity computes 0xE8 genuinely and the block-42 default is no
longer load-bearing — at which point a clean override of block-42 (or just
removing its hard cell-8 stamp) fixes func_add &b without regressing
func_identity. That requires localizing the L8 `_layer8_alu_lea_lo` /
BP-offset → byte-0 path and the block-42 writer's true intent (is it a SP/BP
default? a fall-through?), a multi-block build. NOT a tail capture/restore.

## Probes (banked)

* `tools/_probe_lea_addr_trace.py [pid] [step]` — block-by-block OUTPUT_LO byte-0
  + BP-row + final argmax (the localization probe).
* `tools/_probe_lea_head1_attn.py` / `_cpu.py` / `_fetch.py` — Bug #1 head-1
  re-pin probes.

To re-enable the Bug #2 machinery for the upstream-fix lane:
`C4_FUNC_LEA_B0_RESTORE=1` (inside the campaign config). It flips func_add but
regresses func_identity until the upstream genuine-byte-0 fix lands.
