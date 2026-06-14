# Post-ENT MEM-desync: the brief's L14-heads-4-7 target is DEAD CODE; the real
# value producer is L10→L18, the over-emitter is L20 (2026-06-14)

Worktree base / HEAD: `73ce852b` (this worktree). Smoke at HEAD:
**51 passed / 0 failed** (`CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_smoke.py`).
**No production weight change was made.** The brief's prescribed fix
("rebuild L14 value heads 4-7 attention source") was attempted-by-investigation
and is REFUTED at this HEAD: those heads contribute nothing. All conclusions
below are spec_k=0, hook-free, BUILT-LAYOUT (`probe.model.embed._dim_positions`)
evidence on `func_identity_0` (id 550), the canonical repro.

Reproducer (unchanged): `CUDA_VISIBLE_DEVICES=0 python tools/run_1096_canonical.py
--ids 550-560 --criterion full_trace` → all fail `step=7 (LI) expected ax=val
got ax=0`. Per-step trace: `tools/_probe_func_li.py 550` shows the 37-token
post-ENT desync (steps 2 and 6 emit 2 extra leading 0xFF tokens).

## TL;DR — three precisely-localized roots, NONE of them is L14 attention

| # | symptom | producer (physical block / logical layer) | built dims |
|---|---------|-------------------------------------------|------------|
| R1 | ENT store VALUE = garbage `1` instead of old_BP | **block 30 = logical L18** decodes a thermometer that L10 leaves empty for ENT | `STACK0_B0_H1_PREV`=916-922, `STACK0_B0_H3_PREV`=923-929, `ALU_LO`=330-345, `OUTPUT_LO`=69, `OUTPUT_HI`=85 |
| R2 | ENT store ADDR byte-2 = `0x12`(18) sentinel | **block 39 = logical L25** tail corrector, 6.49e12 write | `OUTPUT_LO`+2 = dim 71 |
| R3 | post-ENT SE row over-emits 0xFF (logit 47635) → 37-token desync → breaks the L15 LI lookup | **block 32 = logical L20** unit 591, `w_out=-2e5` to `OUTPUT_HI`+0, `b_up=-850`, suppressed only by `OP_IMM/OP_ENT/MARK_*` reads | `OUTPUT_HI`+0 = dim 85 |

**R3 is the dominant root for the LI failure.** The PSH store at step 3 lands the
arg at the CORRECT address (65512); the LI at step 7 returns 0 not because the
value is wrong but because the 37-token desync (R3) shifts every position so the
L15 content-addressable lookup can't match the store entry. Fix R3 and the LI
lookup re-aligns. R1/R2 corrupt the *stored bytes* and must also be fixed for a
fully clean frame, but they are downstream of the desync's damage.

## Why the brief's L14-heads-4-7 rebuild CANNOT work (the dead-code proof)

The brief says: "the L14 value heads (4-7) at the ENT step attend the WRONG
source ... they copy garbage into the MEM value bytes." At THIS HEAD that is
false — the heads write **nothing**:

* `tools/_probe_l14_fire.py 550 1` and `... 550 3` (ENT and PSH steps): for
  EVERY row in the step, heads 4-7's O-contribution to `OUTPUT_LO/HI` is below
  0.3 in magnitude. They do not fire.
* `tools/_probe_l14_fire_bc.py 3` (the WORKING `si_li_roundtrip` SI store,
  val=42): heads 4-7 again write ~0. So they are dead on the working path too —
  they are not the value producer for ANY store.
* Direct dim read: at the SI val rows, the heads' declared O-targets
  (`STACK0_BYTE_VAL_1_LO`=602, `_2_LO`=729, `_3_LO`=761, plus 443/874) are all
  exactly 0.0 after blocks 17 and 18. The slot-36/44/45 `_with_overrides`
  machinery in `l14_ops.py:925-999` lowers to weights that produce no residual
  signal — superseded by the L10→L18 path below in a prior migration.

Therefore: rebuilding heads 4-7's Q/K/V is **byte-identical-inert** (it can only
move zeros) and cannot change the store value. Any agent re-reading the brief
should NOT spend a pass on the L14 attention heads.

## The REAL value pipeline (R1), traced dim-by-dim

`tools/_probe_storeval_path.py` + `tools/_probe_val_trace_bc.py` show the store
value enters and relays as:

1. **block 12 = logical L10** populates a thermometer of the store value into
   `STACK0_B0_H1_PREV` (916-922) and `STACK0_B0_H3_PREV` (923-929).
   - Working SI store (val=42): `B0_H1_PREV`=160, `B0_H3_PREV`=170 → decodes 42.
   - ENT store (val should be old_BP=65536): `B0_H1_PREV`=40, `B0_H3_PREV`=4.8 →
     near-empty; AND a spurious `ALU_LO`(330-345) = -37.6 appears (it is 0 for SI).
2. **block 30 = logical L18** reads that thermometer region + `ALU_LO` and writes
   `OUTPUT_LO/HI`. For SI it emits byte 42; for ENT only the byte-default
   baseline survives → `OUTPUT_LO`+1=2.0, `OUTPUT_HI`+0=4.0 → decodes byte `1`.

Discriminator probe (the single most useful artifact for R1):
`tools/_probe_storeval_path.py` — compares the block-30 INPUT residual at the
val_b0 row between the working SI store and the broken ENT store. The dims that
differ are EXACTLY `STACK0_B0_H{1,3}_PREV` (the value is missing) and the whole
`ALU_LO` band (-37.6 garbage for ENT). So **R1's true fix is at L10 (block 12)**:
ENT's store value is old_BP, but L10's value relay only handles the
STACK0/AX-sourced value (SI/SC/PSH). There is no old_BP→`STACK0_B0_*_PREV` path,
so the thermometer is empty and L18 emits the default. This is the "make the ENT
MEM section emit the clean saved-BP store" half of the prior doc's two-part build,
but the producer is **L10's STACK0_B0 thermometer relay, not L14 attention.**

## R3 — the over-emitter (the desync trigger), precisely located

`tools/_probe_desync_logit.py 550 1`:
* The SE row of the post-ENT step (pos 177) predicts token **255 @ logit 47635**
  (vs the post-JSR SE row at pos 142 which cleanly predicts REG_PC=257 @ logit 18).
* The driver is `OUTPUT_HI`+0 going hugely negative: block 11 ≈ 0 → block 12 =
  −240 (uniform across nibbles) → block 32 nibble-0 spikes to **−5242**.
* The −5242 spike at block 32 = logical **L20**, **unit 591**:
  `W_down[OUTPUT_HI+0, 591] = −200000`, `b_up[591] = −850`, and `W_up[591]`
  reads `OP_IMM`(−1e11), `OP_ENT`(−1e8), `MARK_PC/AX/SP/BP/STACK0`(−1e8 each).
  This is a **sentinel-inversion default**: "write OUTPUT_HI = −big UNLESS a
  register/opcode marker is present." On a normal next-step start row a marker
  (MARK_PC / OP_IMM) suppresses it; on the post-ENT SE row none fire, so the
  −2e5 sentinel writes 0xFF. This is the documented 0xFF/tail family
  (`project_ax_ff_leak_is_tail_byte1_ff_emitters`,
  `project_opcode_not_at_mark_ax`).

Companion sentinel R2 (`tools/_probe_addr_trace.py 550 1 2`): ADDR byte-2 = 18
enters at block **39 = logical L25**, dim `OUTPUT_LO`+2 (71) at 6.49e12 — a
separate tail corrector misfiring on the post-ENT addr row.

## The prior worktree stub was inert (reverted)

This worktree carried an UNCOMMITTED, never-committed change adding
`MARK_MEM/MEM_VAL_B0..3 = -2000` NOT-blockers to `l16_ent_frame_sp_byte1_ff`
(L16) behind `C4_ENT_MEM_DESYNC_FIX` (default ON), plus a never-called flag stub
in `l14_ops.py`. It is **inert**: `C4_ENT_MEM_DESYNC_FIX=1` vs `=0` give a
byte-identical `_probe_func_li.py 550` trace (MEM still `0x0012fff0`, LI still 0).
The L16 SP-byte1 sentinel it targets is NOT the corruptor (R1 is L10/L18, R2 is
L25, R3 is L20). Reverted both files to clean HEAD; smoke holds 51/0 and the func
cluster baseline is unchanged (0/3 at 550-552).

## Executable plan for the next agent (do these, in this order)

PART B first (R3 — it gates the LI lookup, highest leverage):
1. Target **L20 (block 32) unit 591** (and its sibling default units 574-578).
   It is the `OUTPUT_HI`+0 sentinel default. Find its declarative owner by
   grepping the L20 FFN op for a `gated_write`/`constant_write` with
   `b_gate/b_up ≈ -850`, `W_down[OUTPUT_HI+0] = -200000`, suppressors
   `OP_IMM/OP_ENT/MARK_*`. The post-ENT SE row needs an ADDITIONAL suppressor so
   the sentinel does not fire there. The row's discriminator: it is the SE→next
   token boundary AFTER an ENT step. Candidate gate dims present on that row but
   absent on a genuine 0xFF-needing row: it carries the STEP_END/next-step-fresh
   context with OP_ENT residue but NO register marker. DO NOT use a bare OP_ENT
   blocker (OP_ENT broadcasts ~10-18 for ≥1 row after its marker — see
   `docs/OPCODE_BROADCAST_BLOCKER_AUDIT_2026_06_11.md`; it will mis-suppress the
   legitimate firing and relocate the 0xFF — the documented zero-sum trap).
2. Verify with `tools/_probe_desync_logit.py 550 1`: the SE-row top-1 must flip
   from 255@47635 back to 257 (REG_PC) and `tools/_probe_func_li.py 550` must
   show step 2/6 back to 35 tokens (clean per-step counts `[35]*9`).

PART A second (R1 — the clean saved-BP store value):
3. Target **L10 (block 12)** `layer10_stack0_byte_relay` (it owns
   `STACK0_B0_*_PREV`). Add an ENT-step path that relays old_BP into the value
   thermometer the same way SI/SC relay the STACK0 value. old_BP at the ENT step
   is the prior frame's BP register (65536 = bytes [0,0,1,0]); content-address it
   from the prior step's BP/JSR-prologue row. Preserve SI/SC/PSH (they read the
   STACK0/AX value, byte-identical). Verify the block-30 OUTPUT decodes old_BP
   with `tools/_probe_storeval_path.py 550 1 4` (and offsets 5/6/7).
4. R2 (ADDR byte-2 L25 sentinel) — gate L25/block39 dim `OUTPUT_LO`+2's 6.49e12
   write off the post-ENT addr row (lower priority; addr byte-2 only needs to be
   0, and the addr low bytes are already correct).

GATES (all required, per the brief): func_identity_0 step 1 → 35 tokens + LI
returns the arg + trace advances; `pytest tests/test_smoke.py` = 51/0 with
SI/SC/LI/LC green; flag-off byte-identical to HEAD (`register_residual_band` for
any new band; one `C4_*` flag default ON); cluster delta over func 550-699 /
nested 950-999 / rec 700-799 / var 250-349 (`--criterion exit_code` AND
`full_trace`); add/sub/mul/if/bool must not regress.

## Diagnostic tools added this session (all `tools/_probe_*`, spec_k=0, hook-free)

* `_probe_func_li.py <ids>` — per-step PC/AX/SP/BP/STACK0/MEM (the desync visible).
* `_probe_l14_fire.py <id> <step>` / `_probe_l14_fire_bc.py <step>` — per-row L14
  value-head O-contribution (proves heads 4-7 are dead).
* `_probe_mem_blocks.py <id> <step>` — OUTPUT byte per MEM row across blocks.
* `_probe_addr_trace.py <id> <step> <byteoff>` — single MEM byte across ALL
  blocks (pinpoints R2's block-39 sentinel and R1's block-30 entry).
* `_probe_storeval_path.py {si|<id>} <step> <byteoff>` — the value relay dims
  (`STACK0_B0_*_PREV`, `STACK0_BYTE_VAL`) across blocks; SI-vs-ENT discriminator.
* `_probe_val_trace_bc.py <step> <byteoff>` — SI store value across blocks.
* `_probe_desync_logit.py <id> <after_step>` — the SE-row over-emit logit + the
  block-32/L20 negative-OUTPUT_HI driver (R3).
