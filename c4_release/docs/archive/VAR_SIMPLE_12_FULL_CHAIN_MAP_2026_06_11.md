# var_simple_12 — FULL remaining-link divergence map (2026-06-11)

Date: 2026-06-11. HEAD `383a6eb7`. Path: **spec_k=0, hook-free**
(`tools/probe_groundtruth.py`). Dims via `probe.model.dim_positions`.
READ-ONLY pass (no weight/ops edits). Tooling added (read-only):
`tools/probe_var_full_chain.py` (whole-program walk + per-divergence 37-block
genesis sweep) and `tools/probe_var_lea_pc_disc.py` (step-2 LEA gate +
PC-flood pin).

Program `var_simple_12` (id **262**): `int main(){int x; x=28; return x;}`,
expected **28**. spec_k=0 neural exit = **240** (wrong). Bytecode:
`[0]JSR3 [3]ENT8 [4]LEA-8 [5]PUSH [6]IMM28 [7]STORE [8]LEA-8 [9]LOAD [10]HALT`.

## TL;DR — the prologue is FIXED; the program desyncs at step 2 (LEA), NOT at the store

The 7-link JSR→ENT prologue + LEA byte0 (0xe8) + IMM decode fixes on main
**hold**: at spec_k=0 the neural trace is byte-perfect for **step 0 (JSR)** and
**step 1 (ENT)** — `PC=26 AX=0 SP=0xfff8 BP=0x00010000` (step0) and
`PC=34 AX=0 SP=0xffe8 BP=0xfff0` (step1) match the oracle exactly.

**The FIRST divergence is now step 2 (LEA −8), and it is in the PC, not the AX.**
The model emits **PC byte0 = 0x00 instead of 0x2a (42)** → neural PC=0 after
step 2. That desyncs the whole program: from step 3 on the neural model is
fetching the WRONG instructions, so the "store / load / LEV" the brief asked
about are **never executed on a clean state** (exactly the same structural
failure the prologue had before its fix, just moved two steps later). The store
(SI x=28), load (LI x), and LEV return therefore have **no clean divergence to
pin yet** — they are gated behind the step-2 PC desync. Fix the step-2 LEA PC
(and the step-2 LEA AX byte1 sibling) and the body re-runs; only then can the
SI/LI/LEV links be re-attributed.

## Full divergence table (every link, in stream order)

`†` = post-desync (neural is on the wrong instruction; oracle opname is the
intended op, not what neural ran). The two **GENUINE** clean-state links are
step-2 LEA PC byte0 and step-2 LEA AX byte1; everything after is desync debris.

| # | step | op | reg.byte | oracle | neural | genesis blk / logical | writer / mechanism | bug CLASS |
|---|------|----|---------|--------|--------|----------------------|--------------------|-----------|
| **1** | **2** | **LEA** | **PC.b0** | **0x2a** | **0x00** | clean→blk27; **flood at blk 28→29 / L18→L19** | OUTPUT_LO+0 jumps 0.00→**40.0** at L19 (OUTPUT_HI+0 →4.0 at L18) on the PC byte row; correct 0x2a sits at OUT_LO+10/HI+2 (≈0.9) and is swamped. **OP_ENT=12.63 broadcast on a LEA-step PC row.** 40.0 scale == L15 `pop_d8_to_e0` O-write (`l15_ops.py:2676`); store/lookup-band emitter mis-fires onto the LEA PC row. | **opcode-broadcast** (OP_ENT leaks onto non-ENT row) |
| **2** | **2** | **LEA** | **AX.b1** | **0xff** | **0x00** | residual 0x00 all blocks; weak +0.2 write at blk≈28 fades | `l16_lea_local_ax_byte1_ff_lo/hi` (`l16_ops.py:1978`) — gate FIRES (score 4.96≥4.5) but writes only ±20/S = **±0.2** to nib15; cannot beat the prevailing 0x00 mass. Sign-extension byte never materializes. | **missing/insufficient-writer** (discriminator OK, strength too weak) |
| 3† | 3 | PSH | PC.b0 | 0x32 | 0x10 | blk3 sets 0x10 | desync: neural PC already 0, re-fetched wrong op | desync debris |
| 4† | 3 | PSH | AX.b0 | 0xe8 | 0xf0 | ALU_LO/HI flips to 0xf at blk8–11 | desync; AX carries 0xf-flood | desync debris |
| 5† | 3 | PSH | AX.b1 | 0xff | 0x00 | 0x00 throughout | desync (same weak-byte1 shape as #2) | desync debris |
| 6† | 3 | PSH | SP.b0 | 0xe0 | 0xe8 | 0xe8 vs 0xe0 (off-by-8) | desync: SP decrement skipped | desync debris |
| 7† | 4 | IMM | PC.b0 | 0x3a | 0x18 | blk3 0x18 | desync | desync debris |
| 8† | 4 | IMM | AX.b0 | 0x1c | 0xf0 | 0xf-flood blk8+ | desync | desync debris |
| 9† | 4 | IMM | SP.b0 | 0xe0 | 0xe8 | off-by-8 | desync | desync debris |
| 10† | 5 | SI | PC.b0 | 0x42 | 0x20 | blk3 0x20 | desync | desync debris |
| 11† | 5 | SI | AX.b0 | 0x1c | 0xf0 | 0xf-flood | desync | desync debris |
| 12† | 5 | SI | BP.b1 | 0xff | 0x02 | runaway 0x02 flood, OUT_LO→1e15 by L25/L26 | desync; numerical blowup (carrier) | desync debris |
| 13† | 5 | SI | BP.b2 | 0x00 | 0x02 | same 0x02 runaway | desync | desync debris |
| 14† | 5 | SI | BP.b3 | 0x00 | 0x02 | same 0x02 runaway | desync | desync debris |

## The two links that actually matter (batch-fix target)

Only **#1 (LEA PC byte0)** and **#2 (LEA AX byte1)** are genuine clean-state
divergences on the real LEA step. Links 3–14 are all **downstream of the #1 PC
desync** and will vanish once #1 is fixed — do NOT chase them.

### Link #1 — LEA PC byte0 (the desync trigger) — OPCODE-BROADCAST class

- Genesis sweep (`probe_var_lea_pc_disc.py`): the PC byte row carries the
  correct PC=42 cleanly (`OUT_LO+10≈0.9, OUT_HI+2≈0.9` → 0x2a) through
  **block 27**; at **block 28 (L18)** OUTPUT_HI+0 jumps to 4.0 and at
  **block 29 (L19)** OUTPUT_LO+0 jumps to **40.0**, flipping the OUTPUT decode
  to 0x00. LM head then emits 0x00 (logit 216) over 0x2a (logit 14).
- **`OP_ENT = +12.63` is broadcast on this LEA-step PC byte row** (a non-ENT
  step). This is the recurring opcode-broadcast signature: a frame-opcode
  flag reaching a row it should not, defeating an intent blocker. ENT's
  broadcast here (12.63) exceeds the audit's 7.3–11.4 (it peaks at the
  ENT-adjacent rows).
- The **40.0** O-write scale is the L15 `pop_d8_to_e0` head
  (`l15_ops.py:2669-2677`, "writes the matched value slot back to
  OUTPUT_LO/HI with scale 40.0"); a store/lookup-band emitter in the
  **L15→L19** range mis-fires onto the LEA PC row. The fix surface is the
  L18/L19 OUTPUT_LO+0 0x00 flood writer; promote its IS_BYTE +
  non-PC-marker / OP_ENT NOT-blockers to hard `-1e6` per the
  `OPCODE_BROADCAST_BLOCKER_AUDIT` pattern.

### Link #2 — LEA AX byte1 (sign extension) — MISSING/WEAK-WRITER class

- `l16_lea_local_ax_byte1_ff_lo/hi` (`l16_ops.py:1978-2001`) gate at the AX
  byte1 row **FIRES** (CMP+7=1, H1+1=1, IS_BYTE=1, HAS_SE=0.99, BYTE_INDEX_0=0.97
  → score 4.96 ≥ thr 4.5) — so the discriminator is NOT the problem. The
  failure is **write strength**: it writes only `±20/S = ±0.2` to OUTPUT nib15.
  That 0.2 cannot overcome the dominant 0x00 default on the AX byte1 row, so the
  residual stays 0x00 and the 0xff sign-extension byte never appears.
- This is NOT the opcode-broadcast class; it's an under-powered corrective
  writer. The batch fix is to raise its write magnitude (and/or add an L10
  `tail_ax_lea_local_addr_byte1_ff_after_*` consumer-side amplifier — that L10
  rule writes 0xFF at `strength=5000` but is keyed on BYTE_INDEX_0, i.e. the
  byte0 row, so it does not reach byte1).

## How many links are the broadcast class (batch-fixable together)

- **1 of the 2 genuine links** (link #1, LEA PC byte0) is the **opcode-broadcast**
  class — the same class as the 5 already-fixed prologue links and the 16-18
  latent families in `docs/OPCODE_BROADCAST_BLOCKER_AUDIT_2026_06_11.md`. It is
  batch-fixable with the proven hard-`-1e6`-NOT-blocker promotion.
- **1 of the 2** (link #2, LEA AX byte1) is a **weak-writer** one-off (raise
  strength; discriminator already correct).
- The 12 downstream links (#3–14) are **desync debris** — zero independent
  fixes; they disappear when #1 lands.

## Next agent — after fixing #1 + #2, RE-RUN this probe

`tools/probe_var_full_chain.py 262` will then re-expose the store (SI), load
(LI), and LEV return on a clean state. Those links CANNOT be pinned now because
they never execute with a valid PC. The brief's "store → load → LEV" sequence is
gated behind the step-2 LEA PC desync; this map converts the chase into:
fix #1 (broadcast) + #2 (weak writer), re-run, re-map the body.
