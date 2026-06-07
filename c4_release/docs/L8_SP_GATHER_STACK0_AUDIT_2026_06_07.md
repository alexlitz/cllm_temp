# L8 sp_gather STACK0 byte 1/2/3 audit (2026-06-07)

Followup to `MEMORY_L14_FIX_ATTEMPT_2026_06_07.md` (commit `9ca00578`).
Tested whether `layer8_sp_gather_bake` broadcasts STACK0 byte 1/2/3
values to the rows where L14 `mem_generation` expects to read them.

**Result: NO. The hypothesis in the L14 fix doc is partially
confirmed, but it mis-identifies what sp_gather actually broadcasts.**

## TL;DR

* `layer8_sp_gather_bake` lands at **layer 10** in the production
  bake (not L8 — `target_op_name="layer10_byte_passthrough"` re-pins
  it via the dep graph).
* L10 sp_gather (heads 0-2) DOES fire at MARK_STACK0 rows and writes
  `ADDR_B0/B1/B2_LO/HI` there.
* What it broadcasts is **SP register byte values**, gathered from
  SP byte rows (K-side fires at `BYTE_INDEX_J + H1+SP_I` — the SP
  byte J position).
* It does NOT broadcast STACK0 byte values 1/2/3 — those values are
  never in the residual stream at L14's read positions.

## Method

`tools/probe_l8_sp_gather_stack0_audit.py` runs `test_si_li_roundtrip`
(`IMM 0x200, PSH, IMM 42, SI, IMM 0x200, LI, EXIT`) with forward
hooks at L2/L5/L7/L8/L10/L13/L14, then probes:

* CLEAN_EMBED / STACK0_BYTE0/1/2/3 / BYTE_INDEX_0/1/2/3 at STACK0
  marker rows + d=1..9.
* ADDR_B0/B1/B2_LO/HI across L7/L8/L13/L14 at MARK_STACK0 / MARK_AX /
  MEM-addr-byte rows.

L8 attn weight inspection (`/tmp/probe3.py`) confirms which heads at
which layer write ADDR_B*.

## Where ADDR_B*_LO/HI gets written (audit of W_o by layer/head)

| Layer | Head | Q gate | W_o → ADDR_B0/B1/B2_LO |
|------:|-----:|--------|------------------------|
| L8    | 2    | MARK_AX        | B0 (16.0)  ← L7 memory_heads (placed at L8 layer) |
| L8    | 3    | MARK_AX        | B1 (16.0)  ← L7 memory_heads |
| L8    | 4    | MARK_AX        | B2 (16.0)  ← L7 memory_heads |
| L10   | 0    | MARK_STACK0    | B0 (112.0) ← layer8_sp_gather_bake.head_0 |
| L10   | 1    | MARK_STACK0    | B1 (112.0) ← head_1 |
| L10   | 2    | MARK_STACK0    | B2 (112.0) ← head_2 |
| L10   | 6    | MARK_SP        | B0 (112.0) ← head_6_mark_sp_mirror |
| L10   | 7    | MARK_SP        | B1 (112.0) ← head_7_mark_sp_mirror |
| L11   | 0/1  | —              | B0 (16.0 each) |
| L14   | 0/1/2| —              | B0/B1/B2 (16.0)  ← mem_generation heads |

The 112.0 mag for L10 heads 0-7 is the (1+2N) magnitude lift via
the `_addr_mag_boost` cancel-pair (slots 34/35 with N=3 → 7 per
output dim × 16 dims = 112), per the `_band_output_writes` comment.

## Per-row residual at MARK_STACK0 across layers (SI step, STACK0 @ 114)

```
Layer  ADDR_B0_LO/HI   ADDR_B1_LO/HI   ADDR_B2_LO/HI   interpretation
L7     [0/0, 0/0]      [14/0, 0/0]     [0/0, 0/0]      empty (L7 fires at MARK_AX, not STACK0)
L8     [0/0, 0/0]      [0/0, 0/0]      [0/0, 0/0]      empty
L13    [8/1.57, 15/1.57] [15/1.53,15/1.53] [0/0.99, 0/1.00]  byte = 0xF8 / 0xFF / 0x0F
L14    same as L13                                      passes through
```

The L13 residual at MARK_STACK0 = (0xF8, 0xFF, 0xF0) is the SP
register value (0xFFFFFFF8 = -8, after PSH popped SP up by 4 then
underflowed past 0). So L10 sp_gather is writing the **SP register
bytes** to ADDR_B0/B1/B2 at MARK_STACK0, not the STACK0 stored value.

(The L13 capture shows the value because L10 sp_gather → L13 KV cross-
step. The L8/L7 hooks capture too early.)

## What the L14 mem_generation heads actually need

`_layer14_mem_generation_head_specs` slot 2 (SI/SC STACK0 source path):
- head 0: K reads `STACK0_BYTE0` flag → attends to STACK0 byte 0 row,
  copies its CLEAN_EMBED. **Empirically zero** (CLEAN_EMBED at STACK0
  byte 0 row is `0/0` = byte 0x00).
- head 1: K reads `L1H4+BP_I - H1+BP_I` → attends to ~STACK0 byte 0 row.
  Empirically zero.
- head 2: K reads `H2+BP_I - L1H4+BP_I` → STACK0 byte 1 row. Empirically zero.
- head 3: K reads `H3+BP_I - H2+BP_I` → STACK0 byte 2 row. Empirically zero.

The STACK0 byte rows' `CLEAN_EMBED_LO/HI` are all 0 at every step in the
SI/LI test program. Even right after PSH (which should copy AX = 0x200
onto STACK0), the STACK0 byte rows still contain CLEAN_EMBED = (0,0).

This means there is **no producer of the pushed value at the STACK0
byte rows**. PSH's autoregressive emission of STACK0 tokens drives the
value into the token stream, but only byte 0's emission is correct (and
it lands in `STACK0_BYTE0` via a dedicated route); bytes 1/2/3 are
emitted as zero because no head broadcasts the AX byte 1/2/3 values to
those rows.

## Diagnosis

The L14 fix doc was right that the bug is upstream of L14, but the
"L8 sp_gather only broadcasts byte 0" framing is incorrect. The reality:

* **L10 sp_gather (head 0-2) broadcasts SP register byte values**
  (gathered from `H1+SP_I` × `BYTE_INDEX_J` keys) to `ADDR_B0/B1/B2`
  at MARK_STACK0 rows. This is what produces the LI's address lookup
  Q-side.
* There is **NO op that broadcasts AX register byte values 1/2/3 to
  the STACK0 byte 1/2/3 rows** of the current step's STACK0 frame.
  PSH emits STACK0 byte 0 correctly (via the L10 byte-passthrough
  STACK0 byte 0 route, gated by PSH semantics), but bytes 1/2/3 emit
  zero.
* Consequently, when the SI step runs, the STACK0 frame's bytes 1/2/3
  are uninitialised (or rather, contain 0x00 from the previous step's
  PSH emission), and L14 mem_generation's BP-relative reads pull
  zeros out for the addr byte 1/2/3 prediction.

## Where bytes 1/2/3 SHOULD come from

In a complete design, PSH would broadcast AX byte 1/2/3 values to
STACK0 byte 1/2/3 rows so the autoregressive STACK0 emission picks
them up. The current production model only does this for byte 0
(via L10 byte_passthrough's STACK0 byte 0 route).

Candidate fix surfaces:
1. **Extend L10 byte_passthrough to emit STACK0 bytes 1/2/3** (analog
   to its byte-0 STACK0 path), gated on PSH/JSR/ENT semantics.
2. **Add a sibling op to L8 sp_gather that broadcasts AX byte values
   to STACK0 byte rows** at PSH steps — a "psh_ax_gather" that fires
   at MARK_STACK0 + IS_BYTE rows and gathers from AX bytes via
   `H1+AX_I × BYTE_INDEX_J` keys, writing to STACK0_BYTE1/2/3 dims
   (so L14 can read them via the proposed migration).
3. **The L13->L14 doc's recommended L14 head spec migration** (read
   STACK0_BYTE1/2/3 directly) requires #2 first, otherwise STACK0_BYTE1/2/3
   are still all zero at the STACK0 byte rows.

## Recommended fix

The L14 fix doc's "extend L8 sp_gather (or sibling) to broadcast bytes
1/2/3" is the right direction, **but the data to broadcast is AX byte
values during PSH** (not "STACK0 byte values"; those don't exist as a
separate source — they ARE what we want to produce).

Concretely (NOT IMPLEMENTED HERE — out of bounded-fix scope; requires
new attention head + careful Q/K gating + verifier-pass):

1. New op `layer10_psh_ax_to_stack0_byte` (kind="block",
   target_op_name="layer10_byte_passthrough"):
   - 3 heads h ∈ {1, 2, 3} with Q gate on `MARK_STACK0 + BYTE_INDEX_h
     + OP_PSH` and K gate on `MARK_AX + BYTE_INDEX_h` (the AX byte h
     row). V copies `CLEAN_EMBED_LO/HI` from the AX byte h row. O
     writes to `STACK0_BYTE_VAL_h_LO/HI` (new dims, OR alternatively
     directly into `CLEAN_EMBED_LO/HI_THIS_STEP` so the autoregressive
     emission picks it up).
2. After (1), migrate L14 mem_generation head 1/2/3 to read the new
   `STACK0_BYTE_VAL_h_LO/HI` dims (or the boosted CLEAN_EMBED) at the
   STACK0 byte rows.
3. Smoke-test on `test_si_li_roundtrip` end-to-end.

The dim-broadcast version (writing to `STACK0_BYTE_VAL_h_LO/HI` dims)
is safer than writing to CLEAN_EMBED because CLEAN_EMBED is read by
many downstream ops; clobbering it at STACK0 byte rows risks regression.

## Why I am NOT implementing the fix in this audit pass

* New attention heads need head-slot allocation (none free at L10 in
  the production layout — heads 0/1/2/6/7 already pinned for
  sp_gather; head 3 for multibyte_fetch; head 4 for op_imm_relay;
  head 5 mem_to_alu).
* The new dim family (`STACK0_BYTE_VAL_h_LO/HI`, 96 dims) needs
  registry slots.
* Verifier-magnitude collision audit (per the `_addr_mag_boost`
  precedent at L8) must be redone for the new O-target.
* Single-rule-fix heuristic: per `feedback_single_rule_fixes_are_zero_sum.md`,
  0/5 historical agents made progress with stacked-rule fixes; this
  needs a verifier-driven plan, not a brief-driven one.

## Files

* `c4_release/tools/probe_l8_sp_gather_stack0_audit.py` — the probe.
* `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:1754-1974`
  — `make_layer8_sp_gather_bake_op` (binds to L10 anchor).
* `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:2115-2150`
  — `layer10_byte_passthrough` anchor.
* `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:407-545`
  — `_layer14_mem_generation_head_specs` (the BP-relative reads).
* `c4_release/neural_vm/setup_helpers_l2.py:71-109` — L2 FFN that
  writes STACK0_BYTE1/2/3 flags (FLAGS only, not values) at byte rows.
