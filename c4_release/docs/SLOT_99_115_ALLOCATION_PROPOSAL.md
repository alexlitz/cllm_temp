# Slot 99-115 Allocation Proposal (Structural Wave B8)

Scope: scaffold the next structural-allocation wave to reclaim the 17
remaining dims at slots 99-115 (the dormant tail of the L0 H5 / H6 / H7
threshold heads). See `CAMPAIGN_SUMMARY.md` §7.5 and §9.5, plus the B6-K
BD dim usage map (`investigation/bd-dim-usage-map`, commit `6e8ab77`).

**Status:** PROPOSAL ONLY. No source files modified. Implementation is
deferred until this proposal is reviewed.

---

## 1. Background

### 1.1 Slot-95-98 baseline (B7-1..5, already landed)

The B7 work consumed the first 4 of the 21 dead H5/H6/H7 dims as
single-bit lifecycle / boolean dims (see `dim_registry.py` lines 609-620):

| Slot | Name                     | Producer                                           | Notes                                            |
|------|--------------------------|----------------------------------------------------|--------------------------------------------------|
| 95   | `SP_BYTE0_IS_F8`         | L7 head 6, V slots 6+7 → O (`l7_ops.py:477-536`)   | aliases H5+0                                     |
| 96   | `IN_STEP_FRESH`          | L1 head 5 (`l1_ops.py:152-168`)                    | aliases H5+1; positive ALiBi decay from MARK_SE  |
| 97   | `ADDR_B0_VALID`          | L13 head 0 slot 34 (`l13_ops.py:33-46`)            | aliases H5+2                                     |
| 98   | `SP_GATHERED_THIS_STEP`  | L8 FFN unit 2055 (`l8_ops.py:1119-1225`)           | aliases H5+3; SwiGLU pulse on MARK_SP             |

Each producer follows the same pattern: (a) hook a single
attention-head V/O slot or FFN unit into an existing layer, (b) add a
`reg.alloc(... semantics="...")` entry, (c) have downstream rules read
it via the standard `("DIM_NAME", weight)` condition tuple. L10 tail
rules (e.g. `l10_ops.py:3700-4135`) now consume `SP_BYTE0_IS_F8`,
`IN_STEP_FRESH`, and `ADDR_B0_VALID` directly.

### 1.2 The 17 dims still free

The full layout of slots 95-115 is:

```
H5: dims 95..101  (slot 95 = SP_BYTE0_IS_F8, 96 = IN_STEP_FRESH,
                   97 = ADDR_B0_VALID,        98 = SP_GATHERED_THIS_STEP,
                   99, 100, 101 — FREE)
H6: dims 102..108 (102..108 — ALL FREE)
H7: dims 109..115 (109..115 — ALL FREE)
```

Total = 3 (H5 tail) + 7 (H6) + 7 (H7) = **17 reclaimable dims**.

L0 still writes the full 7-wide threshold output to H5/H6/H7
(`l0_ops.py:143-179`) but no other layer reads dims 99-115. From the
verifier's perspective these slots are pure waste. Reclaiming requires
either (a) shrinking the L0 head-O footprint to 4 slots and rerouting
the high slots, or (b) leaving the dead L0 writes in place and
overwriting from a later layer (the B7 approach: aliased lifecycle
bits — works because no downstream rule reads the L0 H5/H6/H7 values).

### 1.3 Bug clusters targeted

Per `BUG_CATALOG.md`:

- **#27 / #29 — SP_byte0 step-2 cluster (~275 first-fatal rows).**
  Soft ADDR_B0/B1/B2 evidence misfires; lifecycle witnesses currently
  only available for B0, not B1/B2.
- **#30 / #31 — PC_byte0 / PC_byte1 clusters (42 + 46 rows).**
  No structural lifecycle signal for "PC byte resolved this step".
- **#13 / #16 — `SP byte 0 == 0xF8` proxied via OUTPUT_LO+8 /
  OUTPUT_HI+15 because no SP-byte structural channel exists.** Only a
  single-byte `IS_F8` boolean exists today (slot 95); a wider SP byte 0
  one-hot would let consumers gate on byte values other than 0xF8.
- **#28 — STACK0_byte2 cluster on `var_three_*` (25 rows).** Likely
  needs a STACK0-byte-resolved lifecycle witness.

---

## 2. Candidate allocations

Twelve candidates scoped. Layout choice is a one-pass cost: each
candidate consumes 1 or 32 slots. The 17 free dims allow either a full
32-wide one-hot (NOT possible — only 17 slots free) or a mix of
single-bit lifecycle witnesses and partial multi-bit channels.

### 2.1 Constraint note: 32-wide one-hots will not fit

The user request mentions `SP_BYTE0_VALUE` as a 32-dim one-hot. Only 17
dims are free in the 99-115 window. Options:

- **Option A — partial 8-bit / 16-bit SP byte 0 channel.** Use an 8-bit
  *bit-decomposition* of SP byte 0 (each bit is one slot) rather than a
  256-wide one-hot. Costs 8 slots; covers the full byte equality space.
  This is the recommended fallback for SP_BYTE0_VALUE.
- **Option B — sentinel one-hots for high-value SP byte 0 constants.**
  Allocate `SP_BYTE0_IS_F0`, `SP_BYTE0_IS_E0`, `SP_BYTE0_IS_00`,
  `SP_BYTE0_IS_FF` as four lifecycle bits (4 slots) instead of a full
  one-hot. Cheaper and addresses the actual fatal-byte distribution
  observed in #27 / #29 / #13.
- **Option C — defer SP_BYTE0_VALUE.** Allocate it in a future wave
  after retiring an unused 16-wide slot elsewhere (e.g. one of the
  dormant CLEAN_EMBED aliases at 306..435 if profiling shows the slot
  is dead).

This proposal recommends **Option B** as the lowest-cost path to
covering the SP_byte0 cluster.

### 2.2 Catalog (ordered by expected impact / implementation cost)

Format: `[slot, name, size, semantics, producer, consumer(s), impact, cost]`.

| # | Slot(s)     | Name                   | Size | Semantics                                   | Producer (layer / head)             | Consumer(s)                                 | Impact (bug)         | Cost     |
|---|-------------|------------------------|------|---------------------------------------------|--------------------------------------|----------------------------------------------|----------------------|----------|
| 1 | 99          | `ADDR_B1_VALID`         | 1    | `addr_b1_valid` (mirror B0)                  | L13 head 1 slot 34 (mirror B0)       | L10 tail rules reading ADDR_B1 lanes         | #27 / #29 partial    | LOW      |
| 2 | 100         | `ADDR_B2_VALID`         | 1    | `addr_b2_valid` (mirror B0)                  | L13 head 2 slot 34 (mirror B0)       | L10 tail rules reading ADDR_B2 lanes         | #27 / #29 partial    | LOW      |
| 3 | 101         | `LAST_OP_WAS_PSH`       | 1    | `mark in {SP, STACK0} AND opcode_in_step == PSH AND in_step_fresh` | L1 FFN unit OR L8 SwiGLU pulse | L10 tail PSH stack-top rules (e.g. `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority`) | #1 / #18                | LOW      |
| 4 | 102         | `SP_BYTE0_IS_E0`        | 1    | `mark == SP AND sp_byte0 == 0xE0`            | L7 head 6 (clone of IS_F8 producer, EMBED_LO+0 / EMBED_HI+14) | L10 SP-marker tail rules                     | #13 / #29            | LOW      |
| 5 | 103         | `SP_BYTE0_IS_00`        | 1    | `mark == SP AND sp_byte0 == 0x00`            | L7 head 6 (clone, EMBED_LO+0 / EMBED_HI+0) | L10 tail rules; ADDR_B0=0 disambiguation       | #2 / #29             | LOW      |
| 6 | 104         | `SP_BYTE0_IS_FF`        | 1    | `mark == SP AND sp_byte0 == 0xFF`            | L7 head 6 (clone, EMBED_LO+15 / EMBED_HI+15) | L10 tail rules                                | #29 partial          | LOW      |
| 7 | 105         | `PC_BYTE_VALID`         | 1    | `mark == PC AND pc_resolved_this_step`        | L13 head extension OR L8 FFN at MARK_PC | L10 rules currently using OUTPUT_LO+i proxies for PC byte | #30 / #31         | MEDIUM   |
| 8 | 106         | `STACK0_BYTE_VALID`     | 1    | `mark == STACK0 AND addr_known_this_step`     | L13 head 1 or new L14 FFN            | L16 STACK0 tail rules; L10 STACK0_byte2     | #28                  | MEDIUM   |
| 9 | 107-108     | `STEP_INDEX_LO/HI`      | 2    | `mark == any AND step_index in {0..3} / {4..7}` | L1 FFN unit reading HAS_SE history (running counter) | rules needing "step 2 specifically"     | #29 (gating)         | MEDIUM-HIGH |
| 10| 109-112     | `SP_BYTE0_BITS[0..3]`   | 4    | per-bit value of SP byte 0 low nibble        | L7 head 6 (bit-decomposed read of EMBED_LO) | L10 rules; partial byte equality          | #29 (full byte cmp) | HIGH     |
| 11| 113         | `MEM_VAL_VALID`         | 1    | `mark == MEM AND mem_val_loaded_this_step`   | L13 / L14 FFN; or L10 MEM-VAL phase    | L10 MEM-val tail rules                    | #2 / #12            | MEDIUM   |
| 12| 114-115     | reserve / `BP_FRAME_VALID`, `JSR_RET_VALID` | 2 | BP frame / JSR return-addr resolution gates | L14 / L15 FFN                       | L16 BP_FRAME rules (project_l16_bp_frame_byte1_ff_dual) | open bug (memory)  | MEDIUM   |

Total: **17 dims allocated** (all candidates fit exactly in the 99-115
window).

---

## 3. Top 5 — detailed sketches

### 3.1 #1 — `ADDR_B1_VALID` (slot 99)

**Producer.** L13 head 1 currently writes 32 V slots (`l13_ops.py:29-31`):

```
W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
```

Mirror the B7-4 trick on head 1: extend with slot 34 reading
`L1H1+MEM_I=4` (or a dedicated MEM-row-detect signal), wire W_o to
ADDR_B1_VALID:

```python
# Pseudocode (sketch only):
_claims.add((13, "attn_W_v", "1_34", "L1H1+4"))
_claims.add((13, "attn_W_o", "1_34", "ADDR_B1_VALID+0"))
# In setup_helpers._set_layer13_mem_addr_gather, mirror the ADDR_B0_VALID block
# but for head 1 → ADDR_B1_VALID.
```

**Consumer.** L10 rules `tail_mem_store_*_addr1_*` (e.g. lines 3850+ in
`l10_ops.py` read ADDR_B0_VALID as the proxy freshness witness).
Replace `("ADDR_B0_VALID", 50.0)` with `("ADDR_B1_VALID", 50.0)` on rules
whose dominant evidence lane is the ADDR_B1 nibbles.

**Expected impact.** Closes the asymmetry where B0 has a lifecycle
witness and B1/B2 use B0 as a proxy (see `l10_ops.py:3852` comment:
"no separate VALID bit exists for B1/B2 per B7-4"). Partial fix to
bug #27 / #29 because some SP_byte0 misfires originate in B1/B2 lanes
gated on a stale B0 witness.

### 3.2 #2 — `ADDR_B2_VALID` (slot 100)

**Same producer pattern** as #1 on L13 head 2. Consumers: any
`tail_mem_*_addr2_*` rule and the active_value/active_byte-2 evidence
lanes in `l10_ops.py:4079-4135`.

**Impact / cost.** Same as #1. Allocating both #1 and #2 together is
nearly free (one helper function clone + 2 W_o claim entries each).

### 3.3 #3 — `LAST_OP_WAS_PSH` (slot 101)

**Producer.** Two viable sites:
- L1 FFN: hook a SwiGLU unit reading `OP_PSH` (slot 286 area, see
  OPCODE_FLAGS section in `dim_registry.py:564-582`) at MARK_SP rows.
  Pattern matches `_layer8_sp_gathered_sentinel_rule` (l8_ops.py:1126).
- L8 FFN: pulse at MARK_STACK0 right after `layer8_sp_gather_bake`
  (phase 8.0). Phase=8.65 (after sentinel @ 8.6).

```python
# Pseudocode (L1 FFN, ~slot 286 OP_PSH read):
def _layer1_last_op_was_psh_rule(S: float) -> FFNRule:
    return FFNRule.gated_write(
        name="l1_last_op_was_psh",
        conditions=(("MARK_SP", 1.0), ("OP_PSH", 1.0), ("IN_STEP_FRESH", 1.0)),
        threshold=2.5,
        gate=None, gate_bias=1.0,
        writes=(("LAST_OP_WAS_PSH", 2.0 / S),),
    )
```

**Consumer.** `tail_mem_store_addr0_e0_from_psh_sp_no_addr_src_authority`
(bug #1, the L10 rule that overpowers ENT-main). Gating on
`LAST_OP_WAS_PSH` instead of inferring from MARK_SP residue would
collapse the rule's false-fire surface area.

**Impact.** Directly addresses bug #1 (the open L10 PSH addr0 issue
documented in user memory `project_l10_psh_addr_ent_bug.md`).

### 3.4 #4-6 — `SP_BYTE0_IS_E0` / `IS_00` / `IS_FF` (slots 102-104)

**Producer.** Clone the L7 head-6 spec from `_layer7_sp_byte0_is_f8_spec`
(`l7_ops.py:547-569`). For each byte value, the V reads change to the
appropriate EMBED_LO / EMBED_HI nibble indices:

```
0xE0 = nibble(LO=0, HI=14)
0x00 = nibble(LO=0, HI=0)
0xFF = nibble(LO=15, HI=15)
0xF8 = nibble(LO=8, HI=15)   # already exists
```

Each new sentinel adds two head-6 V/O slots (the head already
contains 6 active slots: 0..7 with 1..5 from PSH/CMP/JSR relay + 6/7
from F8). Slots 8/9, 10/11, 12/13 are free on head 6.

**Consumer.** L10 SP-marker tail rules currently reading
`("ADDR_B0_LO+8", weight)` to identify SP byte 0 == 0xF8 can be
extended:

```python
# Before:
conditions=(("MARK_SP", 1.0), ("ADDR_B0_LO+8", 50.0), ...)
# After (for 0xE0 family):
conditions=(("MARK_SP", 1.0), ("SP_BYTE0_IS_E0", 50.0), ...)
```

This decouples L10 from the soft-ADDR_B0 evidence chain (the actual
B5-D / B6-B regression source per bug #27).

**Impact.** Addresses #13 (the structural-channel gap) and a portion of
#29 (SP byte 0 step-2 cluster) by giving consumers a direct lifecycle
witness instead of the OUTPUT_LO/HI proxy.

### 3.5 #7 — `PC_BYTE_VALID` (slot 105)

**Producer.** L13 currently produces ADDR_B0_VALID at MARK_MEM rows;
PC_BYTE_VALID would be a MARK_PC-row analogue, set when L8's PC byte
gather has populated the PC nibbles for the current step.

```python
# Sketch — new L13 head slot (or new L8 FFN unit):
# Q: MARK_PC, K: MARK_PC, V: L1H1+PC_I (PC-row marker), O: PC_BYTE_VALID
# Same shape as the ADDR_B0_VALID producer; reuse helper.
```

**Consumer.** L10 / L11 rules currently lacking any PC freshness
gate; bug clusters #30 (PC_byte0 in slice 548-821) and #31 (PC_byte1
in slice 822-1095) both have NO existing structural lifecycle channel,
which is part of why they fail open.

**Impact.** Likely partial fix to #30 / #31 — won't solve them alone but
gives consumer rules a witness to gate on instead of proxying via
HAS_SE.

---

## 4. Implementation order (recommended)

**Wave B8-A — cheap symmetry (1 commit, ~80 LOC).**

1. `ADDR_B1_VALID` (slot 99) — L13 head 1 mirror.
2. `ADDR_B2_VALID` (slot 100) — L13 head 2 mirror.
3. Retrofit L10 `tail_*_addr1_*` and `tail_*_addr2_*` rules to read the
   correct VALID bit instead of borrowing ADDR_B0_VALID.

Single commit shape:

```
- dim_registry.py: +2 reg.alloc entries (slots 99, 100).
- l13_ops.py: +2 _claims entries per head; reads/writes set update.
- setup_helpers (_set_layer13_mem_addr_gather): mirror head-0 slot-34 logic on heads 1 and 2.
- l10_ops.py: ~10 rule conditions changed from ADDR_B0_VALID to ADDR_B1_VALID / ADDR_B2_VALID.
- tests/: extend l13_per_op test; add ADDR_B*_VALID asserts.
```

**Wave B8-B — SP byte 0 sentinel family (1 commit, ~120 LOC).**

1. `SP_BYTE0_IS_E0`, `SP_BYTE0_IS_00`, `SP_BYTE0_IS_FF` (slots 102-104).
2. Clone `_layer7_sp_byte0_is_f8_spec` three times with different V
   reads; add to L7 head 6.
3. Migrate L10 SP-marker tail rules to read the dedicated lifecycle
   bits instead of `ADDR_B0_LO+8` / `OUTPUT_LO+8` etc. proxies.

**Wave B8-C — LAST_OP_WAS_PSH (slot 101, 1 commit, ~50 LOC).**

Single FFN unit + L10 rule rewrite for bug #1.

**Wave B8-D — PC / STACK0 / MEM lifecycle bits (multiple commits).**

Bugs #30 / #31 (PC) and #28 (STACK0_byte2) require additional
investigation before allocation — the producer site for
PC_BYTE_VALID isn't obvious from the current PC gather chain. Defer
these until the B8-A and B8-B waves have been measured against the
1096 retest.

**Recommended order:** A → B → C → D. A is the cheapest and unblocks
the asymmetric VALID-witness gap that B7-4 left half-finished.

---

## 5. Re-validation: does the B6-K reclaimable-dim map still apply?

Mostly yes, with one caveat. The B6-K map (`6e8ab77`) inventoried 21
reclaimable dims at H5/H6/H7. The B7-1..5 work consumed 4 of them
(slots 95-98), leaving 17. The 17-dim count is consistent with this
proposal's enumeration of slots 99-115.

**Caveat 1.** B6-K predates the IO-state / IO-tool-call alias additions
in slots 322-359 and the OUTPUT_BYTE / FORMAT_PTR aliasing at 480-502.
None of those allocations touch the 95-115 window, but the rest of the
B6-K map (which catalogued the *entire* dim plan) is now out of date
elsewhere. **If a future wave wants to reclaim outside 99-115, the
audit should be redone.**

**Caveat 2.** B6-K's analysis assumed L0 head writes were the *only*
producers of slots 95-115. Post-B7, slots 95-98 have layered writes
(L0 still writes the threshold values to H5+0..3; B7 producers
overwrite them in later layers). The aliasing trick is sound — no
downstream rule reads the L0 values — but the verifier's
write-before-read check has to special-case the L0 wastes. The
verifier accepts this today (the B7 commits landed), so the same
trick is available for slots 99-115.

**Recommendation.** A fresh B8-K-style audit is **not** required for
this wave (B6-K's slot-95-115 inventory is still accurate). For waves
that touch other dim ranges, redo the audit.

---

## 6. Next steps

**Most valuable to implement first:** Wave B8-A (`ADDR_B1_VALID` +
`ADDR_B2_VALID`). Reasons:
- Lowest cost: ~80 LOC, one commit.
- Closes a known asymmetry (`l10_ops.py:3852` comment).
- Directly addresses bugs #27 / #29 partial — the SP_byte0 cluster
  has B1/B2 lane components that B0-only gating misses.
- Low risk: producer pattern is a literal mirror of the existing
  ADDR_B0_VALID producer; consumer rewrite is a 1-line condition swap.

**Rough commit shape (B8-A):**

1. `dim_registry.py`: 2 new `reg.alloc()` calls (slots 99, 100), each
   with a `semantics="addr_bN_valid"` predicate string.
2. `l13_ops.py`: add 4 new `_claims` entries (2 per new head), update
   `writes={...}` set.
3. `vm_step._set_layer13_mem_addr_gather`: extract the head-0 slot-34
   bake into a helper, invoke it 3 times for heads 0/1/2.
4. `l10_ops.py`: ~10 rule edits changing `ADDR_B0_VALID` reads to
   `ADDR_B1_VALID` / `ADDR_B2_VALID` where the rule's dominant
   evidence is in the B1/B2 lane.
5. `tests/test_l13_per_op.py`: extend with VALID-bit assertions for
   heads 1 and 2.

Expected delta on 1096 retest: +20 to +50 ids (B8-A is a partial
fix to bug #27, not a full one; the remaining surface is bug #29's
SP-byte-value gating, which B8-B addresses).

---

## 7. Out of scope (do NOT implement in this wave)

- Full 256-wide or 32-wide `SP_BYTE0_VALUE` one-hot: does not fit in
  the 17 free slots; would require evicting a slot elsewhere or
  expanding d_model.
- Reclaiming H5/H6/H7 L0 *producer* writes (i.e. shrinking the L0
  attention head O-rows): unnecessary for the consumer reads to
  ignore the L0 values, and structurally invasive.
- B8-D's PC / STACK0 / MEM lifecycle bits: scoped above but
  deliberately not committed to a producer site yet; needs
  investigation before allocation.
