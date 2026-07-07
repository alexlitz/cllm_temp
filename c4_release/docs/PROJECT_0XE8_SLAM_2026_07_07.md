# PROJECT_0XE8_SLAM — the frame-address slam into OUTPUT on VALUE/COMPARISON rows

Phase 1: SCOPE + DESIGN ONLY. Byte-identical to golden
`b1dcae630381bbe93ece7a53efbeadf4a6fefef0227db8fa17fba81d76ad53f5`
throughout. No weight changes. This doc + `tools/probe_e8_slam_rowsig.py`
are the foundation for the Phase-2 fix.

Status: **IN PROGRESS** (this doc is filled in as the probes land).

---

## 0. TL;DR (the one-paragraph root cause)

The two owning rules
`layer16_lev_routing::l16_psh_mem_addr0_restore_lo_8` and
`::l16_psh_mem_addr0_restore_hi_14` (physical block 34 in the campaign
60-block model = logical L16 lev-routing) are supposed to restore a
stack-frame PSH address byte `0xE8`/`0xE0` **only at the MEM store-address
marker** of a genuine local-frame PSH. They are scoped
`mark == MEM`, hard-block every other marker with `-1e6`, and gate
`PSH_AT_SP + MEM_STORE + MARK_MEM`. **But they fire on the AX
comparison/value marker at step 0 of if_gt / if_eq / … and slam
`0xE8`/`0xE0` over the true comparison result** (id360 `8>27` wants AX
byte0 `0x08`, gets `0xe8`; id402 `16==9` wants `0x10`, gets `0xe0`).

**Why the `-1e6 MARK_AX` blocker fails:** the rule's *positive trigger
condition is a self-read of the very OUTPUT nibble it writes* —
`(OUTPUT_LO nibble 8, weight +1.0)` for `restore_lo_8`,
`(OUTPUT_HI nibble 14, weight +1.0)` for `restore_hi_14`. On a clean
GENUINE restore row those nibbles are small one-hots (magnitude ~1–2), so
`S·(+1)·1 = +100` is tiny and the `-1e6 MARK_AX` blocker dominates → the
rule stays OFF on AX rows. On a COMPARISON/VALUE row the OUTPUT nibble
cells carry a HUGE amplitude (~+5.8e6) written by the upstream
comparison/decode band, so the self-read term is `S·(+1)·5.83e6 =
+5.83e8`, which **overwhelms the fixed `-1e6·1 = -1.0e8` MARK_AX blocker by
~5.8×** → `up = +4.83e8 > 0` → the AND spuriously fires and the
`+1e7/S` restore write slams `0xE8`/`0xE0` onto OUTPUT. The blocker was
calibrated for a bounded (0/1) OUTPUT one-hot; the trigger dim is in fact
amplitude-unbounded.

This is an **AND-gate calibration bug**: a fixed hard-blocker weight
against an amplitude-unbounded positive condition term.

---

## 1. The rules (as authored) — `neural_vm/unified_compiler/ops/l16_ops.py`

`_layer16_lev_routing_rules` → `psh_mem_addr0_conditions` (l16_ops.py:1392)
+ `l16_psh_mem_addr0_restore_lo_8` (l16_ops.py:1416) +
`l16_psh_mem_addr0_restore_hi_{k}` for k in 1..15 (l16_ops.py:1427).

```
psh_mem_addr0_conditions = (
    ("PSH_AT_SP", 1.0),
    (OP_JSR, -1000.0), (OP_ENT, -1000.0),
    (MARK_MEM, 1.0), ("MEM_STORE", 1.0), ("HAS_SE", 0.5),
    ("IS_BYTE", -1_000_000.0),
    (MARK_PC, -1e6), (MARK_AX, -1e6), (MARK_SP, -1e6),
    (MARK_BP, -1e6), (MARK_STACK0, -1e6),
)
psh_mem_addr0_restore = 10_000_000.0 / S     # +1e7/S write strength

restore_lo_8:
    conditions = psh_mem_addr0_conditions + ((OUTPUT_LO nibble 8, 1.0),)
    threshold  = 5.9
    writes     = (OUTPUT_LO nibble 8, +1e7/S), (OUTPUT_LO nibble 0, -1e7/S)
    scope="mark == MEM", dominates_at={"OUTPUT_LO": "mark == MEM"}

restore_hi_{k}:
    conditions = psh_mem_addr0_conditions + ((OUTPUT_HI_THIS_STEP+k, 1.0),)
    threshold  = 5.5
    writes     = (OUTPUT_HI_THIS_STEP+k, +1e7/S), (OUTPUT_HI_THIS_STEP+0, -1e7/S)
    scope="mark == MEM", dominates_at={"OUTPUT_HI_THIS_STEP": "mark == MEM"}
```

The `(OUTPUT_LO nibble 8, +1.0)` / `(OUTPUT_HI+k, +1.0)` term is
intended as a "restore only a byte the address decode already partially
produced" self-consistency gate. It is the amplitude-unbounded term.

---

## 2. Row-signature separation (probe results)

TO FILL: probe `tools/probe_e8_slam_rowsig.py` outputs.

---

## 3. Load-bearing memory function

TO FILL: which memory-smoke tests depend on the restore + kill-switch
measurement.

---

## 4. writer_index competition at the leaked OUTPUT dim

TO FILL.

---

## 5. Correct-by-construction Phase-2 design

TO FILL: the exact spec edit + discriminator + guardrail tests.
