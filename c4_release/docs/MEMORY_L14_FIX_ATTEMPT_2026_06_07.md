# L14 mem_generation fix attempt — negative result (2026-06-07)

## What was tried

Per `MEMORY_L13_L14_HANDOFF_2026_06_07.md` (`d3455b96`), replaced BP-relative K reads in `_layer14_mem_generation_head_specs` (`l14_ops.py:478-491`) with direct `STACK0_BYTE1/2/3` reads. Two variants tried:
1. Head h → STACK0_BYTE{h} (per brief)
2. Head h → STACK0_BYTE{h-1} (autoregressive shift per `vm_step.py:6794-6805` comments)

Neither moved the 5 memory tests from FAIL.

## Correction to prior doc

The prior doc claimed STACK0_BYTE1/2/3 are at compact slots 730/731/732. **Actual measurement**: they're at **legacy slots 508/509/510** (dynamic registry pins resolve to legacy because the compact mirror only activates under specific allocator config). Verified via `slots['STACK0_BYTE1'] = start=508`.

## Why the prescribed fix doesn't work

- L2 FFN populates `STACK0_BYTE1/2/3` at distance 7/8/9 from BP marker (= STACK0 byte 1/2/3 rows) per `setup_helpers_l2.py:71-109`.
- Legacy K reads (`L1H4+BP_I − H1+BP_I` etc.) selected d=5..6, 6..7, 7..8 from BP — in legacy STACK0-aliased-onto-BP layout that was STACK0 byte 0/1/2.
- These ROWS are the same in the new layout, so substitution should be byte-equivalent. It isn't.

## Deeper hypothesis

The bug is further upstream:
- **(a)** STACK0 byte values may not be arriving at STACK0 byte rows in `CLEAN_EMBED/OUTPUT`. The L8 `sp_gather_bake` may only set STACK0_BYTE0 (per comments in `setup_helpers_l2.py` about "the OUTPUT contains byte J value" being the original autoregressive trick).
- **(b)** The original BP-relative thresholds had additional implicit behavior — fired where OUTPUT/CLEAN_EMBED actually carried the next byte from L6 wide-ALU, not at the STACK0 byte position itself.

## Next investigation

Investigate whether L8 sp_gather writes STACK0 bytes 1/2/3 to the right rows. If not, that's the missing op — needs to be added. Pattern: extend `layer8_sp_gather_bake` to broadcast bytes 1/2/3 analogously to byte 0, OR add a sibling op for bytes 1/2/3.

## Symptoms vs. value

The user-facing failure is "memory returns address (0x200) instead of value (42)" — the LI operation returns the address itself. Could be that the address-keyed memory_lookup at L15 reads from positions that don't have the value bytes properly broadcast.

## Recommendation

Don't repeat the L14 head spec migration. Investigate L8 sp_gather output for bytes 1/2/3 first.
