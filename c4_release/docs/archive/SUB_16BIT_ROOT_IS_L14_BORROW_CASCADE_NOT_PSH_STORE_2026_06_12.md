# sub_16bit: PSH multi-byte store ALREADY WORKS — root is the L14 borrow cascade

**Date:** 2026-06-12 (spec_k=0, hook-free, GPU 0). **Status:** ROOT
RE-LOCALIZED. Corrects the premise of
`SUB_16BIT_ROOT_IS_PSH_STACK0_BYTE1_TRUNCATION_2026_06_11.md` and the brief
(`project_sub_16bit_root_is_psh_stack0_truncation`). NO code change landed
(correct zero-regression outcome; a working byte-1 staging head was built and
verified but the L14 borrow cascade coupling forces regressions that cannot
be gated away — the documented multi-session Wall-2 surface).

## TL;DR — the brief's premise is FALSE

The brief said *"the PSH store truncates high bytes; STACK0_BYTE1/2/3
(710/711/712) stay 0; extend the PSH store to populate them."* **The PSH store
already populates the minuend high bytes correctly.** Two corrections:

1. **STACK0_BYTE1/2/3 (710/711/712) are POSITION FLAGS, not value dims.** The
   dim registry (`dim_registry_dynamic.py:325-330`) names them "STACK0 byte N
   position flag" (`mark == STACK0 OR (is_byte AND byte_index == N)`). They
   mark *which sequence row* is byte N of the stack top; they never carried the
   value. They ARE correctly set at the stack-byte rows for every push (probe
   `tools/probe_psh_stack0_store.py`).

2. **The pushed value's high bytes ARE stored — in `STACK0_BYTE_VAL_h_LO/HI`
   (dims 734-829), not in the position flags.** The producer is
   `layer10_psh_ax_broadcast` (l10_ops.py:1905, heads 8/9/10, block 12 =
   logical L11). On OP_PSH it broadcasts AX byte h's `CLEAN_EMBED` →
   `STACK0_BYTE_VAL_h_LO/HI` at the STACK0 byte-h rows. **Probe-verified
   (spec_k=0):** for `sub_16bit` (`IMM 0x100; PSH`), `STACK0_BYTE_VAL_1_LO` at
   the PSH frame's STACK0 byte-1 rows (66/101) = nibble 1 = **0x01**. For
   `sub_borrow_cascade` (`IMM 0; PSH`) it = 0x00. **The discriminator the prior
   diag said "does not exist anywhere" exists cleanly in the value-bus dim,
   before the SUB runs.**

## The real root: the L14 borrow cascade never reads STACK0_BYTE_VAL_1

The SUB-borrow loop (`CarryPropagationPostOp`, vm_step.py:855/975, attached as
L10 post_ops → expanded into physical blocks 16-25 / logical L14) reads the
minuend byte from `OUTPUT_LO/HI` at the byte-1 emit row. The L7/L10 stack0
relay only puts byte 0 into that OUTPUT band; byte 1 stays **0x00** for BOTH
`sub_16bit` and `sub_borrow_cascade`. So the loop computes
`(0x00 - borrow) & 0xFF = 0xFF` for both — they are byte-identical at every
block from the truncated OUTPUT on (matching the prior diag's "byte-identical"
observation, but the cause is the relay/OUTPUT band, NOT a missing store).

`STACK0_BYTE_VAL_1` (which HOLDS the 0x01 discriminator) is consumed today only
by `layer13_bitwise_byte1_gather` (l13_ops.py:428, L13 head 3), which stages it
into `AX_FULL` for **OP_OR/OP_XOR only** and explicitly excludes OP_SUB. There
is no SUB path from `STACK0_BYTE_VAL_1` to the borrow loop's input.

## What was tried (and why it's a multi-session arch block)

A new L13 attn head 4 (`layer13_sub_byte1_minuend_gather`) was built, modeled on
head 3: Q gated to the SUB byte-1 emit row (`TEMP+9` = SUB relay AND
`BYTE_INDEX_1`); K selects STACK0_BYTE1 rows with a large magnitude (so the
ALiBi distance bias can't pick a far non-STACK0 row); small negative ALiBi
tie-breaks to the original PSH frame (later carry rows decay to 0x00 —
probe-verified); V copies `STACK0_BYTE_VAL_1_LO/HI` → O writes `OUTPUT_LO/HI`.

**It WORKS as a stager:** verified spec_k=0, the head writes `OUTPUT` byte-1 =
**0x01** at block 14 for `sub_16bit` (and 0x00 for `sub_borrow_cascade`), gated
cleanly (dark on every non-SUB-byte-1 row; guardrails stayed 8/8).

**But the L14 borrow cascade breaks on the injected OUTPUT:**
- The byte-1 borrow stage (`CarryPropagationPostOp byte_idx=1, cascade=True`)
  fires only when the cascade carry `CARRY+3` reaches the byte-1 row. The byte-0
  stage sets `CARRY+3 = +172.9` at the byte-**0** row (156); a relay must shift
  it to the byte-1 row (157). **Writing OUTPUT at row 157 disrupts that
  CARRY+3 cross-row relay**, so the byte-1 stage never fires: the injected
  0x01 passes through unchanged (sub_16bit → `0x1xxxx`) AND
  `sub_borrow_cascade` REGRESSES (its byte-1/2/3 borrow continuation stops →
  `0xFFFF` instead of `0xFFFFFFFF`).
- Net: 33-34 pass / 7-8 fail (−1 to −2 vs the 35 baseline), guardrails intact.
  No gating of the head removes the cascade disruption because the cascade
  reads/relays the same OUTPUT band the staging must write.

This is the documented Wall-2 / borrow-cascade coupling
(`project_mul_div_mod_arch_blocked`, `project_l10_tail_bank_width_sensitive`,
`feedback_single_rule_fixes_are_zero_sum`). A correct fix must co-design the
borrow cascade to read the minuend from `STACK0_BYTE_VAL_1` (and keep the
`CARRY+3` cross-row relay intact) — a change to the imperative
`CarryPropagationPostOp` chain (vm_step.py), not a single declarative head.

## The real fix (multi-session)

Make the L14 `CarryPropagationPostOp` byte-1/2/3 stages take the minuend byte
from `STACK0_BYTE_VAL_h` instead of (or OR'd with) the truncated `OUTPUT` band,
WITHOUT perturbing the `CARRY+3` byte-to-byte borrow relay. The value is
already present and clean at the STACK0 byte-h rows; the cascade just needs to
gather it as its per-byte input. Equivalently, extend `AddSubBytePropagationPostOp`
(vm_step.py:1025, which "materializes the per-byte ADD/SUB base result before
the carry post-op") to source byte h from `STACK0_BYTE_VAL_h` — it is the
designed consumer for high-byte ADD/SUB bases and runs BEFORE the borrow loop,
so feeding it there avoids the cross-row CARRY relay collision.

## State left

- NO ops/weights changed. Tree clean except `tools/probe_psh_stack0_store.py`
  (new diagnostic, no bake side effects). Tail rule count 2059 (untouched). No
  new dims. Smoke reconfirmed **35 pass / 6 fail / 8-of-8 guardrails**;
  sub_16bit still FAIL (4294967295).
- Decisive probes (spec_k=0, `tools/probe_psh_stack0_store.py`):
  STACK0_BYTE_VAL_1_LO = 0x01 at the sub_16bit PSH frame (rows 66/101), 0x00
  for sub_borrow_cascade; OUTPUT byte-1 = 0x00 for BOTH at the SUB emit row;
  byte-1 = 0xFF written at block 17/18 (logical L14) by the borrow loop.
