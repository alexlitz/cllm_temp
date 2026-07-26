"""VANILLA IN-STEP TOMBSTONE — the blog-faithful ``free()`` as a MODEL zero-write
distributed across the free/return step's SPARE token positions (``C4_VANILLA_TOMBSTONE``).

The distinction from ``C4_EXACT_EVICT``
======================================
``C4_EXACT_EVICT`` (``nibble_evict_schedule``) is a DRAFT-DRIVEN CACHE-MANAGER
reclaim: it reads the perfect draft's read-log and DROPS dead KV rows in O(steps)
with ZERO model cost — but that is a fast-path cache manager operation, NOT a
vanilla model op.  The blog spec's ``free()`` is a MODEL OP (§Memory 689-691): the
model WRITES ZEROS to the freed address (a real vanilla store of 0 → the softmax1
CAM returns 0 = ZFOD tombstone), and the cache manager then VALIDLY removes the
zeroed row.  THIS module implements that vanilla path — the FIDELITY path.

The two are REDUNDANT on the fast path (both end with the freed address reading 0
and its KV row reclaimed).  ``exact-evict`` costs 0 model steps but is a manager
reclaim; the vanilla tombstone costs TOKEN SLOTS (it emits real zero-writes) but is
a genuine model operation you could run with no cache manager at all.

The mechanism — one step, N zero-writes
=======================================
Each VM step emits a 30-token register frame.  The §Memory CAM head keys ONLY on
the store overlay (``IS_STORE`` / ``ADDR_BIN`` / ``VAL_NIB``), which is INDEPENDENT
of the token id at a position — so a store KV row can ride ANY token position, not
just the one MEM token.  Freeing an N-slot frame therefore = laying N zero-value
store rows on N SPARE positions of the ONE free/return step's frame (0 extra STEPS;
N real zero-writes).

Which positions are SPARE (byte-exactness)
------------------------------------------
A store row sets ``IS_FRAME_BYTE=0`` at its position, dropping that token from the
next step's register INGEST (the ingest penalty sinks any non-frame-byte row).  So:

  * the 5 register/step MARKER slots (REG_PC=0, REG_AX=5, REG_SP=10, REG_BP=15,
    STEP_END=29) are NEVER ingested (they carry no ROLE / IS_FRAME_BYTE) → ALWAYS
    spare.  (Slot 20 = MEM is the frame's own primary store token, reserved.)
  * a ROLE-BYTE slot (the 20 register byte tokens + STACK0's 4) is spare ONLY when
    its register byte is UNCHANGED this step: dropping it from THIS frame's ingest
    is harmless because the SAME role from the PRIOR frame (identical byte) still
    wins the ingest recency, so the next step reconstructs the byte-identical value.
    On a LEV/return the register frame is mostly RESTORED (AX + STACK0 unchanged, so
    ~8 role bytes + the 5 markers ≈ 13 spare slots), so most tokens are slack.

Both facts are proven byte-exact through the real ``model.forward`` (probe +
``test_vanilla_tombstone``): the freed addresses read 0, the decoded VM trace is
identical to the pre-free path, and the cache manager reclaims the zeroed rows
(compose with ``nibble_evict_schedule``).

Gate: ``C4_VANILLA_TOMBSTONE`` (default OFF ⇒ golden ``8f4dd780`` unchanged; the
overlay's ``frame_spare_stores`` arg defaults to ``None`` = the stock path).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import c4_min.blogspec_vocab as V


# ---------------------------------------------------------------------------
# Frame slot geometry (mirrors nibble_pure_forward._FRAME_ROLE_SLOTS / markers).
# ---------------------------------------------------------------------------
# The 30-token frame:
#   0 REG_PC | 1..4 pc bytes | 5 REG_AX | 6..9 ax | 10 REG_SP | 11..14 sp
#   | 15 REG_BP | 16..19 bp | 20 MEM | 21..24 addr | 25..28 val(STACK0) | 29 STEP_END
MARKER_SLOTS = (0, 5, 10, 15, 29)          # never ingested; slot 20 (MEM) reserved
# per-register the 4 role-byte frame slots (little-endian byte 0..3).
_REG_BYTE_SLOTS = {
    "PC": (1, 2, 3, 4),
    "AX": (6, 7, 8, 9),
    "SP": (11, 12, 13, 14),
    "BP": (16, 17, 18, 19),
    "STACK0": (25, 26, 27, 28),
}


def vanilla_tombstone_enabled() -> bool:
    """``C4_VANILLA_TOMBSTONE`` (default OFF): emit blog-faithful vanilla zero-write
    tombstones on the free/return step's spare token positions.  OFF ⇒ the driver
    never populates ``frame_spare_stores`` ⇒ byte-identical to golden ``8f4dd780``."""
    return os.environ.get("C4_VANILLA_TOMBSTONE", "0") \
        not in ("0", "", "false", "False")


def spare_slots_for_step(changed_regs: Optional[set] = None,
                         reserve_mem: bool = True) -> List[int]:
    """The frame-local token positions that are SPARE this step (safe to carry a
    tombstone store row without corrupting the next step's ingest).

    ``changed_regs`` — the set of register NAMES whose value CHANGED this step
      ("PC"/"AX"/"SP"/"BP"/"STACK0").  A register NOT in this set is unchanged, so
      ALL 4 of its role-byte slots are spare (ingest falls back to the prior frame's
      identical byte).  When ``None`` (unknown) only the marker slots are returned
      (the always-safe minimum).
    ``reserve_mem`` — keep slot 20 (the frame's primary MEM store token) reserved.

    Returns the spare local positions in a STABLE order (markers first, then
    unchanged-register role bytes) so a caller fills the cheapest slots first."""
    slots: List[int] = [s for s in MARKER_SLOTS]     # 5 always-spare markers
    if not reserve_mem:
        slots.append(20)
    if changed_regs is not None:
        changed = set(changed_regs)
        for reg, byte_slots in _REG_BYTE_SLOTS.items():
            if reg not in changed:
                slots.extend(byte_slots)
    return slots


def distribute_tombstones(free_addrs: List[int],
                          changed_regs: Optional[set] = None,
                          reserve_mem: bool = True
                          ) -> Tuple[Dict[int, Tuple[int, int]], List[int]]:
    """Assign the N free addresses to this step's spare token positions.

    Returns ``(spare_stores, spill)`` where ``spare_stores`` maps ``local_pos ->
    (addr, 0)`` (the tombstone zero-writes that FIT this step's slack) and ``spill``
    is the list of free addresses that did NOT fit (the caller must tombstone these
    on the NEXT step(s), each a real vanilla zero-write — reported honestly, no
    silent drop).  0 extra STEPS when ``spill`` is empty."""
    slots = spare_slots_for_step(changed_regs, reserve_mem=reserve_mem)
    spare_stores: Dict[int, Tuple[int, int]] = {}
    spill: List[int] = []
    for i, addr in enumerate(free_addrs):
        if i < len(slots):
            spare_stores[slots[i]] = (int(addr) & 0xFFFFFFFF, 0)
        else:
            spill.append(int(addr) & 0xFFFFFFFF)
    return spare_stores, spill


def frame_tombstone_capacity(changed_regs: Optional[set] = None,
                             reserve_mem: bool = True) -> int:
    """How many tombstones ONE step can carry given its slack (the spare-slot
    budget).  Marker-only (``changed_regs=None``) = 5; a LEV/return step that leaves
    AX + STACK0 unchanged = 5 + 8 = 13."""
    return len(spare_slots_for_step(changed_regs, reserve_mem=reserve_mem))
