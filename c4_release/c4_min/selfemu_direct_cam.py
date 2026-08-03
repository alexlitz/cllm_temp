"""SELF-EMU O(1) direct-CAM memory reads on the BOUNDED-KV pos-sparse path.

Generalises DOOM's ``direct_cam_batched`` O(1) address-CAM to the SELF-EMULATED
c4 VM's bounded/K-batch runner (``pos_sparse_bounded.BoundedBlock`` /
``pf_kbatch.KBatchBoundedBlock``).

THE PROBLEM this fixes
----------------------
#746 bounded-KV collapsed the K/V-over-S floor for the LOCAL (ingest) heads, but the
GLOBAL address-CAM heads (mem LI/LC, stack pop, LEV ret-PC) STILL do a ``softmax1``
over EVERY ``IS_STORE`` row ``<= q`` (``BoundedBlock._store_rows`` /
``KBatchBoundedBlock._store_layout``).  On a self-emulation the emulated VM writes
memory heavily, so ``n_store`` GROWS ~O(S) — that growing softmax is the residual
O(S) attention, and it is ~all wasted: the read resolves to EXACTLY ONE store row
(the address-CAM softmax1 winner), the other n_store-1 rows carry ~0 softmax weight.

THE FIX (O(1))
--------------
The perfect draft already knows, per read, the EXACT store row the softmax1+ALiBi
winner is (``nibble_evict_schedule.resolve_load_rows``: address -> latest-write-wins
store, == the softmax1-CAM+ALiBi winner; a never-written address -> the +1 sink ->
ZFOD 0).  So on the verify path the global CAM heads need not score the store set at
all: each read DIRECT-GATHERS the resolved value's V vector.

WHY BYTE-EXACT (identical argument to ``direct_cam_batched``)
------------------------------------------------------------
For a global CAM head the softmax1 output at a query row is EXACTLY the winner store
row's V vector (weight ~=1 on the exact-address latest-write winner, ~=0 elsewhere,
+1 sink -> 0 for an unwritten address).  ``_bake_cam_head`` makes the head's ``W_v``
read ONLY ``VAL_NIB[j]`` into slots ``ADDR_BITS+3+j``, and ``W_o`` maps those into
the destination band (mem->AX, pop->STACK0, lev->LEV_RET).  So the head's context
vector is a KNOWN function of the resolved value: nibbles(value) at slots
``ADDR_BITS+3+j``, zero elsewhere.  We reconstruct that exact context and SET the
global heads' context slice to it — so ``x + W_o(ctx)`` is byte-identical to the
softmax path, and the value bands drive the WHOLE downstream FFN transition
(BZ/BNZ read AX, LEV reads LEV_RET, ALU pops read STACK0), untouched.

Because a query row's global-CAM output is a pure function of the resolved value,
and the resolver is exact (latest-write-wins == softmax1+ALiBi winner), the direct
gather == softmax over the store rows, bit-for-bit (both ZFOD to 0 on an unwritten
address).

Gate: ``C4_SELFEMU_DIRECT_CAM`` default OFF -> the vanilla store-row softmax path
(byte-exact golden 069cc32f unchanged).  Fast-path only (a gather is not softmax),
same status as ``C4_DIRECT_CAM_BATCHED`` / ``C4_DIRECT_CAM_READ``.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch

from . import blogspec_vocab as V
from .blogspec_layout import NIB_PER_REG
from .blogspec_memory import ADDR_BITS, MEM_ALIBI_SLOPE


def selfemu_direct_cam_enabled() -> bool:
    """``C4_SELFEMU_DIRECT_CAM`` (DEFAULT OFF): O(1) direct-index CAM gather for the
    self-emu bounded/K-batch global heads.  OFF -> the vanilla softmax1+ALiBi over the
    store-row set (byte-exact golden path)."""
    return os.environ.get("C4_SELFEMU_DIRECT_CAM", "0") not in ("0", "", "false", "False")


# ===========================================================================
# The per-query-position resolved global-CAM values, keyed by (position, kind).
# ===========================================================================
class DirectCamTable:
    """Resolved global-CAM values per ABSOLUTE query-row position, per head kind.

    ``val[kind][pos]`` -> the resolved value (int) the ``kind`` head reads at the step
    whose query row is at absolute stream position ``pos``.  Absent -> that head is a
    pure softmax1 sink at that step (ZFOD 0 output).  ``kind`` is "mem"/"pop"/"lev".
    """

    __slots__ = ("mem", "pop", "lev")

    def __init__(self):
        self.mem: Dict[int, int] = {}
        self.pop: Dict[int, int] = {}
        self.lev: Dict[int, int] = {}

    def by_kind(self, kind: str) -> Dict[int, int]:
        return getattr(self, kind)


def build_direct_cam_table(resolved_by_qpos: Dict[int, list]) -> DirectCamTable:
    """From ``{query_position: [ResolvedRead, ...]}`` (as ``bench_pf_kbatch._prep_stream``
    already builds via ``resolve_load_rows``) build the per-(kind, pos) value table."""
    tbl = DirectCamTable()
    for pos, rlist in resolved_by_qpos.items():
        for r in rlist:
            d = tbl.by_kind(r.head) if r.head in ("mem", "pop", "lev") else None
            if d is not None:
                d[int(pos)] = int(r.value) & 0xFFFFFFFF
    return tbl


# ===========================================================================
# The per-block global-head kind map: which of THIS block's global heads is
# mem / pop / lev, so the direct gather pulls the right resolved value.
# ===========================================================================
def block_global_kinds(block_idx: int, L, block_names: List[str],
                       global_idx: torch.Tensor) -> Dict[int, str]:
    """Return ``{head_idx: kind}`` for the global CAM heads on ``block_idx``.

    Mirrors ``bake_global_cam_heads`` (the sole authority): in the golden 3-head build
    the mem-cam block carries head ``N_ROLES`` (kind "mem"); the stack-pop-cam block
    carries head ``N_ROLES+1`` (kind "pop") and ``N_ROLES+2`` (kind "lev").  We map by
    (block name, head index) so a layout shift is caught (an unmapped global head keeps
    the softmax path -> still byte-exact, just not the fast gather).
    """
    from .nibble_pure_forward import N_ROLES
    name = block_names[block_idx] if block_idx < len(block_names) else ""
    kinds: Dict[int, str] = {}
    gset = set(int(h) for h in global_idx.tolist())
    if name == "mem-cam" and (N_ROLES in gset):
        kinds[N_ROLES] = "mem"
    if name == "stack-pop-cam":
        if (N_ROLES + 1) in gset:
            kinds[N_ROLES + 1] = "pop"
        if (N_ROLES + 2) in gset:
            kinds[N_ROLES + 2] = "lev"
    return kinds


# ===========================================================================
# The direct-CAM head-context vector: the exact softmax1 winner's V (the head's
# value-slot space, HD-wide).
# ===========================================================================
def head_ctx_vec(value: int, HD: int, device, dtype) -> torch.Tensor:
    """The global CAM head's attention context vector for the resolved ``value``.

    ``_bake_cam_head``: ``W_v[base+ADDR_BITS+3+j] <- VAL_NIB[j]``, so the winner V
    (softmax1 weight ~=1) is the value's nibbles at head slots ``ADDR_BITS+3+j``.  A
    ZFOD read (value == 0 / unwritten) is the all-zero vector — the softmax1 +1 sink
    output — so value 0 correctly gives 0 here too."""
    out = torch.zeros(HD, device=device, dtype=dtype)
    b0 = ADDR_BITS + 3
    for j, nv in enumerate(V.nibbles_of_value(int(value) & 0xFFFFFFFF, NIB_PER_REG)):
        out[b0 + j] = float(nv)
    return out


def _values_tensor(kind: str, tbl: DirectCamTable, q_idxs: List[int],
                   device) -> torch.Tensor:
    """The ``[Q]`` int64 tensor of resolved values (0 for an unread/ZFOD row) for
    ``kind`` at the query positions ``q_idxs``."""
    d = tbl.by_kind(kind)
    vals = [int(d.get(int(ap), 0)) & 0xFFFFFFFF for ap in q_idxs]
    return torch.tensor(vals, dtype=torch.long, device=device)


def gather_global_ctx(kinds: Dict[int, str], tbl: DirectCamTable,
                      q_idxs: List[int], global_idx: torch.Tensor,
                      HD: int, device, dtype) -> torch.Tensor:
    """Build the ``[Q, nG, HD]`` context for the global heads of a block by DIRECT
    GATHER of the resolved value at each query position — NO softmax over store rows.

    ``global_idx`` is the block's global head-index tensor (the order the caller writes
    ``ctx_full[:, global_idx]`` in).  For head ``h`` at global-slot ``gi``, if ``h`` is
    a mapped CAM kind we place its resolved value's nibble V vector at slots
    ``ADDR_BITS+3 .. +NIB_PER_REG-1``; an unmapped global head (shouldn't happen in the
    golden build) gets 0 (a pure sink — safe).

    VECTORIZED: the nibble decode is a per-kind ``[Q, NIB_PER_REG]`` tensor op (shift +
    mask), scattered into the context — NO Python per-nibble / per-query loop (that
    fixed-overhead loop made the gather SLOWER than the softmax at small n_store)."""
    Q = len(q_idxs)
    nG = int(global_idx.numel())
    ctx = torch.zeros(Q, nG, HD, device=device, dtype=dtype)
    b0 = ADDR_BITS + 3
    # shift amounts for the 16 4-bit nibbles (little-endian, matching nibbles_of_value).
    shifts = torch.arange(NIB_PER_REG, device=device, dtype=torch.long) * 4  # [N]
    for gi, h in enumerate(global_idx.tolist()):
        kind = kinds.get(int(h))
        if kind is None:
            continue
        vals = _values_tensor(kind, tbl, q_idxs, device)                    # [Q]
        # nib[q, j] = (vals[q] >> (4*j)) & 0xF
        nib = (vals.view(Q, 1) >> shifts.view(1, NIB_PER_REG)) & 0xF        # [Q,N]
        ctx[:, gi, b0:b0 + NIB_PER_REG] = nib.to(dtype)
    return ctx


__all__ = [
    "selfemu_direct_cam_enabled",
    "DirectCamTable",
    "build_direct_cam_table",
    "block_global_kinds",
    "head_ctx_vec",
    "gather_global_ctx",
]
