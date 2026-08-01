"""BATCHED per-SPAN block-MoE for the verify_blocks path (C4_BATCHED_BLOCK_SKIP).

The existing ``C4_STEP_BLOCK_SKIP`` (step_block_skip.py) is a PER-STEP block-MoE:
per opcode, run only that op's live blocks.  But the doom run uses the BATCHED
``verify_blocks`` path — one forward over a K-step SPAN with HETEROGENEOUS ops —
so a single forward cannot mask blocks per row.  The correct batched form is the
UNION of every op-in-the-span's live blocks: run exactly that union, skip the
complement (identity passthrough), byte-exact iff the union is a superset of every
op's true decode-live set.

Two coordinate spaces matter (recurrent-divmod build):
  * ``L._block_names`` (95) — the LOGICAL block sequence (one name per DISTINCT
    block object).  ``step_block_skip._LIVE_NAMES`` is keyed here.
  * ``model.blocks`` (242) — the APPLICATION sequence: the div-loop body is applied
    8x (recurrence), so a name maps to MANY application indices.
We build ``name -> distinct-object-index`` from the first occurrence of each block
object, then ``model.blocks[i]`` runs iff its distinct-object name is in the span's
live-name union.  DIV/MOD are OPERAND-DEPENDENT: their static live set is the WHOLE
contiguous divmod span (every recurrence iteration), matching step_block_skip.

Why this is the honest lever for doom: DIV appears every ~12 steps, so at K>=32
EVERY span contains a DIV -> the divmod span (168 of 242 blocks) can never be
skipped at big K.  What CAN be skipped in a big-K span is the blocks NO op in the
span touches — mostly the bitwise/shift/cmp/lea tail.  At small K the divmod span
itself becomes skippable on divmod-free spans, but K<32 explodes the forward count.
This module lets us MEASURE both regimes.

Gate: ``C4_BATCHED_BLOCK_SKIP`` (default OFF -> the full 242-block forward, byte-
identical, golden 069cc32f unchanged).  It changes NO stored weight — a runtime
COMPUTE skip only.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set

import torch

from . import isa
from .step_block_skip import _LIVE_NAMES, _SPAN_OPS, _SPAN_COMMON, \
    _KV_STACK_CHAIN, _POP_CONSUMER_OPS, kv_stack_enabled


def batched_block_skip_enabled() -> bool:
    return os.environ.get("C4_BATCHED_BLOCK_SKIP", "0") == "1"


class BatchedBlockSkipPlan:
    """Precomputed per-op live-mask over ``model.blocks`` (application coords), plus
    a per-span union-mask helper.  ``span_live_mask(ops)`` returns a bool list of
    length ``len(model.blocks)``: block i runs iff some op in ``ops`` is live at
    block i's distinct-object name."""

    def __init__(self, model, L):
        self.model = model
        self.blocks = model.blocks
        self.nb = len(model.blocks)
        names = list(getattr(L, "_block_names", []))
        if not names:
            raise RuntimeError("BatchedBlockSkipPlan: L._block_names missing")

        # distinct-object index (== name index): first occurrence of each block obj.
        first_seen: Dict[int, int] = {}
        distinct_of: List[int] = [0] * self.nb   # sparse idx -> distinct/name index
        for i, b in enumerate(self.blocks):
            oid = id(b)
            if oid not in first_seen:
                first_seen[oid] = len(first_seen)
            distinct_of[i] = first_seen[oid]
        n_distinct = len(first_seen)
        self._name_mismatch = (None if n_distinct == len(names)
                               else (n_distinct, len(names)))
        self.distinct_of = distinct_of
        self.names = names

        # name -> set of distinct-object indices carrying that name (should be 1:1)
        name_to_distinct: Dict[str, Set[int]] = {}
        for i, b in enumerate(self.blocks):
            d = distinct_of[i]
            nm = names[d] if d < len(names) else None
            if nm is not None:
                name_to_distinct.setdefault(nm, set()).add(d)
        self.name_to_distinct = name_to_distinct

        # divmod span in NAME coords -> the set of distinct indices in [start,end].
        div_idx = [i for i, n in enumerate(names) if n.startswith("alu-div")]
        self.div_distinct: Set[int] = (set(range(min(div_idx), max(div_idx) + 1))
                                       if div_idx else set())
        # MUL span: the CFM/addr32 build lowers a WIDER multiply than the code_size=32
        # build ``step_block_skip._LIVE_NAMES[MUL]`` was validated on — it adds the
        # Karatsuba blocks ``alu-mul-gp / alu-mul-ks0..ks2`` that a product >16 bits
        # needs (measured: a MUL of 123201 dropped bit-9 without them).  Like DIV/MOD,
        # run the WHOLE contiguous ``alu-mul-*`` span for MUL (operand-independent,
        # build-robust).  This is the batched-path's honest superset for MUL.
        mul_idx = [i for i, n in enumerate(names) if n.startswith("alu-mul")]
        self.mul_distinct: Set[int] = (set(range(min(mul_idx), max(mul_idx) + 1))
                                       if mul_idx else set())

        # per-op LIVE distinct-index set (resolve names -> distinct indices).
        kv_chain_names = list(_KV_STACK_CHAIN) if kv_stack_enabled() else []

        def names_to_distinct(nms: List[str]) -> Set[int]:
            s: Set[int] = set()
            for nm in nms:
                s |= self.name_to_distinct.get(nm, set())
            return s

        self._opstr_to_int = _build_opstr_to_int()
        kv_chain_d = names_to_distinct(kv_chain_names)
        self.op_live_distinct: Dict[Optional[int], Set[int]] = {}
        for op_int, nms in _LIVE_NAMES.items():
            live = names_to_distinct(nms)
            if op_int in _POP_CONSUMER_OPS:
                live |= kv_chain_d
            if op_int == isa.MUL:                      # full mul span (see mul_distinct)
                live |= self.mul_distinct
            self.op_live_distinct[op_int] = live
        for op_int in _SPAN_OPS:                       # DIV/MOD
            live = names_to_distinct(_SPAN_COMMON) | self.div_distinct
            if op_int in _POP_CONSUMER_OPS:
                live |= kv_chain_d
            self.op_live_distinct[op_int] = live

        # precompute per-op application-coords mask for quick union.
        self._op_app_mask: Dict[Optional[int], torch.Tensor] = {}
        for op_int, dset in self.op_live_distinct.items():
            m = [distinct_of[i] in dset for i in range(self.nb)]
            self._op_app_mask[op_int] = torch.tensor(m, dtype=torch.bool)

        self._full_mask = torch.ones(self.nb, dtype=torch.bool)

    def op_live_count(self, op_str: str) -> int:
        op_int = self._opstr_to_int.get(op_str)
        m = self._op_app_mask.get(op_int)
        return int(m.sum()) if m is not None else self.nb

    def span_live_mask(self, op_strs) -> torch.Tensor:
        """Union of live application-coords masks over the ops in the span.  Any op
        not in the table forces the full mask (safe)."""
        acc = torch.zeros(self.nb, dtype=torch.bool)
        for os_ in op_strs:
            op_int = self._opstr_to_int.get(os_)
            m = self._op_app_mask.get(op_int)
            if m is None:
                return self._full_mask.clone()         # unknown op -> run everything
            acc |= m
        return acc


def _build_opstr_to_int() -> Dict[str, int]:
    """Map the draft frame op STRING (isa opcode name) to the isa.<OP> int used as
    the ``_LIVE_NAMES`` key."""
    m: Dict[str, int] = {}
    for nm in dir(isa):
        if nm.isupper() and not nm.startswith("_"):
            v = getattr(isa, nm)
            if isinstance(v, int):
                m[nm] = v
    return m


__all__ = ["batched_block_skip_enabled", "BatchedBlockSkipPlan"]
