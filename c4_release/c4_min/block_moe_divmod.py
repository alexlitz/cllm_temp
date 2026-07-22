"""BLOCK-LEVEL identity/null expert over the DIV/MOD block span.

BLOG_SPEC §"Mixture of Experts Routing" (566-568):

    "Each opcode only needs a small subset of the FFN ops to complete so it
     very naturally lends itself to using Mixture of Experts (MoE) routing to
     avoid unnecessary computation and masking (to avoid interference)."

``moe_top1.py`` routes the hidden UNITS of ONE dispatch FFN.  This module is the
**block-level** dual: it routes the entire contiguous ``L._divmod_span`` block
range (the ~262 base-16 long-division blocks — 86% of the 305-block stack, the
dominant physical cost) to a **NO-OP identity expert** whenever the step's decoded
opcode is not DIV/MOD.

Why a WHOLE-block skip is byte-identical for a non-DIV/MOD step
==============================================================
Each block's FFN is a pure residual add ``x -> x + down(silu(up)·gate)`` and each
divmod block's attention is all-zero (identity).  The divmod blocks write ONLY
into the ALU-32 scratch bands (``a.QD``/``a.R``/``a.DIV_RES``/``a.MOD_RES`` ...),
which are:

  * re-cleared / re-derived from the operands every step (block-local scratch),
  * consumed ONLY by the OPCODE-GATED ``ax-mux`` block (which sits AFTER the span
    and writes ``AX_nib[c] += OP_IS[op]·RES[op][c]``) — so on a non-DIV/MOD step
    the ax-mux never reads the divmod RES bands.

Therefore running the divmod blocks on a non-DIV/MOD step only scribbles dead
scratch: dropping them leaves the AX (and PC/SP/BP) residual UNCHANGED.  This is
verified empirically by an L∞=0 byte-identity gate over the full op battery
(``test_block_moe_divmod.py``) — a skip that changes ANY output is a bug.

The router
==========
The active opcode is a one-hot ``OP_IS`` band the ``opcode-decode`` block writes
(greedy decode: exactly one lane high).  The router argmaxes the residual over the
``OP_IS`` one-hot window and asks "is the winner DIV or MOD?".  This is a pure
``Gather`` (route dims) + ``ArgMax`` (winner) — ONNX-vanilla, no ``If``/``Loop``/
``Scan``.  The block-skip itself is the top-1 route: expert 1 = the divmod stack,
expert 0 = identity; a non-DIV/MOD step routes to expert 0 and the divmod blocks'
COMPUTE is never touched (the real speedup), while the stored weights are
untouched (static graph, dynamic compute-skip).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import isa


# ---------------------------------------------------------------------------
# Locate the DIV/MOD span on a built model + layout.
# ---------------------------------------------------------------------------
def resolve_divmod_span(L) -> Tuple[int, int]:
    """Return ``(divmod_start, divmod_end)`` — the contiguous physical-block range
    of the DIV/MOD blocks (the block-MoE skip target, ``ax-mux`` sits at
    ``divmod_end``).

    Prefers the ``L._divmod_span`` SEAM recorded by
    ``build_pure_forward_complete_model``.  Falls back to deriving it from
    ``L._block_names`` (the contiguous ``alu-div*`` run) when the layout was built
    by the streaming/compact path (which throws away the seam-carrying ``L``).
    Both paths agree byte-for-byte on the current build (span = (31, 293)).
    """
    span = getattr(L, "_divmod_span", None)
    if span is not None:
        return int(span[0]), int(span[1])
    names = getattr(L, "_block_names", None)
    if not names:
        raise ValueError("layout carries neither _divmod_span nor _block_names")
    div_idx = [i for i, n in enumerate(names) if n.startswith("alu-div")]
    if not div_idx:
        raise ValueError("no alu-div* blocks found in _block_names")
    start, end = min(div_idx), max(div_idx) + 1
    # sanity: the span must be contiguous.
    if sorted(div_idx) != list(range(start, end)):
        raise ValueError("alu-div* blocks are not contiguous")
    return start, end


# ---------------------------------------------------------------------------
# Router: is the active opcode DIV or MOD?  (top-1 over the DIV/MOD guard lanes.)
# ---------------------------------------------------------------------------
class DivModRouter(torch.nn.Module):
    """Top-1 opcode router for the block-level DIV/MOD skip.

    ``forward(state) -> gate`` where ``state`` is the residual at ANY block whose
    input already carries the decoded ``OP_IS`` one-hot (the opcode-decode block
    runs at index 3, long before the divmod span at index 31).  ``gate`` is 1 when
    the step's op is DIV or MOD, else 0 — a pure ``Gather`` (the two DIV/MOD guard
    lanes) + threshold + OR (ONNX-vanilla, no ``If``/``Loop``/``Scan``).  The gate
    multiplies expert selection: gate=1 -> run the divmod stack, gate=0 -> identity
    (skip its compute).

    ROOT CAUSE this design fixes (the prior 0.99x / "run-all" bug)
    -------------------------------------------------------------
    A naive ``argmax`` over the FULL 40-wide ``OP_IS`` window does NOT identify the
    active op at the divmod-span entry: the residual there also carries the query
    row's ``HALT`` (38) and ``NOP`` (39) guard lanes pegged at 1.0 (frame/ingest
    aliasing), so the op band is a THREE-WAY tie {DIV, HALT, NOP} on a DIV step and
    ``argmax`` returns HALT — the router then reads "not divmod" and NEVER skips
    correctly OR always-skips (the prior no-skip / wrong-skip failure).  The fix is
    to read the DIV and MOD guard lanes DIRECTLY (they are 1.0 iff the step's op is
    that op, 0.0 otherwise — cleanly separated, no HALT/NOP contamination) and OR
    their above-threshold indicators.  Robust to the ingest aliasing; still a top-1
    route (expert 1 = divmod stack, expert 0 = identity).
    """

    def __init__(self, L, num_ops: Optional[int] = None, thresh: float = 0.5):
        super().__init__()
        self.op_is_base = int(L.OP_IS)
        self.num_ops = int(num_ops if num_ops is not None else isa.NUM_OPS)
        self.thresh = float(thresh)
        # route over EXACTLY the two DIV/MOD guard lanes (NOT the full OP_IS
        # window, which the ingest row pollutes with HALT/NOP one-hots).
        route_dims = torch.tensor(
            [self.op_is_base + isa.DIV, self.op_is_base + isa.MOD],
            dtype=torch.long)
        self.register_buffer("route_dims", route_dims)

    def active_op(self, state: torch.Tensor) -> torch.Tensor:
        """Best-effort active-op index over the FULL OP_IS window (diagnostic only;
        NOT used for routing — see the class docstring on the HALT/NOP tie)."""
        opis = state.index_select(
            -1, torch.arange(self.op_is_base, self.op_is_base + self.num_ops,
                             device=state.device))
        return opis.argmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return the DIV/MOD run-gate (1.0 run, 0.0 skip) for a residual row.

        ``gate = [OP_IS[DIV] > thr] OR [OP_IS[MOD] > thr]`` — a top-1 route read
        directly off the two guard lanes (immune to the HALT/NOP ingest aliasing).
        """
        guards = state.index_select(-1, self.route_dims)  # [..., 2] = (DIV, MOD)
        hot = guards > self.thresh                          # [..., 2]
        return hot.any(dim=-1).to(state.dtype)              # [...] 0/1


# ---------------------------------------------------------------------------
# The block-MoE forward: run the block stack, skipping the divmod span when the
# router gates it off.  COUNTS executed blocks (the block-count proof).
# ---------------------------------------------------------------------------
def moe_forward_with_count(model, L, x: torch.Tensor,
                           router: Optional[DivModRouter] = None,
                           divmod_span: Optional[Tuple[int, int]] = None
                           ) -> Tuple[torch.Tensor, int, bool]:
    """Apply ``model.blocks`` to residual ``x`` ([1, S, D]) with the block-level
    DIV/MOD skip, returning ``(x_out, n_blocks_executed, skipped)``.

    The router reads the residual AT the divmod-span entry (its input already
    carries the decoded ``OP_IS`` one-hot), decides skip/run, and either applies
    every block (divmod step) or every block EXCEPT the ``[start, end)`` span
    (non-divmod step).  ``n_blocks_executed`` is the HARD count used by the
    block-count assertion — it is ``len(blocks)`` on a divmod step and
    ``len(blocks) - (end - start)`` on a non-divmod step.  A router that fell back
    to running all blocks would report the full count and FAIL the test.
    """
    if router is None:
        router = DivModRouter(L)
    if divmod_span is None:
        divmod_span = resolve_divmod_span(L)
    start, end = divmod_span
    blocks = model.blocks
    n = len(blocks)
    executed = 0
    skipped = False
    bi = 0
    while bi < n:
        if bi == start:
            # ROUTER decision at the span entry (residual carries OP_IS one-hot).
            gate = float(router(x[0, -1]))       # 1.0 run divmod, 0.0 skip
            if gate < 0.5:
                # top-1 route to the IDENTITY expert: skip the whole span's
                # COMPUTE (the residual is unchanged across [start, end)).
                skipped = True
                bi = end
                continue
        x = blocks[bi](x)
        executed += 1
        bi += 1
    return x, executed, skipped


__all__ = [
    "resolve_divmod_span",
    "DivModRouter",
    "moe_forward_with_count",
]
