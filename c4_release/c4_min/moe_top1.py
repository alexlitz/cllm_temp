"""TOP-1-ROUTED opcode dispatch — the structurally-sparse MoE the spec wants.

BLOG_SPEC §"Mixture of Experts Routing" (566-568):

    "Each opcode only needs a small subset of the FFN ops to complete so it
     very naturally lends itself to using Mixture of Experts (MoE) routing to
     avoid unnecessary computation and masking (to avoid interference)."

The c4_min dispatch block (``nibble_pure_forward_complete`` / the compact model)
is a **single dense SwiGLU FFN** whose every hidden unit is guarded by exactly
one opcode's one-hot ``OP_IS[op]`` (``W_up[u, OP_IS+op] = S``, ``b_up[u] =
-S·0.5``). When ``op`` is INACTIVE that unit's pre-activation is ``silu(-0.5·S)
≈ -2.8e-12`` and its residual contribution is ~0; when ACTIVE it is
``silu(+0.5·S) = 0.5·S`` and the down-projection (``1/silu(0.5S)``) recovers the
write expression exactly.  So the dense FFN is *already* a soft MoE — but it
pays the full FLOPs of ALL ~31-38 opcodes' units every step and then blends by
the one-hot.  ~37/38 of that compute multiplies by (a rounding-error away from)
zero.

This module turns that dense soft-MoE into a **top-1 hard-routed** MoE:

  * a router argmaxes the decoded ``OP_IS`` one-hot -> the ONE active opcode
    (greedy decode: exactly one lane is high);
  * a static ``[NUM_OPS, K]`` unit table (``op_units``, K = max units any single
    opcode owns) says which hidden-unit rows belong to each opcode;
  * a single ``Gather`` selects the active opcode's <=K unit rows of
    ``W_up / W_gate / b_up / b_gate`` and the matching columns of ``W_down``;
  * the SwiGLU runs on JUST those K units.

The output is argmax-identical (L-inf ~1e-13) to the dense blend: the selected
units' arithmetic is bit-for-bit the dense FFN's (same weight rows), and the
only thing dropped is the ~1e-13 ``silu(-0.5S)`` residue of the ~37 skipped
inactive opcodes' units -- which is exactly the "unnecessary computation" the
spec says the MoE should avoid, and is crushed by the vanilla requant argmax.

Vanilla / ONNX (spec §357-398)
------------------------------
``forward`` is pure tensor ops: ``ArgMax`` (router) + ``Gather`` (unit select) +
the standard SwiGLU matmuls.  NO Python ``if``/``for`` over opcodes, NO
``.item()``, NO data-dependent Python branch -- so it traces to ONE ONNX graph
with only ``ArgMax`` / ``Gather`` / ``MatMul`` / ``Add`` / ``Mul`` / ``Sigmoid``
(SiLU) ops, none of them the forbidden ``If`` / ``Loop`` / ``Scan``.  Padding
slots in ``op_units`` point at a dedicated all-zero dead unit so a short opcode's
K-wide gather contributes exactly 0 for its unused slots.

Composition with tensor sparsity (``sparse_forward.sparse_mm``)
---------------------------------------------------------------
Orthogonal and multiplicative.  ``sparse_mm`` exploits WEIGHT sparsity (the ~180k
nonzeros in a 7.5B-param dense model) so each mat-mul does work proportional to
its nonzeros.  Top-1 routing exploits ACTIVATION/EXPERT sparsity (only 1 of ~38
experts is live) so the dispatch block only *touches* the active expert's rows.
Together: the dispatch block does ``nnz(active-expert-only)`` work instead of
``nnz(all-experts)`` — the routing shrinks the row set, sparse_mm shrinks the
per-row cost.  The routed expert's small dense gather is already tiny, so its
sub-block is best kept dense; the rest of the stack keeps ``sparse_mm``.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Router: decode the per-unit -> opcode map from a compiled dispatch SwiGLU.
# ---------------------------------------------------------------------------
def unit_opcodes_from_ffn(W_up: torch.Tensor, op_is_base: int, num_ops: int,
                          guard_thresh: float = 1e-3) -> torch.Tensor:
    """Recover, per hidden unit, the opcode ``OP_IS`` lane that guards it.

    ``compile_ffn`` writes exactly ONE ``W_up[u, OP_IS+op] = S`` guard per
    dispatch unit (the opcode one-hot window).  This reads that guard back:
    ``unit_op[u] = argmax_op |W_up[u, OP_IS+op]|`` (the single nonzero lane), or
    ``-1`` if the unit has no ``OP_IS`` guard (an ungated pass-through — always
    run, never routed away).

    Returns a ``[hidden]`` long tensor of opcode indices (or -1 for ungated).
    """
    band = W_up[:, op_is_base:op_is_base + num_ops].abs()   # [hidden, num_ops]
    has_guard = band.max(dim=1).values > guard_thresh        # [hidden]
    unit_op = band.argmax(dim=1)                             # [hidden]
    unit_op = torch.where(has_guard, unit_op,
                          torch.full_like(unit_op, -1))
    return unit_op


def route_dims_from_ffn(W_up: torch.Tensor, op_is_base: int, num_ops: int,
                        guard_thresh: float = 1e-3
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the EXACT residual dims to route over + the opcode each keys.

    The naive ``x[..., OP_IS:OP_IS+num_ops]`` slice is WRONG on the *compact*
    (dim-shared) model: the compaction packs other, unrelated bands (``BP_LOW``,
    ``AXB_LO``, ...) into that dim window, and one of them can carry a value
    (e.g. 240) that beats the true opcode one-hot (~1) in an argmax.  Those alias
    bands are simply *never live at the same block* as ``OP_IS`` — the dispatch
    FFN itself only ever reads the genuine opcode-guard dims — so we route over
    exactly the dims the FFN's ``W_up`` guards use.

    Returns ``(route_dims[R], route_op[R])``: ``route_dims`` = the sorted set of
    genuine opcode-guard dims (one per opcode that owns >=1 unit), and
    ``route_op[i]`` = the opcode index (``0..num_ops-1``) that ``route_dims[i]``
    keys.  The router argmaxes ``x[..., route_dims]`` and maps the winning index
    through ``route_op`` to the active opcode.
    """
    unit_op = unit_opcodes_from_ffn(W_up, op_is_base, num_ops, guard_thresh)
    dim_to_op: Dict[int, int] = {}
    H = W_up.shape[0]
    for u in range(H):
        op = int(unit_op[u])
        if op < 0:
            continue
        d = op_is_base + op                       # the guard column = OP_IS + op
        dim_to_op[d] = op
    dims = sorted(dim_to_op)
    route_dims = torch.tensor(dims, dtype=torch.long)
    route_op = torch.tensor([dim_to_op[d] for d in dims], dtype=torch.long)
    return route_dims, route_op


def build_op_unit_table(unit_op: torch.Tensor, num_ops: int
                        ) -> tuple[torch.Tensor, int, List[int]]:
    """Build the static ``[num_ops, K]`` opcode -> unit-rows gather table.

    ``K`` = the max number of units any single opcode owns.  Row ``op`` lists the
    hidden-unit indices guarded by ``OP_IS[op]``, right-padded with a **dead unit
    index** (``num_units`` — a sentinel row we append below as an all-zero unit)
    so every row is exactly ``K`` wide (a rectangular Gather, ONNX-friendly).

    Any ungated units (``unit_op == -1``) are NOT in this table; the caller runs
    them unconditionally (they always fire regardless of opcode).

    Returns ``(op_units[num_ops, K], K, ungated_unit_indices)``.
    """
    num_units = int(unit_op.shape[0])
    dead = num_units                                     # sentinel: the appended zero unit
    per_op: List[List[int]] = [[] for _ in range(num_ops)]
    ungated: List[int] = []
    for u in range(num_units):
        op = int(unit_op[u])
        if op < 0:
            ungated.append(u)
        else:
            per_op[op].append(u)
    K = max((len(us) for us in per_op), default=1)
    K = max(K, 1)
    table = torch.full((num_ops, K), dead, dtype=torch.long)
    for op in range(num_ops):
        for j, u in enumerate(per_op[op]):
            table[op, j] = u
    return table, K, ungated


# ===========================================================================
# Top-1 routed SwiGLU dispatch FFN (drop-in for blogspec_model.FFN).
# ===========================================================================
class Top1RoutedFFN(nn.Module):
    """A single dense SwiGLU dispatch FFN, re-expressed as a TOP-1-routed MoE.

    Drop-in replacement for the ``blogspec_model.FFN`` sitting in the dispatch
    block: identical ``forward(x) -> x + down(silu(up)·gate)`` contract, identical
    weights, but only the ACTIVE opcode's hidden units are computed (plus any
    ungated units).  Built from a compiled dispatch FFN (``compile_ffn`` output)
    whose units are each ``OP_IS[op]``-guarded.

    The routing is a per-row (per position) ArgMax over the ``OP_IS`` band + a
    Gather of that opcode's unit rows.  Pure tensor ops -> ONNX-vanilla.

    NOTE: assumes GREEDY decode — the decoded ``OP_IS`` one-hot has exactly one
    high lane, so ``argmax`` selects the true active opcode.  (This is the same
    assumption the dense blend makes: it multiplies by the one-hot, which is
    ~1 for the active op and ~0 else.)
    """

    def __init__(self, ffn: nn.Module, op_is_base: int, num_ops: int):
        super().__init__()
        with torch.no_grad():
            W_up = ffn.W_up.detach().clone()
            b_up = ffn.b_up.detach().clone()
            W_gate = ffn.W_gate.detach().clone()
            b_gate = ffn.b_gate.detach().clone()
            W_down = ffn.W_down.detach().clone()
            b_down = ffn.b_down.detach().clone()

        H, D = W_up.shape
        self.dim = int(D)
        self.op_is_base = int(op_is_base)
        self.num_ops = int(num_ops)

        unit_op = unit_opcodes_from_ffn(W_up, self.op_is_base, self.num_ops)
        table, K, ungated = build_op_unit_table(unit_op, self.num_ops)
        self.K = int(K)
        self.n_units = int(H)
        self.n_ungated = len(ungated)

        # Genuine opcode-guard dims to route over (NOT the naive OP_IS window,
        # which the compact/dim-shared layout pollutes with alias bands).
        route_dims, route_op = route_dims_from_ffn(
            W_up, self.op_is_base, self.num_ops)
        self.n_route = int(route_dims.numel())

        # Append ONE all-zero "dead" unit at index H so padded gather slots
        # contribute exactly 0 (silu(0)*0 = 0, down col all-zero).
        Wup_pad = torch.cat([W_up, torch.zeros(1, D)], dim=0)         # [H+1, D]
        bup_pad = torch.cat([b_up, torch.zeros(1)], dim=0)           # [H+1]
        Wg_pad = torch.cat([W_gate, torch.zeros(1, D)], dim=0)       # [H+1, D]
        bg_pad = torch.cat([b_gate, torch.zeros(1)], dim=0)         # [H+1]
        Wd_pad = torch.cat([W_down, torch.zeros(D, 1)], dim=1)       # [D, H+1]

        # Register the padded weights + routing table as buffers (params-free;
        # the wrapper carries no NEW learnable params — same weights, re-indexed).
        self.register_buffer("W_up", Wup_pad)
        self.register_buffer("b_up", bup_pad)
        self.register_buffer("W_gate", Wg_pad)
        self.register_buffer("b_gate", bg_pad)
        self.register_buffer("W_down", Wd_pad)
        self.register_buffer("b_down", b_down)
        self.register_buffer("op_units", table)          # [num_ops, K] long
        self.register_buffer("route_dims", route_dims)   # [R] residual dims to argmax
        self.register_buffer("route_op", route_op)       # [R] opcode of each route dim
        # Ungated (always-run) unit indices, if any. Kept as a buffer so they
        # gather statically at trace time (empty tensor if none).
        self.register_buffer(
            "ungated_units",
            torch.tensor(ungated, dtype=torch.long) if ungated
            else torch.zeros(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Top-1 routed SwiGLU: run ONLY the active opcode's (+ ungated) units.

        ``x`` : ``[batch, pos, dim]``.  Returns ``x + down(silu(up)·gate)`` with
        the down-projection restricted to the routed units.  Pure tensor ops:
        Gather (route dims) + ArgMax (route) + Gather (select rows/cols) + the
        SwiGLU matmuls.
        """
        B, P, D = x.shape
        # -- Router: argmax the opcode one-hot over the GENUINE guard dims. -----
        # (index_select the route dims, argmax, map winner -> opcode.  This is
        # robust to the compact layout's dim-sharing: the alias bands packed into
        # the OP_IS window are NOT in ``route_dims``.)
        opis = x.index_select(-1, self.route_dims)                   # [B,P,R]
        win = opis.argmax(dim=-1)                                    # [B,P] -> route idx
        active_op = self.route_op[win]                               # [B,P] -> opcode

        # -- Gather the active opcode's unit indices (+ ungated). --------------
        sel = self.op_units[active_op]                               # [B,P,K]
        if self.ungated_units.numel() > 0:
            ung = self.ungated_units.view(1, 1, -1).expand(B, P, -1)
            sel = torch.cat([sel, ung], dim=-1)                     # [B,P,K+U]
        Ksel = sel.shape[-1]

        # -- Gather the selected units' weight rows / bias / down cols. ---------
        # W_up[H+1, D] -> per-row selected [B,P,Ksel,D]; do it as a flat gather.
        flat_sel = sel.reshape(-1)                                   # [B*P*Ksel]
        Wup_s = self.W_up.index_select(0, flat_sel).view(B, P, Ksel, D)
        Wg_s = self.W_gate.index_select(0, flat_sel).view(B, P, Ksel, D)
        bup_s = self.b_up.index_select(0, flat_sel).view(B, P, Ksel)
        bg_s = self.b_gate.index_select(0, flat_sel).view(B, P, Ksel)
        # W_down[D, H+1] -> select the same columns -> [B,P,D,Ksel].
        Wd_s = (self.W_down.transpose(0, 1)                          # [H+1, D]
                .index_select(0, flat_sel)                          # [B*P*Ksel, D]
                .view(B, P, Ksel, D)
                .transpose(2, 3))                                   # [B,P,D,Ksel]

        # -- SwiGLU on the routed units only. ----------------------------------
        # up = sum_d Wup_s[...,k,d]*x[...,d] + b : [B,P,Ksel]
        up = torch.einsum("bpkd,bpd->bpk", Wup_s, x) + bup_s
        gate = torch.einsum("bpkd,bpd->bpk", Wg_s, x) + bg_s
        hidden = F.silu(up) * gate                                   # [B,P,Ksel]
        # down = sum_k Wd_s[...,d,k]*hidden[...,k] + b : [B,P,D]
        delta = torch.einsum("bpdk,bpk->bpd", Wd_s, hidden) + self.b_down
        return x + delta


# ---------------------------------------------------------------------------
# Convenience: wrap the dispatch block of a built model in top-1 routing.
# ---------------------------------------------------------------------------
def route_dispatch_block(model, L, block_name: str = "dispatch") -> "Top1RoutedFFN":
    """Replace ``model.blocks[<dispatch>].ffn`` (a dense SwiGLU FFN) with a
    ``Top1RoutedFFN`` and return it.  ``L._block_names`` must hold the block order
    (as ``build_pure_forward_complete_model`` / the compact builder set it).

    The OP_IS band + width are read from the layout (``L.OP_IS`` and the opcode
    one-hot width — the ``num_ops`` the decode fills).  The wrapped model is
    argmax-identical to the dense model under the driver.
    """
    from . import isa
    names = list(L._block_names)
    bi = names.index(block_name)
    routed = Top1RoutedFFN(model.blocks[bi].ffn, op_is_base=int(L.OP_IS),
                           num_ops=int(isa.NUM_OPS))
    model.blocks[bi].ffn = routed
    return routed


__all__ = [
    "Top1RoutedFFN",
    "unit_opcodes_from_ffn",
    "build_op_unit_table",
    "route_dispatch_block",
]
