"""Spec-faithful Mixture-of-Experts routing on the NIBBLE foundation.

BLOG_SPEC §"Mixture of Experts Routing" (566-568) + §"Vanillaness" (§352-399,
the ``StandardMoEFFN`` reference class verbatim).

    "Each opcode only needs a small subset of the FFN ops to complete so it
     very naturally lends itself to using Mixture of Experts (MoE) routing to
     avoid unnecessary computation and masking (to avoid interference). ...
     This also means that we can reuse the same dimensions of the residual
     streams between different opcodes/experts."

What this module is
-------------------
A drop-in replacement for the skeleton's **single dense dispatch FFN**
(``nibble_vm.base_dispatch_rules`` compiled by ``compile_ffn`` into ONE
SwiGLU block). Instead of one FFN whose hidden units are all
``OP_IS[op]``-guarded and coexist in the same block, we split the dispatch
**per opcode**: one ``PureFFN`` expert per opcode, each holding ONLY that
opcode's transition rules, and blend them by the decoded opcode one-hot with
the spec's soft-MoE residual formula:

    forward(x) = x + Σ_i  opcode_weight[expert_op_i] · (expert_i(x) − x)

where ``opcode_weight`` is read straight out of the residual's ``OP_IS`` band
(the decoded opcode one-hot the skeleton's ``compile_opcode_decode`` produces).
This is *exactly* the reference ``StandardMoEFFN.forward`` — the only
adaptation is that ``E.OP_START:E.OP_START+E.NUM_OPS`` is the nibble VM's
``L.OP_IS`` band (``isa.NUM_OPS`` wide).

Why it's byte-exact vs the dense FFN
------------------------------------
Every rule in ``base_dispatch_rules`` is already gated on its opcode's
one-hot ``OP_IS[op]`` inside the SwiGLU (``W_up[u,OP_IS+op]=S``,
``b_up[u]=-S·(n_win-0.5)``): when ``op`` is inactive the guard drives the unit
into ``silu(-0.5S)≈0``, so its down-contribution is ~0 regardless. Therefore:

  * **sum of the per-op experts == the dense FFN, exactly.** Splitting the
    rule list into per-op FFNs and summing ``(E_i(x)-x)`` is an algebraic
    identity — the hidden units are simply partitioned across modules; the
    residual write is linear.  (proven: 0.0 max-abs-diff, ``test_nibble_moe``.)

  * **one-hot-blended MoE == dense, up to fp noise.** With ``w_i = OP_IS[op_i]``
    (exactly 1 for the active op, 0 for all others), the active expert
    contributes its full ``(E-x)`` and every inactive expert is annihilated by
    ``w_i=0``. This removes even the inactive experts' O(1e-14) guard residue,
    so the blend is *cleaner* than the dense FFN, not dirtier. The remaining
    ~1e-13 is fp associativity and is crushed by the vanilla requant argmax.

Why route at all, if it's byte-identical? — exactly the spec's rationale:
"avoid unnecessary computation and masking (to avoid interference)". The MoE
gives a **clean logical separation** — each opcode's gadget is an independent
module — and, crucially, lets experts **reuse the same residual dims**: IMM,
LEA, ADD, SUB all *write ``AX_VAL``*; PSH/ADD/SUB all *write ``SP_VAL``*; every
op writes ``PC_VAL``. In the dense FFN those coexist only because the guards
keep them from firing together. In the MoE they are literally the same
destination dims in different experts, and the one-hot blend guarantees only
the active op's write lands — two experts writing the same dim never interfere.
(proven: ``test_dim_reuse_no_interference``.)

Purity / ONNX (spec §357-361, §383-398)
---------------------------------------
``forward`` is pure tensor ops: the ``for i in range(num_experts)`` loop
unrolls statically at trace time (``num_experts`` and each ``opcode_idx`` are
Python ints), there is NO ``.item()`` / data-dependent Python branch, and the
routing signal is a fixed residual slice — so the whole thing is a single ONNX
graph. All experts run every step (soft MoE); the blend, not a scatter/gather,
selects the winner.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import isa
from .blogspec_model import FFN
from .dsl import FFNRule
from .nibble_vm_layout import NibbleVMLayout
from .nibble_vm import base_dispatch_rules, compile_ffn


# ---------------------------------------------------------------------------
# Build a PureFFN (blogspec_model.FFN) expert from a compiled SwiGLU spec.
# ---------------------------------------------------------------------------
def _ffn_from_spec(spec: Dict[str, torch.Tensor], dim: int) -> FFN:
    """Load a compiled SwiGLU spec (from ``compile_ffn``) into a PureFFN expert.

    ``blogspec_model.FFN`` is the spec's ``PureFFN`` (SwiGLU + additive residual,
    ``x + down(silu(up)·gate)``). The expert is a *self-contained* FFN so it can
    run stand-alone in the MoE (the ``x +`` residual is part of the expert; the
    MoE blends the residual delta ``E(x)-x``).
    """
    hidden = spec["W_up"].shape[0]
    f = FFN(dim, hidden)
    with torch.no_grad():
        f.W_up.copy_(spec["W_up"])
        f.b_up.copy_(spec["b_up"])
        f.W_gate.copy_(spec["W_gate"])
        f.b_gate.copy_(spec["b_gate"])
        f.W_down.copy_(spec["W_down"])
        f.b_down.copy_(spec["b_down"])
    return f


def experts_from_rules(rules: Sequence[FFNRule], L: NibbleVMLayout,
                       dim: int) -> tuple[List[FFN], List[int]]:
    """Partition dispatch rules **by opcode** into one PureFFN expert per opcode.

    Each rule in ``base_dispatch_rules`` is guarded by exactly one opcode's
    one-hot window ``(L.OP_IS + op, 0.5, 1.5)`` (its first guard). We group all
    rules sharing an opcode into one expert FFN (there is one rule per op in the
    base table, but grouping is general — a gadget may append several rules for
    the same op). Returns ``(experts, expert_opcodes)`` where ``expert_opcodes[i]``
    is the opcode INDEX (0..NUM_OPS-1) that keys expert ``i`` — i.e. the offset
    into the ``OP_IS`` routing band.
    """
    by_op: Dict[int, List[FFNRule]] = {}
    for r in rules:
        if not r.when:
            raise ValueError("dispatch rule has no opcode guard; cannot route it")
        # the opcode this rule keys on = its first guard band minus OP_IS base.
        op = r.when[0][0] - L.OP_IS
        if not (0 <= op < isa.NUM_OPS):
            raise ValueError(f"rule guard band {r.when[0][0]} is not an OP_IS lane")
        by_op.setdefault(op, []).append(r)

    experts: List[FFN] = []
    expert_opcodes: List[int] = []
    for op in sorted(by_op):
        spec = compile_ffn(by_op[op], dim)
        experts.append(_ffn_from_spec(spec, dim))
        expert_opcodes.append(op)
    return experts, expert_opcodes


# ===========================================================================
# The spec's StandardMoEFFN — soft MoE, all experts run, blended by op one-hot.
# ===========================================================================
class NibbleStandardMoEFFN(nn.Module):
    """Soft Mixture-of-Experts FFN (BLOG_SPEC §352-399 ``StandardMoEFFN``).

    All experts run in parallel; outputs are blended by the opcode one-hot read
    from the residual ``OP_IS`` band. NO Python control flow in ``forward`` —
    pure tensor ops only.

        forward(x) = x + Σ_i  opcode_weight[expert_op_i] · (expert_i(x) − x)

    This is a verbatim port of the reference class; the only substitution is the
    routing slice: the reference reads ``x[:, 0, E.OP_START:E.OP_START+E.NUM_OPS]``;
    here that band is ``L.OP_IS`` (width ``isa.NUM_OPS``), the decoded opcode
    one-hot the skeleton's ``compile_opcode_decode`` writes.
    """

    def __init__(self, experts: List[FFN], expert_opcodes: List[int],
                 op_start: int, num_ops: int):
        """
        Args
        ----
        experts        : list of PureFFN (``blogspec_model.FFN``) expert modules.
        expert_opcodes : opcode INDEX (offset into the routing band) per expert.
        op_start       : residual dim where the opcode one-hot band begins
                         (the skeleton's ``L.OP_IS``).
        num_ops        : width of the opcode one-hot band (``isa.NUM_OPS``).
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        # Store opcodes as a Python list (NOT a tensor) so the forward loop
        # unrolls statically at ONNX trace time (spec §371-372).
        self.expert_opcode_list = list(expert_opcodes)
        # Also as a buffer, for state_dict round-trips (spec §373-374).
        self.register_buffer(
            "expert_opcodes", torch.tensor(expert_opcodes, dtype=torch.long))
        self.num_experts = len(experts)
        self.op_start = int(op_start)
        self.num_ops = int(num_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-tensor soft-MoE forward (spec §377-398).

        All experts run; each output is weighted by its opcode's one-hot value
        and accumulated as a residual delta. ONNX compatible: opcode indices are
        Python ints (static at trace time); no ``.item()``, no data-dependent
        Python branch.
        """
        # Routing signal: the opcode one-hot band at the (single) VM-step
        # position. Shape [batch, NUM_OPS]. (The step-block residual is one
        # position wide; ``:, 0`` mirrors the reference which reads position 0.)
        opcode_weights = x[:, 0, self.op_start:self.op_start + self.num_ops]

        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)                 # [batch, pos, dim]
            opcode_idx = self.expert_opcode_list[i]         # Python int (static)
            weight = opcode_weights[:, opcode_idx:opcode_idx + 1].unsqueeze(-1)
            output = output + weight * (expert_out - x)     # weighted residual
        return x + output


# ===========================================================================
# TOP-1 hard-routed MoE — compute ONLY the argmax opcode's expert.
# ===========================================================================
class NibbleTop1MoEFFN(nn.Module):
    """Top-1 hard-routed form of :class:`NibbleStandardMoEFFN`.

    The soft MoE above runs ALL ``num_experts`` experts every step and blends by
    the opcode one-hot, so ~37/38 of the compute multiplies by ~0.  This class
    computes ONLY the active opcode's expert: it argmaxes the opcode one-hot to
    the active opcode, ``Gather``s that opcode's expert's (padded) weight slabs,
    and runs one SwiGLU.  The output is argmax-identical to the soft blend (the
    inactive experts' ``weight_i=0`` contribution is dropped).

    Vanilla / ONNX: ``ArgMax`` (route) + ``Gather`` (select the expert's weights)
    + the SwiGLU matmuls.  No Python ``for``/``if`` over experts in ``forward``,
    no ``.item()``.  Experts are padded to a common hidden width ``Hmax`` (pad
    rows are all-zero -> contribute 0); an ``op -> expert-slot`` table maps the
    argmax opcode to its expert slab (opcodes with no expert map to a dead
    all-zero slab, so a mis-decoded opcode is a safe no-op, matching the soft
    blend where its weight would be 0).

    Assumes GREEDY decode (the decoded opcode one-hot has one high lane).
    """

    def __init__(self, experts: List[FFN], expert_opcodes: List[int],
                 op_start: int, num_ops: int):
        super().__init__()
        self.op_start = int(op_start)
        self.num_ops = int(num_ops)
        self.num_experts = len(experts)
        D = experts[0].W_up.shape[1] if experts else 0
        self.dim = int(D)
        Hmax = max((e.W_up.shape[0] for e in experts), default=1)
        self.Hmax = int(Hmax)

        # Stack padded expert weights: [E+1, Hmax, D] (slot E = dead all-zero).
        E = self.num_experts
        Wup = torch.zeros(E + 1, Hmax, D)
        bup = torch.zeros(E + 1, Hmax)
        Wg = torch.zeros(E + 1, Hmax, D)
        bg = torch.zeros(E + 1, Hmax)
        Wd = torch.zeros(E + 1, D, Hmax)
        bd = torch.zeros(E + 1, D)
        with torch.no_grad():
            for i, e in enumerate(experts):
                h = e.W_up.shape[0]
                Wup[i, :h] = e.W_up.detach()
                bup[i, :h] = e.b_up.detach()
                Wg[i, :h] = e.W_gate.detach()
                bg[i, :h] = e.b_gate.detach()
                Wd[i, :, :h] = e.W_down.detach()
                bd[i] = e.b_down.detach()
        self.register_buffer("Wup", Wup)
        self.register_buffer("bup", bup)
        self.register_buffer("Wg", Wg)
        self.register_buffer("bg", bg)
        self.register_buffer("Wd", Wd)
        self.register_buffer("bd", bd)

        # opcode -> expert slot (default = dead slot E). The routing band is the
        # opcode one-hot; the argmax opcode indexes this table to its expert.
        op_to_slot = torch.full((num_ops,), E, dtype=torch.long)
        for i, op in enumerate(expert_opcodes):
            op_to_slot[int(op)] = i
        self.register_buffer("op_to_slot", op_to_slot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Top-1 routed MoE forward: run ONLY the argmax opcode's expert.

        ``x`` : ``[batch, pos, dim]``.  Returns ``x + (E_sel(x) - x)`` for the
        selected expert ``E_sel`` (the SwiGLU with additive residual), i.e. the
        soft blend restricted to the single active expert.  Router reads the
        opcode one-hot at position 0 (the VM-step residual is one position wide),
        matching the soft MoE's routing signal.
        """
        B, P, D = x.shape
        opcode_weights = x[:, 0, self.op_start:self.op_start + self.num_ops]
        active_op = opcode_weights.argmax(dim=-1)            # [B]
        slot = self.op_to_slot[active_op]                    # [B] expert slot

        # Gather the selected expert's padded weight slabs (per batch element).
        Wup = self.Wup.index_select(0, slot)                 # [B, Hmax, D]
        bup = self.bup.index_select(0, slot)                 # [B, Hmax]
        Wg = self.Wg.index_select(0, slot)                   # [B, Hmax, D]
        bg = self.bg.index_select(0, slot)                   # [B, Hmax]
        Wd = self.Wd.index_select(0, slot)                   # [B, D, Hmax]
        bd = self.bd.index_select(0, slot)                   # [B, D]

        up = torch.einsum("bhd,bpd->bph", Wup, x) + bup.unsqueeze(1)
        gate = torch.einsum("bhd,bpd->bph", Wg, x) + bg.unsqueeze(1)
        hidden = F.silu(up) * gate                           # [B, P, Hmax]
        delta = torch.einsum("bdh,bph->bpd", Wd, hidden) + bd.unsqueeze(1)
        return x + delta


# ---------------------------------------------------------------------------
# Convenience: build the MoE dispatch straight from the skeleton's layout.
# ---------------------------------------------------------------------------
def build_moe_dispatch(L: NibbleVMLayout,
                       rules: Sequence[FFNRule] | None = None,
                       top1: bool = False):
    """Build the ``StandardMoEFFN`` that REPLACES the skeleton's dense dispatch
    FFN (``compile_ffn(base_dispatch_rules(L))``, the block-5 SwiGLU).

    ``rules`` defaults to ``base_dispatch_rules(L)``; pass an extended rule list
    (base + a plugged-in gadget's ``OP_IS[op]``-gated rules) to route new opcodes
    the same way — the MoE partitions by opcode automatically. The routing band
    is ``L.OP_IS`` (the decoded opcode one-hot), so the MoE reuses the skeleton's
    existing fetch/decode pipeline unchanged.

    ``top1=True`` returns the :class:`NibbleTop1MoEFFN` (hard-routed: only the
    argmax opcode's expert runs) instead of the soft-blend
    :class:`NibbleStandardMoEFFN`; the two are argmax-identical.
    """
    if rules is None:
        rules = base_dispatch_rules(L)
    experts, expert_opcodes = experts_from_rules(rules, L, L.D)
    cls = NibbleTop1MoEFFN if top1 else NibbleStandardMoEFFN
    return cls(experts, expert_opcodes, op_start=L.OP_IS, num_ops=isa.NUM_OPS)


__all__ = [
    "NibbleStandardMoEFFN",
    "NibbleTop1MoEFFN",
    "experts_from_rules",
    "build_moe_dispatch",
]
