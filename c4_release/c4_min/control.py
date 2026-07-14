"""c4_min control-flow substrate: PC-driven instruction dispatch (depth=time).

See ``CONTROL_FLOW.md`` for the mechanism. In one sentence: each unrolled
transformer block is a **universal VM step** — it reads the scalar ``PC`` band,
selects the instruction living at that PC from the baked code table (a PC-equality
one-hot), applies that instruction's effect (which may itself set ``PC``), and
emits the step's ``AX``. Because the executed instruction is chosen by the
*runtime* PC (not baked per position), a forward branch that skips instructions
Just Works: the block after the branch dispatches on the new PC.

This module is additive to the straight-line slice (``compiler._step_rules`` +
``compile_ffn``); it adds two things:

  1. ``compile_pc_fetch`` — an FFN that materialises, from the scalar ``PC``:
       * a **PC one-hot** ``PC_IS[i] == (PC == i)`` for each code index i, via an
         exact integer triangular-pulse (``relu`` gadget), and
       * ``AX_ZERO == (AX == 0)`` (the BZ/BNZ predicate), via ``relu(1 - AX)``.
     Both are exact 0/1 on integer inputs.

  2. ``dispatch_rules`` — the per-code-index ``FFNRule`` set that, gated on
     ``PC_IS[i]``, applies ``code[i]``'s AX/STACK0 effect **and** the PC update
     (sequential ``PC += 1``, or the branch's ``PC = target`` predicated on
     ``AX_ZERO`` for BZ/BNZ).

The AX/STACK0 effects reuse the exact-integer ``LinearExpr`` writes from the
straight-line slice — only the *guard* changes from the always-on ``ONE`` window
to the ``PC_IS[i]`` window. So there is **zero new arithmetic**; control flow is
purely a matter of *which* guard selects each op.
"""
from __future__ import annotations

from typing import List

import torch

from . import isa
from .dsl import FFNRule, LinearExpr


RELU_S = 200.0  # relu-via-silu scale: silu(RELU_S*z)/RELU_S ~= relu(z) (exact on ints)


S = 60.0  # silu-identity scale (matches compile_ffn.S): silu(60)~=60, silu(-60)~=0


def compile_pc_fetch(pc_band: int, ax_band: int, pc_is_bands: List[int],
                     ax_zero_band: int, one_band: int, dim: int):
    """FFN that (re)writes the PC one-hot ``PC_IS[i]`` + the ``AX_ZERO`` predicate.

    PC one-hot (exact on integers): ``PC_IS[i] = tri_i(PC)`` where
        ``tri_i(x) = relu(x-(i-1)) - 2*relu(x-i) + relu(x-(i+1))``
    is a unit triangular pulse — ``1`` at ``x==i`` and ``0`` at every other
    integer. We bake one shared relu unit per distinct threshold ``t`` in
    ``[-1 .. n_code]`` and route them with ``+1/-2/+1`` coefficients.

    ``AX_ZERO = relu(1 - AX)``: ``1`` iff ``AX == 0`` (AX is a non-negative 8-bit
    int, so ``relu(1-AX)`` is ``1`` at 0 and ``0`` for all AX>=1). One relu unit.

    The residual is **additive**, so before writing the fresh value each step we
    must subtract the band's *previous* value (which was set by the last step's
    fetch). Each target band gets a ``silu``-identity self-clear unit
    (``up=S, gate=old_band, down=-1/S`` → subtracts the old value) so the block is
    a true SET, idempotent across the depth-unrolled steps.
    """
    n = len(pc_is_bands)
    clear_bands = list(pc_is_bands) + [ax_zero_band]
    thresholds = list(range(-1, n + 1))          # relu thresholds for the pulses
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_pc_relu = len(thresholds)
    az_unit = n_pc_relu
    clear0 = az_unit + 1
    n_units = clear0 + len(clear_bands)          # relus + AX_ZERO + self-clears

    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    silu_on = float(torch.nn.functional.silu(torch.tensor(S)))  # ~= S

    # up_j = RELU_S*(PC - t_j); hidden_j = relu(PC - t_j)  (gate = 1)
    for t, j in thr_unit.items():
        W_up[j, pc_band] = RELU_S
        b_up[j] = -RELU_S * t
        W_gate[j, one_band] = 1.0
    # AX_ZERO relu: up = RELU_S*(1 - AX); hidden = relu(1 - AX)
    W_up[az_unit, ax_band] = -RELU_S
    b_up[az_unit] = RELU_S * 1.0
    W_gate[az_unit, one_band] = 1.0

    # self-clear units: hidden = silu(S)*old_band ; down routes -1/silu(S) -> -old_band
    for c, band in enumerate(clear_bands):
        u = clear0 + c
        W_up[u, one_band] = S
        W_gate[u, band] = 1.0
        W_down[band, u] += -1.0 / silu_on

    # PC_IS[i] = relu(PC-(i-1)) - 2 relu(PC-i) + relu(PC-(i+1)).
    # hidden_j == silu(RELU_S*(PC-t)) == RELU_S*relu(PC-t); recover relu via /RELU_S.
    for i, band in enumerate(pc_is_bands):
        W_down[band, thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[band, thr_unit[i]] += -2.0 / RELU_S
        W_down[band, thr_unit[i + 1]] += 1.0 / RELU_S
    # AX_ZERO = relu(1 - AX): same /RELU_S normalisation.
    W_down[ax_zero_band, az_unit] += 1.0 / RELU_S

    return {
        "W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
        "W_down": W_down, "b_down": b_down,
    }


def dispatch_rules(L, code: List[isa.Instr], pc_is_bands: List[int]) -> List[FFNRule]:
    """Per-code-index FFN rules: gated on ``PC_IS[i]``, apply ``code[i]``.

    Reuses the slice's exact-integer AX/STACK0 writes; the guard becomes the PC
    one-hot for index i. Adds the PC update (sequential or branch) as an extra
    write into the PC band. ``AX_ZERO`` (materialised by ``compile_pc_fetch``)
    predicates BZ/BNZ.
    """
    ax, stk, pc = L.AX, L.STACK0, L.PC
    rules: List[FFNRule] = []
    n = len(code)
    for i, ins in enumerate(code):
        G = [(pc_is_bands[i], 0.5, 1.5)]  # fires iff PC == i
        op = ins.op
        writes = {}

        # --- data effect (AX / STACK0), identical algebra to the slice ---
        if op == isa.IMM:
            writes[ax] = LinearExpr.c(float(ins.imm)) + LinearExpr.of(ax, -1.0)
        elif op == isa.LEA:
            writes[ax] = LinearExpr.of(L.BP, 1.0) + LinearExpr.c(float(ins.imm)) + LinearExpr.of(ax, -1.0)
        elif op == isa.PSH:
            writes[stk] = LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0)
        elif op == isa.ADD:
            writes[ax] = LinearExpr.of(stk, 1.0)
        elif op == isa.SUB:
            writes[ax] = LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0) + LinearExpr.c(256.0)
        elif op == isa.AND or op == isa.OR or op == isa.XOR:
            # bitwise handled by caller (needs the bit gadget); dispatch here only
            # covers the ops the slice + branches implement.
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in control dispatch")
        elif op in (isa.JMP, isa.BZ, isa.BNZ):
            pass  # control ops touch only PC (below)
        elif op == isa.HALT:
            writes[L.HALTED] = LinearExpr.c(1.0)
        else:
            raise NotImplementedError(f"op {isa.NAMES.get(op, op)} not in control dispatch")

        # --- PC update (every op) ---
        # sequential default: PC += 1 (i.e. PC becomes i+1, since PC==i here).
        if op == isa.JMP:
            # PC := imm  ==  PC += (imm - i)  (guarded, PC==i)
            writes[pc] = LinearExpr.c(float(ins.imm - i))
        elif op == isa.BZ:
            # taken (AX==0): PC := imm ; else PC += 1.
            #   delta = AX_ZERO*(imm - i) + (1-AX_ZERO)*1 = 1 + AX_ZERO*(imm - i - 1)
            writes[pc] = LinearExpr.c(1.0) + LinearExpr.of(L.AX_ZERO, float(ins.imm - i - 1))
        elif op == isa.BNZ:
            # taken (AX!=0): PC := imm ; else PC += 1.
            #   delta = (1-AX_ZERO)*(imm - i) + AX_ZERO*1 = (imm - i) + AX_ZERO*(1 - (imm - i))
            writes[pc] = LinearExpr.c(float(ins.imm - i)) + LinearExpr.of(L.AX_ZERO, float(1 - (ins.imm - i)))
        elif op == isa.HALT:
            # freeze PC on HALT (no advance) so a halted step re-dispatches HALT.
            writes[pc] = LinearExpr.c(0.0)
        else:
            writes[pc] = LinearExpr.c(1.0)  # PC += 1

        rules.append(FFNRule(G, writes))
    return rules
