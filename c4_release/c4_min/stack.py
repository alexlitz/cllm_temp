"""c4_min SP-INDEXED STACK substrate: an arbitrary-depth stack addressed by SP.

Boundary this replaces
----------------------
The bring-up substrate carried a *single* ``STACK0`` scalar band as the top-of-
stack mirror. That is only correct at stack depth <= 1: a second ``PSH`` clobbers
the one mirror cell, so a depth-2 program (``PSH a; PSH b; ... ADD; ADD``) pops
``b`` twice and never sees ``a`` (verified: ``10 20 30 ADD ADD`` gave 70 not 60).
Nested expressions, function arguments and recursion all need depth > 1, so this
single-cell mirror blocks the whole func/rec cluster.

The mechanism
-------------
We model the stack as an **array of memory cells** ``STACK[0..K-1]`` living in the
residual (one scalar band per cell), addressed by the ``SP`` register used as a
*depth counter* (``SP == number of items on the stack``; ``SP=0`` is empty). This
is the exact analogue of the data-memory the LI/SI ops use, indexed by SP instead
of an arbitrary address:

    PSH:   STACK[SP] = AX ;  SP += 1
    pop:   value = STACK[SP-1] ;  SP -= 1          (used by ADD/SUB/... )

The address match is the SAME content/addr-match gadget the design uses for
pop/memory-load: "select the cell whose index equals SP". At the single recurrent
position (the driver forwards ``seq_len == 1``) cross-position attention cannot
address sibling cells, so the addr-match is realised with the exact-integer
FFN one-hot the PC dispatch already uses — an ``SP``-equality triangular pulse
``SP_IS[i] = (SP == i)`` — which is the residual-band form of the content match
(``q = SP``, ``k = cell_index``, ``v = STACK[i]``). It is exact 0/1 on integers.

Two gadgets, both additive FFN sub-blocks (union-merge friendly — they only
append hidden units and never reorder existing ones):

  * ``compile_sp_fetch`` — materialises, from the scalar ``SP``:
      - ``SP_IS[i] = (SP == i)``        (which cell PSH will write / SP sits at)
      - ``STACK0   = STACK[SP-1]``      (the current top-of-stack mirror, read by
        the ALU pop of ADD/SUB) via ``STACK0 = Σ_i (SP-1 == i) * STACK[i]``.
    Both are recomputed every step and self-clear, so they carry no stale state.

  * ``sp_stack_writes`` — the per-cell ``FFNRule`` writes for a ``PSH`` at code
    index ``i``: for every candidate depth ``d`` it writes ``STACK[d] += AX``
    gated on ``PC_IS[i] AND SP_IS[d]`` (fires for exactly one ``d`` == current SP).
    The dispatch adds ``SP += 1`` for PSH and ``SP -= 1`` for every pop op.

Exactness: every band holds a non-negative integer between steps, the guards are
one-hot 0/1, and the recurrent driver re-quantises to the nearest integer each
step, so the O(1e-6) SwiGLU residue never compounds — the stack is byte-exact to
any depth < K over arbitrarily many steps.
"""
from __future__ import annotations

from typing import List

import torch

from . import isa
from .dsl import FFNRule, LinearExpr


S = 60.0        # silu-identity scale (matches compile_ffn.S)
RELU_S = 200.0  # relu-via-silu scale (matches control.RELU_S)


def compile_sp_fetch(sp_band: int, stack_bands: List[int], sp_is_bands: List[int],
                     stack0_band: int, one_band: int, dim: int):
    """FFN that (re)writes the ``SP_IS`` one-hot addr-match selector.

    ``SP_IS[i] = tri_i(SP)`` where ``tri_i(x)=relu(x-(i-1))-2relu(x-i)+relu(x-(i+1))``
    is the exact unit triangular pulse (1 at ``x==i``, 0 at every other integer) —
    the same addr-match one-hot the PC fetch bakes, but keyed on ``SP``. This is
    the "select the cell whose index == SP" content match, in residual-band form.

    The top-of-stack mirror ``STACK0 = STACK[SP-1]`` is produced by the SEPARATE
    ``compile_stack0_select`` sub-block, which runs AFTER this one so it can read
    the freshly-written ``SP_IS`` selectors out of the residual (a single FFN's
    hidden units cannot read each other, so the two-band derivation is split
    across two additive sub-blocks).

    Each written band self-clears first (subtract its previous value) so the block
    is an idempotent SET across the depth-unrolled / recurrent steps.
    """
    K = len(stack_bands)
    assert len(sp_is_bands) == K, "one SP_IS band per stack cell"

    # relu thresholds for the SP pulses: need SP_IS[i] for i in [0..K-1], which
    # reads relu(SP-(i-1)), relu(SP-i), relu(SP-(i+1)) -> thresholds [-1 .. K].
    thresholds = list(range(-1, K + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_pc_relu = len(thresholds)

    clear_bands = list(sp_is_bands)
    clear0 = n_pc_relu
    n_units = clear0 + len(clear_bands)

    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    silu_on = float(torch.nn.functional.silu(torch.tensor(S)))  # ~= S

    # --- relu units: hidden_j = relu(SP - t_j) via silu(RELU_S*z)/RELU_S ---
    for t, j in thr_unit.items():
        W_up[j, sp_band] = RELU_S
        b_up[j] = -RELU_S * t
        W_gate[j, one_band] = 1.0

    # --- self-clear units for SP_IS[*] (subtract previous value) ---
    for c, band in enumerate(clear_bands):
        u = clear0 + c
        W_up[u, one_band] = S
        W_gate[u, band] = 1.0
        W_down[band, u] += -1.0 / silu_on

    # --- SP_IS[i] = relu(SP-(i-1)) - 2 relu(SP-i) + relu(SP-(i+1)) ---
    for i, band in enumerate(sp_is_bands):
        W_down[band, thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[band, thr_unit[i]] += -2.0 / RELU_S
        W_down[band, thr_unit[i + 1]] += 1.0 / RELU_S

    return {
        "W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
        "W_down": W_down, "b_down": b_down,
    }


def compile_stack0_select(stack_bands: List[int], sp_is_bands: List[int],
                          stack0_band: int, one_band: int, dim: int):
    """FFN materialising the top-of-stack mirror ``STACK0 = STACK[SP-1]``.

    ``STACK0 = Σ_i (SP-1 == i) * STACK[i]``. The selector ``(SP-1 == i)`` equals
    ``SP_IS[i+1]`` — the one-hot bit for depth ``i+1`` — which the preceding
    ``compile_sp_fetch`` sub-block has already written into the residual. So each
    cell ``i`` contributes one hidden unit: ``gate = STACK[i]``,
    ``up = S * SP_IS[i+1]`` routes ``STACK[i]`` into ``STACK0`` iff ``SP-1 == i``.
    When ``SP == 0`` (empty stack) no selector bit is set and ``STACK0`` stays 0.

    ``STACK0`` self-clears first so this is an idempotent SET each step.
    """
    K = len(stack_bands)
    assert len(sp_is_bands) == K, "one SP_IS band per stack cell"

    n_units = 1 + K                            # 1 self-clear + K per-cell selectors
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    silu_on = float(torch.nn.functional.silu(torch.tensor(S)))  # ~= S

    # self-clear STACK0 (subtract previous value)
    W_up[0, one_band] = S
    W_gate[0, stack0_band] = 1.0
    W_down[stack0_band, 0] += -1.0 / silu_on

    # STACK0 += Σ_i SP_IS[i+1] * STACK[i]. Selector for cell i is depth (i+1);
    # the top cell (i == K-1) would need SP_IS[K] which does not exist (depth K is
    # a full stack, guarded against), so it has no selector — never queried.
    for i in range(K):
        sel_depth = i + 1
        if sel_depth >= K:
            continue                           # no SP_IS band for a full stack
        u = 1 + i
        W_up[u, sp_is_bands[sel_depth]] = S    # up = S when SP_IS[i+1] == 1
        b_up[u] = -0.5 * S                     # silu(0.5S) on / silu(-0.5S) off
        W_gate[u, stack_bands[i]] = 1.0        # gate = STACK[i]
        silu_half = float(torch.nn.functional.silu(torch.tensor(0.5 * S)))
        W_down[stack0_band, u] += 1.0 / silu_half

    return {
        "W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
        "W_down": W_down, "b_down": b_down,
    }


def sp_stack_writes(pc_is_band: int, ax_band: int, stack_bands: List[int],
                    sp_is_bands: List[int]) -> List[FFNRule]:
    """Per-cell PSH writes: ``STACK[d] = AX`` gated on ``PC_IS[i] AND SP_IS[d]``.

    Exactly one depth ``d == SP`` fires (the new top). The write is a true SET —
    ``STACK[d] += AX - STACK[d]`` — so it overwrites any *stale* value left in that
    cell by an earlier push that was later popped (a plain additive write would
    accumulate ``old + AX``). Returns one rule per candidate depth ``d``.
    """
    K = len(stack_bands)
    rules: List[FFNRule] = []
    for d in range(K):
        G = [(pc_is_band, 0.5, 1.5), (sp_is_bands[d], 0.5, 1.5)]
        rules.append(FFNRule(
            G, {stack_bands[d]: LinearExpr.of(ax_band, 1.0)
                + LinearExpr.of(stack_bands[d], -1.0)}))
    return rules
