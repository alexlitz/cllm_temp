"""c4_min RECURRENT (universal) interpreter: ONE step-block, applied
autoregressively, with per-step integer RE-QUANTIZATION of the VM state.

Motivation
----------
The straight-line / PC-dispatch compilers in ``compiler.py`` realise the honest
recurrence **depth = time**: they unroll ``max_steps`` physical step-blocks, one
transformer step-block per executed VM step, and flow the residual state through
depth. That is provably exact (``test_slice.py``) but it has two hard limits:

  * it bakes ``4 * max_steps`` physical blocks — for a deep loop that runs 1000
    steps that is 4000 blocks of weights, which is intractable, and
  * ``max_steps`` is a *compile-time* bound: a loop whose real step count exceeds
    the bake truncates (see the countdown demo — 16-block default vs 22 steps).

The 233 deep-loop programs (gcd, ``loop_*``, ``rec_*``) proved to have **no
loop-specific logic** — they are the SAME per-step VM transition repeated. So the
right architecture is a **recurrent universal step**: bake ONE step-block (the
four sub-blocks that make up a single VM step) and apply it REPEATEDLY in a
Python driver, carrying the full VM state (registers + stack + memory bands) from
iteration k to k+1, stopping at HALT/EXIT. The loop count is the program's real
step count — **unbounded**, decoupled from the bake.

Why it stays EXACT over arbitrarily many steps
----------------------------------------------
The one thing that makes the classic C4 neural VM exact over arbitrary steps is
that its state round-trips through *tokens* every step — a re-quantisation to
exact integers that stops fp error from accumulating. We reproduce that here
directly on the residual: after each step-block forward we **round every VM state
scalar to the nearest integer** (and the fold already reduced AX mod 256). Since
every band in this substrate holds an exact non-negative integer between steps
(register value, PC index, stack cell, one-hot 0/1), rounding is a no-op on the
true value but *annihilates* the O(1e-6) fp residue the SwiGLU gadgets leave — so
step k+1 sees a byte-exact integer state, and error can never compound. The
un-requantised depth-unroll drifts after ~O(150) chained blocks; requantised, it
is exact to whatever step budget you give the driver (demonstrated to 1000s).

State carried across iterations
-------------------------------
The full residual vector is carried, but only the *persistent* bands matter:
``AX, SP, BP, PC, STACK0`` (+ the ``STACK`` cell band and ``MEM`` cell band when a
program uses PSH-depth / memory) and the sticky ``HALTED`` flag. The scratch bands
(``PC_IS[*]``, ``AX_ZERO``) are recomputed by the fetch sub-block every step and
each carries a self-clear, so they need no special handling. ``ONE`` is a
constant lane (kept at 1.0). Per-step ``OUT``/``HALT_SEEN`` slots are unused here —
we read ``AX`` straight off the carried state each iteration.

Public API
----------
``build_step_model(code, ...)`` -> ``(model, L, code)`` where ``model`` has exactly
ONE step-block (4 physical sub-blocks). ``run_recurrent(model, L, code, ...)``
loops that step-block, requantising each iteration, and returns the per-step AX
trace (matching ``isa.interpret``). ``StepModel`` bundles the two for convenience.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from . import isa
from . import control
from .compiler import (VOCAB, HALT_TOKEN, _build_pc_layout, _zero_attn,
                       _load_ffn, _load_head, head_matrix)
from .compile_ffn import compile_ffn, compile_fold
from .dsl import FFNRule, LinearExpr
from .model import Transformer


def build_step_model(code: List[isa.Instr], n_heads: int = 4, max_pos: int = 4):
    """Bake ONE universal VM step-block for ``code`` (the PC-dispatch step).

    The step-block is the SAME four sub-blocks the PC-driven unroll uses per step:
      1. fetch    — PC scalar -> PC one-hot ``PC_IS[i]`` + ``AX_ZERO`` predicate.
      2. dispatch — rules gated on ``PC_IS[i]`` apply code[i]'s AX/STACK0 effect
                    AND the PC update (sequential +1 or branch target).
      3. fold     — AX mod-256 (a no-op unless AX overflowed a byte).
      4. emit     — copy post-step AX into ``L.OUT_SLOTS[0]`` (a single reusable
                    slot) and snapshot the sticky ``HALTED`` into ``HALT_SEEN[0]``.

    Only ONE OUT slot / HALT_SEEN slot is allocated (``max_steps=1``): the driver
    reuses it every iteration and reads it back, so the bake is O(n_code) blocks —
    independent of how many VM steps the program actually runs.

    Returns ``(model, layout, code)``. The model has exactly 4 physical blocks.
    """
    n_code = len(code)
    # max_steps=1 -> a single OUT/HALT_SEEN slot; the recurrent driver reuses it.
    L = _build_pc_layout(n_code, max_steps=1, n_heads=n_heads)
    dim = L.D

    fetch = control.compile_pc_fetch(L.PC, L.AX, L.PC_IS, L.AX_ZERO, L.ONE, dim)
    disp_rules = control.dispatch_rules(L, code, L.PC_IS)

    ffn_specs = [
        fetch,
        compile_ffn(disp_rules, dim),
        compile_fold(L.AX, L.ONE, dim, modulus=256),
        compile_ffn([
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]
    n_blocks = len(ffn_specs)                       # == 4
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0                 # initial state: ONE=1, rest 0
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L, code


# Bands that hold a persistent integer VM value between steps and must be
# re-quantised. Scratch bands (PC_IS, AX_ZERO, HALT_SEEN, OUT) are recomputed each
# step, so requantising them too is harmless — we requantise the WHOLE vector
# except the constant ONE lane, which keeps error from hiding anywhere.
def _requantize(state: torch.Tensor, one_band: int) -> torch.Tensor:
    """Round every band to the nearest integer (annihilating fp residue), then
    restore the constant ONE lane to exactly 1.0. Idempotent on integer state."""
    q = torch.round(state)
    q[one_band] = 1.0
    return q


def initial_state(model, L) -> torch.Tensor:
    """The baked initial residual (ONE=1, all registers/PC/stack = 0)."""
    return model.embed[0].clone()


def step_once(model, state: torch.Tensor) -> torch.Tensor:
    """Apply the single baked step-block to a [D] state vector, return new [D]."""
    x = state.view(1, 1, -1)
    for blk in model.blocks:
        x = blk(x)
    return x[0, 0]


def run_recurrent(model, L, code, max_steps: int = 4096,
                  requantize: bool = True, trace_state: bool = False):
    """Run the ONE step-block AUTOREGRESSIVELY until HALT (or ``max_steps``).

    Each iteration: forward the step-block, (optionally) re-quantise the carried
    state to exact integers, decode this step's AX via the LM head, and stop once
    the step executed HALT (``HALT_SEEN`` fired). Returns the per-step AX trace,
    matching ``isa.interpret``.

    ``requantize=False`` runs the identical loop WITHOUT the per-step integer
    rounding — used to demonstrate that fp error then accumulates and eventually
    corrupts the trace (the drift the re-quantisation cures).
    """
    import torch.nn.functional as F

    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    out_slot = L.OUT_SLOTS[0]
    halt_seen = L.HALT_SEEN[0]

    state = initial_state(model, L)
    trace: List[int] = []
    states = [] if trace_state else None
    for _ in range(max_steps):
        state = step_once(model, state)
        if requantize:
            state = _requantize(state, L.ONE)
        if states is not None:
            states.append(state.clone())
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if trace_state:
        return trace, states
    return trace


class StepModel:
    """Bundle: one baked step-block + its layout + code, with a ``run`` method."""

    def __init__(self, prog, n_heads: int = 4):
        self.code = isa.assemble(prog)
        self.model, self.L, _ = build_step_model(self.code, n_heads=n_heads)

    def run(self, max_steps: int = 4096, requantize: bool = True):
        return run_recurrent(self.model, self.L, self.code,
                             max_steps=max_steps, requantize=requantize)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)
