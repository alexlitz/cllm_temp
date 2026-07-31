"""STACK-AWARE lean drivers — a real (depth-N) data stack for the compacted VM.

The compacted lean qwen VM (``qwen_full_vm`` / ``qwen_lean_forward``) mirrors the
data stack with a SINGLE ``STACK0`` register — ``PSH`` sets it, a pop-op reads it —
so the *stock* drivers (``run_program_lean`` / ``draft_program_lean``) only handle a
stack DEPTH of 1.  Depth-2+ pushes and stores of a COMPUTED value therefore diverge
(measured: ``10 + (3+4)`` decodes 10, not 17; ``mem[a] = x + y`` stores 0).

The subroutine MUL/DIV/MOD (:mod:`lean_subroutine_muldiv`) are byte-exact bytecode
but are inherently depth->1 / computed-store programs (a shift-add loop stores the
running product; long division stores the remainder), so they need a real stack.

This module supplies it **the same way the stock driver already handles JSR/ENT/LEV
functions** — with an explicit stack that lives ALONGSIDE the token stream, in the
driver, NOT in the model.  Every step the driver:

  * sets ``STACK0`` = the TRUE top of its Python data stack (so the model's ALU pops
    the correct operand — the model only ever needs the ONE top value per op),
  * on ``PSH`` pushes the real ``AX``; on a pop-op pops the real top;
  * on ``SI``/``SC`` pops the real ADDRESS (not the model's single-slot mirror).

The model still computes EVERY opcode (decode + ALU + control-flow) inside the
shallow ~15-layer forward; only the O(1)-per-step stack bookkeeping is driver-side —
exactly the "complexity is VM STEPS, not model depth" thesis.  Byte-exact vs
``isa.interpret`` for arbitrary-depth programs (verified in
``test_qwen_lean_stack_driver``).

Speculation (``speculative_run_lean_stack``) keeps working: the deterministic VM
drafts the whole trace (including the real stack) for free, and the lean forward
verifies ``block_steps`` steps per batched forward — each step's window carries its
own true ``STACK0``, so B steps verify in ONE forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from .nibble_pure_forward import SP_INIT
from .qwen_full_vm import _snap
from .qwen_lean_forward import (
    LeanQwenVM, CAM_REGS, _build_stream_and_overlay, _build_spec_batch,
    LeanSpecResult,
)

# pop-ops read the data-stack top (operand A / store address); SI/SC also store.
POP_OPS = (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
           isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
           isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
           isa.SI, isa.SC)


# ===========================================================================
# Pure-Python transition mirroring the model EXACTLY (the perfect draft) with a
# REAL data stack + a REAL byte-addressed memory.
# ===========================================================================
@dataclass
class StackDraft:
    steps: List[dict]              # per step: reg_state BEFORE (incl true STACK0), store_log, op
    ref_trace: List[int]
    halted: bool


def draft_program_lean_stack(lean: LeanQwenVM, code: List[isa.Instr],
                             max_steps: int = 200_000) -> StackDraft:
    """Draft the whole program with a REAL depth-N data stack + memory, recording per
    step the exact window the stack-aware driver feeds the model (register file with
    the TRUE top-of-stack in ``STACK0`` + the compacted store log).  Zero forwards.

    8-bit value path (AX & 0xFF), matching the lean model's fold.  Falls back
    (empty draft) on an out-of-slice op (JSR/ENT/LEV/syscalls)."""
    subset = lean.subset
    # full-length reference (isa.interpret defaults to max_steps=256, which TRUNCATES
    # a long subroutine trace and would spuriously fail the byte-identity compare).
    ref_trace = isa.interpret(code, max_steps=max_steps)

    pc = 0
    ax = 0
    sp = SP_INIT
    bp = SP_INIT
    stack: List[int] = []          # REAL data stack (values)
    mem: Dict[int, int] = {}
    store_log: List[dict] = []
    steps: List[dict] = []
    halted = False

    for _ in range(max_steps):
        if not (0 <= pc < len(code)):
            break
        op = code[pc].op
        imm = code[pc].imm
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = ax & 0xFF
        top = stack[-1] & 0xFF if stack else 0        # TRUE top of stack
        steps.append({
            "reg_state": {"PC": pc, "AX": ax & 0xFF, "SP": sp, "BP": bp, "STACK0": top},
            "store_log": [dict(s) for s in store_log],
            "load_addr": load_addr, "op": op, "pc_at": pc,
        })
        npc = pc + 1
        halted_step = False
        v = top
        addr = 0
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            stack.append(ax & 0xFF)
            sp -= 4
        elif op in POP_OPS:
            v = stack.pop() & 0xFF if stack else 0
            sp += 4
            if op == isa.ADD: ax = (v + ax) & 0xFF
            elif op == isa.SUB: ax = (v - ax) & 0xFF
            elif op == isa.MUL: ax = (v * ax) & 0xFF
            elif op == isa.DIV: ax = ((v // ax) if ax else 0) & 0xFF
            elif op == isa.MOD: ax = ((v % ax) if ax else 0) & 0xFF
            elif op == isa.AND: ax = v & ax
            elif op == isa.OR: ax = v | ax
            elif op == isa.XOR: ax = v ^ ax
            elif op == isa.SHL: ax = (v << ax) & 0xFF
            elif op == isa.SHR: ax = (v >> ax) & 0xFF
            elif op == isa.EQ: ax = 1 if v == ax else 0
            elif op == isa.NE: ax = 1 if v != ax else 0
            elif op == isa.LT: ax = 1 if v < ax else 0
            elif op == isa.GT: ax = 1 if v > ax else 0
            elif op == isa.LE: ax = 1 if v <= ax else 0
            elif op == isa.GE: ax = 1 if v >= ax else 0
            elif op in (isa.SI, isa.SC):
                addr = v & 0xFF
                mem[addr] = ax & 0xFF
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax & 0xFF, 0) & 0xFF
        elif op == isa.JMP:
            npc = imm
        elif op == isa.BZ:
            npc = imm if (ax & 0xFF) == 0 else npc
        elif op == isa.BNZ:
            npc = imm if (ax & 0xFF) != 0 else npc
        elif op == isa.HALT:
            halted_step = True
        elif op == isa.NOP:
            pass
        else:
            return StackDraft(steps=[], ref_trace=ref_trace, halted=False)

        if subset.memory and op in (isa.SI, isa.SC):
            store_log = [s for s in store_log if (s["addr"] & 0xFF) != (addr & 0xFF)]
            store_log.append({"addr": addr, "val": ax & 0xFF})
        pc = npc
        if halted_step or pc < 0 or pc >= len(code):
            halted = True
            break
    return StackDraft(steps=steps, ref_trace=ref_trace, halted=halted)


# ===========================================================================
# NAIVE stack-aware driver — one lean forward per VM step, real data stack.
# ===========================================================================
def run_program_lean_stack(lean: LeanQwenVM, code: List[isa.Instr],
                           max_steps: int = 200_000, forward=None,
                           verbose: bool = False) -> Dict[str, object]:
    """Execute ``code`` with a REAL depth-N data stack; one lean forward per step.

    ``forward``: optional callable ``(x, q_positions) -> hidden`` (e.g. a
    ``GraphedLeanForward``); defaults to ``lean.forward(..., past=None)``.  Returns
    ``{"ax_trace","ref_trace","exact","steps","forwards"}``."""
    QL, L = lean.QL, lean.QL.L
    subset = lean.subset
    ref_trace = isa.interpret(code, max_steps=max_steps)

    def _fwd(x, pos):
        if forward is not None:
            return forward(x, pos)
        with torch.no_grad():
            h, _ = lean.forward(x, past=None, q_positions=pos)
        return h

    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    stack: List[int] = []
    store_log: List[dict] = []
    ax_trace: List[int] = []
    cur_pc = 0
    forwards = 0

    for _ in range(max_steps):
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        prev = dict(reg_state)
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = prev["AX"] & 0xFF
        reg_state["STACK0"] = (stack[-1] & 0xFF) if stack else 0   # true top

        x, positions = _build_stream_and_overlay(lean, code, reg_state, store_log, load_addr)
        hidden = _fwd(x, positions)
        forwards += 1
        state = hidden[0, -1]

        pc = _snap(state[L.PC_VAL])
        # efficient-ALU MUL/DIV/MOD (+SHL/SHR under shift_via_mul) decode from the AX
        # NIBBLE band (AX_VAL is the stale popped operand); see LeanQwenVM.decode_ax.
        ax = lean.decode_ax(state, op) & 0xFF
        sp = _snap(state[L.SP_VAL])
        bp = _snap(state[L.BP_VAL])

        if op == isa.PSH:
            stack.append(prev["AX"] & 0xFF)
        elif op in POP_OPS:
            popped = (stack.pop() & 0xFF) if stack else 0
            if subset.memory and op in (isa.SI, isa.SC):
                addr = popped
                store_log = [s for s in store_log
                             if (s["addr"] & 0xFF) != (addr & 0xFF)]
                store_log.append({"addr": addr, "val": ax})

        reg_state = {"PC": pc, "AX": ax, "SP": sp, "BP": bp, "STACK0": reg_state["STACK0"]}
        ax_trace.append(ax)
        if verbose:
            print(f"  step pc={cur_pc} op={isa.NAMES.get(op, op):5s} -> "
                  f"pc={pc} ax={ax} sp={sp} depth={len(stack)}")
        cur_pc = pc
        if float(state[L.HALTED]) > 0.5 or cur_pc < 0 or cur_pc >= len(code):
            break
    return {"ax_trace": ax_trace, "ref_trace": ref_trace,
            "exact": ax_trace == ref_trace, "steps": len(ax_trace),
            "forwards": forwards}


# ===========================================================================
# SPECULATIVE stack-aware driver — perfect-draft, big-K verification.
# ===========================================================================
def speculative_run_lean_stack(lean: LeanQwenVM, code: List[isa.Instr], *,
                               block_steps: int = 256, max_steps: int = 200_000,
                               forward=None, verbose: bool = False) -> LeanSpecResult:
    """Perfect-draft speculation with a REAL data stack.

    Drafts the whole program (deterministic VM, real stack — zero forwards), then
    verifies ``block_steps`` steps per batched lean forward.  Each drafted step's
    window carries its TRUE ``STACK0``, so B steps verify in ONE forward with zero
    approximation.  ``forward`` optionally routes the batched forward through a
    CUDA-graph replay.  Returns forwards-saved + the decoded trace."""
    draft = draft_program_lean_stack(lean, code, max_steps=max_steps)
    ref_trace = draft.ref_trace
    if not draft.steps:
        r = run_program_lean_stack(lean, code, max_steps=max_steps, forward=forward,
                                   verbose=verbose)
        n = r["steps"]
        return LeanSpecResult(
            status="PASS" if r["exact"] else "FAIL", ax_trace=r["ax_trace"],
            ref_trace=r["ref_trace"], exact=r["exact"], steps=n, forwards=r["forwards"],
            naive_forwards=n, speedup=1.0, accepted=n, detail="naive-fallback")

    QL, L = lean.QL, lean.QL.L
    n_steps = len(draft.steps)
    ax_trace: List[int] = []
    forwards = 0
    accepted = 0

    def _fwd(x, pos):
        if forward is not None:
            return forward(x, pos)
        with torch.no_grad():
            h, _ = lean.forward(x, past=None, q_positions=pos)
        return h

    for s0 in range(0, n_steps, block_steps):
        slab = draft.steps[s0:s0 + block_steps]
        x, positions = _build_spec_batch(lean, code, slab)
        hidden = _fwd(x, positions)
        forwards += 1
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if lean.subset.memory else 0
            qrow = (1 + n_store) + len(CAM_REGS)
            state = hidden[i, qrow]
            ax = lean.decode_ax(state, st["op"]) & 0xFF
            ax_trace.append(ax)
            accepted += 1
    exact = ax_trace == ref_trace
    speedup = (n_steps / forwards) if forwards else 0.0
    return LeanSpecResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=n_steps, forwards=forwards, naive_forwards=n_steps,
        speedup=speedup, accepted=accepted,
        detail="" if exact else "spec trace != isa.interpret")
