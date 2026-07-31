"""Shared depth-N DATA-STACK maintenance for the lean CFM drivers (doom wall #3).

## The bug it fixes (#702 1-slot STACK0 wall)

The compacted lean qwen VM mirrors the operand stack with a SINGLE ``STACK0``
register in the windowed token stream — ``PSH`` sets it, a pop-op reads it — and
the *stock* lean drivers (``run_program_lean`` / ``draft_program_lean`` and the
graphed / eviction variants) track that register with ONE scalar ``stk``.  So a
SECOND push overwrites the first: a depth-2 stack expression loses the parked
value.  Measured: ``10 + (3 + 4)`` decodes ``10`` not ``17``; ``mem[a] = x + y``
stores ``0``.  doom's raycaster is full of depth-2+ fixed-point terms
(``a*b + c*d``, ``dist * icos(da) / FP``), so it breaks pervasively.

## The fix (a real Python data stack, driver-side — NOT in the model)

The model only ever needs the ONE top-of-stack value per op (its ALU pops a
single operand).  So the driver keeps a REAL depth-N Python stack ALONGSIDE the
token stream — exactly the way the stock driver already rides a Python CALL
stack for JSR/ENT/LEV — and, every step, writes the TRUE top of that stack into
the ``STACK0`` register frame the model reads.  The model still computes every
opcode (decode + ALU + control-flow) inside the shallow forward; only the
O(1)-per-step stack bookkeeping is driver-side.

This is byte-IDENTICAL to the stock 1-slot path for depth<=1 programs (a
1-element stack's top == the single pushed value == the old ``stk``), and
byte-EXACT (vs ``isa.interpret`` / HF ``run_program(spill_stack_to_kv=True)``)
for arbitrary depth.  It composes with the fast paths (bounded-KV / big-K /
CUDA-graph) unchanged: those verify the model in parallel over per-step windows,
and each window simply carries its own true ``STACK0``.

## Relation to ``spill_stack_to_kv``

``qwen_full_vm.run_program(spill_stack_to_kv=True)`` solves the same wall by
MIRRORING every PSH into the persistent §Memory KV log and reconstructing the
next ``STACK0`` from that log after a pop.  The real-data-stack here is the same
correctness (both reconstruct the true top-of-stack), but keeps the value in a
plain Python list instead of the KV log — so it does NOT pollute the memory-CAM
window with transient stack cells (smaller window, no address collisions with
the program's real ``SI``/``LI`` heap), and needs no ``store_log`` compaction on
the hot path.  For the CFM lean drivers that is strictly better.  A pre-existing
``qwen_lean_stack_driver`` already carried this mechanism for the subroutine
MUL/DIV/MOD; this module GENERALISES it (adds JSR/ENT/ADJ/LEV function support to
the draft) and WIRES it into every lean driver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa
from .nibble_pure_forward import SP_INIT

# pop-ops read the data-stack top (operand A, or the store ADDRESS for SI/SC).
POP_OPS: Tuple[int, ...] = (
    isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD,
    isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
    isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE,
    isa.SI, isa.SC,
)
_STORE_OPS = (isa.SI, isa.SC)


@dataclass
class LeanDataStack:
    """A REAL depth-N operand stack that mirrors what the model computes.

    Drop-in per-step maintenance for a naive lean driver's control loop.  Use::

        ds = LeanDataStack()
        for each VM step:
            reg_state["STACK0"] = ds.top()          # true top -> model frame
            ... build window, forward, decode pc/ax/sp/bp ...
            store_addr = ds.apply(op, prev_ax=prev["AX"], model_ax=ax)
            # store_addr is the popped SI/SC address (None for non-store ops)

    ``apply`` pushes on PSH and pops on a pop-op, returning the popped value for
    a SI/SC (the store address) so the driver can supersede its ``store_log``.
    """

    values: List[int] = field(default_factory=list)

    def top(self) -> int:
        """The TRUE top-of-stack (what the model's ALU must pop this step)."""
        return (self.values[-1] & 0xFF) if self.values else 0

    def depth(self) -> int:
        return len(self.values)

    def apply(self, op, prev_ax: int, model_ax: int) -> Optional[int]:
        """Advance the real stack for one executed op.

        ``prev_ax``  : the AX the model READ this step (pushed by PSH).
        ``model_ax`` : the AX the model PRODUCED this step (stored by SI/SC).
        Returns the popped store ADDRESS for a SI/SC op, else ``None``.
        """
        if op == isa.PSH:
            self.values.append(prev_ax & 0xFF)
            return None
        if op in POP_OPS:
            popped = (self.values.pop() & 0xFF) if self.values else 0
            if op in _STORE_OPS:
                return popped                       # the store ADDRESS
        return None


# ===========================================================================
# UNIFIED perfect-draft with a REAL data stack AND function support.
#
# ``qwen_lean_forward.draft_program_lean`` drafts JSR/ENT/ADJ/LEV but a 1-SLOT
# stack; ``qwen_lean_stack_driver.draft_program_lean_stack`` drafts a real stack
# but falls back on functions.  doom is BOTH function-heavy AND deep-stacked, so
# neither existing draft suffices.  This is the union: a real depth-N data stack
# + a real call stack, so a function that internally computes a depth-2 term
# drafts correctly.
# ===========================================================================
@dataclass
class LeanStackDraft:
    steps: List[dict]              # per step: reg_state BEFORE (true STACK0), store_log, op
    ref_trace: List[int]
    halted: bool


def draft_program_lean_stackfn(code: List[isa.Instr], subset, *,
                               max_steps: int = 200_000,
                               ref_trace: Optional[List[int]] = None
                               ) -> LeanStackDraft:
    """Draft the whole program with a REAL depth-N data stack + a REAL call stack.

    Records, per step, the EXACT window the stack-aware driver feeds the model
    (register file with the TRUE top-of-stack in ``STACK0`` + the compacted store
    log).  Zero forwards.  8-bit value fold (``& 0xFF``), matching the lean model.

    ``ref_trace`` may be supplied by the caller (so it can pick
    ``interpret_with_functions`` vs ``isa.interpret``); if ``None`` a plain
    ``isa.interpret`` is used.  Returns an EMPTY draft only for a genuinely
    out-of-slice op (a syscall outside the slice) -> caller falls back to naive.
    """
    if ref_trace is None:
        ref_trace = isa.interpret(code, max_steps=max_steps)

    pc = 0
    ax = 0
    sp = SP_INIT
    bp = SP_INIT
    stack: List[int] = []                 # REAL data stack (values)
    mem: Dict[int, int] = {}
    store_log: List[dict] = []
    steps: List[dict] = []
    halted = False
    # (ret_pc, saved_bp) frames — the SAME abstraction the naive driver rides.
    call_stack: List[Tuple[Optional[int], Optional[int]]] = []

    for _ in range(max_steps):
        if not (0 <= pc < len(code)):
            break
        op = code[pc].op
        imm = code[pc].imm
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = ax & 0xFF
        top = (stack[-1] & 0xFF) if stack else 0        # TRUE top of stack
        steps.append({
            "reg_state": {"PC": pc, "AX": ax & 0xFF, "SP": sp, "BP": bp, "STACK0": top},
            "store_log": [dict(s) for s in store_log],
            "load_addr": load_addr, "op": op, "pc_at": pc,
        })
        npc = pc + 1
        halted_step = False
        addr = 0
        v = top
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            stack.append(ax & 0xFF); sp -= 4
        elif op in POP_OPS:
            v = (stack.pop() & 0xFF) if stack else 0
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
            elif op in _STORE_OPS:
                addr = v & 0xFF; mem[addr] = ax & 0xFF
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax & 0xFF, 0) & 0xFF
        elif op == isa.JMP:
            npc = imm
        elif op == isa.BZ:
            npc = imm if (ax & 0xFF) == 0 else npc
        elif op == isa.BNZ:
            npc = imm if (ax & 0xFF) != 0 else npc
        elif op == isa.JSR:
            # return address = pc+1 (draft's PRE-increment convention: pc is the
            # CURRENT instruction, so the return is the next one — matches
            # qwen_lean_forward.draft_program_lean).  Push (ret_pc, caller BP).
            call_stack.append((pc + 1, bp)); npc = imm
        elif op == isa.ENT:
            call_stack.append((None, bp)); bp = sp; sp -= imm
        elif op == isa.ADJ:
            sp += imm
        elif op == isa.LEV:
            sp = bp
            saved_bp = ret_pc = None
            if call_stack:
                _, saved_bp = call_stack.pop()
            if call_stack:
                ret_pc, _ = call_stack.pop()
            if saved_bp is not None:
                bp = saved_bp
            if ret_pc is not None:
                npc = ret_pc
        elif op == isa.HALT:
            halted_step = True
        elif op == isa.NOP:
            pass
        else:
            return LeanStackDraft(steps=[], ref_trace=ref_trace, halted=False)

        if subset.memory and op in _STORE_OPS:
            store_log = [s for s in store_log if (s["addr"] & 0xFF) != (addr & 0xFF)]
            store_log.append({"addr": addr, "val": ax & 0xFF})
        pc = npc
        if halted_step or pc < 0 or pc >= len(code):
            halted = True
            break
    return LeanStackDraft(steps=steps, ref_trace=ref_trace, halted=halted)
