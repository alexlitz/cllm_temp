"""OPCODE-DISPATCH of MUL/DIV/MOD (and the low-precision variants) to in-memory
base-ISA BYTECODE SUBROUTINES on the lean shallow model.

The INPUT is ORDINARY C4 bytecode with the REAL ``MUL`` / ``DIV`` / ``MOD`` opcodes
(and the supplemental low-precision opcodes ``MUL8`` / ``DIV8`` / ``MOD8`` /
``MUL16`` / ``DIV16`` / ``MOD16``).  The lean VM has NO hardware multiply/divide —
so, exactly like MALC/FREE/MSET/MCMP ("compiled from C into VM bytecode and execute
entirely neurally", BLOG_SPEC §687/§747) and UNLIKE the *inlined* muldiv of
:mod:`lean_subroutine_muldiv`, each expensive opcode is DISPATCHED as a CALL to a
fixed base-ISA subroutine:

    on fetching MUL/DIV/MOD  ==>  push the return-PC, marshal the two operands into
    the subroutine's operand cells, and JUMP to that op's fixed SUBROUTINE ADDRESS.
    The base-ISA subroutine (shift-and-add MUL / restoring long division, from
    :mod:`lean_subroutine_muldiv`) then RUNS entirely on the shallow ~15-layer
    ``SUBSET_BITWISE`` model and RETURNS the result in AX — no MUL/DIV block in the
    model DEPTH, the cost is VM STEPS.

Mechanism (trap / JSR-to-subroutine)
------------------------------------
:func:`build_dispatch_program` lays out one combined code table::

    [ main program (with real MUL/DIV/MOD opcodes) ] [ subroutine bodies @ fixed addrs ]

and returns a :class:`DispatchTable` mapping each expensive opcode -> its
subroutine entry PC + operand marshalling.  Every subroutine is
``ENT 0 ; <emit_* body> ; LEV`` — so its return rides the SAME driver call-stack the
model already uses for JSR/ENT/LEV functions (see ``qwen_full_vm.run_program``).

The DRIVER and the perfect-draft do the IDENTICAL fetch-time trap: on an expensive
opcode both (a) marshal STACK0 (operand A) + AX (operand B) into the subroutine's
operand cells (an O(1) store-log write, byte-addressed the SAME way SI/SC and the
mem CAM address it), (b) push the return-PC onto the call stack, and (c) set PC to
the subroutine address.  The model then computes the whole multiply/divide LOOP
neurally.  Because the draft's instruction stream is BYTE-IDENTICAL to the model's,
speculation accepts 100% (:func:`speculative_run_dispatch`), and the decoded AX
trace is byte-exact vs ``isa.interpret`` of the ORIGINAL program (the subroutine
computes the same product/quotient).

Integration note (code-from-memory)
-----------------------------------
On a base with CODE-FROM-MEMORY landed the subroutine bodies would live in the
same KV-addressed memory the data does, and the trap would be a genuine indirect
``JSR *dispatch_table[op]``.  Here code-from-memory is NOT yet on this base, so the
subroutines live at FIXED KNOWN ADDRESSES in the combined code table and the trap
consults a static :class:`DispatchTable`.  The operand-marshalling + return-PC push
is the O(1) driver bookkeeping (identical to the existing call-stack / KV-stack
spill), NOT model depth — the integration point is: replace ``table[op]`` with the
in-memory dispatch-vector load once code-from-memory lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import isa
from . import lean_subroutine_muldiv as M


# ===========================================================================
# Supplemental LOW-PRECISION opcodes (driver-recognized trap codes).
#
# These are NOT decoded by the model — the driver TRAPS them at fetch and reroutes
# to a SHORTER subroutine (8/16-bit -> far fewer bit-iterations) for programs that
# don't need the full 32-bit width.  They live above the core ISA one-hot band
# (isa.NUM_OPS = 40); the model only ever runs the base-ISA subroutine bytecode.
# ===========================================================================
MUL8, DIV8, MOD8 = 40, 41, 42
MUL16, DIV16, MOD16 = 43, 44, 45
MUL32, DIV32, MOD32 = isa.MUL, isa.DIV, isa.MOD   # the real opcodes == full 32-bit

DISPATCH_NAMES = {
    isa.MUL: "MUL", isa.DIV: "DIV", isa.MOD: "MOD",
    MUL8: "MUL8", DIV8: "DIV8", MOD8: "MOD8",
    MUL16: "MUL16", DIV16: "DIV16", MOD16: "MOD16",
}

# Every opcode the dispatcher TRAPS (the model never computes these directly).
DISPATCHED_OPS = tuple(DISPATCH_NAMES)


# The reference SEMANTICS of each dispatched opcode (operand A = STACK0, B = AX),
# used by the byte-exact oracle and to validate the subroutine.  Width is the mask.
_OP_SEMANTICS: Dict[int, Tuple[Callable[[int, int], int], int]] = {
    isa.MUL:  (lambda a, b: a * b, 0xFF),          # 8-bit fold (matches isa.interpret)
    isa.DIV:  (lambda a, b: (a // b) if b else 0, 0xFF),
    isa.MOD:  (lambda a, b: (a % b) if b else 0, 0xFF),
    MUL8:  (lambda a, b: a * b, 0xFF),
    DIV8:  (lambda a, b: (a // b) if b else 0, 0xFF),
    MOD8:  (lambda a, b: (a % b) if b else 0, 0xFF),
    MUL16: (lambda a, b: a * b, 0xFFFF),
    DIV16: (lambda a, b: (a // b) if b else 0, 0xFFFF),
    MOD16: (lambda a, b: (a % b) if b else 0, 0xFFFF),
}


# ===========================================================================
# Subroutine bodies (base-ISA, from lean_subroutine_muldiv), wrapped ENT/LEV.
#
# Each subroutine reads its operands from FIXED cells and leaves the result in AX
# (the low result byte for the multi-byte variants; the driver reads the full
# multi-byte result out of the result cells for the >8-bit precision decode).
# ===========================================================================
@dataclass
class _Subr:
    name: str
    body: List[isa.Instr]         # ENT 0 ; <compute> ; LEV
    in_cells: List[int]           # operand A cells (LSB first), operand B cells
    b_cells: List[int]
    res_cells: List[int]          # where the (multi-byte) result lands
    nbytes: int


def _wrap(body_asm) -> List[isa.Instr]:
    """``ENT 0 ; <body> ; LEV`` — a callable subroutine whose return rides the
    driver call-stack (JSR pushes ret-PC, ENT saves BP, LEV pops both)."""
    from .nibble_runtime import Asm
    a = Asm()
    a.emit(isa.ENT, 0)
    a.splice(body_asm)
    a.emit(isa.LEV, 0)
    return a.instrs()


def _make_subroutines() -> Dict[int, _Subr]:
    """One base-ISA subroutine per dispatched opcode (8/16/32-bit precision)."""
    subrs: Dict[int, _Subr] = {}
    # 8-bit (single-cell operands C_X / C_Y -> AX).
    subrs[MUL8] = _Subr("mul8", _wrap(M.emit_mul8()), [M.C_X], [M.C_Y], [M.C_R], 1)
    subrs[DIV8] = _Subr("div8", _wrap(M.emit_divmod8(want_rem=False)),
                        [M.C_X], [M.C_Y], [M.C_Q], 1)
    subrs[MOD8] = _Subr("mod8", _wrap(M.emit_mod8()), [M.C_X], [M.C_Y], [M.C_R], 1)
    # 16-bit (two consecutive cells per operand, result in M_/Q_ cells).
    subrs[MUL16] = _Subr("mul16", _wrap(M.emit_mul_mb(2)),
                         M.A_BYTES[:2], M.B_BYTES[:2], M.M_BYTES[:2], 2)
    subrs[DIV16] = _Subr("div16", _wrap(M.emit_divmod_mb(2, want_rem=False)),
                         M.A_BYTES[:2], M.B_BYTES[:2], M.Q_BYTES[:2], 2)
    subrs[MOD16] = _Subr("mod16", _wrap(M.emit_divmod_mb(2, want_rem=True)),
                         M.A_BYTES[:2], M.B_BYTES[:2], M.M_BYTES[:2], 2)
    # 32-bit == the REAL opcodes (full 4-byte precision).
    subrs[isa.MUL] = _Subr("mul32", _wrap(M.emit_mul_mb(4)),
                           M.A_BYTES[:4], M.B_BYTES[:4], M.M_BYTES[:4], 4)
    subrs[isa.DIV] = _Subr("div32", _wrap(M.emit_divmod_mb(4, want_rem=False)),
                           M.A_BYTES[:4], M.B_BYTES[:4], M.Q_BYTES[:4], 4)
    subrs[isa.MOD] = _Subr("mod32", _wrap(M.emit_divmod_mb(4, want_rem=True)),
                           M.A_BYTES[:4], M.B_BYTES[:4], M.M_BYTES[:4], 4)
    return subrs


# ===========================================================================
# The combined program layout + dispatch table.
# ===========================================================================
@dataclass
class DispatchTable:
    code: List[isa.Instr]                 # combined: [main][subroutine bodies]
    entry: Dict[int, int]                 # opcode -> subroutine entry PC
    subrs: Dict[int, _Subr]               # opcode -> subroutine descriptor
    main_len: int                         # length of the main program prefix


def build_dispatch_program(main_code: List[isa.Instr],
                           dispatched: Optional[Tuple[int, ...]] = None
                           ) -> DispatchTable:
    """Lay out ``main_code`` (ordinary bytecode, real MUL/DIV/MOD + supplemental
    opcodes) followed by the base-ISA subroutine bodies at fixed addresses, and
    return the :class:`DispatchTable`.

    Only the opcodes ACTUALLY USED by ``main_code`` (intersected with ``dispatched``)
    get a subroutine appended, so an 8-bit-only program never pays for the 32-bit
    machinery.  The subroutines' internal branch targets are already absolute
    instruction indices relative to their own start; ``_wrap`` keeps them so, and we
    append each body at a known offset (the entry PC)."""
    if dispatched is None:
        dispatched = DISPATCHED_OPS
    used = [op for op in dispatched
            if any(ins.op == op for ins in main_code)]
    all_subrs = _make_subroutines()
    code = list(main_code)
    entry: Dict[int, int] = {}
    subrs: Dict[int, _Subr] = {}
    for op in used:
        sr = all_subrs[op]
        base = len(code)
        entry[op] = base
        subrs[op] = sr
        # re-base the subroutine body's internal absolute targets to ``base``.
        for ins in sr.body:
            if ins.op in (isa.JMP, isa.BZ, isa.BNZ, isa.JSR):
                code.append(isa.Instr(ins.op, ins.imm + base))
            else:
                code.append(isa.Instr(ins.op, ins.imm))
    return DispatchTable(code=code, entry=entry, subrs=subrs, main_len=len(main_code))


# ===========================================================================
# Operand marshalling (driver-side, O(1) per trap).
# ===========================================================================
def _marshal_operands(store_log: List[dict], sr: _Subr, a_val: int, b_val: int) -> List[dict]:
    """Write operand A -> ``sr.in_cells`` and B -> ``sr.b_cells`` (LSB-first bytes)
    into the byte-addressed store-log the mem CAM reads — the SAME latest-write-wins
    compaction SI/SC use.  Also ZEROES the result cells so a stale prior result can
    never leak into the multi-byte read.  Returns the updated store_log."""
    def _put(sl, addr, val):
        sl = [s for s in sl if (s["addr"] & 0xFF) != (addr & 0xFF)]
        sl.append({"addr": addr, "val": val & 0xFF})
        return sl
    sl = list(store_log)
    for i, c in enumerate(sr.in_cells):
        sl = _put(sl, c, (a_val >> (8 * i)) & 0xFF)
    for i, c in enumerate(sr.b_cells):
        sl = _put(sl, c, (b_val >> (8 * i)) & 0xFF)
    for c in sr.res_cells:
        sl = _put(sl, c, 0)
    return sl


def _read_result(mem: Dict[int, int], sr: _Subr) -> int:
    return sum((mem.get(c & 0xFF, 0) & 0xFF) << (8 * i) for i, c in enumerate(sr.res_cells))


# ===========================================================================
# Pure-Python word-exact REFERENCE dispatch interpreter (the byte-exact golden).
#
# Runs the combined program with the fetch-time TRAP: on a dispatched opcode it
# marshals (STACK0=A, AX=B) into the subroutine cells, pushes the return-PC, and
# jumps to the subroutine entry; the base-ISA subroutine runs (ENT/LEV framed) and
# leaves the result in AX (and the multi-byte result cells).  8-bit AX fold per step
# (matching the lean model + draft); multi-byte values live in memory.
# ===========================================================================
_POP_OPS = (isa.ADD, isa.SUB, isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR,
            isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE, isa.SI, isa.SC)


@dataclass
class _DispatchState:
    """Shared pure-Python transition for the reference / draft (identical logic)."""
    table: DispatchTable
    ax: int = 0
    pc: int = 0
    sp: int = field(default=None)
    bp: int = field(default=None)
    stack: List[int] = field(default_factory=list)
    mem: Dict[int, int] = field(default_factory=dict)
    store_log: List[dict] = field(default_factory=list)
    call_stack: List[Tuple[Optional[int], Optional[int]]] = field(default_factory=list)


def _dispatch_step(st: _DispatchState) -> Tuple[Optional[int], bool, Optional[int]]:
    """Advance one VM step of the combined program with the trap semantics.

    Returns ``(op, halted, load_addr)`` — ``op`` is the fetched opcode (None if PC
    out of range), ``load_addr`` the LI/LC address (for the model window).  Mutates
    ``st`` in place (ax/pc/sp/bp/stack/mem/store_log/call_stack).  A DISPATCHED op is
    trapped: operands marshalled, return-PC pushed, PC := subroutine entry."""
    from .nibble_pure_forward import SP_INIT
    if st.sp is None:
        st.sp = SP_INIT
    if st.bp is None:
        st.bp = SP_INIT
    code = st.table.code
    if not (0 <= st.pc < len(code)):
        return None, True, None
    ins = code[st.pc]
    op, imm = ins.op, ins.imm
    load_addr = (st.ax & 0xFF) if op in (isa.LI, isa.LC) else None

    # -- TRAP: dispatched opcode -> JSR-to-subroutine ------------------------
    if op in st.table.entry:
        sr = st.table.subrs[op]
        a_val = st.stack.pop() if st.stack else 0        # operand A (popped)
        st.sp += 4
        b_val = st.ax                                    # operand B (AX)
        # marshal into the subroutine's operand cells (mask to precision width).
        semantics_mask = _OP_SEMANTICS[op][1]
        st.store_log = _marshal_operands(st.store_log, sr, a_val & semantics_mask,
                                         b_val & semantics_mask)
        for s in st.store_log:                           # keep mem in sync
            st.mem[s["addr"] & 0xFF] = s["val"] & 0xFF
        st.call_stack.append((st.pc + 1, st.bp))         # return to the NEXT main pc
        st.pc = st.table.entry[op]
        return op, False, None

    npc = st.pc + 1
    halted = False
    addr = 0
    v = (st.stack[-1] & 0xFF) if st.stack else 0
    if op == isa.IMM:
        st.ax = imm & 0xFF
    elif op == isa.LEA:
        st.ax = (st.bp + imm) & 0xFF
    elif op == isa.PSH:
        st.stack.append(st.ax & 0xFF); st.sp -= 4
    elif op in _POP_OPS:
        v = st.stack.pop() & 0xFF if st.stack else 0
        st.sp += 4
        if op == isa.ADD: st.ax = (v + st.ax) & 0xFF
        elif op == isa.SUB: st.ax = (v - st.ax) & 0xFF
        elif op == isa.AND: st.ax = v & st.ax
        elif op == isa.OR: st.ax = v | st.ax
        elif op == isa.XOR: st.ax = v ^ st.ax
        elif op == isa.SHL: st.ax = (v << st.ax) & 0xFF
        elif op == isa.SHR: st.ax = (v >> st.ax) & 0xFF
        elif op == isa.EQ: st.ax = 1 if v == st.ax else 0
        elif op == isa.NE: st.ax = 1 if v != st.ax else 0
        elif op == isa.LT: st.ax = 1 if v < st.ax else 0
        elif op == isa.GT: st.ax = 1 if v > st.ax else 0
        elif op == isa.LE: st.ax = 1 if v <= st.ax else 0
        elif op == isa.GE: st.ax = 1 if v >= st.ax else 0
        elif op in (isa.SI, isa.SC):
            addr = v & 0xFF
            st.mem[addr] = st.ax & 0xFF
    elif op in (isa.LI, isa.LC):
        st.ax = st.mem.get(st.ax & 0xFF, 0) & 0xFF
    elif op == isa.JMP:
        npc = imm
    elif op == isa.BZ:
        npc = imm if (st.ax & 0xFF) == 0 else npc
    elif op == isa.BNZ:
        npc = imm if (st.ax & 0xFF) != 0 else npc
    elif op == isa.JSR:
        st.call_stack.append((st.pc + 1, st.bp)); npc = imm
    elif op == isa.ENT:
        st.call_stack.append((None, st.bp)); st.bp = st.sp; st.sp -= imm
    elif op == isa.ADJ:
        st.sp += imm
    elif op == isa.LEV:
        st.sp = st.bp
        saved_bp = ret_pc = None
        if st.call_stack:
            _, saved_bp = st.call_stack.pop()
        if st.call_stack:
            ret_pc, _ = st.call_stack.pop()
        if saved_bp is not None:
            st.bp = saved_bp
        if ret_pc is not None:
            npc = ret_pc
    elif op == isa.HALT:
        halted = True
    elif op == isa.NOP:
        pass
    else:
        raise NotImplementedError(f"dispatch: op {isa.NAMES.get(op, op)} unsupported")

    if op in (isa.SI, isa.SC):
        st.store_log = [s for s in st.store_log if (s["addr"] & 0xFF) != (addr & 0xFF)]
        st.store_log.append({"addr": addr, "val": st.ax & 0xFF})
    st.pc = npc
    if npc < 0 or npc >= len(code):
        halted = True
    return op, halted, load_addr


def interpret_dispatch(table: DispatchTable, max_steps: int = 2_000_000) -> List[int]:
    """Word-exact pure-Python reference: the AX value after every executed VM step
    of the combined program (with the trap dispatch).  This is the byte-exact golden
    the model-driver + draft must match."""
    st = _DispatchState(table=table)
    trace: List[int] = []
    for _ in range(max_steps):
        op, halted, _ = _dispatch_step(st)
        if op is None:
            break
        trace.append(st.ax & 0xFF)
        if halted:
            break
    return trace


def run_original_reference(main_code: List[isa.Instr], max_steps: int = 2_000_000) -> List[int]:
    """The AX trace ``isa.interpret`` would produce for the ORIGINAL program IF it
    computed each dispatched opcode at its precision width (8-bit MUL/DIV/MOD folds
    to isa.interpret exactly; the supplemental MUL8..MOD16 fold to their width).
    Used to prove the dispatch's decoded result equals the op's true semantics."""
    ax = 0
    sp = None
    bp = None
    stack: List[int] = []
    mem: Dict[int, int] = {}
    pc = 0
    trace: List[int] = []
    from .nibble_pure_forward import SP_INIT
    sp = bp = SP_INIT
    for _ in range(max_steps):
        if not (0 <= pc < len(main_code)):
            break
        ins = main_code[pc]
        op, imm = ins.op, ins.imm
        npc = pc + 1
        halted = False
        if op in _OP_SEMANTICS:
            b = ax
            a = stack.pop() if stack else 0
            sp += 4
            fn, mask = _OP_SEMANTICS[op]
            ax = fn(a & mask, b & mask) & mask & 0xFF     # 8-bit fold for the trace
        elif op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            stack.append(ax & 0xFF); sp -= 4
        elif op in _POP_OPS:
            v = stack.pop() & 0xFF if stack else 0
            sp += 4
            if op == isa.ADD: ax = (v + ax) & 0xFF
            elif op == isa.SUB: ax = (v - ax) & 0xFF
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
                mem[v & 0xFF] = ax & 0xFF
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax & 0xFF, 0) & 0xFF
        elif op == isa.JMP:
            npc = imm
        elif op == isa.BZ:
            npc = imm if (ax & 0xFF) == 0 else npc
        elif op == isa.BNZ:
            npc = imm if (ax & 0xFF) != 0 else npc
        elif op == isa.HALT:
            halted = True
        elif op == isa.NOP:
            pass
        else:
            raise NotImplementedError(f"ref: op {isa.NAMES.get(op, op)}")
        trace.append(ax & 0xFF)
        pc = npc
        if halted or pc < 0 or pc >= len(main_code):
            break
    return trace


# ===========================================================================
# The PERFECT DRAFT — the driver's exact per-step transition (incl. the trap) in
# pure Python, recording the register/store window the model is fed each step.
#
# The trap is a TWO-fold record: (1) at the dispatched-op step the draft records the
# window the model would see for that main-level op (its OP is the dispatched code —
# the model NEVER sees it, so the draft marks it a "trap" step with NO forward), and
# (2) every subroutine step records a normal base-ISA window the model verifies.
# Because the draft's transition IS the model's (base-ISA subroutine), draft == model
# byte-for-byte and speculation accepts 100%.
# ===========================================================================
@dataclass
class DispatchDraft:
    steps: List[dict]                     # per model-visible step: window + op
    ref_trace: List[int]                  # interpret_dispatch golden
    halted: bool
    n_trap_steps: int                     # dispatched-op fetches (driver-only, no fwd)


def draft_program_dispatch(lean, table: DispatchTable,
                           max_steps: int = 2_000_000) -> DispatchDraft:
    """Draft the whole combined program (zero forwards), recording per BASE-ISA step
    the exact window the stack-aware driver feeds the model (register file with the
    TRUE top-of-stack in STACK0 + the compacted store log).  A dispatched-op fetch is
    a DRIVER-ONLY trap step (operand marshalling + return-PC push + PC reroute); it is
    NOT fed to the model, so it is recorded separately and carries no window."""
    from .nibble_pure_forward import SP_INIT
    ref_trace = interpret_dispatch(table, max_steps=max_steps)
    st = _DispatchState(table=table)
    st.sp = SP_INIT
    st.bp = SP_INIT
    steps: List[dict] = []
    halted = False
    n_trap = 0
    subset_memory = getattr(lean.subset, "memory", True)
    for _ in range(max_steps):
        if not (0 <= st.pc < len(table.code)):
            halted = True
            break
        op = table.code[st.pc].op
        load_addr = None
        if subset_memory and op in (isa.LI, isa.LC):
            load_addr = st.ax & 0xFF
        is_trap = op in table.entry
        top = st.stack[-1] & 0xFF if st.stack else 0
        pre_ax = st.ax & 0xFF
        if is_trap:
            # DRIVER-ONLY trap step: no model forward.  AX is UNCHANGED (the trap
            # marshals operands + reroutes PC), so its decoded AX is the known pre-op
            # AX.  Recorded so the trace stays aligned with interpret_dispatch, but
            # flagged trap=True so speculation accepts it with ZERO forward.
            steps.append({"trap": True, "op": op, "pc_at": st.pc, "ax": pre_ax})
            n_trap += 1
        else:
            # a real model step: snapshot the window BEFORE the transition.
            steps.append({
                "trap": False,
                "reg_state": {"PC": st.pc, "AX": pre_ax, "SP": st.sp,
                              "BP": st.bp, "STACK0": top},
                "store_log": [dict(s) for s in (st.store_log if subset_memory else [])],
                "load_addr": load_addr, "op": op, "pc_at": st.pc,
            })
        step_op, step_halt, _ = _dispatch_step(st)
        if step_halt:
            halted = True
            break
    return DispatchDraft(steps=steps, ref_trace=ref_trace, halted=halted,
                         n_trap_steps=n_trap)


# ===========================================================================
# NAIVE model driver — one lean forward per BASE-ISA step; trap steps are O(1).
# ===========================================================================
@dataclass
class DispatchResult:
    status: str
    ax_trace: List[int]
    ref_trace: List[int]
    exact: bool
    steps: int                     # total VM steps (model + trap)
    model_steps: int               # steps that ran a lean forward
    trap_steps: int                # dispatched-op fetches (driver-only)
    forwards: int
    naive_forwards: int
    speedup: float
    accepted: int
    detail: str = ""


def run_program_dispatch(lean, table: DispatchTable, max_steps: int = 2_000_000,
                         forward=None, verbose: bool = False) -> DispatchResult:
    """Execute the combined program on the lean model, one forward per base-ISA step;
    a dispatched-op fetch is trapped (marshal operands + push ret-PC + reroute PC), an
    O(1) driver step with NO forward.  Uses the STACK-AWARE window (real depth-N data
    stack) since the subroutines reach data-stack depth 2-3.  Byte-exact vs
    ``interpret_dispatch``."""
    import torch
    from .qwen_lean_forward import _build_stream_and_overlay, _snap
    from .nibble_pure_forward import SP_INIT
    L = lean.QL.L
    subset = lean.subset
    ref_trace = interpret_dispatch(table, max_steps=max_steps)

    def _fwd(x, pos):
        if forward is not None:
            return forward(x, pos)
        with torch.no_grad():
            h, _ = lean.forward(x, past=None, q_positions=pos)
        return h

    st = _DispatchState(table=table)
    st.sp = SP_INIT
    st.bp = SP_INIT
    reg_state = {"PC": 0, "AX": 0, "SP": SP_INIT, "BP": SP_INIT, "STACK0": 0}
    ax_trace: List[int] = []
    forwards = 0
    model_steps = 0
    trap_steps = 0
    for _ in range(max_steps):
        if not (0 <= st.pc < len(table.code)):
            break
        op = table.code[st.pc].op
        if op in table.entry:                      # TRAP: driver-only, no forward.
            pre_ax = st.ax & 0xFF
            _dispatch_step(st)                     # marshal + push ret-PC + reroute
            ax_trace.append(pre_ax)                # AX unchanged across the trap
            trap_steps += 1
            if verbose:
                print(f"  TRAP pc-> {isa.NAMES.get(op, DISPATCH_NAMES.get(op, op))} "
                      f"-> subr@{table.entry[op]}")
            continue
        # a real model step: build the window, forward, decode AX, then transition.
        load_addr = None
        if subset.memory and op in (isa.LI, isa.LC):
            load_addr = st.ax & 0xFF
        reg_state = {"PC": st.pc, "AX": st.ax & 0xFF, "SP": st.sp, "BP": st.bp,
                     "STACK0": (st.stack[-1] & 0xFF) if st.stack else 0}
        store_log = [dict(s) for s in (st.store_log if subset.memory else [])]
        x, positions = _build_stream_and_overlay(lean, table.code, reg_state, store_log,
                                                 load_addr)
        hidden = _fwd(x, positions)
        forwards += 1
        model_steps += 1
        state = hidden[0, -1]
        model_ax = _snap(state[L.AX_VAL]) & 0xFF
        _dispatch_step(st)                         # advance the pure-Python truth
        ax_trace.append(model_ax)
        if float(state[L.HALTED]) > 0.5:
            break
    exact = ax_trace == ref_trace
    speedup = (len(ax_trace) / forwards) if forwards else float("inf")
    return DispatchResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=len(ax_trace), model_steps=model_steps, trap_steps=trap_steps,
        forwards=forwards, naive_forwards=len(ax_trace), speedup=speedup,
        accepted=len(ax_trace), detail="" if exact else "trace != interpret_dispatch")


# ===========================================================================
# SPECULATIVE model driver — perfect-draft, big-K batched verification.
# ===========================================================================
def speculative_run_dispatch(lean, table: DispatchTable, *, block_steps: int = 256,
                             max_steps: int = 2_000_000, forward=None,
                             verbose: bool = False) -> DispatchResult:
    """Perfect-draft speculation on the dispatched program.

    Drafts the whole combined program (deterministic VM incl. the trap — zero
    forwards), then verifies ``block_steps`` MODEL steps per batched lean forward.
    Trap steps carry their known AX (unchanged) and are accepted with ZERO forward.
    Every model step's window carries its TRUE STACK0, so B steps verify in ONE
    forward with zero approximation -> 100% speculation acceptance, byte-exact."""
    import torch
    from .qwen_lean_forward import _build_spec_batch, _snap, CAM_REGS
    L = lean.QL.L
    draft = draft_program_dispatch(lean, table, max_steps=max_steps)
    ref_trace = draft.ref_trace

    def _fwd(x, pos):
        if forward is not None:
            return forward(x, pos)
        with torch.no_grad():
            h, _ = lean.forward(x, past=None, q_positions=pos)
        return h

    ax_trace: List[int] = []
    forwards = 0
    accepted = 0
    model_steps = [s for s in draft.steps if not s["trap"]]
    # verify the MODEL steps in batches; splice the trap AXs back in trace order.
    model_ax: List[int] = []
    for s0 in range(0, len(model_steps), block_steps):
        slab = model_steps[s0:s0 + block_steps]
        x, positions = _build_spec_batch(lean, table.code, slab)
        hidden = _fwd(x, positions)
        forwards += 1
        for i, st in enumerate(slab):
            n_store = len(st["store_log"]) if lean.subset.memory else 0
            qrow = (1 + n_store) + len(CAM_REGS)
            state = hidden[i, qrow]
            model_ax.append(_snap(state[L.AX_VAL]) & 0xFF)
    # reassemble the full trace in step order (model AX from the batch, trap AX known).
    mi = 0
    for s in draft.steps:
        if s["trap"]:
            ax_trace.append(s["ax"])
        else:
            ax_trace.append(model_ax[mi]); mi += 1
        accepted += 1
    exact = ax_trace == ref_trace
    total = len(ax_trace)
    # naive = one forward per model step; trap steps are free either way.
    naive = len(model_steps)
    speedup = (naive / forwards) if forwards else float("inf")
    return DispatchResult(
        status="PASS" if exact else "FAIL", ax_trace=ax_trace, ref_trace=ref_trace,
        exact=exact, steps=total, model_steps=len(model_steps),
        trap_steps=draft.n_trap_steps, forwards=forwards, naive_forwards=naive,
        speedup=speedup, accepted=accepted,
        detail="" if exact else "spec trace != interpret_dispatch")
