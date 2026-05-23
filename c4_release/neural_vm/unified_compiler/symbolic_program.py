"""Whole-program symbolic execution for declarative VM opcode specs.

This module is the program-level companion to ``ir.py``.  ``ir.py`` can run
declarative layer primitives position-wise; this file runs bytecode through a
table of opcode declarations.  The same opcode table gives us a cheap oracle
for debugging declaration semantics before lowering those declarations into
attention/FFN weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..constants import INSTR_WIDTH, PC_OFFSET, STACK_INIT, idx_to_pc, pc_to_idx


Word = int
OpcodeEffect = Callable[["SymbolicProgramState", int], None]


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _s24(value: int) -> int:
    value &= 0xFFFFFF
    if value >= 0x800000:
        value -= 0x1000000
    return value


def _binop(opcode: int, left: int, right: int) -> int:
    left = _u32(left)
    right = _u32(right)
    if opcode == 14:
        return left | right
    if opcode == 15:
        return left ^ right
    if opcode == 16:
        return left & right
    if opcode == 17:
        return 1 if left == right else 0
    if opcode == 18:
        return 1 if left != right else 0
    if opcode == 19:
        return 1 if left < right else 0
    if opcode == 20:
        return 1 if left > right else 0
    if opcode == 21:
        return 1 if left <= right else 0
    if opcode == 22:
        return 1 if left >= right else 0
    if opcode == 23:
        return _u32(left << right)
    if opcode == 24:
        return left >> right
    if opcode == 25:
        return _u32(left + right)
    if opcode == 26:
        return _u32(left - right)
    if opcode == 27:
        return _u32(left * right)
    if opcode == 28:
        return _u32(left // right) if right != 0 else 0
    if opcode == 29:
        return _u32(left % right) if right != 0 else 0
    raise ValueError(f"unknown binary opcode: {opcode}")


@dataclass(frozen=True)
class DeclarativeOpcodeSpec:
    """One opcode declaration for symbolic execution and diagnostics."""

    opcode: int
    name: str
    reads: frozenset[str]
    writes: frozenset[str]
    effect: OpcodeEffect
    section: str = ""


@dataclass(frozen=True)
class SymbolicStepTrace:
    """A compact per-instruction trace entry."""

    step: int
    opcode: int
    name: str
    imm: int
    idx_before: int
    pc_before: int
    idx_after: int
    pc_after: int
    ax_before: int
    ax_after: int
    sp_before: int
    sp_after: int
    bp_before: int
    bp_after: int
    mem_addr: int
    mem_value: int
    halted: bool


@dataclass
class SymbolicProgramState:
    """Mutable VM state used by declarative opcode effects."""

    code: Sequence[int]
    idx: int = 0
    pc: int = PC_OFFSET
    ax: int = 0
    sp: int = STACK_INIT
    bp: int = STACK_INIT
    memory: Dict[int, int] = field(default_factory=dict)
    mem_bytes: Dict[int, int] = field(default_factory=dict)
    halted: bool = False
    output: List[str] = field(default_factory=list)
    stdin_buf: str = ""
    stdin_pos: int = 0
    heap_ptr: int = 0x20000
    steps: int = 0
    last_mem_addr: int = 0
    last_mem_val: int = 0
    trace: List[SymbolicStepTrace] = field(default_factory=list)

    def load_data(self, data: Iterable[int]) -> None:
        for i, byte in enumerate(data):
            self.memory[0x10000 + i] = byte & 0xFF

    def mem_read(self, addr: int) -> int:
        return self.memory.get(_u32(addr), 0)

    def mem_write(self, addr: int, value: int) -> None:
        addr = _u32(addr)
        value = _u32(value)
        self.memory[addr] = value
        self.last_mem_addr = addr
        self.last_mem_val = value

    def push(self, value: int) -> None:
        self.sp = _u32(self.sp - 8)
        self.mem_write(self.sp, value)

    def pop(self) -> int:
        value = self.mem_read(self.sp)
        self.sp = _u32(self.sp + 8)
        return value

    def read_c_string(self, addr: int, *, limit: int = 10000) -> str:
        chars: List[str] = []
        for _ in range(limit):
            byte = self.mem_read(addr) & 0xFF
            if byte == 0:
                break
            chars.append(chr(byte))
            addr = _u32(addr + 1)
        return "".join(chars)

    def resolve_static_target_idx(self, target: int) -> Optional[int]:
        target = int(target)
        if 0 <= target < len(self.code):
            return target

        pc_idx = pc_to_idx(target)
        if 0 <= pc_idx < len(self.code) and idx_to_pc(pc_idx) == target:
            return pc_idx

        byte_idx = target // INSTR_WIDTH
        if (
            0 <= byte_idx < len(self.code)
            and target % INSTR_WIDTH == 0
        ):
            return byte_idx
        return None

    def branch_to(self, target: int) -> None:
        target = _u32(target)
        target_idx = self.resolve_static_target_idx(target)
        if target_idx is not None:
            self.idx = target_idx
            self.pc = idx_to_pc(target_idx)
        else:
            self.idx = len(self.code)
            self.pc = target

    def fetch_memory_instruction(self, pc: int) -> Optional[tuple[int, int]]:
        op = self.mem_bytes.get(pc)
        if op is None:
            return None
        imm = (
            self.mem_bytes.get(pc + 1, 0)
            | (self.mem_bytes.get(pc + 2, 0) << 8)
            | (self.mem_bytes.get(pc + 3, 0) << 16)
        )
        return op & 0xFF, _s24(imm)

    def write_code_to_memory(self, addr: int, opcode: int, imm: int = 0) -> None:
        imm &= 0xFFFFFF
        self.mem_bytes[addr] = opcode & 0xFF
        self.mem_bytes[addr + 1] = imm & 0xFF
        self.mem_bytes[addr + 2] = (imm >> 8) & 0xFF
        self.mem_bytes[addr + 3] = (imm >> 16) & 0xFF

    def get_output(self) -> str:
        return "".join(self.output)


def _decode_static_instruction(instr: int) -> tuple[int, int]:
    return instr & 0xFF, _s24(instr >> 8)


def _op_lea(state: SymbolicProgramState, imm: int) -> None:
    state.ax = _u32(state.bp + imm)


def _op_imm(state: SymbolicProgramState, imm: int) -> None:
    state.ax = _u32(imm)


def _op_jmp(state: SymbolicProgramState, imm: int) -> None:
    state.branch_to(imm)


def _op_jsr(state: SymbolicProgramState, imm: int) -> None:
    state.push(state.pc)
    state.branch_to(imm)


def _op_bz(state: SymbolicProgramState, imm: int) -> None:
    if state.ax == 0:
        state.branch_to(imm)


def _op_bnz(state: SymbolicProgramState, imm: int) -> None:
    if state.ax != 0:
        state.branch_to(imm)


def _op_ent(state: SymbolicProgramState, imm: int) -> None:
    state.push(state.bp)
    state.bp = state.sp
    state.sp = _u32(state.sp - imm)


def _op_adj(state: SymbolicProgramState, imm: int) -> None:
    state.sp = _u32(state.sp + imm)


def _op_lev(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.sp = state.bp
    state.bp = state.mem_read(state.sp)
    state.sp = _u32(state.sp + 8)
    ret_addr = state.mem_read(state.sp)
    state.sp = _u32(state.sp + 8)
    state.branch_to(ret_addr)


def _op_li(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.ax = state.mem_read(state.ax)


def _op_lc(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.ax = state.mem_read(state.ax) & 0xFF


def _op_si(state: SymbolicProgramState, imm: int) -> None:
    del imm
    addr = state.pop()
    state.mem_write(addr, state.ax)


def _op_sc(state: SymbolicProgramState, imm: int) -> None:
    del imm
    addr = state.pop()
    state.mem_write(addr, state.ax & 0xFF)


def _op_psh(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.push(state.ax)


def _op_binop(opcode: int) -> OpcodeEffect:
    def effect(state: SymbolicProgramState, imm: int) -> None:
        del imm
        left = state.pop()
        state.ax = _binop(opcode, left, state.ax)

    return effect


def _op_open(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.ax = _u32(-1)


def _op_read(state: SymbolicProgramState, imm: int) -> None:
    del imm
    fd = state.mem_read(state.sp + 16)
    buf = state.mem_read(state.sp + 8)
    count = state.mem_read(state.sp)
    if fd != 0:
        state.ax = _u32(-1)
        return

    bytes_read = 0
    for i in range(count):
        if state.stdin_pos >= len(state.stdin_buf):
            break
        state.mem_write(buf + i, ord(state.stdin_buf[state.stdin_pos]))
        state.stdin_pos += 1
        bytes_read += 1
    state.ax = bytes_read


def _op_clos(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.ax = 0


def _op_prtf(state: SymbolicProgramState, imm: int) -> None:
    del imm
    argc = 1
    if state.idx < len(state.code):
        next_op, next_imm = _decode_static_instruction(state.code[state.idx])
        if next_op == 7:
            argc = next_imm // 8

    fmt_addr = state.mem_read(state.sp + (argc - 1) * 8)
    fmt = state.read_c_string(fmt_addr)
    pieces: List[str] = []
    arg_idx = 1
    i = 0
    while i < len(fmt):
        ch = fmt[i]
        if ch == "\\" and i + 1 < len(fmt):
            nxt = fmt[i + 1]
            if nxt == "n":
                pieces.append("\n")
                i += 2
                continue
            if nxt == "t":
                pieces.append("\t")
                i += 2
                continue
            if nxt == "\\":
                pieces.append("\\")
                i += 2
                continue
        if ch == "%" and i + 1 < len(fmt):
            spec = fmt[i + 1]
            if spec == "%":
                pieces.append("%")
                i += 2
                continue
            val = 0
            if arg_idx < argc:
                val = state.mem_read(state.sp + (argc - 1 - arg_idx) * 8)
            if spec == "d":
                if val > 0x7FFFFFFF:
                    val -= 0x100000000
                pieces.append(str(val))
            elif spec == "x":
                pieces.append(format(val & 0xFFFFFFFF, "x"))
            elif spec == "c":
                pieces.append(chr(val & 0xFF))
            elif spec == "s":
                pieces.append(state.read_c_string(val))
            else:
                pieces.append("%" + spec)
            arg_idx += 1
            i += 2
            continue
        pieces.append(ch)
        i += 1

    state.output.append("".join(pieces))
    state.ax = 0


def _op_malc(state: SymbolicProgramState, imm: int) -> None:
    del imm
    size = state.mem_read(state.sp)
    if size <= 0:
        state.ax = 0
        return
    ptr = state.heap_ptr
    state.heap_ptr += size
    if state.heap_ptr & 7:
        state.heap_ptr += 8 - (state.heap_ptr & 7)
    state.ax = _u32(ptr)


def _op_free(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.ax = 0


def _op_mset(state: SymbolicProgramState, imm: int) -> None:
    del imm
    ptr = state.mem_read(state.sp + 16)
    value = state.mem_read(state.sp + 8) & 0xFF
    size = state.mem_read(state.sp)
    for i in range(size):
        state.mem_write(ptr + i, value)
    state.ax = ptr


def _op_mcmp(state: SymbolicProgramState, imm: int) -> None:
    del imm
    ptr1 = state.mem_read(state.sp + 16)
    ptr2 = state.mem_read(state.sp + 8)
    size = state.mem_read(state.sp)
    state.ax = 0
    for i in range(size):
        a = state.mem_read(ptr1 + i) & 0xFF
        b = state.mem_read(ptr2 + i) & 0xFF
        if a != b:
            state.ax = _u32(a - b)
            break


def _op_exit(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.halted = True


def _op_nop(state: SymbolicProgramState, imm: int) -> None:
    del state, imm


def _op_getchar(state: SymbolicProgramState, imm: int) -> None:
    del imm
    if state.stdin_pos < len(state.stdin_buf):
        state.ax = ord(state.stdin_buf[state.stdin_pos])
        state.stdin_pos += 1
    else:
        state.ax = _u32(-1)


def _op_putchar(state: SymbolicProgramState, imm: int) -> None:
    del imm
    state.output.append(chr(state.ax & 0xFF))


def _spec(
    opcode: int,
    name: str,
    reads: Iterable[str],
    writes: Iterable[str],
    effect: OpcodeEffect,
    section: str,
) -> DeclarativeOpcodeSpec:
    return DeclarativeOpcodeSpec(
        opcode=opcode,
        name=name,
        reads=frozenset(reads),
        writes=frozenset(writes),
        effect=effect,
        section=section,
    )


def build_default_opcode_declarations() -> Dict[int, DeclarativeOpcodeSpec]:
    """Return the declarative opcode table used by the symbolic backend."""

    section = "docs/OPCODE_TABLE.md"
    specs = [
        _spec(0, "LEA", {"BP", "IMM"}, {"AX"}, _op_lea, section),
        _spec(1, "IMM", {"IMM"}, {"AX"}, _op_imm, section),
        _spec(2, "JMP", {"IMM"}, {"PC"}, _op_jmp, section),
        _spec(3, "JSR", {"PC", "IMM", "SP"}, {"PC", "SP", "MEM"}, _op_jsr, section),
        _spec(4, "BZ", {"AX", "IMM"}, {"PC"}, _op_bz, section),
        _spec(5, "BNZ", {"AX", "IMM"}, {"PC"}, _op_bnz, section),
        _spec(6, "ENT", {"BP", "SP", "IMM"}, {"BP", "SP", "MEM"}, _op_ent, section),
        _spec(7, "ADJ", {"SP", "IMM"}, {"SP"}, _op_adj, section),
        _spec(8, "LEV", {"BP", "SP", "MEM"}, {"BP", "SP", "PC"}, _op_lev, section),
        _spec(9, "LI", {"AX", "MEM"}, {"AX"}, _op_li, section),
        _spec(10, "LC", {"AX", "MEM"}, {"AX"}, _op_lc, section),
        _spec(11, "SI", {"AX", "SP", "MEM"}, {"SP", "MEM"}, _op_si, section),
        _spec(12, "SC", {"AX", "SP", "MEM"}, {"SP", "MEM"}, _op_sc, section),
        _spec(13, "PSH", {"AX", "SP"}, {"SP", "MEM"}, _op_psh, section),
        _spec(30, "OPEN", {"SP", "MEM"}, {"AX"}, _op_open, section),
        _spec(31, "READ", {"SP", "MEM", "STDIN"}, {"AX", "MEM"}, _op_read, section),
        _spec(32, "CLOS", {"SP", "MEM"}, {"AX"}, _op_clos, section),
        _spec(33, "PRTF", {"SP", "MEM"}, {"AX", "STDOUT"}, _op_prtf, section),
        _spec(34, "MALC", {"SP", "MEM", "HEAP"}, {"AX", "HEAP"}, _op_malc, section),
        _spec(35, "FREE", {"SP", "MEM"}, {"AX"}, _op_free, section),
        _spec(36, "MSET", {"SP", "MEM"}, {"AX", "MEM"}, _op_mset, section),
        _spec(37, "MCMP", {"SP", "MEM"}, {"AX"}, _op_mcmp, section),
        _spec(38, "EXIT", {"AX"}, {"HALT"}, _op_exit, section),
        _spec(39, "NOP", set(), set(), _op_nop, section),
        _spec(64, "GETCHAR", {"STDIN"}, {"AX"}, _op_getchar, section),
        _spec(65, "PUTCHAR", {"AX"}, {"STDOUT"}, _op_putchar, section),
    ]
    binop_names = {
        14: "OR",
        15: "XOR",
        16: "AND",
        17: "EQ",
        18: "NE",
        19: "LT",
        20: "GT",
        21: "LE",
        22: "GE",
        23: "SHL",
        24: "SHR",
        25: "ADD",
        26: "SUB",
        27: "MUL",
        28: "DIV",
        29: "MOD",
    }
    for opcode, name in binop_names.items():
        specs.append(
            _spec(
                opcode,
                name,
                {"AX", "SP", "MEM"},
                {"AX", "SP"},
                _op_binop(opcode),
                section,
            )
        )
    return {spec.opcode: spec for spec in specs}


class SymbolicDeclarativeProgramRunner:
    """Run C4 bytecode by interpreting declarative opcode specs."""

    def __init__(
        self,
        opcode_specs: Optional[Mapping[int, DeclarativeOpcodeSpec]] = None,
    ):
        self.opcode_specs = dict(
            opcode_specs or build_default_opcode_declarations()
        )

    def init_state(
        self,
        bytecode: Sequence[int],
        data: Sequence[int] | bytes = (),
        *,
        stdin: str = "",
    ) -> SymbolicProgramState:
        state = SymbolicProgramState(code=list(bytecode), stdin_buf=stdin)
        state.load_data(data)
        return state

    def step(self, state: SymbolicProgramState) -> bool:
        if state.halted:
            return False

        if state.idx < len(state.code):
            instr = state.code[state.idx]
            opcode, imm = _decode_static_instruction(instr)
            idx_before = state.idx
            pc_before = state.pc
            state.idx += 1
            state.pc = idx_to_pc(state.idx)
        else:
            result = state.fetch_memory_instruction(state.pc)
            if result is None:
                return False
            opcode, imm = result
            idx_before = state.idx
            pc_before = state.pc
            state.pc = _u32(state.pc + 4)

        ax_before = state.ax
        sp_before = state.sp
        bp_before = state.bp
        state.last_mem_addr = 0
        state.last_mem_val = 0

        spec = self.opcode_specs.get(opcode)
        if spec is None:
            raise ValueError(f"unsupported opcode {opcode} at idx {idx_before}")
        spec.effect(state, imm)

        state.trace.append(
            SymbolicStepTrace(
                step=state.steps,
                opcode=opcode,
                name=spec.name,
                imm=imm,
                idx_before=idx_before,
                pc_before=pc_before,
                idx_after=state.idx,
                pc_after=state.pc,
                ax_before=ax_before,
                ax_after=state.ax,
                sp_before=sp_before,
                sp_after=state.sp,
                bp_before=bp_before,
                bp_after=state.bp,
                mem_addr=state.last_mem_addr,
                mem_value=state.last_mem_val,
                halted=state.halted,
            )
        )
        state.steps += 1
        return True

    def run(
        self,
        bytecode: Sequence[int],
        data: Sequence[int] | bytes = (),
        *,
        stdin: str = "",
        max_steps: Optional[int] = 2000,
    ) -> SymbolicProgramState:
        state = self.init_state(bytecode, data, stdin=stdin)
        while max_steps is None or state.steps < max_steps:
            if not self.step(state):
                break
            if state.halted:
                break
        return state


__all__ = [
    "DeclarativeOpcodeSpec",
    "SymbolicDeclarativeProgramRunner",
    "SymbolicProgramState",
    "SymbolicStepTrace",
    "build_default_opcode_declarations",
]
