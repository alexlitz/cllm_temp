"""c4_min ARGV reader — argc/argv via the READ opcode + the neural-stdin path.

Per ``docs/BLOG_SPEC.md`` §"Reading Arguments" (751-793):

    The transformer also support argc and argv ... if argc and argv are used the
    caller writes a system prompt in the following format:

        | argc      | 4 bytes  | Little-endian uint32     |
        | argv[0]   | Variable | Null-terminated string   |
        | ...       | ...      | ...                      |
        | argv[n-1] | Variable | Null-terminated string   |

    Note that while this may appear like it is in some sort of equivalent of the
    stack calling convention it very much is not, these need to be **read in the
    same manner as user input is** and then saved to memory using the store
    instruction. So a subroutine has to be run ... in fact **they are read exactly
    as user input is.**

The blog then gives a ``__argv_setup`` C reference that reads that block with
``getchar()``. But GETCHAR has been intentionally REMOVED from the canonical
c4_min ISA (branch ``chk0`` / ``c4min-final``): the single-character neural
stdin/stdout opcodes are waived. What the blog PROSE actually mandates is that
argv is *"read exactly as user input is"* — i.e. through the SAME neural-stdin
mechanism the rest of the VM uses for input: the **READ opcode** over the
position-signature input-KV stream (§Tool Use Mode / §Printing and Reading
Input: "Input bytes are injected into the token stream between
USER_INPUT_START/END markers, and the VM reads them via attention").

So this module re-plumbs ``__argv_setup`` onto **READ** (op 31) instead of
GETCHAR:

    getchar()   ==>   READ(fd=0, scratch, 1) ; LC scratch     (one stdin byte)

READ(fd=0, ...) is the neural-stdin read: the runner (``nibble_filesys``) serves
it from the ``InputKVStream`` — the bytes injected between the input markers,
which is where the caller places the ARGV block. Each stdin byte re-enters VM
memory as a §Memory KV store frame, exactly as user input does, and is then
saved to the argv table / string buffer with SI/SC. No getchar, no new opcode —
the argv reader is a baked bytecode subroutine over the existing ISA + READ, and
"read exactly as user input is" is satisfied literally.

What's here
-----------
* :func:`argv_block`         — serialize [args] -> the ARGV system-prompt bytes
                               (argc uint32 LE + null-terminated strings).
* :func:`argv_stdin`         — the ``InputKVStream`` seeded with that block (the
                               neural-stdin buffer a READ(0,...) pulls from).
* :func:`emit_argv_setup`    — the ``__argv_setup`` bytecode (READ-based, NO
                               getchar), as ``isa.Instr`` for the model path.
* :func:`ref_argv_setup`     — a pure-python GOLDEN: the argv table + string
                               bytes ``__argv_setup`` must lay into memory (the
                               reference the model output is checked against).
* :func:`argv_getchar_block` — one-byte READ gadget (the getchar replacement).

The whole subroutine is proven two ways: the reference ISA VM
(:func:`c4_min.isa.interpret`, extended to serve READ from an ``InputKVStream``)
and the pure-forward transformer (``run_pure_forward_complete``), so there is a
golden AND a byte-exact neural validation (see ``test_argv_read.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from . import isa
from . import nibble_filesys as _FS


# ===========================================================================
# The ARGV system-prompt block (§"Reading Arguments" 753-761).
#
#   argc      : 4 bytes,  little-endian uint32
#   argv[0..]  : null-terminated strings, back to back
#
# This is what the caller writes to the input stream; the neural-stdin READ then
# pulls it out byte-for-byte, exactly as user input.
# ===========================================================================
Arg = Union[str, bytes]


def argv_block(args: List[Arg]) -> bytes:
    """Serialize ``args`` into the ARGV system-prompt block (the blog's format).

    ``argc`` is ``len(args)`` as a 4-byte little-endian uint32, followed by each
    ``argv[i]`` as a null-terminated string. Matches the table in §753-761.
    """
    argc = len(args)
    out = bytearray()
    out += argc.to_bytes(4, "little")             # argc : uint32 LE
    for a in args:
        b = a.encode("latin-1") if isinstance(a, str) else bytes(a)
        out += b
        out += b"\x00"                            # null terminator
    return bytes(out)


def argv_stdin(args: List[Arg]) -> "_FS.InputKVStream":
    """The neural-stdin buffer seeded with the ARGV block.

    A ``READ(fd=0, buf, n)`` in :func:`emit_argv_setup` reads from here — the
    input-KV bytes the caller injected between the USER_INPUT markers. This is
    the SAME stream a normal stdin ``read()`` pulls from (§Tool Use Mode); argv
    is read *exactly as user input is*.
    """
    return _FS.InputKVStream(argv_block(args))


# ===========================================================================
# Memory layout the argv reader lays out.  The pure-forward memory CAM keys a
# load (LI/LC/pop) on the LOW BYTE of the address (``compile_addr_expand``
# ``n_bits=8``), so every LIVE address the reader stores to / loads from must
# have a UNIQUE low byte.  We keep the whole argv region + scratch cells inside a
# single low 256-byte window, clear of the stack (which grows down from
# ``SP_INIT``).  These are byte addresses; each int cell is 4 bytes wide.
#
# The regions are DISJOINT (no low-byte alias between a scratch cell and an argv
# pointer-table slot): scratch cells occupy the lowest bytes, then the argv
# pointer table (``argv_base .. argv_base + argc*SLOT``), then the packed
# strings.  The default window sizes the table for up to ~7 args before it
# reaches ``string_base``; a caller with more args passes a wider ``string_base``.
# ===========================================================================
@dataclass(frozen=True)
class ArgvLayout:
    """Fixed byte addresses the argv reader uses. All in the low-byte window and
    mutually low-byte-distinct so the memory CAM resolves each unambiguously."""

    sc_argc: int = 0x04          # scratch: argc
    sc_i: int = 0x08             # scratch: loop index i
    sc_str: int = 0x0C           # scratch: current str_ptr
    sc_ch: int = 0x10            # scratch: current char / READ-landing byte
    read_buf: int = 0x14         # 1-byte READ landing buffer (the getchar cell)
    argv_base: int = 0x20        # argv[] pointer table (argc entries, SLOT bytes ea)
    string_base: int = 0x40      # the packed null-terminated string bytes


DEFAULT_LAYOUT = ArgvLayout()

#: c4/pure-forward int slot width (bytes). The argv table stores 4-byte pointers
#: and the stack/driver use 4-byte slots (see nibble_pure_forward_complete).
SLOT = 4


# ===========================================================================
# The bytecode assembler — mirrors the reference argv reader on ``c4min-final``
# (nibble_runtime.emit_argv_setup) but emits ``isa.Instr`` on the clean-room ISA
# and uses READ (op 31) where the reference used GETCHAR (op 64).
# ===========================================================================
_Instr = Union[Tuple[str, int], str]


class _Asm:
    """Tiny label-resolving assembler over ``isa.Instr`` (branch targets are
    instruction indices, matching :func:`isa.interpret`)."""

    def __init__(self) -> None:
        self.code: List[_Instr] = []
        self.labels: Dict[str, int] = {}

    def label(self, name: str) -> "_Asm":
        if name in self.labels:
            raise ValueError(f"duplicate label {name!r}")
        self.labels[name] = len(self.code)
        return self

    def op(self, name: str, imm: object = 0) -> "_Asm":
        self.code.append((name, imm))
        return self

    # readable wrappers
    def imm(self, v: int):   return self.op("IMM", v)
    def psh(self):           return self.op("PSH")
    def add(self):           return self.op("ADD")
    def sub(self):           return self.op("SUB")
    def mul(self):           return self.op("MUL")
    def li(self):            return self.op("LI")
    def lc(self):            return self.op("LC")
    def si(self):            return self.op("SI")
    def sc(self):            return self.op("SC")
    def lt(self):            return self.op("LT")
    def read1(self, buf: int):
        """One neural-stdin byte -> AX (the getchar replacement).

        ``READ(fd=0, buf, n=1)`` reads exactly one stdin byte into ``buf`` (fd 0
        is the input-KV / neural-stdin stream, §Tool Use Mode), then ``LC buf``
        loads that byte into AX. The whole gadget is the READ-based ``getchar()``:
        the blog's argv reader, with argv "read exactly as user input is."

        READ marshalling (``nibble_filesys.dispatch_file_op``): ``fd = pop()``,
        ``buf = pop()``, ``n = AX``. So push fd(0), push buf, set AX=1, READ.
        """
        self.imm(0).psh()          # fd = 0 (neural stdin)
        self.imm(buf).psh()        # buf
        self.imm(1)                # n = 1 (AX)
        self.op("READ")            # AX = read(0, buf, 1) = n_read (1)
        self.imm(buf).lc()         # AX = *(char*)buf  -- the stdin byte
        return self

    def jmp(self, lbl: str):  return self.op("JMP", lbl)
    def bz(self, lbl: str):   return self.op("BZ", lbl)
    def bnz(self, lbl: str):  return self.op("BNZ", lbl)
    def halt(self):           return self.op("HALT")

    def assemble(self) -> List[isa.Instr]:
        out: List[isa.Instr] = []
        for name, imm in self.code:
            if isinstance(imm, str):
                if imm not in self.labels:
                    raise KeyError(f"unresolved label {imm!r}")
                imm = self.labels[imm]
            out.append(isa.Instr(isa.BY_NAME[name], int(imm) & 0xFFFFFFFF))
        return out


def _store(a: _Asm, addr: int, value_fn, *, byte: bool = False) -> None:
    """``*(addr) = value`` — the c4 store idiom: push ADDRESS, produce the value
    into AX stack-neutrally, then SI (word) / SC (byte)."""
    a.imm(addr).psh()
    value_fn(a)
    a.op("SC" if byte else "SI")


def _load(a: _Asm, addr: int) -> None:
    """AX = ``*(addr)`` (word)."""
    a.imm(addr).li()


def _add_const(a: _Asm, addr: int, delta: int) -> None:
    """``*(addr) += delta`` (word)."""
    _store(a, addr, lambda a: (_load(a, addr), a.psh(), a.imm(delta), a.add()))


def emit_argv_setup(layout: ArgvLayout = DEFAULT_LAYOUT) -> _Asm:
    """The ``__argv_setup`` subroutine (§765-791), re-plumbed onto **READ**.

    This is a direct bytecode translation of the spec's C, with every
    ``getchar()`` replaced by the READ-based one-byte neural-stdin read
    (:meth:`_Asm.read1`) — argv "read exactly as user input is", but on the
    canonical (getchar-free) ISA. On completion:

      * ``sc_argc``                 holds argc,
      * ``argv_base + i*4`` (i<argc) holds the pointer to argv[i]'s bytes,
      * ``string_base ...``         holds each argv[i] as a null-terminated string.

    The subroutine ends by leaving argc in AX and HALTing, so a standalone proof
    can read argc back; when used as a prologue it is concatenated with the user
    body (see :func:`emit_argv_program`) and the trailing HALT is dropped.
    """
    L = layout
    a = _Asm()
    # argc = read() + read()*256 + read()*65536 + read()*16777216  (LE uint32)
    _store(a, L.sc_argc, lambda a: (
        a.read1(L.read_buf)
         .psh().read1(L.read_buf).psh().imm(256).mul().add()
         .psh().read1(L.read_buf).psh().imm(65536).mul().add()
         .psh().read1(L.read_buf).psh().imm(16777216).mul().add()))
    # str_ptr = string_base
    _store(a, L.sc_str, lambda a: a.imm(L.string_base))
    # i = 0
    _store(a, L.sc_i, lambda a: a.imm(0))
    a.label("outer")
    # while (i < argc)
    _load(a, L.sc_i); a.psh(); _load(a, L.sc_argc); a.lt()   # AX = (i < argc)
    a.bz("end")
    # *(int*)(argv_base + i*SLOT) = str_ptr
    a.imm(L.argv_base).psh()                                 # [addr_lo]
    _load(a, L.sc_i); a.psh(); a.imm(SLOT).mul(); a.add()    # AX = argv_base + i*SLOT
    a.psh()                                                  # [addr]
    _load(a, L.sc_str)                                       # AX = str_ptr
    a.si()                                                   # *(argv_base+i*SLOT) = str_ptr
    # ch = read()
    _store(a, L.sc_ch, lambda a: a.read1(L.read_buf))
    a.label("inner")
    _load(a, L.sc_ch); a.bz("inner_done")                    # while (ch)
    # *(char*)str_ptr = ch
    _load(a, L.sc_str); a.psh()                              # [str_ptr]
    _load(a, L.sc_ch); a.sc()                                # *(str_ptr) = ch
    _add_const(a, L.sc_str, 1)                               # str_ptr += 1
    _store(a, L.sc_ch, lambda a: a.read1(L.read_buf))        # ch = read()
    a.jmp("inner")
    a.label("inner_done")
    # *(char*)str_ptr = 0
    _load(a, L.sc_str); a.psh(); a.imm(0).sc()
    _add_const(a, L.sc_str, 1)                               # str_ptr += 1
    _add_const(a, L.sc_i, 1)                                 # i += 1
    a.jmp("outer")
    a.label("end")
    _load(a, L.sc_argc)                                      # AX = argc
    a.halt()
    return a


def emit_argv_program(body: List[Tuple[str, int]],
                      layout: ArgvLayout = DEFAULT_LAYOUT) -> List[isa.Instr]:
    """Prepend the READ-based argv prologue to a user ``body``.

    The argv reader runs first (populating the argv table + strings from the
    neural-stdin ARGV block), then control falls into ``body`` (which reads argv
    with LI/LC and computes its result). The prologue's trailing HALT is dropped
    so control flows into the body; the body must HALT.  ``body`` is a list of
    ``(op_name, imm)`` tuples on the same clean-room ISA. Branch targets in the
    body (JMP/BZ/BNZ) are absolute instruction indices; they are RE-BASED by the
    prologue length so they stay correct after the prologue is prepended.
    """
    prologue = emit_argv_setup(layout).assemble()
    assert prologue[-1].op == isa.HALT
    prologue = prologue[:-1]                                 # drop the reader's HALT
    base = len(prologue)
    body_code = isa.assemble(body)
    rebased: List[isa.Instr] = []
    for ins in body_code:
        if ins.op in (isa.JMP, isa.BZ, isa.BNZ):
            rebased.append(isa.Instr(ins.op, (ins.imm + base) & 0xFFFFFFFF))
        else:
            rebased.append(isa.Instr(ins.op, ins.imm))
    return prologue + rebased


# ===========================================================================
# The reference GOLDEN — the memory image ``__argv_setup`` must produce.  This is
# the value-faithful oracle the model output is checked against (it does NOT run
# bytecode; it computes the intended argv-table + string layout directly from the
# ARGV block, so a bug in the bytecode is caught by disagreement with THIS).
# ===========================================================================
@dataclass
class ArgvImage:
    """The intended post-``__argv_setup`` memory state."""

    argc: int
    argv_ptrs: List[int]                 # argv[i] pointer stored at argv_base+i*SLOT
    mem: Dict[int, int] = field(default_factory=dict)   # byte addr -> byte value

    def argv(self, i: int) -> bytes:
        """The bytes of argv[i] (up to its null terminator), from ``mem``."""
        p = self.argv_ptrs[i]
        out = bytearray()
        while self.mem.get(p, 0) != 0:
            out.append(self.mem[p])
            p += 1
        return bytes(out)


def ref_argv_setup(args: List[Arg], layout: ArgvLayout = DEFAULT_LAYOUT) -> ArgvImage:
    """Compute the golden argv memory image the reader must lay out for ``args``.

    Mirrors the C in §765-791 exactly (argv_base pointer table + packed
    null-terminated strings at string_base), independent of any bytecode. The
    model / reference-VM memory after running :func:`emit_argv_setup` must match
    this byte-for-byte at the argv table and string region.
    """
    L = layout
    argc = len(args)
    mem: Dict[int, int] = {}
    argv_ptrs: List[int] = []
    str_ptr = L.string_base
    for i, a in enumerate(args):
        b = a.encode("latin-1") if isinstance(a, str) else bytes(a)
        argv_ptrs.append(str_ptr)
        # store the pointer to argv[i] at argv_base + i*SLOT (little-endian word)
        addr = L.argv_base + i * SLOT
        for k in range(SLOT):
            mem[addr + k] = (str_ptr >> (8 * k)) & 0xFF
        for ch in b:                          # the string bytes
            mem[str_ptr] = ch & 0xFF
            str_ptr += 1
        mem[str_ptr] = 0                      # null terminator
        str_ptr += 1
    # argc also lives in its scratch cell (the reader stores it there first)
    for k in range(SLOT):
        mem[L.sc_argc + k] = (argc >> (8 * k)) & 0xFF
    return ArgvImage(argc=argc, argv_ptrs=argv_ptrs, mem=mem)


# ===========================================================================
# Convenience: a self-contained program that reads argv then computes a value the
# caller can check (used by the byte-exact neural validation).
# ===========================================================================
def emit_print_argv_i_char0(i: int, layout: ArgvLayout = DEFAULT_LAYOUT
                            ) -> List[isa.Instr]:
    """argv reader + body ``return argv[i][0]`` (the first char of argv[i]).

    C equivalent:  ``int main(int argc,char**argv){ return argv[i][0]; }``
    The body: AX = *(argv_base + i*SLOT)  (the pointer), then LC that pointer.
    """
    L = layout
    body = [
        ("IMM", L.argv_base + i * SLOT), ("LI", 0),   # AX = argv[i] (pointer)
        ("LC", 0),                                    # AX = *argv[i] = argv[i][0]
        ("HALT", 0),
    ]
    return emit_argv_program(body, layout)


def emit_return_argc(layout: ArgvLayout = DEFAULT_LAYOUT) -> List[isa.Instr]:
    """argv reader + body ``return argc``.  C equivalent:
    ``int main(int argc,char**argv){ return argc; }``."""
    return emit_argv_setup(layout).assemble()   # already ends AX=argc; HALT


def emit_read_nth_stdin_byte(n: int, buf: int = 0x40) -> List[isa.Instr]:
    """A COMPACT (straight-line, no-loop) neural proof of the READ-based argv path.

    Reads and DISCARDS the first ``n`` bytes of the ARGV/stdin block, then reads
    the ``n``-th byte into ``buf`` via ``READ(fd=0, buf, 1)`` and returns it
    (``LC buf``).  This exercises the EXACT neural mechanism the full
    ``__argv_setup`` uses — the READ opcode serviced from the position-signature
    ``InputKVStream``, the read byte re-entering VM memory as a §Memory KV frame,
    and an ``LC`` reading it back — but as ~``5*(n+1)`` straight-line instructions,
    so it fits a SMALL ``code_size`` model (no 145-instruction loop reader).

    For the ARGV block of ``args``, byte ``n`` picks a known target: bytes 0-3 are
    ``argc`` (LE uint32), byte 4 is ``argv[0][0]``, etc.  The caller checks the
    returned byte against the known ARGV-block byte (== gcc's view of that byte).
    """
    prog: List[Tuple[str, int]] = []
    for _ in range(n + 1):                       # read n discards + 1 kept byte
        prog += [("IMM", 0), ("PSH", 0),         # fd = 0 (neural stdin)
                 ("IMM", buf), ("PSH", 0),        # buf
                 ("IMM", 1), ("READ", 0)]         # AX = read(0, buf, 1)
    prog += [("IMM", buf), ("LC", 0),            # AX = *(char*)buf = the n-th byte
             ("HALT", 0)]
    return isa.assemble(prog)


__all__ = [
    "Arg", "argv_block", "argv_stdin", "ArgvLayout", "DEFAULT_LAYOUT", "SLOT",
    "emit_argv_setup", "emit_argv_program", "emit_print_argv_i_char0",
    "emit_return_argc", "emit_read_nth_stdin_byte", "ArgvImage", "ref_argv_setup",
]
