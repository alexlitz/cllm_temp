"""End-to-end LIB-PROGRAM CORPUS: real C programs that exercise the C4 runtime
library (malloc / free / memset / memcmp) + file IO (OPEN/READ/CLOS) + printf,
with a compile -> run -> byte-exact-stdout harness.

WHAT THIS IS
============
Each corpus entry is a small, real C program (``c4_min/libprog/*.c``) that calls
the c4 runtime-library routines and/or the file-IO / printf tool-call opcodes.
The harness:

  1. COMPILES the .c to C4 bytecode with the repo's real compiler
     (:func:`src.compiler.compile_c`), which parses the C, and — because these
     programs call ``malloc`` / ``free`` / ``memset`` / ``memcmp`` — LINKS the
     C4 standard library (``src/stdlib/memory.c4``) so those functions are
     compiled from C into bytecode subroutines (NOT tool calls, per BLOG_SPEC
     §"Memory Allocation and Freeing" / §"Memset, Memcmp and Memcpy"). File ops
     (OPEN/READ/CLOS) and ``printf`` (PRTF) stay tool calls (§Tool Use Mode).
  2. RUNS the bytecode through a REFERENCE C4 VM (:class:`RefVM`) — a plain-python
     word-addressed interpreter that executes the full C4 ISA (JSR/ENT/LEV/ADJ,
     the ALU, LI/LC/SI/SC) so the linked malloc/memset/memcmp subroutines run
     *as bytecode*, and services the file / printf tool-call boundary (OPEN/READ/
     CLOS via a stub filesystem, PRTF via the c4 printf format subset).
  3. Asserts stdout == the GOLDEN captured from real ``gcc`` (a byte-exact oracle).

The GOLDEN for each program is captured by compiling the SAME .c with the system
``gcc`` (standard headers prepended, so ``malloc`` / ``printf`` resolve to libc)
and capturing its stdout.  ``libprog/GOLDENS.txt`` records the frozen goldens;
:func:`regenerate_goldens` rebuilds them.

TWO ENGINES (--engine reference|model)
======================================
  * ``reference`` (this file, available NOW): runs the bytecode on :class:`RefVM`.
  * ``model``     (PENDING on #646): once the runtime library is integrated into
    the unified neural model (``build_pure_forward_complete_model`` +
    ``nibble_runtime``), the SAME bytecode runs through ``model.forward`` and the
    same byte-exact stdout assertion applies.  :func:`run_model` is the hook; it
    reports PENDING until #646's ``c4_min.nibble_runtime`` (the lib) lands.  It is
    NOT faked.

The pytest wrapper is ``c4_min/test_libprog_corpus.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# The c4 printf format subset (%d %u %x %c %s %%) — reused so the reference VM's
# PRTF formats bytes identically to the tool-call runner.
from .nibble_filesys import _format_printf, _sign32

_HERE = Path(__file__).resolve().parent
LIBPROG_DIR = _HERE / "libprog"
GOLDENS_PATH = LIBPROG_DIR / "GOLDENS.txt"

# Instruction-index PC model (matches src.compiler's `current_addr = len(code)`
# branch/call targets).  Stack descends by SLOT bytes per push; data segment
# base is where the compiler places string literals.
SLOT = 8
STACK_TOP = 0x100000        # SP/BP init (well above the data seg + heap)
DATA_BASE = 0x10000         # src.compiler places string literals here


# ===========================================================================
# Opcode ids (src.compiler.Op numbering; identical to c4_min.isa for the shared
# subset).  Kept local so the harness does not depend on the src.Op enum import
# at module load (the enum is used only inside compile helpers).
# ===========================================================================
LEA, IMM, JMP, JSR, BZ, BNZ, ENT, ADJ, LEV = 0, 1, 2, 3, 4, 5, 6, 7, 8
LI, LC, SI, SC, PSH = 9, 10, 11, 12, 13
OR, XOR, AND = 14, 15, 16
EQ, NE, LT, GT, LE, GE = 17, 18, 19, 20, 21, 22
SHL, SHR = 23, 24
ADD, SUB, MUL, DIV, MOD = 25, 26, 27, 28, 29
OPEN, READ, CLOS, PRTF = 30, 31, 32, 33
EXIT, NOP = 38, 39


# ===========================================================================
# THE CORPUS: program -> (source file, lib functions exercised, what it shows).
# Keep programs SMALL so the neural VM run (Phase 2) stays tractable.
# ===========================================================================
@dataclass
class CorpusEntry:
    name: str
    source: str                       # .c filename under libprog/
    lib_funcs: Tuple[str, ...]        # runtime-lib / IO functions exercised
    demonstrates: str                 # the read/print capacity shown
    # per-program file inputs for OPEN/READ (name -> bytes) and stdin bytes.
    files: Dict[str, bytes] = field(default_factory=dict)
    stdin: bytes = b""


CORPUS: List[CorpusEntry] = [
    CorpusEntry(
        "memtest", "memtest.c",
        ("malloc", "memset", "memcmp", "printf"),
        "heap alloc -> fill -> byte-read-back print (%c loop) -> equal-buffer compare",
    ),
    CorpusEntry(
        "malloc_free_reuse", "malloc_free_reuse.c",
        ("malloc", "free", "memset", "printf"),
        "alloc/fill/print, free, re-alloc/fill/print + %d",
    ),
    CorpusEntry(
        "memset_fill", "memset_fill.c",
        ("malloc", "memset", "printf"),
        "memset fill then verify every byte via a %c read-back loop",
    ),
    CorpusEntry(
        "memcmp_eq", "memcmp_eq.c",
        ("malloc", "memset", "memcmp", "printf"),
        "memcmp of two identical buffers -> 0 (equal branch)",
    ),
    CorpusEntry(
        "memcmp_ne", "memcmp_ne.c",
        ("malloc", "memset", "memcmp", "printf"),
        "memcmp with a perturbed byte -> non-zero (differ branch)",
    ),
    CorpusEntry(
        "filecat", "filecat.c",
        ("malloc", "open", "read", "close", "printf"),
        "OPEN+READ+CLOS a file into a heap buffer, then print it (cat-style)",
        files={"greeting.txt": b"hello, file!\n"},
    ),
    CorpusEntry(
        "printf_int", "printf_int.c",
        ("printf",),
        "printf %d (signed decimal) + %c (char) formatting across calls",
    ),
    CorpusEntry(
        "printf_str", "printf_str.c",
        ("printf",),
        "printf %s (data-segment string pointer) + %d",
    ),
    CorpusEntry(
        "printf_hex", "printf_hex.c",
        ("printf",),
        "printf %x (lowercase hex) + %d",
    ),
    CorpusEntry(
        "malloc_printf", "malloc_printf.c",
        ("malloc", "memset", "printf"),
        "alloc+memset then report the fill via %d and %c reading a heap byte",
    ),
]

CORPUS_BY_NAME = {e.name: e for e in CORPUS}


# ===========================================================================
# COMPILE: .c -> C4 bytecode via the repo's real compiler (src.compiler).
# ===========================================================================
def _repo_root() -> Path:
    # c4_min/ lives at <root>/c4_release/c4_min ; src/ at <root>/c4_release/src
    return _HERE.parent


def compile_c_source(source: str) -> Tuple[List[Tuple[int, int]], bytes]:
    """Compile C ``source`` -> (instructions, data_bytes).

    Uses :func:`src.compiler.compile_c` (which links the C4 stdlib for
    malloc/free/memset/memcmp).  The compiler emits packed words ``op + (imm<<8)``;
    we unpack to ``(op, imm)`` instruction tuples (PC + branch targets are
    instruction indices).  ``data_bytes`` are the static data-segment bytes that
    load at :data:`DATA_BASE`.
    """
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.compiler import compile_c   # real C4 compiler (+ stdlib link)

    code_words, data = compile_c(source)
    instrs: List[Tuple[int, int]] = []
    for w in code_words:
        op = w & 0xFF
        imm = w >> 8
        # sign-extend a negative frame offset (LEA -8 etc.)
        if imm >= (1 << 55):
            imm -= (1 << 56)
        instrs.append((op, imm))
    return instrs, bytes(data)


def compile_entry(entry: CorpusEntry) -> Tuple[List[Tuple[int, int]], bytes]:
    """Compile a corpus entry's .c to (instructions, data_bytes)."""
    src = (LIBPROG_DIR / entry.source).read_text()
    return compile_c_source(src)


# ===========================================================================
# THE REFERENCE C4 VM — word-addressed, full ISA, tool-call IO.
#
# This is the byte-exact oracle for the LINKED-bytecode programs: it runs the
# compiled malloc/memset/memcmp subroutines AS BYTECODE (JSR/ENT/LEV + the ALU),
# and services the file / printf tool-call boundary.  PC and all branch/call
# targets are INSTRUCTION INDICES (src.compiler's native form).
# ===========================================================================
class _StubFS:
    """In-memory filesystem for OPEN/READ/CLOS: name -> bytes."""

    def __init__(self, files: Optional[Dict[str, bytes]] = None):
        self.files = dict(files or {})
        self._open: Dict[int, Tuple[bytes, int]] = {}   # fd -> (data, pos)
        self._next_fd = 3
        self.stdin = b""
        self.stdin_pos = 0

    def open(self, path: str) -> int:
        if path not in self.files:
            return -1
        fd = self._next_fd
        self._next_fd += 1
        self._open[fd] = (self.files[path], 0)
        return fd

    def read(self, fd: int, n: int) -> bytes:
        if fd == 0:                                    # stdin
            chunk = self.stdin[self.stdin_pos:self.stdin_pos + n]
            self.stdin_pos += len(chunk)
            return chunk
        st = self._open.get(fd)
        if st is None:
            return b""
        data, pos = st
        chunk = data[pos:pos + n]
        self._open[fd] = (data, pos + len(chunk))
        return chunk

    def close(self, fd: int) -> int:
        self._open.pop(fd, None)
        return 0


class RefVM:
    """Plain-python word-addressed C4 reference VM (the byte-exact oracle).

    Memory is a ``{byte_addr: byte}`` dict.  SI/LI are 32-bit word store/load;
    SC/LC are byte store/load (matching C ``char`` access and the neural LC/SC).
    The stack descends from :data:`STACK_TOP` by :data:`SLOT` per push.  File
    ops go through :class:`_StubFS`; PRTF formats via the c4 printf subset.
    """

    def __init__(self, instrs: List[Tuple[int, int]], data: bytes,
                 files: Optional[Dict[str, bytes]] = None, stdin: bytes = b""):
        self.code = instrs
        self.mem: Dict[int, int] = {}
        for i, b in enumerate(data):
            self.mem[DATA_BASE + i] = b
        self.fs = _StubFS(files)
        self.fs.stdin = stdin
        self.stdout = bytearray()
        self.ax = 0
        self.pc = 0
        self.sp = STACK_TOP
        self.bp = STACK_TOP

    # -- memory helpers ------------------------------------------------------
    def _lw(self, addr: int) -> int:
        return sum(self.mem.get(addr + i, 0) << (8 * i) for i in range(4))

    def _sw(self, addr: int, v: int) -> None:
        for i in range(4):
            self.mem[addr + i] = (v >> (8 * i)) & 0xFF

    def _cstr(self, addr: int, limit: int = 4096) -> str:
        out = bytearray()
        for i in range(limit):
            b = self.mem.get(addr + i, 0) & 0xFF
            if b == 0:
                break
            out.append(b)
        return out.decode("latin-1")

    def _push(self, v: int) -> None:
        self.sp -= SLOT
        self._sw(self.sp, v & 0xFFFFFFFF)

    def _pop(self) -> int:
        v = self._lw(self.sp)
        self.sp += SLOT
        return v

    # -- printf --------------------------------------------------------------
    def _do_printf(self, n_pushed: int) -> None:
        """PRTF: the format ptr was pushed FIRST, then each arg, so the stack is
        (top->down) last-arg ... first-arg, fmt_ptr.  ``n_pushed`` is the number of
        pushed slots (fmt + args), derived deterministically from the compiler's
        trailing ``ADJ (n_args+1)*8`` (see :meth:`_prtf_pushed`).  We format via the
        c4 subset, append to stdout, and set AX = n_written.  The compiler's ADJ
        reclaims the pushed slots — PRTF itself does not pop."""
        n_args = max(0, n_pushed - 1)
        fmt_ptr = self._lw(self.sp + n_args * SLOT)   # deepest slot = fmt
        fmt = self._cstr(fmt_ptr)
        args: List[int] = []
        for k in range(n_args):
            # arg k (call order) sits at depth (n_args-1-k) above fmt.
            args.append(self._lw(self.sp + (n_args - 1 - k) * SLOT))
        strings = {str(a): self._cstr(a) for a in args}
        text = _format_printf(fmt, args, strings)
        self.stdout.extend(text.encode("latin-1"))
        self.ax = len(text.encode("latin-1"))

    def _prtf_pushed(self, prtf_idx: int) -> int:
        """Number of slots pushed for the PRTF at instruction ``prtf_idx`` (fmt +
        args).  The compiler ALWAYS emits ``ADJ (n_args+1)*8`` immediately after a
        syscall, so the pushed-slot count is ``ADJ.imm / SLOT`` — a deterministic
        read, not a format-string guess.  Falls back to a format-string spec count
        if the next instruction is not an ADJ (defensive)."""
        nxt = self.code[prtf_idx + 1] if prtf_idx + 1 < len(self.code) else None
        if nxt is not None and nxt[0] == ADJ:
            return max(1, nxt[1] // SLOT)
        # defensive fallback: derive from the format string via a bounded scan.
        return _count_format_args(_peek_fmt(self)) + 1

    # -- run -----------------------------------------------------------------
    def run(self, max_steps: int = 200000) -> str:
        steps = 0
        while 0 <= self.pc < len(self.code) and steps < max_steps:
            steps += 1
            op, imm = self.code[self.pc]
            i = self.pc
            self.pc += 1
            if op == IMM:
                self.ax = imm & 0xFFFFFFFF
            elif op == LEA:
                # src.compiler emits LEA/ENT/ADJ immediates as BYTE offsets (frame
                # locals at -8, -16, ...; params at 16 + i*8), so imm is already a
                # byte offset — do NOT re-scale by SLOT.
                self.ax = (self.bp + imm) & 0xFFFFFFFF
            elif op == PSH:
                self._push(self.ax)
            elif op == ADD:
                self.ax = (self._pop() + self.ax) & 0xFFFFFFFF
            elif op == SUB:
                self.ax = (self._pop() - self.ax) & 0xFFFFFFFF
            elif op == MUL:
                self.ax = (self._pop() * self.ax) & 0xFFFFFFFF
            elif op == DIV:
                v = self._pop(); self.ax = (v // self.ax if self.ax else 0) & 0xFFFFFFFF
            elif op == MOD:
                v = self._pop(); self.ax = (v % self.ax if self.ax else 0) & 0xFFFFFFFF
            elif op == OR:
                self.ax = (self._pop() | self.ax) & 0xFFFFFFFF
            elif op == XOR:
                self.ax = (self._pop() ^ self.ax) & 0xFFFFFFFF
            elif op == AND:
                self.ax = (self._pop() & self.ax) & 0xFFFFFFFF
            elif op == SHL:
                self.ax = (self._pop() << self.ax) & 0xFFFFFFFF
            elif op == SHR:
                self.ax = (self._pop() >> self.ax) & 0xFFFFFFFF
            elif op in (EQ, NE, LT, GT, LE, GE):
                v = self._pop()
                r = {EQ: v == self.ax, NE: v != self.ax, LT: v < self.ax,
                     GT: v > self.ax, LE: v <= self.ax, GE: v >= self.ax}[op]
                self.ax = 1 if r else 0
            elif op == LI:
                self.ax = self._lw(self.ax)
            elif op == LC:
                self.ax = self.mem.get(self.ax, 0) & 0xFF
            elif op == SI:
                self._sw(self._pop(), self.ax)
            elif op == SC:
                self.mem[self._pop()] = self.ax & 0xFF
            elif op == JMP:
                self.pc = imm
            elif op == BZ:
                self.pc = imm if self.ax == 0 else self.pc
            elif op == BNZ:
                self.pc = imm if self.ax != 0 else self.pc
            elif op == JSR:
                self._push(i + 1)                      # return = next instr index
                self.pc = imm
            elif op == ENT:
                self._push(self.bp)                    # save caller BP
                self.bp = self.sp
                self.sp -= imm                         # reserve locals (BYTE size)
            elif op == ADJ:
                self.sp += imm                         # pop args (BYTE size)
            elif op == LEV:
                self.sp = self.bp
                self.bp = self._pop()                  # restore caller BP
                self.pc = self._pop()                  # restore return index
            elif op in (OPEN, READ, CLOS, PRTF):
                # Syscalls (§Tool Use Mode): args were pushed left-to-right, so the
                # stack (top->down) holds the LAST arg first.  Like PRTF, they PEEK
                # their args off the stack and do NOT pop — the compiler's trailing
                # ADJ reclaims the pushed slots.  For open(name,flags): top=flags,
                # then name.  For read(fd,buf,n): top=n, then buf, then fd.
                if op == OPEN:
                    name_ptr = self._lw(self.sp + 1 * SLOT)   # under flags
                    self.ax = self.fs.open(self._cstr(name_ptr)) & 0xFFFFFFFF
                elif op == READ:
                    n = self._lw(self.sp + 0 * SLOT)
                    buf = self._lw(self.sp + 1 * SLOT)
                    fd = self._lw(self.sp + 2 * SLOT)
                    chunk = self.fs.read(fd, n)
                    for k, b in enumerate(chunk):
                        self.mem[buf + k] = b
                    self.ax = len(chunk)
                elif op == CLOS:
                    fd = self._lw(self.sp + 0 * SLOT)
                    self.ax = self.fs.close(fd) & 0xFFFFFFFF
                else:  # PRTF
                    self._do_printf(self._prtf_pushed(i))
            elif op == NOP:
                pass
            elif op == EXIT:
                break
            else:
                raise NotImplementedError(f"op {op} not in RefVM ISA")
        return bytes(self.stdout).decode("latin-1")


def _peek_fmt(vm: "RefVM") -> str:
    """Read the format string for the pending PRTF WITHOUT knowing the arg count
    yet: the fmt ptr is the DEEPEST pushed slot, but we don't yet know the depth.
    We resolve it by scanning outward — the c4 lowering pushes fmt first, so the
    fmt ptr is the value at the largest SP offset that points at a valid C string
    whose %-spec count is self-consistent with that depth.  In practice the
    compiler always emits ``ADJ (n_args+1)`` so we instead derive n_args from the
    format found at each candidate depth; the smallest depth whose fmt's spec
    count == depth is the answer.  See :func:`_count_format_args`.
    """
    # Candidate depths 0..8 (printf in this corpus has <=3 args).  For each, read
    # the value at sp + d*SLOT as a would-be fmt ptr and count its specs; the
    # consistent depth is the one where spec-count == d.
    for d in range(0, 9):
        cand = vm._lw(vm.sp + d * SLOT)
        fmt = vm._cstr(cand)
        if fmt and _count_format_args(fmt) == d:
            return fmt
    # Fallback: depth 0 (no-arg printf) — fmt is the top of stack.
    return vm._cstr(vm._lw(vm.sp))


def _count_format_args(fmt: str) -> int:
    """Number of conversion args a c4 printf format string consumes (%% is not
    an arg; %d/%u/%x/%c/%s each consume one)."""
    n = 0
    i = 0
    while i < len(fmt):
        if fmt[i] == "%":
            i += 1
            if i < len(fmt) and fmt[i] != "%":
                n += 1
            i += 1
        else:
            i += 1
    return n


# ===========================================================================
# ENGINE: reference (now) | model (PENDING on #646).
# ===========================================================================
def run_reference(entry: CorpusEntry, max_steps: int = 200000) -> str:
    """Compile the entry and run its bytecode on :class:`RefVM`; return stdout."""
    instrs, data = compile_entry(entry)
    vm = RefVM(instrs, data, files=dict(entry.files), stdin=entry.stdin)
    return vm.run(max_steps=max_steps)


class ModelEnginePending(RuntimeError):
    """Raised when the neural (model.forward) engine is requested but the runtime
    library (#646, ``c4_min.nibble_runtime`` on the unified model) has not landed
    in this checkout yet."""


def lib_integrated() -> bool:
    """True iff #646's runtime library — AND a stdout-capable neural model build —
    are present in this checkout (i.e. the neural engine can actually run these
    stdout-producing programs).

    #646 (branch ``lib-into-unified-model``) lands ``c4_min.nibble_runtime`` (the
    baked malloc/free/memset/memcmp gadgets + the word-width oracle) and
    ``c4_min.lib_neural.build_lib_model`` (the 32-bit-load-address model build).
    We require BOTH modules before claiming the neural engine is runnable.
    """
    try:
        from . import nibble_runtime      # noqa: F401  (present only after #646)
        from . import lib_neural          # noqa: F401  (the 32-bit-addr model build)
    except Exception:
        return False
    return True


def _bytes_to_seg(data: bytes) -> Dict[int, int]:
    """Byte data-segment -> {byte_addr: byte} at :data:`DATA_BASE` (the form the
    pure-forward driver's ``data_seg`` expects)."""
    return {DATA_BASE + i: b for i, b in enumerate(data)}


def run_model(entry: CorpusEntry, max_steps: int = 4000) -> str:
    """PENDING on #646 — run the entry's bytecode THROUGH THE TRANSFORMER.

    Once #646 lands, this compiles the entry, builds the unified pure-forward
    model with the runtime library on it (``lib_neural.build_lib_model``), runs
    every VM step as one ``model.forward`` via ``run_pure_forward_complete`` with
    a file-op context (so OPEN/READ/CLOS/PRTF cross the tool boundary), and
    returns the PRTF stdout — the SAME byte-exact assertion as
    :func:`run_reference`.  Until then it raises :class:`ModelEnginePending` so
    callers report PENDING (never fake a pass).

    NOTE (honest scope): #646's landed neural tests exercise the lib on the
    AX-VALUE path (``run_pure_forward_cached`` at ``mask=0xFFFFFFFF``, checking the
    returned AX trace for heap word round-trips).  This corpus additionally needs
    the STDOUT path (PRTF + file IO via ``fio``).  The wiring below uses the
    stdout-capable ``run_pure_forward_complete`` driver; if #646's
    ``build_lib_model`` does not yet expose a PRTF-capable build, this raises
    :class:`ModelEnginePending` with that reason rather than silently passing.
    Memory: #646's dense build is ~62 GB before the sparse conversion — keep the
    corpus SMALL and build the model per-program on demand.
    """
    if not lib_integrated():
        raise ModelEnginePending(
            "neural engine PENDING: #646 (c4_min.nibble_runtime + "
            "c4_min.lib_neural on the unified model) is not in this checkout — "
            "run with --engine reference")
    from . import nibble_filesys as FS
    from .lib_neural import build_lib_model            # type: ignore
    from .nibble_pure_forward_complete import run_pure_forward_complete

    instrs, data = compile_entry(entry)
    try:
        model, L = build_lib_model(code_size=len(instrs) + 2)
    except TypeError as exc:  # signature drift once #646 lands
        raise ModelEnginePending(
            f"#646 build_lib_model signature differs ({exc}); wire the model "
            "builder args here") from exc
    fio = FS.FileOpState(runner=FS.FileRunner(
        fs=FS.StubFilesystem(dict(entry.files)),
        stdin=FS.InputKVStream(entry.stdin)))
    try:
        run_pure_forward_complete(
            model, L, instrs, max_steps=max_steps, mask=0xFFFFFFFF,
            fio=fio, data_seg=_bytes_to_seg(data))
    finally:
        import gc
        del model
        gc.collect()
    return bytes(fio.runner.stdout).decode("latin-1")


def run_engine(entry: CorpusEntry, engine: str = "reference", **kw) -> str:
    if engine == "reference":
        return run_reference(entry, **kw)
    if engine == "model":
        return run_model(entry, **kw)
    raise ValueError(f"unknown engine {engine!r} (want reference|model)")


# ===========================================================================
# GOLDEN generation + storage (real gcc, byte-exact oracle).
# ===========================================================================
_GCC_HEADERS = (
    "#include <stdlib.h>\n#include <string.h>\n#include <stdio.h>\n"
    "#include <fcntl.h>\n#include <unistd.h>\n"
)


def gcc_available() -> bool:
    from shutil import which
    return which("gcc") is not None


def gen_golden(entry: CorpusEntry, tmpdir: Optional[str] = None) -> bytes:
    """Compile the entry's .c with real gcc (stdlib headers prepended so libc
    resolves malloc/printf) and return its stdout bytes.  Files the program opens
    are materialised in the run cwd so gcc's ``open``/``read`` see the same bytes
    the c4 stub filesystem seeds.
    """
    import tempfile
    src = (LIBPROG_DIR / entry.source).read_text()
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        cfile = Path(d) / "prog.c"
        cfile.write_text(_GCC_HEADERS + src)
        binf = Path(d) / "prog"
        # -static avoids the -lgcc_s link issue in this sandbox.
        cp = subprocess.run(
            ["gcc", "-w", "-static", "-o", str(binf), str(cfile)],
            capture_output=True)
        if cp.returncode != 0:
            raise RuntimeError(f"gcc failed for {entry.name}:\n{cp.stderr.decode()}")
        # materialise the program's input files in the run cwd.
        for fname, data in entry.files.items():
            (Path(d) / fname).write_bytes(data)
        rp = subprocess.run([str(binf)], capture_output=True,
                            input=entry.stdin, cwd=d)
        return rp.stdout


def regenerate_goldens(tmpdir: Optional[str] = None) -> Dict[str, bytes]:
    """Rebuild every golden from gcc and write ``libprog/GOLDENS.txt`` (one
    ``name=<hex>`` line per program).  Returns {name: golden_bytes}."""
    if not gcc_available():
        raise RuntimeError("gcc not available — cannot regenerate goldens")
    goldens: Dict[str, bytes] = {}
    for e in CORPUS:
        goldens[e.name] = gen_golden(e, tmpdir=tmpdir)
    lines = [f"{name}={g.hex()}" for name, g in goldens.items()]
    GOLDENS_PATH.write_text("\n".join(lines) + "\n")
    return goldens


def load_goldens() -> Dict[str, bytes]:
    """Read the frozen goldens from ``libprog/GOLDENS.txt`` -> {name: bytes}."""
    if not GOLDENS_PATH.exists():
        return {}
    out: Dict[str, bytes] = {}
    for line in GOLDENS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        name, hexv = line.split("=", 1)
        out[name] = bytes.fromhex(hexv)
    return out


def golden_for(entry: CorpusEntry) -> bytes:
    """The byte-exact golden for ``entry``: prefer the frozen file, else gcc."""
    frozen = load_goldens()
    if entry.name in frozen:
        return frozen[entry.name]
    return gen_golden(entry)


# ===========================================================================
# CLI: regenerate goldens / run the corpus on an engine, print a PASS/FAIL table.
# ===========================================================================
def _run_table(engine: str) -> int:
    goldens = load_goldens()
    n_pass = n_fail = n_pending = 0
    print(f"{'PROGRAM':<20} {'LIB FUNCS':<34} {'ENGINE':<10} RESULT")
    print("-" * 90)
    for e in CORPUS:
        libs = ",".join(e.lib_funcs)
        want = goldens.get(e.name)
        if want is None:
            try:
                want = gen_golden(e)
            except Exception as exc:  # noqa: BLE001
                print(f"{e.name:<20} {libs:<34} {engine:<10} NO-GOLDEN ({exc})")
                n_fail += 1
                continue
        try:
            got = run_engine(e, engine=engine).encode("latin-1")
        except ModelEnginePending as exc:
            print(f"{e.name:<20} {libs:<34} {engine:<10} PENDING (#646)")
            n_pending += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{e.name:<20} {libs:<34} {engine:<10} ERROR: {exc!r}")
            n_fail += 1
            continue
        if got == want:
            print(f"{e.name:<20} {libs:<34} {engine:<10} PASS  {want!r}")
            n_pass += 1
        else:
            print(f"{e.name:<20} {libs:<34} {engine:<10} FAIL")
            print(f"    want={want!r}")
            print(f"    got ={got!r}")
            n_fail += 1
    print("-" * 90)
    print(f"{n_pass} pass / {n_fail} fail / {n_pending} pending  "
          f"({len(CORPUS)} programs, engine={engine})")
    return 1 if n_fail else 0


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="C4 lib-program corpus harness")
    ap.add_argument("--engine", choices=["reference", "model"], default="reference",
                    help="run bytecode on the reference VM (now) or the neural "
                         "model (PENDING on #646)")
    ap.add_argument("--regen-goldens", action="store_true",
                    help="rebuild libprog/GOLDENS.txt from gcc")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.regen_goldens:
        g = regenerate_goldens()
        print(f"wrote {len(g)} goldens -> {GOLDENS_PATH}")
        for name, gv in g.items():
            print(f"  {name:<20} {gv!r}")
        return 0
    return _run_table(args.engine)


if __name__ == "__main__":
    raise SystemExit(main())
