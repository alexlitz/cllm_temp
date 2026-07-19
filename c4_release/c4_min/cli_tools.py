"""The classic streaming CLI programs (echo / cat / yes) run THROUGH THE C4 VM
transformer via the tool-use I/O path (READ / PRTF).

The reference ``*_cllm.c`` programs (from the c-runtime branch) are written with
``getchar`` / ``putchar``.  The small (Qwen-0.5B-scale) model in this branch has
the *neural* PUTCHAR / GETCHAR opcodes REMOVED — the only I/O boundary left is the
§Tool Use Mode tool-call path (OPEN / READ / CLOS / PRTF, ``nibble_filesys``).  So
these programs are re-expressed in that I/O model:

    echo   putchar('h')…putchar(10)          -> PRTF("hello\n")           (fixed str)
    cat    c=getchar(); while(c>=0) putchar   -> READ(0,buf,n); PRTF(buf)  (echo input)
    yes    while(1){putchar('y');putchar(10)} -> PRTF("y\n") ×N            (bounded)

Each is one turn of the SAME driver ELIZA uses: the program runs whole through
``run_pure_forward_complete`` (every VM step is one ``model.forward``), input
enters via the input-KV (READ fd 0), output leaves via PRTF's stdout.  The SAME
bytecode also runs on a plain-python C4 reference so every result is byte-exact.

``cat`` copies stdin to stdout: it READs the user line into a low-window buffer,
NUL-terminates it, and PRTFs that buffer as a format string — the input bytes are
loaded back out of the §Memory KV log (the READ byte-stores) and formatted to
stdout, so the round trip (input-KV -> VM memory -> stdout) is entirely through
the transformer.  (The format-string print requires the input to contain no ``%``;
a real cat would loop char-by-char, but that is the O(n²)-token step-count wall the
minimal chat note describes, so the buffer-print form is used for the proof.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import isa
from . import nibble_filesys as FS
from .chat_eliza import asm, run_eliza_reference

# Low-256 window layout (8-bit pointers): scratch below, buffer, then data.
BUF = 0x08                 # cat's stdin read buffer
_DATA_BASE = 0x20          # format-string / literal data segment
CAT_READ_N = 23            # cat reads up to 23 bytes/turn (BUF window 0x08..0x1F)


@dataclass
class Program:
    """A built CLI program: bytecode + its seeded data segment + a label."""
    name: str
    code: List[isa.Instr]
    data_seg: Dict[int, int]


def _c(seg: Dict[int, int], addr: int, s) -> int:
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        seg[addr + i] = b
    seg[addr + len(s)] = 0
    return len(s) + 1


# ---------------------------------------------------------------------------
# echo — print a fixed string.  putchar('h')…putchar(10)  ->  PRTF("hello\n").
# ---------------------------------------------------------------------------
def build_echo(text: str = "hello\n") -> Program:
    seg: Dict[int, int] = {}
    _c(seg, _DATA_BASE, text)
    prog = [("IMM", _DATA_BASE), ("PSH", 0), ("PRTF", 0), ("HALT", 0)]
    return Program("echo", asm(prog), seg)


# ---------------------------------------------------------------------------
# yes — print a short string N times.  while(1){putchar('y');putchar(10)} bounded.
# ---------------------------------------------------------------------------
def build_yes(text: str = "y\n", n: int = 3) -> Program:
    """``yes`` outputs ``text`` forever; the demo bounds it to ``n`` repetitions
    (a counted loop) so it terminates.  This exercises a genuine JMP/BZ loop with a
    PRTF body — the streaming-output half of a conversational program.
    """
    seg: Dict[int, int] = {}
    _c(seg, _DATA_BASE, text)
    I_CELL = 0x00
    prog: List[Tuple] = []
    # i = n
    prog += [("IMM", n), ("PSH", 0), ("IMM", I_CELL), ("PSH", 0)]  # (unused push tidy)
    # store i = n:  addr=I_CELL ; PSH ; val=n ; SI
    prog = [("IMM", I_CELL), ("PSH", 0), ("IMM", n), ("SI", 0)]
    prog += [("LABEL", "loop")]
    # if i == 0 -> done
    prog += [("IMM", I_CELL), ("LI", 0), ("BZ", "done")]
    # PRTF(text)
    prog += [("IMM", _DATA_BASE), ("PSH", 0), ("PRTF", 0)]
    # i = i - 1:  addr=I_CELL ; PSH ; val = i-1 ; SI
    prog += [("IMM", I_CELL), ("PSH", 0),
             ("IMM", I_CELL), ("LI", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),
             ("SI", 0)]
    prog += [("JMP", "loop")]
    prog += [("LABEL", "done"), ("HALT", 0)]
    return Program("yes", asm(prog), seg)


# ---------------------------------------------------------------------------
# cat — copy stdin to stdout.  READ the line into BUF, NUL-terminate, PRTF(BUF).
# ---------------------------------------------------------------------------
def build_cat(n_read: int = CAT_READ_N) -> Program:
    seg: Dict[int, int] = {}   # cat has no static data — the "format string" IS the
    # read-back input buffer, which lives in the KV store log, not the data seg.
    prog: List[Tuple] = []
    prog += [("IMM", 0), ("PSH", 0)]            # fd = 0
    prog += [("IMM", BUF), ("PSH", 0)]          # buf
    prog += [("IMM", n_read)]                   # n
    prog += [("READ", 0)]                       # AX = n_read
    # NUL-terminate: BUF[n_read] = 0
    prog += [("PSH", 0)]                        # save n_read
    prog += [("IMM", BUF), ("ADD", 0)]          # AX = BUF + n_read
    prog += [("PSH", 0), ("IMM", 0), ("SI", 0)]  # BUF[n_read] = 0
    # PRTF(BUF): the input line is the format string -> echoes it to stdout.
    prog += [("IMM", BUF), ("PSH", 0), ("PRTF", 0)]
    prog += [("HALT", 0)]
    return Program("cat", asm(prog), seg)


# ===========================================================================
# Runners — the SAME two paths the chat driver uses.
# ===========================================================================
def _fresh_fio(stdin_bytes: bytes = b"") -> FS.FileOpState:
    return FS.FileOpState(
        runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                             stdin=FS.InputKVStream(stdin_bytes)))


def run_ref(prog: Program, stdin_bytes: bytes = b"", max_steps: int = 20000) -> str:
    """Run ``prog`` on the plain-python C4 reference (byte-exact oracle)."""
    fio = _fresh_fio(stdin_bytes)
    return run_eliza_reference(prog.code, dict(prog.data_seg), fio,
                               max_steps=max_steps)


def run_model(model, L, prog: Program, stdin_bytes: bytes = b"",
              max_steps: int = 200) -> str:
    """Run ``prog`` THROUGH THE TRANSFORMER (every VM step is one model.forward)."""
    from .nibble_pure_forward_complete import run_pure_forward_complete
    fio = _fresh_fio(stdin_bytes)
    run_pure_forward_complete(
        model, L, prog.code, max_steps=max_steps, mask=0xFF,
        fio=fio, data_seg=dict(prog.data_seg))
    return bytes(fio.runner.stdout).decode("latin-1")
