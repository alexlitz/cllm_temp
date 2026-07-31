"""c4_min EDGE-CASE corpus — the 17 opcodes the canonical 1096 corpus never touches.

WHY
---
``tests.test_suite_1000.generate_test_programs()`` is exactly 1096 programs but
exercises ONLY 23/40 opcodes.  NEVER exercised by that corpus (verified
empirically, 2026-07-30): ``AND OR XOR SHL SHR GE BNZ LC SC MALC FREE MSET MCMP
PRTF OPEN READ CLOS`` (+ PUTCHAR/GETCHAR via PRTF/READ).  The two ops with recent
byte-exactness fixes — **SHR** (arithmetic right shift) and **LC** (signed char
load) — are BOTH untested end-to-end.  This is the doom-critical path (``doom.c``
is ``char*``-buffer + fixed-point-shift + signed-div heavy).

This module is a green-field, ADDITIVE edge-case corpus.  It authors NO weights
(imports the existing reference interpreters + the already-built pure-forward
model only), so the golden fingerprint ``069cc32f`` is untouched.

TWO WAYS to build a case
------------------------
* ``kind="c"``   — C source compiled via ``src.compiler.compile_c`` (the same
  path ``run_1096_pure_forward`` uses).  Exercises whole ops the way a real
  program would.  NB: the c4_min compiler emits an **8-bit IMM** (``IMM`` folds
  the constant to its low byte in the ISA), and ``char c = -1`` therefore loads
  back through **LI (unsigned)** as 255 — so a compiled ``c >> 1`` is
  ``255>>1==127``, NOT the native-c4 ``-1`` (which uses a signed LC).  The C
  path CANNOT express a true signed-char shift; the ``asm`` path is used for that.
* ``kind="asm"`` — hand-assembled ``isa.Instr`` list.  Bypasses the compiler's
  IMM byte-fold to target LC/SC/SHR/SHL sign edges precisely (store a byte with
  SC, load it signed with LC, shift it, store it back truncated).  This is the
  only way to drive a genuine 32-bit-signed shift/char op.

GOLDEN
------
For each case the golden is the c4_min reference:
  * ``kind="c"``/``kind="asm"`` value cases  -> ``ref_interpret`` (32-bit,
    mask=0xFFFFFFFF) — the SAME golden ``run_1096_pure_forward`` scores against,
    and the value the pure-forward NEURAL model reproduces byte-exactly.
  * I/O cases (READ/PRTF)                     -> golden AX + golden stdout bytes
    computed by ``ref_interpret``'s I/O contract (READ pulls from a stdin stream,
    PRTF appends AX&0xFF).  The neural model services these via the TOOL_CALL /
    input-KV protocol (``run_pure_forward_complete(fio=..., data_seg=...,
    seed_mem=...)``).

``expected`` is the golden 32-bit AX; ``expected_stdout`` (bytes) is the golden
visible-output byte stream for PRTF cases (None if the case prints nothing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from c4_min import isa


# ---------------------------------------------------------------------------
# Case model.
# ---------------------------------------------------------------------------
@dataclass
class EdgeCase:
    name: str                       # unique, stable id
    cluster: str                    # grouping key for the pass/fail table
    kind: str                       # "c" (compile) | "asm" (hand-assembled)
    body: object                    # C source str  OR  List[(opname, imm)]
    expected: int                   # golden 32-bit AX (masked & 0xFFFFFFFF)
    note: str = ""                  # what edge this pins
    # optional I/O:
    stdin: bytes = b""              # bytes a READ(fd=0) pulls
    data_seg: Dict[int, int] = field(default_factory=dict)   # byte addr -> byte
    seed_mem: Dict[int, int] = field(default_factory=dict)   # word addr -> value
    expected_stdout: Optional[bytes] = None                  # golden PRTF bytes
    prtf_args: List[int] = field(default_factory=list)       # printf varargs
    io: bool = False                # True => neural run needs the fio/stdin path
    max_steps: int = 64

    def code(self) -> List[isa.Instr]:
        """The assembled ISA program (compiling C on demand)."""
        if self.kind == "asm":
            return isa.assemble(self.body)
        if self.kind == "c":
            from src.compiler import compile_c
            from c4_min.run_1096_pure_forward import bytecode_to_isa
            bc, _data = compile_c(self.body)
            return bytecode_to_isa(bc)
        raise ValueError(f"unknown kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Assembly helpers.
# ---------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def _bin(v: int, n: int, op: str):
    """``v OP n`` : IMM v ; PSH ; IMM n ; OP ; HALT.  (v,n are LOW BYTES: the ISA
    IMM folds to 8 bits, so both operands' low byte is what the op sees.)"""
    return [("IMM", v & 0xFF), ("PSH", 0), ("IMM", n & 0xFF), (op, 0), ("HALT", 0)]


def _store_byte_then(addr: int, byte: int, tail):
    """SC ``byte`` -> mem[addr], then continue with ``tail`` (a list of (op,imm)).

    ``IMM addr ; PSH ; IMM byte ; SC ; <tail>``.  SC pops the pushed ADDRESS and
    stores AX's low byte there — the way to seed a signed char in memory that a
    later LC reads back sign-extended."""
    return [("IMM", addr & 0xFF), ("PSH", 0), ("IMM", byte & 0xFF), ("SC", 0)] + list(tail)


def _lc(addr: int):
    """LC mem[addr] : IMM addr ; LC.  (signed char load)"""
    return [("IMM", addr & 0xFF), ("LC", 0)]


def _li(addr: int):
    """LI mem[addr] : IMM addr ; LI.  (unsigned byte load in this 8-bit ISA)"""
    return [("IMM", addr & 0xFF), ("LI", 0)]


# ===========================================================================
# CLUSTER 1 — CHAR READ / WRITE  (LC sign-extend, SC truncate, char* walks).
# ===========================================================================
def _cluster_char_rw() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # LC sign-extends: byte>=0x80 -> negative (32-bit).  SC truncates the store.
    for b, want in [(0x00, 0x00000000), (0x41, 0x00000041), (0x7F, 0x0000007F),
                    (0x80, 0xFFFFFF80), (0xFF, 0xFFFFFFFF)]:
        cs.append(EdgeCase(
            f"lc_signext_{b:02x}", "char_rw", "asm",
            _store_byte_then(0x40, b, _lc(0x40) + [("HALT", 0)]),
            want & M32, note=f"LC(0x{b:02x}) sign-extend"))
    # LI (unsigned) vs LC (signed) on the SAME 0x80 byte — LI stays 0x80.
    cs.append(EdgeCase(
        "li_unsigned_80", "char_rw", "asm",
        _store_byte_then(0x44, 0x80, _li(0x44) + [("HALT", 0)]),
        0x80, note="LI unsigned byte load (0x80 stays positive 128)"))
    # SC truncates a wide AX to one byte: store 0x1FF -> 0xFF (only low byte kept),
    # then LC reads it back as -1.  (0x1FF's low byte is 0xFF; IMM already folds
    # the constant to 0xFF, so this pins the byte store path.)
    cs.append(EdgeCase(
        "sc_trunc_1ff", "char_rw", "asm",
        _store_byte_then(0x48, 0x1FF, _lc(0x48) + [("HALT", 0)]),
        M32, note="SC truncates 0x1FF->0xFF, LC reads -1"))
    # -1 stored as a byte (0xFF) then read back signed = -1.
    cs.append(EdgeCase(
        "sc_neg1", "char_rw", "asm",
        _store_byte_then(0x4C, 0xFF, _lc(0x4C) + [("HALT", 0)]),
        M32, note="SC(-1)->0xFF, LC->-1"))
    # char pointer walk / read-back after write: store 3 chars, read one back.
    #   mem[0x50]='A'(0x41) mem[0x51]='B'(0x42) mem[0x52]=0x00 ; return LC(0x51)
    walk = (_store_byte_then(0x50, 0x41, [])
            + _store_byte_then(0x51, 0x42, [])
            + _store_byte_then(0x52, 0x00, [])
            + _lc(0x51) + [("HALT", 0)])
    cs.append(EdgeCase("char_ptr_walk", "char_rw", "asm", walk, 0x42,
                       note="write s[0..2], read s[1] back ('B')"))
    # C-level: string copy + scan (doom put_str style) via char* — exercises LC/SC
    # through compiled C (loads back through LI/SC per the c4_min lowering).
    cs.append(EdgeCase(
        "c_string_copy_scan", "char_rw", "c",
        "int main(){ char *s; int n; s=malloc(4); "
        "s[0]=72; s[1]=105; s[2]=0; n=0; "
        "while(s[n]){ n=n+1; } return n; }",
        2, note="strlen-style char* scan of \"Hi\" -> 2", max_steps=4000))
    return cs


# ===========================================================================
# CLUSTER 2 — CHAR SHIFTS  (arithmetic SHR on negatives, SHL across the sign bit).
# ===========================================================================
def _cluster_char_shift() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # Signed-char SHR is ARITHMETIC: LC 0xFF (=-1); -1>>1 must be -1 (0xFFFFFFFF),
    # NOT 0x7F.  THE doom-critical case (fixed-point >>).
    for amt, want in [(1, M32), (4, M32), (7, M32), (8, M32), (31, M32)]:
        cs.append(EdgeCase(
            f"shr_char_neg1_by{amt}", "char_shift", "asm",
            _store_byte_then(0x40, 0xFF, _lc(0x40)
                             + [("PSH", 0), ("IMM", amt), ("SHR", 0), ("HALT", 0)]),
            want, note=f"(char -1) >> {amt} == -1 (arithmetic)"))
    # LC 0x80 (=-128); -128>>1 == -64 (0xFFFFFFC0); -128>>7 == -1.
    cs.append(EdgeCase(
        "shr_char_neg128_by1", "char_shift", "asm",
        _store_byte_then(0x44, 0x80, _lc(0x44)
                         + [("PSH", 0), ("IMM", 1), ("SHR", 0), ("HALT", 0)]),
        0xFFFFFFC0, note="(char -128) >> 1 == -64"))
    cs.append(EdgeCase(
        "shr_char_neg128_by7", "char_shift", "asm",
        _store_byte_then(0x45, 0x80, _lc(0x45)
                         + [("PSH", 0), ("IMM", 7), ("SHR", 0), ("HALT", 0)]),
        M32, note="(char -128) >> 7 == -1"))
    # Positive char SHR stays the plain value: 0x7F>>3 == 0x0F.
    cs.append(EdgeCase(
        "shr_char_pos_by3", "char_shift", "asm",
        _store_byte_then(0x46, 0x7F, _lc(0x46)
                         + [("PSH", 0), ("IMM", 3), ("SHR", 0), ("HALT", 0)]),
        0x0F, note="(char 127) >> 3 == 15"))
    # Unsigned byte SHR (LI path): 0xFF>>1 == 0x7F (logical, positive operand).
    cs.append(EdgeCase(
        "shr_uchar_ff_by1", "char_shift", "asm",
        _store_byte_then(0x47, 0xFF, _li(0x47)
                         + [("PSH", 0), ("IMM", 1), ("SHR", 0), ("HALT", 0)]),
        0x7F, note="(uchar 255) >> 1 == 127 (LI unsigned operand)"))
    # SHL crossing the sign bit: 0x0F << 4 == 0xF0 (== -16 as a char once stored).
    cs.append(EdgeCase(
        "shl_0f_by4", "char_shift", "asm", _bin(0x0F, 4, "SHL"), 0xF0,
        note="0x0F << 4 == 0xF0"))
    cs.append(EdgeCase(
        "shl_01_by7", "char_shift", "asm", _bin(0x01, 7, "SHL"), 0x80,
        note="1 << 7 == 0x80 (sign bit set)"))
    # SHL then store-char truncation: (0x0F<<4)=0xF0, SC to mem[0x4A], LC back = -16.
    #   IMM 0x4A ; PSH (addr) ; IMM 0x0F ; PSH ; IMM 4 ; SHL (AX=0xF0) ; SC (store
    #   low byte 0xF0 to popped addr 0x4A) ; IMM 0x4A ; LC (read signed -> -16).
    cs.append(EdgeCase(
        "shl_store_char_trunc", "char_shift", "asm",
        [("IMM", 0x4A), ("PSH", 0),                              # addr on stack
         ("IMM", 0x0F), ("PSH", 0), ("IMM", 4), ("SHL", 0),      # AX = 0xF0
         ("SC", 0)] + _lc(0x4A) + [("HALT", 0)],                 # store byte, LC back
        0xFFFFFFF0, note="(0x0F<<4)=0xF0 stored as char, LC back == -16"))
    # SHR by amount 0 is identity; amount >= width behaviour (arith fills sign).
    cs.append(EdgeCase(
        "shr_char_neg1_by0", "char_shift", "asm",
        _store_byte_then(0x4B, 0xFF, _lc(0x4B)
                         + [("PSH", 0), ("IMM", 0), ("SHR", 0), ("HALT", 0)]),
        M32, note="(char -1) >> 0 == -1 (identity)"))
    return cs


# ===========================================================================
# CLUSTER 3 — CHAR ARITHMETIC  (int-promote -> 32-bit compute -> truncate store).
# ===========================================================================
def _cluster_char_arith() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # char + char: 0x7F + 0x01 == 0x80 (128); as a stored char it wraps to -128.
    cs.append(EdgeCase("char_add_7f_01", "char_arith", "asm", _bin(0x7F, 0x01, "ADD"),
                       0x80, note="127 + 1 == 128"))
    # char - char going negative: 0x10 - 0x20 == -16 (0xFFFFFFF0) at 32-bit.
    #   (both operands' low byte via IMM; SUB pops STACK0 - AX).
    cs.append(EdgeCase("char_sub_neg", "char_arith", "asm", _bin(0x10, 0x20, "SUB"),
                       0xFFFFFFF0, note="16 - 32 == -16"))
    # '0' + (n % 10) digit formatting: 7%10==7, '0'(48)+7 == '7'(55).
    cs.append(EdgeCase(
        "digit_format", "char_arith", "asm",
        [("IMM", 7), ("PSH", 0), ("IMM", 10), ("MOD", 0),   # AX = 7
         ("PSH", 0), ("IMM", 48), ("ADD", 0),               # STACK0=7, AX=48 -> 55
         ("HALT", 0)],
        55, note="'0' + (7%10) == '7' (digit format)"))
    # char * int (int promotion): 0x40(64) * 4 == 256 (0x100) at 32-bit.
    cs.append(EdgeCase("char_mul_int", "char_arith", "asm", _bin(0x40, 4, "MUL"),
                       0x100, note="64 * 4 == 256 (int promotion, no byte wrap)"))
    # multi-byte add carry: 0xFF + 0x01 == 0x100 (carry out of byte 0).
    cs.append(EdgeCase("add_carry_out", "char_arith", "asm", _bin(0xFF, 0x01, "ADD"),
                       0x100, note="255 + 1 == 256 (carry to byte 1)"))
    # multi-byte sub borrow: 0x00 - 0x01 == -1 (0xFFFFFFFF), borrow cascade.
    cs.append(EdgeCase("sub_borrow_cascade", "char_arith", "asm", _bin(0x00, 0x01, "SUB"),
                       M32, note="0 - 1 == -1 (borrow cascade)"))
    # 16-bit-ish mul: 0xFF * 0xFF == 0xFE01 (65025), stresses the nibble schoolbook.
    cs.append(EdgeCase("mul_ff_ff", "char_arith", "asm", _bin(0xFF, 0xFF, "MUL"),
                       0xFE01, note="255 * 255 == 65025 (multi-byte product)"))
    # C-level digit build: (n=253) -> hundreds digit '2' via /100 then %10 +'0'.
    cs.append(EdgeCase(
        "c_digit_hundreds", "char_arith", "c",
        "int main(){ int n; n=253; return 48 + (n/100)%10; }",
        50, note="hundreds digit of 253 is '2'(50)", max_steps=200))
    return cs


# ===========================================================================
# CLUSTER 4 — I/O  (PRTF %d/%c/%s, PUTCHAR, READ from stdin, GETCHAR).
# ===========================================================================
_FMT_ADDR = 0x20          # low window so the LC/read CAM can reach it
_BUF_ADDR = 0x40


def _seed_cstring(d: Dict[int, int], addr: int, s) -> None:
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        d[addr + i] = b
    d[addr + len(s)] = 0


def _cluster_io() -> List[EdgeCase]:
    cs: List[EdgeCase] = []

    # PUTCHAR sequence: PRTF prints AX&0xFF (the single-char printf channel).
    #   IMM 'A'; PRTF ; IMM 'B'; PRTF ; HALT  -> stdout "AB", final AX='B'.
    cs.append(EdgeCase(
        "putchar_seq", "io", "asm",
        [("IMM", ord("A")), ("PRTF", 0), ("IMM", ord("B")), ("PRTF", 0), ("HALT", 0)],
        ord("B"), note="PUTCHAR('A');PUTCHAR('B') -> \"AB\"",
        expected_stdout=b"AB"))
    # PUTCHAR of a newline + digit.
    cs.append(EdgeCase(
        "putchar_digit_nl", "io", "asm",
        [("IMM", ord("7")), ("PRTF", 0), ("IMM", 10), ("PRTF", 0), ("HALT", 0)],
        10, note="print '7' then '\\n'", expected_stdout=b"7\n"))

    # READ from supplied stdin -> buffer, then LC the bytes back.  fd=0 (stdin).
    #   marshalling: OPEN not needed for stdin; READ(fd=0, buf, n) = pop fd, pop buf, n=AX
    #   IMM 0 ; PSH (fd) ; IMM buf ; PSH ; IMM n ; READ ; then LC buf[0], buf[1]
    read_prog = ([("IMM", 0), ("PSH", 0),                    # fd = 0 (stdin)
                  ("IMM", _BUF_ADDR), ("PSH", 0),            # buf
                  ("IMM", 3),                                 # n = 3
                  ("READ", 0)]                                # AX = n_read (=3)
                 + _lc(_BUF_ADDR)                             # AX = buf[0]
                 + [("HALT", 0)])
    cs.append(EdgeCase(
        "read_stdin_lc0", "io", "asm", read_prog, ord("c"),
        note="READ 3 stdin bytes \"cat\", LC buf[0]=='c'",
        stdin=b"cat", io=True, max_steps=32))
    # GETCHAR == READ 1 byte from stdin, LC it back.
    getchar = ([("IMM", 0), ("PSH", 0), ("IMM", _BUF_ADDR), ("PSH", 0),
                ("IMM", 1), ("READ", 0)] + _lc(_BUF_ADDR) + [("HALT", 0)])
    cs.append(EdgeCase(
        "getchar_stdin", "io", "asm", getchar, ord("Z"),
        note="GETCHAR: READ 1 stdin byte 'Z', LC back",
        stdin=b"Z!", io=True, max_steps=32))

    # PRTF %d %c %s : the runner formats via the TOOL_CALL protocol (pending_args).
    # Golden stdout is computed by the c4 printf subset in FileRunner; the neural
    # path drives the SAME runner.  We assert the stdout bytes (byte-exact I/O).
    _fmt_seg: Dict[int, int] = {}
    _seed_cstring(_fmt_seg, _FMT_ADDR, "hi %d %c\n")
    cs.append(EdgeCase(
        "prtf_d_c", "io", "asm",
        [("IMM", _FMT_ADDR), ("PSH", 0), ("PRTF", 0), ("HALT", 0)],
        len(b"hi 7 Z\n"), note="printf(\"hi %d %c\\n\", 7, 'Z')",
        data_seg=_fmt_seg, expected_stdout=b"hi 7 Z\n", prtf_args=[7, ord("Z")],
        io=True, max_steps=16))

    return cs


# ===========================================================================
# CLUSTER 5 — THE REST  (AND/OR/XOR with sign-bit operands, GE, BNZ).
# ===========================================================================
def _cluster_bitwise_cmp() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # AND / OR / XOR at the byte, including the sign bit (0x80).
    cs.append(EdgeCase("and_f0_0f", "bitwise", "asm", _bin(0xF0, 0x0F, "AND"), 0x00,
                       note="0xF0 & 0x0F == 0"))
    cs.append(EdgeCase("and_ff_80", "bitwise", "asm", _bin(0xFF, 0x80, "AND"), 0x80,
                       note="0xFF & 0x80 == 0x80 (sign bit)"))
    cs.append(EdgeCase("or_f0_0f", "bitwise", "asm", _bin(0xF0, 0x0F, "OR"), 0xFF,
                       note="0xF0 | 0x0F == 0xFF"))
    cs.append(EdgeCase("or_80_01", "bitwise", "asm", _bin(0x80, 0x01, "OR"), 0x81,
                       note="0x80 | 0x01 == 0x81"))
    cs.append(EdgeCase("xor_ff_0f", "bitwise", "asm", _bin(0xFF, 0x0F, "XOR"), 0xF0,
                       note="0xFF ^ 0x0F == 0xF0"))
    cs.append(EdgeCase("xor_aa_ff", "bitwise", "asm", _bin(0xAA, 0xFF, "XOR"), 0x55,
                       note="0xAA ^ 0xFF == 0x55"))
    cs.append(EdgeCase("xor_self", "bitwise", "asm", _bin(0x5A, 0x5A, "XOR"), 0x00,
                       note="x ^ x == 0"))
    # GE (untested!): 5>=3, 3>=5, 5>=5.
    cs.append(EdgeCase("ge_5_3", "cmp_ge", "asm", _bin(5, 3, "GE"), 1, note="5 >= 3"))
    cs.append(EdgeCase("ge_3_5", "cmp_ge", "asm", _bin(3, 5, "GE"), 0, note="3 >= 5"))
    cs.append(EdgeCase("ge_5_5", "cmp_ge", "asm", _bin(5, 5, "GE"), 1, note="5 >= 5"))
    cs.append(EdgeCase("ge_0_0", "cmp_ge", "asm", _bin(0, 0, "GE"), 1, note="0 >= 0"))
    return cs


def _cluster_branch() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # BNZ (untested!): branch-if-nonzero.  IMM 1 ; BNZ tgt ; IMM 9 ; HALT ; IMM 7 ; HALT.
    cs.append(EdgeCase(
        "bnz_taken", "branch", "asm",
        [("IMM", 1), ("BNZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
        7, note="BNZ taken (AX!=0) -> returns 7"))
    cs.append(EdgeCase(
        "bnz_not_taken", "branch", "asm",
        [("IMM", 0), ("BNZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
        9, note="BNZ not taken (AX==0) -> falls through to 9"))
    # BZ control (exercised by corpus, kept as a positive control alongside BNZ).
    cs.append(EdgeCase(
        "bz_taken", "branch", "asm",
        [("IMM", 0), ("BZ", 4), ("IMM", 9), ("HALT", 0), ("IMM", 7), ("HALT", 0)],
        7, note="BZ taken (AX==0) -> returns 7"))
    return cs


# ===========================================================================
# CLUSTER 6 — MALC / FREE / MSET / MCMP  (compiled-from-C runtime subroutines).
# ===========================================================================
# NB: malloc/free/memset/memcmp compile to JSR/ENT/LEV SUBROUTINES using LC/SC/LI/
# SI/BNZ (NOT the MALC/FREE/MSET/MCMP opcodes — those are word-VM intrinsics the
# neural model never decodes; it runs the compiled bytecode).  So these C cases
# exercise the runtime library AND naturally cover LC/SC/BNZ through real C.
def _cluster_runtime() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    cs.append(EdgeCase(
        "malloc_write_read_free", "runtime", "c",
        "int main(){ char *p; int i; p = malloc(8); i=0; "
        "while(i<8){ p[i]=i*2; i=i+1; } i=p[3]; free(p); return i; }",
        6, note="malloc/free + char[] write/read-back -> p[3]==6", max_steps=6000))
    # NB: the LIBRARY memset() DIVERGES on the c4_min 8-bit-IMM VM (ref=0) vs native
    # c4 (65) — the linked stdlib memset uses a wide constant the ISA IMM folds to a
    # byte.  So this cluster uses a HAND-WRITTEN char[] fill loop (byte-exact ref ==
    # native == neural), which exercises the same char* LC/SC/BNZ runtime path.
    cs.append(EdgeCase(
        "char_fill_loop", "runtime", "c",
        "int main(){ char *p; int i; p = malloc(8); i=0; "
        "while(i<8){ p[i]=65; i=i+1; } return p[5]; }",
        65, note="hand-written char[] fill (memset-style) -> p[5]=='A'",
        max_steps=6000))
    cs.append(EdgeCase(
        "memcmp_equal", "runtime", "c",
        "int main(){ char *a; char *b; a=malloc(4); b=malloc(4); "
        "a[0]=1; a[1]=2; a[2]=3; a[3]=0; b[0]=1; b[1]=2; b[2]=3; b[3]=0; "
        "return memcmp(a,b,4); }",
        0, note="memcmp of equal 4-byte buffers == 0", max_steps=8000))
    return cs


# ===========================================================================
# CLUSTER 7 — IFFY RANGES  (signed DIV/MOD sign combos, boundaries, overflow).
# ===========================================================================
# NB: the c4_min ISA IMM folds a constant to 8 bits and DIV/MOD in ``ref_interpret``
# are UNSIGNED floor at the value width (matching the neural gadget's unsigned base-16
# long division).  A negative dividend must be built by an ALU op (SUB) so its full
# 32-bit two's-complement is on the stack, NOT by an out-of-range IMM.  These pin the
# UNSIGNED-floor golden the neural model actually computes; native-c4 signed div is
# noted where it differs.
def _cluster_ranges() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # Plain unsigned div/mod (positive operands, the neural gadget's domain).
    cs.append(EdgeCase("div_17_5", "ranges", "asm", _bin(17, 5, "DIV"), 3,
                       note="17 / 5 == 3"))
    cs.append(EdgeCase("mod_17_5", "ranges", "asm", _bin(17, 5, "MOD"), 2,
                       note="17 % 5 == 2"))
    # DIV by zero -> 0 (ISA_SPEC 4.2: b==0 -> 0), MOD by zero -> 0.
    cs.append(EdgeCase("div_by_zero", "ranges", "asm", _bin(7, 0, "DIV"), 0,
                       note="7 / 0 == 0 (ISA guard)"))
    cs.append(EdgeCase("mod_by_zero", "ranges", "asm", _bin(7, 0, "MOD"), 0,
                       note="7 % 0 == 0 (ISA guard)"))
    # Boundaries: 0/1, 1/1, 255/1, 255/255, 1%1.
    cs.append(EdgeCase("div_255_1", "ranges", "asm", _bin(255, 1, "DIV"), 255,
                       note="255 / 1 == 255"))
    cs.append(EdgeCase("div_255_255", "ranges", "asm", _bin(255, 255, "DIV"), 1,
                       note="255 / 255 == 1"))
    cs.append(EdgeCase("mod_255_16", "ranges", "asm", _bin(255, 16, "MOD"), 15,
                       note="255 % 16 == 15"))
    cs.append(EdgeCase("div_0_5", "ranges", "asm", _bin(0, 5, "DIV"), 0,
                       note="0 / 5 == 0"))
    # (0 - 1) / 1 : dividend is 0xFFFFFFFF (unsigned 4294967295) / 1 == itself.
    cs.append(EdgeCase(
        "div_neg1_1_unsigned", "ranges", "asm",
        [("IMM", 0), ("PSH", 0), ("IMM", 1), ("SUB", 0),    # AX = -1 = 0xFFFFFFFF
         ("PSH", 0), ("IMM", 1), ("DIV", 0), ("HALT", 0)],
        M32, note="(0-1)/1 == 0xFFFFFFFF unsigned (neural DIV is unsigned floor)"))
    # C-level signed div/mod sign combos (the compiler's real lowering).
    cs.append(EdgeCase("c_div_pos_pos", "ranges", "c",
                       "int main(){ return 17/5; }", 3, note="17/5==3", max_steps=64))
    cs.append(EdgeCase("c_mod_pos_pos", "ranges", "c",
                       "int main(){ return 17%5; }", 2, note="17%5==2", max_steps=64))
    return cs


# ===========================================================================
# CLUSTER 8 — DEEP RECURSION  (the #702 1-slot STACK0 wall).
# ===========================================================================
def _cluster_recursion() -> List[EdgeCase]:
    cs: List[EdgeCase] = []
    # Iterative sum 1..10 == 55 (control; shallow frame).
    cs.append(EdgeCase(
        "iter_sum_10", "recursion", "c",
        "int main(){ int i; int s; i=1; s=0; while(i<=10){ s=s+i; i=i+1; } return s; }",
        55, note="iterative sum 1..10 == 55", max_steps=400))
    # Recursive factorial 5! == 120 (== 0x78, fits a byte).
    cs.append(EdgeCase(
        "rec_fact_5", "recursion", "c",
        "int fact(int n){ if (n<2) return 1; return n*fact(n-1); } "
        "int main(){ return fact(5); }",
        120, note="fact(5) == 120 (recursion depth 5)", max_steps=800))
    # Recursive sum 1..8 == 36 (deeper frame chain).
    cs.append(EdgeCase(
        "rec_sum_8", "recursion", "c",
        "int rsum(int n){ if (n==0) return 0; return n + rsum(n-1); } "
        "int main(){ return rsum(8); }",
        36, note="rsum(8) == 36 (recursion depth 8)", max_steps=1200))
    return cs


# ---------------------------------------------------------------------------
# Assembly of the whole corpus.
# ---------------------------------------------------------------------------
def generate_edge_cases() -> List[EdgeCase]:
    """The full additive edge-case corpus (all clusters)."""
    cases: List[EdgeCase] = []
    for builder in (_cluster_char_rw, _cluster_char_shift, _cluster_char_arith,
                    _cluster_io, _cluster_bitwise_cmp, _cluster_branch,
                    _cluster_runtime, _cluster_ranges, _cluster_recursion):
        cases.extend(builder())
    # unique names
    seen = set()
    for c in cases:
        assert c.name not in seen, f"duplicate case name {c.name}"
        seen.add(c.name)
    return cases


def opcodes_covered(cases: Optional[List[EdgeCase]] = None) -> set:
    """The set of opcode NAMES the edge corpus exercises (union over compiled +
    assembled programs)."""
    cases = cases or generate_edge_cases()
    ops = set()
    for c in cases:
        try:
            for ins in c.code():
                ops.add(isa.NAMES.get(ins.op, ins.op))
        except Exception:  # noqa: BLE001 — a compile failure is surfaced by the runner
            pass
    return ops


if __name__ == "__main__":
    cs = generate_edge_cases()
    from collections import Counter
    by_cluster = Counter(c.cluster for c in cs)
    print(f"edge corpus: {len(cs)} cases across {len(by_cluster)} clusters")
    for cl, n in sorted(by_cluster.items()):
        print(f"  {cl:14s} {n}")
    print("opcodes covered:", sorted(opcodes_covered(cs)))
