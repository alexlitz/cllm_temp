#!/usr/bin/env python3
"""_agent_io_progs.py — MODE-1 (STRICT per-byte pointer-walk) and MODE-2 (BURST /
syscall) I/O programs for the all-C native VM, all reading/writing via §Memory
(NOT hardcoded IMM(b);PRTF(b) literals).

Two selectable modes, byte-exact to each other and to the reference:

  MODE 1  STRICT   the PROGRAM loops out of §Memory: put the string+NUL in §Memory,
                   ``ptr=base; loop: AX = mem[ptr]; BZ done; PRTF; ptr+=1; JMP loop``.
                   ONE VM step (forward + think-frame) per byte; single-char PRTF.
                   Faithful c4-VM loop; the LC/LI reads are computed NEURALLY.

  MODE 2  BURST    ``IMM base; PRTF`` — ONE PRTF whose AX is the POINTER; the RUNTIME
                   walks §Memory mem[ptr..NUL] and writes the whole string in one op
                   (a C loop in the opcode handler).  READ likewise injects the whole
                   stdin buffer into §Memory in one op.  No per-byte forward.

Every program's payload lives in §Memory (``seed_mem`` for echo/yes, or the stdin
input-KV for cat/eliza) — never as an IMM literal per byte.

The 8-bit pointer window (SP_INIT = 0xFC): stack descends from 252, so the data
segment 0x00..~0xC0 is free.  Messages must be < ~48 bytes.

Returns ``IOProg(code, seed_mem, expected, stdin, mode)``:
  * ``code``      : list[(op_name, imm)]  (assemble with isa.assemble)
  * ``seed_mem``  : {addr: byte}  leading §Memory data-segment stores
  * ``expected``  : bytes  the visible stdout the run must produce (reference)
  * ``stdin``     : bytes  the stdin the run is fed (cat/eliza), or b""
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---- 8-bit data-segment layout (all addresses < SP_INIT=0xFC) --------------
BUF = 0x10          # read buffer base (cat / eliza input)
MSG_BASE = 0x00     # echo / yes message string base (mirror quine Q_BASE=0)
IDX_CELL = 0xF0     # pointer scratch cell (p); below SP, never collided


@dataclass
class IOProg:
    code: List[Tuple[str, int]]
    seed_mem: Dict[int, int]
    expected: bytes
    stdin: bytes = b""
    mode: str = "strict"
    note: str = ""


# ---------------------------------------------------------------------------
# The STRICT pointer-walk print loop (prints mem[base..NUL], one PRTF/byte).
# This is the classic c4 printf(ptr) loop: it READS §Memory with LC and branches
# on the NUL, exactly like real printf walking a C string.  ``base`` is a byte
# address (< 256).  Uses the IDX_CELL scratch counter (i), and computes the
# byte address base+i, loads it (LC) and PRTFs.  A NUL byte ends the loop.
# ---------------------------------------------------------------------------
# targets of these ops are absolute PC indices and must be relocated when a code
# block is spliced at a non-zero offset.
_JUMP_OPS = frozenset({"JMP", "BZ", "BNZ", "JSR"})


def _relocate(block: List[Tuple[str, int]], offset: int) -> List[Tuple[str, int]]:
    """Shift every absolute jump/branch target in ``block`` by ``offset``."""
    out = []
    for name, imm in block:
        out.append((name, imm + offset) if name in _JUMP_OPS else (name, imm))
    return out


def _strict_print_loop(base: int, idx_cell: int = IDX_CELL) -> List[Tuple[str, int]]:
    """The classic c4 ``while (*p) putchar(*p++)`` pointer-walk.

    A pointer ``p`` (initialised to ``base``) lives in the scratch cell ``idx_cell``.
    Each iteration: LOAD p, deref (LC = ``mem[p]``), NUL-test (BZ done), PRTF the
    byte, then p = p + 1 and loop.  The only arithmetic is the pointer INCREMENT
    ``p + 1`` (ADD with immediate 1) — the exact ADD the proven quine loop uses;
    there is NO ``base + i`` re-add (that untested ADD-with-large-immediate mis-
    computes in the neural ALU for some operand pairs).  Works for ANY ``base``.
    """
    prog: List[Tuple[str, int]] = []

    def E(name, imm=0):
        prog.append((name, imm))
        return len(prog) - 1

    # p = base :  mem[idx_cell] = base
    E("IMM", idx_cell); E("PSH"); E("IMM", base); E("SI")
    TOP = len(prog)
    # AX = mem[p]  (load p, then deref with LC)
    E("IMM", idx_cell); E("LI")                 # AX = p
    E("LC")                                      # AX = mem[p]   (byte load)
    bz = E("BZ", 0)                              # if AX == 0 (NUL) -> DONE
    E("PRTF")                                    # printf("%c", AX)  <-- one byte
    # p = p + 1  :  mem[idx_cell] = mem[idx_cell] + 1
    E("IMM", idx_cell); E("PSH")                # push &p
    E("IMM", idx_cell); E("LI"); E("PSH"); E("IMM", 1); E("ADD")   # AX = p + 1
    E("SI")                                      # mem[&p] = p + 1
    E("JMP", TOP)
    DONE = len(prog)
    E("HALT")
    prog[bz] = ("BZ", DONE)
    return prog


# The BURST print: one PRTF whose AX is the POINTER; the runtime bursts the whole
# string from §Memory.  (mem[base..NUL] -> stdout in the opcode handler.)
def _burst_print(base: int) -> List[Tuple[str, int]]:
    return [("IMM", base), ("PRTF", 0), ("HALT", 0)]


def _seed_string(base: int, text: bytes) -> Dict[int, int]:
    """Lay ``text`` + a trailing NUL into §Memory at ``base`` (8-bit addresses)."""
    seed = {}
    for i, b in enumerate(text):
        seed[(base + i) & 0xFF] = int(b)
    seed[(base + len(text)) & 0xFF] = 0        # NUL terminator
    return seed


# ---------------------------------------------------------------------------
# echo:  print a fixed message from §Memory.
# ---------------------------------------------------------------------------
def build_echo(text: str, mode: str) -> IOProg:
    payload = text.encode()
    assert len(payload) < 48, "message too long for the 8-bit window"
    seed = _seed_string(MSG_BASE, payload)
    if mode == "burst":
        code = _burst_print(MSG_BASE)
    else:
        code = _strict_print_loop(MSG_BASE)
    return IOProg(code, seed, payload, b"", mode,
                  note=f"echo {text!r} via §Memory ({mode})")


# ---------------------------------------------------------------------------
# yes:  print a message N times (finite for the 8-bit demo).  Each repeat walks
# §Memory again (strict) or bursts it again (burst).
# ---------------------------------------------------------------------------
def build_yes(text: str, n: int, mode: str) -> IOProg:
    payload = text.encode()
    assert len(payload) < 48, "message too long for the 8-bit window"
    seed = _seed_string(MSG_BASE, payload)
    expected = payload * n
    if mode == "burst":
        # N burst PRTFs of the same pointer (each bursts the whole string)
        code = []
        for _ in range(n):
            code += [("IMM", MSG_BASE), ("PRTF", 0)]
        code += [("HALT", 0)]
    else:
        # unroll N strict print-loops (each a full pointer-walk of the string).
        # Each copy's absolute jump targets are RELOCATED by the running offset so
        # a copy's DONE falls through to the NEXT copy (its HALT is dropped); only
        # the last copy keeps its HALT.
        code = []
        for k in range(n):
            body = _strict_print_loop(MSG_BASE)
            if k != n - 1:
                body = body[:-1]                     # drop this copy's HALT
            code += _relocate(body, len(code))
    return IOProg(code, seed, expected, b"", mode,
                  note=f"yes {text!r} x{n} via §Memory ({mode})")


# ---------------------------------------------------------------------------
# cat:  READ a line from stdin into §Memory (BUF), then print BUF back.
# ---------------------------------------------------------------------------
def build_cat(stdin_text: str, mode: str) -> IOProg:
    payload = stdin_text.encode()
    assert len(payload) < 48, "stdin too long for the 8-bit window"
    n = len(payload)

    # READ(fd=0, buf=BUF, n) : lowering pushes fd then buf, n in AX.
    #   IMM 0 ; PSH        -> fd = 0 on stack
    #   IMM BUF ; PSH      -> buf = BUF on stack
    #   IMM n              -> AX = n (count)
    #   READ               -> reads n bytes stdin -> §Memory[BUF..], AX = n_read
    read_seq = [("IMM", 0), ("PSH", 0),
                ("IMM", BUF), ("PSH", 0),
                ("IMM", n), ("READ", 0)]
    # ensure a NUL terminates the printed buffer even if input has no NUL:
    # store 0 at BUF+n  (IMM BUF+n ; PSH ; IMM 0 ; SC)
    nul_seq = [("IMM", (BUF + n) & 0xFF), ("PSH", 0), ("IMM", 0), ("SC", 0)]

    prelude = read_seq + nul_seq
    if mode == "burst":
        code = prelude + _burst_print(BUF)
    else:
        # RELOCATE the print loop's absolute jump targets past the prelude.
        code = prelude + _relocate(_strict_print_loop(BUF), len(prelude))
    # cat echoes stdin verbatim
    return IOProg(code, {}, payload, payload, mode,
                  note=f"cat (READ stdin -> §Memory -> print) ({mode})")


# ---------------------------------------------------------------------------
# catchunk:  ARBITRARY-LENGTH streaming cat (Part B).  Loop: READ(fd=0, BUF, CHUNK)
# a chunk into §Memory, NUL-terminate it, burst-print it, repeat until READ returns
# 0 (EOF).  Each new READ turn writes the SAME addresses (BUF..BUF+CHUNK) -> the
# previous chunk's store rows are SUPERSEDED (dropped by free-driven eviction), so an
# arbitrary-size file streams through with the KV pinned at ~chunk+window (NO OOM).
# ---------------------------------------------------------------------------
def build_catchunk(mode: str, chunk: int = 32, stdin_text: str = "") -> IOProg:
    """A LOOPING cat: read up to `chunk` bytes per turn into §Memory and print them,
    until EOF.  `stdin_text` is only the reference oracle (the runtime reads real
    stdin); expected == the whole stdin verbatim.  chunk<=~200 for the 8-bit window."""
    assert 1 <= chunk <= 200, "chunk must fit the 8-bit §Memory window"
    prog: List[Tuple[str, int]] = []

    def E(name, imm=0):
        prog.append((name, imm)); return len(prog) - 1

    TOP = len(prog)
    # READ(fd=0, buf=BUF, n=chunk): push fd, push buf, AX=chunk, READ -> AX=got
    E("IMM", 0); E("PSH"); E("IMM", BUF); E("PSH"); E("IMM", chunk); E("READ")
    bz = E("BZ", 0)                                  # got==0 -> EOF -> DONE
    # NUL-terminate at BUF+got (READ left AX=got; ADD = popped + AX; SC stores AX at
    # the pushed address):  AX=BUF+got, then mem[BUF+got]=0 so the burst stops there.
    E("PSH")                                          # push got (current AX)
    E("IMM", BUF); E("ADD")                           # AX = got + BUF = BUF+got (NUL addr)
    E("PSH"); E("IMM", 0); E("SC")                    # push addr; AX=0; SC -> mem[addr]=0
    # burst-print BUF (mode burst: one PRTF, runtime walks §Memory BUF..NUL)
    if mode == "burst":
        E("IMM", BUF); E("PRTF")
    else:
        # strict: relocate a print-loop of BUF inline (rare; burst is the streaming path)
        body = _relocate(_strict_print_loop(BUF), len(prog))
        body = body[:-1]                              # drop the loop's HALT (we JMP back)
        prog += body
    E("JMP", TOP)
    DONE = len(prog)
    E("HALT")
    prog[bz] = ("BZ", DONE)
    payload = stdin_text.encode("latin-1")
    return IOProg(prog, {}, payload, payload, mode,
                  note=f"catchunk chunk={chunk} ({mode})")


# ---------------------------------------------------------------------------
# keepheap:  the FREE-DRIVEN-EVICTION correctness test (Part A #2).  Write several
# distinct §Memory addresses, then do ENOUGH "other work" (many dummy stores to a
# scratch cell — each is a fresh VM step / frame) to blow PAST any recency window,
# then LC-read the ORIGINAL addresses back and PRTF them.  If a live store were
# evicted for being old, the read-back would return 0/garbage; a correct keep-heap
# returns the stored bytes verbatim.  Byte-exact between base (full stream) and
# --evict (compacted keep-set) is the proof that the LIVE HEAP survives eviction.
# ---------------------------------------------------------------------------
def build_keepheap(text: str, mode: str, churn: int = 8) -> IOProg:
    """text = the payload bytes to store at distinct addresses then read back.
    churn = number of dummy scratch stores between the writes and the reads (each is
    ~1 VM step / frame ~30 tokens; the recency window is 60 tokens ~2 frames, so
    churn>=3 forces the writes out of the window; default 8 = ~240 tokens behind, a
    generous margin while keeping the BASE (full-stream) run tractable to compare)."""
    payload = text.encode()
    assert len(payload) < 40, "payload too long for the 8-bit window"
    # store addresses 0x20, 0x21, ... one per payload byte (distinct, below SP/scratch)
    HEAP = 0x20
    SCRATCH = 0xE8            # dummy-churn cell (below SP_INIT=0xFC, above heap)
    prog: List[Tuple[str, int]] = []

    def E(name, imm=0):
        prog.append((name, imm)); return len(prog) - 1

    # 1) WRITE the payload: for each byte, mem[HEAP+i] = byte  (SI: push &addr, IMM v, SI)
    for i, b in enumerate(payload):
        E("IMM", (HEAP + i) & 0xFF); E("PSH"); E("IMM", int(b)); E("SI")

    # 2) CHURN: `churn` dummy stores to SCRATCH (each a fresh frame -> pushes the
    #    payload writes far behind the recency window).  These are LIVE stores too,
    #    but all to the SAME address -> each supersedes the last, so only ONE survives
    #    (proves supersede-eviction keeps the heap bounded while the reads still work).
    for j in range(churn):
        E("IMM", SCRATCH); E("PSH"); E("IMM", (j & 0xFF)); E("SI")

    # 3) READ BACK the ORIGINAL heap addresses (now long past the window) and PRTF.
    for i in range(len(payload)):
        E("IMM", (HEAP + i) & 0xFF); E("LC"); E("PRTF")
    E("HALT")
    return IOProg(prog, {}, payload, b"", mode,
                  note=f"keepheap {text!r} churn={churn} ({mode})")


# ---------------------------------------------------------------------------
# ELIZA:  READ a line from stdin (§Memory input-KV), prefix-match it against the
# keyword table, PRTF the matching response (else fallback).  Both keyword +
# response tables live in §Memory (seed_mem = data segment).
#
#   BURST : the response is emitted by ONE PRTF whose AX is the response pointer
#           (the runtime walks §Memory) — this is exactly chat_eliza.build_chat_min.
#   STRICT: the response is walked out one byte at a time by an in-program pointer
#           loop (LC mem[p]; PRTF; p++), one VM step per byte.
# The pattern-match (READ + LC + EQ + BZ) is IDENTICAL in both modes; only the
# response-emit differs.  Byte-exact vs chat_eliza's reference (run_eliza_reference).
# ---------------------------------------------------------------------------
ELIZA_PTR_CELL = 0x04       # response-walk pointer scratch (below BUF=0x08, free)


def build_eliza(message: str, mode: str, rules=None, fallback=None) -> IOProg:
    from c4_min import chat_eliza as CE
    from c4_min import isa as _isa
    rules = rules or CE.ELIZA_RULES
    fallback = fallback if fallback is not None else CE.ELIZA_FALLBACK

    # data segment: keyword + response C-strings (same layout as build_chat_min)
    seg: Dict[int, int] = {}
    addr = CE._DATA_BASE
    kw_ptr: List[int] = []
    resp_ptr: List[int] = []
    for kw, _ in rules:
        kw_ptr.append(addr); addr += CE._c(seg, addr, kw)
    for _, resp in rules:
        resp_ptr.append(addr); addr += CE._c(seg, addr, resp)
    fb_ptr = addr; addr += CE._c(seg, addr, fallback)
    assert addr < 256, f"eliza data segment overflows the 8-bit window ({addr})"

    BUF_E = CE.BUF
    n_read = CE.READ_N

    prog: List[Tuple] = []

    def emit_response(rp: int):
        """Emit the response at pointer ``rp`` — burst (one pointer PRTF) or strict
        (an in-program §Memory pointer-walk loop)."""
        out: List[Tuple] = []
        if mode == "burst":
            out += [("IMM", rp), ("PSH", 0), ("PRTF", 0)]
        else:
            # p = rp ; while (c = mem[p]) { putchar(c); p++ }
            out += [("IMM", ELIZA_PTR_CELL), ("PSH", 0), ("IMM", rp), ("SI", 0)]
            out += [("LABEL", ("wtop", rp))]
            out += [("IMM", ELIZA_PTR_CELL), ("LI", 0), ("LC", 0)]
            out += [("BZ", ("wend", rp)), ("PRTF", 0)]
            out += [("IMM", ELIZA_PTR_CELL), ("PSH", 0),
                    ("IMM", ELIZA_PTR_CELL), ("LI", 0), ("PSH", 0),
                    ("IMM", 1), ("ADD", 0), ("SI", 0)]
            out += [("JMP", ("wtop", rp)), ("LABEL", ("wend", rp))]
        return out

    # --- READ one line from stdin into BUF (+ NUL-terminate) -----------------
    prog += [("IMM", 0), ("PSH", 0), ("IMM", BUF_E), ("PSH", 0),
             ("IMM", n_read), ("READ", 0)]
    # NUL-terminate at a FIXED offset (len(message)) so the strict walker stops.
    msg = (message.rstrip("\n") + "\n").encode("latin-1")[:n_read]
    prog += [("IMM", (BUF_E + len(msg)) & 0xFF), ("PSH", 0), ("IMM", 0), ("SI", 0)]

    # --- per rule: unrolled prefix compare of BUF[0..klen) vs keyword ---------
    for ri, (kw, _resp) in enumerate(rules):
        rp = resp_ptr[ri]
        kwb = kw.encode("latin-1")
        Lnext = ("m", ri, "next")
        for k, kb in enumerate(kwb):
            prog += [("IMM", BUF_E + k), ("LC", 0),
                     ("PSH", 0), ("IMM", kb), ("EQ", 0), ("BZ", Lnext)]
        prog += emit_response(rp)
        prog += [("JMP", "DONE"), ("LABEL", Lnext)]
    # fallback
    prog += emit_response(fb_ptr)
    prog += [("LABEL", "DONE"), ("HALT", 0)]

    # resolve labels (chat_eliza's two-pass asm handles str/obj labels; ours are
    # tuples, so do a tiny two-pass resolver here).
    code = _resolve_labels(prog)

    # reference reply (byte-exact oracle via chat_eliza's own runner)
    expected = _eliza_reference_reply(CE, rules, fallback, message)
    return IOProg(code, dict(seg), expected.encode("latin-1"),
                  msg, mode, note=f"eliza reply to {message!r} ({mode})")


def _resolve_labels(prog):
    """Two-pass resolver: ('LABEL', name) defines; a jump imm that is a non-int
    name resolves to that label's instruction index.  Returns [(name, imm)]."""
    labels = {}
    pc = 0
    for op, arg in prog:
        if op == "LABEL":
            labels[arg] = pc
        else:
            pc += 1
    out = []
    for op, arg in prog:
        if op == "LABEL":
            continue
        if not isinstance(arg, int):
            arg = labels[arg]
        out.append((op, arg))
    return out


def _eliza_reference_reply(CE, rules, fallback, message):
    """The byte-exact reply from chat_eliza's own reference runner (same I/O)."""
    eliza = CE.build_chat_min(rules=rules, fallback=fallback)
    return CE.chat_turn_ref(eliza, message)


def build(mode_name: str, io_mode: str, text: str = "hello\n",
          n: int = 8, stdin_text: str = "", chunk_size: int = 32) -> IOProg:
    """Dispatch a named utility (echo/yes/cat/eliza/catchunk/keepheap) in io_mode
    'strict'/'burst'.  ``chunk_size`` = the per-READ chunk for the streaming catchunk."""
    if mode_name == "echo":
        return build_echo(text, io_mode)
    if mode_name == "yes":
        return build_yes(text, n, io_mode)
    if mode_name == "eliza":
        return build_eliza(stdin_text or text, io_mode)
    if mode_name == "cat":
        return build_cat(stdin_text or text, io_mode)
    if mode_name == "catchunk":
        return build_catchunk(io_mode, chunk=chunk_size, stdin_text=stdin_text)
    if mode_name == "keepheap":
        return build_keepheap(text, io_mode, churn=n)
    raise ValueError(f"unknown io prog mode {mode_name!r}")
