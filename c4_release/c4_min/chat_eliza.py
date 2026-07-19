"""A CHAT INTERFACE running through the C4 VM transformer.

A user types a message; the model runs a conversational C4 program (ELIZA) that
READS the message off stdin (via the tool-use input-KV, ``READ`` fd 0),
pattern-matches it, and WRITES a response back with ``PRTF`` — a real read/write
user-message loop, all through the pure-forward transformer (``model.forward``).

This satisfies CHK-4 ("IO behavior with the pure autoregressive transformer works
with reading and writing user messages") via the TOOL-USE I/O path (READ / PRTF),
since neural PUTCHAR/GETCHAR were removed from the small model — the only I/O is
the §Tool Use Mode tool-call boundary (``nibble_filesys``).

    program  --READ(0,buf,n)-->  runner reads the user line from the input-KV
    model    --LC buf[i]------->  the message bytes re-enter as §Memory KV frames
    model    --contains()------>  substring match in bytecode (LC + EQ + branches)
    program  --PRTF(resp)------>  runner formats the response into stdout

The ELIZA bytecode is authored in the c4_min ISA (``isa.assemble``); the SAME
bytecode runs on (a) the pure-forward transformer and (b) a plain-python C4
reference so the exchange is proven BYTE-EXACT.  The two paths differ ONLY in
whether the VM transition is computed by the transformer or by python; the I/O
runner (``nibble_filesys.FileRunner`` + ``InputKVStream``) is identical for both.

Addressing note (the 8-bit boundary): the small model's emitted registers are
8-bit (``mask=0xFF``), so every POINTER (string / buffer address) must live in the
low-256 byte window, and the LI/LC §Memory CAM keys on the low byte of the
address.  So the input buffer + the keyword/response tables are laid out in
[0, 256).  User messages are short (< ~48 bytes) so they fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import isa
from . import nibble_filesys as FS


# ===========================================================================
# A tiny label-based assembler over ``isa.assemble`` so ELIZA's branches/loops
# are readable.  A program is a list of ("OP", arg) where arg may be a str label
# (resolved to the instruction index of that label) or an int immediate.
# ===========================================================================
LABEL = object()   # sentinel op meaning "define a label here" (arg = name)


def asm(prog: List[Tuple]) -> List[isa.Instr]:
    """Two-pass assemble: pass 1 records label -> pc, pass 2 resolves label refs.

    ``("LABEL", name)`` defines a label (emits no instruction); a branch/jump op
    whose immediate is a ``str`` is resolved to that label's instruction index.
    """
    labels: Dict[str, int] = {}
    pc = 0
    for op, arg in prog:
        if op == "LABEL":
            labels[arg] = pc
        else:
            pc += 1
    out: List[Tuple[str, int]] = []
    for op, arg in prog:
        if op == "LABEL":
            continue
        if isinstance(arg, str):
            if arg not in labels:
                raise KeyError(f"undefined label {arg!r}")
            arg = labels[arg]
        out.append((op, arg))
    return isa.assemble(out)


# ===========================================================================
# ELIZA data layout in the low-256 byte window.
#
# The small model's emitted registers are 8-bit, so EVERY pointer (buffer +
# string addresses) must be < 256, and the LI/LC §Memory CAM keys on the low byte
# of the address.  So the ENTIRE working set — scan-index scratch cells, the
# user-message read buffer, the keyword table, the response table — is packed into
# [0, 256).  Layout (low to high), NON-OVERLAPPING:
#     0x00 I_CELL, 0x04 J_CELL            (scan indices)
#     0x08..0x1F  BUF  (24-byte read window: message ≤ 23 bytes + NUL)
#     0x20..0xF5  data segment (keywords + responses + fallback, 214 bytes)
# The buffer window (BUF .. BUF+READ_N) MUST NOT overlap the data segment: the
# scan reads BUF[i+j], and if the buffer ran into the keyword table it would find
# a keyword's OWN bytes and false-match.  So DATA_BASE = BUF + READ_N + 1, and the
# program NUL-terminates the buffer after the READ (BUF[n_read]=0) so the scan
# stops at the real message end.  Responses must be TERSE (the whole table < ~224
# bytes) — the honest 8-bit-window boundary.  A wider model (16-bit pointers)
# would lift this cap.
# ===========================================================================
BUF = 0x08                 # user-message read buffer (buf[0..]) — low window
_DATA_BASE = 0x20          # keyword/response string table base (after BUF+READ_N)


@dataclass
class Eliza:
    """A built ELIZA program: the bytecode + its data segment + the rule table."""
    code: List[isa.Instr]
    data_seg: Dict[int, int]
    # (keyword bytes, response string) pairs, in priority order (for reference).
    rules: List[Tuple[bytes, str]]
    fallback: str


def _c(seg: Dict[int, int], addr: int, s) -> int:
    """Write a NUL-terminated C string ``s`` at ``addr``; return len+1 (bytes used)."""
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        seg[addr + i] = b
    seg[addr + len(s)] = 0
    return len(s) + 1


# ELIZA's keyword -> response table (the classic 1966 reflection style).  Terse
# responses so the whole table fits the 8-bit pointer window (< 256).
ELIZA_RULES: List[Tuple[str, str]] = [
    ("bye",    "BYE. TAKE CARE.\n"),
    ("hello",  "HI. HOW DO YOU FEEL?\n"),
    ("sad",    "WHY ARE YOU SAD?\n"),
    ("happy",  "WHAT MAKES YOU HAPPY?\n"),
    ("mother", "TELL ME ABOUT YOUR MOTHER.\n"),
    ("dream",  "WHAT DO DREAMS MEAN?\n"),
    ("yes",    "YOU SEEM CERTAIN.\n"),
    ("no",     "WHY NOT?\n"),
]
ELIZA_FALLBACK = "PLEASE GO ON.\n"
READ_N = 23   # bytes to read per turn; BUF window = 0x08..0x1F (24 B: 23 + NUL)


def build_eliza(rules: Optional[List[Tuple[str, str]]] = None,
                fallback: str = ELIZA_FALLBACK,
                n_read: int = READ_N) -> Eliza:
    """Assemble the ELIZA chat program: one turn = READ a line, scan it for each
    keyword with an inline ``contains`` loop, PRTF the first matching response
    (else the fallback), and stop when the ``bye`` rule fires.

    The program is a single turn (the driver re-runs it per user message, feeding
    each message through the input-KV) — a clean per-turn read/compute/write.  The
    reference interpreter runs the identical bytecode so the exchange is byte-exact.
    """
    rules = rules or ELIZA_RULES
    seg: Dict[int, int] = {}
    addr = _DATA_BASE

    # lay out the keyword C-strings + response C-strings in the data segment.
    kw_ptr: List[int] = []
    resp_ptr: List[int] = []
    for kw, resp in rules:
        kw_ptr.append(addr); addr += _c(seg, addr, kw)
    for kw, resp in rules:
        resp_ptr.append(addr); addr += _c(seg, addr, resp)
    fb_ptr = addr; addr += _c(seg, addr, fallback)
    assert addr < 256, f"ELIZA data segment overflows the 8-bit window ({addr})"
    assert _DATA_BASE >= BUF + n_read + 1, (
        f"BUF window (0x{BUF:02x}..0x{BUF + n_read:02x}) overlaps data seg "
        f"(base 0x{_DATA_BASE:02x}) — the scan would false-match the keyword table")

    # Emit the program in the index-in-memory form (a scratch cell holds the scan
    # index i and the inner match index j — the robust flat-asm form).
    prog = _emit_eliza_indexed(rules, kw_ptr, resp_ptr, fb_ptr, seg, n_read)

    code = asm(prog)
    return Eliza(code=code, data_seg=seg,
                 rules=[(kw.encode("latin-1"), resp) for kw, resp in rules],
                 fallback=fallback)


# ===========================================================================
# THE TRANSFORMER-TRACTABLE chat: a COMPACT prefix-match variant.
#
# The full ELIZA (``build_eliza``) uses an O(len) inline ``contains`` scan per
# keyword, so a single turn is ~140-1200 VM steps.  Because the pure-forward VM
# does ONE ``model.forward`` over the WHOLE growing token stream PER VM step (the
# state lives in the KV log), an N-step turn is O(N²) tokens of compute — so the
# scan-based ELIZA is only tractable THROUGH THE TRANSFORMER for the very first
# handful of steps.  ``build_chat_min`` is the same read/compute/write loop with a
# CHEAP match: it reads the message, matches each keyword as a PREFIX (compares the
# first ``len(kw)`` bytes directly, no scan loop), and PRTFs the matching response.
# A turn is ~15-40 VM steps, so it runs end-to-end through ``model.forward``.  The
# match is genuine (the model reads the user's bytes with LC and branches on them);
# only the match shape (prefix vs substring) is cheaper.  Same byte-exact oracle.
# ===========================================================================
def build_chat_min(rules: Optional[List[Tuple[str, str]]] = None,
                   fallback: str = ELIZA_FALLBACK,
                   n_read: int = READ_N) -> Eliza:
    """A compact PREFIX-match chat (transformer-tractable, ~15-40 steps/turn).

    One turn = READ a line into BUF, then for each rule compare the message's
    leading bytes to the keyword (an unrolled per-byte compare, no scan loop) and
    PRTF the first matching response, else the fallback.  Because there is no
    O(len) scan, a turn is short enough to run whole through ``model.forward``.
    The reference interpreter runs the identical bytecode → byte-exact.
    """
    rules = rules or ELIZA_RULES
    seg: Dict[int, int] = {}
    addr = _DATA_BASE
    kw_ptr: List[int] = []
    resp_ptr: List[int] = []
    for kw, resp in rules:
        kw_ptr.append(addr); addr += _c(seg, addr, kw)
    for kw, resp in rules:
        resp_ptr.append(addr); addr += _c(seg, addr, resp)
    fb_ptr = addr; addr += _c(seg, addr, fallback)
    assert addr < 256, f"chat_min data segment overflows the 8-bit window ({addr})"
    assert _DATA_BASE >= BUF + n_read + 1, "BUF window overlaps data seg"

    prog = _emit_chat_min_prefix(rules, kw_ptr, resp_ptr, fb_ptr, n_read)
    code = asm(prog)
    return Eliza(code=code, data_seg=seg,
                 rules=[(kw.encode("latin-1"), resp) for kw, resp in rules],
                 fallback=fallback)


def _emit_chat_min_prefix(rules, kw_ptr, resp_ptr, fb_ptr, n_read):
    """Emit the compact prefix-match chat: READ into BUF, then per rule do an
    UNROLLED per-byte compare of BUF[0..klen) against the keyword bytes; on a full
    match PRTF the response and jump to DONE; else fall to the next rule; finally
    the fallback.  No loop → ~ (5 + sum(3·klen+3) + ...) instructions, and only
    ~15-40 executed per turn (each mismatch short-circuits to the next rule).
    """
    prog: List[Tuple] = []

    # --- READ one line from stdin into BUF (+ NUL-terminate) -----------------
    prog += [("IMM", 0), ("PSH", 0)]            # fd = 0
    prog += [("IMM", BUF), ("PSH", 0)]          # buf
    prog += [("IMM", n_read)]                   # n
    prog += [("READ", 0)]                       # AX = n_read
    prog += [("PSH", 0)]                        # save n_read
    prog += [("IMM", BUF), ("ADD", 0)]          # AX = BUF + n_read
    prog += [("PSH", 0)]                        # addr for store
    prog += [("IMM", 0), ("SI", 0)]             # BUF[n_read] = 0

    for ri, (kw, resp) in enumerate(rules):
        rp = resp_ptr[ri]
        kwb = kw.encode("latin-1")
        Lnext = f"m{ri}_next"
        # unrolled per-byte compare: if BUF[k] != kw[k] -> next rule
        for k, kb in enumerate(kwb):
            prog += [("IMM", BUF + k), ("LC", 0)]     # AX = BUF[k]
            prog += [("PSH", 0), ("IMM", kb), ("EQ", 0)]  # AX = (BUF[k] == kw[k])
            prog += [("BZ", Lnext)]                    # mismatch -> next rule
        # full prefix matched -> PRTF response, then DONE
        prog += [("IMM", rp), ("PSH", 0), ("PRTF", 0)]
        prog += [("JMP", "DONE")]
        prog += [("LABEL", Lnext)]

    # fallback
    prog += [("IMM", fb_ptr), ("PSH", 0), ("PRTF", 0)]
    prog += [("LABEL", "DONE")]
    prog += [("HALT", 0)]
    return prog


def _emit_eliza_indexed(rules, kw_ptr, resp_ptr, fb_ptr, seg, n_read):
    """Emit ELIZA using a memory scratch cell for the scan index i and the inner
    match index j — the robust flat-asm form (no stack juggling of live pointers).

    Scratch cells (low window, BELOW the buffer, distinct from BUF and the data
    seg; their full 32-bit address has high bits 0, so they never alias the
    descending stack, whose addresses carry non-zero high bits):
        I_CELL  = 0x00   (outer scan index i)
        J_CELL  = 0x04   (inner keyword index j)
    A cell read is ``IMM addr ; LI`` (LI loads MEM[AX]).  A cell write is the c4
    store idiom ``<addr into AX> ; PSH ; <value into AX> ; SI`` (SI pops the
    address off the stack and stores the current AX).  BUF[i] is
    ``IMM BUF ; PSH ; IMM i ; ADD ; LI``.
    """
    I_CELL, J_CELL = 0x00, 0x04
    prog: List[Tuple] = []

    def LOAD(addr):   # AX = MEM[addr]
        return [("IMM", addr), ("LI", 0)]

    def ST(addr_setup, val_setup):
        # addr_setup leaves the address in AX; PSH it; val_setup leaves value in AX; SI.
        return addr_setup + [("PSH", 0)] + val_setup + [("SI", 0)]

    def IMMV(v):
        return [("IMM", v)]

    # --- READ one line from stdin into BUF -----------------------------------
    prog += [("IMM", 0), ("PSH", 0)]          # fd = 0
    prog += [("IMM", BUF), ("PSH", 0)]        # buf
    prog += [("IMM", n_read)]                 # n
    prog += [("READ", 0)]                     # AX = n_read
    # NUL-terminate the message: BUF[n_read] = 0.  Without this, the contains()
    # scan runs off the end of a message that did NOT fill the buffer into the
    # KEYWORD TABLE that follows BUF, and false-matches a keyword's own bytes.
    # AX holds n_read after READ; PSH it, then addr = BUF + n_read (ADD pops it),
    # then store 0 there.  ``ADD`` is 8-bit but n_read ≤ READ_N (< 24) so no wrap.
    prog += [("PSH", 0)]                       # save n_read
    prog += [("IMM", BUF), ("ADD", 0)]         # AX = BUF + n_read  (the NUL slot)
    prog += [("PSH", 0)]                       # push addr for the store
    prog += [("IMM", 0), ("SI", 0)]            # BUF[n_read] = 0

    for ri, (kw, resp) in enumerate(rules):
        kp = kw_ptr[ri]
        rp = resp_ptr[ri]
        klen = len(kw)
        Louter = f"r{ri}_outer"
        Linner = f"r{ri}_inner"
        Lok = f"r{ri}_ok"
        Lnext = f"r{ri}_next"
        Ladv = f"r{ri}_adv"

        # i = 0
        prog += ST(IMMV(I_CELL), IMMV(0))
        prog += [("LABEL", Louter)]
        # c = BUF[i]; if c == 0 -> no match (next rule)
        prog += IMMV(BUF); prog += [("PSH", 0)]        # push BUF
        prog += LOAD(I_CELL)                            # AX = i
        prog += [("ADD", 0), ("LI", 0)]                # AX = BUF[i]
        prog += [("BZ", Lnext)]                         # end of string -> no match
        # try to match kw at position i: j = 0
        prog += ST(IMMV(J_CELL), IMMV(0))
        prog += [("LABEL", Linner)]
        # if j == klen -> full keyword matched at i -> Lok
        prog += LOAD(J_CELL); prog += [("PSH", 0)]; prog += IMMV(klen)
        prog += [("EQ", 0)]                             # AX = (j == klen)
        prog += [("BNZ", Lok)]
        # a = BUF[i+j]
        prog += IMMV(BUF); prog += [("PSH", 0)]
        prog += LOAD(I_CELL); prog += [("PSH", 0)]; prog += LOAD(J_CELL)
        prog += [("ADD", 0)]                            # AX = i + j
        prog += [("ADD", 0), ("LI", 0)]                # AX = BUF[i+j]
        prog += [("PSH", 0)]                            # save a
        # b = kw[j]
        prog += IMMV(kp); prog += [("PSH", 0)]; prog += LOAD(J_CELL)
        prog += [("ADD", 0), ("LI", 0)]                # AX = kw[j]
        # if a != b -> ok = 0 ; break inner (go advance i)
        prog += [("EQ", 0)]                            # AX = (a == b)  (pop a, cmp)
        prog += [("BZ", Ladv)]                          # mismatch -> advance i
        # j = j + 1 ; loop inner.  Store idiom: PSH addr, compute value, SI.
        prog += ST(IMMV(J_CELL),
                   LOAD(J_CELL) + [("PSH", 0)] + IMMV(1) + [("ADD", 0)])
        prog += [("JMP", Linner)]
        # advance i: i = i + 1 ; loop outer
        prog += [("LABEL", Ladv)]
        prog += ST(IMMV(I_CELL),
                   LOAD(I_CELL) + [("PSH", 0)] + IMMV(1) + [("ADD", 0)])
        prog += [("JMP", Louter)]
        # matched: PRTF(resp) ; if this is the 'bye' rule -> HALT else -> DONE(read next)
        prog += [("LABEL", Lok)]
        prog += IMMV(rp); prog += [("PSH", 0), ("PRTF", 0)]
        prog += [("JMP", "DONE")]
        prog += [("LABEL", Lnext)]

    # fallback: PRTF(fb) ; DONE
    prog += IMMV(fb_ptr); prog += [("PSH", 0), ("PRTF", 0)]
    prog += [("LABEL", "DONE")]
    prog += [("HALT", 0)]
    return prog


# ===========================================================================
# REFERENCE C4 VM — plain python, services the SAME tool-call I/O (READ from the
# input-KV, PRTF via the FileRunner) as the transformer path.  This is the
# byte-exact oracle: the two paths differ ONLY in whether the VM transition is
# computed by the transformer or by python; the I/O runner is IDENTICAL.
# ===========================================================================
SP_INIT_REF = 0x10000


def run_eliza_reference(code: List[isa.Instr], data_seg: Dict[int, int],
                        fio: FS.FileOpState, max_steps: int = 20000) -> str:
    """Run ``code`` on a plain-python C4 VM, servicing READ/PRTF through ``fio``
    (the same runner + input-KV the transformer path uses).  Returns stdout text.

    The byte-addressed memory holds the data segment (string literals) + the
    program's stores; the stack descends from ``SP_INIT_REF``.  Every file op is
    dispatched via ``FS.dispatch_file_op`` — byte-for-byte the transformer's tool
    boundary.
    """
    mem: Dict[int, int] = dict(data_seg)
    sp = SP_INIT_REF
    ax = pc = 0
    stack: Dict[int, int] = {}
    steps = 0

    class _Mem:
        def load_int(self, a, w=1):
            return sum((mem.get(a + i, 0) & 0xFF) << (8 * i) for i in range(w))

        def store_int(self, a, val, w=1):
            for i in range(w):
                mem[a + i] = (val >> (8 * i)) & 0xFF
    M = _Mem()

    def push(v):
        nonlocal sp
        sp -= 4; stack[sp] = v & 0xFFFFFFFF

    def pop():
        nonlocal sp
        v = stack.get(sp, 0); sp += 4; return v

    while 0 <= pc < len(code) and steps < max_steps:
        steps += 1
        op, imm = code[pc].op, code[pc].imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & 0xFF
        elif op == isa.SUB:
            ax = (pop() - ax) & 0xFF
        elif op == isa.EQ:
            ax = 1 if (pop() & 0xFF) == (ax & 0xFF) else 0
        elif op == isa.NE:
            ax = 1 if (pop() & 0xFF) != (ax & 0xFF) else 0
        elif op == isa.LI or op == isa.LC:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.SI or op == isa.SC:
            addr = pop(); mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op in FS.FILE_OPCODES:
            # READ(fd=pop, buf=pop, n=AX) / PRTF(fmt=pop) — the tool boundary.
            ax = FS.dispatch_file_op(op, ax & 0xFFFFFFFF, imm, pop, M, fio) & 0xFF
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
    return bytes(fio.runner.stdout).decode("latin-1")


# ===========================================================================
# THE CHAT DRIVER — a multi-turn conversation THROUGH THE TRANSFORMER.
#
# Each user message is fed as INPUT to the model via the tool-use input-KV (a
# READ on fd 0 pulls the message bytes), the model runs the ELIZA bytecode
# (pattern-match + respond) entirely in ``model.forward``, and the response comes
# back out via PRTF.  A turn = one ``run_pure_forward_complete`` over the ELIZA
# program with the turn's message loaded into the input-KV; the driver collects
# the PRTF stdout as ELIZA's reply and loops to the next message.
# ===========================================================================
@dataclass
class Turn:
    user: str
    eliza_model: str      # response produced THROUGH THE TRANSFORMER
    eliza_ref: str        # response from the plain-python reference (same I/O)
    byte_exact: bool


def _fresh_fio(message: str) -> FS.FileOpState:
    """A file-op context whose stdin input-KV carries one user ``message`` (the
    line the user typed, newline-terminated) — the bytes a READ(0,...) pulls in."""
    line = (message.rstrip("\n") + "\n").encode("latin-1")
    return FS.FileOpState(
        runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                             stdin=FS.InputKVStream(line)))


def chat_turn_model(model, L, eliza: Eliza, message: str,
                    max_steps: int = 4000):
    """Run ONE ELIZA turn through the transformer for ``message``; return the PRTF
    reply string.  The message enters via the input-KV (READ fd 0)."""
    from .nibble_pure_forward_complete import run_pure_forward_complete
    fio = _fresh_fio(message)
    run_pure_forward_complete(
        model, L, eliza.code, max_steps=max_steps, mask=0xFF,
        fio=fio, data_seg=dict(eliza.data_seg))
    return bytes(fio.runner.stdout).decode("latin-1")


def chat_turn_ref(eliza: Eliza, message: str) -> str:
    """Run ONE ELIZA turn on the plain-python reference for ``message``."""
    fio = _fresh_fio(message)
    return run_eliza_reference(eliza.code, dict(eliza.data_seg), fio)


def run_chat(messages: List[str], eliza: Optional[Eliza] = None,
             model=None, L=None, max_steps: int = 4000,
             verbose: bool = True) -> List[Turn]:
    """Multi-turn ELIZA conversation THROUGH THE TRANSFORMER, byte-exact-checked
    against the plain-python reference each turn.

    Builds the pure-forward model once (if not supplied), then for each user
    ``message`` runs the ELIZA program through ``model.forward`` (message in via
    the input-KV, reply out via PRTF) AND on the reference, and asserts the two
    replies are byte-identical.  Returns the list of ``Turn`` records.

    NOTE the STEP-COUNT WALL: the pure-forward VM does one ``model.forward`` over
    the WHOLE growing token stream PER VM step, so an N-step turn is O(N²) tokens.
    The full substring-scan ``build_eliza`` runs ~140-1200 steps/turn → only the
    first handful are tractable on CPU.  Pass ``eliza=build_chat_min()`` for the
    compact prefix-match variant (~15-40 steps/turn) that runs end-to-end.
    """
    eliza = eliza or build_chat_min()
    if model is None:
        from .nibble_pure_forward_complete import build_pure_forward_complete_model
        model, L = build_pure_forward_complete_model(
            code_size=len(eliza.code) + 2)
    turns: List[Turn] = []
    for msg in messages:
        ref = chat_turn_ref(eliza, msg)
        mdl = chat_turn_model(model, L, eliza, msg, max_steps=max_steps)
        ok = (ref == mdl)
        turns.append(Turn(user=msg, eliza_model=mdl, eliza_ref=ref, byte_exact=ok))
        if verbose:
            print(f"USER:  {msg}")
            print(f"ELIZA: {mdl}", end="" if mdl.endswith("\n") else "\n")
            print(f"       [byte-exact vs reference: {'YES' if ok else 'NO'}]")
            if not ok:
                print(f"       ref  = {ref!r}")
                print(f"       model= {mdl!r}")
    return turns


if __name__ == "__main__":
    # The transformer-tractable compact chat (prefix match, ~15-40 steps/turn).
    # For the FULL substring-scan ELIZA byte-exact on the plain-python reference,
    # use ``build_eliza()`` + ``chat_turn_ref`` (the model path is O(N²)-token slow
    # for the 140-1200-step scan turns; see ``run_chat``'s note).
    convo = ["sad here", "yes ok", "no thanks", "bye now"]
    run_chat(convo, eliza=build_chat_min())
