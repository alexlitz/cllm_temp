"""c4_min PRTF STRING-QUINE — a C4-C program that prints its own source, run on
the SMALL pure-forward transformer, with byte-exact self-output.

Authority: ``docs/BLOG_SPEC.md`` §"Making a Quine" (line 922, "the classic string
quine method") + §"Printing and Reading Input" (line 851, the think-tag output
contract). The output opcode is **PRTF** (printf, op 33 in the spec's ISA table,
line 148) — the spec's I/O opcode — NOT ``PUTCHAR``: PRTF single-char output is
``printf("%c", AX)`` and is what the classic C4 quine loop uses.

What "print" and "source" mean on this substrate
-------------------------------------------------
Output is the spec's think-tag protocol (§Printing): the 30-token register frames
are the model's internal *thinking* (inside THINK tags), and a ``PRTF`` step exits
the think block, emits ONE visible byte token (decoded from the model's own AX
byte-0 nibbles by the LM byte-head — a genuine argmax, not a python copy), then
re-enters thinking. The VISIBLE OUTPUT of a run is exactly those outside-think
bytes — the byte-stream a terminal would show (``blogspec_vocab.visible_output``).

The "source" of a c4_min program is its **code table** of ``(op, imm)`` cells (the
same object a baker bakes into weights). A STRING QUINE is a program whose VISIBLE
OUTPUT equals the ``WIDTH=2`` byte serialization of its own code table
``S = [op0, imm0, op1, imm1, ...]``.

The classic string-quine trick (data that encodes the code + code that prints it)
----------------------------------------------------------------------------------
  * Serialize the code table to a flat byte list ``S = [op0, imm0, ...]``.
  * Put that SAME byte list into DATA MEMORY as a table ``Q`` (one byte per cell at
    ``Q_BASE + i``) — the analogue of the string literal a textbook C quine embeds.
    On the pure-forward VM (memory is built from stores only, no python-seeded
    ``mem``) this is materialised as leading MEM-STORE frames (the ``seed_mem`` data
    segment the driver lays into the KV memory before step 0). It IS the program's
    own bytes — bundled with the program exactly as the data segment of a C quine.
  * The code is a loop that walks ``Q`` and emits each byte via ``PRTF`` for
    ``len(S)`` bytes. Because ``Q`` *is* ``S``, the emitted VISIBLE stream equals
    the source. The loop is "the code that prints the data"; ``Q`` is the
    self-referential encoding of the whole program.

Self-reference closure: the loop's length sets the byte count ``N`` (an ``IMM N``)
and the ``BZ`` exit target, but neither literal changes the instruction *count*, so
the fixed-point is reached in one pass (assemble the skeleton, read its length,
patch the two literals, then serialize — see ``build_quine``).

Addressing note (8-bit IMM)
---------------------------
``IMM`` masks its immediate to a byte (``AX = imm & 0xFF``) on the 8-bit AX path,
so all addresses formable by ``IMM`` are 0..255. ``Q`` lives in a low byte window
(``Q_BASE=0``) that never collides with the stack (which descends from ``SP_INIT``
and stays near the top), and ``SI``/``LI`` are 1-byte cells at the exact address
(the pure-forward memory reference stores/loads a single byte per address).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from . import isa


Q_BASE = 0x00       # source-byte table base (byte cells Q[i] = mem[Q_BASE+i])
PTR_CELL = 0xF0     # loop counter i lives here (0xF0 == SP_INIT; the stack DESCENDS
                    # below it so the counter cell is never a stack slot, and it is
                    # distinct from every Q address, which occupy 0..N-1, N < 0xF0).


def build_quine_prog() -> Tuple[List[Tuple[str, int]], int]:
    """Build the quine as a raw ``[(name, imm), ...]`` program.  Returns
    ``(prog, N)`` where ``N`` = source byte count = ``2 * len(prog)`` (the value the
    loop counts up to).  ``Q`` is walked from ``Q_BASE`` with ``LI`` and each byte
    printed with ``PRTF``; the counter ``i`` lives at ``mem[PTR_CELL]``.

    Memory convention (``nibble_pure_forward_complete.ref_interpret``):
        SI:  mem[pop()] = AX & 0xFF     (pop the address, store the byte AX)
        LI:  AX = mem[AX] & 0xFF        (load the byte at the address in AX)
    """
    src: List[Tuple[str, int]] = []

    def E(name, imm=0) -> int:
        src.append((name, imm))
        return len(src) - 1

    # --- init i = 0 : mem[PTR_CELL] = 0 -------------------------------------
    E("IMM", PTR_CELL)                  # 0: AX = &i
    E("PSH")                            # 1: push &i
    E("IMM", 0)                         # 2: AX = 0
    E("SI")                             # 3: mem[pop()=&i] = 0   -> i = 0

    TOP = len(src)                      # loop-head instruction index
    # --- test i == N ? -> BZ END --------------------------------------------
    E("IMM", PTR_CELL)                  # AX = &i
    E("LI")                             # AX = i
    E("PSH")                            # push i
    n_pc = E("IMM", 0)                  # AX = N       (patched: source byte count)
    E("SUB")                            # AX = pop() - AX = i - N   (0 iff i==N)
    bz_pc = E("BZ", 0)                  # if i == N goto END        (patched)

    # --- emit Q[Q_BASE + i] via PRTF (VISIBLE output) -----------------------
    E("IMM", PTR_CELL)                  # AX = &i
    E("LI")                             # AX = i
    E("PSH")                            # push i
    E("IMM", Q_BASE)                    # AX = Q_BASE
    E("ADD")                            # AX = pop() + AX = Q_BASE + i   (byte addr)
    E("LI")                             # AX = Q[i]    (the source byte)
    E("PRTF")                           # printf("%c", AX)  <-- EMIT visible byte

    # --- i = i + 1 : mem[PTR_CELL] = i + 1 ----------------------------------
    E("IMM", PTR_CELL)                  # AX = &i          [stack: &i]
    E("PSH")                            # push &i
    E("IMM", PTR_CELL)                  # AX = &i
    E("LI")                             # AX = i
    E("PSH")                            # push i           [stack: &i, i]
    E("IMM", 1)                         # AX = 1
    E("ADD")                            # AX = pop() + AX = i + 1
    E("SI")                             # mem[pop()=&i] = i + 1
    E("JMP", TOP)                       # loop

    END = len(src)
    E("HALT")

    # close the self-reference: N = 2 * instr count (WIDTH=2 bytes/instr)
    N = isa.WIDTH * len(src)
    src[n_pc] = ("IMM", N & 0xFF)
    src[bz_pc] = ("BZ", END)
    assert N < 0xF0, "quine too long: Q would collide with the stack/PTR_CELL"
    return src, N


def source_bytes(code: List[isa.Instr]) -> List[int]:
    """Flatten a code table to its WIDTH=2 byte serialization
    ``[op0, imm0, op1, imm1, ...]`` — the program's own source, as bytes."""
    out: List[int] = []
    for ins in code:
        out.append(ins.op & 0xFF)
        out.append(ins.imm & 0xFF)
    return out


def build_quine_source() -> Tuple[List[isa.Instr], int]:
    """Assemble the quine program into a code table.  Returns ``(code, N)``."""
    prog, N = build_quine_prog()
    return isa.assemble(prog), N


def build_quine() -> Tuple[List[isa.Instr], Dict[int, int], List[int]]:
    """Return ``(code, seed_mem, S)``: the quine code table, the data segment
    ``seed_mem = {Q_BASE+i: S[i]}`` (the program's own source bytes as the table
    ``Q``, the string-literal step of the classic quine), and the source byte list
    ``S``.  Running ``code`` with ``seed_mem`` on the model emits exactly ``S``."""
    code, _N = build_quine_source()
    S = source_bytes(code)
    seed_mem = {Q_BASE + i: b for i, b in enumerate(S)}
    return code, seed_mem, S


def run_reference() -> Tuple[List[int], List[int]]:
    """Run the quine on the REFERENCE interpreter (no transformer), via a byte-cell
    memory pre-seeded with ``Q``.  Returns ``(visible, S)`` where ``visible`` is the
    PRTF byte stream — proves the algorithm before the neural run."""
    code, seed_mem, S = build_quine()
    # SP-addressed byte-cell reference (matches the model's 1-byte LI/SI cells).
    from .nibble_pure_forward_complete import SP_INIT
    mem: Dict[int, int] = dict(seed_mem)
    sp = SP_INIT
    ax = pc = 0
    out: List[int] = []
    steps = 0
    while 0 <= pc < len(code) and steps < 200000:
        steps += 1
        ins = code[pc]; op, imm = ins.op, ins.imm; pc += 1
        if op == isa.IMM:      ax = imm & 0xFF
        elif op == isa.PSH:    sp -= 4; mem[sp] = ax & 0xFF
        elif op == isa.ADD:    v = mem.get(sp, 0); sp += 4; ax = (v + ax) & 0xFF
        elif op == isa.SUB:    v = mem.get(sp, 0); sp += 4; ax = (v - ax) & 0xFF
        elif op == isa.LI:     ax = mem.get(ax, 0) & 0xFF
        elif op == isa.SI:     addr = mem.get(sp, 0); sp += 4; mem[addr] = ax & 0xFF
        elif op == isa.PRTF:   out.append(ax & 0xFF)
        elif op == isa.JMP:    pc = imm
        elif op == isa.BZ:     pc = imm if ax == 0 else pc
        elif op == isa.HALT:   break
        else: raise NotImplementedError(isa.NAMES.get(op, op))
    return out, S


if __name__ == "__main__":
    code, seed_mem, S = build_quine()
    print("instructions: %d   source bytes: %d" % (len(code), len(S)))
    vis, S = run_reference()
    print("source :", S)
    print("printed:", vis)
    print("QUINE OK (reference visible output == own source):", vis == S)
