"""c4_min MODEL-RUNS-C, FULL PATH: a COMPILER in the weights/bytecode that reads
C SOURCE (as data) -> emits bytecode into memory -> runs it. No tool calls.

THE HEADLINE GOAL (BLOG_SPEC.md §"Model that Directly Runs C Code"): *C source in
-> the transformer compiles it to bytecode -> the transformer runs that bytecode.*
``nibble_handoff.py`` proved the compile-then-execute HANDOFF *mechanism* (EMIT a
word into code memory, JMP to it, the universal fetch runs it), but its generator
had the produced program's opcodes/immediates **hardcoded in its own bytecode** —
it was a code *emitter*, not a *compiler*: it never read any source.

This module closes that gap. It adds the ONE capability a real compiler needs that
the handoff machine lacked: **reading the SOURCE at a runtime address**. The C
source ``2+3*4`` is loaded into a data band ``SRC[i]`` (the "input file"); a new
``LC`` opcode (load char) selects the source byte at the address in AX:

    LC:  AX := sum_i AX_IS[i] * SRC[i]        # read source[AX]

— the exact bilinear read-select the universal fetch uses for code memory
(``WORD = sum_i PC_IS[i]*CODE_WORD[i]``), only the address one-hot is AX's value
and the memory band is the SOURCE. With ``LC`` the generator can be a genuine
compiler: it *scans* the source characters, recognises digit / ``+`` / ``*``
tokens, and EMITs the corresponding bytecode with correct operator precedence
(``*`` binds tighter than ``+``). Nothing about ``2+3*4`` is in the generator's
bytecode — the digits ``2``,``3``,``4`` and the operators are read out of the SRC
data band at runtime and turned into ``IMM``/``MUL``/``ADD`` words.

The full pipeline (one fixed-weight step-block, applied recurrently):

  1. the COMPILER bytecode (in code memory, OR baked into weights) runs. It:
       a. reads a source char via LC (source[cursor]),
       b. if it is a digit, computes the number and EMITs ``IMM d; PSH``,
       c. reads the operator; ``*`` is emitted deferred-after-operand-precedence,
          ``+`` after the following ``*``-term — yielding, for ``2+3*4``:
             IMM 2; PSH; IMM 3; PSH; IMM 4; MUL; ADD; HALT
       d. HANDOFF: JMP to the produced code.
  2. the UNIVERSAL FETCH at the produced PC reads the freshly-EMITted bytecode
     and runs it: 2+(3*4) == 14.

``MUL`` is wired into the universal dispatch here too (the handoff ISA had only
IMM/LEA/PSH/ADD/SUB/JMP/BZ/BNZ/HALT — a real ``2+3*4`` needs multiply).

Relationship to the FULL c4 compiler (bundler/c4_compile.c, 964 LOC): the real c4
compiler's ``emit(op,imm) { code[code_pos] = op | (imm<<8); }`` IS a sequence of
these EMITs, and its ``next()`` lexer reads source with the same ``src[pos]`` byte
loads that ``LC`` implements. This module builds a *minimal-subset* instance of the
same machine (integer ``+``/``*`` expressions), proving the end-to-end
C-source-in -> result-out path with the compiler in weights; the full-compiler bake
size is scoped in docs/NIBBLE_MODEL_RUNS_C_FULL_2026_07_14.md.

Opcode numbering note: ``nibble_handoff.EMIT`` reused value 27, which in the full
isa.py is ``MUL`` — a collision that was harmless for the handoff (it never used
MUL) but not here. We put ``EMIT`` on the free opcode 30 so EMIT and MUL coexist.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from .compiler import (VOCAB, _zero_attn, _load_ffn, _load_head, head_matrix,
                       vanilla_requantize)
from .compile_ffn import compile_ffn, compile_fold, S as FFN_S, RELU_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer
from . import universal as U

# EMIT on a FREE opcode (30..37 are unused in isa.py; 27 is MUL, which we need).
EMIT = 30
isa.NAMES.setdefault(EMIT, "EMIT")
isa.BY_NAME.setdefault("EMIT", EMIT)


# ------------------------------------------------------------------ layout ----

def build_compiler_layout(code_size: int, src_size: int, mem_size: int = 0,
                          stack_depth: int = 8, n_heads: int = 4) -> Layout:
    """Universal PACKED layout + EMIT store-addr one-hot + a SOURCE data band, an
    AX read-address one-hot (LC/LI), and (optionally) a general scratch DATA memory
    ``MEM[i]`` with a STACK0 store-address one-hot (SI).

    Adds on top of the handoff layout:
      * ``SRC[i]``    (src_size)  — the C SOURCE bytes, loaded as INPUT data (LC).
      * ``AX_IS[i]``  (max(src,mem)) — AX read-address one-hot ``AX_IS[i]=(AX==i)``:
        the read-address one-hot LC (source) and LI (mem) route on (mirror of PC_IS).
      * ``MEM[i]``    (mem_size)  — general scratch data memory (compiler variables),
        read by LI (``AX := MEM[AX]``), written by SI (``MEM[STACK0] := AX``).
      * ``MEM_IS[i]`` (mem_size)  — STACK0 store-address one-hot for SI.
    """
    L = Layout(n_heads=n_heads)
    L.CODE_SIZE = code_size
    L.SRC_SIZE = src_size
    L.MEM_SIZE = mem_size
    L.ADDR_MAX = max(src_size, mem_size)   # AX_IS covers both SRC and MEM indexing
    L.PACKED = True
    L.CODE_WORD = [L._band(f"CODE_WORD_{i}", 1) for i in range(code_size)]
    L.WORD = L._band("WORD", 1)
    L.PC_IS = [L._band(f"PC_IS_{i}", 1) for i in range(code_size)]
    L.ADDR_IS = [L._band(f"ADDR_IS_{i}", 1) for i in range(code_size)]  # EMIT store addr
    L.SRC = [L._band(f"SRC_{i}", 1) for i in range(src_size)]           # C source bytes
    L.AX_IS = [L._band(f"AX_IS_{i}", 1) for i in range(L.ADDR_MAX)]     # LC/LI read addr
    L.MEM = [L._band(f"MEM_{i}", 1) for i in range(mem_size)]           # scratch data mem
    L.MEM_IS = [L._band(f"MEM_IS_{i}", 1) for i in range(mem_size)]     # SI store addr
    # SP-indexed arbitrary-depth stack (replaces the single STACK0 mirror): the
    # produced program's IMM/PSH/ADD/MUL chain needs depth > 1 (2+3*4 is depth 2).
    L.STACK_DEPTH = stack_depth
    L.STACK = [L._band(f"STACK_{i}", 1) for i in range(stack_depth)]    # stack cells
    L.SP_IS = [L._band(f"SP_IS_{i}", 1) for i in range(stack_depth)]    # SP one-hot
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


# ------------------------------ EMIT store-address one-hot (ADDR_IS == IMM) ----

def compile_store_addr_onehot(L: Layout, dim: int):
    """ADDR_IS[i] = (IMM == i): the store-address one-hot for EMIT (over CODE_SIZE).

    Triangular-pulse over IMM (the EMIT instruction's own immediate == target slot).
    SET semantics (self-clear first). Same gadget as nibble_handoff.
    """
    n = L.CODE_SIZE
    silu_S = float(F.silu(torch.tensor(FFN_S)))
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    clear0 = n_relu
    n_units = clear0 + n
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for t, j in thr_unit.items():
        W_up[j, L.IMM] = RELU_S; b_up[j] = -RELU_S * t; W_gate[j, L.ONE] = 1.0
    for c in range(n):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, L.ADDR_IS[c]] = 1.0
        W_down[L.ADDR_IS[c], u] += -1.0 / silu_S
    for i in range(n):
        W_down[L.ADDR_IS[i], thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[L.ADDR_IS[i], thr_unit[i]] += -2.0 / RELU_S
        W_down[L.ADDR_IS[i], thr_unit[i + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ----------------------------- AX read-address one-hot (AX_IS == AX) for LC ----

def compile_ax_addr_onehot(L: Layout, dim: int):
    """AX_IS[i] = (AX == i): the read-address one-hot (mirror of PC_IS) shared by
    LC (source read) and LI (mem read).

    Triangular-pulse over AX (the load address). Range 0..ADDR_MAX-1. SET semantics
    (self-clear first). This is what turns AX into an address one-hot so the LC/LI
    read-selects can dot it against the SRC / MEM bands.
    """
    n = L.ADDR_MAX
    silu_S = float(F.silu(torch.tensor(FFN_S)))
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    clear0 = n_relu
    n_units = clear0 + n
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for t, j in thr_unit.items():
        W_up[j, L.AX] = RELU_S; b_up[j] = -RELU_S * t; W_gate[j, L.ONE] = 1.0
    for c in range(n):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, L.AX_IS[c]] = 1.0
        W_down[L.AX_IS[c], u] += -1.0 / silu_S
    for i in range(n):
        W_down[L.AX_IS[i], thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[L.AX_IS[i], thr_unit[i]] += -2.0 / RELU_S
        W_down[L.AX_IS[i], thr_unit[i + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ---------------------- indexed memory READ: AX := MEM_BAND[AX], gated on op ----

def _compile_indexed_read(L: Layout, dim: int, op: int, mem_bands: List[int]):
    """``AX := sum_i AX_IS[i] * MEM_BAND[i]`` when OP_IS[op] fires; PC += 1.

    The generic bilinear read-select over any memory band list, addressed by the
    AX one-hot ``AX_IS`` (mirror of the fetch's code read). Used for LC (band=SRC)
    and LI (band=MEM). A 2-input AND guard (OP_IS[op] AND AX_IS[i]) sharp-silu ``up``
    gate; the memory value rides the linear ``gate``. Untouched when op mismatches.
    """
    n = len(mem_bands)
    BIG = 256.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))   # == 128.0 exactly
    # single-band guard up = FFN_S*(guard-0.5) => ON hidden == silu(0.5*FFN_S).
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    op_band = L.OP_IS + op
    n_units = n + 2                       # n products + clear-AX + PC-bump (both on op)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_band, coeff, dst):
        W_up[u, op_band] += BIG
        W_up[u, addr_band] += BIG
        b_up[u] += -BIG * 1.5
        W_gate[u, gate_band] += 1.0
        W_down[dst, u] += coeff / silu_big

    u = 0
    for i in range(n):
        _and_unit(u, L.AX_IS[i], mem_bands[i], +1.0, L.AX); u += 1
    # clear AX_old on op (SET). up = FFN_S*(OP_IS[op]-0.5); gate = AX_old.
    W_up[u, op_band] += FFN_S; b_up[u] += -FFN_S * 0.5
    W_gate[u, L.AX] += 1.0
    W_down[L.AX, u] += -1.0 / silu_S
    u += 1
    # PC += 1 on op.
    W_up[u, op_band] += FFN_S; b_up[u] += -FFN_S * 0.5
    W_gate[u, L.ONE] += 1.0
    W_down[L.PC, u] += 1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_lc_source_read(L: Layout, dim: int):
    """LC: ``AX := SRC[AX]`` (read the C source char at address AX). The compiler's
    window into its input file — the same ``src[pos]`` load the c4 lexer does."""
    return _compile_indexed_read(L, dim, isa.LC, L.SRC[: L.SRC_SIZE])


def compile_li_mem_read(L: Layout, dim: int):
    """LI: ``AX := MEM[AX]`` (read a compiler variable from scratch data memory)."""
    if L.MEM_SIZE == 0:
        return _empty_ffn(dim)
    return _compile_indexed_read(L, dim, isa.LI, L.MEM[: L.MEM_SIZE])


def compile_mem_store_addr_onehot(L: Layout, dim: int):
    """MEM_IS[i] = (STACK0 == i): the SI store-address one-hot (indexed by STACK0,
    the popped address). Triangular pulse over STACK0. SET semantics."""
    if L.MEM_SIZE == 0:
        return _empty_ffn(dim)
    n = L.MEM_SIZE
    silu_S = float(F.silu(torch.tensor(FFN_S)))
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    clear0 = n_relu
    n_units = clear0 + n
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for t, j in thr_unit.items():
        W_up[j, L.STACK0] = RELU_S; b_up[j] = -RELU_S * t; W_gate[j, L.ONE] = 1.0
    for c in range(n):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, L.MEM_IS[c]] = 1.0
        W_down[L.MEM_IS[c], u] += -1.0 / silu_S
    for i in range(n):
        W_down[L.MEM_IS[i], thr_unit[i - 1]] += 1.0 / RELU_S
        W_down[L.MEM_IS[i], thr_unit[i]] += -2.0 / RELU_S
        W_down[L.MEM_IS[i], thr_unit[i + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_si_mem_store(L: Layout, dim: int):
    """SI: ``MEM[STACK0] := AX`` (store a compiler variable into scratch data mem),
    gated on OP_IS[SI]. Store-SELECT over MEM addressed by the MEM_IS[i]=(STACK0==i)
    one-hot. PC += 1. In the c4 ISA SI pops the address off the stack; here the
    address rides STACK0 (the stack-top mirror).
    """
    if L.MEM_SIZE == 0:
        return _empty_ffn(dim)
    n = L.MEM_SIZE
    BIG = 256.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))   # == 128.0
    op_si = L.OP_IS + isa.SI
    n_units = 2 * n + 1                  # (set AX, clear old) per slot + PC-bump
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_band, coeff, dst):
        W_up[u, op_si] += BIG
        W_up[u, addr_band] += BIG
        b_up[u] += -BIG * 1.5
        W_gate[u, gate_band] += 1.0
        W_down[dst, u] += coeff / silu_big

    u = 0
    for i in range(n):
        _and_unit(u, L.MEM_IS[i], L.AX, +1.0, L.MEM[i]); u += 1       # MEM[i] += AX
        _and_unit(u, L.MEM_IS[i], L.MEM[i], -1.0, L.MEM[i]); u += 1   # clear old (SET)
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    W_up[u, op_si] += FFN_S; b_up[u] += -FFN_S * 0.5
    W_gate[u, L.ONE] += 1.0
    W_down[L.PC, u] += 1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def _empty_ffn(dim: int):
    """A no-op FFN block (one dead unit) — used when MEM_SIZE==0 disables LI/SI."""
    W_up = torch.zeros(1, dim); b_up = torch.full((1,), -1e4)
    W_gate = torch.zeros(1, dim); b_gate = torch.zeros(1)
    W_down = torch.zeros(dim, 1); b_down = torch.zeros(dim)
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------- EMIT store-select into code memory ----

def compile_emit_store(L: Layout, dim: int):
    """EMIT: ``CODE_WORD[IMM] := AX + 256*STACK0``, gated on OP_IS[EMIT]. PC += 1;
    SP -= 1 (pop the produced-immediate that rode STACK0).

    Same store-SELECT algebra as nibble_handoff.compile_emit_store (write-mirror of
    the fetch read-select over the shared CODE_WORD band), but EMIT is opcode 30
    here (27 == MUL) and it POPS the immediate it consumed so a compiler that emits
    many instructions (``PSH imm; IMM op; EMIT`` per produced word) keeps a balanced
    stack — otherwise every emit would leak one cell and overflow the SP-stack.
    """
    n = L.CODE_SIZE
    BIG = 256.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))   # == 128.0 exactly
    op_emit = L.OP_IS + EMIT
    n_units = 3 * n + 2       # 3 store units/slot + PC-bump + SP-pop (the immediate)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_terms, dst, coeff):
        W_up[u, op_emit] += BIG
        W_up[u, addr_band] += BIG
        b_up[u] += -BIG * 1.5
        for band, coeff_g in gate_terms:
            W_gate[u, band] += coeff_g
        W_down[dst, u] += coeff / silu_big

    u = 0
    for i in range(n):
        _and_unit(u, L.ADDR_IS[i], [(L.AX, 1.0)], L.CODE_WORD[i], +1.0); u += 1
        _and_unit(u, L.ADDR_IS[i], [(L.STACK0, 256.0)], L.CODE_WORD[i], +1.0); u += 1
        _and_unit(u, L.ADDR_IS[i], [(L.CODE_WORD[i], 1.0)], L.CODE_WORD[i], -1.0); u += 1
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    # PC += 1 on EMIT.
    W_up[u, op_emit] += FFN_S; b_up[u] += -FFN_S * 0.5
    W_gate[u, L.ONE] += 1.0
    W_down[L.PC, u] += 1.0 / silu_S; u += 1
    # SP -= 1 on EMIT (pop the produced-immediate the store consumed from STACK0).
    W_up[u, op_emit] += FFN_S; b_up[u] += -FFN_S * 0.5
    W_gate[u, L.ONE] += 1.0
    W_down[L.SP, u] += -1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ------------------------------------------- MUL dispatch (STACK0 * AX) --------

def compile_mul_dispatch(L: Layout, dim: int):
    """MUL: ``AX := STACK0 * AX`` (folded mod 256 downstream), gated on OP_IS[MUL].

    A single bilinear product ``STACK0 * AX`` (guarded by the decoded opcode) plus a
    clear of AX's old value and PC += 1. The product ``STACK0*AX`` reaches
    255*255 = 65025 < 2**24, exact in fp32; we normalise with the exact power-of-two
    ``POW2 = 256`` (``silu(256)=256``, ``1/256`` exact). The result is folded mod 256
    by the shared fold block, matching the 8-bit reference ``(pop()*ax) & 0xFF``.

    Guard = OP_IS[MUL] via a sharp-silu ``up`` gate; the product value rides the
    linear ``gate = STACK0``, whose ``up`` is ``POW2*AX`` scaled — but we need the
    guard too. We use a 3-input structure: up encodes (OP_IS[MUL] AND a strong AX
    ramp); simpler and exact is to gate the whole product on OP_IS[MUL] with a
    2-factor SwiGLU: ``silu(up)*gate`` with ``up`` carrying AX (so silu(POW2*AX)
    = POW2*AX for AX>=1, 0 for AX==0) and ``gate = STACK0``, then a SEPARATE opcode
    mask multiplies it. Since a single unit gives only one product, we fold the
    opcode guard into the down-route by making the whole block a no-op unless MUL:
    we bake ``up = POW2*AX`` and ``gate = STACK0`` for the product, but ROUTE it
    through the opcode mask by adding an ``-huge*(1-OP_IS[MUL])`` term into ``up``
    so silu saturates to 0 when the op isn't MUL. Concretely up has two regimes:
      op==MUL:  up = POW2*AX               -> silu(up) = POW2*AX  (AX in 0..255)
      op!=MUL:  up = POW2*AX - GATEOFF     -> silu(up) ~= 0        (GATEOFF huge)
    """
    POW2 = 256.0
    op_mul = L.OP_IS + isa.MUL
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    GATEOFF = 1.0e6           # push up strongly negative when op != MUL
    # product(masked) + clear-AX + PC-bump + SP-1 (pop the multiplicand) on MUL
    n_units = 4
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    # unit 0: masked product. up = POW2*AX + GATEOFF*(OP_IS[MUL]-1) so:
    #   MUL:  up = POW2*AX          -> silu(up)=POW2*AX (AX in 0..255, exact int)
    #   !MUL: up = POW2*AX-GATEOFF  -> silu(up)~=0
    # gate = STACK0 ; down = (1/POW2)*silu(up)*gate = AX*STACK0 on MUL, 0 else.
    W_up[0, L.AX] = POW2
    W_up[0, op_mul] = GATEOFF
    b_up[0] = -GATEOFF
    W_gate[0, L.STACK0] = 1.0
    W_down[L.AX, 0] += 1.0 / POW2
    # unit 1: clear AX_old on MUL. up = S*(OP_IS[MUL]-0.5); gate = AX_old.
    W_up[1, op_mul] = FFN_S; b_up[1] = -FFN_S * 0.5
    W_gate[1, L.AX] = 1.0
    W_down[L.AX, 1] += -1.0 / silu_S
    # unit 2: PC += 1 on MUL.
    W_up[2, op_mul] = FFN_S; b_up[2] = -FFN_S * 0.5
    W_gate[2, L.ONE] = 1.0
    W_down[L.PC, 2] += 1.0 / silu_S
    # unit 3: SP -= 1 on MUL (pop the multiplicand that STACK0 mirrored).
    W_up[3, op_mul] = FFN_S; b_up[3] = -FFN_S * 0.5
    W_gate[3, L.ONE] = 1.0
    W_down[L.SP, 3] += -1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------- stack-aware dispatch (SP-indexed) -----

def compiler_dispatch_rules(L: Layout) -> List[FFNRule]:
    """Universal dispatch, but PSH/ADD/SUB use the SP-INDEXED stack so nested
    expressions (depth > 1) evaluate correctly.

    vs ``universal_dispatch_rules`` (single STACK0 mirror):
      * PSH:  STACK[SP] := AX (per-cell, gated on OP_IS[PSH] AND SP_IS[d]); SP += 1.
      * ADD:  AX := STACK0 + AX ; SP -= 1        (STACK0 == STACK[SP-1] mirror)
      * SUB:  AX := STACK0 - AX ; SP -= 1
    IMM/LEA/JMP/BZ/BNZ/HALT are unchanged (they don't touch the stack). MUL's SP-1
    lives in its own block; the STACK0 mirror is (re)derived by compile_sp_fetch +
    compile_stack0_select each step from SP and the STACK cells.
    """
    from .stack import sp_stack_writes
    ax, stk, pc = L.AX, L.STACK0, L.PC
    imm, bp, sp = L.IMM, L.BP, L.SP
    OP = L.OP_IS

    def G(op):
        return [(OP + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    # IMM: AX = imm
    rules.append(FFNRule(G(isa.IMM),
                         {ax: LinearExpr.of(imm, 1.0) + LinearExpr.of(ax, -1.0),
                          pc: LinearExpr.c(1.0)}))
    # LEA: AX = BP + imm
    rules.append(FFNRule(G(isa.LEA),
                         {ax: LinearExpr.of(bp, 1.0) + LinearExpr.of(imm, 1.0)
                          + LinearExpr.of(ax, -1.0),
                          pc: LinearExpr.c(1.0)}))
    # PSH: STACK[SP] := AX (per-cell), SP += 1. Guard the per-cell write on the
    # DECODED OP_IS[PSH] (the "pc-is" slot in sp_stack_writes is a bool guard band).
    rules += sp_stack_writes(OP + isa.PSH, ax, L.STACK, L.SP_IS)
    rules.append(FFNRule(G(isa.PSH),
                         {sp: LinearExpr.c(1.0), pc: LinearExpr.c(1.0)}))
    # ADD: AX = STACK0 + AX ; SP -= 1 (pop)
    rules.append(FFNRule(G(isa.ADD),
                         {ax: LinearExpr.of(stk, 1.0),
                          sp: LinearExpr.c(-1.0), pc: LinearExpr.c(1.0)}))
    # SUB: AX = STACK0 - AX (+256 mod fold downstream) ; SP -= 1
    rules.append(FFNRule(G(isa.SUB),
                         {ax: LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0)
                          + LinearExpr.c(256.0),
                          sp: LinearExpr.c(-1.0), pc: LinearExpr.c(1.0)}))
    # JMP: PC = imm
    rules.append(FFNRule(G(isa.JMP),
                         {pc: LinearExpr.of(imm, 1.0) + LinearExpr.of(pc, -1.0)}))
    # BZ / BNZ: PC update deferred to compile_branch_delta (bilinear).
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    # HALT: latch HALTED, freeze PC.
    rules.append(FFNRule(G(isa.HALT), {L.HALTED: LinearExpr.c(1.0)}))
    return rules


# ------------------------------------------------------- the compiler step -----

def build_compiler_step(code_size: int, src_size: int, mem_size: int = 0,
                        stack_depth: int = 8, n_heads: int = 4, max_pos: int = 4):
    """Universal PACKED interpreter + SP-indexed stack + LC/LI (read source/mem)
    + SI (write mem) + EMIT (write code) + MUL. ONE fixed-weight step-block.

    Physical FFN sub-blocks (single residual position, no attention):
      1.  fetch      : PC -> PC_IS[i] + AX_ZERO.
      2.  word_select: WORD = sum_i PC_IS[i]*CODE_WORD[i]   (read code memory).
      3a. word>>8    : IMM = WORD>>8.
      3b. word&0xff  : OP_VAL = WORD&0xFF.
      3c. decode     : OP_VAL -> OP_IS[op] one-hot.
      3d. store1hot  : ADDR_IS[i] = (IMM == i)     (EMIT target, over CODE_SIZE).
      3e. read1hot   : AX_IS[i]   = (AX  == i)     (LC/LI read address, ADDR_MAX).
      3f. mem1hot    : MEM_IS[i]  = (STACK0 == i)  (SI store address, MEM_SIZE).
      4.  dispatch   : universal per-opcode rules (IMM/LEA/PSH/ADD/SUB/JMP/HALT).
      5.  mul        : MUL -> AX := STACK0*AX.
      6.  lc         : LC  -> AX := SRC[AX]   (READ SOURCE).
      7.  li         : LI  -> AX := MEM[AX]   (read compiler variable).
      8.  si         : SI  -> MEM[STACK0] := AX (write compiler variable).
      9.  emit_store : EMIT -> CODE_WORD[IMM] := AX+256*STACK0 (WRITE code) + PC+1.
      10. branch     : BZ/BNZ bilinear PC update.
      11. fold+emit  : AX mod-256, AX -> OUT slot, HALTED -> HALT_SEEN.

    SRC is read-only input; MEM is scratch variables; CODE_WORD is the shared
    read/write code memory (fetch reads, EMIT writes). Returns ``(model, L)``.
    ``mem_size=0`` drops LI/SI (no compiler variables). Returns ``(model, L)``.
    """
    from .stack import compile_sp_fetch, compile_stack0_select
    L = build_compiler_layout(code_size, src_size, mem_size=mem_size,
                              stack_depth=stack_depth, n_heads=n_heads)
    dim = L.D

    ffn_specs = [
        U.compile_fetch_select(L, dim),          # 1. PC one-hot + AX_ZERO
        U.compile_word_select(L, dim),           # 2. WORD <- code memory at PC
        U.compile_word_decode_imm(L, dim),       # 3a. IMM = WORD>>8
        U.compile_word_decode_op(L, dim),        # 3b. OP_VAL = WORD&0xFF
        U.compile_opcode_decode(L, dim),         # 3c. OP_VAL -> OP_IS[op]
        # SP-indexed stack: SP_IS one-hot then STACK0 = STACK[SP-1] mirror. Runs
        # BEFORE the address one-hots (MEM_IS reads STACK0) and dispatch.
        compile_sp_fetch(L.SP, L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),   # 3s1
        compile_stack0_select(L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),    # 3s2
        compile_store_addr_onehot(L, dim),       # 3d. ADDR_IS[i] = (IMM==i)
        compile_ax_addr_onehot(L, dim),          # 3e. AX_IS[i]   = (AX==i)
        compile_mem_store_addr_onehot(L, dim),   # 3f. MEM_IS[i]  = (STACK0==i)
        compile_ffn(compiler_dispatch_rules(L), dim),  # 4. dispatch (SP-indexed)
        compile_mul_dispatch(L, dim),            # 5. MUL
        compile_lc_source_read(L, dim),          # 6. LC -> read SRC[AX]
        compile_li_mem_read(L, dim),             # 7. LI -> read MEM[AX]
        compile_si_mem_store(L, dim),            # 8. SI -> write MEM[STACK0]
        compile_emit_store(L, dim),              # 9. EMIT -> write code memory
        U.compile_branch_delta(L, dim),          # 10. BZ/BNZ bilinear PC
        compile_fold(L.AX, L.ONE, dim, modulus=256),      # 11a. AX mod-256
        compile_ffn([                            # 11b. emit + halt snapshot
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


# ============================================================================ #
#  COMPILER IN THE WEIGHTS: the hybrid baked fetch                             #
# ============================================================================ #

def compile_hybrid_word_select(L: Layout, gen_code: List[isa.Instr], dim: int):
    """Hybrid WORD select — the COMPILER lives in the WEIGHTS, produced code in mem.

        WORD = Σ_{i<GEN} PC_IS[i]·word_i(BAKED constant)          # compiler code
             + Σ_{i>=GEN} PC_IS[i]·CODE_WORD[i](DATA memory)       # produced code

    For the compiler's own slots (``i < len(gen_code)``) the instruction word
    ``word_i = op_i | imm_i<<8`` is a CONSTANT baked into the weights (it rides the
    unit's constant gate bias, gated on the ``PC_IS[i]`` detector) — the spec's
    read-only code segment. For the produced slots (``i >= GEN``) the word is read
    from the writable ``CODE_WORD`` data band exactly like ``compile_word_select``,
    so EMIT can fill it at runtime. There is NO compiler bytecode in the input
    state; ``initial_state_baked`` writes only the SOURCE.

    So a single fetch dots the PC one-hot against a code table that is HALF weights
    (the compiler) and HALF memory (the program it produces). WORD self-clears
    first (SET). Precision: exact power-of-two ``POW2=256`` normaliser.
    """
    n = L.CODE_SIZE
    gen = len(gen_code)
    POW2 = 256.0                                  # silu(256)=256 exactly; 1/256 exact
    # gen baked-word units + (n-gen) memory-product units + 1 self-clear
    n_units = n + 1
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for i in range(n):
        if i < gen:
            # BAKED: WORD += PC_IS[i] * word_i  (word_i is a constant on the gate
            # bias). EXACT power-of-two normaliser: silu(256*PC_IS[i]) = 256 iff
            # PC==i, gate = word_i constant, down = 1/256 -> WORD += word_i (exact,
            # 256*65535 < 2**24). Using silu(60)/60 here injected a WORD-proportional
            # residue that flipped the produced word's low byte (0x201 -> 0x200).
            word_i = float((gen_code[i].op & 0xFF) | ((gen_code[i].imm & 0xFF) << 8))
            W_up[i, L.PC_IS[i]] = POW2           # detector: silu(256)=256 iff PC==i
            b_gate[i] = word_i                   # gate = constant baked word
            W_down[L.WORD, i] += 1.0 / POW2
        else:
            # MEMORY: WORD += PC_IS[i] * CODE_WORD[i]  (the produced program)
            W_up[i, L.PC_IS[i]] = POW2            # silu(256)=256; gate on PC one-hot
            W_gate[i, L.CODE_WORD[i]] = 1.0
            W_down[L.WORD, i] += 1.0 / POW2
    # self-clear WORD (SET)
    W_up[n, L.ONE] = POW2
    W_gate[n, L.WORD] = 1.0
    W_down[L.WORD, n] += -1.0 / POW2
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def build_baked_compiler_step(gen_code: List[isa.Instr], code_size: int,
                              src_size: int, mem_size: int = 0,
                              stack_depth: int = 8, n_heads: int = 4,
                              max_pos: int = 4):
    """The COMPILER-IN-WEIGHTS model. Identical to ``build_compiler_step`` except
    the code-fetch block is the HYBRID baked/memory word-select: the compiler
    bytecode ``gen_code`` is baked into the WEIGHTS, the produced program is read
    from CODE_WORD memory. The input state carries ONLY the C source — no compiler
    bytecode anywhere in the data. Returns ``(model, L)``.
    """
    from .stack import compile_sp_fetch, compile_stack0_select
    L = build_compiler_layout(code_size, src_size, mem_size=mem_size,
                              stack_depth=stack_depth, n_heads=n_heads)
    L.GEN_SIZE = len(gen_code)
    dim = L.D

    ffn_specs = [
        U.compile_fetch_select(L, dim),
        compile_hybrid_word_select(L, gen_code, dim),     # <-- compiler in WEIGHTS
        U.compile_word_decode_imm(L, dim),
        U.compile_word_decode_op(L, dim),
        U.compile_opcode_decode(L, dim),
        compile_sp_fetch(L.SP, L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_stack0_select(L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_store_addr_onehot(L, dim),
        compile_ax_addr_onehot(L, dim),
        compile_mem_store_addr_onehot(L, dim),
        compile_ffn(compiler_dispatch_rules(L), dim),
        compile_mul_dispatch(L, dim),
        compile_lc_source_read(L, dim),
        compile_li_mem_read(L, dim),
        compile_si_mem_store(L, dim),
        compile_emit_store(L, dim),
        U.compile_branch_delta(L, dim),
        compile_fold(L.AX, L.ONE, dim, modulus=256),
        compile_ffn([
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)
    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


def initial_state_baked(model, L, src: List[int]) -> torch.Tensor:
    """Initial state for the BAKED compiler: ONLY the C source in the SRC band.

    THE PROOF SURFACE: unlike ``load_program`` (which writes the compiler bytecode
    into CODE_WORD), this writes NO compiler bytecode — the compiler is entirely in
    the weights. CODE_WORD starts all-zero (the produced program is not there; it
    is EMITted at runtime). Only the source is input.
    """
    assert len(src) <= L.SRC_SIZE
    state = model.embed[0].clone()
    for i, ch in enumerate(src):
        state[L.SRC[i]] = float(ch & 0xFF)
    return state


def run_baked_compiler(model, L, src: List[int], max_steps: int = 8192,
                       requantize: bool = True, return_code: bool = False):
    """Run the COMPILER-IN-WEIGHTS on C ``src`` — with NO compiler bytecode in the
    input state (only the source). The weights ARE the compiler."""
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]
    state = initial_state_baked(model, L, src)
    trace: List[int] = []
    for _ in range(max_steps):
        state = _step_once(model, state)
        if requantize:
            state = _requantize(state, L)
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if return_code:
        words = [int(round(float(state[L.CODE_WORD[i]]))) for i in range(L.CODE_SIZE)]
        return trace, words
    return trace


class BakedCompilerMachine:
    """A C COMPILER that IS a transformer: the compiler bytecode is baked into the
    WEIGHTS. Feed only the C source; the weights read it, emit bytecode into memory,
    and run it — no compiler bytecode anywhere in the input.

        bcm = BakedCompilerMachine(expr_compiler_bytecode(), code_size=176, src_size=8)
        bcm.run("2+3*4")   # -> trace[-1] == 14, with NO bytecode in the input
    """

    def __init__(self, prog, code_size: int, src_size: int, mem_size: int = 8,
                 stack_depth: int = 8, n_heads: int = 4):
        self.gen_code = _assemble(prog)
        self.code_size = code_size
        self.model, self.L = build_baked_compiler_step(
            self.gen_code, code_size, src_size, mem_size=mem_size,
            stack_depth=stack_depth, n_heads=n_heads)

    def run(self, source, max_steps: int = 8192, requantize: bool = True,
            return_code: bool = False):
        return run_baked_compiler(self.model, self.L, _source_bytes(source),
                                  max_steps=max_steps, requantize=requantize,
                                  return_code=return_code)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)


# --------------------------------------------------- reference interpreter -----

def interpret_full(code: List[isa.Instr], src: List[int], code_size: int,
                   mem_size: int = 256, max_steps: int = 8192):
    """Reference interpreter mirroring the compiler MODEL EXACTLY: the handoff ISA
    (IMM/LEA/ADD/SUB/MUL/JMP/BZ/BNZ/HALT/EMIT) with an SP-INDEXED stack (PSH/pop)
    plus LC (read source[AX]), LI (read mem[AX]) and SI (write mem[STACK0]).

    Stack semantics match the model's SP-indexed cells (SP is a depth counter, top
    == STACK[SP-1]):
      PSH: STACK[SP]=AX ; SP+=1        ADD/SUB/MUL: use STACK[SP-1] then SP-=1
    Memory:
      LI:  AX := mem[AX]               SI: mem[STACK0] := AX   (address = stack top,
                                                                NOT popped)
      LC:  AX := src[AX]               EMIT: code[imm] := (AX&0xFF)|((STACK0&0xFF)<<8)

    Returns (per-step AX trace, final code table).
    """
    ax = bp = sp = 0
    pc = 0
    stack = [0] * (mem_size + 8)
    mem = [0] * mem_size
    words = [0] * code_size
    for i, ins in enumerate(code):
        words[i] = (ins.op & 0xFF) | ((ins.imm & 0xFF) << 8)
    source = list(src) + [0] * code_size            # tolerate over-read as 0
    emitted = []

    def top():                                      # STACK0 mirror == STACK[SP-1]
        return stack[sp - 1] if sp >= 1 else 0

    steps = 0
    while pc < code_size and steps < max_steps:
        steps += 1
        w = words[pc]
        op = w & 0xFF
        imm = (w >> 8) & 0xFF
        stack0 = top()
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + imm) & 0xFF
        elif op == isa.PSH:
            stack[sp] = ax & 0xFF; sp += 1
        elif op == isa.ADD:
            ax = (top() + ax) & 0xFF; sp -= 1
        elif op == isa.SUB:
            ax = (top() - ax) & 0xFF; sp -= 1
        elif op == isa.MUL:
            ax = (top() * ax) & 0xFF; sp -= 1
        elif op == isa.LC:
            ax = source[ax] & 0xFF if ax < len(source) else 0
        elif op == isa.LI:
            ax = mem[ax] & 0xFF if ax < len(mem) else 0
        elif op == isa.SI:
            if stack0 < len(mem):
                mem[stack0] = ax & 0xFF
        elif op == EMIT:
            words[imm % code_size] = (ax & 0xFF) | ((stack0 & 0xFF) << 8)
            sp = max(0, sp - 1)          # EMIT pops the produced-immediate it used
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            emitted.append(ax); break
        else:
            raise NotImplementedError(f"op {op} not in compiler ISA")
        emitted.append(ax)
    return emitted, words


# ------------------------------------------------------------- run driver ------

def _step_once(model, state):
    x = state.view(1, 1, -1)
    for blk in model.blocks:
        x = blk(x)
    return x[0, 0]


def _requantize(state, L):
    """Snap every band to its exact integer via the VANILLA LM-head argmax
    (``argmax_v (2*v*x - v^2)`` — the model's own emit-token snap, NO
    ``torch.round``), then pin ONE=1. The vocab covers the packed CODE_WORD data
    band (the compiler bytecode + the freshly-EMITted target words up to 0xFFFF),
    so the program-in-data and produced code are preserved byte-exact."""
    return vanilla_requantize(state, L.ONE)


def load_program(model, L, code: List[isa.Instr], src: List[int]) -> torch.Tensor:
    """Initial state: ``code`` (the COMPILER bytecode) in CODE_WORD memory, the C
    ``src`` bytes in the SOURCE band. Produced-code slots stay 0 (empty)."""
    assert len(code) <= L.CODE_SIZE
    assert len(src) <= L.SRC_SIZE
    state = model.embed[0].clone()
    for i, ins in enumerate(code):
        state[L.CODE_WORD[i]] = float((ins.op & 0xFF) | ((ins.imm & 0xFF) << 8))
    for i, ch in enumerate(src):
        state[L.SRC[i]] = float(ch & 0xFF)
    return state


def run_compiler(model, L, code: List[isa.Instr], src: List[int],
                 max_steps: int = 8192, requantize: bool = True,
                 return_code: bool = False):
    """Run the compiler step-block recurrently on ``code`` (compiler bytecode) with
    ``src`` (C source) loaded as data. Returns the per-step AX trace; if
    ``return_code`` also returns the final CODE_WORD memory (the produced bytecode).
    """
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]
    state = load_program(model, L, code, src)
    trace: List[int] = []
    for _ in range(max_steps):
        state = _step_once(model, state)
        if requantize:
            state = _requantize(state, L)
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if return_code:
        words = [int(round(float(state[L.CODE_WORD[i]]))) for i in range(L.CODE_SIZE)]
        return trace, words
    return trace


class CompilerMachine:
    """ONE fixed-weight interpreter that COMPILES C source to bytecode and runs it.

        cm = CompilerMachine(code_size=64, src_size=16)
        trace = cm.run(compiler_bytecode, source="2+3*4")   # -> ... trace[-1] == 14

    The compiler bytecode reads the source out of the SRC data band (LC), emits the
    produced program (EMIT), and JMPs to it (the universal fetch runs it). Compile
    ONCE (build the model); run MANY sources / compilers by swapping the data bands.
    """

    def __init__(self, code_size: int, src_size: int, mem_size: int = 0,
                 stack_depth: int = 8, n_heads: int = 4):
        self.code_size = code_size
        self.src_size = src_size
        self.mem_size = mem_size
        self.model, self.L = build_compiler_step(
            code_size, src_size, mem_size=mem_size, stack_depth=stack_depth,
            n_heads=n_heads)

    def run(self, prog, source, max_steps: int = 8192, requantize: bool = True,
            return_code: bool = False):
        code = _assemble(prog)
        src = _source_bytes(source)
        return run_compiler(self.model, self.L, code, src, max_steps=max_steps,
                            requantize=requantize, return_code=return_code)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)


def _assemble(prog) -> List[isa.Instr]:
    """assemble that also understands the EMIT mnemonic (opcode 30)."""
    out = []
    for entry in prog:
        name, imm = (entry if isinstance(entry, tuple) else (entry, 0))
        op = EMIT if name == "EMIT" else isa.BY_NAME[name]
        out.append(isa.Instr(op, imm & 0xFF))
    return out


def _source_bytes(source) -> List[int]:
    if isinstance(source, str):
        return [ord(c) for c in source]
    return list(source)


def make_word(op_name: str, imm: int) -> int:
    op = EMIT if op_name == "EMIT" else isa.BY_NAME[op_name]
    return (op & 0xFF) | ((imm & 0xFF) << 8)


# ======================================================================== #
#  THE MINIMAL-SUBSET C COMPILER, IN c4_min BYTECODE                        #
# ======================================================================== #
#
#  This is the headline artifact: a COMPILER written in c4_min bytecode that,
#  when run on ``CompilerMachine`` (or baked into the weights), reads a C
#  integer expression out of the SOURCE data band, parses it with correct
#  operator precedence (``*`` binds tighter than ``+``), EMITs the compiled
#  bytecode into empty code memory, and JMPs to it — the universal fetch then
#  runs the freshly-produced program. NOTHING about the input expression is in
#  this bytecode: the operand digits and the operators (hence the precedence
#  decisions) are all read from SRC at runtime via ``LC``.
#
#  Grammar (minimal C-subset instance of the c4 expr()/stmt() descent): a
#  three-operand two-operator integer expression ``D op D op D`` over single
#  ASCII digits and the operators ``+`` and ``*`` (e.g. ``2+3*4``). All four
#  precedence cases (+/+, +/*, */+, */*) are handled by RUNTIME branches on the
#  operator characters — exactly the precedence lookahead the c4 compiler does.
#  The produced code is the SAME ``op | imm<<8`` packed encoding the real c4
#  compiler emits (``code[code_pos] = op | (imm<<8)``, c4_compile.c:352): for
#  ``2+3*4`` this is ``IMM 2; PSH; IMM 3; PSH; IMM 4; MUL; ADD; HALT`` == 14,
#  byte-identical to c4's own output (JSR/ENT/LEV framing aside).

# Where the produced program is EMITted (must be PAST the compiler's own code).
COMPILER_OUTBASE = 160
# ASCII of the two operators the compiler recognises.
_STAR = ord("*")   # 42
_PLUS = ord("+")   # 43


class _Asm:
    """Tiny label assembler for hand-writing the compiler bytecode."""

    def __init__(self):
        self.code: List[list] = []
        self.labels: dict = {}

    def emit(self, op: int, imm=0) -> int:
        self.code.append([op, imm])
        return len(self.code) - 1

    def label(self, name: str):
        self.labels[name] = len(self.code)

    def resolve(self) -> List[Tuple[str, int]]:
        out = []
        for op, imm in self.code:
            if isinstance(imm, str):
                imm = self.labels[imm]
            name = "EMIT" if op == EMIT else isa.NAMES[op]
            out.append((name, imm))
        return out


def _emit_produced(a: _Asm, out_slot: int, opcode: int,
                   digit_at=None, imm_const=None):
    """Compiler bytecode that EMITs ONE produced instruction into ``out_slot``.

    EMIT writes ``CODE[out_slot] = AX + 256*STACK0`` (low byte = produced OPCODE,
    high byte = produced IMMEDIATE), and pops the immediate. So: put the produced
    immediate on the stack (``PSH``), then AX := produced opcode, then ``EMIT``.

      * ``digit_at``: the produced immediate is ``src[digit_at] - '0'`` — READ FROM
        SOURCE at runtime (``LC``), the operand value the compiler is parsing.
      * ``imm_const``: the produced immediate is a constant (0 for PSH/MUL/ADD/HALT).
    """
    if digit_at is not None:
        a.emit(isa.IMM, digit_at); a.emit(isa.LC, 0)          # AX = src[digit_at]
        a.emit(isa.PSH, 0); a.emit(isa.IMM, 48); a.emit(isa.SUB, 0)  # AX = char-'0'
    else:
        a.emit(isa.IMM, int(imm_const or 0))
    a.emit(isa.PSH, 0)                                         # STACK0 = produced imm
    a.emit(isa.IMM, opcode)                                    # AX = produced opcode
    a.emit(EMIT, out_slot)                                     # CODE[out_slot] := word


def _test_op_is_star(a: _Asm, src_pos: int, target_label: str):
    """Compiler bytecode: read src[src_pos]; if it is '*' branch to target_label.

    ``AX := src[pos] - '*'`` (== 0 iff the operator is ``*``), then ``BZ``. This is
    the runtime operator test that drives the precedence decision."""
    a.emit(isa.IMM, src_pos); a.emit(isa.LC, 0)               # AX = src[pos] = operator
    a.emit(isa.PSH, 0); a.emit(isa.IMM, _STAR); a.emit(isa.SUB, 0)  # AX = op - '*'
    a.emit(isa.BZ, target_label)


def expr_compiler_bytecode(outbase: int = COMPILER_OUTBASE):
    """Assemble the minimal-subset expression compiler as c4_min bytecode.

    Returns a list of ``(mnemonic, imm)`` — the COMPILER program. Load it as the
    ``prog`` of ``CompilerMachine.run(prog, source="2+3*4")`` (or bake it) and it
    reads the source, emits the compiled bytecode at ``outbase``.., and JMPs there.

    Precedence handling for ``d0 op1 d1 op2 d2`` (positions 0..4 of the source):
      op1=='*' : fold d0*d1 immediately (``IMM d0;PSH;IMM d1;MUL``), then op2.
      op1=='+' : defer; if op2=='*' the ``d1*d2`` term binds first, so the ADD is
                 emitted LAST (``IMM d0;PSH;IMM d1;PSH;IMM d2;MUL;ADD``); if op2=='+'
                 it left-folds (``IMM d0;PSH;IMM d1;ADD;PSH;IMM d2;ADD``).
    """
    O = outbase
    a = _Asm()
    # --- op1 = src[1]: choose the multiply-first vs add-first skeleton ---
    _test_op_is_star(a, 1, "op1_star")
    a.emit(isa.JMP, "op1_plus")

    # ===================== op1 == '*' : d0*d1 folds first =====================
    a.label("op1_star")
    _emit_produced(a, O + 0, isa.IMM, digit_at=0)             # IMM d0
    _emit_produced(a, O + 1, isa.PSH, imm_const=0)            # PSH
    _emit_produced(a, O + 2, isa.IMM, digit_at=2)             # IMM d1
    _emit_produced(a, O + 3, isa.MUL, imm_const=0)            # MUL  (d0*d1)
    _emit_produced(a, O + 4, isa.PSH, imm_const=0)            # PSH
    _emit_produced(a, O + 5, isa.IMM, digit_at=4)             # IMM d2
    _test_op_is_star(a, 3, "star_op2_mul")                    # op2 ?
    _emit_produced(a, O + 6, isa.ADD, imm_const=0)            # op2=='+' : ADD
    a.emit(isa.JMP, "star_done")
    a.label("star_op2_mul")
    _emit_produced(a, O + 6, isa.MUL, imm_const=0)            # op2=='*' : MUL
    a.label("star_done")
    _emit_produced(a, O + 7, isa.HALT, imm_const=0)           # HALT
    a.emit(isa.JMP, O)                                        # HANDOFF -> run it

    # ===================== op1 == '+' : deferred add =====================
    a.label("op1_plus")
    _emit_produced(a, O + 0, isa.IMM, digit_at=0)             # IMM d0
    _emit_produced(a, O + 1, isa.PSH, imm_const=0)            # PSH
    _emit_produced(a, O + 2, isa.IMM, digit_at=2)             # IMM d1
    _test_op_is_star(a, 3, "plus_op2_star")                   # op2 ?
    # op2 == '+' : left-fold  (d0+d1) then +d2
    _emit_produced(a, O + 3, isa.ADD, imm_const=0)            # ADD  (d0+d1)
    _emit_produced(a, O + 4, isa.PSH, imm_const=0)            # PSH
    _emit_produced(a, O + 5, isa.IMM, digit_at=4)             # IMM d2
    _emit_produced(a, O + 6, isa.ADD, imm_const=0)            # ADD
    a.emit(isa.JMP, "plus_done")
    # op2 == '*' : d1*d2 binds first, ADD emitted last
    a.label("plus_op2_star")
    _emit_produced(a, O + 3, isa.PSH, imm_const=0)            # PSH
    _emit_produced(a, O + 4, isa.IMM, digit_at=4)             # IMM d2
    _emit_produced(a, O + 5, isa.MUL, imm_const=0)            # MUL  (d1*d2)
    _emit_produced(a, O + 6, isa.ADD, imm_const=0)            # ADD  (d0 + d1*d2)
    a.label("plus_done")
    _emit_produced(a, O + 7, isa.HALT, imm_const=0)           # HALT
    a.emit(isa.JMP, O)                                        # HANDOFF -> run it
    return a.resolve()
