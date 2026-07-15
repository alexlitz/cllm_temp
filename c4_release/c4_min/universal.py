"""c4_min UNIVERSAL interpreter: the C4 INTERPRETER compiled to a transformer.

THE ENDGAME. Where ``recurrent.py`` bakes each *program's* code table into the
dispatch weights (``control.dispatch_rules`` gates every op's effect on the baked
PC one-hot ``PC_IS[i]`` and hard-codes that instruction's opcode + immediate), so
one transformer == one program, THIS module moves the program **out of the
weights and into DATA MEMORY**. The result is ONE fixed-weight model — the
interpreter itself — that runs ANY program loaded into it as data.

The move (program: WEIGHTS -> DATA)
-----------------------------------
1. LOAD.   The program is a list of ``[opcode, imm]`` cells (DESIGN.md §(a): each
   instruction is a fixed WIDTH=2 slot). We write those cells into two data-memory
   band arrays ``CODE_OP[i]`` / ``CODE_IMM[i]`` via the *initial state* (the
   embedding row) — as INPUT, not baked into any FFN rule. The dispatch weights
   never see the program.

2. FETCH.  Read the instruction cell at the current PC from data memory. PC is a
   scalar band; the fetch materialises the exact-integer PC one-hot ``PC_IS[i]``
   (the triangular-pulse gadget from ``control.compile_pc_fetch``) and then does a
   content/addr-match *select*: ``OP_VAL = sum_i PC_IS[i] * CODE_OP[i]`` and
   ``IMM = sum_i PC_IS[i] * CODE_IMM[i]``. This is a bilinear read — the SwiGLU
   ``silu(up)*gate`` computes the product ``PC_IS[i] * CODE_x[i]`` per slot and
   the down-projection sums them. The weights are program-INDEPENDENT: they only
   know "multiply the PC one-hot by whatever data sits in the CODE bands".

3. DECODE. Turn the fetched scalar ``OP_VAL`` into an opcode one-hot
   ``OP_IS[op] == (OP_VAL == op)`` — the SAME exact-integer triangular-pulse
   gadget used for the PC one-hot, applied to ``OP_VAL`` instead of ``PC``. Now
   the current opcode is a dynamic one-hot decoded from the fetched value.
   (``IMM`` is already the fetched scalar immediate.)

4. DISPATCH. Execute the transition for the DECODED opcode. There is ONE fixed
   ``FFNRule`` per opcode *value* (not per program position), gated on
   ``OP_IS[op]``, reading its immediate from the fetched ``IMM`` band. These are
   the same exact-integer AX/STACK0/PC gadgets as ``recurrent.py`` — only the
   guard changed from a baked ``PC_IS[i]`` (which knew the instruction) to the
   dynamically-decoded ``OP_IS[op]`` (which does not).

The transformer WEIGHTS = the interpreter (fetch-select + decode + the op
transitions). The PROGRAM = data in ``CODE_OP``/``CODE_IMM``. Apply the step-block
recurrently (``recurrent.py``'s driver + per-step re-quantisation) -> runs any
program, any length. Compile ONCE, run MANY different programs by swapping the
data bands (``load_program``), never recompiling.

Ops covered: IMM, LEA, PSH, ADD, SUB, JMP, BZ, BNZ, HALT (the built subset). The
dispatch table is trivially extensible: add one ``FFNRule`` per new opcode value.
"""
from __future__ import annotations

from typing import List

import torch

from . import isa
from . import control
from .compiler import (VOCAB, _zero_attn, _load_ffn, _load_head, head_matrix)
from .compile_ffn import compile_ffn, compile_fold, S as FFN_S
from .control import RELU_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer


# ------------------------------------------------------------------ layout ----

def build_universal_layout(code_size: int, n_heads: int = 4,
                           packed: bool = False) -> Layout:
    """Layout for the universal interpreter.

    Adds, on top of the base register bands:
      * DATA-MEMORY code table (``code_size`` slots), in one of two encodings:
          - two-cell (default): ``CODE_OP[i]`` / ``CODE_IMM[i]`` (DESIGN.md WIDTH=2).
          - packed (``packed=True``): ``CODE_WORD[i] == op | imm<<8`` (one scalar
            word per slot, the literal C4 ``instr`` encoding) + a ``WORD`` fetch
            scratch band that ``compile_word_decode`` splits into OP_VAL/IMM.
      * ``PC_IS[i]``   — PC one-hot scratch (exact, recomputed each step by fetch).
      * ``OP_VAL``     — the fetched scalar opcode (dynamic decode input).
      * ``OP_IS[op]``  — the DECODED opcode one-hot (size NUM_OPS).
      * one reusable ``OUT_SLOTS[0]`` / ``HALT_SEEN[0]`` (recurrent driver reuses).

    None of these are weights: ``load_program`` writes the code bands into the
    initial state as data.
    """
    L = Layout(n_heads=n_heads)
    L.CODE_SIZE = code_size
    L.PACKED = packed
    if packed:
        L.CODE_WORD = [L._band(f"CODE_WORD_{i}", 1) for i in range(code_size)]
        L.WORD = L._band("WORD", 1)
    else:
        L.CODE_OP = [L._band(f"CODE_OP_{i}", 1) for i in range(code_size)]
        L.CODE_IMM = [L._band(f"CODE_IMM_{i}", 1) for i in range(code_size)]
    L.PC_IS = [L._band(f"PC_IS_{i}", 1) for i in range(code_size)]
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)  # decoded opcode one-hot region
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


# ------------------------------------------------------- fetch + decode FFN ----

def compile_fetch_select(L: Layout, dim: int):
    """FFN: PC one-hot + fetch-select ``OP_VAL``/``IMM`` from data memory + AX_ZERO.

    Combines three program-INDEPENDENT jobs into one block:

      (a) PC one-hot  ``PC_IS[i] = (PC == i)`` and ``AX_ZERO = (AX == 0)`` — the
          exact triangular-pulse / relu gadgets from ``control.compile_pc_fetch``.
          These are pure functions of PC and AX; no program in them.

      (b) fetch-SELECT the code cell at PC out of DATA MEMORY:
              OP_VAL = sum_i PC_IS[i] * CODE_OP[i]
              IMM    = sum_i PC_IS[i] * CODE_IMM[i]
          via the SwiGLU product ``silu(S*PC_IS[i]) * CODE_x[i] / silu(S)`` per slot
          (silu(S*1)=S, silu(S*0)=0, so the unit passes CODE_x[i] iff PC==i). The
          weights only know "PC one-hot times data band"; the data is the program.

    Both (a) and (b) are SET (not increment): each written scratch band gets a
    silu-identity self-clear reading its previous value, so re-running fetch every
    recurrent step is idempotent.

    NOTE the two stages are DATA-DEPENDENT within one block: the product in (b)
    reads ``PC_IS[i]`` which (a) writes. A single additive-residual FFN cannot read
    its own freshly-written band, so (a) and (b) are baked as TWO separate blocks
    by the caller (``build_universal_step``); this function returns (a)'s spec and
    ``compile_code_select`` returns (b)'s. They are documented together because
    conceptually they are one fetch.
    """
    return control.compile_pc_fetch(L.PC, L.AX, L.PC_IS, L.AX_ZERO, L.ONE, dim)


def compile_code_select(L: Layout, dim: int):
    """FFN block (b): OP_VAL / IMM <- select code cell at PC from DATA MEMORY.

        OP_VAL = sum_i PC_IS[i] * CODE_OP[i]
        IMM    = sum_i PC_IS[i] * CODE_IMM[i]

    Program-independent product-select. Two hidden units per code slot (one for
    OP_OP, one for OP_IMM) plus two self-clears (OP_VAL, IMM).
    """
    n = L.CODE_SIZE
    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    # units: [ n selects for OP_VAL | n selects for IMM | clear OP_VAL | clear IMM ]
    n_units = 2 * n + 2
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    def _product_unit(u, sel_band, data_band, dst_band):
        # up = S * sel_band  (silu(S)=S when sel==1, silu(0)=0 when sel==0)
        W_up[u, sel_band] = FFN_S
        # gate = data_band
        W_gate[u, data_band] = 1.0
        # down: dst += silu(up)*gate / silu(S) = sel * data
        W_down[dst_band, u] += 1.0 / silu_S

    u = 0
    for i in range(n):
        _product_unit(u, L.PC_IS[i], L.CODE_OP[i], L.OP_VAL); u += 1
    for i in range(n):
        _product_unit(u, L.PC_IS[i], L.CODE_IMM[i], L.IMM); u += 1
    # self-clear OP_VAL and IMM (SET, not increment)
    for band in (L.OP_VAL, L.IMM):
        W_up[u, L.ONE] = FFN_S
        W_gate[u, band] = 1.0
        W_down[band, u] += -1.0 / silu_S
        u += 1

    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_word_select(L: Layout, dim: int):
    """FFN block (b, PACKED variant): WORD <- select the packed code word at PC.

        WORD = sum_i PC_IS[i] * CODE_WORD[i]     (CODE_WORD[i] == op | imm<<8)

    The PACKED representation stores the whole instruction as ONE scalar word in
    data memory (the literal ``instr = opcode | (imm << 8)`` C4 encoding), rather
    than the two-cell ``[op, imm]`` slot. The subsequent word-decode splits ``WORD``
    into ``OP_VAL`` and ``IMM`` dynamically. Program-independent.

    Precision: WORD reaches 65535, so the product-select normaliser must be EXACT
    on the round-trip. We use scale ``POW2 = 256`` (``silu(256)=256`` exactly) and
    the exact power-of-two reciprocal ``1/256`` (exact in fp32) instead of the
    generic ``1/silu(60)`` (whose ~0.0167 reciprocal injects a WORD-proportional
    error that flips staircase steps at large immediates). ``256*65535 < 2**24`` so
    every intermediate is an exact fp32 integer.
    """
    n = L.CODE_SIZE
    POW2 = 256.0                                      # silu(256)=256 exactly; 1/256 exact
    n_units = n + 1                                   # n selects + 1 self-clear
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for i in range(n):
        W_up[i, L.PC_IS[i]] = POW2                    # gate on PC one-hot (silu(256)=256)
        W_gate[i, L.CODE_WORD[i]] = 1.0               # value = packed word
        W_down[L.WORD, i] += 1.0 / POW2               # exact reciprocal
    W_up[n, L.ONE] = POW2                             # self-clear WORD (SET)
    W_gate[n, L.WORD] = 1.0
    W_down[L.WORD, n] += -1.0 / POW2
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_word_decode_imm(L: Layout, dim: int):
    """FFN block (PACKED variant, stage 1): ``IMM = WORD >> 8 = floor(WORD/256)``.

    Exact-integer staircase: ``imm = sum_{k>=1} step(WORD >= 256*k)`` over
    ``k in 1..255`` (WORD < 65536), each step being the clamped-relu difference
    ``relu(WORD-(256k-1)) - relu(WORD-256k)`` (exact 0/1 on integers, and each
    relu is normalised to its own small value BEFORE summing, so the per-step
    cancellation is between ~O(WORD) values -> fp32-safe). SET (self-clear first).

    ``OP_VAL = WORD & 0xFF = WORD - 256*IMM`` is deliberately deferred to stage 2
    (``compile_word_decode_op``): forming ``256*step`` here would scale each relu to
    ~O(256*WORD) ~ 1.4e7, whose fp32 ULP (~1) destroys the -256 cancellation. Using
    the already-materialised small ``IMM`` band in stage 2 keeps both operands
    moderate.
    """
    KMAX = 255                                        # WORD < 256*256
    # EXACT power-of-two normaliser: relu(z) = silu(POW2*z)/POW2 with POW2=256
    # (silu(256)=256 exactly, 1/256 exact in fp32). The generic 1/RELU_S=1/200 is
    # NOT exact and leaves a ~1e-5 residue on IMM; harmless on its own, but stage 2
    # amplifies it by 256 -> ~0.004, which de-sharpens the opcode one-hot. POW2
    # keeps IMM an exact integer (256*65535 < 2**24, so no intermediate rounds).
    POW2 = 256.0
    n_step_units = 2 * KMAX
    clear_imm = n_step_units
    n_units = n_step_units + 1
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    W_up[clear_imm, L.ONE] = POW2                     # self-clear IMM (SET), exact
    W_gate[clear_imm, L.IMM] = 1.0
    W_down[L.IMM, clear_imm] += -1.0 / POW2

    for k in range(1, KMAX + 1):
        thr = 256 * k
        a = 2 * (k - 1)                               # relu(WORD-(thr-1))
        b = 2 * (k - 1) + 1                           # relu(WORD-thr)
        W_up[a, L.WORD] = POW2; b_up[a] = -POW2 * (thr - 1); W_gate[a, L.ONE] = 1.0
        W_up[b, L.WORD] = POW2; b_up[b] = -POW2 * thr;       W_gate[b, L.ONE] = 1.0
        W_down[L.IMM, a] += 1.0 / POW2                # +step (imm += 1 per crossed multiple)
        W_down[L.IMM, b] += -1.0 / POW2
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_word_decode_op(L: Layout, dim: int):
    """FFN block (PACKED variant, stage 2): ``OP_VAL = WORD & 0xFF = WORD - 256*IMM``.

    Reads the already-materialised small ``IMM`` band (<=255). A single SET write
    ``OP_VAL := WORD - 256*IMM``, hand-built with the EXACT power-of-two normaliser
    (``silu(256)=256`` gate, ``1/256`` reciprocal) rather than the generic
    ``compile_ffn`` (whose ``1/silu(30)`` scale leaves a ~0.004 residue that then
    de-sharpens the ``OP_IS`` one-hot and scales the dispatched write by <1). Both
    ``WORD``<=65535 and ``256*IMM``<=65280 are exact fp32 integers with no giant
    intermediate, so ``OP_VAL`` is exact. Completes the literal ``word & 0xFF`` /
    ``word >> 8`` dynamic decode — no per-position table.

    Three silu-identity units (all gated on ``ONE``, so unconditional SET):
      +WORD, -256*IMM, -OP_VAL_old, each routed with the exact ``1/256`` scale.
    """
    POW2 = 256.0
    n_units = 3
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    # unit u: hidden = silu(POW2)*gate = 256*gate ; down = coeff/256 -> coeff*gate
    specs = [(L.WORD, 1.0), (L.IMM, -256.0), (L.OP_VAL, -1.0)]
    for u, (band, coeff) in enumerate(specs):
        W_up[u, L.ONE] = POW2
        W_gate[u, band] = 1.0
        W_down[L.OP_VAL, u] += coeff / POW2
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_opcode_decode(L: Layout, dim: int):
    """FFN block: DECODE the fetched ``OP_VAL`` scalar into an opcode one-hot.

        OP_IS[op] = (OP_VAL == op)   for op in 0..NUM_OPS-1

    The SAME exact-integer triangular-pulse gadget as the PC one-hot, applied to
    ``OP_VAL`` instead of ``PC``:
        OP_IS[op] = relu(OP_VAL-(op-1)) - 2 relu(OP_VAL-op) + relu(OP_VAL-(op+1))
    One shared relu unit per threshold in [-1 .. NUM_OPS], routed +1/-2/+1. This is
    the DYNAMIC decode: which opcode fires is a function of the fetched value, not a
    baked per-position table. Each OP_IS band gets a self-clear (SET semantics).
    """
    n = isa.NUM_OPS
    op_is = [L.OP_IS + op for op in range(n)]
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    clear0 = n_relu
    n_units = clear0 + n

    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    # relu units: hidden_j = relu(OP_VAL - t_j)
    for t, j in thr_unit.items():
        W_up[j, L.OP_VAL] = RELU_S
        b_up[j] = -RELU_S * t
        W_gate[j, L.ONE] = 1.0
    # self-clear each OP_IS band first
    for c, band in enumerate(op_is):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S
        W_gate[u, band] = 1.0
        W_down[band, u] += -1.0 / silu_S
    # OP_IS[op] = triangular pulse of OP_VAL at op
    for op, band in enumerate(op_is):
        W_down[band, thr_unit[op - 1]] += 1.0 / RELU_S
        W_down[band, thr_unit[op]] += -2.0 / RELU_S
        W_down[band, thr_unit[op + 1]] += 1.0 / RELU_S

    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------------------- opcode dispatch table ----

def universal_dispatch_rules(L: Layout) -> List[FFNRule]:
    """ONE FFNRule per OPCODE VALUE, gated on the DECODED one-hot ``OP_IS[op]``.

    This is the interpreter's op table — fixed weights, program-independent. Each
    rule reads its immediate from the DYNAMIC ``IMM`` band (fetched from data
    memory), not a baked constant. The AX/STACK0/PC algebra is identical to
    ``control.dispatch_rules`` / the recurrent slice; only the guard (decoded
    opcode) and the immediate source (fetched band) changed.

    PC update, per op (guarded on OP_IS[op], so PC == the current index i is NOT
    known statically). We update PC RELATIVE to itself:
      * non-branch: PC += 1
      * JMP:  PC := IMM          == PC += (IMM - PC)
      * BZ:   AX==0 -> PC := IMM ; else PC += 1
      * BNZ:  AX!=0 -> PC := IMM ; else PC += 1
      * HALT: PC += 0 (freeze)
    The branch/jump deltas reference the *fetched* IMM band and the current PC
    band, so they are program-independent linear writes.
    """
    ax, stk, pc = L.AX, L.STACK0, L.PC
    imm, bp = L.IMM, L.BP
    OP = L.OP_IS

    def G(op):
        return [(OP + op, 0.5, 1.5)]  # fires iff decoded opcode == op

    rules: List[FFNRule] = []

    # --- data effect + PC update, one rule per opcode value ---
    # IMM: AX = imm
    rules.append(FFNRule(G(isa.IMM),
                         {ax: LinearExpr.of(imm, 1.0) + LinearExpr.of(ax, -1.0),
                          pc: LinearExpr.c(1.0)}))
    # LEA: AX = BP + imm
    rules.append(FFNRule(G(isa.LEA),
                         {ax: LinearExpr.of(bp, 1.0) + LinearExpr.of(imm, 1.0)
                          + LinearExpr.of(ax, -1.0),
                          pc: LinearExpr.c(1.0)}))
    # PSH: STACK0 = AX
    rules.append(FFNRule(G(isa.PSH),
                         {stk: LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0),
                          pc: LinearExpr.c(1.0)}))
    # ADD: AX = STACK0 + AX  (fold mod 256 downstream)
    rules.append(FFNRule(G(isa.ADD),
                         {ax: LinearExpr.of(stk, 1.0),
                          pc: LinearExpr.c(1.0)}))
    # SUB: AX = STACK0 - AX  == (stk - 2*ax + 256), folded mod 256 downstream
    rules.append(FFNRule(G(isa.SUB),
                         {ax: LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0)
                          + LinearExpr.c(256.0),
                          pc: LinearExpr.c(1.0)}))
    # JMP: PC = imm  == PC += (imm - PC)
    rules.append(FFNRule(G(isa.JMP),
                         {pc: LinearExpr.of(imm, 1.0) + LinearExpr.of(pc, -1.0)}))
    # BZ / BNZ: the branch PC update is BILINEAR in the runtime bands (the taken
    # target is AX_ZERO gated with (imm - pc)), which a single LINEAR write cannot
    # express. Dispatch therefore leaves PC UNTOUCHED for BZ/BNZ; a dedicated
    # bilinear branch block (``compile_branch_delta``, run immediately after this
    # dispatch and BEFORE PC is otherwise read) computes the full PC update from
    # OP_IS[BZ]/OP_IS[BNZ], AX_ZERO, the fetched IMM and the current (pre-update)
    # PC. See that function for the exact product algebra.
    #   (no PC write here for BZ/BNZ)
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    # HALT: latch HALTED, freeze PC (PC += 0 -> no write needed).
    rules.append(FFNRule(G(isa.HALT),
                         {L.HALTED: LinearExpr.c(1.0)}))
    return rules


def compile_branch_delta(L: Layout, dim: int):
    """FFN block: the BILINEAR PC update for BZ / BNZ (runs after dispatch).

    Dispatch left PC untouched for BZ/BNZ, so PC still holds ``pc_pre`` (the
    branch's own index) here. We add the full branch delta:

        BZ  taken (AX_ZERO==1):  PC := IMM   -> delta = IMM - pc_pre
        BZ  not     (AX_ZERO==0): PC := pc_pre+1 -> delta = 1
        BNZ taken (AX_ZERO==0):  PC := IMM   -> delta = IMM - pc_pre
        BNZ not     (AX_ZERO==1): PC := pc_pre+1 -> delta = 1

    Both are the SAME shape: ``delta = TAKEN*(IMM - pc_pre) + (1-TAKEN)*1`` where
    ``TAKEN`` is ``AX_ZERO`` for BZ and ``1-AX_ZERO`` for BNZ, all further gated by
    the decoded ``OP_IS[BZ]`` / ``OP_IS[BNZ]``. We bake four SwiGLU product units
    per branch op (guard AND boolean via the sharp-silu gate, value via the linear
    gate):

        u1: gate=(IMM - pc_pre),  up = big*(OP_IS[op] + TAKEN - 1.5)   -> fires iff
            op-decoded AND taken       -> contributes (IMM - pc_pre)
        u2: gate=1(ONE),          up = big*(OP_IS[op] + NOTTAKEN - 1.5)-> fires iff
            op-decoded AND not-taken   -> contributes +1

    where TAKEN/NOTTAKEN read AX_ZERO with the right sign per op. The two guards
    are mutually exclusive, so exactly one contributes when the op is BZ/BNZ, and
    none contributes otherwise (this block is a no-op for every non-branch op).
    """
    pc, imm, azero, one = L.PC, L.IMM, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ

    # Two branch ops x (taken-unit + nottaken-unit) = 4 units.
    n_units = 4
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    BIG = 200.0
    silu_big = float(torch.nn.functional.silu(torch.tensor(0.5 * BIG)))  # ~=0.5*BIG

    def _and_unit(u, op_band, bool_terms, gate_expr):
        """up = BIG*(op_band + bool - 1.5): +0.5*BIG iff BOTH the decoded op-hot AND
        the boolean hold (each in {0,1}); <= -0.5*BIG otherwise. gate = gate_expr."""
        W_up[u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                W_up[u, band] += BIG * coeff
            b_up[u] += BIG * const
        b_up[u] += -BIG * 1.5
        for band, coeff in gate_expr[0]:
            W_gate[u, band] += coeff
        b_gate[u] += gate_expr[1]

    # BZ taken: AX_ZERO==1 -> bool = AX_ZERO ; delta = (IMM - pc_pre)
    _and_unit(0, BZ, [(azero, 1.0, 0.0)],
              ([(imm, 1.0), (pc, -1.0)], 0.0))
    W_down[pc, 0] += 1.0 / silu_big
    # BZ not-taken: AX_ZERO==0 -> bool = (1 - AX_ZERO) ; delta = +1
    _and_unit(1, BZ, [(azero, -1.0, 1.0)],
              ([(one, 1.0)], 0.0))
    W_down[pc, 1] += 1.0 / silu_big
    # BNZ taken: AX_ZERO==0 -> bool = (1 - AX_ZERO) ; delta = (IMM - pc_pre)
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)],
              ([(imm, 1.0), (pc, -1.0)], 0.0))
    W_down[pc, 2] += 1.0 / silu_big
    # BNZ not-taken: AX_ZERO==1 -> bool = AX_ZERO ; delta = +1
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)],
              ([(one, 1.0)], 0.0))
    W_down[pc, 3] += 1.0 / silu_big

    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ----------------------------------------------------- the universal model ----

def build_universal_step(code_size: int, n_heads: int = 4, max_pos: int = 4,
                         packed: bool = False):
    """Bake ONE universal VM step-block: the INTERPRETER, program-INDEPENDENT.

    The physical FFN sub-blocks (single residual position, no attention) are:

      1. fetch      : PC -> PC_IS[i] one-hot  +  AX_ZERO predicate.
      2. code_select: fetch the code cell at PC from DATA MEMORY (product of
                      PC_IS[i] and the data band). Two-cell mode selects OP_VAL/IMM
                      directly; PACKED mode selects the packed WORD.
      3. decode     : two-cell mode -> OP_VAL is already the opcode; PACKED mode
                      first splits WORD into OP_VAL/IMM (``compile_word_decode``).
                      Then OP_VAL scalar -> OP_IS[op] decoded opcode one-hot.
      4. dispatch   : per-OPCODE rules gated on OP_IS[op] apply the op's AX/STACK0
                      effect + PC update (branches deferred to block 5).
      5. branch     : bilinear BZ/BNZ PC update (product of AX_ZERO, IMM, PC).
      6. fold+emit  : AX mod-256 fold, then AX -> OUT slot + HALTED -> HALT_SEEN.

    NONE of these weights depend on the program: ``code_size`` only sizes the
    PC/data bands. The program is loaded into the CODE bands of the *initial state*
    by ``load_program``. ``packed=True`` stores the literal ``op | imm<<8`` word per
    slot and decodes it dynamically. Returns ``(model, layout)``.
    """
    L = build_universal_layout(code_size, n_heads=n_heads, packed=packed)
    dim = L.D

    if packed:
        fetch_decode = [
            compile_word_select(L, dim),                    # 2. fetch WORD
            compile_word_decode_imm(L, dim),                # 3a. IMM = WORD>>8
            compile_word_decode_op(L, dim),                 # 3b. OP_VAL = WORD&0xFF
            compile_opcode_decode(L, dim),                  # 3c. OP_VAL -> OP_IS[op]
        ]
    else:
        fetch_decode = [
            compile_code_select(L, dim),                    # 2. fetch OP_VAL/IMM
            compile_opcode_decode(L, dim),                  # 3. OP_VAL -> OP_IS[op]
        ]

    ffn_specs = [
        compile_fetch_select(L, dim),                       # 1. PC one-hot + AX_ZERO
        *fetch_decode,
        compile_ffn(universal_dispatch_rules(L), dim),      # 4. dispatch (non-branch PC)
        compile_branch_delta(L, dim),                       # 5. BZ/BNZ bilinear PC
        compile_fold(L.AX, L.ONE, dim, modulus=256),        # 6a. AX mod-256
        compile_ffn([                                       # 6b. emit + halt snapshot
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]

    n_blocks = len(ffn_specs)                                # == 7
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0                          # ONE lane = 1
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


def load_program(model, L, code: List[isa.Instr]) -> torch.Tensor:
    """Return the INITIAL STATE with ``code`` loaded into DATA MEMORY.

    This is the whole point of universality: the program is written into the
    data bands of the initial residual as INPUT — NOT baked into any weight.
    Different programs = different data in these bands; the model weights are
    untouched. PC/AX/SP/BP/STACK0 start at 0, ONE=1. Two-cell mode writes
    ``CODE_OP``/``CODE_IMM``; packed mode writes the ``op | imm<<8`` word into
    ``CODE_WORD``.
    """
    assert len(code) <= L.CODE_SIZE, f"program has {len(code)} > {L.CODE_SIZE} slots"
    state = model.embed[0].clone()
    if getattr(L, "PACKED", False):
        for i, ins in enumerate(code):
            state[L.CODE_WORD[i]] = float((ins.op & 0xFF) | ((ins.imm & 0xFF) << 8))
    else:
        for i, ins in enumerate(code):
            state[L.CODE_OP[i]] = float(ins.op)
            state[L.CODE_IMM[i]] = float(ins.imm)
    # slots past the program length stay 0 (opcode 0 == LEA, never reached: HALT
    # freezes PC, and every valid program ends in HALT before the tail).
    return state


def _step_once(model, state: torch.Tensor) -> torch.Tensor:
    x = state.view(1, 1, -1)
    for blk in model.blocks:
        x = blk(x)
    return x[0, 0]


def _requantize(state: torch.Tensor, L) -> torch.Tensor:
    """Round every band to the nearest integer (annihilating fp residue), then
    pin ONE=1 and re-assert the (constant) CODE data bands so nothing can perturb
    the program-in-data. Idempotent on the exact-integer VM state."""
    q = torch.round(state)
    q[L.ONE] = 1.0
    return q


def run_universal(model, L, code: List[isa.Instr], max_steps: int = 100000,
                  requantize: bool = True, trace_state: bool = False):
    """Run the ONE universal step-block AUTOREGRESSIVELY on ``code`` loaded as DATA.

    Each iteration: forward the interpreter step-block, re-quantise to exact
    integers (kills fp residue -> exact over unbounded steps), decode this step's
    AX via the LM head, stop once HALT executed. Returns the per-step AX trace,
    matching ``isa.interpret``. The MODEL WEIGHTS ARE NEVER TOUCHED — only the
    initial CODE data bands change per program.
    """
    import torch.nn.functional as F

    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]

    state = load_program(model, L, code)
    trace: List[int] = []
    states = [] if trace_state else None
    for _ in range(max_steps):
        state = _step_once(model, state)
        if requantize:
            state = _requantize(state, L)
        if states is not None:
            states.append(state.clone())
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if trace_state:
        return trace, states
    return trace


class UniversalInterpreter:
    """ONE fixed-weight interpreter model that runs ANY program loaded as data.

    Compile ONCE (``UniversalInterpreter(code_size)``); run MANY different programs
    via ``run(prog)`` — each loads a new program into the data bands, never
    recompiling. ``code_size`` bounds the number of instruction slots in data
    memory (the code-table capacity), independent of the run's step count.
    """

    def __init__(self, code_size: int, n_heads: int = 4, packed: bool = False):
        self.code_size = code_size
        self.packed = packed
        self.model, self.L = build_universal_step(code_size, n_heads=n_heads,
                                                  packed=packed)

    def run(self, prog, max_steps: int = 100000, requantize: bool = True):
        code = isa.assemble(prog)
        return run_universal(self.model, self.L, code,
                             max_steps=max_steps, requantize=requantize)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)
