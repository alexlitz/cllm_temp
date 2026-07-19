"""c4_min NIBBLE-BAKING: compile a *program* into the transformer WEIGHTS.

THE ORIGINAL GOAL (BLOG_SPEC.md §"Baking Prompts/Programs into the Transformer
Weights", §"Model that Directly Runs C Code"): *C code in -> transformer weights
out*. The universal interpreter (``greenfield-universal:c4_min/universal.py``)
reads the bytecode from DATA (the ``CODE_OP``/``CODE_IMM`` bands of the initial
state); this module moves the bytecode OUT OF DATA and INTO THE WEIGHTS. The
result is a transformer that runs one specific program *with no bytecode anywhere
in its input state* — a compiled C program that IS a transformer.

The contrast (the whole point)
------------------------------
UNIVERSAL fetch (``universal.compile_code_select``), program lives in DATA::

    OP_VAL = sum_i  PC_IS[i] * CODE_OP[i]        # CODE_OP[i] is a DATA band
    IMM    = sum_i  PC_IS[i] * CODE_IMM[i]       #   (loaded per program)

    the weights only know "PC one-hot times whatever data sits in CODE" —
    program-INDEPENDENT; you must LOAD the program into the state to run it.

BAKED fetch (``compile_baked_fetch`` below), program lives in WEIGHTS::

    OP_VAL = sum_i  ADDR_IS[i] * op_i            # op_i is a baked CONSTANT
    IMM    = sum_i  ADDR_IS[i] * imm_i           #   (a bias in the weights)

    the constants ``op_i``/``imm_i`` ARE the program. There is NO CODE band and
    ``load_program`` writes NOTHING — the initial state is just ONE=1, all
    registers zero. This is a **read-only code segment made of weights**.

The baking mechanism, exactly as the spec prescribes (§ line 830)
-----------------------------------------------------------------
"take the binary key, break it up into bytes or nibbles, perform an equality
check for each byte or nibble, then in a subsequent layer perform a logical AND
over those results ... use the outputs of the second layer as a one-hot mask for
the next layer, which can be implemented as a MoE router ... route to a layer that
simply returns the value associated with the key, e.g. via biases, zero gate and
up matrices and an identity down matrix."

We implement that as an FFN pipeline keyed on the scalar PC (the fetch key):

  1a. SPLIT     (``compile_pc_split``).   Break the PC key into NIBBLES
     (PC < 256 -> 2 nibbles: ``PC_LO = PC & 0xF``, ``PC_HI = PC >> 4``) with the
     exact-integer staircase, plus ``AX_ZERO = (AX==0)``.

  1b. EQ-CHECK  (``compile_nibble_eq``).  Per-nibble EQUALITY point-indicators
     ``EQ_LO[d] = (PC_LO == d)`` / ``EQ_HI[d] = (PC_HI == d)`` (the exact
     triangular pulse). SHARED across addresses (spec: "if multiple addresses have
     the same value for a given byte/nibble the first-layer results can be shared,
     bounding the size"). Split (1a) and eq-check (1b) are TWO blocks because 1b
     reads the ``PC_LO``/``PC_HI`` that 1a writes and one additive FFN cannot read
     its own fresh write.

  2. AND -> ONE-HOT  (``compile_addr_and``).  For each code slot i with address
     nibbles (lo_i, hi_i), AND its two nibble-equalities into a one-hot mask
     ``ADDR_IS[i] = EQ_LO[lo_i] * EQ_HI[hi_i]`` (a SwiGLU product of two 0/1
     indicators -> exact 0/1). Exactly one ``ADDR_IS[i]`` is hot per step: it is
     the address one-hot, computed by eq+AND rather than the triangular pulse.

  3. MoE-VALUE  (``compile_baked_value``).  Gated on the ``ADDR_IS[i]`` router
     one-hot, write the BAKED CONSTANTS ``OP_VAL := op_i`` and ``IMM := imm_i``.
     This is the "expert that simply returns the value associated with the key,
     via biases / zero gate / identity down" — here the value is a constant so it
     rides the unit's constant gate (b_gate) and the ``ADDR_IS[i]`` router in the
     up-detector; ``W_down`` places it into OP_VAL/IMM. Duplicate (op,imm) values
     could share one expert (spec) — we keep one unit per slot for clarity.

Everything downstream (decode OP_VAL -> OP_IS one-hot, per-opcode dispatch, the
bilinear branch delta, the mod-256 fold, emit) is the SAME program-independent
interpreter machinery as ``universal.py``: baking only swaps the *fetch* — "we
simply have the network check in these FFNs instead of before the attention.
Interestingly enough this effectively creates a read only code segment."

Composition with the universal fetch
------------------------------------
``universal.build_universal_step`` chains: fetch(PC one-hot) -> code_select(DATA)
-> decode -> dispatch -> branch -> fold -> emit. This module chains: eqcheck ->
and(one-hot) -> baked_value(WEIGHTS) -> decode -> dispatch -> branch -> fold ->
emit. Blocks 4..7 are byte-identical between the two; only blocks 1..3 (the
fetch) differ — universal reads a data band, baked reads a bias. You can bake a
program into the universal model's data OR into a per-program model's weights and
get the same trace; ``run_baked`` proves the baked run needs no program in data.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from . import isa
from . import control
from .compiler import VOCAB, _zero_attn, _load_ffn, _load_head, head_matrix
from .compile_ffn import compile_ffn, compile_fold, S as FFN_S, RELU_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer

# The universal interpreter's DECODE / DISPATCH / BRANCH are program-independent;
# import them so the baked model shares BYTE-IDENTICAL downstream blocks and the
# contrast is provably only the fetch. universal.py lives on greenfield-universal;
# if it is not present on this branch we fall back to a local copy of the three
# program-independent functions (kept in sync, same algebra).
try:  # pragma: no cover - import path depends on branch merge state
    from .universal import (compile_opcode_decode, universal_dispatch_rules,
                            compile_branch_delta)
    _HAVE_UNIVERSAL = True
except Exception:  # universal.py not merged onto this branch yet
    _HAVE_UNIVERSAL = False


NIBBLE = 16  # nibble radix (4 bits)


# ------------------------------------------------------------------ layout ----

def build_baked_layout(code_size: int, n_heads: int = 4) -> Layout:
    """Layout for a BAKED program. Note what is ABSENT vs the universal layout:
    there are **NO CODE_OP / CODE_IMM / CODE_WORD data bands** — the program is
    not data anywhere. We add only the fetch SCRATCH the baked pipeline needs:

      * ``EQ_LO[d]`` / ``EQ_HI[d]``  (NIBBLE=16 each) — per-nibble PC equality.
      * ``PC_LO`` / ``PC_HI``        — the PC split into low/high nibble scalars.
      * ``ADDR_IS[i]``  (code_size)  — the eq+AND address one-hot (router).
      * ``OP_VAL``                   — the fetched opcode scalar (decode input).
      * ``OP_IS``       (NUM_OPS)    — the decoded opcode one-hot.
      * one reusable ``OUT_SLOTS[0]`` / ``HALT_SEEN[0]`` (recurrent driver reuses).

    ``AX_ZERO`` is reused as the BZ/BNZ branch predicate (materialised alongside
    the PC split). The baked constants ``op_i``/``imm_i`` live in the WEIGHTS
    (``compile_baked_value``), not here.
    """
    L = Layout(n_heads=n_heads)
    L.CODE_SIZE = code_size
    L.PC_LO = L._band("PC_LO", 1)
    L.PC_HI = L._band("PC_HI", 1)
    L.EQ_LO = L._band("EQ_LO", NIBBLE)   # one-hot: (PC & 0xF) == d
    L.EQ_HI = L._band("EQ_HI", NIBBLE)   # one-hot: (PC >> 4) == d
    L.ADDR_IS = [L._band(f"ADDR_IS_{i}", 1) for i in range(code_size)]
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


# --------------------------------------- stage 1a: split PC key into nibbles --

def compile_pc_split(L: Layout, dim: int):
    """STAGE 1a: split the fetch KEY (scalar PC) into its two nibbles + AX_ZERO.

        PC_HI = floor(PC / 16) = PC >> 4
        PC_LO = PC - 16*PC_HI  = PC & 0xF
        AX_ZERO = (AX == 0)                     (the BZ/BNZ branch predicate)

    ``PC_HI`` is the exact-integer staircase ``sum_k step(PC >= 16k)`` (k=1..15,
    since PC < 256), each step the clamped-relu difference
    ``relu(PC-(16k-1)) - relu(PC-16k)`` (exact 0/1 on integers). ``PC_LO`` reuses
    the SAME step units: ``PC_LO = PC - 16*sum_k step(PC>=16k)`` — routed as +PC
    (identity) and -16 per crossed multiple. Both single-pass (read PC, an OLD
    band), so this whole block is exact in one forward.

    The EQ pulses (stage 1b, ``compile_nibble_eq``) MUST be a SEPARATE block: they
    read ``PC_LO``/``PC_HI`` which are WRITTEN here, and an additive FFN cannot read
    its own fresh write. SET semantics (self-clear PC_LO/PC_HI/AX_ZERO first).
    """
    POW2 = 16.0  # exact power-of-two normaliser: silu(16)=16, 1/16 exact in fp32
    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    KMAX = 15                                   # PC < 256 -> PC_HI in 0..15

    clear_bands = [L.PC_LO, L.PC_HI, L.AX_ZERO]
    n_step = 2 * KMAX
    u_axz = n_step
    u_pc = u_axz + 1                            # +PC identity into PC_LO
    u_clear0 = u_pc + 1
    n_units = u_clear0 + len(clear_bands)

    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    for k in range(1, KMAX + 1):
        thr = 16 * k
        a = 2 * (k - 1)
        b = 2 * (k - 1) + 1
        W_up[a, L.PC] = POW2; b_up[a] = -POW2 * (thr - 1); W_gate[a, L.ONE] = 1.0
        W_up[b, L.PC] = POW2; b_up[b] = -POW2 * thr;       W_gate[b, L.ONE] = 1.0
        W_down[L.PC_HI, a] += 1.0 / POW2       # PC_HI += step
        W_down[L.PC_HI, b] += -1.0 / POW2
        W_down[L.PC_LO, a] += -16.0 / POW2     # PC_LO -= 16*step
        W_down[L.PC_LO, b] += 16.0 / POW2
    # PC_LO += PC (silu-identity of PC): PC_LO = PC - 16*floor(PC/16) = PC & 0xF.
    W_up[u_pc, L.ONE] = FFN_S; W_gate[u_pc, L.PC] = 1.0
    W_down[L.PC_LO, u_pc] = 1.0 / silu_S
    # AX_ZERO = relu(1 - AX)
    W_up[u_axz, L.AX] = -RELU_S; b_up[u_axz] = RELU_S * 1.0; W_gate[u_axz, L.ONE] = 1.0
    W_down[L.AX_ZERO, u_axz] += 1.0 / RELU_S
    # self-clears (SET): subtract each band's OLD value.
    for c, band in enumerate(clear_bands):
        u = u_clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, band] = 1.0
        W_down[band, u] = -1.0 / silu_S

    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ---------------------------------- stage 1b: per-nibble equality (eq-check) --

def compile_nibble_eq(L: Layout, dim: int):
    """STAGE 1b (EQ-CHECK): per-nibble EQUALITY one-hots on the split PC nibbles.

    The spec's "equality check for each nibble", the exact triangular pulse:

        EQ_LO[d] = (PC_LO == d),   EQ_HI[d] = (PC_HI == d),   d in 0..15

    ``tri_d(x) = relu(x-(d-1)) - 2 relu(x-d) + relu(x-(d+1))`` — 1 at x==d, 0 at
    every other integer. One SHARED relu unit per threshold (in ``[-1..16]``) per
    nibble field, routed +1/-2/+1. SHARED across all code addresses: an address's
    nibble value indexes into these 16 shared indicators (spec: "if multiple
    addresses have the same value for a given nibble the first-layer results can be
    shared, bounding the size needed"). SET (self-clear first).
    """
    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    thr = list(range(-1, NIBBLE + 1))          # shared relu thresholds
    n_relu = len(thr)
    lo0, hi0 = 0, n_relu                        # PC_LO relus | PC_HI relus
    clear0 = 2 * n_relu
    clear_bands = [L.EQ_LO + d for d in range(NIBBLE)] + [L.EQ_HI + d for d in range(NIBBLE)]
    n_units = clear0 + len(clear_bands)

    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    lo_unit = {t: lo0 + j for j, t in enumerate(thr)}
    hi_unit = {t: hi0 + j for j, t in enumerate(thr)}
    for t, u in lo_unit.items():
        W_up[u, L.PC_LO] = RELU_S; b_up[u] = -RELU_S * t; W_gate[u, L.ONE] = 1.0
    for t, u in hi_unit.items():
        W_up[u, L.PC_HI] = RELU_S; b_up[u] = -RELU_S * t; W_gate[u, L.ONE] = 1.0
    # self-clears (SET)
    for c, band in enumerate(clear_bands):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, band] = 1.0
        W_down[band, u] = -1.0 / silu_S
    # triangular pulses
    for d in range(NIBBLE):
        W_down[L.EQ_LO + d, lo_unit[d - 1]] += 1.0 / RELU_S
        W_down[L.EQ_LO + d, lo_unit[d]]     += -2.0 / RELU_S
        W_down[L.EQ_LO + d, lo_unit[d + 1]] += 1.0 / RELU_S
        W_down[L.EQ_HI + d, hi_unit[d - 1]] += 1.0 / RELU_S
        W_down[L.EQ_HI + d, hi_unit[d]]     += -2.0 / RELU_S
        W_down[L.EQ_HI + d, hi_unit[d + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ------------------------------------------------- stage 2: AND -> one-hot ----

def compile_addr_and(L: Layout, code_size: int, dim: int):
    """STAGE 2 (AND -> ONE-HOT): AND the two per-nibble equalities per address.

        ADDR_IS[i] = EQ_LO[lo_i] * EQ_HI[hi_i]        (lo_i = i & 0xF, hi_i = i>>4)

    A SwiGLU product of two exact 0/1 indicators is an exact logical AND: exactly
    one ``ADDR_IS[i]`` is 1 (the current PC's slot), the rest 0. This IS the spec's
    "logical AND over those results ... used as a one-hot mask for the next layer"
    — the mask that the MoE value stage routes on. Program-INDEPENDENT in *shape*
    (the (lo_i, hi_i) pairs are just i's nibbles); the baked constants come later.

    Product unit for slot i: ``up = S*EQ_LO[lo_i]`` (silu(S)=S iff lo matches),
    ``gate = EQ_HI[hi_i]`` (=1 iff hi matches), ``down = 1/silu(S)`` -> writes
    ``1`` iff BOTH match. Plus a self-clear per ADDR_IS band (SET).
    """
    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    n_units = 2 * code_size          # one product + one self-clear per slot
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    for i in range(code_size):
        lo_i, hi_i = i & 0xF, (i >> 4) & 0xF
        prod = i
        clr = code_size + i
        # self-clear ADDR_IS[i] first (SET)
        W_up[clr, L.ONE] = FFN_S
        W_gate[clr, L.ADDR_IS[i]] = 1.0
        W_down[L.ADDR_IS[i], clr] = -1.0 / silu_S
        # AND product: EQ_LO[lo_i] * EQ_HI[hi_i]
        W_up[prod, L.EQ_LO + lo_i] = FFN_S      # silu(S*1)=S when lo matches
        W_gate[prod, L.EQ_HI + hi_i] = 1.0      # gate = hi-match (0/1)
        W_down[L.ADDR_IS[i], prod] += 1.0 / silu_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ----------------------------------------------- stage 3: MoE value (baked) ----

def compile_baked_value(L: Layout, code: List[isa.Instr], dim: int):
    """STAGE 3 (MoE-VALUE): the baked read-only code segment.

    Gated on the address one-hot ``ADDR_IS[i]`` (the router), write the BAKED
    CONSTANTS for slot i::

        OP_VAL := op_i           # a constant baked into the WEIGHTS
        IMM    := imm_i          # a constant baked into the WEIGHTS

    THIS is where the program lives — ``op_i``/``imm_i`` are the bytecode, now
    residing in ``W_down``/``b_gate`` (the "expert that returns the value
    associated with the key via biases / zero gate / identity down"). There is no
    CODE data band: the router picks the expert, the expert emits the constant.
    Exactly one expert fires per step (the ADDR_IS one-hot). SET semantics
    (self-clear OP_VAL/IMM first).

    Efficiency (spec): experts for slots with the SAME (op,imm) could be shared;
    we keep one unit per slot for a 1:1 read of the code table.
    """
    silu_S = float(torch.nn.functional.silu(torch.tensor(FFN_S)))
    n = len(code)
    # 2 self-clears (OP_VAL, IMM) + 2 value units per slot (op, imm)
    n_units = 2 + 2 * n
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    # self-clear OP_VAL and IMM (SET)
    for u, band in enumerate((L.OP_VAL, L.IMM)):
        W_up[u, L.ONE] = FFN_S
        W_gate[u, band] = 1.0
        W_down[band, u] = -1.0 / silu_S
    u = 2
    for i, ins in enumerate(code):
        # OP_VAL += op_i * ADDR_IS[i]  (constant value rides the gate BIAS)
        W_up[u, L.ADDR_IS[i]] = FFN_S       # detector: fires iff this slot's PC
        b_gate[u] = float(ins.op)           # gate = constant op_i (the baked value)
        W_down[L.OP_VAL, u] = 1.0 / silu_S
        u += 1
        # IMM += imm_i * ADDR_IS[i]
        W_up[u, L.ADDR_IS[i]] = FFN_S
        b_gate[u] = float(ins.imm)          # gate = constant imm_i (baked value)
        W_down[L.IMM, u] = 1.0 / silu_S
        u += 1
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# ----------------------------- program-independent downstream (fallback) ------
# If universal.py is not merged on this branch, provide the three shared blocks
# LOCALLY (identical algebra) so the baked model is self-contained. When
# universal.py IS present these are unused (we import its versions above), which
# keeps the "only the fetch differs" contrast literally true.

def _local_opcode_decode(L: Layout, dim: int):
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
    for t, j in thr_unit.items():
        W_up[j, L.OP_VAL] = RELU_S; b_up[j] = -RELU_S * t; W_gate[j, L.ONE] = 1.0
    for c, band in enumerate(op_is):
        u = clear0 + c
        W_up[u, L.ONE] = FFN_S; W_gate[u, band] = 1.0
        W_down[band, u] += -1.0 / silu_S
    for op, band in enumerate(op_is):
        W_down[band, thr_unit[op - 1]] += 1.0 / RELU_S
        W_down[band, thr_unit[op]] += -2.0 / RELU_S
        W_down[band, thr_unit[op + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def _local_dispatch_rules(L: Layout) -> List[FFNRule]:
    ax, stk, pc = L.AX, L.STACK0, L.PC
    imm, bp = L.IMM, L.BP
    OP = L.OP_IS

    def G(op):
        return [(OP + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    rules.append(FFNRule(G(isa.IMM),
                         {ax: LinearExpr.of(imm, 1.0) + LinearExpr.of(ax, -1.0),
                          pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.LEA),
                         {ax: LinearExpr.of(bp, 1.0) + LinearExpr.of(imm, 1.0)
                          + LinearExpr.of(ax, -1.0), pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.PSH),
                         {stk: LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0),
                          pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.ADD),
                         {ax: LinearExpr.of(stk, 1.0), pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.SUB),
                         {ax: LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0)
                          + LinearExpr.c(256.0), pc: LinearExpr.c(1.0)}))
    rules.append(FFNRule(G(isa.JMP),
                         {pc: LinearExpr.of(imm, 1.0) + LinearExpr.of(pc, -1.0)}))
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    rules.append(FFNRule(G(isa.HALT), {L.HALTED: LinearExpr.c(1.0)}))
    return rules


def _local_branch_delta(L: Layout, dim: int):
    pc, imm, azero, one = L.PC, L.IMM, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ
    n_units = 4
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    BIG = 200.0
    silu_big = float(torch.nn.functional.silu(torch.tensor(0.5 * BIG)))

    def _and_unit(u, op_band, bool_terms, gate_expr):
        W_up[u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                W_up[u, band] += BIG * coeff
            b_up[u] += BIG * const
        b_up[u] += -BIG * 1.5
        for band, coeff in gate_expr[0]:
            W_gate[u, band] += coeff
        b_gate[u] += gate_expr[1]

    _and_unit(0, BZ, [(azero, 1.0, 0.0)], ([(imm, 1.0), (pc, -1.0)], 0.0))
    W_down[pc, 0] += 1.0 / silu_big
    _and_unit(1, BZ, [(azero, -1.0, 1.0)], ([(one, 1.0)], 0.0))
    W_down[pc, 1] += 1.0 / silu_big
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)], ([(imm, 1.0), (pc, -1.0)], 0.0))
    W_down[pc, 2] += 1.0 / silu_big
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)], ([(one, 1.0)], 0.0))
    W_down[pc, 3] += 1.0 / silu_big
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------------------- the baked step-block -----

def _decode_spec(L, dim):
    return (compile_opcode_decode(L, dim) if _HAVE_UNIVERSAL
            else _local_opcode_decode(L, dim))


def _dispatch_rules(L):
    return (universal_dispatch_rules(L) if _HAVE_UNIVERSAL
            else _local_dispatch_rules(L))


def _branch_spec(L, dim):
    return (compile_branch_delta(L, dim) if _HAVE_UNIVERSAL
            else _local_branch_delta(L, dim))


def build_baked_step(code: List[isa.Instr], n_heads: int = 4, max_pos: int = 4):
    """Bake ONE step-block for a SPECIFIC ``code`` — the program IS the weights.

    Physical FFN sub-blocks (single residual position, no attention):

      1a. split  : PC -> PC_LO/PC_HI nibbles + AX_ZERO         [BAKED FETCH]
      1b. eqcheck: PC_LO/PC_HI -> EQ_LO[d]/EQ_HI[d] equalities [BAKED FETCH]
      2.  and    : ADDR_IS[i] = EQ_LO[lo_i] * EQ_HI[hi_i]      [BAKED FETCH]
      3.  value  : ADDR_IS[i] -> OP_VAL := op_i, IMM := imm_i  [BAKED FETCH]
      4.  decode : OP_VAL scalar -> OP_IS[op] one-hot          (program-INDEPENDENT)
      5.  dispatch: per-opcode rules gated on OP_IS[op]        (program-INDEPENDENT)
      6.  branch : bilinear BZ/BNZ PC update                   (program-INDEPENDENT)
      7.  fold   : AX mod-256                                  (program-INDEPENDENT)
      8.  emit   : AX -> OUT slot, HALTED -> HALT_SEEN         (program-INDEPENDENT)

    Blocks 1a-3 (fetch) carry the program in WEIGHTS. Blocks 4-8 are byte-identical
    to ``universal.build_universal_step``. Returns ``(model, layout)``.
    """
    L = build_baked_layout(len(code), n_heads=n_heads)
    dim = L.D

    ffn_specs = [
        compile_pc_split(L, dim),                     # 1a. PC -> nibbles + AX_ZERO
        compile_nibble_eq(L, dim),                    # 1b. per-nibble eq-check
        compile_addr_and(L, len(code), dim),          # 2. AND -> ADDR_IS one-hot
        compile_baked_value(L, code, dim),            # 3. MoE value (BAKED op/imm)
        _decode_spec(L, dim),                         # 4. OP_VAL -> OP_IS
        compile_ffn(_dispatch_rules(L), dim),         # 5. dispatch (non-branch PC)
        _branch_spec(L, dim),                         # 6. BZ/BNZ bilinear PC
        compile_fold(L.AX, L.ONE, dim, modulus=256),  # 7. AX mod-256
        compile_ffn([                                 # 8. emit + halt snapshot
            FFNRule([(L.ONE, 0.5, 1.5)], {L.OUT_SLOTS[0]: LinearExpr.of(L.AX, 1.0)
                                          + LinearExpr.of(L.OUT_SLOTS[0], -1.0)}),
            FFNRule([(L.ONE, 0.5, 1.5)], {L.HALT_SEEN[0]: LinearExpr.of(L.HALTED, 1.0)
                                          + LinearExpr.of(L.HALT_SEEN[0], -1.0)}),
        ], dim),
    ]
    n_blocks = len(ffn_specs)                          # == 9
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, max_pos=max_pos, vocab=VOCAB)
    with torch.no_grad():
        model.embed.zero_()
        model.embed[0, L.ONE] = 1.0                    # ONE lane = 1; NO program data
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


def initial_state_no_program(model, L) -> torch.Tensor:
    """The baked initial residual: ONE=1, every register/PC/scratch = 0.

    THE PROOF SURFACE: unlike ``universal.load_program`` (which writes the whole
    code table into DATA bands), this writes NOTHING program-specific. The bytecode
    is entirely in the weights; the input state has no CODE band at all. Assert
    ``(state == model.embed[0]).all()`` — the run starts from the bare embedding.
    """
    return model.embed[0].clone()


def _requantize(state: torch.Tensor, one_band: int) -> torch.Tensor:
    q = torch.round(state)
    q[one_band] = 1.0
    return q


def _step_once(model, state: torch.Tensor) -> torch.Tensor:
    x = state.view(1, 1, -1)
    for blk in model.blocks:
        x = blk(x)
    return x[0, 0]


def run_baked(model, L, max_steps: int = 100000, requantize: bool = True,
              trace_state: bool = False):
    """Run the BAKED program recurrently — WITH NO BYTECODE IN THE INPUT.

    The whole point: ``run_baked`` never loads a program. It starts from the bare
    embedding (ONE=1, all else 0) and the weights alone drive the fetch. Each
    iteration forwards the step-block, re-quantises to exact integers (kills fp
    residue -> exact over unbounded steps), decodes AX via the LM head, stops at
    HALT. Returns the per-step AX trace, matching ``isa.interpret``.
    """
    import torch.nn.functional as F
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]

    state = initial_state_no_program(model, L)
    trace: List[int] = []
    states = [] if trace_state else None
    for _ in range(max_steps):
        state = _step_once(model, state)
        if requantize:
            state = _requantize(state, L.ONE)
        if states is not None:
            states.append(state.clone())
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if trace_state:
        return trace, states
    return trace


class BakedProgram:
    """A C-program-turned-transformer: compile a program straight into WEIGHTS,
    then run it with NO bytecode in the input.

        bp = BakedProgram([("IMM", 5), ("PSH", 0), ("IMM", 3), ("ADD", 0),
                           ("HALT", 0)])
        bp.run()   # -> [5, 5, 3, 8, 8]   (no program in data — it's the weights)

    Contrast ``recurrent.StepModel`` (also bakes per-program, but via the baked
    PC_IS dispatch) and ``universal.UniversalInterpreter`` (ONE fixed model, program
    in DATA). This class realises the blogspec's eq-check -> AND -> one-hot -> MoE
    value read-only code segment explicitly.
    """

    def __init__(self, prog, n_heads: int = 4):
        self.code = isa.assemble(prog)
        self.model, self.L = build_baked_step(self.code, n_heads=n_heads)

    def run(self, max_steps: int = 100000, requantize: bool = True):
        return run_baked(self.model, self.L, max_steps=max_steps,
                         requantize=requantize)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)
