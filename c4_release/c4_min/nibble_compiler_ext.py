"""RUNG-2 EXTENSION of the compiler-in-weights (task #848): add the C RELATIONAL
operators (``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``) to the BakedCompilerMachine
so the transformer can compile-and-run a genuine C *comparison* expression
(``5>3``) BYTE-IDENTICAL to the real c4 compiler, not just ``+``/``*`` arithmetic.

WHY this is the honest next rung and where the taller rungs wall
----------------------------------------------------------------
The base ``nibble_compiler`` machine's produced-program executor
(``compiler_dispatch_rules`` + the ``interpret_full`` reference) implements only
IMM/LEA/PSH/ADD/SUB/MUL/JMP/BZ/BNZ/HALT + LC/LI/SI/EMIT.  A produced ``GT`` word
raises ``NotImplementedError op 20`` — so even the *comparison fragment* of the
task's rung-2 (``if``/comparison + ``while``) could not run.  This module adds a
single fixed-weight COMPARISON block (all six relational ops, exact-integer
step-function gadget) to the machine's dispatch AND to the reference interpreter,
then a compiler bytecode that reads ``d0 <op> d1`` from the SOURCE and emits
``IMM d0; PSH; IMM d1; <CMP>`` — the SAME core the real c4 compiler emits for
``return d0 <op> d1;`` (verified byte-identical against ``./c4c``).

The taller rungs (a local variable ``int x; x=..; return x*4;`` and a ``while``
loop) are **architecturally blocked**, NOT a missing-emit / decode bug: real c4
addresses locals frame-relative (``LEA -8`` / ``LEV`` / ``ENT``) with a 32-bit
signed immediate (``0xfffff800``), and the whole c4_min machine is an *8-bit-only*
subset (``isa.MASK == 0xFF``) whose EMIT packs ``op | imm<<8`` (imm in 0..255) and
whose ``LEA`` is ``AX := BP+imm`` with no negative frame offset.  So a produced
``LEA -8`` word is UNENCODABLE by construction, and the model's produced core can
never be byte-identical to c4's frame-relative local-variable body.  This module
does NOT try to fake it; it delivers the tallest rung that IS byte-exact-reachable
and documents the wall precisely (see ``docs`` string of ``cmp_compiler_bytecode``).

GOLDEN: this module is a STANDALONE artifact on the ``nibble_compiler`` lineage; it
imports the base machine and only ADDS an extra FFN block + a bytecode.  It does
NOT touch the main pure-forward VM that ``_fingerprint_build`` hashes, so the
golden ``7d4afe61`` is unchanged (verified).
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from . import isa
from .compile_ffn import compile_ffn, compile_fold, RELU_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer
from . import universal as U
from . import nibble_compiler as C
from .compiler import VOCAB, _zero_attn, _load_ffn, _load_head

EMIT = C.EMIT

# The six relational opcodes and their Python predicate on (a=STACK0, b=AX).
# c4 semantics: ``ax = (pop() <op> ax)`` -> a is the popped stack operand, b is AX.
_CMP_OPS: List[Tuple[int, str]] = [
    (isa.EQ, "=="), (isa.NE, "!="), (isa.LT, "<"),
    (isa.GT, ">"), (isa.LE, "<="), (isa.GE, ">="),
]


# --------------------------------------------------------------------------- #
#  COMPARISON dispatch block: AX := (STACK0 <cmp> AX) ? 1 : 0 ; SP -= 1 ; PC++  #
# --------------------------------------------------------------------------- #

def compile_cmp_dispatch(L: Layout, dim: int):
    """One fixed-weight FFN block implementing ALL SIX relational ops.

    For each op, gated on the decoded one-hot ``OP_IS[op]``, it writes into AX the
    0/1 result of ``STACK0 <op> AX`` (both integer operands in 0..255), then
    ``SP -= 1`` (the comparison pops the left operand) and ``PC += 1``.

    Exact-integer step gadget (no fp slop): let ``d = STACK0 - AX`` (an integer in
    [-255, 255]).  A crisp 0/1 indicator ``ge1(x) = relu(x) - relu(x-1)`` fires iff
    the integer ``x >= 1``; ``le0`` / ``eq0`` are built from the same clamped-relu
    differences.  Each relu is ``silu(RELU_S*z)/RELU_S`` (the ``compile_fold``
    identity).  The six predicates:

        LT (a<b)  : d <= -1  ->  ge1(-d)
        GT (a>b)  : d >=  1  ->  ge1( d)
        LE (a<=b) : d <=  0  ->  ge1(1 - d)
        GE (a>=b) : d >=  0  ->  ge1(1 + d)
        EQ (a==b) : d == 0   ->  1 - ge1(d) - ge1(-d)
        NE (a!=b) : d != 0   ->  ge1(d) + ge1(-d)

    Each predicate's contribution is itself AND-gated on the decoded ``OP_IS[op]``
    so the block is a NO-OP for every non-comparison opcode.  Because only one
    ``OP_IS`` is hot per step, exactly one predicate's writes are active, and they
    SET AX (self-clear ``-AX`` on the same op) so AX becomes exactly the 0/1 flag.
    """
    one = L.ONE
    ax, stk, sp, pc = L.AX, L.STACK0, L.SP, L.PC
    OP = L.OP_IS

    # We assemble the tensors directly (multi-relu products need explicit units).
    units = []   # each: dict of (up_terms, up_bias, gate_terms, gate_bias, down_dst, down_coeff)

    def relu_pair(d_terms, d_bias, op_band, ax_write_coeff):
        """Emit the two relu units for ge1(d) = relu(d) - relu(d-1), AND-gated on
        the decoded op.  Since ``ge1`` is 0/1 and the op one-hot is 0/1, we gate the
        relu VALUE on the op one-hot via a product: hidden = silu(op AND d-ramp).

        To keep it exact we use the SAME trick the base ops use: the relu computes
        ``relu(d - thr)`` ONLY when the op is hot, by adding ``-HUGE*(1-OP_IS[op])``
        into the pre-activation so the ramp is pushed below zero (silu ~ 0) when the
        op is cold.  On the hot op, ``OP_IS[op]==1`` cancels the HUGE term and the
        ramp is the plain ``relu(d-thr)``.
        """
        HUGE = 1.0e5
        for thr, coeff in ((0, +1.0), (1, -1.0)):
            up_terms = [(b, RELU_S * c) for (b, c) in d_terms]
            up_terms.append((op_band, HUGE))
            up_bias = -RELU_S * thr - HUGE
            units.append(dict(up_terms=up_terms, up_bias=up_bias,
                              gate_terms=[(one, 1.0)], gate_bias=0.0,
                              down_dst=ax, down_coeff=(ax_write_coeff * coeff) / RELU_S))

    # ---- clear AX_old and bump SP/PC, per comparison op (SET semantics) ----
    from .compile_ffn import S as FFN_S
    silu_S = float(torch.nn.functional.silu(torch.tensor(0.5 * FFN_S)))
    for op, _sym in _CMP_OPS:
        op_band = OP + op
        # clear AX_old: up = FFN_S*(OP_IS[op]-0.5) ; gate = AX_old ; down -1/silu_S
        units.append(dict(up_terms=[(op_band, FFN_S)], up_bias=-FFN_S * 0.5,
                          gate_terms=[(ax, 1.0)], gate_bias=0.0,
                          down_dst=ax, down_coeff=-1.0 / silu_S))
        # SP -= 1
        units.append(dict(up_terms=[(op_band, FFN_S)], up_bias=-FFN_S * 0.5,
                          gate_terms=[(one, 1.0)], gate_bias=0.0,
                          down_dst=sp, down_coeff=-1.0 / silu_S))
        # PC += 1
        units.append(dict(up_terms=[(op_band, FFN_S)], up_bias=-FFN_S * 0.5,
                          gate_terms=[(one, 1.0)], gate_bias=0.0,
                          down_dst=pc, down_coeff=+1.0 / silu_S))

    # ---- the predicate writes into AX (after the clear), per op ----
    # d = STACK0 - AX  (as linear terms)
    dpos = [(stk, 1.0), (ax, -1.0)]      #  d
    dneg = [(stk, -1.0), (ax, 1.0)]      # -d
    # LT: ge1(-d)
    relu_pair(dneg, 0.0, OP + isa.LT, +1.0)
    # GT: ge1(d)
    relu_pair(dpos, 0.0, OP + isa.GT, +1.0)
    # LE: ge1(1 - d) -> d_terms = (1 - d), realised via constant on ONE
    relu_pair([(one, 1.0), (stk, -1.0), (ax, 1.0)], 0.0, OP + isa.LE, +1.0)
    # GE: ge1(1 + d)
    relu_pair([(one, 1.0), (stk, 1.0), (ax, -1.0)], 0.0, OP + isa.GE, +1.0)
    # NE: ge1(d) + ge1(-d)
    relu_pair(dpos, 0.0, OP + isa.NE, +1.0)
    relu_pair(dneg, 0.0, OP + isa.NE, +1.0)
    # EQ: 1 - ge1(d) - ge1(-d)  -> add +1 (gated on op) then subtract the two ge1
    #   the +1 constant, AND-gated on OP_IS[EQ]:
    units.append(dict(up_terms=[(OP + isa.EQ, FFN_S)], up_bias=-FFN_S * 0.5,
                      gate_terms=[(one, 1.0)], gate_bias=0.0,
                      down_dst=ax, down_coeff=+1.0 / silu_S))
    relu_pair(dpos, 0.0, OP + isa.EQ, -1.0)
    relu_pair(dneg, 0.0, OP + isa.EQ, -1.0)

    n_units = len(units)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for u, spec in enumerate(units):
        for band, coeff in spec["up_terms"]:
            W_up[u, band] += coeff
        b_up[u] += spec["up_bias"]
        for band, coeff in spec["gate_terms"]:
            W_gate[u, band] += coeff
        b_gate[u] += spec["gate_bias"]
        W_down[spec["down_dst"], u] += spec["down_coeff"]
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------------------------------------------- #
#  RUNG-3 (Doom fixed-point) SHIFT dispatch: AX := STACK0 <<|>> AX ; SP-=1 ; PC++
#
#  Doom's m_fixed.c / tables.c fixed-point math is dominated by ``x << FRACBITS`` /
#  ``x >> n`` / masks.  These are the byte-exact-reachable Doom-arithmetic ops (no
#  frame / 32-bit immediate needed).  We add SHL and SHR to the produced-program
#  executor.  The shift amount rides AX (0..8 for single-digit sources), gated by a
#  LOCAL exact one-hot ``AX==k``; the value ``STACK0`` is shifted by the CONSTANT
#  ``2**k`` for that branch, so every arithmetic sub-unit is an exact integer op
#  (mul by 2**k for SHL; the exact integer floor-divide STAIRCASE for SHR).
# --------------------------------------------------------------------------- #

def compile_shift_dispatch(L: Layout, dim: int, max_shift: int = None):
    """SHL (``AX := STACK0 << AX``) and SHR (``AX := STACK0 >> AX``), exact-integer,
    gated on the decoded ``OP_IS[SHL]`` / ``OP_IS[SHR]`` and a LOCAL ``AX==k``
    one-hot (k in 0..max_shift).  SP -= 1, PC += 1 on either op.

    For a fixed shift amount ``k`` the ops are constant-coefficient integer maps:
        SHL:  STACK0 * 2**k                       (mod-256 fold downstream)
        SHR:  floor(STACK0 / 2**k) = sum_{m>=1} [ STACK0 >= m * 2**k ]
    where each ``[STACK0 >= T]`` is the exact clamped-relu step
    ``relu(STACK0-(T-1)) - relu(STACK0-T)`` on integers.  Every unit is AND-gated
    on ``OP_IS[op] AND (AX==k)`` via the sharp-silu product (both in {0,1}), so the
    block is a NO-OP unless the op is SHL/SHR with that exact amount.
    """
    one = L.ONE
    ax, stk, sp, pc = L.AX, L.STACK0, L.SP, L.PC
    OP = L.OP_IS
    # Cover shift amounts 0..(ADDR_MAX-1): AX_IS[k]=(AX==k) exists for k < ADDR_MAX.
    # Any single-digit shift amount >= ADDR_MAX zeros the byte anyway (x<<8 & 0xFF ==
    # 0, x>>8 == 0 for a byte x), so amounts beyond the covered range are a no-op ->
    # AX stays cleared, which is the correct byte result for those amounts too.
    if max_shift is None:
        max_shift = L.ADDR_MAX - 1
    from .compile_ffn import S as FFN_S
    silu_S = float(torch.nn.functional.silu(torch.tensor(0.5 * FFN_S)))
    HUGE = 1.0e5
    units = []

    # --- AX==k amount one-hot: reuse the layout's AX_IS one-hot when it covers
    #     0..max_shift; the compiler layout sizes AX_IS to ADDR_MAX = max(src,mem)
    #     which is >= 8 in practice, so AX_IS[k] is exactly (AX==k).  The AX_IS
    #     block (``compile_ax_addr_onehot``) runs BEFORE this block in the pipeline
    #     (it is used by LC/LI), so AX_IS already holds (AX==k) for the fetched AX.
    #     compiler layout sizes AX_IS to ADDR_MAX = max(src,mem) which is >= 8 in
    #     practice, so AX_IS[k] is exactly (AX==k).  Guard that assumption.
    assert L.ADDR_MAX > max_shift, (
        f"AX_IS one-hot (ADDR_MAX={L.ADDR_MAX}) must cover shift amounts 0..{max_shift}")
    AXK = [L.AX_IS[k] for k in range(max_shift + 1)]

    def and3_step(op_band, k_band, thr, dst, coeff):
        """coeff * ( relu(STACK0-(thr-1)) - relu(STACK0-thr) ), AND-gated on
        OP_IS[op] AND AX==k.  Two relu units."""
        for t, c in ((thr - 1, +1.0), (thr, -1.0)):
            up_terms = [(stk, RELU_S), (op_band, HUGE), (k_band, HUGE)]
            up_bias = -RELU_S * t - 2 * HUGE
            units.append(dict(up=up_terms, ub=up_bias, gate=[(one, 1.0)], gb=0.0,
                              dst=dst, dc=(coeff * c) / RELU_S))

    def and_mul(op_band, k_band, factor, dst):
        """dst += factor * STACK0, AND-gated on OP_IS[op] AND AX==k.  ONE unit:
        gate = STACK0 (linear), up = HUGE*(op + k - 2) + big so silu ~ big iff both
        one-hots hold, else ~0; down = factor / silu(big)."""
        BIG = 200.0
        silu_big = float(torch.nn.functional.silu(torch.tensor(0.5 * BIG)))
        up_terms = [(op_band, BIG), (k_band, BIG)]
        up_bias = -BIG * 1.5
        units.append(dict(up=up_terms, ub=up_bias, gate=[(stk, 1.0)], gb=0.0,
                          dst=dst, dc=factor / silu_big))

    # clear AX_old + SP-=1 + PC+=1, once per shift op (SET semantics), gated on op.
    for op in (isa.SHL, isa.SHR):
        ob = OP + op
        units.append(dict(up=[(ob, FFN_S)], ub=-FFN_S * 0.5, gate=[(ax, 1.0)], gb=0.0,
                          dst=ax, dc=-1.0 / silu_S))
        units.append(dict(up=[(ob, FFN_S)], ub=-FFN_S * 0.5, gate=[(one, 1.0)], gb=0.0,
                          dst=sp, dc=-1.0 / silu_S))
        units.append(dict(up=[(ob, FFN_S)], ub=-FFN_S * 0.5, gate=[(one, 1.0)], gb=0.0,
                          dst=pc, dc=+1.0 / silu_S))

    # SHL: AX += STACK0 * 2**k  for the active k.
    for k in range(max_shift + 1):
        and_mul(OP + isa.SHL, AXK[k], float(1 << k), ax)
    # SHR: AX += floor(STACK0 / 2**k) = sum_{m>=1, m*2**k <= 255} [STACK0 >= m*2**k].
    for k in range(max_shift + 1):
        step = 1 << k
        m = 1
        while m * step <= 255:
            and3_step(OP + isa.SHR, AXK[k], m * step, ax, +1.0)
            m += 1

    n_units = len(units)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    for u, s in enumerate(units):
        for band, coeff in s["up"]:
            W_up[u, band] += coeff
        b_up[u] += s["ub"]
        for band, coeff in s["gate"]:
            W_gate[u, band] += coeff
        b_gate[u] += s["gb"]
        W_down[s["dst"], u] += s["dc"]
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# --------------------------------------------------------------------------- #
#  build the RUNG-2 baked compiler step (base blocks + the comparison block)   #
# --------------------------------------------------------------------------- #

def build_baked_compiler_step_ext(gen_code, code_size, src_size, mem_size=0,
                                  stack_depth=8, n_heads=4, max_pos=4):
    """``nibble_compiler.build_baked_compiler_step`` + the comparison dispatch
    block inserted right after MUL (so a produced ``d0 <cmp> d1`` runs)."""
    from .stack import compile_sp_fetch, compile_stack0_select
    L = C.build_compiler_layout(code_size, src_size, mem_size=mem_size,
                                stack_depth=stack_depth, n_heads=n_heads)
    L.GEN_SIZE = len(gen_code)
    dim = L.D

    ffn_specs = [
        U.compile_fetch_select(L, dim),
        C.compile_hybrid_word_select(L, gen_code, dim),
        U.compile_word_decode_imm(L, dim),
        U.compile_word_decode_op(L, dim),
        U.compile_opcode_decode(L, dim),
        compile_sp_fetch(L.SP, L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_stack0_select(L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        C.compile_store_addr_onehot(L, dim),
        C.compile_ax_addr_onehot(L, dim),
        C.compile_mem_store_addr_onehot(L, dim),
        compile_ffn(C.compiler_dispatch_rules(L), dim),
        C.compile_mul_dispatch(L, dim),
        compile_cmp_dispatch(L, dim),                     # <-- NEW: relational ops
        compile_shift_dispatch(L, dim),                   # <-- NEW: SHL/SHR (Doom fixed-point)
        C.compile_lc_source_read(L, dim),
        C.compile_li_mem_read(L, dim),
        C.compile_si_mem_store(L, dim),
        C.compile_emit_store(L, dim),
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


class BakedCompilerMachineExt(C.BakedCompilerMachine):
    """BakedCompilerMachine + the relational-op comparison block."""

    def __init__(self, prog, code_size, src_size, mem_size=8, stack_depth=8,
                 n_heads=4):
        self.gen_code = C._assemble(prog)
        self.code_size = code_size
        self.model, self.L = build_baked_compiler_step_ext(
            self.gen_code, code_size, src_size, mem_size=mem_size,
            stack_depth=stack_depth, n_heads=n_heads)


# --------------------------------------------------------------------------- #
#  reference interpreter that mirrors the extended machine (adds relational)   #
# --------------------------------------------------------------------------- #

def interpret_full_ext(code, src, code_size, mem_size=256, max_steps=8192):
    """``nibble_compiler.interpret_full`` + the six relational ops (op pops the
    left operand, pushes the 0/1 result into AX)."""
    ax = bp = sp = 0
    pc = 0
    stack = [0] * (mem_size + 8)
    mem = [0] * mem_size
    words = [0] * code_size
    for i, ins in enumerate(code):
        words[i] = (ins.op & 0xFF) | ((ins.imm & 0xFF) << 8)
    source = list(src) + [0] * code_size
    emitted = []

    def top():
        return stack[sp - 1] if sp >= 1 else 0

    _CMP = {isa.EQ: lambda a, b: 1 if a == b else 0,
            isa.NE: lambda a, b: 1 if a != b else 0,
            isa.LT: lambda a, b: 1 if a < b else 0,
            isa.GT: lambda a, b: 1 if a > b else 0,
            isa.LE: lambda a, b: 1 if a <= b else 0,
            isa.GE: lambda a, b: 1 if a >= b else 0}

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
        elif op in _CMP:
            ax = _CMP[op](top(), ax); sp -= 1
        elif op == isa.SHL:
            ax = (top() << ax) & 0xFF; sp -= 1
        elif op == isa.SHR:
            ax = (top() >> ax) & 0xFF; sp -= 1
        elif op == isa.LC:
            ax = source[ax] & 0xFF if ax < len(source) else 0
        elif op == isa.LI:
            ax = mem[ax] & 0xFF if ax < len(mem) else 0
        elif op == isa.SI:
            if stack0 < len(mem):
                mem[stack0] = ax & 0xFF
        elif op == EMIT:
            words[imm % code_size] = (ax & 0xFF) | ((stack0 & 0xFF) << 8)
            sp = max(0, sp - 1)
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            emitted.append(ax); break
        else:
            raise NotImplementedError(f"op {op} not in ext compiler ISA")
        emitted.append(ax)
    return emitted, words


# --------------------------------------------------------------------------- #
#  the RUNG-2 comparison compiler bytecode: compile ``d0 <op> d1``             #
# --------------------------------------------------------------------------- #

# ASCII of the six relational operator lead chars the compiler recognises.
_LT = ord("<")   # 60
_GT = ord(">")   # 62
_EQ = ord("=")   # 61  (in "==")
_BANG = ord("!") # 33  (in "!=")


def cmp_compiler_bytecode(outbase: int = C.COMPILER_OUTBASE):
    """Compiler bytecode: read ``d0 <op> d1`` from SOURCE and emit the c4 core
    ``IMM d0; PSH; IMM d1; <CMP>; HALT``.

    Source layout (single-digit operands): ``src[0]=d0, src[1..]=operator,
    src[k]=d1`` where the operator is one of ``<  >  <= >= == !=`` (1 or 2 chars).
    The compiler reads ``src[1]`` (operator lead char) and, if it is ``<`` or
    ``>`` or ``=``/``!``, reads ``src[2]`` to disambiguate ``<`` vs ``<=`` etc.,
    then emits the matching relational opcode.  d1 sits at ``src[2]`` for a
    1-char operator and ``src[3]`` for a 2-char operator.

    The produced core is byte-identical to what real c4 emits for
    ``int main(){ return d0 <op> d1; }`` (JSR/ENT prologue + trailing LEV stripped
    by ``c4_arith_core``): e.g. ``5>3`` -> ``IMM 5; PSH; IMM 3; GT`` (== 1).
    """
    O = outbase
    a = C._Asm()

    def emit_imm_digit(slot, digit_pos):
        # AX := src[digit_pos] - '0' ; push produced-imm ; AX := opcode IMM ; EMIT
        a.emit(isa.IMM, digit_pos); a.emit(isa.LC, 0)
        a.emit(isa.PSH, 0); a.emit(isa.IMM, 48); a.emit(isa.SUB, 0)
        a.emit(isa.PSH, 0)
        a.emit(isa.IMM, isa.IMM)
        a.emit(EMIT, slot)

    def emit_simple(slot, opcode):
        a.emit(isa.IMM, 0); a.emit(isa.PSH, 0)
        a.emit(isa.IMM, opcode)
        a.emit(EMIT, slot)

    def read_src_minus(pos, ch, into_label_if_zero=None):
        """AX := src[pos] - ch ; (== 0 iff src[pos]==ch)."""
        a.emit(isa.IMM, pos); a.emit(isa.LC, 0)
        a.emit(isa.PSH, 0); a.emit(isa.IMM, ch); a.emit(isa.SUB, 0)

    # produced slots 0..1: IMM d0 ; PSH  (d0 = src[0])
    emit_imm_digit(O + 0, 0)
    emit_simple(O + 1, isa.PSH)

    # --- decode the operator at src[1] (lead char) ---
    # branch on lead char: '<' -> lt group, '>' -> gt group, '=' -> EQ, '!' -> NE
    read_src_minus(1, _LT); a.emit(isa.BZ, "op_lt")
    read_src_minus(1, _GT); a.emit(isa.BZ, "op_gt")
    read_src_minus(1, _EQ); a.emit(isa.BZ, "op_eq")   # '==' : lead '='
    # else assume '!=' (lead '!')
    a.emit(isa.JMP, "op_ne")

    # ---- '<' group: '<' (d1 at src[2]) vs '<=' (d1 at src[3]) ----
    a.label("op_lt")
    read_src_minus(2, _EQ); a.emit(isa.BZ, "op_le")   # src[2]=='=' -> '<='
    emit_imm_digit(O + 2, 2)                          # d1 at src[2]
    emit_simple(O + 3, isa.LT)
    a.emit(isa.JMP, "fin")
    a.label("op_le")
    emit_imm_digit(O + 2, 3)                          # d1 at src[3]
    emit_simple(O + 3, isa.LE)
    a.emit(isa.JMP, "fin")

    # ---- '>' group ----
    a.label("op_gt")
    read_src_minus(2, _EQ); a.emit(isa.BZ, "op_ge")   # src[2]=='=' -> '>='
    emit_imm_digit(O + 2, 2)
    emit_simple(O + 3, isa.GT)
    a.emit(isa.JMP, "fin")
    a.label("op_ge")
    emit_imm_digit(O + 2, 3)
    emit_simple(O + 3, isa.GE)
    a.emit(isa.JMP, "fin")

    # ---- '==' (d1 at src[3]) ----
    a.label("op_eq")
    emit_imm_digit(O + 2, 3)
    emit_simple(O + 3, isa.EQ)
    a.emit(isa.JMP, "fin")

    # ---- '!=' (d1 at src[3]) ----
    a.label("op_ne")
    emit_imm_digit(O + 2, 3)
    emit_simple(O + 3, isa.NE)

    a.label("fin")
    emit_simple(O + 4, isa.HALT)
    a.emit(isa.JMP, O)                                # HANDOFF -> run produced
    return a.resolve()


def shift_compiler_bytecode(outbase: int = C.COMPILER_OUTBASE):
    """RUNG-3 (Doom fixed-point): compile ``d0 << d1`` / ``d0 >> d1`` from SOURCE to
    the c4 core ``IMM d0; PSH; IMM d1; <SHL|SHR>; HALT`` — byte-identical to what
    real c4 emits for ``int main(){ return d0 << d1; }``.

    Doom's ``m_fixed.c`` / ``tables.c`` fixed-point math is exactly this shape
    (``x << FRACBITS``, ``x >> n``), so this is the genuine Doom-arithmetic rung
    that IS byte-exact reachable on the 8-bit machine.  Source: ``d0`` at src[0],
    the 2-char operator (``<<`` or ``>>``) at src[1..2], ``d1`` at src[3]."""
    O = outbase
    a = C._Asm()

    def emit_imm_digit(slot, digit_pos):
        a.emit(isa.IMM, digit_pos); a.emit(isa.LC, 0)
        a.emit(isa.PSH, 0); a.emit(isa.IMM, 48); a.emit(isa.SUB, 0)
        a.emit(isa.PSH, 0); a.emit(isa.IMM, isa.IMM); a.emit(EMIT, slot)

    def emit_simple(slot, opcode):
        a.emit(isa.IMM, 0); a.emit(isa.PSH, 0)
        a.emit(isa.IMM, opcode); a.emit(EMIT, slot)

    emit_imm_digit(O + 0, 0)                          # IMM d0
    emit_simple(O + 1, isa.PSH)                       # PSH
    emit_imm_digit(O + 2, 3)                          # IMM d1 (past the 2-char op)
    # operator: src[1] is '<' (SHL) or '>' (SHR)
    a.emit(isa.IMM, 1); a.emit(isa.LC, 0)
    a.emit(isa.PSH, 0); a.emit(isa.IMM, _LT); a.emit(isa.SUB, 0)
    a.emit(isa.BZ, "shl")                             # src[1]=='<' -> SHL
    emit_simple(O + 3, isa.SHR)                       # else '>>' -> SHR
    a.emit(isa.JMP, "fin")
    a.label("shl")
    emit_simple(O + 3, isa.SHL)
    a.label("fin")
    emit_simple(O + 4, isa.HALT)
    a.emit(isa.JMP, O)
    return a.resolve()
