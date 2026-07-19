"""c4_min MODEL-RUNS-C, FETCH-DEDUP: shrink the O(code_size) fetch toward
O(distinct) so the FULL c4 compiler fits a tractable residual width D.

THE WALL (see docs/NIBBLE_MODEL_RUNS_C_VALIDATION_2026_07_18.md): the baseline
``build_baked_compiler_step`` allocates, PER code slot, THREE residual bands
``CODE_WORD[i] + PC_IS[i] + ADDR_IS[i]`` and an O(code_size)-wide fetch/decode
FFN. Measured ``D = 3.00*code_size + 148`` -> the full ~4000-instr c4 compiler
projects to D~12k (a 12k-wide residual with a 4000-wide FFN per block). That is a
SIZE wall, not a research wall.

THE FIX (two structural dedups, both bit-exact vs the reference interpreter):

  (1) REGION SPLIT.  The compiler bytecode (the BAKED region, PC in
      ``0..gen_size``) is CONSTANT -- its words ride weight biases, so it needs NO
      ``CODE_WORD`` memory dim and is NEVER an EMIT target, so it needs NO
      ``ADDR_IS`` store-address dim.  Only the small WRITABLE produced-code region
      (``out_size`` slots, contiguous right after the compiler at PC
      ``gen_size..gen_size+out_size``) needs ``OUT_WORD`` + ``OUT_ADDR_IS``.  This
      turns the ``3*code_size`` term into ``code_size + 2*out_size`` with
      ``out_size << code_size`` (the produced program is short).

  (2) FACTORED PC ONE-HOT.  ``PC_IS`` is consumed ONLY by the word-fetch (nothing
      else reads it; branches read PC/IMM/AX scalars).  Replace the full
      ``code_size``-wide PC one-hot with a base-``B`` DIGIT factorization
      ``PC = B*PC_HI + PC_LO``: two SMALL one-hots ``PC_HI_IS`` (``ceil(N/B)``
      bits) and ``PC_LO_IS`` (``B`` bits).  With ``B ~= sqrt(N)`` the PC
      addressing costs ``~2*sqrt(N)`` dims instead of ``N``.  The baked word at
      slot ``i = B*h + l`` is selected by a 2-input AND ``PC_HI_IS[h] AND
      PC_LO_IS[l]`` gated by the constant word ``w_i`` (the same exact-integer AND
      gadget EMIT already uses).

Net scaling: ``D ~= base + 2*sqrt(code_size) + 2*out_size + (src/mem/stack
overhead)`` -- SUB-LINEAR in code_size.  The produced-program semantics are
UNCHANGED; every gadget is the same exact-integer staircase / bilinear-AND used by
the baseline, so the model is byte-identical to
``nibble_compiler.interpret_full`` (proven in test_fetch_dedup.py) and hence to
the real c4 compiler for the arithmetic cores.

This module reuses ``nibble_compiler`` for everything except the layout and the
fetch/word-select (the two O(code_size) pieces).  The op dispatch (IMM/LEA/PSH/
ADD/SUB/MUL/JMP/BZ/BNZ/LC/LI/SI/EMIT/HALT), the SP-stack, the address one-hots,
and the run driver are imported unchanged.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

from . import isa
from . import nibble_compiler as C
from .compiler import VOCAB, _zero_attn, _load_ffn, _load_head, head_matrix
from .compile_ffn import compile_ffn, compile_fold, S as FFN_S, RELU_S
from .dsl import FFNRule, LinearExpr
from .layout import Layout
from .model import Transformer
from . import universal as U

EMIT = C.EMIT
# EMITP: emit at the RUNTIME output pointer OUT_PTR (a moving code-emit cursor),
# then OUT_PTR += 1.  Needed by a LOOP compiler that emits a variable number of
# produced instructions (its output address is not a compile-time constant).
EMITP = 31
isa.NAMES.setdefault(EMITP, "EMITP")
isa.BY_NAME.setdefault("EMITP", EMITP)
silu_S_full = float(F.silu(torch.tensor(FFN_S)))


# ------------------------------------------------------------------ layout ----

def _choose_base(n: int) -> int:
    """Base B ~= sqrt(n) that minimises the two-digit one-hot dim cost H + B where
    H = ceil(n / B). Small n -> B=n (degenerate to a full one-hot, cheapest)."""
    if n <= 1:
        return 1
    best_b, best_cost = 1, n + 1
    lo = max(1, int(math.isqrt(n)) - 2)
    for b in range(lo, int(math.isqrt(n)) + 4):
        if b < 1:
            continue
        h = (n + b - 1) // b
        cost = h + b
        if cost < best_cost:
            best_cost, best_b = cost, b
    return best_b


def build_dedup_compiler_layout(gen_size: int, out_size: int, src_size: int,
                                mem_size: int = 0, stack_depth: int = 8,
                                n_heads: int = 4, base=None) -> Layout:
    """Deduped compiler layout.  Address space is contiguous:

        PC in [0, gen_size)                 -> BAKED compiler (constants, factored PC)
        PC in [gen_size, gen_size+out_size) -> WRITABLE produced code (memory)
    """
    N = gen_size + out_size
    B = base if base is not None else _choose_base(N)
    H = (N + B - 1) // B

    L = Layout(n_heads=n_heads)
    L.GEN_SIZE = gen_size
    L.OUT_SIZE = out_size
    L.CODE_SIZE = N
    L.SRC_SIZE = src_size
    L.MEM_SIZE = mem_size
    L.ADDR_MAX = max(src_size, mem_size)
    L.PACKED = True
    L.PC_BASE = B
    L.PC_HI_N = H

    L.PC_HI = L._band("PC_HI", 1)
    L.PC_LO = L._band("PC_LO", 1)
    L.PC_HI_IS = [L._band(f"PC_HI_IS_{h}", 1) for h in range(H)]
    L.PC_LO_IS = [L._band(f"PC_LO_IS_{l}", 1) for l in range(B)]
    L.WORD = L._band("WORD", 1)
    L.OUT_WORD = [L._band(f"OUT_WORD_{k}", 1) for k in range(out_size)]
    L.OUT_ADDR_IS = [L._band(f"OUT_ADDR_IS_{k}", 1) for k in range(out_size)]
    L.SRC = [L._band(f"SRC_{i}", 1) for i in range(src_size)]
    L.AX_IS = [L._band(f"AX_IS_{i}", 1) for i in range(L.ADDR_MAX)]
    L.MEM = [L._band(f"MEM_{i}", 1) for i in range(mem_size)]
    L.MEM_IS = [L._band(f"MEM_IS_{i}", 1) for i in range(mem_size)]
    L.STACK_DEPTH = stack_depth
    L.STACK = [L._band(f"STACK_{i}", 1) for i in range(stack_depth)]
    L.SP_IS = [L._band(f"SP_IS_{i}", 1) for i in range(stack_depth)]
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


# ------------------------------------------- factored PC one-hot + AX_ZERO -----

def compile_factored_pc_fetch(L: Layout, dim: int):
    """(re)write the FACTORED PC one-hot + AX_ZERO from the scalar PC.

    Returns a LIST of two FFN specs:
      block A: PC_HI, PC_LO scalars + AX_ZERO (read PC / AX only).
      block B: PC_HI_IS, PC_LO_IS one-hots (read PC_HI / PC_LO).
    """
    B = L.PC_BASE
    H = L.PC_HI_N

    # ---- block A ----
    thr_set = sorted({B * h - 1 for h in range(1, H)} | {B * h for h in range(1, H)})
    thr_unit = {t: j for j, t in enumerate(thr_set)}
    n_relu = len(thr_set)
    clearA0 = n_relu
    clear_bandsA = [L.PC_HI, L.PC_LO, L.AX_ZERO]
    azA = clearA0 + len(clear_bandsA)
    idU = azA + 1
    nA = idU + 1
    Wu = torch.zeros(nA, dim); bu = torch.zeros(nA)
    Wg = torch.zeros(nA, dim); bg = torch.zeros(nA)
    Wd = torch.zeros(dim, nA); bd = torch.zeros(dim)
    for t, j in thr_unit.items():
        Wu[j, L.PC] = RELU_S; bu[j] = -RELU_S * t; Wg[j, L.ONE] = 1.0
    for c, band in enumerate(clear_bandsA):
        u = clearA0 + c
        Wu[u, L.ONE] = FFN_S; Wg[u, band] = 1.0
        Wd[band, u] += -1.0 / silu_S_full
    # PC_HI = sum_{h>=1} step(PC >= B*h)
    for h in range(1, H):
        Wd[L.PC_HI, thr_unit[B * h - 1]] += 1.0 / RELU_S
        Wd[L.PC_HI, thr_unit[B * h]] += -1.0 / RELU_S
    # AX_ZERO = relu(1 - AX)
    Wu[azA, L.AX] = -RELU_S; bu[azA] = RELU_S * 1.0; Wg[azA, L.ONE] = 1.0
    Wd[L.AX_ZERO, azA] += 1.0 / RELU_S
    # PC_LO = PC - B*PC_HI : +PC identity ...
    Wu[idU, L.PC] = RELU_S; Wg[idU, L.ONE] = 1.0
    Wd[L.PC_LO, idU] += 1.0 / RELU_S
    # ... - B*PC_HI (route each step unit with -B)
    for h in range(1, H):
        Wd[L.PC_LO, thr_unit[B * h - 1]] += (-B) / RELU_S
        Wd[L.PC_LO, thr_unit[B * h]] += (+B) / RELU_S
    specA = {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
             "W_down": Wd, "b_down": bd}

    # ---- block B ----
    def _onehot_spec(src_band, is_bands):
        n = len(is_bands)
        thr = list(range(-1, n + 1))
        tu = {t: j for j, t in enumerate(thr)}
        nrel = len(thr)
        c0 = nrel
        nu = c0 + n
        wu = torch.zeros(nu, dim); b_u = torch.zeros(nu)
        wg = torch.zeros(nu, dim); b_g = torch.zeros(nu)
        wd = torch.zeros(dim, nu); b_d = torch.zeros(dim)
        for t, j in tu.items():
            wu[j, src_band] = RELU_S; b_u[j] = -RELU_S * t; wg[j, L.ONE] = 1.0
        for c, band in enumerate(is_bands):
            u = c0 + c
            wu[u, L.ONE] = FFN_S; wg[u, band] = 1.0
            wd[band, u] += -1.0 / silu_S_full
        for i, band in enumerate(is_bands):
            wd[band, tu[i - 1]] += 1.0 / RELU_S
            wd[band, tu[i]] += -2.0 / RELU_S
            wd[band, tu[i + 1]] += 1.0 / RELU_S
        return wu, b_u, wg, b_g, wd, b_d

    hu, hbu, hg, hbg, hd, hbd = _onehot_spec(L.PC_HI, L.PC_HI_IS)
    lu, lbu, lg, lbg, ld, lbd = _onehot_spec(L.PC_LO, L.PC_LO_IS)
    WuB = torch.cat([hu, lu], 0); buB = torch.cat([hbu, lbu], 0)
    WgB = torch.cat([hg, lg], 0); bgB = torch.cat([hbg, lbg], 0)
    WdB = torch.cat([hd, ld], 1); bdB = hbd + lbd
    specB = {"W_up": WuB, "b_up": buB, "W_gate": WgB, "b_gate": bgB,
             "W_down": WdB, "b_down": bdB}
    return [specA, specB]


# ---------------------------------- deduped word-select (baked AND + memory) ----

def compile_dedup_word_select(L: Layout, gen_code: List[isa.Instr], dim: int):
    """WORD = baked-compiler word (factored PC one-hot AND, constant words)
              + produced-code word (OUT memory, factored one-hot AND).

    Baked slot i = B*h + l < gen_size:
        WORD += (PC_HI_IS[h] AND PC_LO_IS[l]) * word_i     # word_i constant bias
    Produced slot i (k = i - gen_size, 0 <= k < out_size):
        WORD += (PC_HI_IS[h] AND PC_LO_IS[l]) * OUT_WORD[k]

    Exact-integer 2-input AND gadget: up = BIG*(a+b-1.5), gate=value.  Only NONZERO
    baked words get a unit -> baked fetch width is O(#nonzero baked slots).
    """
    B = L.PC_BASE
    gen = len(gen_code)
    N = L.CODE_SIZE
    BIG = 512.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))

    units = []
    for i in range(N):
        h, l = divmod(i, B)
        if i < gen:
            w = float((gen_code[i].op & 0xFF) | ((gen_code[i].imm & 0xFF) << 8))
            if w != 0.0:
                units.append((h, l, ('const', w)))
        else:
            k = i - gen
            if k < L.OUT_SIZE:
                units.append((h, l, ('mem', L.OUT_WORD[k])))

    n_units = len(units) + 1
    Wu = torch.zeros(n_units, dim); bu = torch.zeros(n_units)
    Wg = torch.zeros(n_units, dim); bg = torch.zeros(n_units)
    Wd = torch.zeros(dim, n_units); bd = torch.zeros(dim)
    for u, (h, l, val) in enumerate(units):
        Wu[u, L.PC_HI_IS[h]] += BIG
        Wu[u, L.PC_LO_IS[l]] += BIG
        bu[u] += -BIG * 1.5
        if val[0] == 'const':
            bg[u] = val[1]
        else:
            Wg[u, val[1]] = 1.0
        Wd[L.WORD, u] += 1.0 / silu_big
    # self-clear WORD (SET).  MUST use the EXACT power-of-two normaliser POW2=256
    # (silu(256)=256 exactly, 1/256 exact in fp32): WORD reaches 65535, and a
    # generic 1/silu(60)=1/60 reciprocal is INEXACT (~1.7e-2), so at large WORD_prev
    # the self-clear undershoots by ~WORD_prev*3e-5 -> the freshly-selected word
    # lands ~0.001 low -> OP_VAL/OP_IS decode 0.9999 not 1.0 -> the SUB `+256`
    # constant lands at 255.96 < the fold threshold 256 -> the fold silently fails
    # and AX carries garbage.  256*65535 < 2**24 so every intermediate is exact.
    POW2 = 256.0
    uc = len(units)
    Wu[uc, L.WORD] = POW2; Wg[uc, L.ONE] = 1.0
    Wd[L.WORD, uc] += -1.0 / POW2
    return {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
            "W_down": Wd, "b_down": bd}


# --------------------------------- EMIT store-addr one-hot over OUT region ----

def compile_out_store_addr_onehot(L: Layout, dim: int):
    """OUT_ADDR_IS[k] = (IMM - gen_size == k): the EMIT store-address one-hot over
    the WRITABLE produced-code region only.  Triangular pulse over (IMM - gen)."""
    n = L.OUT_SIZE
    gen = L.GEN_SIZE
    thr = list(range(-1, n + 1))
    tu = {t: j for j, t in enumerate(thr)}
    nrel = len(thr)
    c0 = nrel
    nu = c0 + n
    Wu = torch.zeros(nu, dim); bu = torch.zeros(nu)
    Wg = torch.zeros(nu, dim); bg = torch.zeros(nu)
    Wd = torch.zeros(dim, nu); bd = torch.zeros(dim)
    for t, j in tu.items():
        Wu[j, L.IMM] = RELU_S; bu[j] = -RELU_S * (t + gen); Wg[j, L.ONE] = 1.0
    for c, band in enumerate(L.OUT_ADDR_IS):
        u = c0 + c
        Wu[u, L.ONE] = FFN_S; Wg[u, band] = 1.0
        Wd[band, u] += -1.0 / silu_S_full
    for i, band in enumerate(L.OUT_ADDR_IS):
        Wd[band, tu[i - 1]] += 1.0 / RELU_S
        Wd[band, tu[i]] += -2.0 / RELU_S
        Wd[band, tu[i + 1]] += 1.0 / RELU_S
    return {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
            "W_down": Wd, "b_down": bd}


# ----------------------------------------- EMIT store into OUT_WORD memory ----

def compile_out_emit_store(L: Layout, dim: int):
    """EMIT: ``OUT_WORD[IMM-gen] := AX + 256*STACK0``, gated on OP_IS[EMIT].
    PC += 1; SP -= 1.  Store-select over OUT_WORD addressed by OUT_ADDR_IS."""
    n = L.OUT_SIZE
    BIG = 256.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))
    op_emit = L.OP_IS + EMIT
    n_units = 3 * n + 2
    Wu = torch.zeros(n_units, dim); bu = torch.zeros(n_units)
    Wg = torch.zeros(n_units, dim); bg = torch.zeros(n_units)
    Wd = torch.zeros(dim, n_units); bd = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_terms, dst, coeff):
        Wu[u, op_emit] += BIG
        Wu[u, addr_band] += BIG
        bu[u] += -BIG * 1.5
        for band, cg in gate_terms:
            Wg[u, band] += cg
        Wd[dst, u] += coeff / silu_big

    u = 0
    for k in range(n):
        _and_unit(u, L.OUT_ADDR_IS[k], [(L.AX, 1.0)], L.OUT_WORD[k], +1.0); u += 1
        _and_unit(u, L.OUT_ADDR_IS[k], [(L.STACK0, 256.0)], L.OUT_WORD[k], +1.0); u += 1
        _and_unit(u, L.OUT_ADDR_IS[k], [(L.OUT_WORD[k], 1.0)], L.OUT_WORD[k], -1.0); u += 1
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    Wu[u, op_emit] += FFN_S; bu[u] += -FFN_S * 0.5
    Wg[u, L.ONE] += 1.0
    Wd[L.PC, u] += 1.0 / silu_S; u += 1
    Wu[u, op_emit] += FFN_S; bu[u] += -FFN_S * 0.5
    Wg[u, L.ONE] += 1.0
    Wd[L.SP, u] += -1.0 / silu_S
    return {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
            "W_down": Wd, "b_down": bd}


# ------------------------------------------------------- the deduped step -----

def build_dedup_baked_compiler_step(gen_code: List[isa.Instr], out_size: int,
                                    src_size: int, mem_size: int = 0,
                                    stack_depth: int = 8, n_heads: int = 4,
                                    max_pos: int = 4, base=None):
    """COMPILER-IN-WEIGHTS with the DEDUPED (region-split + factored PC) fetch.
    Same op semantics as ``nibble_compiler.build_baked_compiler_step``; only the
    fetch / word-select / EMIT-store blocks changed.  Returns ``(model, L)``.
    """
    from .stack import compile_sp_fetch, compile_stack0_select
    gen_size = len(gen_code)
    L = build_dedup_compiler_layout(gen_size, out_size, src_size, mem_size=mem_size,
                                    stack_depth=stack_depth, n_heads=n_heads, base=base)
    dim = L.D

    pc_fetch_specs = compile_factored_pc_fetch(L, dim)
    ffn_specs = [
        *pc_fetch_specs,
        compile_dedup_word_select(L, gen_code, dim),
        U.compile_word_decode_imm(L, dim),
        U.compile_word_decode_op(L, dim),
        U.compile_opcode_decode(L, dim),
        compile_sp_fetch(L.SP, L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_stack0_select(L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_out_store_addr_onehot(L, dim),
        C.compile_ax_addr_onehot(L, dim),
        C.compile_mem_store_addr_onehot(L, dim),
        compile_ffn(C.compiler_dispatch_rules(L), dim),
        C.compile_mul_dispatch(L, dim),
        C.compile_lc_source_read(L, dim),
        C.compile_li_mem_read(L, dim),
        C.compile_si_mem_store(L, dim),
        compile_out_emit_store(L, dim),
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


def initial_state_dedup(model, L, src: List[int]) -> torch.Tensor:
    """Initial state: ONLY the C source in SRC.  OUT_WORD starts all-zero."""
    assert len(src) <= L.SRC_SIZE
    state = model.embed[0].clone()
    for i, ch in enumerate(src):
        state[L.SRC[i]] = float(ch & 0xFF)
    return state


def run_dedup_compiler(model, L, src: List[int], max_steps: int = 8192,
                       requantize: bool = True, return_code: bool = False):
    """Run the deduped compiler-in-weights on C ``src`` (source only in the state)."""
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]
    state = initial_state_dedup(model, L, src)
    trace: List[int] = []
    for _ in range(max_steps):
        state = C._step_once(model, state)
        if requantize:
            state = C._requantize(state, L)
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if return_code:
        words = [int(round(float(state[L.OUT_WORD[k]]))) for k in range(L.OUT_SIZE)]
        return trace, words
    return trace


class DedupBakedCompilerMachine:
    """A C compiler that IS a transformer, with the DEDUPED sub-linear fetch.

    NOTE: the compiler bytecode's EMIT/JMP targets must point at the OUT region,
    which starts at ``gen_size`` (contiguous).  Build the compiler with
    ``outbase = len(gen_code)`` (see ``expr_compiler_bytecode(outbase=...)``).
    """

    def __init__(self, prog, out_size: int, src_size: int, mem_size: int = 8,
                 stack_depth: int = 8, n_heads: int = 4, base=None):
        self.gen_code = C._assemble(prog)
        self.out_size = out_size
        self.model, self.L = build_dedup_baked_compiler_step(
            self.gen_code, out_size, src_size, mem_size=mem_size,
            stack_depth=stack_depth, n_heads=n_heads, base=base)

    def run(self, source, max_steps: int = 8192, requantize: bool = True,
            return_code: bool = False):
        return run_dedup_compiler(self.model, self.L, C._source_bytes(source),
                                  max_steps=max_steps, requantize=requantize,
                                  return_code=return_code)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)


# ============================================================================ #
#  LOOP COMPILER: a variable-length code emitter with a MOVING output pointer   #
#                                                                               #
#  The fixed-slot EMIT (CODE[imm] := ...) forces a compile-time-constant target #
#  slot per emit-site, so a compiler that emits an UNBOUNDED number of produced #
#  instructions (a loop over a variable-length source) cannot use it.  EMITP    #
#  writes at the RUNTIME cursor OUT_PTR and bumps it, so a bytecode LOOP can     #
#  emit as many words as the source demands.  This is the mechanism a real      #
#  recursive/looping c4 codegen needs (``code[code_pos++] = word``).            #
# ============================================================================ #

def build_loop_compiler_layout(gen_size: int, out_size: int, src_size: int,
                               mem_size: int = 0, stack_depth: int = 8,
                               n_heads: int = 4, base=None) -> Layout:
    """As ``build_dedup_compiler_layout`` plus an OUT_PTR emit cursor + its
    one-hot (OUT_PTR_IS over the OUT region) for the moving-pointer EMITP."""
    N = gen_size + out_size
    B = base if base is not None else _choose_base(N)
    H = (N + B - 1) // B

    L = Layout(n_heads=n_heads)
    L.GEN_SIZE = gen_size
    L.OUT_SIZE = out_size
    L.CODE_SIZE = N
    L.SRC_SIZE = src_size
    L.MEM_SIZE = mem_size
    L.ADDR_MAX = max(src_size, mem_size)
    L.PACKED = True
    L.PC_BASE = B
    L.PC_HI_N = H

    L.PC_HI = L._band("PC_HI", 1)
    L.PC_LO = L._band("PC_LO", 1)
    L.PC_HI_IS = [L._band(f"PC_HI_IS_{h}", 1) for h in range(H)]
    L.PC_LO_IS = [L._band(f"PC_LO_IS_{l}", 1) for l in range(B)]
    L.WORD = L._band("WORD", 1)
    L.OUT_WORD = [L._band(f"OUT_WORD_{k}", 1) for k in range(out_size)]
    L.OUT_ADDR_IS = [L._band(f"OUT_ADDR_IS_{k}", 1) for k in range(out_size)]
    # moving emit cursor: OUT_PTR is an ABSOLUTE code address (gen_size..N);
    # OUT_PTR_IS[k] = (OUT_PTR - gen_size == k) is the store-address one-hot.
    L.OUT_PTR = L._band("OUT_PTR", 1)
    L.OUT_PTR_IS = [L._band(f"OUT_PTR_IS_{k}", 1) for k in range(out_size)]
    L.SRC = [L._band(f"SRC_{i}", 1) for i in range(src_size)]
    L.AX_IS = [L._band(f"AX_IS_{i}", 1) for i in range(L.ADDR_MAX)]
    L.MEM = [L._band(f"MEM_{i}", 1) for i in range(mem_size)]
    L.MEM_IS = [L._band(f"MEM_IS_{i}", 1) for i in range(mem_size)]
    L.STACK_DEPTH = stack_depth
    L.STACK = [L._band(f"STACK_{i}", 1) for i in range(stack_depth)]
    L.SP_IS = [L._band(f"SP_IS_{i}", 1) for i in range(stack_depth)]
    L.OP_VAL = L._band("OP_VAL", 1)
    L.OP_IS = L._band("OP_IS", isa.NUM_OPS)
    L.OUT_SLOTS = [L._band("OUT_0", 1)]
    L.HALT_SEEN = [L._band("HSEEN_0", 1)]
    while L._off % n_heads != 0:
        L._band(f"_pad2_{L._off}", 1)
    L.D = L._off
    return L


def compile_out_ptr_onehot(L: Layout, dim: int):
    """OUT_PTR_IS[k] = (OUT_PTR - gen_size == k): the EMITP store-address one-hot
    over the OUT region (indexed by the moving emit cursor OUT_PTR)."""
    n = L.OUT_SIZE
    gen = L.GEN_SIZE
    thr = list(range(-1, n + 1))
    tu = {t: j for j, t in enumerate(thr)}
    nrel = len(thr)
    c0 = nrel
    nu = c0 + n
    Wu = torch.zeros(nu, dim); bu = torch.zeros(nu)
    Wg = torch.zeros(nu, dim); bg = torch.zeros(nu)
    Wd = torch.zeros(dim, nu); bd = torch.zeros(dim)
    for t, j in tu.items():
        Wu[j, L.OUT_PTR] = RELU_S; bu[j] = -RELU_S * (t + gen); Wg[j, L.ONE] = 1.0
    for c, band in enumerate(L.OUT_PTR_IS):
        u = c0 + c
        Wu[u, L.ONE] = FFN_S; Wg[u, band] = 1.0
        Wd[band, u] += -1.0 / silu_S_full
    for i, band in enumerate(L.OUT_PTR_IS):
        Wd[band, tu[i - 1]] += 1.0 / RELU_S
        Wd[band, tu[i]] += -2.0 / RELU_S
        Wd[band, tu[i + 1]] += 1.0 / RELU_S
    return {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
            "W_down": Wd, "b_down": bd}


def compile_emitp_store(L: Layout, dim: int):
    """EMITP: ``OUT_WORD[OUT_PTR-gen] := AX + 256*STACK0``, gated on OP_IS[EMITP].
    Then PC += 1; SP -= 1 (pop produced-imm); OUT_PTR += 1 (advance the cursor).
    Store-select over OUT_WORD addressed by OUT_PTR_IS (the moving cursor one-hot).
    """
    n = L.OUT_SIZE
    BIG = 256.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))
    op = L.OP_IS + EMITP
    n_units = 3 * n + 3
    Wu = torch.zeros(n_units, dim); bu = torch.zeros(n_units)
    Wg = torch.zeros(n_units, dim); bg = torch.zeros(n_units)
    Wd = torch.zeros(dim, n_units); bd = torch.zeros(dim)

    def _and_unit(u, addr_band, gate_terms, dst, coeff):
        Wu[u, op] += BIG
        Wu[u, addr_band] += BIG
        bu[u] += -BIG * 1.5
        for band, cg in gate_terms:
            Wg[u, band] += cg
        Wd[dst, u] += coeff / silu_big

    u = 0
    for k in range(n):
        _and_unit(u, L.OUT_PTR_IS[k], [(L.AX, 1.0)], L.OUT_WORD[k], +1.0); u += 1
        _and_unit(u, L.OUT_PTR_IS[k], [(L.STACK0, 256.0)], L.OUT_WORD[k], +1.0); u += 1
        _and_unit(u, L.OUT_PTR_IS[k], [(L.OUT_WORD[k], 1.0)], L.OUT_WORD[k], -1.0); u += 1
    silu_S = float(F.silu(torch.tensor(0.5 * FFN_S)))
    Wu[u, op] += FFN_S; bu[u] += -FFN_S * 0.5; Wg[u, L.ONE] += 1.0
    Wd[L.PC, u] += 1.0 / silu_S; u += 1
    Wu[u, op] += FFN_S; bu[u] += -FFN_S * 0.5; Wg[u, L.ONE] += 1.0
    Wd[L.SP, u] += -1.0 / silu_S; u += 1
    Wu[u, op] += FFN_S; bu[u] += -FFN_S * 0.5; Wg[u, L.ONE] += 1.0
    Wd[L.OUT_PTR, u] += 1.0 / silu_S
    return {"W_up": Wu, "b_up": bu, "W_gate": Wg, "b_gate": bg,
            "W_down": Wd, "b_down": bd}


def build_loop_compiler_step(gen_code: List[isa.Instr], out_size: int,
                             src_size: int, mem_size: int = 8,
                             stack_depth: int = 8, n_heads: int = 4,
                             max_pos: int = 4, base=None, out_ptr_init=None):
    """COMPILER-IN-WEIGHTS with the deduped fetch AND a moving-pointer EMITP, so a
    bytecode LOOP can emit a variable number of produced instructions.  OUT_PTR is
    initialised to ``out_ptr_init`` (default gen_size, the start of the OUT region)
    in the embedding.  Returns ``(model, L)``.
    """
    from .stack import compile_sp_fetch, compile_stack0_select
    gen_size = len(gen_code)
    L = build_loop_compiler_layout(gen_size, out_size, src_size, mem_size=mem_size,
                                   stack_depth=stack_depth, n_heads=n_heads, base=base)
    L.OUT_PTR_INIT = gen_size if out_ptr_init is None else out_ptr_init
    dim = L.D

    pc_fetch_specs = compile_factored_pc_fetch(L, dim)
    ffn_specs = [
        *pc_fetch_specs,
        compile_dedup_word_select(L, gen_code, dim),
        U.compile_word_decode_imm(L, dim),
        U.compile_word_decode_op(L, dim),
        U.compile_opcode_decode(L, dim),
        compile_sp_fetch(L.SP, L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_stack0_select(L.STACK, L.SP_IS, L.STACK0, L.ONE, dim),
        compile_out_store_addr_onehot(L, dim),
        compile_out_ptr_onehot(L, dim),
        C.compile_ax_addr_onehot(L, dim),
        C.compile_mem_store_addr_onehot(L, dim),
        compile_ffn(C.compiler_dispatch_rules(L), dim),
        C.compile_mul_dispatch(L, dim),
        C.compile_lc_source_read(L, dim),
        C.compile_li_mem_read(L, dim),
        C.compile_si_mem_store(L, dim),
        compile_out_emit_store(L, dim),
        compile_emitp_store(L, dim),
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
        model.embed[0, L.OUT_PTR] = float(L.OUT_PTR_INIT)
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
        _load_head(model, L)
    return model, L


def run_loop_compiler(model, L, src: List[int], max_steps: int = 8192,
                      requantize: bool = True, return_code: bool = False):
    W, b = head_matrix(L, model.dim, L.OUT_SLOTS[0], halt_band=None)
    halt_seen = L.HALT_SEEN[0]
    state = model.embed[0].clone()
    for i, ch in enumerate(src):
        state[L.SRC[i]] = float(ch & 0xFF)
    trace: List[int] = []
    for _ in range(max_steps):
        state = C._step_once(model, state)
        if requantize:
            state = C._requantize(state, L)
        logits = F.linear(state, W, b)
        trace.append(int(logits.argmax().item()))
        if float(state[halt_seen]) > 0.5:
            break
    if return_code:
        words = [int(round(float(state[L.OUT_WORD[k]]))) for k in range(L.OUT_SIZE)]
        return trace, words
    return trace


class LoopCompilerMachine:
    """A C compiler that IS a transformer, deduped fetch + moving-pointer EMITP.
    Its bytecode can LOOP over a variable-length source, emitting a variable number
    of produced instructions at the runtime cursor OUT_PTR (start = gen_size)."""

    def __init__(self, prog, out_size: int, src_size: int, mem_size: int = 8,
                 stack_depth: int = 8, n_heads: int = 4, base=None):
        self.gen_code = _assemble_loop(prog)
        self.out_size = out_size
        self.model, self.L = build_loop_compiler_step(
            self.gen_code, out_size, src_size, mem_size=mem_size,
            stack_depth=stack_depth, n_heads=n_heads, base=base)

    def run(self, source, max_steps: int = 8192, requantize: bool = True,
            return_code: bool = False):
        return run_loop_compiler(self.model, self.L, C._source_bytes(source),
                                 max_steps=max_steps, requantize=requantize,
                                 return_code=return_code)

    @property
    def n_blocks(self) -> int:
        return len(self.model.blocks)


def _assemble_loop(prog) -> List[isa.Instr]:
    """assemble understanding EMIT (30) and EMITP (31) mnemonics."""
    out = []
    for entry in prog:
        name, imm = (entry if isinstance(entry, tuple) else (entry, 0))
        if name == "EMIT":
            op = EMIT
        elif name == "EMITP":
            op = EMITP
        else:
            op = isa.BY_NAME[name]
        out.append(isa.Instr(op, imm & 0xFF))
    return out
