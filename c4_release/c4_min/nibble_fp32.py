"""NATIVE IEEE-754 single (binary32) ALU as persistent transformer MEGABLOCKS.

The gated c4+FP ISA extension (``C4_FLOAT_OPS``): F_ADD/F_SUB/F_MUL/F_DIV as
SINGLE VM opcodes, so the efficient full-C90 float path is ONE op instead of the
hundreds-of-steps soft-float routine.  A float32 value is carried as its raw
32-bit int bit pattern in the register nibble bands (STACK0 = popped operand a,
AX = operand b), exactly like the integer ALU; these blocks DECODE the bit
pattern (sign / exponent / significand), compute the correctly-rounded op, and
RE-ENCODE the result bits into a dedicated ``F_RES`` nibble band the ax-mux copies
into AX on the active float opcode.

Design (all fp32-exact — no hidden unit ever holds a value >= 2^24):
  * DECODE: extract the 32 individual bits (``_step_ge`` staircase over the 8
    nibbles, reusing the same bit-plane trick as ``nibble_bitwise``), then compose
    ``S`` (sign 0/1), ``E`` (exponent 0..255), ``M`` (stored mantissa 0..2^23-1).
    The significand ``F = (E>0)*2^23 + M`` (implicit leading 1 for normals) and the
    "is-zero / is-inf / is-nan / is-subnormal" class predicates are small integers.
  * F_MUL: multiply the two 24-bit significands (schoolbook nibble multiply, the
    ``_mul_gate`` + carry-round chain), add the unbiased exponents, normalise the
    48-bit product to a 24-bit significand + guard/round/sticky, round-to-nearest-
    even, re-bias, encode.  Special cases (0/inf/nan) via ``_guard`` overrides.
  * F_ADD/F_SUB: align the smaller significand by the exponent difference (a
    variable right shift realised as a floor-divide by ``2**d`` staircase, keeping
    guard+round+sticky), add or subtract (SUB flips b's sign), renormalise (leading
    bit search up OR down), round, encode.
  * F_DIV: divide the dividend significand (shifted up by 2 extra bits for round
    material) by the divisor significand via the base-16 long divider (reused),
    subtract exponents, normalise, round, encode.

Every construction is validated BIT-EXACT vs gcc + the ``isa.f32_op_bits`` oracle
by ``_fp32_verify.py``.  HONEST scope of the megablock form is documented per-op
in that verifier's report (subnormal results and the full tie-to-even set are the
hard cases; the covered range is stated explicitly).

The public builders return ``(name, spec)`` FFN sub-block lists (the same SwiGLU
tensor-dict container the integer ALU uses) that ``build_pure_forward_complete_model``
wires like any other op block, gated behind ``C4_FLOAT_OPS`` (default OFF -> the
golden ``_fingerprint_build`` build is byte-identical: 7d4afe61 by default, or
069cc32f under the ``C4_BP_RESTORE_HIBYTE=0`` escape hatch).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from . import isa
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF, _empty_spec

# Reuse the exact-integer SwiGLU primitives from the integer ALU verbatim: the
# same _ident / _relu / _step_ge / _guard / _mul_gate / carry-round gadgets, so a
# hidden unit is fp32-exact on integer forms and the block container matches.
from . import nibble_alu32 as A

_SILU_G = SILU_HALF


def _set_one(L):
    """Point the shared ALU emitters at this layout's ONE lane (they read the
    module-global ``_ONE``)."""
    A._ONE = L.ONE


FLOAT_OPS = (isa.F_ADD, isa.F_SUB, isa.F_MUL, isa.F_DIV)


def compile_all_fp_blocks(L, dim):
    """The full gated FLOAT block stack: each op computes its result into F_RES; the
    per-op blocks are UNCONDITIONAL (they always run) so only ONE op's result is
    valid per step — the ax-mux copies F_RES into AX gated on the active OP_IS[F_*].

    Since all four ops share the decode + F_RES band, we run them SEQUENTIALLY, each
    guarded so it only writes F_RES when its opcode is active.  To keep it simple and
    correct, the four op stacks are emitted back-to-back but each op's FINAL F_RES
    write is OP-gated; a cleaner integration (block-MoE per opcode) is the runtime
    concern.  Here we build the DECODE once, then the four op cores, then the mux."""
    # NOTE: the four op pipelines write shared scratch (SIG/MROUND/F_RES/...), so we
    # cannot naively concatenate all four (later ops clobber earlier F_RES).  The mux
    # therefore reads F_RES right after EACH op's core, latching the active op's
    # result.  Implemented as: decode | MUL-core | latch(F_MUL) | ADD-core |
    # latch(F_ADD) | SUB-core | latch(F_SUB) | DIV-core | latch(F_DIV).  Each latch
    # copies F_RES -> the AX nibble band gated on that op.  All ops run every step
    # (unconditional compute); only the matching latch fires.
    blocks = list(compile_fp_decode_blocks(L, dim))
    blocks += _fp_mul_blocks(L, dim)
    blocks.append(("fp-mul-norm", _fp_mul_normalize_block(L, dim)))
    blocks.append(("fp-mul-sticky", _fp_mul_sticky_block(L, dim)))
    blocks += _fp_round_encode_tail(L, dim)
    blocks.append(("fp-mul-special", _fp_mul_special_flags_block(L, dim)))
    blocks.append(("fp-special-combine", _fp_special_combine_block(L, dim)))
    blocks.append(("fp-mul-override", _fp_special_override_block(L, dim)))
    blocks.append(("fp-latch-mul", _fp_latch_block(L, dim, isa.F_MUL)))
    # ADD (re-decode not needed; decode fields persist).  b is untouched for ADD.
    blocks += _fp_addsub_core(L, dim)
    blocks.append(("fp-latch-add", _fp_latch_block(L, dim, isa.F_ADD)))
    # SUB: flip b's sign, run the add core.  The flip mutates B_S / decode; so we
    # re-decode first to restore B_S, then flip.
    blocks += list(compile_fp_decode_blocks(L, dim))
    blocks.append(("fp-sub-flip", _fp_sub_flip_block(L, dim)))
    blocks += _fp_addsub_core(L, dim)
    blocks.append(("fp-latch-sub", _fp_latch_block(L, dim, isa.F_SUB)))
    # DIV: re-decode (the add core consumed scratch), run div core.
    blocks += list(compile_fp_decode_blocks(L, dim))
    blocks.append(("fp-div-init", _fp_div_init_block(L, dim)))
    for k in range(_DIV_SHIFT + 1):
        blocks.append((f"fp-div-sh{k}", _fp_div_shift_block(L, dim, k)))
        blocks.append((f"fp-div-q{k}", _fp_div_q_block(L, dim, k)))
        blocks.append((f"fp-div-qc{k}", _fp_div_qcombine_block(L, dim, k)))
        blocks.append((f"fp-div-sb{k}", _fp_div_sub_block(L, dim, k)))
    blocks.append(("fp-div-lead", _fp_div_lead_block(L, dim)))
    blocks.append(("fp-div-norm", _fp_div_normalize_block(L, dim)))
    blocks.append(("fp-div-stickyf", _fp_div_sticky_final_block(L, dim)))
    blocks += _fp_round_encode_tail(L, dim)
    blocks.append(("fp-div-special", _fp_div_special_flags_block(L, dim)))
    blocks.append(("fp-special-combine", _fp_special_combine_block(L, dim)))
    blocks.append(("fp-div-override", _fp_special_override_block(L, dim)))
    blocks.append(("fp-latch-div", _fp_latch_block(L, dim, isa.F_DIV)))
    return blocks


def _fp_latch_block(L, dim, op):
    """Copy the current F_RES nibbles into the AX nibble band, gated on OP_IS[op].
    SET: clear AX[c] then add F_RES[c], both gated on the op (so a non-op step or a
    different-op step leaves AX untouched)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 8 * 2 + 4)
    u = 0
    g = (L.OP_IS + op, 1.0, 0.0)
    for c in range(8):
        u = A._guard(spec, u, [g], {L.AX + c: -1.0}, 0.0, L.AX + c, 1.0)   # clear AX[c]
        u = A._guard(spec, u, [g], {f.F_RES + c: 1.0}, 0.0, L.AX + c, 1.0)  # + F_RES[c]
    return A._truncate(spec, u, dim)


# ===========================================================================
# Scratch bands for the float ALU (allocated only when C4_FLOAT_OPS is on, LAST,
# so a flag-OFF build's L.D + every earlier band offset is byte-identical to the
# golden layout).
# ===========================================================================
class FP32Bands:
    def __init__(self, L):
        # Per-operand decoded fields (a = STACK0, b = AX).
        self.A_BIT = L._band("FP_A_BIT", 32)     # the 32 bits of operand a
        self.B_BIT = L._band("FP_B_BIT", 32)     # the 32 bits of operand b
        self.A_S = L._scalar("FP_A_S")           # sign bit (0/1)
        self.B_S = L._scalar("FP_B_S")
        self.A_E = L._scalar("FP_A_E")           # biased exponent 0..255
        self.B_E = L._scalar("FP_B_E")
        self.A_M = L._band("FP_A_M", 24)         # significand bits (bit0..23), incl implicit
        self.B_M = L._band("FP_B_M", 24)
        self.A_FP = L._scalar("FP_A_FP")         # [frac(a) != 0]  (fraction bits nonzero)
        self.B_FP = L._scalar("FP_B_FP")
        # class predicates per operand.
        self.A_ISZ = L._scalar("FP_A_ISZ")       # value is +/-0 (E==0 and M==0)
        self.B_ISZ = L._scalar("FP_B_ISZ")
        self.A_ISINF = L._scalar("FP_A_ISINF")   # E==255 and frac==0
        self.B_ISINF = L._scalar("FP_B_ISINF")
        self.A_ISNAN = L._scalar("FP_A_ISNAN")   # E==255 and frac!=0
        self.B_ISNAN = L._scalar("FP_B_ISNAN")
        # Result assembly.
        self.R_S = L._scalar("FP_R_S")           # result sign bit
        self.R_E = L._scalar("FP_R_E")           # result biased exponent (int)
        self.R_M = L._band("FP_R_M", 32)         # result significand bits (working, incl guard)
        self.R_SPECIAL = L._scalar("FP_R_SPECIAL")  # 1 -> a special value is forced (skip normal encode)
        self.R_BITS = L._band("FP_R_BITS", 32)   # the assembled 32 result bits
        self.F_RES = L._band("FP_F_RES", 8)      # result nibbles (what the ax-mux reads)
        # --- F_MUL significand multiply scratch ---
        # significands as 6 nibbles each (24 bits); the schoolbook product is 12
        # nibble columns (48 bits).  Every column kept < 256 for fp32-exact carries.
        self.AN = L._band("FP_AN", 6)             # A significand nibbles
        self.BN = L._band("FP_BN", 6)             # B significand nibbles
        self.PPM = L._band("FP_PPM", 36)          # raw nibble partial products a_i*b_j
        self.PCOL = L._band("FP_PCOL", 12)        # 12 product columns (col c weight 16^c)
        self.PCOL2 = L._band("FP_PCOL2", 12)      # carry double-buffer
        self.PBIT = L._band("FP_PBIT", 48)        # the 48 product bits
        # --- normalise / round / encode working (shared by all ops) ---
        self.SIG = L._band("FP_SIG", 27)          # normalised 24-bit significand + G/R/S at [-1..-3]
        self.EXPADJ = L._scalar("FP_EXPADJ")      # exponent adjust from normalise
        self.GRS = L._band("FP_GRS", 3)           # guard / round / sticky
        self.STICKY = L._scalar("FP_STICKY")      # OR of the dropped low bits
        self.ROUNDUP = L._scalar("FP_ROUNDUP")    # 1 iff round increments significand
        self.RE_ROUND = L._scalar("FP_RE_ROUND")  # exp bump if rounding overflows the significand
        self.MROUND = L._band("FP_MROUND", 25)    # 24-bit significand + 1 carry slot after round
        # --- special-case selector flags (own band; never aliased with ADD scratch) ---
        self.FLAGS = L._band("FP_FLAGS", 12)      # NANF,INF0,INFR,ZR,OVF,UDF,ANYF,QF,IF_,ZF,CANCEL
        # --- F_ADD/F_SUB scratch ---
        self.WIDE = L._band("FP_WIDE", 56)        # aligned a significand (bits, high anchor)
        self.WIDE2 = L._band("FP_WIDE2", 56)      # aligned b significand (bits)
        self.SUMB = L._band("FP_SUMB", 56)        # the added / subtracted significand bits
        self.SHD = L._scalar("FP_SHD")            # alignment shift distance (exp difference)
        self.SHD_OH = L._band("FP_SHD_OH", 32)    # one-hot(SHD) for the variable shift
        self.LZ = L._band("FP_LZ", 56)            # leading-one position search for renorm
        self.ASWAP = L._scalar("FP_ASWAP")        # 1 iff B has the larger magnitude (swap)
        self.ESUB = L._scalar("FP_ESUB")          # 1 iff effective subtraction (signs differ)
        # --- F_DIV scratch (bit-serial restoring division of the significands) ---
        # The remainder is kept as an EXACT BIT-VECTOR (each slot 0/1), NOT a big
        # scalar: a scalar remainder doubled 26x amplifies the ~value*2^-24 gadget
        # residue past 1.0 and corrupts the quotient (see _fp_div_debug_findings.md).
        # DRB holds R as 26 bits (R < 2*B_M < 2^25, +1 headroom for the pre-compare
        # left shift).  BBIT holds the divisor significand B_M as 24 bits.
        self.DRB = L._band("FP_DRB", 27)          # remainder bit-vector (bit i = 2^i)
        self.DRB2 = L._band("FP_DRB2", 27)        # remainder double-buffer (shifted S bits)
        self.BBIT = L._band("FP_BBIT", 24)        # divisor significand B_M bits
        # nibble-domain subtract scratch: S nibbles, ~B nibbles, two column buffers
        # (double-buffered carry-settle), and the extracted quotient bit / diff bits.
        self.DSN = L._band("FP_DSN", 8)           # S significand as 8 nibbles (< 2^26 -> 7 nib)
        self.DCOL = L._band("FP_DCOL", 8)         # subtract column accumulator
        self.DCOL2 = L._band("FP_DCOL2", 8)       # carry-settle double-buffer
        self.DDIFF = L._band("FP_DDIFF", 32)      # S - B result bits (from settled nibbles)
        self.DCMP = L._band("FP_DCMP", 8)         # quotient bit + flags scratch
        self.QBIT = L._band("FP_QBIT", 28)        # quotient bits (MSB-first fill)


def extend_layout_for_fp32(L):
    """Allocate FP32 scratch bands on ``L`` (only when ``C4_FLOAT_OPS`` on)."""
    if getattr(L, "FP32", None) is not None:
        return L.FP32
    L.FP32 = FP32Bands(L)
    while L._off % L.n_heads != 0:
        L._scalar(f"_fppad{L._off}")
    L.D = L._off
    return L.FP32


# ===========================================================================
# 0. DECODE : the 32 register nibbles of a/b -> bits -> S / E / significand.
#    Reuses the bit-plane staircase (bit p of nibble v = floor(v/2^p) mod 2).
# ===========================================================================
def _bit_extract_units(spec, u, L, src_nib, bit_base):
    """Materialise bit_base+0..3 = the 4 bits of the nibble at ``src_nib`` (0..15).
    bit p = [v>=..] staircase: floor(v/2^p) - 2*floor(v/2^(p+1)), each floor an
    integer staircase of unit-rise relu steps.  SET each."""
    def floor_terms(M):
        terms = {}
        m = M
        while m <= 15:
            terms[m - 1] = terms.get(m - 1, 0.0) + 1.0
            terms[m] = terms.get(m, 0.0) - 1.0
            m += M
        return terms

    for p in range(4):
        dst = bit_base + p
        u = A._clear(spec, u, dst)
        hi = floor_terms(1 << p)
        lo = floor_terms(1 << (p + 1))
        net = {}
        for t, c in hi.items():
            net[t] = net.get(t, 0.0) + c
        for t, c in lo.items():
            net[t] = net.get(t, 0.0) - 2.0 * c
        for t, c in net.items():
            if c == 0.0:
                continue
            # step(v>=t+1) contribution == relu(v-t) - relu(v-(t+1)); but 'net'
            # already encodes the relu-threshold deltas: coeff c at threshold t
            # means +c*relu(v - t).
            u = A._relu(spec, u, {src_nib: 1.0}, -float(t), dst, c)
    return u


def _decode_operand_block(L, dim, reg_base, bit_base):
    """Extract the 32 bits of a register (8 nibbles) into ``bit_base+0..31``."""
    _set_one(L)
    # per nibble: 4 bit-planes; bit 0's floor staircases alone use ~22 relu units.
    # Budget generously (truncated to the real ``u`` on return).
    spec = _empty_spec(dim, 8 * 4 * 40)
    u = 0
    for j in range(8):
        u = _bit_extract_units(spec, u, L, reg_base + j, bit_base + 4 * j)
    return A._truncate(spec, u, dim)


def _fields_block(L, dim, bit_base, S_dst, E_dst, M_dst, FP_dst):
    """Compose sign / exponent / significand + the frac-nonzero flag from the 32
    raw bits (all read the BLOCK INPUT):
      S = bit31 ; E = Σ_{k=0..7} 2^k * bit(23+k) ;
      significand M[i] = bit(i) for i<23, M[23] = [E>0] (implicit leading 1) ;
      FP = [frac != 0] = [Σ_{i<23} bit(i) >= 1].
    """
    _set_one(L)
    spec = _empty_spec(dim, 600)
    u = 0
    b = bit_base
    exp_sum = {b + 23 + k: 1.0 for k in range(8)}              # Σ exponent bits
    frac_sum = {b + i: 1.0 for i in range(23)}                 # Σ fraction bits
    # sign
    u = A._clear(spec, u, S_dst)
    u = A._ident(spec, u, {b + 31: 1.0}, 0.0, S_dst, 1.0)
    # exponent E = Σ 2^k bit(23+k)
    u = A._clear(spec, u, E_dst)
    for k in range(8):
        u = A._ident(spec, u, {b + 23 + k: float(1 << k)}, 0.0, E_dst, 1.0)
    # significand bits: copy fraction bits, set implicit bit 23 = [E>0] = [exp_sum>=1]
    for i in range(23):
        u = A._clear(spec, u, M_dst + i)
        u = A._ident(spec, u, {b + i: 1.0}, 0.0, M_dst + i, 1.0)
    u = A._clear(spec, u, M_dst + 23)
    u = A._step_ge(spec, u, exp_sum, 0.0, 1, M_dst + 23, 1.0)      # [E>=1]
    # frac-nonzero flag
    u = A._clear(spec, u, FP_dst)
    u = A._step_ge(spec, u, frac_sum, 0.0, 1, FP_dst, 1.0)         # [frac>=1]
    return A._truncate(spec, u, dim)


def _class_block(L, dim, E_src, FP_src, ISZ, ISINF, ISNAN):
    """Class predicates from the already-materialised E (0..255) and FP=[frac!=0].
    A SEPARATE block so it reads the WRITTEN E/FP (not the stale block input).
      ISZ   = [E==0] AND [frac==0]  = 1 - [E + FP >= 1]     (E>=1 or frac -> not zero)
      ISINF = [E==255] AND [frac==0] = [E>=255] - [E+FP>=256]
      ISNAN = [E==255] AND [frac!=0] = [E + FP >= 256]      (E==255 -> E+FP==256 iff FP=1)
    (E in [0,255]; E+FP in [0,256]; the thresholds isolate each class exactly.)"""
    _set_one(L)
    spec = _empty_spec(dim, 40)
    u = 0
    ef = {E_src: 1.0, FP_src: 1.0}
    u = A._clear(spec, u, ISZ)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, ISZ, 1.0)
    u = A._step_ge(spec, u, ef, 0.0, 1, ISZ, -1.0)               # 1 - [E+FP>=1]
    u = A._clear(spec, u, ISINF)
    u = A._step_ge(spec, u, {E_src: 1.0}, 0.0, 255, ISINF, 1.0)  # [E>=255]
    u = A._step_ge(spec, u, ef, 0.0, 256, ISINF, -1.0)          # - [E+FP>=256]
    u = A._clear(spec, u, ISNAN)
    u = A._step_ge(spec, u, ef, 0.0, 256, ISNAN, 1.0)           # [E+FP>=256] (E==255 & FP)
    return A._truncate(spec, u, dim)


def compile_fp_decode_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    f = L.FP32
    return [
        ("fp-dec-a-bits", _decode_operand_block(L, dim, L.STACK0, f.A_BIT)),
        ("fp-dec-b-bits", _decode_operand_block(L, dim, L.AX, f.B_BIT)),
        ("fp-dec-a-fields", _fields_block(L, dim, f.A_BIT, f.A_S, f.A_E, f.A_M, f.A_FP)),
        ("fp-dec-b-fields", _fields_block(L, dim, f.B_BIT, f.B_S, f.B_E, f.B_M, f.B_FP)),
        ("fp-dec-a-class", _class_block(L, dim, f.A_E, f.A_FP, f.A_ISZ, f.A_ISINF, f.A_ISNAN)),
        ("fp-dec-b-class", _class_block(L, dim, f.B_E, f.B_FP, f.B_ISZ, f.B_ISINF, f.B_ISNAN)),
    ]


# ===========================================================================
# Shared helpers: bits <-> nibbles, and a generic base-16 carry round on columns.
# ===========================================================================
def _bits_to_nibbles_block(L, dim, bit_base, nib_base, n_nib):
    """Group ``n_nib`` nibbles from a bit band: nib[c] = Σ_{p<4} 2^p bit(4c+p)."""
    _set_one(L)
    spec = _empty_spec(dim, n_nib * 5)
    u = 0
    for c in range(n_nib):
        u = A._clear(spec, u, nib_base + c)
        for p in range(4):
            u = A._ident(spec, u, {bit_base + 4 * c + p: float(1 << p)}, 0.0, nib_base + c, 1.0)
    return A._truncate(spec, u, dim)


def _nibbles_to_bits_block(L, dim, nib_base, bit_base, n_nib):
    """Explode ``n_nib`` nibbles (0..15) into 4*n_nib bits: bit(4c+p) = plane p of
    nib[c].  Reuses the bit-plane staircase (block input = the nibbles)."""
    _set_one(L)
    spec = _empty_spec(dim, n_nib * 4 * 40)
    u = 0
    for c in range(n_nib):
        u = _bit_extract_units(spec, u, L, nib_base + c, bit_base + 4 * c)
    return A._truncate(spec, u, dim)


# ===========================================================================
# F_MUL : significand multiply + normalise + round + encode.
# ===========================================================================
_MUL_PAIRS6 = [(i, j) for i in range(6) for j in range(6) if i + j < 12]


def _fp_mul_products_block(L, dim):
    """Raw nibble partial products PPM[idx] = AN[i]*BN[j] (<=225)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, len(_MUL_PAIRS6) * 3)
    u = 0
    for idx, (i, j) in enumerate(_MUL_PAIRS6):
        u = A._clear(spec, u, f.PPM + idx)
        u = A._mul_gate(spec, u, f.AN + i, f.BN + j, f.PPM + idx, 1.0)
    return A._truncate(spec, u, dim)


def _fp_mul_split_block(L, dim):
    """Split each PPM into low nibble (col i+j) + high nibble (col i+j+1)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 12 + len(_MUL_PAIRS6) * (1 + 15 * 2))
    u = 0
    for c in range(12):
        u = A._clear(spec, u, f.PCOL + c)
    for idx, (i, j) in enumerate(_MUL_PAIRS6):
        c = i + j
        pp = f.PPM + idx
        u = A._ident(spec, u, {pp: 1.0}, 0.0, f.PCOL + c, 1.0)
        if c + 1 < 12:
            u = A._floor_div_pow2(spec, u, {pp: 1.0}, 0.0, 16, 15,
                                  f.PCOL + c, -16.0, f.PCOL + c + 1, 1.0)
        else:
            u = A._floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, f.PCOL + c, -16.0)
    return A._truncate(spec, u, dim)


def _fp_mul_blocks(L, dim):
    """AN/BN significand nibbles -> 48 product bits in PBIT (via 12 carry-settled
    nibble columns).  Reuses the integer ALU's schoolbook multiply structure."""
    f = L.FP32
    blocks = [
        ("fp-mul-an", _bits_to_nibbles_block(L, dim, f.A_M, f.AN, 6)),
        ("fp-mul-bn", _bits_to_nibbles_block(L, dim, f.B_M, f.BN, 6)),
        ("fp-mul-products", _fp_mul_products_block(L, dim)),
        ("fp-mul-split", _fp_mul_split_block(L, dim)),
    ]
    # carry-settle the 12 columns (each < 256): ripple rounds.
    src, dst = f.PCOL, f.PCOL2
    for r in range(A._MUL_CARRY_ROUNDS):
        blocks.append((f"fp-mul-carry{r}", A._carry_round_block(L, dim, src, dst, 12)))
        src, dst = dst, src
    # explode the settled 12 nibbles into the 48 product bits.
    blocks.append(("fp-mul-pbits", _nibbles_to_bits_block(L, dim, src, f.PBIT, 12)))
    return blocks


def _fp_mul_normalize_block(L, dim):
    """From the 48 product bits PBIT (P = Am*Bm in [2^46, 2^48)) extract the
    normalised 24-bit significand into MROUND[0..23], the guard bit GRS[0], set
    R_S, R_E, and a scratch bit STICKY_HIP22 = hi AND P[22] (used by the next
    block's sticky).  hi = P[47] (1 iff P>=2^47).

    hi=1: product [2^47,2^48), sig = P[47:24], guard = P[23], sticky over P[0..22],
          result exp += 1.
    hi=0: product [2^46,2^47), sig = P[46:23], guard = P[22], sticky over P[0..21].
    Each significand/guard bit is a data-selected mux on hi (a _guard AND)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 24 * 4 + 40)
    u = 0
    hi = f.PBIT + 47
    for i in range(24):
        dst = f.MROUND + i
        u = A._clear(spec, u, dst)
        u = A._guard(spec, u, [(hi, 1.0, 0.0)], {f.PBIT + 24 + i: 1.0}, 0.0, dst, 1.0)
        u = A._guard(spec, u, [(hi, -1.0, 1.0)], {f.PBIT + 23 + i: 1.0}, 0.0, dst, 1.0)
    u = A._clear(spec, u, f.MROUND + 24)             # carry slot above significand
    # guard bit: hi ? P[23] : P[22]
    u = A._clear(spec, u, f.GRS + 0)
    u = A._guard(spec, u, [(hi, 1.0, 0.0)], {f.PBIT + 23: 1.0}, 0.0, f.GRS + 0, 1.0)
    u = A._guard(spec, u, [(hi, -1.0, 1.0)], {f.PBIT + 22: 1.0}, 0.0, f.GRS + 0, 1.0)
    # hi AND P[22]  (P[22] joins the sticky only in the hi branch).
    u = A._clear(spec, u, f.GRS + 2)                 # reuse GRS[2] as HIP22 scratch
    u = A._guard(spec, u, [(hi, 1.0, 0.0)], {f.PBIT + 22: 1.0}, 0.0, f.GRS + 2, 1.0)
    # result sign = A_S xor B_S
    u = A._clear(spec, u, f.R_S)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 1, f.R_S, 1.0)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 2, f.R_S, -1.0)
    # result biased exponent = A_E + B_E - 127 + hi.
    u = A._clear(spec, u, f.R_E)
    u = A._ident(spec, u, {f.A_E: 1.0, f.B_E: 1.0, hi: 1.0}, -127.0, f.R_E, 1.0)
    return A._truncate(spec, u, dim)


def _fp_mul_sticky_block(L, dim):
    """STICKY = [ Σ_{k<22} P[k] + HIP22 >= 1 ]  (HIP22 = GRS[2] = hi AND P[22],
    written by the normalize block; read here as the block input)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 8)
    u = 0
    terms = {f.PBIT + k: 1.0 for k in range(22)}
    terms[f.GRS + 2] = 1.0
    u = A._clear(spec, u, f.STICKY)
    u = A._step_ge(spec, u, terms, 0.0, 1, f.STICKY, 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# Shared ROUND + ENCODE (used by MUL/ADD/SUB/DIV): given the 24-bit significand
# MROUND[0..23] (MROUND[23] the leading 1), guard GRS[0], sticky STICKY, result
# sign R_S and biased exponent R_E, apply round-to-nearest-even, handle the
# rounding significand-overflow (all-ones -> 2^24 -> shift, exp+1), and assemble
# the 32 result bits into R_BITS, then the 8 result nibbles F_RES.
# ===========================================================================
def _fp_roundup_block(L, dim):
    """ROUNDUP = guard AND (sticky OR lsb).  A SEPARATE block from the ripple so the
    ripple reads the WRITTEN ROUNDUP (not the stale block input)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 8)
    u = 0
    # ROUNDUP = GRS AND (STICKY OR lsb).  As two DISJOINT guarded terms:
    #   t1 = GRS AND STICKY                (guard & sticky)
    #   t2 = GRS AND lsb AND NOT STICKY    (guard & lsb, when sticky is 0)
    # Both are exact ANDs via _guard windows (0/1 bands); their sum is 0/1.
    # NOT STICKY window: (STICKY, -1.0, 1.0) -> 1 - STICKY.
    u = A._clear(spec, u, f.ROUNDUP)
    u = A._guard(spec, u, [(f.GRS + 0, 1.0, 0.0), (f.STICKY, 1.0, 0.0)],
                 {L.ONE: 1.0}, 0.0, f.ROUNDUP, 1.0)
    u = A._guard(spec, u, [(f.GRS + 0, 1.0, 0.0), (f.MROUND + 0, 1.0, 0.0),
                           (f.STICKY, -1.0, 1.0)],
                 {L.ONE: 1.0}, 0.0, f.ROUNDUP, 1.0)
    return A._truncate(spec, u, dim)


def _fp_round_block(L, dim):
    """Increment the 24-bit significand MROUND[0..23] by ROUNDUP (written by the
    PRIOR :func:`_fp_roundup_block`) via a BIT-RIPPLE, writing the rounded bits into
    SIG[0..24] (SIG[24] = carry-out = significand overflow, set iff all 24 bits
    were 1 and ROUNDUP=1).

      carry_in[i] = ROUNDUP AND (MROUND[0..i-1] all 1) = ROUNDUP AND [Σ_{k<i} MROUND[k] >= i]
      new[i]      = MROUND[i] XOR carry_in[i]
      SIG[24]     = ROUNDUP AND [Σ_{k<24} MROUND[k] >= 24]
    All read the block INPUT (MROUND + the already-written ROUNDUP)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 24 * 4 + 8)
    u = 0
    # carry_in[i] = ROUNDUP AND MROUND[0] AND ... AND MROUND[i-1]  — an EXACT AND of
    # 0/1 bits via _guard (windows), no big-coefficient residue amplification.
    #   new[i] = MROUND[i] XOR carry_in[i] = MROUND[i] + carry_in[i] - 2*(MROUND[i] AND carry_in[i]).
    for i in range(24):
        dst = f.SIG + i
        u = A._clear(spec, u, dst)
        u = A._ident(spec, u, {f.MROUND + i: 1.0}, 0.0, dst, 1.0)                 # + MROUND[i]
        ci_wins = [(f.ROUNDUP, 1.0, 0.0)] + [(f.MROUND + k, 1.0, 0.0) for k in range(i)]
        u = A._guard(spec, u, ci_wins, {L.ONE: 1.0}, 0.0, dst, 1.0)              # + carry_in[i]
        # - 2*(MROUND[i] AND carry_in[i]) = -2 * AND(ci_wins + MROUND[i]).
        u = A._guard(spec, u, ci_wins + [(f.MROUND + i, 1.0, 0.0)],
                     {L.ONE: 1.0}, 0.0, dst, -2.0)
    # carry-out / significand overflow: SIG[24] = ROUNDUP AND (all 24 bits == 1).
    u = A._clear(spec, u, f.SIG + 24)
    co_wins = [(f.ROUNDUP, 1.0, 0.0)] + [(f.MROUND + k, 1.0, 0.0) for k in range(24)]
    u = A._guard(spec, u, co_wins, {L.ONE: 1.0}, 0.0, f.SIG + 24, 1.0)
    return A._truncate(spec, u, dim)


def _fp_encode_block(L, dim):
    """Assemble the 32 result bits R_BITS from the rounded significand SIG[0..24],
    result sign R_S, and biased exponent R_E:
      ovf = SIG[24] (significand overflowed to 2^24 on rounding) -> exponent += 1,
            and the stored fraction is 0 (2^24 has fraction 0).
      Efin = R_E + ovf.
      stored fraction bit i (i=0..22) = SIG[i]  when NOT ovf, else 0.
      bit31 = R_S ; bits 23..30 = the 8 bits of Efin ; bits 0..22 = fraction.
    This is the NORMAL finite path; special values are overridden downstream."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 300)
    u = 0
    ovf = f.SIG + 24
    # Efin = R_E + ovf.  ALL forms read the BLOCK INPUT (R_E, ovf, SIG) — never an
    # intermediate this block writes — so Efin is used as the linear form {R_E,ovf}.
    Efin_form = {f.R_E: 1.0, ovf: 1.0}
    # fraction bits 0..22 = SIG[i] AND NOT ovf.
    for i in range(23):
        dst = f.R_BITS + i
        u = A._clear(spec, u, dst)
        # SIG[i] - (SIG[i] AND ovf) = SIG[i]*(1-ovf).
        u = A._ident(spec, u, {f.SIG + i: 1.0}, 0.0, dst, 1.0)
        u = A._guard(spec, u, [(ovf, 1.0, 0.0)], {f.SIG + i: -1.0}, 0.0, dst, 1.0)
    # exponent nibbles: e_lo = Efin mod 16, e_hi = floor(Efin/16)  (Efin <= 256; a
    # rounded-finite Efin is <= 254, overflow-to-inf is handled downstream, so
    # e_hi <= 15).  Computed from the Efin FORM (block input).  Exploded to bits by
    # the NEXT block.
    u = A._clear(spec, u, f.PCOL2 + 0)
    u = A._ident(spec, u, Efin_form, 0.0, f.PCOL2 + 0, 1.0)
    u = A._floor_div_pow(spec, u, Efin_form, 0.0, 16, 16, f.PCOL2 + 0, -16.0)  # mod 16
    u = A._clear(spec, u, f.PCOL2 + 1)
    u = A._floor_div_pow(spec, u, Efin_form, 0.0, 16, 16, f.PCOL2 + 1, 1.0)    # floor/16
    # sign bit 31.
    u = A._clear(spec, u, f.R_BITS + 31)
    u = A._ident(spec, u, {f.R_S: 1.0}, 0.0, f.R_BITS + 31, 1.0)
    return A._truncate(spec, u, dim)


def _fp_encode_exp_block(L, dim):
    """Explode the exponent nibbles (PCOL2[0]=Efin mod 16, PCOL2[1]=Efin//16) into
    result bits 23..30 (a SEPARATE block, since it reads the nibbles the encode
    block wrote)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 4 * 40 * 2)
    u = 0
    u = _bit_extract_units(spec, u, L, f.PCOL2 + 0, f.R_BITS + 23)
    u = _bit_extract_units(spec, u, L, f.PCOL2 + 1, f.R_BITS + 27)
    return A._truncate(spec, u, dim)


def _fp_result_nibbles_block(L, dim):
    """Pack the 32 result bits R_BITS into the 8 result nibbles F_RES the ax-mux
    copies into AX: F_RES[c] = Σ_{p<4} 2^p R_BITS[4c+p]."""
    return _bits_to_nibbles_block(L, dim, L.FP32.R_BITS, L.FP32.F_RES, 8)


# ===========================================================================
# MUL special-case override.  After the finite path assembled F_RES (correct for
# in-range finite products), FORCE the IEEE special result when the operands or
# the product exponent hit a special case.  All conditions read the decoded class
# flags + the biased result exponent R_E (from the normalize block).
#
#   nan_in   = A_ISNAN OR B_ISNAN                             -> qNaN
#   inf0     = (A_ISINF AND B_ISZ) OR (A_ISZ AND B_ISINF)     -> qNaN  (inf*0)
#   inf_res  = (A_ISINF OR B_ISINF) AND NOT inf0 AND NOT nan  -> signed inf
#   zero_res = (A_ISZ OR B_ISZ) AND NOT inf AND NOT nan       -> signed 0
#   ovf      = finite AND R_E (+ovf-bump) >= 255              -> signed inf
#   udf      = finite AND R_E <= 0                            -> signed 0 (flush;
#             gradual-underflow subnormals are the documented GAP)
# The forced result overwrites F_RES via a SET on the 8 nibbles, each gated on the
# matching condition.  Sign = R_S (already A_S xor B_S).  qNaN = 0x7FC00000.
# ===========================================================================
def _fp_mul_special_flags_block(L, dim):
    """Compute the MUL special-case selector flags into scratch scalars (read by
    the override block).  Uses GRS[1] area + spare scalars; each flag 0/1."""
    _set_one(L)
    f = L.FP32
    # scratch scalars for the flags (reuse WIDE band head as flag slots).
    NANF, INF0, INFR, ZR, OVF, UDF = (f.FLAGS + 0, f.FLAGS + 1, f.FLAGS + 2,
                                      f.FLAGS + 3, f.FLAGS + 4, f.FLAGS + 5)
    spec = _empty_spec(dim, 60)
    u = 0
    # nan_in = A_ISNAN OR B_ISNAN = [A_ISNAN + B_ISNAN >= 1].
    u = A._clear(spec, u, NANF)
    u = A._step_ge(spec, u, {f.A_ISNAN: 1.0, f.B_ISNAN: 1.0}, 0.0, 1, NANF, 1.0)
    # inf0 = (A_ISINF AND B_ISZ) OR (A_ISZ AND B_ISINF).
    u = A._clear(spec, u, INF0)
    u = A._guard(spec, u, [(f.A_ISINF, 1.0, 0.0), (f.B_ISZ, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, INF0, 1.0)
    u = A._guard(spec, u, [(f.A_ISZ, 1.0, 0.0), (f.B_ISINF, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, INF0, 1.0)
    # inf_res = (A_ISINF OR B_ISINF) AND NOT inf0 AND NOT nan.
    #   base = [A_ISINF + B_ISINF >= 1] ; then subtract if inf0 or nan (they win).
    u = A._clear(spec, u, INFR)
    u = A._step_ge(spec, u, {f.A_ISINF: 1.0, f.B_ISINF: 1.0}, 0.0, 1, INFR, 1.0)
    u = A._guard(spec, u, [(INF0, 1.0, 0.0)], {INFR: -1.0}, 0.0, INFR, 1.0)   # -= INF0*INFR
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {INFR: -1.0}, 0.0, INFR, 1.0)   # -= NAN*INFR
    # zero_res = (A_ISZ OR B_ISZ) AND NOT inf0 AND NOT nan  (inf handled by inf_res).
    u = A._clear(spec, u, ZR)
    u = A._step_ge(spec, u, {f.A_ISZ: 1.0, f.B_ISZ: 1.0}, 0.0, 1, ZR, 1.0)
    u = A._guard(spec, u, [(INF0, 1.0, 0.0)], {ZR: -1.0}, 0.0, ZR, 1.0)
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {ZR: -1.0}, 0.0, ZR, 1.0)
    # finite = NOT (nan OR any-inf OR any-zero).  Overflow/underflow only apply to
    # finite non-zero operands.  finite_op = NOT A_ISNAN,B_ISNAN,A_ISINF,B_ISINF,
    # A_ISZ,B_ISZ.
    fin_terms = {f.A_ISNAN: 1.0, f.B_ISNAN: 1.0, f.A_ISINF: 1.0, f.B_ISINF: 1.0,
                 f.A_ISZ: 1.0, f.B_ISZ: 1.0}
    # ovf = finite AND [R_E + SIG24 >= 255].  SIG24 is the rounding exp bump.
    u = A._clear(spec, u, OVF)
    u = A._step_ge(spec, u, {f.R_E: 1.0, f.SIG + 24: 1.0}, 0.0, 255, OVF, 1.0)   # [Efin>=255]
    # gate out non-finite: subtract OVF when any special-operand flag set.
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {OVF: -1.0}, 0.0, OVF, 1.0)
    # (inf/zero operands already produce inf/zero via INFR/ZR; their OVF is moot but
    #  we zero it to avoid double-forcing.)
    for fl in (f.A_ISINF, f.B_ISINF, f.A_ISZ, f.B_ISZ):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {OVF: -1.0}, 0.0, OVF, 1.0)
    # udf = finite AND [Efin <= 0] = [Efin < 1] = 1 - [Efin >= 1], gated finite.
    #   Efin can be negative (R_E as computed can be < 0).  Use [R_E + SIG24 <= 0].
    u = A._clear(spec, u, UDF)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, UDF, 1.0)
    u = A._step_ge(spec, u, {f.R_E: 1.0, f.SIG + 24: 1.0}, 0.0, 1, UDF, -1.0)    # 1-[Efin>=1]
    for fl in (f.A_ISNAN, f.B_ISNAN, f.A_ISINF, f.B_ISINF, f.A_ISZ, f.B_ISZ):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {UDF: -1.0}, 0.0, UDF, 1.0)
    return A._truncate(spec, u, dim)


def _fp_special_combine_block(L, dim):
    """Combine the base special flags (NANF,INF0,INFR,ZR,OVF,UDF at WIDE+0..5) into
    the class selectors the override consumes.  A SEPARATE block so it reads the
    WRITTEN base flags (not the stale block input):
      QF (qNaN) = NANF OR INF0 ; IF_ (inf) = INFR OR OVF ; ZF (zero) = ZR OR UDF ;
      ANYF = QF OR IF_ OR ZF."""
    _set_one(L)
    f = L.FP32
    NANF, INF0, INFR, ZR, OVF, UDF = (f.FLAGS + 0, f.FLAGS + 1, f.FLAGS + 2,
                                      f.FLAGS + 3, f.FLAGS + 4, f.FLAGS + 5)
    ANYF, QF, IF_, ZF = f.FLAGS + 6, f.FLAGS + 7, f.FLAGS + 8, f.FLAGS + 9
    # PRIORITY qNaN > inf > zero (inf*0 sets both INF0 and ZR, but the IEEE result is
    # qNaN, so qNaN dominates).  All selectors derived from the BASE flags (block
    # inputs) so no same-block staleness:
    #   QF  = NANF OR INF0
    #   IF_ = (INFR OR OVF) AND NOT(NANF OR INF0)
    #   ZF  = (ZR OR UDF) AND NOT(NANF OR INF0) AND NOT(INFR OR OVF)
    # A step over a combined form isolates each: e.g. [x>=1] - [x + hi_prio >= 1]
    # would need hi as a BIG term.  Cleaner: realise NOT via a big-coefficient
    # exclusion: IF_ = [ (INFR+OVF) - BIG*(NANF+INF0) >= 1 ] (drops to <=0 when a
    # higher class fires).
    BIG = 100.0
    spec = _empty_spec(dim, 12)
    u = 0
    u = A._clear(spec, u, QF)
    u = A._step_ge(spec, u, {NANF: 1.0, INF0: 1.0}, 0.0, 1, QF, 1.0)
    u = A._clear(spec, u, IF_)
    u = A._step_ge(spec, u, {INFR: 1.0, OVF: 1.0, NANF: -BIG, INF0: -BIG}, 0.0, 1, IF_, 1.0)
    u = A._clear(spec, u, ZF)
    u = A._step_ge(spec, u, {ZR: 1.0, UDF: 1.0, NANF: -BIG, INF0: -BIG,
                             INFR: -BIG, OVF: -BIG}, 0.0, 1, ZF, 1.0)
    u = A._clear(spec, u, ANYF)
    u = A._step_ge(spec, u, {NANF: 1.0, INF0: 1.0, INFR: 1.0, ZR: 1.0, OVF: 1.0, UDF: 1.0},
                   0.0, 1, ANYF, 1.0)
    return A._truncate(spec, u, dim)


def _fp_special_override_block(L, dim):
    """Override F_RES with the IEEE special result when a special class flag is set
    (flags from the prior flags block, read as block input):
      QF (qNaN) -> F_RES = nibbles(0x7FC00000)   [sign-agnostic canonical qNaN]
      IF_ (inf) -> F_RES = R_S<<31 | 0x7F800000  (signed inf)
      ZF (zero) -> F_RES = R_S<<31 | 0           (signed zero)
    ANYF gates the clear of the finite F_RES; each class then adds its nibbles.
    Non-special step: ANYF=0 -> F_RES untouched (the finite path result stands)."""
    _set_one(L)
    f = L.FP32
    from .blogspec_vocab import nibbles_of_value
    ANYF, QF, IF_, ZF = f.FLAGS + 6, f.FLAGS + 7, f.FLAGS + 8, f.FLAGS + 9
    qnan = nibbles_of_value(0x7FC00000, 8)
    inf = nibbles_of_value(0x7F800000, 8)
    spec = _empty_spec(dim, 8 * 6 + 8)
    u = 0
    for c in range(8):
        # clear the finite F_RES[c] gated on ANY special.
        u = A._guard(spec, u, [(ANYF, 1.0, 0.0)], {f.F_RES + c: -1.0}, 0.0, f.F_RES + c, 1.0)
        # qNaN nibble (sign-agnostic).
        if qnan[c]:
            u = A._guard(spec, u, [(QF, 1.0, 0.0)], {L.ONE: float(qnan[c])}, 0.0, f.F_RES + c, 1.0)
        # inf nibbles (exp field); sign nibble (c==7 top bit) added below.
        if inf[c]:
            u = A._guard(spec, u, [(IF_, 1.0, 0.0)], {L.ONE: float(inf[c])}, 0.0, f.F_RES + c, 1.0)
    # sign bit (nibble 7, bit 3 = value 8) for inf and zero: add 8 to F_RES[7] gated
    # on (IF_ or ZF) AND R_S.
    u = A._guard(spec, u, [(IF_, 1.0, 0.0), (f.R_S, 1.0, 0.0)], {L.ONE: 8.0}, 0.0, f.F_RES + 7, 1.0)
    u = A._guard(spec, u, [(ZF, 1.0, 0.0), (f.R_S, 1.0, 0.0)], {L.ONE: 8.0}, 0.0, f.F_RES + 7, 1.0)
    return A._truncate(spec, u, dim)


# ===========================================================================
# Shared finite normalise+round+encode tail (MUL/ADD/SUB/DIV all feed it).
# ===========================================================================
def _fp_round_encode_tail(L, dim):
    return [
        ("fp-roundup", _fp_roundup_block(L, dim)),
        ("fp-round", _fp_round_block(L, dim)),
        ("fp-encode", _fp_encode_block(L, dim)),
        ("fp-encode-exp", _fp_encode_exp_block(L, dim)),
        ("fp-resnib", _fp_result_nibbles_block(L, dim)),
    ]


def compile_fp_mul_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Full F_MUL block stack: decode | significand multiply | normalise | round |
    encode | special-case override.  Result nibbles land in F_RES (the ax-mux copies
    them into AX on OP_IS[F_MUL])."""
    blocks = list(compile_fp_decode_blocks(L, dim))
    blocks += _fp_mul_blocks(L, dim)
    blocks.append(("fp-mul-norm", _fp_mul_normalize_block(L, dim)))
    blocks.append(("fp-mul-sticky", _fp_mul_sticky_block(L, dim)))
    blocks += _fp_round_encode_tail(L, dim)
    blocks.append(("fp-mul-special", _fp_mul_special_flags_block(L, dim)))
    blocks.append(("fp-special-combine", _fp_special_combine_block(L, dim)))
    blocks.append(("fp-mul-override", _fp_special_override_block(L, dim)))
    return blocks


# ===========================================================================
# F_ADD / F_SUB : sign-magnitude align + add/sub + renormalise.
#
# Anchor layout (bit index within the WIDE band, 0 = LSB):
#   The larger operand's 24-bit significand sits at bits [26..3] (i.e. <<3), so 3
#   low bits (2,1,0) are round material below the significand LSB.  The smaller
#   operand's significand is shifted RIGHT by the exponent difference d, with all
#   bits that fall below bit 0 OR'd into a sticky.  ADD keeps the sum in [0, 2^28);
#   SUB (signs differ) yields a difference in [0, 2^27).  A leading-one search then
#   renormalises so the leading 1 is at bit 26 and sets the guard(2)/round(1)/
#   sticky(0) for the shared round tail.
# ===========================================================================
_ADD_ANCHOR = 3        # significand LSB sits at bit 3 (bits 0,1,2 are GRS room)
_ADD_TOP = 27          # WIDE band bits 0..27 (24-bit sig <<3 tops at bit 26, +carry 27)


def _fp_add_prep_block(L, dim):
    """Decide effective-subtraction ESUB (signs differ, with F_SUB flipping B's sign
    upstream), the swap flag ASWAP (B magnitude > A magnitude), the shift distance
    SHD = |A_E - B_E| adjusted, and the result sign R_S.  Magnitude compare: (E,sig)
    lexicographic — compare exponents, tie-break on the 24-bit significand value."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 80)
    u = 0
    # ESUB = A_S xor B_S.
    u = A._clear(spec, u, f.ESUB)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 1, f.ESUB, 1.0)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 2, f.ESUB, -1.0)
    # magnitude(a) vs magnitude(b): compare E first; on tie compare significand VALUE.
    # bigform = 2^24 * E + sig  (E<=255 so 2^24*E up to ~4.3e9 > 2^24 -> NOT fp32
    # exact!).  Instead compare via two staircases: EBIG = [A_E>B_E], ELT=[A_E<B_E],
    # then significand compare only used on E-tie.
    # magnitude-compare scalars (clean 0/1, fp32-exact; combined in the NEXT block):
    #   EGT_B = [B_E>A_E] ; EEQ = [A_E==B_E] ; SGT_B = [Bsig>Asig].
    # The significands are 24-bit, so a single ``Bsig-Asig`` diff reaches 2^24 where
    # the _step_ge ramp loses fp32 precision.  Compare on a HIGH-12 / LOW-12 split
    # (each diff <= 4095, safely fp32-exact through the ramp):
    #   SGT_B = [Bhi>Ahi] OR ([Bhi==Ahi] AND [Blo>Alo]).
    Ahi = {f.A_M + 12 + i: float(1 << i) for i in range(12)}
    Bhi = {f.B_M + 12 + i: float(1 << i) for i in range(12)}
    Alo = {f.A_M + i: float(1 << i) for i in range(12)}
    Blo = {f.B_M + i: float(1 << i) for i in range(12)}
    dhi = dict(Bhi)
    for k, v in Ahi.items():
        dhi[k] = dhi.get(k, 0.0) - v            # Bhi - Ahi  (in [-4095, 4095])
    dlo = dict(Blo)
    for k, v in Alo.items():
        dlo[k] = dlo.get(k, 0.0) - v            # Blo - Alo
    EGT_B, EEQ, SGT_B = f.LZ + 0, f.LZ + 1, f.LZ + 2   # scratch scalars in LZ band
    HGT, HEQ, LGT = f.LZ + 3, f.LZ + 4, f.LZ + 5       # significand hi/lo compare
    u = A._clear(spec, u, EGT_B)
    u = A._step_ge(spec, u, {f.B_E: 1.0, f.A_E: -1.0}, 0.0, 1, EGT_B, 1.0)         # [B_E>A_E]
    u = A._clear(spec, u, EEQ)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, EEQ, 1.0)
    u = A._step_ge(spec, u, {f.A_E: 1.0, f.B_E: -1.0}, 0.0, 1, EEQ, -1.0)          # -[A_E>B_E]
    u = A._step_ge(spec, u, {f.B_E: 1.0, f.A_E: -1.0}, 0.0, 1, EEQ, -1.0)          # -[B_E>A_E]
    # hi/lo significand compares (each bounded diff).
    u = A._clear(spec, u, HGT)
    u = A._step_ge(spec, u, dhi, 0.0, 1, HGT, 1.0)                                 # [Bhi>Ahi]
    u = A._clear(spec, u, HEQ)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, HEQ, 1.0)
    u = A._step_ge(spec, u, dhi, 0.0, 1, HEQ, -1.0)                                # -[Bhi>Ahi]
    u = A._step_ge(spec, u, {k: -v for k, v in dhi.items()}, 0.0, 1, HEQ, -1.0)    # -[Ahi>Bhi]
    u = A._clear(spec, u, LGT)
    u = A._step_ge(spec, u, dlo, 0.0, 1, LGT, 1.0)                                 # [Blo>Alo]
    # SGT_B combined in the swap block (needs HGT/HEQ/LGT written -> block input there).
    u = A._clear(spec, u, SGT_B)                    # placeholder cleared; set in swap block
    # SHD = |A_E - B_E|  = (A_E-B_E) if A>=B else (B_E-A_E).
    u = A._clear(spec, u, f.SHD)
    u = A._relu(spec, u, {f.A_E: 1.0, f.B_E: -1.0}, 0.0, f.SHD, 1.0)   # relu(A_E-B_E)
    u = A._relu(spec, u, {f.B_E: 1.0, f.A_E: -1.0}, 0.0, f.SHD, 1.0)   # relu(B_E-A_E)
    return A._truncate(spec, u, dim)


def _fp_add_swap_block(L, dim):
    """SGT_B = HGT OR (HEQ AND LGT) ; ASWAP = EGT_B OR (EEQ AND SGT_B) ;
    R_S = ASWAP ? B_S : A_S.  Reads the WRITTEN compare scalars (LZ+0..5).  SGT_B and
    ASWAP compose in ONE block: ASWAP = EGT_B OR (EEQ AND (HGT OR (HEQ AND LGT))).
    Expand to guarded ANDs (all 0/1, exact)."""
    _set_one(L)
    f = L.FP32
    EGT_B, EEQ = f.LZ + 0, f.LZ + 1
    HGT, HEQ, LGT = f.LZ + 3, f.LZ + 4, f.LZ + 5
    spec = _empty_spec(dim, 24)
    u = 0
    u = A._clear(spec, u, f.ASWAP)
    u = A._ident(spec, u, {EGT_B: 1.0}, 0.0, f.ASWAP, 1.0)                          # + EGT_B
    # + EEQ AND HGT
    u = A._guard(spec, u, [(EEQ, 1.0, 0.0), (HGT, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, f.ASWAP, 1.0)
    # + EEQ AND HEQ AND LGT
    u = A._guard(spec, u, [(EEQ, 1.0, 0.0), (HEQ, 1.0, 0.0), (LGT, 1.0, 0.0)],
                 {L.ONE: 1.0}, 0.0, f.ASWAP, 1.0)
    return A._truncate(spec, u, dim)


def _fp_add_rsign_block(L, dim):
    """R_S = ASWAP ? B_S : A_S  (reads the written ASWAP)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 8)
    u = 0
    u = A._clear(spec, u, f.R_S)
    u = A._guard(spec, u, [(f.ASWAP, -1.0, 1.0)], {f.A_S: 1.0}, 0.0, f.R_S, 1.0)    # (1-ASWAP)*A_S
    u = A._guard(spec, u, [(f.ASWAP, 1.0, 0.0)], {f.B_S: 1.0}, 0.0, f.R_S, 1.0)     # ASWAP*B_S
    return A._truncate(spec, u, dim)


# Alignment anchor: the LARGER significand's LSB (bit 0 of its 24 bits) sits at
# WIDE bit ``_ADD_ANCHOR`` (=3); bits 0,1,2 are round material.  The 24-bit sig
# thus spans WIDE bits 3..26.  A right shift of the smaller by d moves its bit p to
# WIDE bit ``3 + p - d``.  Shifts of d >= 27 leave only sticky.
_SHD_MAX = 32


def _fp_add_shd_oh_block(L, dim):
    """One-hot SHD: SHD_OH[s] = [SHD == s] for s=0..31 (SHD is the exponent diff,
    clamped: for s==31 the one-hot also catches SHD>=31, so any larger shift routes
    through the s=31 slot which shifts everything to sticky)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 32 * 5)
    u = 0
    for s in range(32):
        oh = f.SHD_OH + s
        u = A._clear(spec, u, oh)
        u = A._step_ge(spec, u, {f.SHD: 1.0}, 0.0, s, oh, 1.0)
        if s < 31:
            u = A._step_ge(spec, u, {f.SHD: 1.0}, 0.0, s + 1, oh, -1.0)
        # s==31 slot = [SHD>=31] (catches all larger shifts).
    return A._truncate(spec, u, dim)


def _fp_add_align_block(L, dim):
    """Build the aligned significand bit-vectors WIDE (larger, fixed at anchor) and
    WIDE2 (smaller, right-shifted by SHD), plus STICKY = OR of the smaller's bits
    that shifted below WIDE bit 0.

    larger_M[i]  = ASWAP ? B_M[i] : A_M[i]     -> WIDE[i+3]
    smaller_M[i] = ASWAP ? A_M[i] : B_M[i]     -> WIDE2[i+3-SHD]  (dropped if <0)
    Uses the one-hot SHD to gather each WIDE2 destination bit from the correct
    source, and accumulates the dropped bits into STICKY.  All 0/1, exact."""
    _set_one(L)
    f = L.FP32
    an = _ADD_ANCHOR
    NW = 52                                        # WIDE band bit width used
    spec = _empty_spec(dim, NW * 4 + 24 * 34 + 24 * 2 + 8)
    u = 0
    # larger significand into WIDE at anchor.
    for w in range(NW):
        u = A._clear(spec, u, f.WIDE + w)
        u = A._clear(spec, u, f.WIDE2 + w)
    for i in range(24):
        # larger bit i -> WIDE[i+an]
        dst = f.WIDE + i + an
        u = A._guard(spec, u, [(f.ASWAP, -1.0, 1.0)], {f.A_M + i: 1.0}, 0.0, dst, 1.0)  # A when !swap
        u = A._guard(spec, u, [(f.ASWAP, 1.0, 0.0)], {f.B_M + i: 1.0}, 0.0, dst, 1.0)   # B when swap
    # smaller significand, right-shifted by SHD.  smaller bit i lands at WIDE2[i+an-SHD].
    # For each destination position w and each shift s (one-hot), the source is bit
    # (w - an + s).  We gather: WIDE2[w] = Σ_s SHD_OH[s] * smaller_bit(w-an+s).
    for w in range(NW):
        dst = f.WIDE2 + w
        for s in range(32):
            src_i = w - an + s
            if 0 <= src_i < 24:
                # smaller_bit = ASWAP ? A_M[src_i] : B_M[src_i].
                u = A._guard(spec, u, [(f.SHD_OH + s, 1.0, 0.0), (f.ASWAP, -1.0, 1.0)],
                             {f.B_M + src_i: 1.0}, 0.0, dst, 1.0)     # B is smaller when !swap
                u = A._guard(spec, u, [(f.SHD_OH + s, 1.0, 0.0), (f.ASWAP, 1.0, 0.0)],
                             {f.A_M + src_i: 1.0}, 0.0, dst, 1.0)     # A is smaller when swap
    return A._truncate(spec, u, dim)


def _fp_add_sticky_block(L, dim):
    """STICKY = OR of the smaller significand's bits that shifted BELOW WIDE bit 0
    (i.e. smaller bit i with i + anchor - SHD < 0 <=> SHD > i + anchor).  Gathered
    via the one-hot SHD: for each (i, s) with i + anchor - s < 0, the smaller bit i
    is dropped.  STICKY = [ Σ dropped smaller bits >= 1 ].  Reuses STICKY scalar."""
    _set_one(L)
    f = L.FP32
    an = _ADD_ANCHOR
    # collect the (i, s) pairs where bit i is dropped at shift s.  Build a sum of the
    # smaller bits weighted by whether they are dropped at the ACTIVE shift.
    spec = _empty_spec(dim, 24 * 34 + 8)
    u = 0
    u = A._clear(spec, u, f.STICKY)
    # dropped_contrib = Σ_{i,s: i+an-s<0} SHD_OH[s] * smaller_bit(i).  A single
    # threshold over that sum >= 1 gives the OR.  But a _step_ge over a guarded sum
    # isn't one unit; accumulate the guarded ANDs into STICKY (could exceed 1) then a
    # SEPARATE OR-threshold.  Since STICKY only needs to be 0/1, accumulate into a
    # count scratch (LZ+6) then threshold in the add block.
    cnt = f.LZ + 6
    u = A._clear(spec, u, cnt)
    for i in range(24):
        for s in range(32):
            if i + an - s < 0:      # smaller bit i is dropped at shift s
                u = A._guard(spec, u, [(f.SHD_OH + s, 1.0, 0.0), (f.ASWAP, -1.0, 1.0)],
                             {f.B_M + i: 1.0}, 0.0, cnt, 1.0)   # B smaller when !swap
                u = A._guard(spec, u, [(f.SHD_OH + s, 1.0, 0.0), (f.ASWAP, 1.0, 0.0)],
                             {f.A_M + i: 1.0}, 0.0, cnt, 1.0)   # A smaller when swap
    return A._truncate(spec, u, dim)


def _fp_add_astk_block(L, dim):
    """ASTK (0/1) = [ align-sticky count (LZ+6) >= 1 ] — a clean flag the sum block
    reads to decide the two's-complement +1 (dropped-bits borrow).  SEPARATE block so
    the sum reads the WRITTEN 0/1 flag, not the raw count."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 4)
    u = 0
    u = A._clear(spec, u, f.LZ + 7)
    u = A._step_ge(spec, u, {f.LZ + 6: 1.0}, 0.0, 1, f.LZ + 7, 1.0)
    return A._truncate(spec, u, dim)


# The aligned significands span WIDE bits 0..30 (24-bit sig <<3 tops at bit 26; the
# smaller never exceeds the larger).  Represent each as TWO fp32-exact limbs:
#   LO = bits 0..15 (<= 2^16-1) ; HI = bits 16..30 (<= 2^15-1).  Sum/diff of the two
# aligned significands fits: sum < 2^28, diff >= 0.  We carry between limbs at 2^16.
def _wide_limbs(f, base):
    lo = {base + i: float(1 << i) for i in range(16)}
    hi = {base + 16 + i: float(1 << i) for i in range(16)}   # bits 16..31
    return lo, hi


_ADD_NIB = 8            # 8 nibbles cover the 31-bit aligned sum


def _fp_add_sumnib_block(L, dim):
    """Sum the aligned significands as nibble columns.  Convert WIDE / WIDE2 bit
    vectors to 8 nibbles each; ADD (ESUB=0) forms ``WIDE + WIDE2`` per column;
    SUB (ESUB=1) forms ``WIDE + (~WIDE2 & mask) + 1`` (two's complement over 32
    bits) so the non-negative difference appears in the low 8 nibbles.  Columns
    (<= 15+15+1 raw) go to SUMB nibble band; carry-settled by the next blocks."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 8 * 30)
    u = 0
    for c in range(_ADD_NIB):
        col = f.SUMB + c
        u = A._clear(spec, u, col)
        for p in range(4):
            wbit = f.WIDE + 4 * c + p
            sbit = f.WIDE2 + 4 * c + p
            # + WIDE nibble bit p  (weight 2^p)
            u = A._ident(spec, u, {wbit: float(1 << p)}, 0.0, col, 1.0)
            # ADD: + WIDE2 bit ; SUB: + (1 - WIDE2 bit)  (ones' complement)
            u = A._guard(spec, u, [(f.ESUB, -1.0, 1.0)], {sbit: float(1 << p)}, 0.0, col, 1.0)   # !ESUB * sbit
            u = A._guard(spec, u, [(f.ESUB, 1.0, 0.0)], {sbit: -float(1 << p), L.ONE: float(1 << p)}, 0.0, col, 1.0)  # ESUB*(1-sbit)*2^p
        if c == 0:
            # SUB two's-complement +1 into the lowest column — but ONLY when the
            # aligned smaller had NO dropped low bits (align-sticky == 0).  When bits
            # were dropped in alignment, the truncated WIDE2 undercounts the true
            # smaller by a fraction of a ULP, so the +1 is absorbed by the implicit
            # borrow (``L - (S + frac)`` = ``L + ~S`` with no +1), and the dropped
            # bits become the round sticky.  align_sticky = [LZ+6 >= 1].
            u = A._guard(spec, u, [(f.ESUB, 1.0, 0.0), (f.LZ + 7, -1.0, 1.0)],
                         {L.ONE: 1.0}, 0.0, col, 1.0)   # +1 only if !ASTK (LZ+7)
    return A._truncate(spec, u, dim)


def _fp_add_sum_settle_blocks(L, dim):
    """Carry-settle the 8 SUMB nibble columns (raw <= 31) into clean nibbles, then
    explode to the 32 sum bits (SUMB2 band = WIDE2 reused after align is consumed —
    no: use PBIT which ADD does not use).  Uses the shared carry-round + bit-explode."""
    f = L.FP32
    blocks = []
    src, dst = f.SUMB, f.SUMB + 8       # double-buffer within the SUMB band region
    # SUMB band is SUMB..SUMB+55; columns at 0..7, buffer at 8..15.  A two's-complement
    # subtract can produce an all-0xF chain (e.g. 1.0-1.0 -> ~S = 0xFB..) whose carry
    # ripples through ALL 8 nibbles, so 9 rounds (>= 8 + headroom) fully settle it.
    for r in range(9):
        blocks.append((f"fp-add-sumc{r}", A._carry_round_block(L, dim, src, dst, 8)))
        src, dst = dst, src
    # explode the settled 8 nibbles into the 32 sum bits at PBIT.
    blocks.append(("fp-add-sumbits", _nibbles_to_bits_block(L, dim, src, f.PBIT, 8)))
    return blocks


def _fp_add_lead_block(L, dim):
    """Leading-one position LZ of the 32-bit sum (bits at PBIT).  LZ = the highest
    set bit index (0..31); LZ = Σ_{k=0..31} [ any bit >= k is set ] - 1... simpler:
    LZ_ge[k] = [ Σ_{i>=k} PBIT[i] >= 1 ] (a 1 iff some bit at or above k is set);
    the leading position = Σ_k LZ_ge[k] - 1.  We store LZ (the leading index) as a
    scalar."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 80)
    u = 0
    u = A._clear(spec, u, f.LZ + 8)          # LZ scalar (avoid the +0..6 compare scratch)
    for k in range(32):
        hi_terms = {f.PBIT + i: 1.0 for i in range(k, 32)}
        u = A._step_ge(spec, u, hi_terms, 0.0, 1, f.LZ + 8, 1.0)   # + [any bit >= k]
    # LZ counts the number of k for which a bit >= k exists = leading_index + 1.
    u = A._ident(spec, u, {L.ONE: -1.0}, 0.0, f.LZ + 8, 1.0)       # - 1
    return A._truncate(spec, u, dim)


def _fp_add_lead_oh_block(L, dim):
    """One-hot of the leading index LZ (0..31) into LZ+16..LZ+47: LZ_OH[p]=[LZ==p]."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 32 * 5)
    u = 0
    for p in range(32):
        oh = f.LZ + 16 + p
        u = A._clear(spec, u, oh)
        u = A._step_ge(spec, u, {f.LZ + 8: 1.0}, 0.0, p, oh, 1.0)
        u = A._step_ge(spec, u, {f.LZ + 8: 1.0}, 0.0, p + 1, oh, -1.0)
    return A._truncate(spec, u, dim)


def _fp_add_normalize_block(L, dim):
    """Normalise the sum: extract the 24-bit significand (bits LZ..LZ-23) into
    MROUND[0..23], guard = bit LZ-24 into GRS[0], sticky_low = OR(bits < LZ-24) via
    a count into GRS[2], and R_E = E_large + (LZ - 26).  Uses the LZ one-hot to
    gather each significand bit from PBIT[LZ-23+j].  (LZ >= 0; if the whole sum is 0
    the result is signed zero, handled by the special override.)"""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 24 * 34 + 34 + 40 + 24 * 34)
    u = 0
    # significand bit j (j=0..23, j=23 the leading) = PBIT[ LZ - 23 + j ].
    for j in range(24):
        dst = f.MROUND + j
        u = A._clear(spec, u, dst)
        for p in range(32):
            src = p - 23 + j            # when LZ==p, source bit index
            if 0 <= src < 32:
                u = A._guard(spec, u, [(f.LZ + 16 + p, 1.0, 0.0)], {f.PBIT + src: 1.0}, 0.0, dst, 1.0)
    # guard = PBIT[LZ-24].
    u = A._clear(spec, u, f.GRS + 0)
    for p in range(32):
        src = p - 24
        if 0 <= src < 32:
            u = A._guard(spec, u, [(f.LZ + 16 + p, 1.0, 0.0)], {f.PBIT + src: 1.0}, 0.0, f.GRS + 0, 1.0)
    # sticky_low count = Σ PBIT[i] for i < LZ-24, gated on the one-hot.  Accumulate.
    u = A._clear(spec, u, f.GRS + 2)
    for p in range(32):
        for i in range(p - 24):
            if 0 <= i < 32:
                u = A._guard(spec, u, [(f.LZ + 16 + p, 1.0, 0.0)], {f.PBIT + i: 1.0}, 0.0, f.GRS + 2, 1.0)
    # R_E = E_large + LZ - 26.  E_large = ASWAP ? B_E : A_E.
    u = A._clear(spec, u, f.R_E)
    u = A._guard(spec, u, [(f.ASWAP, -1.0, 1.0)], {f.A_E: 1.0}, 0.0, f.R_E, 1.0)   # (1-ASWAP)*A_E
    u = A._guard(spec, u, [(f.ASWAP, 1.0, 0.0)], {f.B_E: 1.0}, 0.0, f.R_E, 1.0)    # ASWAP*B_E
    u = A._ident(spec, u, {f.LZ + 8: 1.0, L.ONE: -26.0}, 0.0, f.R_E, 1.0)          # + LZ - 26
    return A._truncate(spec, u, dim)


def _fp_add_sticky_final_block(L, dim):
    """STICKY = [ align-sticky-count (LZ+6) + normalize-sticky-count (GRS[2]) >= 1 ].
    The align sticky (bits dropped in alignment) and the normalize sticky (bits
    below the guard after renorm) together form the round sticky."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 6)
    u = 0
    u = A._clear(spec, u, f.STICKY)
    u = A._step_ge(spec, u, {f.LZ + 6: 1.0, f.GRS + 2: 1.0}, 0.0, 1, f.STICKY, 1.0)
    return A._truncate(spec, u, dim)


def _step_ge_gated(spec, u, L, windows, terms, thr, dst, scale):
    """dst += scale * AND(windows) * [terms >= thr].  windows are 0/1 (band,coeff,
    const); the step is realised on a combined form with a MODERATE big-coefficient
    (not 1e9) so fp32 stays exact: form = terms + BIG*(Σ window_form - n), threshold
    thr (window all-1 keeps the BIG term at 0; any window 0 pushes form very
    negative so the step never fires).  BIG chosen >> |terms| but << 2^24/RELU_S."""
    BIG = 70000.0                    # > 2^16 (max terms) ; RELU_S*BIG ~ 1.4e7 < 2^24
    n = len(windows)
    form = dict(terms)
    const = 0.0
    for band, coeff, c0 in windows:
        form[band] = form.get(band, 0.0) + BIG * coeff
        const += BIG * c0
    # all windows satisfied -> Σ BIG*window_form == BIG*n ; subtract BIG*n so the
    # window contribution is 0 when all-on, <= -BIG when any off.
    u = A._step_ge(spec, u, form, const - BIG * n, thr, dst, scale)
    return u


# ===========================================================================
# ADD/SUB special-case override + SUB sign flip.
# ===========================================================================
def _fp_sub_flip_block(L, dim):
    """F_SUB: negate operand b's SIGN bit before the shared add machinery, so
    ``a - b`` == ``a + (-b)``.  Flips B_S = 1 - B_S and B_BIT[31] (the sign bit the
    class predicates already read from is B_S; B_BIT[31] is not re-read downstream,
    but flip it too for consistency).  Gated: this block is ONLY in the SUB pipeline
    (there is no OP gate here — the whole block stack is opcode-selected upstream)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 6)
    u = 0
    # B_S := 1 - B_S.
    u = A._ident(spec, u, {L.ONE: 1.0, f.B_S: -2.0}, 0.0, f.B_S, 1.0)   # +1 - 2*B_S (SET delta)
    return A._truncate(spec, u, dim)


def _fp_add_special_flags_block(L, dim):
    """ADD/SUB special-case base flags (b already sign-flipped for SUB):
      nan_in = A_ISNAN OR B_ISNAN
      infdiff = A_ISINF AND B_ISINF AND (A_S != B_S)   -> inf + (-inf) = qNaN
      inf_res = (A_ISINF OR B_ISINF) AND NOT infdiff AND NOT nan  -> signed inf of the inf operand
      bothzero = A_ISZ AND B_ISZ  -> signed zero (sign = A_S AND B_S per IEEE +0 rule;
                for round-to-nearest, (+0)+(+0)=+0, (-0)+(-0)=-0, (+0)+(-0)=+0)
      ovf/udf as in MUL, on R_E.
    The zero-result of a general cancellation (x + -x) is handled by the LZ==... path
    producing a zero significand -> we detect sum==0 via LZ and force +0."""
    _set_one(L)
    f = L.FP32
    NANF, INF0, INFR, ZR, OVF, UDF = (f.FLAGS + 0, f.FLAGS + 1, f.FLAGS + 2,
                                      f.FLAGS + 3, f.FLAGS + 4, f.FLAGS + 5)
    spec = _empty_spec(dim, 80)
    u = 0
    u = A._clear(spec, u, NANF)
    u = A._step_ge(spec, u, {f.A_ISNAN: 1.0, f.B_ISNAN: 1.0}, 0.0, 1, NANF, 1.0)
    # infdiff = A_ISINF AND B_ISINF AND (A_S xor B_S).  A_S xor B_S == ESUB flag
    # BEFORE the sub-flip; here b is already flipped, so opposite-sign infs now have
    # EQUAL signs.  Recompute the sign-difference from A_S,B_S (post-flip): inf+inf
    # with DIFFERING post-flip signs is the qNaN case for ADD; for SUB the flip makes
    # a-b's inf-inf land as equal signs -> qNaN.  Use [A_S != B_S] post-state.
    u = A._clear(spec, u, INF0)
    u = A._guard(spec, u, [(f.A_ISINF, 1.0, 0.0), (f.B_ISINF, 1.0, 0.0), (f.A_S, 1.0, 0.0), (f.B_S, -1.0, 1.0)],
                 {L.ONE: 1.0}, 0.0, INF0, 1.0)   # A_ISINF & B_ISINF & A_S & !B_S
    u = A._guard(spec, u, [(f.A_ISINF, 1.0, 0.0), (f.B_ISINF, 1.0, 0.0), (f.A_S, -1.0, 1.0), (f.B_S, 1.0, 0.0)],
                 {L.ONE: 1.0}, 0.0, INF0, 1.0)   # A_ISINF & B_ISINF & !A_S & B_S
    # inf_res = (A_ISINF OR B_ISINF) AND NOT infdiff AND NOT nan.
    u = A._clear(spec, u, INFR)
    u = A._step_ge(spec, u, {f.A_ISINF: 1.0, f.B_ISINF: 1.0}, 0.0, 1, INFR, 1.0)
    u = A._guard(spec, u, [(INF0, 1.0, 0.0)], {INFR: -1.0}, 0.0, INFR, 1.0)
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {INFR: -1.0}, 0.0, INFR, 1.0)
    # bothzero -> signed zero; general x+-x cancellation -> +0 (the sum is exactly
    # 0, so LZ = -1).  Detect the zero sum via [ Σ PBIT bits < 1 ] and force ZR
    # (a cancellation-to-zero rounds to +0 under round-to-nearest).
    u = A._clear(spec, u, ZR)
    u = A._guard(spec, u, [(f.A_ISZ, 1.0, 0.0), (f.B_ISZ, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, ZR, 1.0)
    # sum==0 (cancellation): 1 - [Σ PBIT >= 1], gated finite (not nan/inf).
    sumbits = {f.PBIT + i: 1.0 for i in range(32)}
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, ZR, 1.0)                     # +1
    u = A._step_ge(spec, u, sumbits, 0.0, 1, ZR, -1.0)                    # -[sum>=1]
    # but the +1-[sum>=1] wrongly adds 1 when a NORMAL nonzero result exists AND
    # bothzero already fired; clamp ZR to 0/1 by re-thresholding is not same-block
    # safe.  Instead: the +1 - [sum>=1] term is 0 for any nonzero sum, 1 for zero
    # sum; bothzero also has sum==0 so it double-counts -> clamp in combine via ZF
    # being a threshold (ZF = [ZR+UDF>=1]).  Subtract nan/inf influence:
    for fl in (NANF, f.A_ISINF, f.B_ISINF):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {ZR: -1.0}, 0.0, ZR, 1.0)
    # ovf = finite AND [R_E + SIG24 >= 255].
    u = A._clear(spec, u, OVF)
    u = A._step_ge(spec, u, {f.R_E: 1.0, f.SIG + 24: 1.0}, 0.0, 255, OVF, 1.0)
    for fl in (NANF, f.A_ISINF, f.B_ISINF):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {OVF: -1.0}, 0.0, OVF, 1.0)
    # udf = finite AND [R_E <= 0].
    u = A._clear(spec, u, UDF)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, UDF, 1.0)
    u = A._step_ge(spec, u, {f.R_E: 1.0, f.SIG + 24: 1.0}, 0.0, 1, UDF, -1.0)
    for fl in (NANF, f.A_ISINF, f.B_ISINF):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {UDF: -1.0}, 0.0, UDF, 1.0)
    # CANCEL = sum==0 AND NOT bothzero AND finite (nonzero operands cancel to +0).
    CANCEL = f.FLAGS + 10
    sumbits2 = {f.PBIT + i: 1.0 for i in range(32)}
    u = A._clear(spec, u, CANCEL)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, CANCEL, 1.0)                 # +1
    u = A._step_ge(spec, u, sumbits2, 0.0, 1, CANCEL, -1.0)              # -[sum>=1] -> sum==0
    # exclude bothzero (its sign rule is handled by ZF) and nan/inf.
    u = A._guard(spec, u, [(f.A_ISZ, 1.0, 0.0), (f.B_ISZ, 1.0, 0.0)], {CANCEL: -1.0}, 0.0, CANCEL, 1.0)
    for fl in (NANF, f.A_ISINF, f.B_ISINF):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {CANCEL: -1.0}, 0.0, CANCEL, 1.0)
    return A._truncate(spec, u, dim)


def _fp_add_inf_sign_block(L, dim):
    """For an inf RESULT (INFR), R_S must be the sign of the (single) inf operand,
    not the magnitude-larger operand (both may be finite-vs-inf).  Override R_S when
    INFR: R_S = A_ISINF ? A_S : B_S."""
    _set_one(L)
    f = L.FP32
    INFR = f.FLAGS + 2
    spec = _empty_spec(dim, 8)
    u = 0
    # only touch R_S when INFR; SET via subtract-old+add-new gated.
    u = A._guard(spec, u, [(INFR, 1.0, 0.0)], {f.R_S: -1.0}, 0.0, f.R_S, 1.0)    # clear R_S if INFR
    u = A._guard(spec, u, [(INFR, 1.0, 0.0), (f.A_ISINF, 1.0, 0.0)], {f.A_S: 1.0}, 0.0, f.R_S, 1.0)
    u = A._guard(spec, u, [(INFR, 1.0, 0.0), (f.A_ISINF, -1.0, 1.0)], {f.B_S: 1.0}, 0.0, f.R_S, 1.0)
    # CANCELLATION to zero (nonzero operands summing to exactly 0) rounds to +0 under
    # round-to-nearest: force R_S = 0.  CANCEL = FLAGS+10 (set in the special flags).
    CANCEL = f.FLAGS + 10
    u = A._guard(spec, u, [(CANCEL, 1.0, 0.0)], {f.R_S: -1.0}, 0.0, f.R_S, 1.0)   # clear R_S if CANCEL
    return A._truncate(spec, u, dim)


def _fp_addsub_core(L, dim):
    """The shared ADD/SUB core (b already sign-adjusted for SUB): prep | swap | rsign
    | shift-oh | align | align-sticky | astk | sum-nibbles | settle | leading-one |
    normalise | sticky | round | encode | special."""
    blocks = [
        ("fp-add-prep", _fp_add_prep_block(L, dim)),
        ("fp-add-swap", _fp_add_swap_block(L, dim)),
        ("fp-add-rsign", _fp_add_rsign_block(L, dim)),
        ("fp-add-shdoh", _fp_add_shd_oh_block(L, dim)),
        ("fp-add-align", _fp_add_align_block(L, dim)),
        ("fp-add-astky", _fp_add_sticky_block(L, dim)),
        ("fp-add-astk", _fp_add_astk_block(L, dim)),
        ("fp-add-sumnib", _fp_add_sumnib_block(L, dim)),
    ]
    blocks += _fp_add_sum_settle_blocks(L, dim)
    blocks += [
        ("fp-add-lead", _fp_add_lead_block(L, dim)),
        ("fp-add-leadoh", _fp_add_lead_oh_block(L, dim)),
        ("fp-add-norm", _fp_add_normalize_block(L, dim)),
        ("fp-add-stickyf", _fp_add_sticky_final_block(L, dim)),
    ]
    blocks += _fp_round_encode_tail(L, dim)
    blocks += [
        ("fp-add-special", _fp_add_special_flags_block(L, dim)),
        ("fp-special-combine", _fp_special_combine_block(L, dim)),
        ("fp-add-infsign", _fp_add_inf_sign_block(L, dim)),
        ("fp-add-override", _fp_special_override_block(L, dim)),
    ]
    return blocks


def compile_fp_add_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Full F_ADD block stack: decode | (align/add/normalise/round/encode) | special."""
    return list(compile_fp_decode_blocks(L, dim)) + _fp_addsub_core(L, dim)


def compile_fp_sub_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Full F_SUB block stack: decode | flip b's sign | (shared add core)."""
    blocks = list(compile_fp_decode_blocks(L, dim))
    blocks.append(("fp-sub-flip", _fp_sub_flip_block(L, dim)))
    blocks += _fp_addsub_core(L, dim)
    return blocks


# ===========================================================================
# F_DIV : bit-serial restoring division of the significands (§Division).
#
# significand quotient q = A_M / B_M in [0.5, 2).  Compute the integer
# ``QINT = floor(A_M * 2^_DIV_SHIFT / B_M)`` (a ``_DIV_SHIFT+1``-bit number whose
# leading 1 is at position _DIV_SHIFT or _DIV_SHIFT-1) by restoring division:
#   R = A_M ; for k in 1.._DIV_SHIFT+1:  R = 2R ; q_k = [R>=B_M] ; R -= q_k*B_M.
# R stays < 2*B_M < 2^25 (fp32-exact as a scalar).  The final remainder R!=0 is the
# division sticky.  Then normalise QINT's leading 1 to bit 23 (significand), take
# guard/round/sticky, round, encode.  Result exp = A_E - B_E + 127 (+ the 1-bit
# normalise when A_M < B_M -> quotient < 1).
# ===========================================================================
_DIV_SHIFT = 26        # quotient integer bits (24 sig + guard + round; sticky = R!=0)


# The remainder is a BIT-VECTOR (each slot 0/1) to avoid the residue amplification
# a big-scalar remainder suffers under the per-iteration doubling (2^-24 gadget
# residue * 2^26 -> corrupt quotient; see _fp_div_debug_findings.md).  DRB holds R
# in bits 0..DRW-1; the divisor B_M in BBIT bits 0..23.  Per iteration:
#   S = (k==0) ? R : 2R            (a BIT RE-INDEX, exact — no scalar doubling)
#   q = [S >= B_M]                 (bit-serial subtract borrow-out == 0)
#   R = q ? S - B_M : S            (restoring)
# recorded MSB-first: QBIT[_DIV_SHIFT - k] = q.
_DRW = 26              # remainder bit-vector width (R < 2*B_M < 2^25 -> 25 bits; +1)
_DIV_Q = 24            # DCMP slot holding this iteration's quotient bit
_DIV_BRW = 25          # DCMP slot region base for the borrow chain (uses DRB2 tail)


def _fp_div_init_block(L, dim):
    """Set up the bit-vector divider: remainder DRB = A_M bits (0..23), divisor BBIT
    = B_M bits (0..23), high DRB slots and QBIT cleared."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, (24 * 3) + _DRW + 28 + 16)
    u = 0
    for i in range(_DRW):
        u = A._clear(spec, u, f.DRB + i)
    for i in range(24):
        u = A._ident(spec, u, {f.A_M + i: 1.0}, 0.0, f.DRB + i, 1.0)   # R = A_M
        u = A._clear(spec, u, f.BBIT + i)
        u = A._ident(spec, u, {f.B_M + i: 1.0}, 0.0, f.BBIT + i, 1.0)  # B = B_M bits
    for k in range(28):
        u = A._clear(spec, u, f.QBIT + k)
    return A._truncate(spec, u, dim)


def _fp_div_shift_block(L, dim, k):
    """Iteration part A: S = 2R for k>0 (bit re-index: S[i]=R[i-1]), S = R for k==0.
    Writes S into DRB2 (double-buffer).  All bits stay 0/1 — no scalar doubling, so
    NO residue.  k==0 is the integer bit: compares A_M directly (quotient in [1,2))."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, _DRW * 2 + 4)
    u = 0
    for i in range(_DRW):
        u = A._clear(spec, u, f.DRB2 + i)
    if k == 0:
        for i in range(_DRW):
            u = A._ident(spec, u, {f.DRB + i: 1.0}, 0.0, f.DRB2 + i, 1.0)
    else:
        for i in range(1, _DRW):
            u = A._ident(spec, u, {f.DRB + (i - 1): 1.0}, 0.0, f.DRB2 + i, 1.0)   # S[i]=R[i-1]
    return A._truncate(spec, u, dim)


_DIV_NIB = 8           # nibble columns for the 32-bit two's-complement subtract
_DIV_CARRY_ROUNDS = 8  # carry-settle rounds (== nibble count: worst-case 0xF-chain ripple)


def _fp_div_sncol_block(L, dim, k):
    """Iteration part B.0: form the two's-complement subtract columns S + (~B) + 1.
    Pack S (in DRB2 bits, < 2^26) into 8 nibbles and add the 32-bit ones' complement
    of B_M plus 1:
        col[c] = S_nibble[c] + (0xF - B_nibble[c]) + (c==0 ? 1 : 0)
    So Σ col[c]*16^c = S + (2^32 - 1 - B) + 1 = S - B + 2^32.  After carry-settle the
    bit-32 carry-out overflows past nibble 7 and is dropped; the surviving bit 31
    region tells the sign: since S,B < 2^26, S-B+2^32 lands in nibble 7 as either
    0xF.. (S<B, the top borrow) or 0x0.. with the +2^32 wrapping — cleaner: we read
    q = [S >= B] as bit 31 of (S - B) two's complement over 32 bits being 0.  With the
    +2^32 wrap, S>=B -> result S-B (bit31=0), S<B -> result 2^32+(S-B) (bit31=1).
    So q = 1 - bit31.  Columns start <= 30, carry-settled next."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, _DIV_NIB * 24)
    u = 0
    for c in range(_DIV_NIB):
        col = f.DCOL + c
        u = A._clear(spec, u, col)
        for p in range(4):
            sidx = 4 * c + p
            sbit = f.DRB2 + sidx if sidx < _DRW else None
            if sbit is not None:
                u = A._ident(spec, u, {sbit: float(1 << p)}, 0.0, col, 1.0)       # + S nibble
            bbit = f.BBIT + sidx if sidx < 24 else None
            # + (1 - B_bit)*2^p   (ones' complement of B over 32 bits)
            if bbit is not None:
                u = A._ident(spec, u, {L.ONE: float(1 << p), bbit: -float(1 << p)}, 0.0, col, 1.0)
            else:
                u = A._ident(spec, u, {L.ONE: float(1 << p)}, 0.0, col, 1.0)      # B bit is 0
        if c == 0:
            u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, col, 1.0)                     # +1 (two's comp)
    return A._truncate(spec, u, dim)


def _fp_div_carry_blocks(L, dim):
    """Carry-settle the 8 subtract columns (raw <= 30) into clean nibbles over
    _DIV_CARRY_ROUNDS, ping-ponging DCOL<->DCOL2.  The bit-32 carry-out overflows past
    the top nibble and is dropped (exactly the mod-2^32 the two's complement wants).
    Returns the settled band base (== DCOL for an odd/even round count)."""
    f = L.FP32
    blocks = []
    src, dst = f.DCOL, f.DCOL2
    for r in range(_DIV_CARRY_ROUNDS):
        blocks.append((f"fp-div-carry{r}", A._carry_round_block(L, dim, src, dst, _DIV_NIB)))
        src, dst = dst, src
    return blocks, src


def _fp_div_ddiff_block(L, dim, settled):
    """Iteration part B.1a: explode the settled subtract nibbles into the 32 result
    bits DDIFF (each 0/1).  DDIFF holds (S - B) mod 2^32 = S - B + 2^32; bit 31 == 1
    iff S < B (since S,B < 2^26 -> S-B+2^32 in [2^31, 2^32) when S<B, else in [2^32,..)
    which wraps to [0, 2^26))."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, _DIV_NIB * 60 + 8)
    u = 0
    u = _nibble_bits_inline(spec, u, L, settled, f.DDIFF, _DIV_NIB)
    return A._truncate(spec, u, dim)


def _fp_div_q_block(L, dim, k):
    """Iteration part B.1b: q = 1 - DDIFF[31] = [S >= B] (SEPARATE block so it reads the
    WRITTEN DDIFF).  Records QBIT[_DIV_SHIFT-k] = q."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 12)
    u = 0
    q = f.DCMP + 0
    # both q and the QBIT record read DDIFF[31] (a block input) directly.
    u = A._clear(spec, u, q)
    u = A._ident(spec, u, {L.ONE: 1.0, f.DDIFF + 31: -1.0}, 0.0, q, 1.0)   # q = 1 - bit31
    dst = f.QBIT + (_DIV_SHIFT - k)
    u = A._clear(spec, u, dst)
    u = A._ident(spec, u, {L.ONE: 1.0, f.DDIFF + 31: -1.0}, 0.0, dst, 1.0)  # = q
    return A._truncate(spec, u, dim)


def _nibble_bits_inline(spec, u, L, nib_base, bit_base, n):
    """Explode n settled nibbles (each 0..15) into 4n bits (each 0/1).  bit p of nibble
    c = [ (nib - 16*hi... ) ] — use the standard nibble->bit extraction: bit3 = [nib>=8],
    bit2=[nib-8*bit3>=4], etc.  Reuses the module bit-extract helper."""
    for c in range(n):
        u = _bit_extract_units(spec, u, L, nib_base + c, bit_base + 4 * c)
    return u


def _fp_div_sub_block(L, dim, k):
    """Iteration part D: R = q ? (S - B) : S.  S bits in DRB2, (S-B) bits in DDIFF
    (low _DRW relevant), q in DCMP.  Each new R bit is SNAPPED to a clean 0/1 via a
    half-threshold so the ~value residue the select/explode leave cannot accumulate
    across the 26 iterations (the bit-vector analogue of keeping ALU columns < 256):
        R[i] = [ S[i] + q*(DDIFF[i] - S[i]) >= 0.5 ]  = [ (q ? DDIFF[i] : S[i]) == 1 ].
    Since q and the bits are 0/1, the selected value is exactly 0 or 1 (± tiny residue),
    so the half-threshold recovers the clean bit.  _step_ge is residue-immune at this
    scale (form magnitude ~1)."""
    _set_one(L)
    f = L.FP32
    q = f.DCMP + 0
    spec = _empty_spec(dim, _DRW * 20)
    u = 0
    for i in range(_DRW):
        # sel = S[i] + q*(DDIFF[i]-S[i]).  Snap via [sel >= 0.5].  _step_ge only reads
        # the block INPUT so we express sel as a single linear form + a guarded term:
        #   [sel >= 0.5] with sel = (1-q)*S[i] + q*DDIFF[i].  Two disjoint guarded
        #   half-thresholds keep it exact:
        #     when q: R[i] = [DDIFF[i] >= 0.5]
        #     when !q: R[i] = [S[i]  >= 0.5]
        u = A._clear(spec, u, f.DRB + i)
        u = _step_ge_gated(spec, u, L, [(q, 1.0, 0.0)], {f.DDIFF + i: 1.0}, 1, f.DRB + i, 1.0)
        u = _step_ge_gated(spec, u, L, [(q, -1.0, 1.0)], {f.DRB2 + i: 1.0}, 1, f.DRB + i, 1.0)
    return A._truncate(spec, u, dim)


def _fp_div_lead_block(L, dim):
    """Leading-one index of the quotient QINT (bits QBIT[0.._DIV_SHIFT]).  QINT =
    A_M/B_M * 2^_DIV_SHIFT with A_M,B_M in [2^23,2^24), so QINT in [2^(_DIV_SHIFT-1),
    2^(_DIV_SHIFT+1)) -> leading bit at _DIV_SHIFT or _DIV_SHIFT-1.  LZ = leading
    index into LZ+8."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 20)
    u = 0
    u = A._clear(spec, u, f.LZ + 8)
    for k in (_DIV_SHIFT, _DIV_SHIFT - 1):
        # any bit >= k set?  For a normalized quotient only the top two positions
        # matter for the leading index.
        hi_terms = {f.QBIT + i: 1.0 for i in range(k, _DIV_SHIFT + 1)}
        u = A._step_ge(spec, u, hi_terms, 0.0, 1, f.LZ + 8, 1.0)
    u = A._ident(spec, u, {L.ONE: float(_DIV_SHIFT - 2)}, 0.0, f.LZ + 8, 1.0)   # base index
    # hi = [LZ == _DIV_SHIFT]  (else LZ == _DIV_SHIFT-1) — computed HERE (not in the
    # normalize block) so the normalize's significand mux reads a WRITTEN hi flag, not
    # the stale block input.  QBIT bit _DIV_SHIFT set <=> LZ == _DIV_SHIFT.
    u = A._clear(spec, u, f.LZ + 9)
    u = A._step_ge(spec, u, {f.QBIT + _DIV_SHIFT: 1.0}, 0.0, 1, f.LZ + 9, 1.0)
    return A._truncate(spec, u, dim)


def _fp_div_normalize_block(L, dim):
    """Normalise the quotient: leading index LZ (either _DIV_SHIFT or _DIV_SHIFT-1).
    significand bit j = QBIT[LZ-23+j] (j=0..23) ; guard = QBIT[LZ-24] ; sticky =
    OR(QBIT bits below LZ-24) OR (final remainder DR != 0).  R_E = A_E - B_E + 127 +
    (LZ - _DIV_SHIFT).  R_S = A_S xor B_S.  Two LZ cases only, so a small mux.
    ``hi`` is computed by the PRIOR lead block (block-input clean)."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 24 * 6 + 60)
    u = 0
    hi = f.LZ + 9        # hi = [LZ == _DIV_SHIFT]  (written by lead block)
    top = _DIV_SHIFT
    # significand bit j = hi ? QBIT[top-23+j] : QBIT[top-1-23+j].
    for j in range(24):
        dst = f.MROUND + j
        u = A._clear(spec, u, dst)
        u = A._guard(spec, u, [(hi, 1.0, 0.0)], {f.QBIT + (top - 23 + j): 1.0}, 0.0, dst, 1.0)
        u = A._guard(spec, u, [(hi, -1.0, 1.0)], {f.QBIT + (top - 1 - 23 + j): 1.0}, 0.0, dst, 1.0)
    # guard = hi ? QBIT[top-24] : QBIT[top-25].
    u = A._clear(spec, u, f.GRS + 0)
    u = A._guard(spec, u, [(hi, 1.0, 0.0)], {f.QBIT + (top - 24): 1.0}, 0.0, f.GRS + 0, 1.0)
    if top - 25 >= 0:
        u = A._guard(spec, u, [(hi, -1.0, 1.0)], {f.QBIT + (top - 25): 1.0}, 0.0, f.GRS + 0, 1.0)
    # sticky (round bits below guard) + remainder-nonzero.
    #   hi -> bits QBIT[0..top-25] ; !hi -> bits QBIT[0..top-26].
    u = A._clear(spec, u, f.GRS + 2)
    hi_terms = {f.QBIT + i: 1.0 for i in range(0, top - 24)}      # bits below guard (hi case)
    lo_terms = {f.QBIT + i: 1.0 for i in range(0, max(0, top - 25))}
    u = _guarded_or(spec, u, L, [(hi, 1.0, 0.0)], hi_terms, f.GRS + 2)
    u = _guarded_or(spec, u, L, [(hi, -1.0, 1.0)], lo_terms, f.GRS + 2)
    # remainder != 0 -> sticky.  (bit-vector remainder: OR of all DRB bits)
    u = A._step_ge(spec, u, {f.DRB + i: 1.0 for i in range(_DRW)}, 0.0, 1, f.GRS + 2, 1.0)
    # R_E = A_E - B_E + 127 + (LZ - _DIV_SHIFT).
    u = A._clear(spec, u, f.R_E)
    u = A._ident(spec, u, {f.A_E: 1.0, f.B_E: -1.0, f.LZ + 8: 1.0, L.ONE: 127.0 - _DIV_SHIFT},
                 0.0, f.R_E, 1.0)
    # R_S = A_S xor B_S.
    u = A._clear(spec, u, f.R_S)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 1, f.R_S, 1.0)
    u = A._step_ge(spec, u, {f.A_S: 1.0, f.B_S: 1.0}, 0.0, 2, f.R_S, -1.0)
    return A._truncate(spec, u, dim)


def _guarded_or(spec, u, L, windows, terms, dst):
    """dst += AND(windows) * [Σ terms >= 1]  (a guarded OR of the terms).  Uses the
    moderate-BIG combined step so it stays fp32-exact."""
    if not terms:
        return u
    return _step_ge_gated(spec, u, L, windows, terms, 1, dst, 1.0)


def _fp_div_sticky_final_block(L, dim):
    """STICKY = [GRS[2] >= 1]  (the div normalize accumulated the round+remainder
    sticky as a count in GRS[2])."""
    _set_one(L)
    f = L.FP32
    spec = _empty_spec(dim, 4)
    u = 0
    u = A._clear(spec, u, f.STICKY)
    u = A._step_ge(spec, u, {f.GRS + 2: 1.0}, 0.0, 1, f.STICKY, 1.0)
    return A._truncate(spec, u, dim)


def _fp_div_special_flags_block(L, dim):
    """F_DIV special-case base flags:
      nan_in = A_ISNAN OR B_ISNAN
      inf0   = (A_ISINF AND B_ISINF) OR (A_ISZ AND B_ISZ)     -> qNaN (inf/inf, 0/0)
      inf_res= (A_ISINF AND NOT B_ISINF) OR (B_ISZ AND NOT A_ISZ AND NOT A_ISINF)
                                                              -> signed inf (finite/0, inf/finite)
      zero_res=(A_ISZ AND NOT B_ISZ) OR (B_ISINF AND NOT A_ISINF)  -> signed 0
      ovf/udf on R_E (finite/finite).
    Sign = A_S xor B_S (already R_S)."""
    _set_one(L)
    f = L.FP32
    NANF, INF0, INFR, ZR, OVF, UDF = (f.FLAGS + 0, f.FLAGS + 1, f.FLAGS + 2,
                                      f.FLAGS + 3, f.FLAGS + 4, f.FLAGS + 5)
    spec = _empty_spec(dim, 100)
    u = 0
    u = A._clear(spec, u, NANF)
    u = A._step_ge(spec, u, {f.A_ISNAN: 1.0, f.B_ISNAN: 1.0}, 0.0, 1, NANF, 1.0)
    u = A._clear(spec, u, INF0)
    u = A._guard(spec, u, [(f.A_ISINF, 1.0, 0.0), (f.B_ISINF, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, INF0, 1.0)
    u = A._guard(spec, u, [(f.A_ISZ, 1.0, 0.0), (f.B_ISZ, 1.0, 0.0)], {L.ONE: 1.0}, 0.0, INF0, 1.0)
    # inf_res: A inf (B finite/zero, not inf) OR B zero (A finite, not zero/inf).
    u = A._clear(spec, u, INFR)
    u = A._guard(spec, u, [(f.A_ISINF, 1.0, 0.0), (f.B_ISINF, -1.0, 1.0)], {L.ONE: 1.0}, 0.0, INFR, 1.0)
    u = A._guard(spec, u, [(f.B_ISZ, 1.0, 0.0), (f.A_ISZ, -1.0, 1.0), (f.A_ISINF, -1.0, 1.0)],
                 {L.ONE: 1.0}, 0.0, INFR, 1.0)
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {INFR: -1.0}, 0.0, INFR, 1.0)
    # zero_res: A zero (B finite, not zero) OR B inf (A finite, not inf).
    u = A._clear(spec, u, ZR)
    u = A._guard(spec, u, [(f.A_ISZ, 1.0, 0.0), (f.B_ISZ, -1.0, 1.0), (f.B_ISINF, -1.0, 1.0)],
                 {L.ONE: 1.0}, 0.0, ZR, 1.0)
    u = A._guard(spec, u, [(f.B_ISINF, 1.0, 0.0), (f.A_ISINF, -1.0, 1.0)], {L.ONE: 1.0}, 0.0, ZR, 1.0)
    u = A._guard(spec, u, [(NANF, 1.0, 0.0)], {ZR: -1.0}, 0.0, ZR, 1.0)
    # ovf/udf finite/finite.
    u = A._clear(spec, u, OVF)
    u = A._step_ge(spec, u, {f.R_E: 1.0}, 0.0, 255, OVF, 1.0)
    for fl in (NANF, f.A_ISINF, f.B_ISINF, f.A_ISZ, f.B_ISZ):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {OVF: -1.0}, 0.0, OVF, 1.0)
    u = A._clear(spec, u, UDF)
    u = A._ident(spec, u, {L.ONE: 1.0}, 0.0, UDF, 1.0)
    u = A._step_ge(spec, u, {f.R_E: 1.0}, 0.0, 1, UDF, -1.0)
    for fl in (NANF, f.A_ISINF, f.B_ISINF, f.A_ISZ, f.B_ISZ):
        u = A._guard(spec, u, [(fl, 1.0, 0.0)], {UDF: -1.0}, 0.0, UDF, 1.0)
    return A._truncate(spec, u, dim)


def compile_fp_div_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Full F_DIV block stack: decode | init | 27 x (shift|q|sub) restoring division
    | leading-one | normalise | round | encode | special."""
    blocks = list(compile_fp_decode_blocks(L, dim))
    blocks.append(("fp-div-init", _fp_div_init_block(L, dim)))
    for k in range(_DIV_SHIFT + 1):
        blocks.append((f"fp-div-sh{k}", _fp_div_shift_block(L, dim, k)))
        blocks.append((f"fp-div-sn{k}", _fp_div_sncol_block(L, dim, k)))
        carry_blocks, settled = _fp_div_carry_blocks(L, dim)
        for (nm, spec) in carry_blocks:
            blocks.append((f"{nm}-k{k}", spec))
        blocks.append((f"fp-div-dd{k}", _fp_div_ddiff_block(L, dim, settled)))
        blocks.append((f"fp-div-q{k}", _fp_div_q_block(L, dim, k)))
        blocks.append((f"fp-div-sb{k}", _fp_div_sub_block(L, dim, k)))
    blocks.append(("fp-div-lead", _fp_div_lead_block(L, dim)))
    blocks.append(("fp-div-norm", _fp_div_normalize_block(L, dim)))
    blocks.append(("fp-div-stickyf", _fp_div_sticky_final_block(L, dim)))
    blocks += _fp_round_encode_tail(L, dim)
    blocks.append(("fp-div-special", _fp_div_special_flags_block(L, dim)))
    blocks.append(("fp-special-combine", _fp_special_combine_block(L, dim)))
    blocks.append(("fp-div-override", _fp_special_override_block(L, dim)))
    return blocks


