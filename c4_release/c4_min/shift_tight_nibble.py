"""TIGHT SHL/SHR — a DIRECT 8x8 nibble select (NO log stages).

The landed shifters realise a 32-bit shift as a *log*-shifter: the bit-granular
gadget in ``nibble_bitwise`` uses 5 conditional ``2**k`` mux stages over 32
bit-planes (~5 269 nz / 8 blocks), and the nibble-granular mux-tree in
``shifter_bakeoff`` uses 3 coarse log stages over 8 nibble-planes plus a per-value
amount decode over all 64 shift counts (4 140 nz / 8 blocks).  Both spend most of
their weight on machinery that is NOT the shift itself: the per-value amount
one-hot and the pipeline of log stages.

This module builds the **tightest** nibble shifter by removing BOTH overheads:

  1. **Amount decode via the equality-pulse gadget** (ONE early FFN).  Decompose
     ``n = 4*c + r`` with ``c = n // 4`` (coarse whole-nibble shift, 0..7) and
     ``r = n mod 4`` (fine sub-nibble shift, 0..3).  We emit the coarse one-hot
     ``ceq[k] = [c == k]`` (k=0..7, 8 equality pulses) and the fine one-hot
     ``feq[m] = [r == m]`` (m=0..3, 4 pulses) DIRECTLY from ``n`` — one shared
     relu staircase each, NOT a 64-wide per-value scan.  Plus the ``keep =
     [n < 32]`` kill flag.

  2. **Coarse = a DIRECT 8x8 nibble select** (the ~64-term core, NO log stages).
     For each output nibble ``j`` the coarse-shifted nibble is the single-term
     sum

         out_nib[j] = sum_k in_nib[j-k] * ceq[k]      (SHL)
         out_nib[j] = sum_k in_nib[j+k] * ceq[k]      (SHR)

     over the VALID neighbours only (triangular: SHL keeps k=0..j, SHR keeps
     k=0..7-j).  Each term is ONE gated SwiGLU unit — the ``_guard`` AND
     primitive with window ``ceq[k]==1`` and gate value ``in_nib[j-k]`` (a nibble
     <= 15, tiny).  That is ~36 terms per direction, and it IS the whole coarse
     shift: no intermediate stage buffers, no log pipeline.

  3. **Fine = a minimal sub-nibble shift** (0..3 bits).  Per coarse-shifted
     nibble ``v`` form ``p = v * 2**r`` (gated by ``feq[r]`` -> multiply by one of
     {1,2,4,8}), then split ``p`` (<= 15*8 = 120) into the low nibble ``p mod 16``
     and the carry ``floor(p/16)`` (kmax=7) that flows into the neighbour nibble
     (UP for SHL, DOWN for SHR).

  4. **Lean muxes.**  Every gated unit is the bare ``_guard`` (1 window + 1 gate
     term -> 4 nz) or a bare copy/step (3-8 nz) — no guard-window lowering
     overhead.  The achieved nz/unit is reported by the driver.

  5. **n >= 32 -> 0.**  ``keep = [n < 32]`` gates the final recompose write, so any
     shift count >= 32 (any bit of ``n`` at position >= 5) zeroes the result,
     matching ``ref_interpret(mask=0xFFFFFFFF)`` (``(pop <</>> n) & 0xFFFFFFFF``,
     ``n`` UNMASKED).

fp32 discipline: every value is a nibble (<= 15), a one-hot (0/1), a small carry
(<= 7) or a product ``v*2**r <= 120`` — no hidden unit ever forms an argument
anywhere near 2**24.  No MAGIC constant, no fp64.

Reuses primitives from ``nibble_alu32`` (``_empty_spec`` / ``_floor_div_pow`` /
``_mul_gate`` / ``_guard`` / ``_step_ge`` / ``_ident`` / RELU_S) and the
equality-pulse / one-hot construction pattern (``_step_ge`` differences, as
``shifter_bakeoff._pow_from_n`` does).  Does NOT edit any existing file; does NOT
touch ``shifter_bakeoff`` / ``shift_attention_bench`` / ``mul_*``.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import nibble_alu32 as alu
from .nibble_alu32 import _empty_spec, _floor_div_pow, _mul_gate, RELU_S

MASK32 = 0xFFFFFFFF
FP32_INT_LIMIT = 1 << 24          # fp32 loses unit precision above 2**24.

N_NIB = 8                         # 8 nibbles carry the low 32 bits.
COARSE_MAX = N_NIB - 1            # c = n // 4 in 0..7 (coarse whole-nibble shift).
FINE_MAX = 3                      # r = n mod 4 in 0..3 (fine sub-nibble shift).


# ===========================================================================
# Residual layout — just the bands the tight shifter touches (NOT a DIM-8192
# model residual).  A scalar ONE lane, the source/amount scalars, the 8 source
# nibble planes, the coarse/fine one-hots, and the scratch/result nibble bands.
# ===========================================================================
class _TightLayout:
    def __init__(self):
        off = 0

        def band(sz):
            nonlocal off
            b = off
            off += sz
            return b

        self.ONE = band(1)
        self.POP = band(1)                    # source value (load/decode reference)
        self.N = band(1)                      # shift amount n
        self.IN_NIB = band(N_NIB)             # the 8 source nibbles (0..15)
        # amount decode outputs:
        self.C = band(1)                      # c = n // 4  (0..7)
        self.R = band(1)                      # r = n mod 4  (0..3) = n - 4c
        self.CEQ = band(N_NIB)                # ceq[k] = [c == k]  (k=0..7)
        self.FEQ = band(FINE_MAX + 1)         # feq[m] = [r == m]  (m=0..3)
        self.POW = band(1)                    # fine multiplier (SHL 2**r ; SHR 2**(4-r))
        self.RNZ = band(1)                    # rnz = [r != 0]  (SHR r=0 whole-nibble bypass)
        self.KEEP = band(1)                   # keep = [n < 32]  (n>=32 -> 0)
        # coarse-select output:
        self.COARSE = band(N_NIB)             # coarse-shifted nibbles
        # fine scratch:
        self.PROD = band(N_NIB)               # coarse_nib * 2**r  (<= 120)
        self.FINE_LO = band(N_NIB)            # prod mod 16
        self.FINE_CO = band(N_NIB)            # floor(prod / 16)  (carry, <= 7)
        # result:
        self.OUT_NIB = band(N_NIB)            # final result nibbles (post keep-gate)
        self.D = off

    def load(self, pop: int, n: int) -> torch.Tensor:
        x = torch.zeros(self.D)
        x[self.ONE] = 1.0
        pop &= MASK32
        x[self.POP] = float(pop)
        x[self.N] = float(n)
        for j in range(N_NIB):
            x[self.IN_NIB + j] = float((pop >> (4 * j)) & 0xF)
        return x

    def decode(self, x: torch.Tensor) -> int:
        val = 0
        for j in range(N_NIB):
            val |= (int(round(float(x[self.OUT_NIB + j]))) & 0xF) << (4 * j)
        return val & MASK32


def _set_one(one_band: int) -> None:
    """Point the shared ALU emitters' ONE lane at ``one_band``."""
    alu._ONE = one_band


def _truncate(spec, u, dim):
    u = max(1, u)
    return {
        "W_up": spec["W_up"][:u].contiguous(), "b_up": spec["b_up"][:u].contiguous(),
        "W_gate": spec["W_gate"][:u].contiguous(), "b_gate": spec["b_gate"][:u].contiguous(),
        "W_down": spec["W_down"][:, :u].contiguous(), "b_down": spec["b_down"].contiguous(),
    }


def _clear(spec, u, band):
    """dst -= old(dst)  (SET semantics; every unit reads the block INPUT)."""
    return alu._ident(spec, u, {band: 1.0}, 0.0, band, -1.0)


# ===========================================================================
# BLOCK 1/2 — the AMOUNT DECODE (the equality-pulse gadget).
# ===========================================================================
def _one_hot_pulse(spec, u, src_band: int, k: int, dst_band: int, one_band: int):
    """dst_band += [src == k]  for an integer ``src``, as ONE §510 point pulse
    ``step(src>=k) - step(src>=k+1)`` (a difference of two sharp relu ramps).
    Exactly the equality-pulse the reference one-hot builders lower to; used for
    ceq[k] and feq[m]."""
    u = alu._step_ge(spec, u, {src_band: 1.0}, 0.0, k, dst_band, 1.0)
    u = alu._step_ge(spec, u, {src_band: 1.0}, 0.0, k + 1, dst_band, -1.0)
    return u


def _pow_of_r(r: int, left: bool) -> int:
    """The fine multiplier for sub-nibble shift ``r`` (0..3).

    SHL: ``p = coarse * 2**r`` -> low nibble ``p mod 16`` stays, carry
    ``floor(p/16)`` flows UP.  POW = 2**r (1,2,4,8).

    SHR: form ``p = coarse * 2**(4-r)`` (r>0) so the peel gives HIGH part
    ``floor(p/16) = coarse >> r`` (surviving bits) and LOW part
    ``p mod 16 = (coarse mod 2**r)*2**(4-r)`` (the r bits that fall DOWN,
    already left-aligned in the neighbour nibble).  r=0 is the pure
    whole-nibble shift (POW unused; bypassed via RNZ)."""
    if left:
        return 1 << r
    return 0 if r == 0 else (1 << (4 - r))


def build_amount_blocks(left: bool) -> Callable[["_TightLayout"], List[dict]]:
    """Return a builder that emits the amount-decode blocks for a layout.

    TWO tiny blocks (each unit reads the block INPUT, so ``c``/``r`` and the
    one-hots that depend on them cannot share a block):

      block 1:  c = n//4  and  r = n mod 4 = n - 4c  (ONE shared floor staircase
                routed to both), plus keep = [n < 32].  The floor(n/4) staircase
                needs only kmax=7 (n<32 -> c<=7; n>=32 is zeroed by keep).
      block 2:  the equality-pulse one-hots ceq[k]=[c==k] (8) and feq[m]=[r==m]
                (4), plus POW = sum_m pow_of_r(m)*feq[m] and RNZ = [r != 0] — all
                read from the SCALAR c/r, so they are 4/8-wide, NOT a 64-wide scan.

    Direction-aware only in POW (SHL 2**r, SHR 2**(4-r)) and RNZ (SHR r=0 bypass)."""

    def emit(L: "_TightLayout") -> List[dict]:
        # ---- block 1: c = floor(n/4), r = n - 4c, keep = [n<32]. ----
        b1 = _empty_spec(L.D, 120)
        u = 0
        u = _clear(b1, u, L.C)
        u = _clear(b1, u, L.R)
        # r seeded with +n; the SHARED floor(n/4) staircase routes +1 into C and
        # -4 into R, giving c and r = n - 4c in ONE staircase (kmax=7).
        u = alu._ident(b1, u, {L.N: 1.0}, 0.0, L.R, 1.0)                    # r += n
        for k in range(1, 8):                                              # floor(n/4), kmax=7
            u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, 4 * k, L.C, 1.0)      # c += [n>=4k]
            u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, 4 * k, L.R, -4.0)     # r -= 4*[n>=4k]
        # keep = [n < 32] = 1 - step(n >= 32).
        u = alu._ident(b1, u, {L.ONE: 1.0}, 0.0, L.KEEP, 1.0)
        u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, 32, L.KEEP, -1.0)
        block1 = _truncate(b1, u, L.D)

        # ---- block 2: the one-hots + POW + RNZ, all from the SCALAR c / r. ----
        b2 = _empty_spec(L.D, 120)
        u = 0
        # ceq[k] = [c == k]  (k = 0..7).
        for k in range(N_NIB):
            u = _clear(b2, u, L.CEQ + k)
            u = _one_hot_pulse(b2, u, L.C, k, L.CEQ + k, L.ONE)
        # feq[m] = [r == m]  (m = 0..3), the spec's fine one-hot (§ amount decode).
        for m in range(FINE_MAX + 1):
            u = _clear(b2, u, L.FEQ + m)
            u = _one_hot_pulse(b2, u, L.R, m, L.FEQ + m, L.ONE)
        # POW = sum_m pow_of_r(m) * [r == m].  feq is a block-2 output (every unit
        # reads the block INPUT), so POW reads the SCALAR r directly — the SAME
        # 4-wide one-hot the feq bands hold, just fused into POW so no extra block.
        u = _clear(b2, u, L.POW)
        for m in range(FINE_MAX + 1):
            p = _pow_of_r(m, left)
            if p != 0:
                u = alu._step_ge(b2, u, {L.R: 1.0}, 0.0, m, L.POW, float(p))
                u = alu._step_ge(b2, u, {L.R: 1.0}, 0.0, m + 1, L.POW, -float(p))
        # RNZ = [r != 0] = step(r >= 1).
        u = _clear(b2, u, L.RNZ)
        u = alu._step_ge(b2, u, {L.R: 1.0}, 0.0, 1, L.RNZ, 1.0)
        block2 = _truncate(b2, u, L.D)
        return [block1, block2]

    return emit


# ===========================================================================
# BLOCK 3 — the DIRECT 8x8 nibble select (the ~64-term coarse core, NO log
# stages).  out_nib[j] = sum over valid k of in_nib[j-/+k] gated on ceq[k].
# ===========================================================================
def build_coarse_select(L: "_TightLayout", left: bool) -> dict:
    """One FFN block: the whole coarse whole-nibble shift as a direct triangular
    select.  For each output nibble ``j`` and each valid source distance ``k``, one
    ``_guard`` unit routes ``in_nib[j-k]`` (SHL) / ``in_nib[j+k]`` (SHR) into
    ``COARSE[j]`` when ``ceq[k] == 1``.  No stage pipeline; ONE block.

    Valid neighbours (others are shifted-in zeros, so their terms are omitted):
        SHL: source j-k in [0,7] -> k in 0..j          (triangular)
        SHR: source j+k in [0,7] -> k in 0..7-j        (triangular)
    Total terms = sum (j+1) = 36 per direction."""
    spec = _empty_spec(L.D, N_NIB * (N_NIB + 1))
    u = 0
    for j in range(N_NIB):
        u = _clear(spec, u, L.COARSE + j)                 # SET
        for k in range(N_NIB):
            src = (j - k) if left else (j + k)
            if 0 <= src < N_NIB:
                # COARSE[j] += in_nib[src]   when   ceq[k] == 1.
                u = alu._guard(spec, u, [(L.CEQ + k, 1.0, 0.0)],
                               {L.IN_NIB + src: 1.0}, 0.0, L.COARSE + j, 1.0)
    return _truncate(spec, u, L.D)


# ===========================================================================
# BLOCK 4/5 — the FINE sub-nibble shift (0..3 bits).
# ===========================================================================
def build_fine_product(L: "_TightLayout") -> dict:
    """FINE step 1: PROD[j] = COARSE[j] * POW, one 6-weight silu-gated multiply
    per nibble.  SHL POW = 2**r (1,2,4,8) -> PROD <= 15*8 = 120; SHR POW =
    2**(4-r) for r>0 (2,4,8) and 0 for r=0 -> PROD <= 15*8 = 120."""
    spec = _empty_spec(L.D, N_NIB * 4)
    u = 0
    for j in range(N_NIB):
        u = _clear(spec, u, L.PROD + j)
        u = _mul_gate(spec, u, L.COARSE + j, L.POW, L.PROD + j, 1.0)
    return _truncate(spec, u, L.D)


def build_fine_peel(L: "_TightLayout") -> dict:
    """FINE step 2: split PROD[j] (0..120) into FINE_LO[j] = prod mod 16 and
    FINE_CO[j] = floor(prod / 16)  (carry into the neighbour nibble; <= 7, so
    kmax=7).  ONE shared floor(prod/16) staircase per nibble routed to BOTH
    destinations via ``_floor_div_pow2`` (co = +floor into FINE_CO; -16*floor into
    FINE_LO), plus a direct prod-read for FINE_LO — half the staircase units of
    two separate ``_floor_div_pow`` calls."""
    kmax = 120 // 16                                        # = 7
    spec = _empty_spec(L.D, N_NIB * (3 + kmax * 2))
    u = 0
    for j in range(N_NIB):
        u = _clear(spec, u, L.FINE_LO + j)
        u = _clear(spec, u, L.FINE_CO + j)
        # low = prod  (read directly; the -16*floor is folded into the shared
        # staircase below, so low = prod - 16*floor(prod/16) = prod mod 16).
        u = alu._ident(spec, u, {L.PROD + j: 1.0}, 0.0, L.FINE_LO + j, 1.0)
        # ONE floor(prod/16) staircase -> FINE_CO += floor ; FINE_LO += -16*floor.
        u = alu._floor_div_pow2(spec, u, {L.PROD + j: 1.0}, 0.0, 16, kmax,
                                L.FINE_CO + j, 1.0, L.FINE_LO + j, -16.0)
    return _truncate(spec, u, L.D)


def build_assemble(L: "_TightLayout", left: bool) -> dict:
    """FINE step 3 + recompose: merge each nibble's peeled parts with the carry
    from its neighbour, gated on KEEP (n>=32 -> 0), into OUT_NIB.

    SHL (``p = coarse*2**r``, FINE_LO = p mod 16 stays, FINE_CO = floor(p/16)
    flows UP):
        OUT_NIB[j] = FINE_LO[j] + FINE_CO[j-1]

    SHR (``p = coarse*2**(4-r)`` for r>0, so FINE_CO = coarse>>r stays and
    FINE_LO = the r bits that fall DOWN into the nibble below; r=0 is the pure
    whole-nibble shift so OUT_NIB[j] = COARSE[j]):
        r != 0 (RNZ=1):  OUT_NIB[j] = FINE_CO[j] + FINE_LO[j+1]
        r == 0 (RNZ=0):  OUT_NIB[j] = COARSE[j]

    The two merged parts occupy DISJOINT bit ranges of the nibble (the low r bits
    vs the high 4-r bits) so their sum is the exact merged nibble < 16.  Gated on
    KEEP so an ``n >= 32`` shift leaves the result all-zero.  The SHR r==0 term is
    ``COARSE·KEEP - COARSE·KEEP·RNZ`` (a NOT via subtract)."""
    spec = _empty_spec(L.D, N_NIB * 8)
    u = 0
    if left:
        for j in range(N_NIB):
            u = _clear(spec, u, L.OUT_NIB + j)
            u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)], {L.FINE_LO + j: 1.0},
                           0.0, L.OUT_NIB + j, 1.0)
            nb = j - 1
            if 0 <= nb < N_NIB:
                u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)],
                               {L.FINE_CO + nb: 1.0}, 0.0, L.OUT_NIB + j, 1.0)
        return _truncate(spec, u, L.D)
    # SHR
    for j in range(N_NIB):
        u = _clear(spec, u, L.OUT_NIB + j)
        # r != 0: FINE_CO[j] + FINE_LO[j+1]   gated on KEEP AND RNZ.
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                       {L.FINE_CO + j: 1.0}, 0.0, L.OUT_NIB + j, 1.0)
        nb = j + 1
        if 0 <= nb < N_NIB:
            u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                           {L.FINE_LO + nb: 1.0}, 0.0, L.OUT_NIB + j, 1.0)
        # r == 0: COARSE[j] gated on KEEP AND NOT RNZ  (= KEEP - KEEP·RNZ).
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)], {L.COARSE + j: 1.0}, 0.0,
                       L.OUT_NIB + j, 1.0)
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                       {L.COARSE + j: 1.0}, 0.0, L.OUT_NIB + j, -1.0)
    return _truncate(spec, u, L.D)


# ===========================================================================
# The assembled tight shifter.
# ===========================================================================
def build_tight(op: int) -> Tuple[List[dict], "_TightLayout"]:
    """Assemble the tight direct-8x8 shifter for SHL/SHR.

    Blocks (a short pipeline; each block reads the previous block's outputs):
        1-2  amount decode (c, feq, POW, keep ; then ceq)
        3    DIRECT 8x8 coarse select
        4    fine product  (coarse * 2**r)
        5    fine peel      (low + carry)
        6    assemble       (merge carry + KEEP-gated recompose)
    """
    if op not in (isa.SHL, isa.SHR):
        raise ValueError(f"build_tight: op {op} not SHL/SHR")
    left = (op == isa.SHL)
    L = _TightLayout()
    _set_one(L.ONE)
    blocks: List[dict] = list(build_amount_blocks(left)(L))
    blocks.append(build_coarse_select(L, left))
    blocks.append(build_fine_product(L))
    blocks.append(build_fine_peel(L))
    blocks.append(build_assemble(L, left))
    return blocks, L


# ===========================================================================
# Forward + nz measurement.
# ===========================================================================
def _apply(x: torch.Tensor, w: Dict[str, torch.Tensor]) -> torch.Tensor:
    up = F.linear(x, w["W_up"]) + w["b_up"]
    gate = F.linear(x, w["W_gate"]) + w["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, w["W_down"], w["b_down"])


def run_blocks(blocks: List[dict], L: "_TightLayout", pop: int, n: int) -> int:
    x = L.load(pop, n)
    for w in blocks:
        x = _apply(x, w)
    return L.decode(x)


def _nz(w: Dict[str, torch.Tensor]) -> int:
    return sum(int((w[k] != 0).sum())
               for k in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"))


def _units(w: Dict[str, torch.Tensor]) -> int:
    return int(w["W_up"].shape[0])


def _blocks_nz(blocks: List[dict]) -> int:
    return sum(_nz(w) for w in blocks)


def _blocks_units(blocks: List[dict]) -> int:
    return sum(_units(w) for w in blocks)


def _max_relu_arg(blocks: List[dict]) -> float:
    m = 0.0
    for w in blocks:
        if w["b_up"].numel():
            m = max(m, float(w["b_up"].abs().max()))
        if w["b_gate"].numel():
            m = max(m, float(w["b_gate"].abs().max()))
    return m


def measure(op: int) -> dict:
    """Depth, nz (split coarse-select / fine / amount-decode), nz/unit, fp32-ok."""
    blocks, L = build_tight(op)
    # block indices: [amount1, amount2, coarse, fine_prod, fine_peel, assemble]
    amount = blocks[0:2]
    coarse = blocks[2:3]
    fine = blocks[3:6]
    return {
        "op": isa.NAMES[op],
        "depth": len(blocks),
        "nz_total": _blocks_nz(blocks),
        "nz_amount": _blocks_nz(amount),
        "nz_coarse": _blocks_nz(coarse),
        "nz_fine": _blocks_nz(fine),
        "units_total": _blocks_units(blocks),
        "nz_per_unit": _blocks_nz(blocks) / max(1, _blocks_units(blocks)),
        "max_arg": _max_relu_arg(blocks),
        "fp32_ok": _max_relu_arg(blocks) < FP32_INT_LIMIT,
    }


# ===========================================================================
# BYTE-EXACT VERIFICATION — the lean arithmetic sim (seconds, no DIM-8192 run).
# ===========================================================================
_EDGE_XS = [0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x1]
_EDGE_NS = [0, 1, 7, 15, 16, 31, 32, 40]


def ref_shift32(pop: int, n: int, left: bool) -> int:
    """32-bit reference: ``(pop <</>> n) & 0xFFFFFFFF`` with ``n`` UNMASKED, so
    ``n >= 32 -> 0`` (matches ref_interpret(mask=0xFFFFFFFF))."""
    pop &= MASK32
    if left:
        return (pop << n) & MASK32 if n < 32 else 0
    return (pop >> n) & MASK32 if n < 32 else 0


def check_ref_agreement(op: int, n_random: int = 200) -> Tuple[int, int]:
    """Confirm ``ref_shift32`` == the NAMED reference ``ref_interpret(mask=
    0xFFFFFFFF)`` on the BYTE-operand path a real ``PSH x ; SHL/SHR cnt`` program
    exercises (IMM loads a byte operand and a 0..255 count).  ``ref_interpret`` can
    only push a byte-bounded AX (IMM sets ``ax = imm & 0xFF``), so this pins the
    equivalence on that path; the full 32-bit operand grid uses ``ref_shift32``
    directly (proven equal by ``(v <</>> n) & MASK32`` construction).  Returns
    (agree, total)."""
    from .nibble_pure_forward_complete import ref_interpret
    rng = random.Random(0x5A + op)
    agree = total = 0
    for _ in range(n_random):
        x = rng.randint(0, 0xFF)
        cnt = rng.randint(0, 40)
        code = isa.assemble([("IMM", x), ("PSH", 0), ("IMM", cnt),
                             ("SHL" if op == isa.SHL else "SHR", 0)])
        ref = ref_interpret(code, mask=MASK32)[-1]
        mine = ref_shift32(x, cnt, op == isa.SHL)
        total += 1
        if ref == mine:
            agree += 1
    return agree, total


def byte_exact_32(op: int, n_random: int = 300) -> Tuple[int, int, List[str]]:
    """Run the gadget over the edge grid + random (x, n) vs the 32-bit reference
    ``ref_shift32`` (== ``ref_interpret(mask=0xFFFFFFFF)`` by construction, and the
    two agree on the corpus — asserted in ``check_ref_agreement``).  Covers ``n``
    up to 63 so both the ``n < 32`` and the ``n >= 32 -> 0`` folds are exercised.
    Returns (passed, total, fails)."""
    left = (op == isa.SHL)
    blocks, L = build_tight(op)
    rng = random.Random(0xC4 + op)
    cases: List[Tuple[int, int]] = [(x, n) for x in _EDGE_XS for n in _EDGE_NS]
    for _ in range(n_random):
        cases.append((rng.randint(0, MASK32), rng.randint(0, 0x3F)))
    passed = 0
    fails: List[str] = []
    for x, n in cases:
        want = ref_shift32(x, n, left)
        got = run_blocks(blocks, L, x, n)
        if got == want:
            passed += 1
        elif len(fails) < 12:
            fails.append(f"{isa.NAMES[op]} x={x:#010x} n={n}: got {got:#010x} want {want:#010x}")
    return passed, len(cases), fails


def byte_exact_isa8(op: int, n_random: int = 60) -> Tuple[int, int, List[str]]:
    """Cross-check the low byte vs the 8-bit reference ``isa.interpret``
    (IMM x, PSH, IMM cnt, SHL/SHR ; AX = shift count).  Confirms the tight
    gadget also matches the 8-bit ISA on byte-sized operands/counts."""
    blocks, L = build_tight(op)
    rng = random.Random(0x88 + op)
    passed = 0
    total = 0
    fails: List[str] = []
    for _ in range(n_random):
        x = rng.randint(0, 0xFF)
        cnt = rng.randint(0, 7)
        code = isa.assemble([("IMM", x), ("PSH", 0), ("IMM", cnt),
                             ("SHL" if op == isa.SHL else "SHR", 0)])
        want = isa.interpret(code)[-1] & 0xFF
        got = run_blocks(blocks, L, x, cnt) & 0xFF
        total += 1
        if got == want:
            passed += 1
        elif len(fails) < 12:
            fails.append(f"{isa.NAMES[op]}8 x={x:#04x} n={cnt}: got {got:#04x} want {want:#04x}")
    return passed, total, fails


# ===========================================================================
# Report — the `tight-direct-8x8` row next to the references.
# ===========================================================================
_NIBBLE_MUXTREE_NZ = 4140          # shifter_bakeoff nibble-granular SHL (measured).
_BIT_LOGSHIFTER_NZ = 5269          # nibble_bitwise landed bit-granular log-shifter.


def report() -> str:
    lines: List[str] = []
    rows: List[dict] = []
    for op in (isa.SHL, isa.SHR):
        m = measure(op)
        p32, t32, f32 = byte_exact_32(op)
        p8, t8, f8 = byte_exact_isa8(op)
        ar, at = check_ref_agreement(op)
        m["byte32"] = (p32, t32)
        m["byte8"] = (p8, t8)
        m["ref_agree"] = (ar, at)
        m["fails"] = f32 + f8
        rows.append(m)

    lines.append("TIGHT DIRECT-8x8 NIBBLE SHIFTER — measurement")
    lines.append("=" * 78)
    hdr = ("| variant            | op  | depth | weights | coarse | fine | amount "
           "| nz/unit | fp32 | byte-exact |")
    sep = ("|--------------------|-----|-------|---------|--------|------|--------"
           "|---------|------|------------|")
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        be = f"{r['byte32'][0]}/{r['byte32'][1]} 32b, {r['byte8'][0]}/{r['byte8'][1]} 8b"
        lines.append(
            f"| tight-direct-8x8   | {r['op']:<3} | {r['depth']:<5} | {r['nz_total']:<7} "
            f"| {r['nz_coarse']:<6} | {r['nz_fine']:<4} | {r['nz_amount']:<6} "
            f"| {r['nz_per_unit']:.2f}    | {'yes' if r['fp32_ok'] else 'NO':<4} | {be} |")
    # references
    lines.append(
        f"| nibble mux-tree    | SHL | 8     | {_NIBBLE_MUXTREE_NZ:<7} "
        f"| (log)  | peel | 64-scan| ~8.5    | yes  | (reference)         |")
    lines.append(
        f"| bit log-shifter    | SHL | 8     | {_BIT_LOGSHIFTER_NZ:<7} "
        f"| (log)  | -    | -      | -       | yes  | (reference)         |")
    lines.append("")
    for r in rows:
        under = _NIBBLE_MUXTREE_NZ - r["nz_total"]
        pct = 100.0 * r["nz_total"] / _NIBBLE_MUXTREE_NZ
        lines.append(
            f"{r['op']}: total {r['nz_total']} nz = coarse-select {r['nz_coarse']} "
            f"+ fine {r['nz_fine']} + amount-decode {r['nz_amount']}; "
            f"{r['nz_per_unit']:.2f} nz/unit over {r['units_total']} units; "
            f"{under} nz UNDER the 4140 mux-tree ({pct:.0f}% of it). "
            f"ref_interpret(mask=0xFFFFFFFF) agreement {r['ref_agree'][0]}/{r['ref_agree'][1]}.")
        if r["fails"]:
            lines.append("  FAILS: " + " ; ".join(r["fails"][:6]))
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
