"""Universal NIBBLE VM — the coherent BLOG_SPEC-faithful base.

ONE fixed-weight transformer step-block that runs ANY program supplied as DATA,
on the spec's nibble representation, emitting the 30-token register frame each
step, re-quantised by the vanilla autoregressive token round-trip (NO
``torch.round``).

It ports three proven scalar-track mechanisms onto the nibble foundation
(``blogspec_layout`` / ``blogspec_vocab`` / ``blogspec_model``):

  1. RUNTIME-PC DISPATCH  (``git show gf-assembled:.../control.py``)
        Each step reads the LIVE PC, builds the exact-integer PC one-hot
        ``PC_IS[i] == (PC == i)`` (triangular-pulse gadget), and the executed
        instruction is selected by that runtime one-hot — a forward branch that
        skips instructions Just Works because the next step dispatches on the new
        PC. Here PC is a NIBBLE band; ``compile_nibble_to_scalar`` recomposes it
        to the scalar ``PC_VAL`` the pulse operates on.

  2. UNIVERSAL FETCH-FROM-DATA  (``git show greenfield-universal:.../universal.py``)
        The program is NOT baked into the weights. It lives in DATA-MEMORY bands
        ``CODE_OP[i]`` / ``CODE_IMM[i]`` (written by ``load_program`` as INPUT).
        Fetch@PC does the bilinear select ``OP_VAL = Σ_i PC_IS[i]·CODE_OP[i]`` /
        ``IMM = Σ_i PC_IS[i]·CODE_IMM[i]`` (SwiGLU ``silu(up)·gate`` product). The
        opcode is DECODED to a one-hot ``OP_IS[op]`` and dispatch executes ONE
        fixed rule per opcode VALUE ⇒ ONE model, MANY programs (swap the data).

  3. VANILLA REQUANT  (``git show greenfield-vanilla-requant:.../recurrent_vanilla.py``)
        Between steps every register round-trips through the LM head:
        ``token = argmax_v (2·v·band − v²)`` (the integer snap, no round) then
        re-embed. On the nibble foundation this IS the 30-token register emission:
        each register byte is the argmax over 256 byte logits read from the scalar
        next-state lane, and the emitted byte token re-enters through the byte
        embedding — writing its two nibbles back into the register's NIBBLE band,
        annihilating the O(1e-6) SwiGLU residue and closing the scalar→nibble loop.

DISPATCH INTERFACE (the contract the fan-out agents target)
-----------------------------------------------------------
An opcode's effect is ONE ``FFNRule`` set gated on the DECODED opcode one-hot
``OP_IS[op]`` (see ``base_dispatch_rules``). It reads the current register value
lanes (``AX_VAL SP_VAL BP_VAL STK_VAL PC_VAL``) + the fetched ``IMM`` and writes
the next-state directly into those lanes (SET semantics: each op cancels the old
image) plus the PC delta. A new gadget (comparison / bitwise / muldiv / memory /
callconv) plugs in by APPENDING its own ``OP_IS[op]``-gated ``FFNRule`` set —
nothing else in the interpreter changes. See ``docs/NIBBLE_SKELETON_2026_07_14.md``.

The persistent VM value is the NIBBLE band; the scalar value lane is that band's
per-step image (recomposed by ``compile_nibble_to_scalar``) so the PROVEN exact
scalar dispatch algebra runs unchanged; the driver's frame round-trip writes the
scalar next-state back into the nibble bands.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import blogspec_vocab as V
from .nibble_vm_layout import NibbleVMLayout
from .blogspec_model import Transformer
from .dsl import FFNRule, LinearExpr


# --- exact-integer silu scales (shared with the proven gadgets) -------------
S = 60.0        # silu identity: silu(60)≈60, silu(-60)≈0 (fp32-exact on ints)
RELU_S = 200.0  # relu-via-silu: silu(RELU_S·z)/RELU_S ≈ relu(z), exact on ints
SILU_S = float(F.silu(torch.tensor(S)))          # ≈ 60.0
SILU_HALF = float(F.silu(torch.tensor(0.5 * S)))  # ≈ 30.0


# ---------------------------------------------------------------------------
# Value-lane WIDTH (Family-2 fix, #667).
#
# The folded corpus (``isa.interpret``, ``MASK=0xFF``) runs the whole scalar
# value substrate at **8 bits**: the recompose reads 5 nibbles, ADD/SUB carry a
# ``+256`` byte wrap, the ``fold`` block does ``AX mod 256``, LEA adds BP's LOW
# byte, and the requant snaps a single flat argmax over ``VALVOCAB`` (~0x10100).
# That makes loop counters and MOD operands **effectively 8-bit** — a
# ``while (i < n)`` with ``n > 255`` diverges from ideal C because the operand is
# truncated at the fold BEFORE the (already 32-bit-capable) compare ever sees it.
#
# ``C4_VM_WIDTH32=1`` widens the substrate to full 32 bits, exact (fp64 exec):
#   * recompose reads all 8 nibbles (16^7 needs the high bytes)   -> full value
#   * ADD/SUB drop the ``+256`` byte hack (SUB shift = 0); the requant wraps a
#     negative SUB result mod 2^32 (two's complement) directly — routing a 2^32
#     constant through the SwiGLU gate is NOT fp-exact even in fp64 (the
#     silu-identity's ~1e-9 relative error times 2^32 is ~4 units)
#   * the mod-256 ``fold`` becomes a NO-OP (a 2^32 ramp is not materialisable);
#     the requant carries the mod-2^32 wrap
#   * LEA uses the full BP value + imm
#   * the requant (``_snap_lane`` -> ``_snap_lane_bytes``) snaps to the nearest
#     integer via ``floor(x+½)`` (the round-free argmax equivalent — the AST guard
#     forbids the ``round`` builtin here), reduces to ``v mod 2^32`` (two's
#     complement, so a large loop counter / negative never aliases to zero), and
#     splits into 4 little-endian bytes — the §720 high-to-low cascade at the token
#     round-trip, exact to the full 2^32 with NO 2^32-wide vocab and NO fp64
#     nibble-pack (§590).  The model runs in fp64 so 16^7 recompose + requant are
#     exact past 2^24.  Unsigned 32-bit ordering (compare/branch) is exact.
# Default OFF -> byte-identical to the 8-bit folded corpus.  ON -> 32-bit.
# ---------------------------------------------------------------------------
def vm_width32() -> bool:
    """True iff the full-32-bit value substrate is enabled (``C4_VM_WIDTH32=1``)."""
    return os.environ.get("C4_VM_WIDTH32", "0") == "1"


# ---------------------------------------------------------------------------
# TWO-LIMB fp32 dispatch (kills the fp64 requirement of the width-32 substrate).
#
# The width-32 substrate historically ran the whole step in **fp64** for ONE
# reason: ``compile_nibble_to_scalar`` recomposed each register's 8 nibbles into a
# single scalar ``VAL = Σ 16^j·nib_j`` and ``16^7 = 2^28 > 2^24`` overflows fp32's
# integer precision, so the ~2^32-wide AX/STACK0 value lanes and the ADD/SUB
# algebra on them lost bits below fp64.  (PC < 4096 and SP/BP ≈ 65536 are small and
# fp32-fine — only the DATA values AX and STACK0 force fp64.)
#
# ``two_limb`` represents AX and STACK0 as TWO fp32-exact limbs instead:
#   ``*_LO`` = low 4 nibbles (bits 0..15) — a scalar ≤ 2^16-1 < 2^24 (fp32-exact).
#   ``*_HI`` = high 4 nibbles (bits 16..31) kept as their own small integer (≤ 2^16-1).
# ADD/SUB add/sub the limbs with a carry/borrow across the 2^16 boundary realised
# by fp32 relu steps at HALF-INTEGER thresholds (exact — the residue never crosses,
# and both the ``RELU_S·lo`` threshold arithmetic AND the ramp stay < 2^24).  The
# requant recomposes ``lo + 2^16·hi`` in Python int (exact) and wraps mod 2^32.
# NOTHING is fp64 — ``model_dtype == torch.float32``.  ``two_limb=False`` keeps the
# historical single-scalar fp64 path intact as a fallback.
#
# Both limbs are ALWAYS present (no data-dependent branch); for small values the
# HI limb is simply 0, so it degenerates to the plain fp32 scalar path for free.
# ---------------------------------------------------------------------------
# Low-limb width: 4 nibbles = bits 0..15, so the carry/borrow boundary is 2^16.
# BOTH limbs are then ≤ 2^16-1 = 65535 < 2^24, and — critically — the CARRY step's
# ``RELU_S·lo`` threshold arithmetic (scale 200) stays at ≈ 200·2^16 = 2^23.6 where
# the fp32 ulp (≈ 1) is far smaller than the 0.25-wide ramp height (RELU_S·W = 50),
# so the carry saturates to EXACTLY 1.  A 5-nibble low limb (boundary 2^20) FAILS
# here: ``RELU_S·2^20 ≈ 2^27.6`` has ulp ≈ 16, catastrophically cancelling the
# narrow ramp (carry came out ≈ 0.96, not 1).  The high limb takes the remaining 4
# nibbles (bits 16..31, ≤ 65535 < 2^24, also fp32-exact).
_LO_NIBBLES = 4
_LO_MOD = 1 << (4 * _LO_NIBBLES)     # 2^16  — low-limb / carry boundary
_HI_NIBBLES = 8 - _LO_NIBBLES        # 4 nibbles (bits 16..31)
_HI_MASK = (1 << (4 * _HI_NIBBLES)) - 1   # 0xFFFF


# A build/run may pin the limb mode for the WHOLE compile+drive (the shared
# recompose/dispatch/branch/requant functions all key off ``vm_two_limb()``, so a
# consistent model needs ONE decision).  The larger unified build (``nibble_unified``)
# has single-scalar CMP/MUL/DIV lanes NOT ported to two-limb, so it pins
# single-scalar; the base ``build_step_model`` uses the flag default.  ``None`` = use
# the ``vm_width32``/``C4_VM_TWO_LIMB`` default below.
_TWO_LIMB_OVERRIDE: "bool | None" = None


def vm_two_limb() -> bool:
    """True iff the fp32 TWO-LIMB AX/STACK0 dispatch is enabled.

    ON (default when width-32 is on) -> AX/STACK0 carried as ``*_LO``/``*_HI`` fp32
    limbs, the whole model runs in **fp32** (no fp64).  An in-process build context
    (``two_limb_mode``) can PIN it (the unified build pins single-scalar).  Explicit
    env override: ``C4_VM_TWO_LIMB=1``/``0``.  When width-32 is OFF the value lanes
    are 8-bit and two-limb is irrelevant (HI limb always 0), so it defaults OFF
    there — byte-identical to the folded 8-bit corpus.  ``C4_VM_TWO_LIMB=0`` with
    width-32 ON is the fp64 single-scalar FALLBACK (the historical path, intact)."""
    if _TWO_LIMB_OVERRIDE is not None:
        return _TWO_LIMB_OVERRIDE
    env = os.environ.get("C4_VM_TWO_LIMB")
    if env is not None:
        return env == "1"
    return vm_width32()          # default: two-limb whenever the substrate is 32-bit


import contextlib


@contextlib.contextmanager
def two_limb_mode(enabled: bool):
    """Pin ``vm_two_limb()`` to ``enabled`` for the whole build+drive inside the
    ``with`` block (all shared recompose/dispatch/requant functions read it).  Used
    so a model is compiled AND driven with ONE consistent limb decision."""
    global _TWO_LIMB_OVERRIDE
    prev = _TWO_LIMB_OVERRIDE
    _TWO_LIMB_OVERRIDE = enabled
    try:
        yield
    finally:
        _TWO_LIMB_OVERRIDE = prev


def _layout_two_limb(L) -> bool:
    """Whether the model built for layout ``L`` is a two-limb (fp32) build.

    A build STAMPS its decision onto the layout (``L.two_limb``) so the recurrent
    driver / requant honour how the model was actually compiled, regardless of the
    ambient flag at run time (the base ``build_step_model`` stamps the flag default;
    the single-scalar unified build stamps ``False``).  Falls back to the ambient
    flag for layouts built before this stamp existed."""
    v = getattr(L, "two_limb", None)
    return vm_two_limb() if v is None else bool(v)


def _recompose_hi_nibbles() -> int:
    """Nibbles the recompose reads into a value lane: 8 (32-bit) when width-32,
    else 5 (the 8-bit-substrate foundation range)."""
    return 8 if vm_width32() else 5


def _fold_modulus() -> int:
    """The AX value-lane fold modulus: 2^32 (a no-op ramp, requant carries the
    wrap) under width-32, else 256 (the 8-bit fold)."""
    return (1 << 32) if vm_width32() else 256

# The opcode VALUES the interpreter decodes + dispatches (the built base subset).
BASE_OPS = [isa.IMM, isa.LEA, isa.PSH, isa.ADD, isa.SUB,
            isa.JMP, isa.BZ, isa.BNZ, isa.HALT]


# ---------------------------------------------------------------------------
# FFN spec container + the generic FFNRule compiler (SwiGLU block).
# ---------------------------------------------------------------------------
def _empty_spec(dim: int, n_units: int) -> Dict[str, torch.Tensor]:
    return {
        "W_up": torch.zeros(n_units, dim), "b_up": torch.zeros(n_units),
        "W_gate": torch.zeros(n_units, dim), "b_gate": torch.zeros(n_units),
        "W_down": torch.zeros(dim, n_units), "b_down": torch.zeros(dim),
    }


def compile_ffn(rules: List[FFNRule], dim: int) -> Dict[str, torch.Tensor]:
    """FFNRule list -> SwiGLU tensors (exact-integer gadget, from compile_ffn.py).

    Per (rule, dst): one hidden unit with gate = write-expr, up = S·(guard
    indicator), down = 1/silu(0.5S) routing ``guard·expr`` into dst.
    """
    n_units = sum(max(1, len(r.write)) for r in rules) or 1
    spec = _empty_spec(dim, n_units)
    u = 0
    for rule in rules:
        n_win = len(rule.when)
        for dst, expr in rule.write.items():
            for band, coeff in expr.terms.items():
                spec["W_gate"][u, band] += coeff
            spec["b_gate"][u] += expr.const
            for (band, lo, hi) in rule.when:
                spec["W_up"][u, band] += S
            spec["b_up"][u] += -S * (n_win - 0.5)
            spec["W_down"][dst, u] += 1.0 / SILU_HALF
            u += 1
    return spec


# ---------------------------------------------------------------------------
# (1) NIBBLE -> SCALAR recompose. value_lane = Σ_j 16^j · nibble_j.
# ---------------------------------------------------------------------------
def compile_nibble_to_scalar(L: NibbleVMLayout, dim: int,
                             hi_nibbles: int = None) -> Dict[str, torch.Tensor]:
    """Recompose each register's nibble band into its scalar value lane (SET).

    ``VAL = Σ_{j<hi_nibbles} 16^j · nibble_j`` via silu-identity reads
    (``silu(S·nib_j)/S ≈ nib_j``, exact for nibbles 0..15) weighted by ``16^j`` in
    the down-projection. Default ``hi_nibbles`` = 8 under ``C4_VM_WIDTH32`` (full
    32-bit): ``16^7`` exceeds fp32's 2^24 unit precision, so the width-32 model runs
    in fp64 (2^32 < 2^53) where the recompose is exact and the requant re-snaps the
    lane to the exact integer.  Else 5 (the 8-bit-substrate foundation range: 8-bit
    AX/STACK0, small PC, SP/BP = 0x10000 at nibble 4; ``16^4 = 65536 < 2^24``
    fp32-exact).

    This is the bridge that lets the PROVEN scalar dispatch algebra run on the
    spec's canonical nibble state: the nibble band is canonical; the scalar lane
    is its per-step image.
    """
    if vm_two_limb():
        return _compile_nibble_to_scalar_two_limb(L, dim)
    if hi_nibbles is None:
        hi_nibbles = _recompose_hi_nibbles()
    # (nibble_base, value_lane, n_read): full registers read hi_nibbles; BP_LOW
    # reads only nibbles 0,1 (the frame-pointer's low byte for the 8-bit LEA).
    pairs = [(L.PC, L.PC_VAL, hi_nibbles), (L.AX, L.AX_VAL, hi_nibbles),
             (L.SP, L.SP_VAL, hi_nibbles), (L.BP, L.BP_VAL, hi_nibbles),
             (L.STACK0, L.STK_VAL, hi_nibbles), (L.BP, L.BP_LOW, 2)]
    n_units = sum(n + 1 for _, _, n in pairs)    # per lane: 1 self-clear + n reads
    spec = _empty_spec(dim, n_units)
    u = 0
    for reg_base, val_lane, n_read in pairs:
        spec["W_up"][u, L.ONE] = S               # self-clear the old lane (SET)
        spec["W_gate"][u, val_lane] = 1.0
        spec["W_down"][val_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(n_read):
            spec["W_up"][u, L.ONE] = S            # gate passes nibble via silu-id
            spec["W_gate"][u, reg_base + j] = 1.0
            spec["W_down"][val_lane, u] += (16.0 ** j) / SILU_S
            u += 1
    return spec


def _compile_nibble_to_scalar_two_limb(L: NibbleVMLayout, dim: int
                                       ) -> Dict[str, torch.Tensor]:
    """TWO-LIMB recompose (fp32-exact 32-bit).  PC/SP/BP recompose to their full
    single scalar (small, fp32-fine: PC < 4096, SP/BP ≈ 65536 at nibble 4).  The
    DATA registers AX and STACK0 recompose to two limbs each:

      ``*_LO = Σ_{j<4} 16^j·nib_j``          (bits 0..15, ≤ 2^16-1 < 2^24)
      ``*_HI = Σ_{j=4,5,6,7} 16^(j-4)·nib_j``  (bits 16..31 as a 0..2^16-1 integer)

    Every read is the same silu-identity nibble pass (``silu(S·nib)/S ≈ nib``,
    exact for nibbles 0..15) the single-scalar recompose uses; the low limb caps
    its coefficient at ``16^3`` and the high limb re-bases at ``16^0`` so BOTH lanes
    stay ≤ 2^16-1 — every product and sum is an integer < 2^24, fp32-exact.  The
    16^7 term that forced fp64 never appears.  SET (self-clears keep it idempotent
    across recurrent steps)."""
    # PC/SP/BP: full single scalar (≤ 5 nibbles is plenty; these are small).
    scalar_pairs = [(L.PC, L.PC_VAL, 5), (L.SP, L.SP_VAL, 5), (L.BP, L.BP_VAL, 5),
                    (L.BP, L.BP_LOW, 2)]
    # AX/STACK0: (nibble_base, LO_lane, HI_lane).
    limb_pairs = [(L.AX, L.AX_LO, L.AX_HI), (L.STACK0, L.STK_LO, L.STK_HI)]
    n_units = (sum(n + 1 for _, _, n in scalar_pairs)
               + sum((1 + _LO_NIBBLES) + (1 + (8 - _LO_NIBBLES)) for _ in limb_pairs))
    spec = _empty_spec(dim, n_units)
    u = 0
    for reg_base, val_lane, n_read in scalar_pairs:
        spec["W_up"][u, L.ONE] = S               # self-clear (SET)
        spec["W_gate"][u, val_lane] = 1.0
        spec["W_down"][val_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(n_read):
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, reg_base + j] = 1.0
            spec["W_down"][val_lane, u] += (16.0 ** j) / SILU_S
            u += 1
    for reg_base, lo_lane, hi_lane in limb_pairs:
        # LOW limb: nibbles 0.._LO_NIBBLES-1 at coeff 16^j (top 16^4 = 65536).
        spec["W_up"][u, L.ONE] = S               # self-clear LO (SET)
        spec["W_gate"][u, lo_lane] = 1.0
        spec["W_down"][lo_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(_LO_NIBBLES):
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, reg_base + j] = 1.0
            spec["W_down"][lo_lane, u] += (16.0 ** j) / SILU_S
            u += 1
        # HIGH limb: nibbles _LO_NIBBLES..7 RE-BASED to coeff 16^(j-_LO_NIBBLES).
        spec["W_up"][u, L.ONE] = S               # self-clear HI (SET)
        spec["W_gate"][u, hi_lane] = 1.0
        spec["W_down"][hi_lane, u] += -1.0 / SILU_S
        u += 1
        for j in range(_LO_NIBBLES, 8):
            spec["W_up"][u, L.ONE] = S
            spec["W_gate"][u, reg_base + j] = 1.0
            spec["W_down"][hi_lane, u] += (16.0 ** (j - _LO_NIBBLES)) / SILU_S
            u += 1
    return spec


# ---------------------------------------------------------------------------
# (2) FETCH: PC one-hot + AX_ZERO (block a); code-select from DATA (block b).
# ---------------------------------------------------------------------------
def compile_pc_fetch(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block (a): PC one-hot ``PC_IS[i] = (PC_VAL == i)`` + ``AX_ZERO = (AX==0)``.

    PC one-hot is the exact-integer triangular pulse
    ``tri_i(x) = relu(x-(i-1)) - 2·relu(x-i) + relu(x-(i+1))`` (1 at x==i, 0 at
    every other integer), sharing one relu unit per threshold. ``AX_ZERO =
    relu(1 - AX_VAL)`` (1 iff AX==0). SETs (self-clears keep them idempotent
    across recurrent steps). Ported from ``control.compile_pc_fetch``.
    """
    n = L.code_size
    clear_bands = list(L.PC_IS) + [L.AX_ZERO]
    thresholds = list(range(-1, n + 1))
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_pc_relu = len(thresholds)
    az_unit = n_pc_relu
    clear0 = az_unit + 1
    n_units = clear0 + len(clear_bands)
    spec = _empty_spec(dim, n_units)

    for t, j in thr_unit.items():
        spec["W_up"][j, L.PC_VAL] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    # AX_ZERO = (AX == 0).  Single-scalar: relu(1 - AX_VAL).  Two-limb: relu(1 -
    # AX_LO - AX_HI) — both limbs are non-negative integers, so their SUM is 0 iff
    # both are 0, and the sum (≤ 2·(2^16-1) < 2^24) is fp32-exact.
    if vm_two_limb():
        spec["W_up"][az_unit, L.AX_LO] = -RELU_S
        spec["W_up"][az_unit, L.AX_HI] = -RELU_S
    else:
        spec["W_up"][az_unit, L.AX_VAL] = -RELU_S
    spec["b_up"][az_unit] = RELU_S * 1.0
    spec["W_gate"][az_unit, L.ONE] = 1.0
    for c, band in enumerate(clear_bands):
        uu = clear0 + c
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, band] = 1.0
        spec["W_down"][band, uu] += -1.0 / SILU_S
    for i, band in enumerate(L.PC_IS):
        spec["W_down"][band, thr_unit[i - 1]] += 1.0 / RELU_S
        spec["W_down"][band, thr_unit[i]] += -2.0 / RELU_S
        spec["W_down"][band, thr_unit[i + 1]] += 1.0 / RELU_S
    spec["W_down"][L.AX_ZERO, az_unit] += 1.0 / RELU_S
    return spec


def compile_code_select(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Block (b): fetch the code cell at PC out of DATA MEMORY.

        OP_VAL = Σ_i PC_IS[i]·CODE_OP[i]      IMM = Σ_i PC_IS[i]·CODE_IMM[i]

    Bilinear product-select ``silu(S·PC_IS[i])·CODE_x[i]/silu(S)``: passes
    ``CODE_x[i]`` iff PC==i (silu(S)=S, silu(0)=0). The weights only know "PC
    one-hot times whatever data band" — the data IS the program. Ported from
    ``universal.compile_code_select``.
    """
    n = L.code_size
    two_limb = vm_two_limb()
    # OP_VAL select (n) + IMM select (n, or 2n split into LO/HI limbs) + self-clears.
    imm_selects = [(L.CODE_IMM_LO, L.IMM_LO), (L.CODE_IMM_HI, L.IMM_HI)] if two_limb \
        else [(L.CODE_IMM, L.IMM)]
    clear_bands = [L.OP_VAL] + [dst for _, dst in imm_selects]
    n_units = n + n * len(imm_selects) + len(clear_bands)
    spec = _empty_spec(dim, n_units)

    def product(u, sel_band, data_band, dst_band):
        spec["W_up"][u, sel_band] = S
        spec["W_gate"][u, data_band] = 1.0
        spec["W_down"][dst_band, u] += 1.0 / SILU_S

    u = 0
    for i in range(n):
        product(u, L.PC_IS[i], L.CODE_OP[i], L.OP_VAL); u += 1
    for code_band, dst in imm_selects:           # IMM (single scalar or LO/HI limbs)
        for i in range(n):
            product(u, L.PC_IS[i], code_band[i], dst); u += 1
    for band in clear_bands:                      # self-clear (SET)
        spec["W_up"][u, L.ONE] = S
        spec["W_gate"][u, band] = 1.0
        spec["W_down"][band, u] += -1.0 / SILU_S
        u += 1
    return spec


# ---------------------------------------------------------------------------
# (3) DECODE: OP_VAL scalar -> opcode one-hot OP_IS[op].
# ---------------------------------------------------------------------------
def compile_opcode_decode(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """``OP_IS[op] = (OP_VAL == op)`` — the SAME triangular-pulse gadget as the PC
    one-hot, applied to the fetched ``OP_VAL``. Turns the dynamically-fetched
    scalar opcode into the one-hot the dispatch gates on. Only the decoded VALUES
    in ``BASE_OPS`` are materialised (others stay 0). Ported from
    ``universal.compile_opcode_decode``."""
    ops = sorted(BASE_OPS)
    thresholds = sorted({t for op in ops for t in (op - 1, op, op + 1)})
    thr_unit = {t: j for j, t in enumerate(thresholds)}
    n_relu = len(thresholds)
    n_units = n_relu + len(ops)
    spec = _empty_spec(dim, n_units)

    for t, j in thr_unit.items():
        spec["W_up"][j, L.OP_VAL] = RELU_S
        spec["b_up"][j] = -RELU_S * t
        spec["W_gate"][j, L.ONE] = 1.0
    clear0 = n_relu
    for c, op in enumerate(ops):                  # self-clear each OP_IS lane
        uu = clear0 + c
        spec["W_up"][uu, L.ONE] = S
        spec["W_gate"][uu, L.OP_IS + op] = 1.0
        spec["W_down"][L.OP_IS + op, uu] += -1.0 / SILU_S
    for op in ops:
        band = L.OP_IS + op
        spec["W_down"][band, thr_unit[op - 1]] += 1.0 / RELU_S
        spec["W_down"][band, thr_unit[op]] += -2.0 / RELU_S
        spec["W_down"][band, thr_unit[op + 1]] += 1.0 / RELU_S
    return spec


# ===========================================================================
# THE DISPATCH INTERFACE — one FFNRule set per OPCODE VALUE, gated on OP_IS[op].
# Reads the current value lanes + fetched IMM, writes them (SET) + PC delta.
# A fan-out gadget plugs in by APPENDING its own OP_IS[op]-gated rules.
# ===========================================================================
def base_dispatch_rules(L: NibbleVMLayout) -> List[FFNRule]:
    """The base-op transition table on the value lanes (§dispatch interface).

    Every rule is gated on the DECODED opcode one-hot ``OP_IS[op]`` (universal —
    not a baked PC position), reads the pre-op value lanes + fetched ``IMM``, and
    writes the next-state directly into the value lanes (SET semantics: the write
    includes ``-old`` where it replaces, so the additive residual lands on exactly
    the new value). PC is updated as a DELTA (``+1`` sequential, ``IMM-PC`` for
    JMP; BZ/BNZ deferred to the bilinear ``compile_branch_delta``).
    """
    if vm_two_limb():
        return _base_dispatch_rules_two_limb(L)
    ax, sp, bp, stk, pc = L.AX_VAL, L.SP_VAL, L.BP_VAL, L.STK_VAL, L.PC_VAL
    IMM = L.IMM
    # Width-32: NO non-negativity shift — the per-byte requant wraps a negative
    # SUB result mod 2^32 (two's complement) directly, and routing a 2^32 constant
    # through the SwiGLU gate is NOT fp-exact (the silu-identity's ~1e-9 relative
    # error times 2^32 is ~4 units).  LEA uses the full BP value, not just its low
    # byte.  8-bit: +256 wrap (the requant there is the flat non-negative argmax)
    # + BP low byte (the folded-corpus behaviour).
    w32 = vm_width32()
    SUB_SHIFT = 0.0 if w32 else 256.0
    lea_bp = LinearExpr.of(bp, 1.0) if w32 else LinearExpr.of(L.BP_LOW, 1.0)

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]         # fires iff decoded opcode == op

    rules: List[FFNRule] = []
    # IMM: AX = imm ; PC += 1
    rules.append(FFNRule(G(isa.IMM), {
        ax: LinearExpr.of(IMM, 1.0) + LinearExpr.of(ax, -1.0), pc: LinearExpr.c(1.0)}))
    # LEA: AX = (BP + imm) ; PC += 1.  8-bit: BP LOW byte + fold; 32-bit: full BP.
    rules.append(FFNRule(G(isa.LEA), {
        ax: lea_bp + LinearExpr.of(IMM, 1.0) + LinearExpr.of(ax, -1.0),
        pc: LinearExpr.c(1.0)}))
    # PSH: STACK0 = AX ; SP -= 4 ; PC += 1
    rules.append(FFNRule(G(isa.PSH), {
        stk: LinearExpr.of(ax, 1.0) + LinearExpr.of(stk, -1.0),
        sp: LinearExpr.c(-4.0), pc: LinearExpr.c(1.0)}))
    # ADD: AX = STACK0 + AX ; SP += 4 ; PC += 1   (per-byte requant wraps mod 2^W)
    rules.append(FFNRule(G(isa.ADD), {
        ax: LinearExpr.of(stk, 1.0), sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # SUB: AX = STACK0 - AX == stk - 2·AX + SHIFT ; SP += 4 ; PC += 1.  The SHIFT
    # keeps the lane non-negative for the requant; the requant wraps mod 2^W so the
    # extra SHIFT (a multiple of 2^W) vanishes.  8-bit SHIFT=256, 32-bit=2^32.
    rules.append(FFNRule(G(isa.SUB), {
        ax: LinearExpr.of(stk, 1.0) + LinearExpr.of(ax, -2.0) + LinearExpr.c(SUB_SHIFT),
        sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # JMP: PC = imm == PC += (imm - PC)
    rules.append(FFNRule(G(isa.JMP), {
        pc: LinearExpr.of(IMM, 1.0) + LinearExpr.of(pc, -1.0)}))
    # BZ / BNZ: PC update deferred to compile_branch_delta (bilinear). No write.
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    # HALT / EXIT: latch HALTED, freeze PC (no PC write == PC += 0).
    rules.append(FFNRule(G(isa.HALT), {L.HALTED: LinearExpr.c(1.0)}))
    return rules


def _base_dispatch_rules_two_limb(L: NibbleVMLayout) -> List[FFNRule]:
    """TWO-LIMB base dispatch (fp32-exact 32-bit).  The DATA registers AX and
    STACK0 are carried as ``*_LO``/``*_HI`` fp32 limbs; every op writes the RAW
    per-limb sum/difference and the CARRY/BORROW across the 2^16 limb boundary is
    normalised by the separate ``compile_limb_normalize`` block that runs right
    after dispatch (the non-linear relu carry step can't live in an additive
    ``LinearExpr``).  PC/SP/BP stay single scalars (small).  ``IMM_LO/IMM_HI`` are
    the fetched immediate's two limbs (value literals); PC-target immediates
    (JMP/branch) are small and live entirely in ``IMM_LO``.

    SET semantics per limb (the lane holds the OLD value):
      IMM: ``AX_LO = IMM_LO``  == write ``IMM_LO - AX_LO``     (replace)
      ADD: ``AX_LO = AX_LO + STK_LO`` == write ``+STK_LO``     (accumulate)
      SUB: ``AX_LO = STK_LO - AX_LO`` == write ``STK_LO - 2·AX_LO``  (a+ (s-2a)=s-a)
    All limb sums/diffs land in ``[-2^16, 2^17)`` (fp32-exact); normalize folds them
    back into ``[0, 2^16)`` + the carried high limb.
    """
    ax_lo, ax_hi = L.AX_LO, L.AX_HI
    stk_lo, stk_hi = L.STK_LO, L.STK_HI
    sp, bp, pc = L.SP_VAL, L.BP_VAL, L.PC_VAL
    imm_lo, imm_hi = L.IMM_LO, L.IMM_HI

    def G(op):
        return [(L.OP_IS + op, 0.5, 1.5)]

    rules: List[FFNRule] = []
    # IMM: AX = imm  (both limbs replaced) ; PC += 1
    rules.append(FFNRule(G(isa.IMM), {
        ax_lo: LinearExpr.of(imm_lo, 1.0) + LinearExpr.of(ax_lo, -1.0),
        ax_hi: LinearExpr.of(imm_hi, 1.0) + LinearExpr.of(ax_hi, -1.0),
        pc: LinearExpr.c(1.0)}))
    # LEA: AX = BP + imm.  BP is a small full scalar (≈ 0x10000); it lands in the LOW
    # limb (BP=2^16 sits right at the boundary, so normalize carries the bit-16 up),
    # IMM's high limb passes through, and normalize folds LO->HI.  (SET: replace the
    # old AX limbs, then add BP + IMM.)
    rules.append(FFNRule(G(isa.LEA), {
        ax_lo: LinearExpr.of(bp, 1.0) + LinearExpr.of(imm_lo, 1.0) + LinearExpr.of(ax_lo, -1.0),
        ax_hi: LinearExpr.of(imm_hi, 1.0) + LinearExpr.of(ax_hi, -1.0),
        pc: LinearExpr.c(1.0)}))
    # PSH: STACK0 = AX  (copy both limbs) ; SP -= 4 ; PC += 1
    rules.append(FFNRule(G(isa.PSH), {
        stk_lo: LinearExpr.of(ax_lo, 1.0) + LinearExpr.of(stk_lo, -1.0),
        stk_hi: LinearExpr.of(ax_hi, 1.0) + LinearExpr.of(stk_hi, -1.0),
        sp: LinearExpr.c(-4.0), pc: LinearExpr.c(1.0)}))
    # ADD: AX = STACK0 + AX  (per-limb accumulate; normalize carries) ; SP += 4 ; PC += 1
    rules.append(FFNRule(G(isa.ADD), {
        ax_lo: LinearExpr.of(stk_lo, 1.0), ax_hi: LinearExpr.of(stk_hi, 1.0),
        sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # SUB: AX = STACK0 - AX  (per-limb, ax+  (stk-2ax) = stk-ax; normalize borrows) ;
    # SP += 4 ; PC += 1.  No 2^32 shift — the two's-complement wrap is done by the
    # high-limb borrow chain + the requant mod-2^32, so no fp-inexact big constant.
    rules.append(FFNRule(G(isa.SUB), {
        ax_lo: LinearExpr.of(stk_lo, 1.0) + LinearExpr.of(ax_lo, -2.0),
        ax_hi: LinearExpr.of(stk_hi, 1.0) + LinearExpr.of(ax_hi, -2.0),
        sp: LinearExpr.c(4.0), pc: LinearExpr.c(1.0)}))
    # JMP: PC = imm  (PC target is small -> IMM_LO) == PC += (IMM_LO - PC)
    rules.append(FFNRule(G(isa.JMP), {
        pc: LinearExpr.of(imm_lo, 1.0) + LinearExpr.of(pc, -1.0)}))
    # BZ / BNZ: PC update deferred to compile_branch_delta (bilinear). No write.
    rules.append(FFNRule(G(isa.BZ), {}))
    rules.append(FFNRule(G(isa.BNZ), {}))
    # HALT / EXIT: latch HALTED, freeze PC.
    rules.append(FFNRule(G(isa.HALT), {L.HALTED: LinearExpr.c(1.0)}))
    return rules


def compile_limb_normalize(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Normalise the two-limb AX after dispatch: fold the raw LOW-limb sum/diff back
    into ``[0, 2^16)`` and propagate the CARRY / BORROW into the HIGH limb.

    After dispatch the LOW limb holds an INTEGER + a small silu-identity residue
    (|res| < 0.5) in ``[-2^16, 2^17)``:
      * ADD / LEA  -> ``lo ∈ [0, 2^17)``; if the integer part ``≥ 2^16`` CARRY 1 up.
      * SUB        -> ``lo ∈ (-2^16, 2^16)``; if the integer part ``< 0`` BORROW 1 down.
    The carry/borrow are SHARP UNIT STEPS placed at HALF-INTEGER thresholds (so the
    ±0.5 residue never crosses the boundary), realised as the difference of two
    ReLU-via-scale units divided by the ramp width:

        carry  = [lo ≥ 2^16-0.5]  (ramp 2^16-0.625 .. 2^16-0.375)   lo -= 2^16·carry ; hi += carry
        borrow = [lo ≤ -0.5]      (ramp on -lo, 0.375 .. 0.625)      lo += 2^16·borrow ; hi -= borrow

    The ramp ENDPOINTS are exact multiples of 0.125 = 2^-3, and the CARRY step's
    ``RELU_S·lo`` threshold arithmetic stays at ≈ 200·2^16 = 2^23.6 where the fp32 ulp
    (≈ 1) is far below the ramp height (RELU_S·W = 50), so the step is fp32-exact and
    saturates to EXACTLY 1.  (A 5-nibble low limb — boundary 2^20 — FAILS: ``RELU_S·
    2^20 ≈ 2^27.6`` has ulp ≈ 16, cancelling the narrow ramp → carry ≈ 0.96.)  A plain
    integer-threshold ``relu(x-(t-1))-relu(x-t)`` step would also fail: dispatch leaves
    a fractional residue and that ramp returns the *fractional* residue itself (not 0)
    near the boundary.  The HIGH limb is left in ``(-2^16, 2^17)`` (a single carry/
    borrow can push it just past its 4-nibble range or negative); the requant
    reconstructs ``lo + 2^16·hi`` in Python int and wraps mod 2^32, so the high-limb
    overflow / underflow IS the 32-bit two's-complement wrap.  Only AX is normalised
    (STACK0 is only ever COPIED from an already-normal AX)."""
    M = _LO_MOD                                   # 2^16
    W = 0.25                                       # ramp width (2 fp32 ulps at 2^16)
    lo, hi = L.AX_LO, L.AX_HI
    spec = _empty_spec(dim, 4)

    def sharp_step(u, band, sign, thr):
        # step ≈ [sign·band ≥ thr], a sharp ramp of width W centred on thr; endpoints
        # thr∓W/2 are exact fp32 multiples of 0.125.  Two hidden units at u, u+1;
        # step = (relu(z-(thr-W/2)) - relu(z-(thr+W/2)))/W with z = sign·band.
        a, b = thr - W / 2, thr + W / 2
        spec["W_up"][u, band] = RELU_S * sign
        spec["b_up"][u] = -RELU_S * a
        spec["W_gate"][u, L.ONE] = 1.0
        spec["W_up"][u + 1, band] = RELU_S * sign
        spec["b_up"][u + 1] = -RELU_S * b
        spec["W_gate"][u + 1, L.ONE] = 1.0
        return u + 2

    # CARRY step [lo ≥ M-0.5]: units 0,1.  carry = (relu(...) - relu(...))/(RELU_S·W).
    sharp_step(0, lo, +1.0, M - 0.5)
    # BORROW step [-lo ≥ 0.5] i.e. lo ≤ -0.5: units 2,3.
    sharp_step(2, lo, -1.0, 0.5)

    # lo -= M·carry ; hi += carry     (carry = (u0 - u1)/(RELU_S·W))
    spec["W_down"][lo, 0] += -M / (RELU_S * W)
    spec["W_down"][lo, 1] += +M / (RELU_S * W)
    spec["W_down"][hi, 0] += +1.0 / (RELU_S * W)
    spec["W_down"][hi, 1] += -1.0 / (RELU_S * W)
    # lo += M·borrow ; hi -= borrow   (borrow = (u2 - u3)/(RELU_S·W))
    spec["W_down"][lo, 2] += +M / (RELU_S * W)
    spec["W_down"][lo, 3] += -M / (RELU_S * W)
    spec["W_down"][hi, 2] += -1.0 / (RELU_S * W)
    spec["W_down"][hi, 3] += +1.0 / (RELU_S * W)
    return spec


def compile_branch_delta(L: NibbleVMLayout, dim: int) -> Dict[str, torch.Tensor]:
    """The BILINEAR PC update for BZ / BNZ (runs after dispatch; PC untouched).

        BZ  taken (AX_ZERO==1):  PC := IMM   -> delta = IMM - pc_pre
        BZ  not     (AX_ZERO==0): PC := pc_pre+1 -> delta = 1
        BNZ taken (AX_ZERO==0):  PC := IMM   -> delta = IMM - pc_pre
        BNZ not     (AX_ZERO==1): PC := pc_pre+1 -> delta = 1

    Four SwiGLU product units: ``up = BIG·(OP_IS[op] + TAKEN − 1.5)`` fires iff
    the decoded op-hot AND the boolean both hold (each 0/1); ``gate`` carries the
    delta value. Ported from ``universal.compile_branch_delta``.
    """
    # The branch TARGET is a small PC index (< code_size).  Single-scalar: the full
    # IMM lane.  Two-limb: it lives entirely in IMM_LO (IMM_HI == 0 for a PC target).
    imm = L.IMM_LO if vm_two_limb() else L.IMM
    pc, azero, one = L.PC_VAL, L.AX_ZERO, L.ONE
    BZ, BNZ = L.OP_IS + isa.BZ, L.OP_IS + isa.BNZ
    spec = _empty_spec(dim, 4)
    BIG = 200.0
    silu_big = float(F.silu(torch.tensor(0.5 * BIG)))

    def _and_unit(u, op_band, bool_terms, gate_terms, gate_const):
        spec["W_up"][u, op_band] += BIG
        for band, coeff, const in bool_terms:
            if band is not None:
                spec["W_up"][u, band] += BIG * coeff
            spec["b_up"][u] += BIG * const
        spec["b_up"][u] += -BIG * 1.5
        for band, coeff in gate_terms:
            spec["W_gate"][u, band] += coeff
        spec["b_gate"][u] += gate_const

    _and_unit(0, BZ,  [(azero, 1.0, 0.0)],  [(imm, 1.0), (pc, -1.0)], 0.0)
    spec["W_down"][pc, 0] += 1.0 / silu_big
    _and_unit(1, BZ,  [(azero, -1.0, 1.0)], [(one, 1.0)], 0.0)
    spec["W_down"][pc, 1] += 1.0 / silu_big
    _and_unit(2, BNZ, [(azero, -1.0, 1.0)], [(imm, 1.0), (pc, -1.0)], 0.0)
    spec["W_down"][pc, 2] += 1.0 / silu_big
    _and_unit(3, BNZ, [(azero, 1.0, 0.0)],  [(one, 1.0)], 0.0)
    spec["W_down"][pc, 3] += 1.0 / silu_big
    return spec


def compile_fold(band: int, one_band: int, dim: int, modulus: int = None
                 ) -> Dict[str, torch.Tensor]:
    """Exact mod-``modulus`` fold on ``band`` in [0, 2M): ``band -= M·(band>=M)``,
    a sharp unit ramp at M-0.5. Ported from ``compile_ffn.compile_fold``.

    Under ``C4_VM_WIDTH32`` the modulus is 2^32; the fp32 ramp at ``2^32-0.5``
    cannot be materialised (it exceeds fp32 unit precision), so this block becomes
    a NO-OP and the mod-2^32 wrap is done by the PER-BYTE requant round-trip
    (``_emit_and_reembed`` snaps each byte mod 256 -> the whole value mod 2^32).
    """
    if modulus is None:
        modulus = _fold_modulus()
    if modulus >= (1 << 32):
        return _empty_spec(dim, 1)                 # no-op; requant carries the wrap
    M, w = modulus, 0.2
    lo = M - 0.5
    spec = _empty_spec(dim, 2)
    for i, thr in enumerate((lo, lo + w)):
        spec["W_up"][i, band] = RELU_S
        spec["b_up"][i] = -RELU_S * thr
        spec["W_gate"][i, one_band] = 1.0
    spec["W_down"][band, 0] = -M / (RELU_S * w)
    spec["W_down"][band, 1] = +M / (RELU_S * w)
    return spec


# ===========================================================================
# BUILD the universal step-block (the interpreter — program INDEPENDENT).
# ===========================================================================
def build_step_model(code_size: int, n_heads: int = 4):
    """Bake ONE universal nibble VM step-block. The weights are program-
    INDEPENDENT — ``code_size`` only sizes the PC/data bands.

    Physical FFN sub-blocks (single residual position, attention zeroed=identity):
      1. recompose : nibble bands -> scalar value lanes (§the bridge).
      2. fetch     : PC_VAL -> PC_IS[i] one-hot + AX_ZERO predicate.
      3. code_sel  : fetch OP_VAL/IMM at PC from DATA MEMORY (product-select).
      4. decode    : OP_VAL scalar -> OP_IS[op] decoded opcode one-hot.
      5. dispatch  : per-OPCODE rules gated on OP_IS[op] apply the op's value-lane
                     effect + PC delta (branches deferred).
      6. branch    : bilinear BZ/BNZ PC update (AX_ZERO · IMM · PC).
      7. fold      : AX_VAL mod-256.
    The next-state lives in the scalar value lanes (AX_VAL/SP_VAL/BP_VAL/STK_VAL +
    PC_VAL) after this block; the driver's frame round-trip writes them back into
    the canonical nibble bands. Returns ``(model, L)``.
    """
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    L.two_limb = vm_two_limb()                # STAMP the build decision (driver honours it)
    dim = L.D
    ffn_specs = [
        compile_nibble_to_scalar(L, dim),
        compile_pc_fetch(L, dim),
        compile_code_select(L, dim),
        compile_opcode_decode(L, dim),
        compile_ffn(base_dispatch_rules(L), dim),
    ]
    if L.two_limb:
        # normalise the two-limb AX (carry/borrow across the 2^16 boundary) BEFORE
        # the branch/fold; STACK0 is only ever copied from an already-normal AX.
        ffn_specs.append(compile_limb_normalize(L, dim))
    ffn_specs += [
        compile_branch_delta(L, dim),
        compile_fold(L.AX_VAL, L.ONE, dim),        # width-aware modulus (256 / 2^32 / two-limb no-op)
    ]
    n_blocks = len(ffn_specs)
    hidden = max(f["W_up"].shape[0] for f in ffn_specs)

    model = Transformer(dim=dim, n_heads=n_heads, hidden=hidden,
                        n_blocks=n_blocks, vocab=V.VOCAB, max_seq_len=64)
    with torch.no_grad():
        model.embed.zero_()
        # NB: the recurrent driver seeds the state directly (load_program); the
        # embedding table still serves the standard token interface for the frame
        # emit/ingest proof (blogspec_run.ingest_ax_lowbyte).
        _bake_frame_embedding(model, L)
        for blk, spec in zip(model.blocks, ffn_specs):
            _zero_attn(blk.attn)
            _load_ffn(blk.ffn, spec, hidden)
    maybe_cast_model_for_width(model)
    return model, L


def maybe_cast_model_for_width(model) -> None:
    """Cast the model to fp64 ONLY on the single-scalar width-32 FALLBACK path.

    The single-scalar width-32 substrate needs fp64 because its ``16^7`` recompose
    + ADD/SUB algebra exceed fp32's 2^24 integer precision.  The TWO-LIMB path
    (``vm_two_limb``, the default when width-32 is on) carries AX/STACK0 as two
    fp32-exact limbs, so it stays **fp32** — this cast is a NO-OP there and no fp64
    appears anywhere.  Also a no-op under the 8-bit substrate (byte-identical to the
    folded corpus)."""
    if vm_width32() and not vm_two_limb():
        model.double()


def _zero_attn(attn) -> None:
    for p in (attn.W_q, attn.W_k, attn.W_v, attn.W_o):
        p.zero_()


def _load_ffn(ffn, spec: Dict[str, torch.Tensor], hidden: int) -> None:
    """Load a compiled spec into an FFN, zero-padding the hidden dim to ``hidden``."""
    h = spec["W_up"].shape[0]
    ffn.W_up.zero_();   ffn.W_up[:h] = spec["W_up"]
    ffn.b_up.zero_();   ffn.b_up[:h] = spec["b_up"]
    ffn.W_gate.zero_(); ffn.W_gate[:h] = spec["W_gate"]
    ffn.b_gate.zero_(); ffn.b_gate[:h] = spec["b_gate"]
    ffn.W_down.zero_(); ffn.W_down[:, :h] = spec["W_down"]
    ffn.b_down.zero_(); ffn.b_down.copy_(spec["b_down"])


def _bake_frame_embedding(model, L) -> None:
    """Byte tokens embed their two nibbles into CUR_NIB; markers light CTX. (The
    same foundation embedding — used by the ingest/emit frame proof.)"""
    E = torch.zeros(V.VOCAB, model.dim)
    E[:, L.ONE] = 1.0
    for b in range(256):
        lo, hi = V.nibbles_of_byte(b)
        E[b, L.CUR_NIB + 0] = float(lo)
        E[b, L.CUR_NIB + 1] = float(hi)
    model.embed.copy_(E)


# ===========================================================================
# UNIVERSAL: load a program into DATA memory (INPUT, not baked).
# ===========================================================================
def load_program(model, L: NibbleVMLayout, code: List[isa.Instr]) -> torch.Tensor:
    """The INITIAL STATE with ``code`` written into the DATA-MEMORY code bands.

    This is universality: different programs = different data in these bands; the
    model weights are untouched. Registers start at the spec init (§C4 Registers):
    PC=AX=0, SP=BP=0x10000, written as NIBBLE bands. ONE=1.
    """
    assert len(code) <= L.code_size, f"{len(code)} slots > code_size {L.code_size}"
    # The TWO-LIMB width-32 path runs the whole step in fp32 (AX/STACK0 carried as
    # two fp32-exact limbs, immediates stored as split limbs).  The single-scalar
    # width-32 FALLBACK runs in fp64 (its 16^7 recompose + ADD/SUB algebra exceed
    # fp32's 2^24).  The 8-bit substrate stays fp32 (byte-identical to the folded
    # corpus).
    two_limb = _layout_two_limb(L)
    dtype = torch.float64 if (vm_width32() and not two_limb) else torch.float32
    state = torch.zeros(L.D, dtype=dtype)
    state[L.ONE] = 1.0
    _write_reg_nibbles(state, L.PC, 0)
    _write_reg_nibbles(state, L.AX, 0)
    _write_reg_nibbles(state, L.SP, 0x10000)
    _write_reg_nibbles(state, L.BP, 0x10000)
    _write_reg_nibbles(state, L.STACK0, 0)
    for i, ins in enumerate(code):
        state[L.CODE_OP[i]] = float(ins.op)
        if two_limb:
            # store the immediate as two fp32-exact limbs (LOW/HIGH 4 nibbles each).
            imm = int(ins.imm) & 0xFFFFFFFF
            state[L.CODE_IMM_LO[i]] = float(imm & (_LO_MOD - 1))
            state[L.CODE_IMM_HI[i]] = float((imm // _LO_MOD) & _HI_MASK)
        else:
            state[L.CODE_IMM[i]] = float(ins.imm)
    return state


def _write_reg_nibbles(state: torch.Tensor, reg_base: int, value: int) -> None:
    for j, nv in enumerate(V.nibbles_of_value(value, 16)):
        state[reg_base + j] = float(nv)


# ===========================================================================
# VANILLA REQUANT DRIVER — the frame round-trip re-quantises AND writes the
# scalar next-state back into the nibble bands. NO torch.round.
# ===========================================================================
def _emit_and_reembed(state: torch.Tensor, L: NibbleVMLayout) -> torch.Tensor:
    """The vanilla token round-trip = the 30-token register frame requant.

    For every register, read its scalar next-state lane, EMIT the 4 little-endian
    byte tokens (argmax over the 256-byte value head — the integer snap, no
    round), and RE-EMBED each byte token by writing its two nibbles back into the
    register's NIBBLE band. Returns the fresh nibble-band state for the next step.
    This closes the loop: dispatch computed the next state on scalar lanes, and
    this writes it back into the spec's canonical nibble representation while
    annihilating all fp residue via the argmax.
    """
    two_limb = _layout_two_limb(L)
    new = torch.zeros_like(state)
    new[L.ONE] = 1.0
    # carry the immutable program-in-data bands untouched.
    for i in range(L.code_size):
        new[L.CODE_OP[i]] = state[L.CODE_OP[i]]
        if two_limb:
            new[L.CODE_IMM_LO[i]] = state[L.CODE_IMM_LO[i]]
            new[L.CODE_IMM_HI[i]] = state[L.CODE_IMM_HI[i]]
        else:
            new[L.CODE_IMM[i]] = state[L.CODE_IMM[i]]
    new[L.HALTED] = state[L.HALTED]
    # (nibble_base, snapped 32-bit value): AX/STACK0 are reconstructed from their
    # two fp32 limbs (lo + 2^16·hi, exact in Python int) under two-limb; PC/SP/BP
    # are single scalars.  All snapped to an exact integer and re-embedded as bytes.
    if two_limb:
        lanes = [(L.PC, _snap_lane(state[L.PC_VAL])),
                 (L.AX, _snap_two_limb(state[L.AX_LO], state[L.AX_HI])),
                 (L.SP, _snap_lane(state[L.SP_VAL])),
                 (L.BP, _snap_lane(state[L.BP_VAL])),
                 (L.STACK0, _snap_two_limb(state[L.STK_LO], state[L.STK_HI]))]
    else:
        lanes = [(L.PC, _snap_lane(state[L.PC_VAL])), (L.AX, _snap_lane(state[L.AX_VAL])),
                 (L.SP, _snap_lane(state[L.SP_VAL])), (L.BP, _snap_lane(state[L.BP_VAL])),
                 (L.STACK0, _snap_lane(state[L.STK_VAL]))]
    for reg_base, value in lanes:
        # EMIT its 4 little-endian byte tokens and RE-EMBED each byte's two nibbles
        # into the register's nibble band. The byte split uses the spec's own
        # floor/mod (§Efficient Floor), not python rounding.
        for bi in range(4):
            byte = (value >> (8 * bi)) & 0xFF          # spec floor/mod byte split
            lo, hi = V.nibbles_of_byte(byte)
            new[reg_base + 2 * bi + 0] = float(lo)     # RE-EMBED (byte -> nibbles)
            new[reg_base + 2 * bi + 1] = float(hi)
    return new


def _snap_two_limb(lo_lane: torch.Tensor, hi_lane: torch.Tensor) -> int:
    """Reconstruct the exact unsigned 32-bit word from the two fp32 limbs.

    Each limb is snapped to its nearest integer (the round-free ``floor(x+½)``
    LM-head requant — annihilating the O(1e-6) SwiGLU residue) in fp32, where each
    limb is < 2^24 and therefore fp32-exact.  The full value ``lo + 2^16·hi`` is
    assembled in Python int (exact, no float) and reduced to the two's-complement
    32-bit word ``& 0xFFFFFFFF`` — so a SUB borrow chain (negative high limb) or an
    ADD carry chain (high limb past its 4-nibble range) wraps correctly and a large
    loop counter never aliases to zero.  NO ``round`` (the AST guard forbids it);
    NO fp64."""
    import math
    lo = float(lo_lane)
    hi = float(hi_lane)
    lo_i = math.floor(lo + 0.5) if lo >= 0 else -math.floor(-lo + 0.5)
    hi_i = math.floor(hi + 0.5) if hi >= 0 else -math.floor(-hi + 0.5)
    return (lo_i + _LO_MOD * hi_i) & 0xFFFFFFFF


# The value vocabulary: token v decodes to the scalar integer v. Sized to cover
# every reachable register value in the foundation (SP/BP reach ~0x10000 as they
# move by ±4 around the init). The LM-head argmax over this vocab is the vanilla
# re-quantiser (``argmax_v(2·v·x − v²) == round(x)`` for v in range) — the same
# emit-token snap the classic C4 neural VM relies on, with NO ``torch.round``.
VALVOCAB = 0x10100   # 0..0x100FF: covers SP/BP = 0x10000 ± small, plus head-room


def _snap_lane(lane: torch.Tensor) -> int:
    """The LM-head requant: emit the value token ``argmax_v (2·v·x − v²)`` over the
    value vocabulary — the exact-integer snap of the lane, NO ``round``. Vectorised
    so the wide vocab is a single argmax (the standard decode-step argmax).

    Under ``C4_VM_WIDTH32`` the flat argmax (capped at ``VALVOCAB`` ≈ 0x10100)
    would clip any value > ~65K, so the snap descends to ``_snap_lane_bytes``: the
    fp64-exact integer snap (``floor(x+½)``, the round-free argmax equivalent)
    reduced to the unsigned 32-bit word ``v mod 2^32`` — exact to the full 2^32 with
    no 2^32-wide vocab (the §720-style cascade at the token round-trip)."""
    if vm_width32():
        return _snap_lane_bytes(lane)
    x = float(lane)
    v = torch.arange(VALVOCAB, dtype=torch.float64)
    logits = 2.0 * v * x - v * v            # LM-head value logits
    return int(logits.argmax().item())      # argmax == the emitted value token


def _snap_lane_bytes(lane: torch.Tensor) -> int:
    """PER-BYTE requant for the full-32-bit substrate (Family-2 fix).

    The lane holds a signed integer image carrying an O(1e-9) SwiGLU residue (a
    SUB may dip negative, an ADD may overflow past 2^32).  We first snap it to the
    nearest integer — the vanilla LM-head re-quantiser
    ``round(x) = argmax_v (2·v·x − v²)`` — realised round-free as
    ``floor(x + ½)`` (``math.floor``, NOT the ``round`` builtin the AST guard
    forbids; identical to the argmax snap the 8-bit ``_snap_lane`` does, only that
    flat argmax caps at ~0x10100 and cannot reach 2^32).  In fp64 the lane is exact
    to 2^53 ≫ 2^32, so ``floor(x+½)`` recovers the exact integer (annihilating the
    residue — the same job the argmax does).  The integer is then reduced to the
    unsigned 32-bit word ``v mod 2^32`` and split into 4 little-endian bytes by
    exact integer arithmetic.

    Sign / overflow: ``% 2^32`` gives the two's-complement word, so a borrow
    (negative lane) or overflow (≥ 2^32) wraps correctly and a large loop counter
    never aliases a large value to zero.  fp64 required (the width-32 driver runs
    the model in fp64)."""
    import math
    r = float(lane)
    # snap to the nearest integer (== the LM-head argmax; floor(x+½) is round-free)
    ivalue = math.floor(r + 0.5) if r >= 0 else -math.floor(-r + 0.5)
    return ivalue & 0xFFFFFFFF                        # two's-complement 32-bit word


def run_program(model, L: NibbleVMLayout, code: List[isa.Instr],
                max_steps: int = 100000, verbose: bool = False
                ) -> Tuple[List[int], List[Dict]]:
    """Run ``code`` on the universal nibble VM, emitting a 30-token register frame
    per step. Returns ``(tokens, frames)``: the flat autoregressive token stream
    and the per-step decoded register dicts.

    Each iteration: apply the ONE baked step-block, then the vanilla frame
    round-trip (``_emit_and_reembed``) re-quantises the scalar next-state into the
    nibble bands. NO torch.round anywhere. Stops at HALT.
    """
    state = load_program(model, L, code)
    tokens: List[int] = [V.BOS]
    frames: List[Dict] = []
    for _ in range(max_steps):
        # apply the step-block (the interpreter) to the current nibble state.
        x = state.view(1, 1, -1)
        for blk in model.blocks:
            x = blk(x)
        out = x[0, 0]
        halted = float(out[L.HALTED]) > 0.5
        # decode this step's registers from the scalar next-state lanes (AX is
        # reconstructed from its two limbs under the fp32 two-limb path).
        ax_val = (_snap_two_limb(out[L.AX_LO], out[L.AX_HI]) if _layout_two_limb(L)
                  else _decode_lane(out, L.AX_VAL))
        dec = {"pc": _decode_lane(out, L.PC_VAL), "ax": ax_val,
               "sp": _decode_lane(out, L.SP_VAL), "bp": _decode_lane(out, L.BP_VAL)}
        frame = V.build_step_frame(dec["pc"], dec["ax"], dec["sp"], dec["bp"])
        tokens += frame
        frames.append({**dec, "op": _op_name_at(out, code, L)})
        if verbose:
            print(f"  step op={frames[-1]['op']:5s} -> pc={dec['pc']} ax={dec['ax']} "
                  f"sp={dec['sp']} bp={dec['bp']}")
        # vanilla requant: emit the frame's byte tokens and re-embed -> nibbles.
        state = _emit_and_reembed(out, L)
        if halted:
            tokens.append(V.HALT)
            break
    return tokens, frames


def _decode_lane(state: torch.Tensor, val_lane: int) -> int:
    """Decode a full register value from its scalar lane via the LM-head value
    argmax (the vanilla re-quant snap; no python round)."""
    return _snap_lane(state[val_lane])


def _op_name_at(state: torch.Tensor, code, L) -> str:
    """The op the step just executed (from the PC one-hot), for the trace."""
    for i in range(L.code_size):
        if float(state[L.PC_IS[i]]) > 0.5 and i < len(code):
            return isa.NAMES.get(code[i].op, str(code[i].op))
    return "?"


def decode_trace(frames) -> List[int]:
    return [f["ax"] for f in frames]


# ---------------------------------------------------------------------------
# Convenience bundle.
# ---------------------------------------------------------------------------
class NibbleVM:
    """ONE universal nibble VM step-block + its layout; ``run`` any program."""

    def __init__(self, code_size: int, n_heads: int = 4):
        self.model, self.L = build_step_model(code_size, n_heads=n_heads)
        self.code_size = code_size

    def run(self, prog, max_steps: int = 100000, verbose: bool = False):
        code = isa.assemble(prog) if prog and isinstance(prog[0], (tuple, str)) else prog
        return run_program(self.model, self.L, code, max_steps=max_steps, verbose=verbose)

    def weight_hash(self) -> str:
        """A hash of the model weights (proves one FIXED model runs many programs)."""
        import hashlib
        h = hashlib.sha256()
        for p in self.model.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
