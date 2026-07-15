"""Full-32-bit MUL / DIV / MOD as *nibble* SiLU gadgets (BLOG_SPEC-faithful).

This module implements the three arithmetic ops on the **nibble/byte
representation** exactly as ``docs/BLOG_SPEC.md`` describes them, rather than
the scalar 8-bit shortcut used by the ``gf-muldivmod`` completion:

  * **MUL** — byte-level *schoolbook* (``§Multiplication Implementation``):
    the two 32-bit operands are split into 4 bytes each, the 10 partial
    products ``a_i * b_j`` with ``i+j < 4`` are formed with the 6-weight
    SiLU-gated multiply gadget (``§Basic Arithmetic``), placed at byte
    position ``i+j`` (products with ``i+j >= 4`` overflow past bit 32 and are
    discarded), then **3 carry rounds** normalise the accumulator into 4
    result bytes.  Products with ``i+j >= 4`` are skipped per spec.

  * **DIV** — base-16 **long division** (``§Division Implementation``): each
    quotient nibble ``q in {0..15}`` is computed by *threshold counting*
    ``q = sum_{k=1..15} step(remainder - k * divisor)``; each iteration
    (MSB nibble first) folds the next dividend nibble into the running
    remainder, extracts the quotient digit, then subtracts ``q * divisor``.
    ``divisor == 0`` yields ``0`` per the ISA (``ISA_SPEC 4.2``).

  * **MOD** — ``§Mod``: division as a subroutine, then multiply the quotient
    by the divisor (schoolbook MUL) and subtract from the dividend.

Everything is built from three SiLU primitives shared across the three ops:

  * ``_relu(z)``     ``= silu(RELU_S * z) / RELU_S``  — clamped ReLU, exact on
                     integers (the ``compile_ffn`` / ``compile_fold`` gadget).
  * ``_step_ge(x,t)````= relu(x-(t-1)) - relu(x-t)``  — exact ``[x >= t]`` for
                     integer ``x, t`` (two hidden units; the DIV threshold
                     unit and the MUL carry-fold both use it).
  * ``_mul(a,b)``    ``= (silu(S a) + silu(-S a)) * b / S`` — the 6-weight
                     SiLU-*gated* multiply (``§Basic Arithmetic``); exact for
                     bounded ``a`` (a byte, 0..255).

**fp discipline (the gf-muldivmod fix).**  Read every staircase from
**bounded** operands and keep hidden magnitudes small:
  * MUL forms products byte-wise (``a_i, b_j <= 255``), so each SiLU-multiply
    argument is ``O(S * 255)`` — no 32-bit value ever enters a hidden node.
  * The DIV / MOD staircase thresholds ``k * divisor`` and the running
    remainder reach ``~15 * 2^32 ~ 6e10``, which *exceeds fp32's 2^24 unit
    precision*.  The spec sanctions doubles for the full-32-bit case ("We can
    of course use doubles"); the internal staircase arithmetic therefore runs
    in **float64** (52-bit mantissa comfortably resolves the unit step at 6e10)
    and uses a *reduced* threshold scale (``§Division`` "reduced scale to
    avoid float32 precision problems").  MUL's carry fold stays byte-bounded
    and is fp32-safe.

The public API takes/returns **16-nibble** little-endian arrays (nibble 0 =
least significant), matching the nibble-skeleton dispatch contract of two
16-nibble operands to a 16-nibble result.  ``dispatch`` maps an ISA opcode to
the corresponding gadget for wiring into the skeleton.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Scales (mirroring c4_min.compile_ffn conventions).
# ---------------------------------------------------------------------------
S = 60.0        # SiLU-identity / gated-multiply scale: silu(S)~=S, silu(-S)~=0
RELU_S = 200.0  # relu-via-silu scale for the byte-bounded MUL carry fold (fp32-ok)
DIV_S = 50.0    # reduced staircase scale for DIV/MOD (§Division), float64 domain

WIDTH_NIBBLES = 16   # 64-bit-wide nibble frame; low 8 nibbles = 32-bit result
RESULT_NIBBLES = 8   # 32-bit result occupies 8 nibbles
_MASK32 = 0xFFFFFFFF


# ===========================================================================
# SiLU primitives (the three shared gadgets)
# ===========================================================================
def _relu(z, s: float = RELU_S, dtype=torch.float64):
    """Clamped ReLU ``silu(s*z)/s``; exact ``max(0, z)`` for integer ``z``."""
    z = torch.as_tensor(z, dtype=dtype)
    return F.silu(s * z) / s


def _step_ge(x, t, s: float = RELU_S, dtype=torch.float64):
    """Exact ``[x >= t]`` for integer ``x, t`` via the clamped-relu difference
    ``relu(x-(t-1)) - relu(x-t)`` (== 1 iff ``x >= t``).  Two hidden units.

    ``t`` may be a scalar or a 1-D tensor of thresholds (batched staircase).
    """
    return _relu(x - (t - 1), s, dtype) - _relu(x - t, s, dtype)


def _count_ge(x, thresholds, s: float = RELU_S) -> int:
    """``sum_t [x >= t]`` over a batch of integer ``thresholds`` (one silu call)."""
    if len(thresholds) == 0:
        return 0
    ts = torch.as_tensor(list(thresholds), dtype=torch.float64)
    return int(round(float(_step_ge(float(x), ts, s).sum())))


def _mul(a, b, s: float = S, dtype=torch.float64):
    """6-weight SiLU-gated multiply ``(silu(S a)+silu(-S a))*b/S ~= a*b``
    (``§Basic Arithmetic``).  Exact for a *bounded* multiplicand ``a`` (a byte,
    0..255): with ``a`` non-negative, ``silu(S a) ~= S a`` and ``silu(-S a) ~= 0``
    so the gate reproduces ``a``; the ``* b / S`` then yields ``a * b``.
    """
    a = torch.as_tensor(a, dtype=dtype)
    b = torch.as_tensor(b, dtype=dtype)
    return (F.silu(s * a) + F.silu(-s * a)) * b / s


def _floor_div_staircase(x, m: int, kmax: int, s: float = DIV_S) -> int:
    """``floor(x / m) = count(x >= k*m for k=1..kmax)`` via threshold counting.

    The single fold behind MUL's carry (``m = 256``) and, in normalised form,
    DIV's quotient digit.  ``kmax`` bounds the number of thresholds; only the
    thresholds ``<= x`` can fire, so the batch is trimmed to ``floor(x/m)+1``.
    """
    x = int(round(float(x)))
    hi = min(kmax, x // m + 1)          # thresholds above x never fire
    return _count_ge(x, [k * m for k in range(1, hi + 1)], s)


# ===========================================================================
# Nibble <-> int helpers (16-nibble little-endian frame)
# ===========================================================================
def to_nibbles(value: int, width: int = WIDTH_NIBBLES) -> List[int]:
    """Little-endian nibble array (nibble 0 = least significant)."""
    value &= (1 << (4 * width)) - 1
    return [(value >> (4 * j)) & 0xF for j in range(width)]


def from_nibbles(nibs: Sequence[int]) -> int:
    """Inverse of :func:`to_nibbles`."""
    v = 0
    for j, n in enumerate(nibs):
        v |= (int(n) & 0xF) << (4 * j)
    return v


def _bytes_of(value: int, n: int = 4) -> List[int]:
    """Little-endian bytes of a value (n bytes)."""
    return [(value >> (8 * i)) & 0xFF for i in range(n)]


# ===========================================================================
# §Multiplication — byte-level schoolbook, 10 partial products, 3 carry rounds
# ===========================================================================
def mul32(a: int, b: int) -> int:
    """32-bit ``(a * b) & 0xFFFFFFFF`` via byte schoolbook (``§Multiplication``).

    * Split ``a, b`` into 4 bytes each (LSB first).
    * Form the 10 partial products ``a_i * b_j`` with ``i + j < 4`` (the SiLU
      multiply gadget; ``i + j >= 4`` products only touch bits >= 32 and are
      discarded), accumulating each into result-byte position ``i + j``.
    * 3 carry rounds: each byte position exports ``floor(acc/256)`` (staircase)
      to the next, keeping ``acc % 256``.
    """
    A = _bytes_of(a & _MASK32, 4)
    B = _bytes_of(b & _MASK32, 4)

    # 10 partial products a_i*b_j (i+j < 4) into 4 result-byte accumulators.
    acc = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        for j in range(4):
            if i + j >= 4:
                continue  # overflow past bit 32 — discarded per spec
            acc[i + j] += float(_mul(A[i], B[j]))

    # 3 carry rounds (byte). Max per-position accumulator ~ 4*255*255 = 260100,
    # so carry reaches ~1016 -> kmax 1024 thresholds is sufficient and byte-bounded.
    for _ in range(3):
        carry = [0, 0, 0, 0]
        for p in range(4):
            xv = int(round(acc[p]))
            c = _floor_div_staircase(xv, 256, kmax=1024, s=RELU_S)
            acc[p] = float(xv - 256 * c)
            if p + 1 < 4:
                carry[p + 1] = c
        for p in range(4):
            acc[p] += carry[p]

    res = 0
    for p in range(4):
        res |= (int(round(acc[p])) & 0xFF) << (8 * p)
    return res & _MASK32


# ===========================================================================
# §Division — base-16 long division, threshold-counting quotient digits
# ===========================================================================
def _udivmod32(a: int, b: int):
    """Unsigned 32-bit ``(a // b, a % b)`` — the magnitude long-division core.

    Each of the 8 quotient nibbles (MSB first) is
    ``q = sum_{k=1..15} step(remainder - k * divisor)`` (threshold counting).
    Per iteration: fold the next dividend nibble into the remainder, extract
    the digit, subtract ``q * divisor``.

    Runs in **float64** at a reduced scale: the staircase operands reach
    ``~15 * 2^32`` which exceeds fp32's 2^24 unit precision (the gf-muldivmod
    fp lesson); doubles resolve the unit step cleanly and the reduced
    ``DIV_S`` keeps the SiLU argument well-conditioned.
    """
    a &= _MASK32
    b &= _MASK32
    if b == 0:
        return 0, a  # ISA_SPEC 4.2: div/mod by zero -> 0

    # dividend nibbles, MSB first (8 nibbles = 32 bits)
    nibs = [(a >> (4 * j)) & 0xF for j in range(RESULT_NIBBLES - 1, -1, -1)]

    r = 0.0                       # running remainder (float64, < 16*b)
    q_nibs: List[int] = []        # quotient nibbles, MSB first
    for nib in nibs:
        r = r * 16 + nib
        # quotient digit q = sum_{k=1..15} [r >= k*b]  (batched staircase)
        qd = _count_ge(r, [k * b for k in range(1, 16)], DIV_S)
        r = r - qd * b            # subtract q*divisor
        q_nibs.append(qd)

    quotient = 0
    for qd in q_nibs:
        quotient = quotient * 16 + qd
    return quotient & _MASK32, int(round(r)) & _MASK32


def _to_signed(v: int) -> int:
    """Interpret a 32-bit word as two's-complement signed (ISA_SPEC 3.45)."""
    v &= _MASK32
    return v - (1 << 32) if v & 0x80000000 else v


def divmod32(a: int, b: int, signed: bool = False):
    """Return ``(a // b, a % b)`` masked to 32 bits, via base-16 long division.

    ``signed=False`` (default, the operational reference in
    ``neural_vm.nibble_bytecode_executor``): unsigned 32-bit floor division.

    ``signed=True`` (the C4-faithful ISA_SPEC 4.2 / 3.45 semantics): two's-
    complement **truncate-toward-zero**.  The magnitude long-division core
    runs on ``|a|, |b|``; the quotient sign is ``sign(a) xor sign(b)`` and the
    remainder sign follows the dividend.

    ``b == 0 -> (0, a)`` per ISA (no trap).
    """
    if not signed:
        return _udivmod32(a, b)

    sa, sb = _to_signed(a), _to_signed(b)
    if sb == 0:
        return 0, sa & _MASK32
    q_mag, r_mag = _udivmod32(abs(sa), abs(sb))
    q = -q_mag if (sa < 0) != (sb < 0) else q_mag   # sign(a) xor sign(b)
    r = -r_mag if sa < 0 else r_mag                 # remainder follows dividend
    return q & _MASK32, r & _MASK32


def div32(a: int, b: int, signed: bool = False) -> int:
    """32-bit ``a // b`` (``b == 0 -> 0``); see :func:`divmod32` for ``signed``."""
    return divmod32(a, b, signed)[0]


# ===========================================================================
# §Mod — division as a subroutine, then quotient*divisor and subtract
# ===========================================================================
def mod32(a: int, b: int, signed: bool = False) -> int:
    """32-bit ``a % b`` via ``§Mod``: divide, then ``quotient * divisor`` with
    the schoolbook MUL gadget and subtract from the dividend (``b == 0 -> 0``).

    For ``signed=True`` the multiply is on the two's-complement bit patterns
    (``mul32`` is sign-agnostic — the low 32 bits of ``q*b`` are identical for
    signed and unsigned operands), so ``a - q*b`` reproduces C truncate-toward-
    zero remainder directly.
    """
    a &= _MASK32
    b &= _MASK32
    if (_to_signed(b) if signed else b) == 0:
        return 0  # ISA_SPEC 4.2
    q = divmod32(a, b, signed)[0]
    prod = mul32(q, b)            # quotient * divisor via schoolbook MUL
    return (a - prod) & _MASK32


# ===========================================================================
# Nibble-array wrappers — the nibble-skeleton dispatch contract
# ===========================================================================
def nibble_mul(a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Two 16-nibble operands -> 16-nibble ``(a*b) & 0xFFFFFFFF`` result."""
    return to_nibbles(mul32(from_nibbles(a_nibs), from_nibbles(b_nibs)))


def nibble_div(a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Two 16-nibble operands -> 16-nibble ``a // b`` (``b==0 -> 0``)."""
    return to_nibbles(div32(from_nibbles(a_nibs), from_nibbles(b_nibs)))


def nibble_mod(a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Two 16-nibble operands -> 16-nibble ``a % b`` (``b==0 -> 0``)."""
    return to_nibbles(mod32(from_nibbles(a_nibs), from_nibbles(b_nibs)))


# Opcode -> nibble gadget (for plugging into the skeleton dispatch).
try:  # tolerate import outside the c4_min package tree
    from . import isa as _isa
    DISPATCH = {
        _isa.MUL if hasattr(_isa, "MUL") else 27: nibble_mul,
        _isa.DIV if hasattr(_isa, "DIV") else 28: nibble_div,
        _isa.MOD if hasattr(_isa, "MOD") else 29: nibble_mod,
    }
except Exception:  # pragma: no cover - standalone use
    DISPATCH = {27: nibble_mul, 28: nibble_div, 29: nibble_mod}


def dispatch(opcode: int, a_nibs: Sequence[int], b_nibs: Sequence[int]) -> List[int]:
    """Apply the nibble gadget for ``opcode`` (MUL/DIV/MOD) to the operands.

    ``a`` is the popped stack top, ``b`` the accumulator, matching the c4
    stack-machine convention ``AX = pop() OP AX`` (``a OP b``).
    """
    fn = DISPATCH.get(opcode)
    if fn is None:
        raise KeyError(f"nibble_muldivmod.dispatch: opcode {opcode} is not MUL/DIV/MOD")
    return fn(a_nibs, b_nibs)
