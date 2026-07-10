"""SHIFT-semantics DSL — derive the SHL/SHR result from ``BLOG_SPEC`` building
blocks instead of the hand-authored bit-shift operators.

``docs/BLOG_SPEC.md`` §597-599 ("Shifts"):

    Left and right shifts could simply be handled by 32 different
    multiplications by powers of two each, but it does not properly handle
    overflow, however overflow can be handled via the modulus by floor
    mentioned above.

and §555 (the MAGIC floor) + §560 (bit-range extraction — "multiply by
negative powers of two, do a floor, then a mod by a power of two"). Reading
those three passages literally gives the whole family with NO per-op magic:

  * a **power-of-two table** ``2**s`` (the "multiplications by powers of two");
  * a **floor** primitive; and
  * a **mod-by-a-power-of-two** primitive, expressed as the generic
    ``x mod m = x - m*floor(x/m)`` (the "modulus by floor").

From those three the two shift directions are:

  * ``SHL(v, s) = (v * 2**s) mod 2**8``  — multiply by ``2**s``, then the
    mod-by-floor caps the result to 8 bits (handles overflow, per §599).
  * ``SHR(v, s) = floor(v / 2**s)``      — divide by ``2**s``, then floor
    (drops the shifted-out low bits, per §560).

This module supplies ``shift_result(value, shift, direction=...)`` which the
L13 shifts lookup table (``ops/l13_ops.py``) uses in place of the hand-authored
``lambda v, s: (v << s) & 0xFF`` / ``lambda v, s: v >> s`` when
``C4_DERIVE_SHIFT=1`` (``ops/shared.derive_shift_enabled``). The lookup-table
STRUCTURE (per-``(shift, a_hi, a_lo)`` one-hot rule, 5-way AND, one-hot nibble
write) is unchanged — only the stored result VALUE is now derived from the spec
rather than typed as a bit-op. Because the derived value equals the bit-op for
every ``(v, s)`` (proven exhaustively below), the resulting FFN weights are
byte-identical and the whole-model golden hash is unchanged.

There are ZERO Python bit-shift operators (``<<`` / ``>>``) and ZERO per-op
mask constants (``& 0xFF``) in the derivation: the byte width and the powers of
two are the ONLY structural inputs, exactly matching the spec's "powers of two
+ mod-by-floor" recipe.
"""

from __future__ import annotations

from typing import Literal


# ---------------------------------------------------------------------------
# BLOG_SPEC building blocks (§555 MAGIC floor / §560 bit-range extraction /
# §599 shifts). Integer arithmetic ONLY — no bit-shift operators, no masks.
# ---------------------------------------------------------------------------

# The ISA word is byte-addressed: each ALU nibble-pair is one 8-bit byte, so
# the mod-by-floor modulus for a left shift is 2**8. This is the ONE structural
# constant (the byte width the whole VM is built on), not a per-op magic mask.
_BYTE_BITS = 8
_BYTE_MODULUS = 2 ** _BYTE_BITS  # 256 — the "mod by a power of two" of §560


def power_of_two(exponent: int) -> int:
    """The spec's "multiplication by powers of two" factor ``2**exponent``.

    §599: shifts are "multiplications by powers of two". This is the single
    derived table the whole family reads — no per-shift hand-typed constant.
    """
    if exponent < 0:
        raise ValueError(f"power_of_two: exponent must be >= 0; got {exponent}")
    return 2 ** exponent


def magic_floor_div(numerator: int, denominator: int) -> int:
    """Floor division ``floor(numerator / denominator)`` — the §555 floor.

    In the network this is the MAGIC-floor trick (§555:
    ``floor(x) = ((x - 0.5) + MAGIC) - MAGIC``) or the fp32-safe nibble-window
    floor (``alu/ops/floor_nibble_extract.py``); at derivation time the exact
    integer floor is used so the baked one-hot lands on the same result cell
    the running network would round to. For non-negative operands (the ISA
    nibbles are always positive — §593) Python ``//`` IS floor division.
    """
    if numerator < 0 or denominator <= 0:
        raise ValueError(
            "magic_floor_div: operands must be non-negative / positive; "
            f"got {numerator} / {denominator}"
        )
    return numerator // denominator


def mod_by_floor(value: int, modulus: int) -> int:
    """The §560 "mod by a power of two", written as the §555 mod-by-floor.

    ``x mod m = x - m * floor(x / m)`` — the modulus expressed purely through
    the floor primitive (no ``%`` / no mask). §599 names this exact construct
    ("overflow can be handled via the modulus by floor") as the overflow
    handler for the left shift.
    """
    return value - modulus * magic_floor_div(value, modulus)


# ---------------------------------------------------------------------------
# Derived shift result (the sole replacement for the hand-authored lambdas).
# ---------------------------------------------------------------------------

def shift_result(value: int, shift: int, *, direction: Literal["left", "right"]) -> int:
    """Derive ``value <shift> shift`` from the spec's powers-of-two + floor/mod.

    * ``direction="left"``  -> ``(value * 2**shift) mod 256`` (§599 mul + §560
      mod-by-floor overflow handling).
    * ``direction="right"`` -> ``floor(value / 2**shift)`` (§599 div + §555
      floor).

    No bit-shift operators, no mask constants — only ``power_of_two``,
    ``magic_floor_div`` and ``mod_by_floor``.
    """
    factor = power_of_two(shift)
    if direction == "left":
        # Multiply by the power of two, cap to the byte via mod-by-floor.
        return mod_by_floor(value * factor, _BYTE_MODULUS)
    if direction == "right":
        # Floor-divide by the power of two.
        return magic_floor_div(value, factor)
    raise ValueError(
        f"shift_result: direction must be 'left' or 'right'; got {direction!r}"
    )


def shl_result(value: int, shift: int) -> int:
    """SHL: ``(value * 2**shift) mod 256`` — spec-derived, matches ``(v<<s)&0xFF``."""
    return shift_result(value, shift, direction="left")


def shr_result(value: int, shift: int) -> int:
    """SHR: ``floor(value / 2**shift)`` — spec-derived, matches ``v>>s``."""
    return shift_result(value, shift, direction="right")


# ---------------------------------------------------------------------------
# Exhaustive byte-identity proof: the derived formula == the hand-authored
# bit-op for every 8-bit value and every 3-bit shift the L13 table enumerates.
# Computed at import so any regression trips immediately (and the golden hash
# gate proves it end-to-end at the weight level).
# ---------------------------------------------------------------------------

def _spec_matches_bitops() -> bool:
    for v in range(256):
        for s in range(8):
            if shl_result(v, s) != ((v << s) & 0xFF):
                return False
            if shr_result(v, s) != (v >> s):
                return False
    return True


_SPEC_MATCHES_BITOPS = _spec_matches_bitops()
assert _SPEC_MATCHES_BITOPS, (
    "shift_semantics_dsl: spec-derived SHL/SHR diverges from the reference "
    "bit-ops — the derivation is not byte-identical."
)
