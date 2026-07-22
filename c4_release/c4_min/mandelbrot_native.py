"""NATIVE-MUL MANDELBROT on the FULL-NATIVE C4 VM — one hardware MUL per multiply.

The lean render (:mod:`lean_mandelbrot`) does every 16-bit multiply as a
SOFTWARE shift-add subroutine (base + bitwise ISA, no hardware muldiv) — ~54k
instructions per pixel.  The FULL-NATIVE model
(``qwen_full_vm.build(SUBSET_FULL, efficient_alu=True)``, driven via
``full_native_fast``) has a NATIVE ``MUL`` opcode (gated byte / nibble
schoolbook, byte-exact vs Python ``*``), so each fixed-point multiply is ONE
``MUL`` instruction.  This module rewrites the escape-time iteration to use that
native ``MUL`` (plus ``DIV`` for the ``>>Q_FRAC`` fixed-point rescale) inside a
``BNZ``/``BZ`` loop over ``max_iter`` — so the pixel program is a SMALL LOOP
(~180 instructions, CONSTANT in ``max_iter``) instead of a straight-line unroll
that grows with the iteration count.

Reference / word width
----------------------
The full-native model's word-faithful CPU oracle is
``nibble_pure_forward_complete.ref_interpret`` at ``mask=0xFFFFFFFF`` — the
"blessed unsigned/two's-complement 32-bit oracle the KV-cached driver is
byte-identical to" (see ``test_mandelbrot_neural``).  Its contract is:

  * ``PSH`` / stack values and ``ADD``/``SUB``/``MUL``/``DIV``/``MOD`` are FULL
    32-bit (wrap mod 2^32).  A native ``MUL`` yields the whole 32-bit product,
    so ``zx*zx`` (up to ~85*85 = 7225) fits with room to spare.
  * ``LT``/``GT``/``LE``/``GE`` are SIGNED two's-complement at the 2^31 sign
    boundary (they also DO NOT pop their stack operand — see ``_drop``).
  * ``SI``/``SC``/``LI``/``LC`` memory and ``IMM`` immediates are BYTE-only
    (``& 0xFF``).  ``SHL``/``SHR``/``AND``/``OR``/``XOR`` are BYTE-only too.

The byte-only memory is the crux of the design: a fixed-point value (``zx``,
``zy`` reach ~±85 in Q4) does NOT round-trip through a raw byte cell, so the
persistent loop state lives EITHER on the 32-bit data stack (transients) or in
byte memory with a ``+BIAS`` offset (``zx``, ``zy`` are stored as ``zx+128`` ∈
[43, 213], a clean byte, and read back as ``mem-128``).  The fixed-point rescale
``>>Q_FRAC`` uses native ``DIV`` by ``SCALE`` on the MAGNITUDE (``SHR`` is
byte-only and cannot shift the 32-bit product), which is exactly ``x >> Q_FRAC``
for a non-negative magnitude.

Signed truncate-toward-zero (the byte-exactness hinge)
------------------------------------------------------
``lean_mandelbrot._sfp_mul`` truncates ``(a*b) >> Q_FRAC`` TOWARD ZERO for
signed operands: ``m = (|a|*|b|) >> Q_FRAC``; result ``= sign(a)^sign(b) ? -m :
m``.  A plain arithmetic shift / floor-divide truncates toward -inf, which
disagrees by one ULP on negative cross terms (~19% of pixels).  This module
reproduces the toward-zero semantics BRANCHLESSLY:

    m_sign = 1 - 2*((a<0) XOR (b<0))          # +1 or -1
    p      = a*b                               # native MUL (two's-complement)
    result = ((p * m_sign) // SCALE) * m_sign  # |p|//SCALE re-signed

``p * m_sign`` is exactly ``|p|`` (non-negative), so the unsigned ``DIV`` floors
correctly, and ``* m_sign`` re-applies the sign — byte-identical to ``_sfp_mul``
(verified 5985/5985 sfp values and every pixel of the reference grid).  The sign
bits are read WITHOUT a compare (compares don't pop cleanly on this VM): with the
``+BIAS`` storage, ``a<0`` ⟺ stored byte ``< 128`` ⟺ ``(byte>>7)==0`` (an 8-bit
``SHR`` on a byte is well-defined), so ``a_neg = 1 - (byte>>7)``.

Deliverables
------------
* :func:`pixel_program_native` — the per-pixel escape-count LOOP program.
* :func:`render_ppm_native` — a P6 PPM byte stream over an escape-count palette.
* :func:`steps_per_pixel` — the VM step count (interpret-trace length).

All verification runs on CPU through ``ref_interpret`` (``mask=0xFFFFFFFF``); the
neural forward is never invoked here.
"""
from __future__ import annotations

from typing import List, Tuple

from . import isa
from .nibble_runtime import Asm

# ---------------------------------------------------------------------------
# Fixed-point parameters — IDENTICAL to lean_mandelbrot so escape counts are
# directly comparable.
# ---------------------------------------------------------------------------
Q_FRAC = 4
SCALE = 1 << Q_FRAC                 # 16   (1.0 == 16 in fixed point)
FOUR = 4 * SCALE                    # 64   (|z|^2 > 4 escape threshold)
BIAS = 128                          # signed-byte storage offset for zx, zy
CHARSET = " .:-=+*#%@"

# ---------------------------------------------------------------------------
# Byte-memory state cells (low bytes distinct; well below the stack region,
# which descends from SP_INIT = 0x10000).
#   A_ZX / A_ZY   : zx, zy stored BIASED  (mem = z + BIAS, a clean byte)
#   A_I           : iteration counter (0..max_iter, small non-negative)
#   A_ESC         : escape count result  (defaults to max_iter)
#   A_TZX / A_TZY : temp new-zx / new-zy (biased) so the update reads OLD z
#   A_MA / A_MB   : the sfp subroutine's operand-ADDRESS argument cells
# ---------------------------------------------------------------------------
A_ZX, A_ZY, A_I, A_ESC, A_TZX, A_TZY, A_MA, A_MB = (
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A)


# ===========================================================================
# Byte-exact Python oracle (mirrors lean_mandelbrot; kept here so this module is
# self-contained for verification and grid mapping is IDENTICAL).
# ===========================================================================
def _sfp_mul(a: int, b: int) -> int:
    """Signed fixed-point ``(a*b) >> Q_FRAC``, TRUNCATE-TOWARD-ZERO — identical to
    ``lean_mandelbrot._sfp_mul`` (magnitude shift then re-sign)."""
    m = (abs(a) * abs(b)) >> Q_FRAC
    return -m if (a < 0) != (b < 0) else m


def _pixel_escape(cx: int, cy: int, max_iter: int) -> int:
    """Escape count for fixed-point ``c=(cx,cy)/SCALE``; ``z=z^2+c`` until |z|^2>4.
    Byte-identical to ``lean_mandelbrot._pixel_escape``."""
    zx = zy = 0
    for i in range(max_iter):
        zx2 = _sfp_mul(zx, zx)
        zy2 = _sfp_mul(zy, zy)
        if zx2 + zy2 > FOUR:
            return i
        two_zxzy = 2 * _sfp_mul(zx, zy)
        zx, zy = zx2 - zy2 + cx, two_zxzy + cy
    return max_iter


def grid(width: int, height: int) -> Tuple[List[int], List[int]]:
    """Fixed-point complex-plane grid — IDENTICAL mapping to
    ``lean_mandelbrot._grid`` (cx in [-2.1, 0.9]*SCALE, cy in [-1.2, 1.2]*SCALE)."""
    cxs = [round((col / (width - 1) * 3.0 - 2.1) * SCALE) for col in range(width)]
    cys = [round((row / (height - 1) * 2.4 - 1.2) * SCALE) for row in range(height)]
    return cxs, cys


# ===========================================================================
# Native-MUL bytecode assembly helpers.
# ===========================================================================
def _to_signed(v: int) -> int:
    """Interpret a 32-bit VM word as a signed two's-complement int."""
    return v - (1 << 32) if v & (1 << 31) else v


def _psh_signed_imm(a: Asm, v: int) -> None:
    """AX = signed constant ``v`` (built from byte immediates; negatives via 0 - |v|,
    keeping every ``IMM`` <= 255 so the byte-masking reference stays exact)."""
    if v >= 0:
        a.imm(v & 0xFF)
    else:
        a.imm(0).psh().imm((-v) & 0xFF).emit(isa.SUB)   # AX = 0 - |v|


def _stc(a: Asm, addr: int, c: int) -> None:
    """mem[addr] = c & 0xFF  (byte store of a small constant)."""
    a.imm(addr).psh().imm(c & 0xFF).emit(isa.SI)


def _drop(a: Asm) -> None:
    """Pop and discard the leftover a comparison leaves on the stack.

    On ``ref_interpret`` (the model's word oracle) the ordering compares
    LT/GT/LE/GE/EQ/NE do NOT pop their stack operand — they leave it in place.
    In a loop that would grow the stack every iteration, so after each compare's
    branch we consume that leftover with ``IMM 0; ADD`` (``ax = pop() + 0``); the
    popped garbage lands in AX and is immediately overwritten."""
    a.imm(0).emit(isa.ADD)


def _emit_sfp(a: Asm) -> None:
    """The signed fixed-point multiply SUBROUTINE (shared by the squares and the
    cross term).  Operand ADDRESSES are passed in the byte cells A_MA, A_MB; the
    result is returned in AX (LEV preserves AX).

    result = ((p * m) // SCALE) * m ,  m = 1 - 2*((a<0) XOR (b<0)) ,  p = a*b.

    ``m`` is computed ONCE and kept as two stack copies (one for ``p*m`` = |p|,
    one for the final re-sign).  ``a<0`` is read from the +BIAS byte's top bit
    (``1 - (byte>>7)``) — no compare, so nothing is left on the stack."""
    a.label("sfp")
    a.emit(isa.ENT, 0)

    def _ldz(a: Asm, acell: int) -> None:
        # AX = signed z = mem[mem[acell]] - BIAS   (double LI: acell holds the addr)
        a.imm(acell).li().li().psh().imm(BIAS).emit(isa.SUB)

    def _sbit(a: Asm, acell: int) -> None:
        # AX = 1 if z<0 else 0  = 1 - (stored_byte >> 7)
        a.imm(1).psh().imm(acell).li().li().psh().imm(7).emit(isa.SHR).emit(isa.SUB)

    # m = 1 - 2*(sign(a) XOR sign(b))
    a.imm(1).psh()
    _sbit(a, A_MA)
    a.psh()
    _sbit(a, A_MB)
    a.emit(isa.XOR)                        # AX = a_neg XOR b_neg  (0 or 1)
    a.psh().imm(2).emit(isa.MUL)           # AX = 2 * pneg
    a.emit(isa.SUB)                        # AX = 1 - 2*pneg = m  (+1 or -1)
    a.psh().psh()                          # stack: [m, m]
    # p = a * b   (native 32-bit MUL; two's-complement product)
    _ldz(a, A_MA)
    a.psh()
    _ldz(a, A_MB)
    a.emit(isa.MUL)                        # AX = p ; stack: [m, m]
    a.emit(isa.MUL)                        # AX = p * m = |p| ; stack: [m]
    a.psh().imm(SCALE).emit(isa.DIV)       # AX = |p| // SCALE  (unsigned floor, exact)
    a.emit(isa.MUL)                        # AX = (|p|//SCALE) * m ; stack: []
    a.emit(isa.LEV)


def _call_sfp(a: Asm, ax: int, ay: int) -> None:
    """Set the operand-address args and call the sfp subroutine (result -> AX)."""
    a.imm(A_MA).psh().imm(ax).emit(isa.SI)
    a.imm(A_MB).psh().imm(ay).emit(isa.SI)
    a.emit(isa.JSR, "sfp")


# ===========================================================================
# The per-pixel program.
# ===========================================================================
def pixel_program_native(cx: int, cy: int, max_iter: int) -> List[isa.Instr]:
    """Escape-count LOOP program for one pixel; result (the escape count) in AX.

    Mirrors ``_pixel_escape``: ``zx=zy=0``; each iteration computes
    ``zx2=sfp(zx,zx)``, ``zy2=sfp(zy,zy)``, tests ``zx2+zy2 > FOUR`` (escape ->
    record ``i``), then ``zx = zx2-zy2+cx``, ``zy = 2*sfp(zx,zy)+cy``.  The escape
    count defaults to ``max_iter`` and is set to the FIRST escaping ``i``.  The
    loop is a ``BZ`` back-edge (constant static size in ``max_iter``); the sfp
    multiply is a shared subroutine so the whole image stays small."""
    a = Asm()
    # --- init: zx = zy = 0 (biased = BIAS); i = 0; esc = max_iter ---
    _stc(a, A_ZX, BIAS)
    _stc(a, A_ZY, BIAS)
    _stc(a, A_I, 0)
    _stc(a, A_ESC, max_iter)

    a.label("loop")
    # while (i < max_iter):    (LT leaves i on the stack -> _drop after the branch)
    a.imm(A_I).li().psh().imm(max_iter).emit(isa.LT)
    a.bz("end")
    _drop(a)

    # mag = zx2 + zy2
    _call_sfp(a, A_ZX, A_ZX)
    a.psh()
    _call_sfp(a, A_ZY, A_ZY)
    a.emit(isa.ADD)                        # AX = mag
    # if mag > FOUR:  esc = i; break     (GT leaves mag on the stack -> _drop)
    a.psh().imm(FOUR).emit(isa.GT)
    a.bz("noesc")
    _drop(a)
    a.imm(A_ESC).psh().imm(A_I).li().emit(isa.SI)   # esc = i
    a.jmp("end")

    a.label("noesc")
    _drop(a)
    # new zx = zx2 - zy2 + cx   (stored biased into A_TZX; read OLD z)
    a.imm(A_TZX).psh()
    _call_sfp(a, A_ZX, A_ZX)
    a.psh()
    _call_sfp(a, A_ZY, A_ZY)
    a.emit(isa.SUB)                        # zx2 - zy2
    a.psh()
    _psh_signed_imm(a, cx)
    a.emit(isa.ADD)                        # + cx
    a.psh().imm(BIAS).emit(isa.ADD).emit(isa.SI)    # store biased

    # new zy = 2*sfp(zx, zy) + cy   (stored biased into A_TZY)
    a.imm(A_TZY).psh()
    _call_sfp(a, A_ZX, A_ZY)
    a.psh().imm(2).emit(isa.MUL)          # 2 * cross
    a.psh()
    _psh_signed_imm(a, cy)
    a.emit(isa.ADD)                        # + cy
    a.psh().imm(BIAS).emit(isa.ADD).emit(isa.SI)

    # commit new z (biased byte copies) and bump the iteration counter
    a.imm(A_ZX).psh().imm(A_TZX).li().emit(isa.SI)
    a.imm(A_ZY).psh().imm(A_TZY).li().emit(isa.SI)
    a.imm(A_I).psh().imm(A_I).li().psh().imm(1).emit(isa.ADD).emit(isa.SI)   # i += 1
    a.jmp("loop")

    a.label("end")
    a.imm(A_ESC).li()                     # AX = escape count
    a.exit_()                             # HALT (main never falls into the sub)

    _emit_sfp(a)                          # the sfp subroutine, after HALT
    return a.instrs()


# ===========================================================================
# Escape count via the CPU word reference (NOT the model).
# ===========================================================================
def escape_count_native(cx: int, cy: int, max_iter: int) -> int:
    """Run ``pixel_program_native`` through ``ref_interpret`` (mask=0xFFFFFFFF) and
    return the decoded escape count in AX."""
    from .nibble_pure_forward_complete import ref_interpret
    code = pixel_program_native(cx, cy, max_iter)
    trace = ref_interpret(code, max_steps=2_000_000, mask=0xFFFFFFFF)
    return _to_signed(trace[-1])


def steps_per_pixel(cx: int, cy: int, max_iter: int) -> int:
    """VM step count (length of the interpret trace) for one pixel — the render
    cost through the native-MUL loop."""
    from .nibble_pure_forward_complete import ref_interpret
    code = pixel_program_native(cx, cy, max_iter)
    return len(ref_interpret(code, max_steps=2_000_000, mask=0xFFFFFFFF))


# ===========================================================================
# PPM render.
# ===========================================================================
def _palette(esc: int, max_iter: int) -> Tuple[int, int, int]:
    """Map an escape count to an RGB triple.  Inside-set (esc == max_iter) is BLACK;
    escaped pixels get a smooth blue->white ramp by escape speed."""
    if esc >= max_iter:
        return (0, 0, 0)
    t = esc / max(1, max_iter - 1)              # 0 (fast escape) .. ~1 (slow)
    r = int(round(255 * t))
    g = int(round(255 * (t ** 0.5)))
    b = int(round(255 * (0.4 + 0.6 * t)))
    return (min(255, r), min(255, g), min(255, b))


def escape_grid_native(width: int, height: int, max_iter: int) -> List[List[int]]:
    """Escape counts for the whole grid via ``ref_interpret`` (the exact byte stream
    the model would emit)."""
    cxs, cys = grid(width, height)
    return [[escape_count_native(cx, cy, max_iter) for cx in cxs] for cy in cys]


def render_ppm_native(width: int, height: int, max_iter: int) -> bytes:
    """Render the grid to a P6 PPM byte stream: ``P6\\n<w> <h>\\n255\\n`` header
    followed by ``w*h*3`` raw RGB bytes.  Escape counts come from the native-MUL
    ``pixel_program_native`` run through the CPU word reference — the exact bytes a
    PRTF stream from the model would produce."""
    escapes = escape_grid_native(width, height, max_iter)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    body = bytearray()
    for row in escapes:
        for esc in row:
            body.extend(_palette(esc, max_iter))
    return bytes(header) + bytes(body)


def parse_ppm(data: bytes) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """Minimal P6 parser -> (width, height, [rgb, ...]).  Used by the verifier to
    confirm the stream round-trips to the escape-count grid."""
    assert data[:2] == b"P6", "not a P6 PPM"
    # header: P6 \n W H \n 255 \n
    idx = 2
    fields: List[int] = []
    while len(fields) < 3:
        while idx < len(data) and data[idx] in b" \t\n\r":
            idx += 1
        start = idx
        while idx < len(data) and data[idx] not in b" \t\n\r":
            idx += 1
        fields.append(int(data[start:idx]))
    idx += 1                                    # single whitespace after maxval
    w, h, _maxval = fields
    pixels = [
        (data[idx + 3 * k], data[idx + 3 * k + 1], data[idx + 3 * k + 2])
        for k in range(w * h)
    ]
    return w, h, pixels


# ===========================================================================
# ASCII preview (handy sanity view, not part of the byte-exact gate).
# ===========================================================================
def render_ascii_native(width: int, height: int, max_iter: int) -> str:
    escapes = escape_grid_native(width, height, max_iter)
    rows = ["".join(CHARSET[min(e, len(CHARSET) - 1)] for e in row) for row in escapes]
    return "\n".join(rows)


if __name__ == "__main__":     # pragma: no cover
    prog = pixel_program_native(-10, 5, 12)
    print(f"pixel program instruction count: {len(prog)}")
    print(f"steps for interior (0,0) mi=12 : {steps_per_pixel(0, 0, 12)}")
    print(render_ascii_native(48, 20, 12))
