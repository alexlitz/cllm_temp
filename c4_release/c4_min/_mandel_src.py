"""Tiny fixed-point Mandelbrot C-source generator for the neural-VM demo.

Two properties make ``model.forward`` byte-exact against the reference:

1. **Every emitted ``IMM`` immediate is <= 255.**  The VM's ``IMM`` is a byte; the
   byte-masking reference ``ref_interpret`` treats ``IMM 1024`` as ``1024 & 0xFF
   = 0`` while the neural model keeps the full overlaid value, so they DISAGREE
   for ``IMM > 255``.  Larger constants (scale, window bases, steps) are built via
   MUL/ADD of byte literals, which the c4_min compiler does NOT constant-fold.

2. **All intermediate AX values stay NON-NEGATIVE** (no value ever wraps to the
   2^31..2^32 range).  A wrapped-negative in AX corrupts the model's next ``LEA``
   (the documented AX high-byte / 0xFF-leak weakness — verified: ``SUB`` to -64
   then ``LEA`` gives 0 on the model vs the correct address on the reference,
   while plain negative ARITHMETIC like ``(-20)*(-20)=400`` is byte-exact).  The
   classic ``zx2 - zy2`` term can go negative, so this generator restructures it:
   compute ``t = zx2 + cx`` first, and if ``zy2 > t`` the point has escaped (the
   real part would go negative -> the orbit is diverging) so we bail with ' ';
   otherwise ``t = t - zy2`` is a NON-NEGATIVE subtraction.  The window is the
   first quadrant (cx, cy >= 0), keeping every value >= 0.

The VM is unsigned 32-bit; the render is the VM's own escape-time set over that
non-negative window (honestly reported — a bounded slice, not a textbook float
mandelbrot).  A full render is a ~200-hour perf wall; this is a few-hundred-step
byte-exact demonstration.  Only the verified op set is used: IMM/LEA/PSH/ADD/SUB/
MUL/DIV/LT/GT + JSR/ENT/ADJ/LEV framing + LI/SI locals + PRTF (op 33).
"""
from __future__ import annotations

#: fixed-point scale = SCALE_SQRT**2 (== 64), built via MUL of byte literals.
SCALE_SQRT = 8


def mandel_c(width: int, height: int, maxiter: int, x0: int = 0, dx: int = 96,
             y0: int = 0, dy: int = 64) -> str:
    """Return C source for a ``width`` x ``height`` escape-time render.

    Fixed-point ``scale = 8*8 = 64`` (so 1.0 == 64).  Window: ``cx = x0 + px*dx``,
    ``cy = y0 + py*dy`` in fixed-point units (all NON-NEGATIVE, all <= 255 so the
    step immediates ``dx``/``dy``/``x0``/``y0`` stay byte-sized).  Defaults render
    the first-quadrant window cx in [0, width*1.5), cy in [0, height*1.0).  Emits
    '*' (bounded through ``maxiter``) / ' ' (escaped) per cell over the PRTF
    stdout channel, newline (printf(10)) per row.
    """
    s = SCALE_SQRT
    for name, v in (("dx", dx), ("dy", dy), ("x0", x0), ("y0", y0),
                    ("width", width), ("height", height), ("maxiter", maxiter)):
        assert 0 <= v <= 255, f"{name}={v} must be a byte literal (0..255)"
    return f'''
int main() {{
  int scale, four, px, py, cx, cy, zx, zy, a, b, t, it, hit;
  scale = {s} * {s};
  four = scale + scale + scale + scale;
  py = 0;
  while (py < {height}) {{
    cy = {y0} + py * {dy};
    px = 0;
    while (px < {width}) {{
      cx = {x0} + px * {dx};
      zx = 0; zy = 0; it = 0; hit = 0;
      while (it < {maxiter}) {{
        a = zx * zx / scale;
        b = zy * zy / scale;
        if (a + b > four) {{ hit = 1; it = {maxiter}; }}
        if (hit == 0) {{
          t = a + cx;
          if (b > t) {{ hit = 1; it = {maxiter}; }}
          if (hit == 0) {{
            t = t - b;
            zy = zx * zy / scale * 2 + cy;
            zx = t;
            it = it + 1;
          }}
        }}
      }}
      if (hit == 0) {{ printf(42); }}
      if (hit == 1) {{ printf(32); }}
      px = px + 1;
    }}
    printf(10);
    py = py + 1;
  }}
  return 0;
}}
'''
