"""Tiny fixed-point Mandelbrot C-source generator for the neural-VM demo.

All integer literals emitted are <= 255 (the VM's IMM immediate is a byte; the
neural model and the byte-masking reference ``ref_interpret`` only AGREE for
byte-sized IMMs — for IMM > 255 the model keeps the full overlaid value while
``ref_interpret`` masks to 0xFF, so they would DISAGREE).  Larger constants
(scale, coordinate bases) are therefore built via MUL/ADD/SUB of byte literals,
which the c4_min compiler does NOT constant-fold — so the emitted bytecode has
no ``IMM > 255`` and ``model.forward`` output is byte-exact against
``ref_interpret``.

The VM is unsigned 32-bit (mod 2^32).  Squares recover correctly for wrapped
negatives ((2^32 - k)^2 == k^2 mod 2^32) and the escape test is on the always
non-negative sum ``a + b`` (= zx^2/scale + zy^2/scale), so the escape-time
pattern is well defined.  The cross term ``2*zx*zy`` is divided under UNSIGNED
semantics; the reference (``ref_interpret``) and the neural model share those
EXACT semantics, so they agree byte-for-byte.  The render is therefore the VM's
own escape-time set (honestly reported: an unsigned-fixed-point mandelbrot, not
a textbook float render).
"""
from __future__ import annotations


def mandel_c(width: int, height: int, maxiter: int) -> str:
    """Return C source for a ``width`` x ``height`` escape-time render at
    ``maxiter`` iterations, fixed-point ``scale = 8*8 = 64``.

    Lean body (13 locals) to keep the per-cell VM step count small.  Window:
    cx in [-2, width-2) scale-units, cy in [-1, height-1) scale-units (one
    scale-unit == 1.0).  Emits one char per cell ('*' inside / ' ' escaped)
    and a newline (printf(10)) per row, over the PRTF visible-output channel.
    """
    return f'''
int main() {{
  int scale, four, px, py, cx, cy, zx, zy, a, b, t, it, hit;
  scale = 8 * 8;
  four = scale * 4;
  py = 0;
  while (py < {height}) {{
    cy = py * scale - scale;
    px = 0;
    while (px < {width}) {{
      cx = px * scale - scale - scale;
      zx = 0; zy = 0; it = 0; hit = 0;
      while (it < {maxiter}) {{
        a = zx * zx / scale;
        b = zy * zy / scale;
        if (a + b > four) {{ hit = 1; it = {maxiter}; }}
        if (hit == 0) {{
          t = a - b + cx;
          zy = zx * zy / scale * 2 + cy;
          zx = t;
          it = it + 1;
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
