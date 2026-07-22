"""C-source generators for the ONE-LAYER SELF-EMULATION demo — a fixed-point
MatMul / dot-product kernel (the load-bearing inner op of a transformer layer, and
of the ONNX runtime that would self-host the model) rendered so it runs BYTE-EXACT
through the actual c4_min ``model.forward``.

This is the honest, minimal instance of self-hosting Rel-3 THROUGH THE NEURAL
FORWARD: the neural transformer executing c4 bytecode that computes a piece of a
transformer layer — a dot product / matmul, one operand of which is a SLICE OF THE
MODEL'S OWN WEIGHTS (a row / block of the c4_min embedding matrix, quantized to
fixed-point).  It is the matmul analogue of the bounded Mandelbrot
(``c4_min/_mandel_src.py``): a real, byte-exact computation through the neural
forward, deliberately BOUNDED, NOT the full self-forward (the ~2.4M-step / ~88-day
wall — see ``docs/SELFHOST_3LAYER_FEASIBILITY.md`` and
``docs/ONE_LAYER_SELF_EMULATION_2026_07_20.md``).

Three properties make ``model.forward`` byte-exact against the reference interpreter
(``nibble_pure_forward_complete.ref_interpret``), as for the Mandelbrot:

1. **Every emitted ``IMM`` immediate is <= 255.**  The VM's ``IMM`` is a byte; the
   byte-masking reference treats ``IMM 4096`` as ``4096 & 0xFF = 0`` while the
   neural model keeps the full overlaid value, so they DISAGREE for ``IMM > 255``.
   The fixed-point scale + the matrix entries are therefore byte-sized, and scaled
   operands (``entry * scale``) are built via a runtime ``MUL`` of two byte literals
   (which the c4_min compiler does NOT constant-fold).  Referencing a GLOBAL emits
   its data-segment ADDRESS (>255) as an ``IMM``, so the kernel uses ONLY locals
   (``scale`` is a ``fpmul`` argument).

2. **Every value STORED to a local stays <= 255 and NON-NEGATIVE.**  VM memory is
   byte-addressed: a ``SI``/``SC`` store keeps only ``AX & 0xFF`` (a local is one
   byte wide), and a wrapped-negative AX corrupts the model's next ``LEA`` (the
   AX high-byte / 0xFF-leak weakness).  So ``scale`` is small (``16`` = 4 fractional
   bits) and the operands are small enough that every accumulator fits in a byte.
   The MAC product ``a*b`` (e.g. ``16*48 = 768``) briefly EXCEEDS a byte, but it
   lives only in the AX register (32-bit) and is divided back down by ``scale``
   BEFORE any store — so ``fpmul`` is written ``return a*b/s`` with NO intermediate
   local ``p = a*b`` (which would truncate the product to a byte).

3. **The VM step count stays within the model's memory-CAM fidelity window.**  As
   the emitted 30-token/step context accumulates (``O(S^2)`` attention), the model's
   KV-memory address-CAM begins to alias local loads (a documented weakness).
   Measured on this build (GPU cuda:0): a **2-element dot product (96 VM steps)**
   and a **2x2 @ 2x1 matrix-vector product (168 VM steps, 2 outputs)** are BOTH
   byte-exact end-to-end; a full 2x2 @ 2x2 matmul (296 steps via ``fpmul`` calls)
   diverges partway (3/4 result bytes correct).  So the byte-exact demonstrated
   instances are the dot product (atomic matmul op = one layer neuron) and the
   matrix-vector product (a weight matrix applied to an input vector = a layer's
   forward on one token); the full matmul is the largest ATTEMPTED.  The
   ``fpmul``-CALL form is the faithful one; an INLINED ``a*b/s`` variant diverges
   EARLIER (worse CAM traffic).

The kernel is the exact per-MAC inner loop of the fixed-point ONNX runtime
(``onnx_kernel_c4subset.c`` / ``onnx_runtime_nibble_fixedpoint_c4.c``).  Only the
verified op set is used: IMM / PSH / MUL / DIV / ADD + JSR/ENT/ADJ/LEV framing +
LEA/LI/SI locals + PRTF (op 33).  The result fixed-point bytes are emitted over the
PRTF stdout channel — the model's OWN LM-head decode of the numbers it computed.
"""
from __future__ import annotations

from typing import List

#: Fixed-point scale (2^4).  1.0 == ``SCALE``.  Small enough that every stored
#: accumulator byte stays <= 255 for the small operands used here.
SCALE = 16


def dot_c(w: List[int], x: List[int], scale: int = SCALE) -> str:
    """Return C source for a fixed-point 2-element dot product ``c = w . x`` — the
    ATOMIC operation of a matmul (one output element = one weight-row . input, i.e.
    one layer neuron).  ``w`` (a row of the model's own weight matrix) and ``x`` are
    length-2 INTEGER vectors; each entry is scaled by ``scale`` at runtime.  Emits
    the single fixed-point result byte over PRTF.  This is the BYTE-EXACT
    demonstrated instance (96 VM steps).
    """
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    assert len(w) == len(x) == 2
    for label, v in (("w", w), ("x", x)):
        for e in v:
            assert 0 <= e <= 255, f"{label} entry {e} must be a byte literal (0..255)"
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s, w0, w1, x0, x1, c;
  s = {scale};
  w0 = {w[0]}*s; w1 = {w[1]}*s;
  x0 = {x[0]}*s; x1 = {x[1]}*s;
  c = fpmul(w0, x0, s) + fpmul(w1, x1, s);
  printf(c);
  return 0;
}}
'''


def dot_reference(w: List[int], x: List[int], scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point dot product: ``c = sum_k (w[k]*scale *
    x[k]*scale) // scale``.  Returns the single fixed-point result byte (a 1-list,
    to match the PRTF stream).  Asserts it fits in a byte."""
    wq = [w[k] * scale for k in range(2)]
    xq = [x[k] * scale for k in range(2)]
    acc = sum((wq[k] * xq[k]) // scale for k in range(2))
    assert 0 <= acc <= 255, (
        f"dot result {acc} exceeds a byte — shrink the operands")
    return [acc]


def matvec_c(A: List[List[int]], x: List[int], scale: int = SCALE) -> str:
    """Return C source for a fixed-point MATRIX-VECTOR product ``c = A @ x`` — a
    2x2 weight matrix (a slice of the model's OWN weights) applied to a length-2
    input vector, i.e. a genuine "a layer's weight matrix times an input" (2 output
    neurons, 4 MACs).  This is the LARGEST byte-exact demonstrated instance (168 VM
    steps).  Emits the two fixed-point result bytes over PRTF."""
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    flatA = [A[0][0], A[0][1], A[1][0], A[1][1]]
    for label, flat in (("A", flatA), ("x", x)):
        for v in flat:
            assert 0 <= v <= 255, f"{label} entry {v} must be a byte literal (0..255)"
    a0, a1, a2, a3 = flatA
    x0, x1 = x
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s, a0, a1, a2, a3, x0, x1, c0, c1;
  s = {scale};
  a0 = {a0}*s; a1 = {a1}*s; a2 = {a2}*s; a3 = {a3}*s;
  x0 = {x0}*s; x1 = {x1}*s;
  c0 = fpmul(a0, x0, s) + fpmul(a1, x1, s);
  c1 = fpmul(a2, x0, s) + fpmul(a3, x1, s);
  printf(c0);
  printf(c1);
  return 0;
}}
'''


def matvec_reference(A: List[List[int]], x: List[int],
                     scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point matrix-vector product: ``c[i] = sum_k
    (A[i,k]*scale * x[k]*scale) // scale``.  Returns the two fixed-point result
    bytes.  Asserts each fits in a byte."""
    Aq = [[A[i][k] * scale for k in range(2)] for i in range(2)]
    xq = [x[k] * scale for k in range(2)]
    C = []
    for i in range(2):
        acc = sum((Aq[i][k] * xq[k]) // scale for k in range(2))
        assert 0 <= acc <= 255, (
            f"c[{i}]={acc} exceeds a byte — shrink the operands")
        C.append(acc)
    return C


def matmul_c(A: List[List[int]], B: List[List[int]], scale: int = SCALE) -> str:
    """Return C source for a full 2x2 @ 2x2 fixed-point MatMul ``C = A @ B`` (the
    ``fpmul``-call form).  NOTE: at 296 VM steps this EXCEEDS the model's memory-CAM
    fidelity window and diverges partway through the neural forward (3/4 result bytes
    correct) — it is retained as the largest ATTEMPTED instance and for the
    reference/native path; the largest byte-exact demonstrated instance is
    ``matvec_c`` (168 steps, 2 outputs).
    Emits the four fixed-point result bytes over PRTF."""
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    flatA = [A[0][0], A[0][1], A[1][0], A[1][1]]
    flatB = [B[0][0], B[0][1], B[1][0], B[1][1]]
    for label, flat in (("A", flatA), ("B", flatB)):
        for v in flat:
            assert 0 <= v <= 255, f"{label} entry {v} must be a byte literal (0..255)"
    a0, a1, a2, a3 = flatA
    b0, b1, b2, b3 = flatB
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;
  s = {scale};
  a0 = {a0}*s; a1 = {a1}*s; a2 = {a2}*s; a3 = {a3}*s;
  b0 = {b0}*s; b1 = {b1}*s; b2 = {b2}*s; b3 = {b3}*s;
  c0 = fpmul(a0, b0, s) + fpmul(a1, b2, s);
  c1 = fpmul(a0, b1, s) + fpmul(a1, b3, s);
  c2 = fpmul(a2, b0, s) + fpmul(a3, b2, s);
  c3 = fpmul(a2, b1, s) + fpmul(a3, b3, s);
  printf(c0);
  printf(c1);
  printf(c2);
  printf(c3);
  return 0;
}}
'''


def matmul_reference(A: List[List[int]], B: List[List[int]],
                     scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point 2x2 MatMul: ``C[i,j] = sum_k
    (A[i,k]*scale * B[k,j]*scale) // scale``.  Returns the four fixed-point result
    bytes row-major.  Asserts each fits in a byte."""
    Aq = [[A[i][k] * scale for k in range(2)] for i in range(2)]
    Bq = [[B[i][k] * scale for k in range(2)] for i in range(2)]
    C = []
    for i in range(2):
        for j in range(2):
            acc = sum((Aq[i][k] * Bq[k][j]) // scale for k in range(2))
            assert 0 <= acc <= 255, (
                f"C[{i},{j}]={acc} exceeds a byte — shrink the operands")
            C.append(acc)
    return C
