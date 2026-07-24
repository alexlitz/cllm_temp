"""GENERAL fixed-point matmul C-source generator in the c4 subset — a byte-exact
arbitrary ``M x K @ K x N`` kernel that runs through the ACTUAL draft VM
(``nibble_pure_forward_complete.ref_interpret``), used to MEASURE the true
``steps/MAC`` rate at real scale.

Why this module exists
----------------------
``docs/SELFHOST_3LAYER_FEASIBILITY.md`` reports **≈ 4.75 VM steps / MAC**, a
LINEAR FIT of a 2x2x2 vs 8x8x8 matmul.  That figure is the marginal rate of
``onnx_kernel_c4subset.c`` on the UNMASKED ``src.compiler`` reference VM
(``run_selfhost_feasibility.count_vm_steps``) — a machine with a real ``malloc``
heap, full 32-bit ``LEA`` addressing, and an INLINED ``p >> 12`` fpmul.  The
tiny-model self-forward step count ``503,776 MACs x 4.75 ≈ 2,392,936`` then
attributes that rate to the DRAFT VM (``ref_interpret``, the byte-masking,
model-faithful interpreter the whole self-emulation story is about).  Those are
DIFFERENT machines: the draft VM masks ``LEA`` to a byte (needs a paged /
windowed local-array walk, the COO trick), and a byte-exact fixed-point divide is
a ``MUL``+``DIV`` (or a ``fpmul`` function call), not a single ``SHR``.

This kernel is the honest general matmul in the SAME c4 subset as
``_matmul_src.py`` / ``_matmul_coo_src.py`` — locals-only, ``IMM <= 255``, stores
``<= 255`` non-negative, ``fpmul = a*b/s`` — so it is byte-exact through
``ref_interpret`` and its measured ``steps/MAC`` is the GROUNDED draft-VM rate.

Byte-safety envelope (same as the COO kernel)
--------------------------------------------
The three operand/result matrices ``A`` (``M*K``), ``B`` (``K*N``), ``C``
(``M*N``) are laid out as consecutive **REVERSE-declared locals** and walked with
a hand-built byte-safe address (``base = &x0``; ``xp = base + idx*4``; a ``char*``
cast gives the ``+`` a stride of 1 and ``idx*4`` is the byte offset between
adjacent int locals).  ``LEA`` masks the frame address to its low byte
(``(bp + 4*imm) & 0xFF``), and ``bp ≈ 0xF8`` after ``main``'s ``ENT``, so every
walked address ``base + idx*4`` must stay in ``[0, 256)``.  That bounds the number
of locals a SINGLE call can hold (empirically ~35 int slots), which caps the
directly-runnable ``M,K,N``.  This bounds the DEMO size, not the algorithm: the
marginal ``steps/MAC`` is a CONSTANT within the window (each extra inner-loop MAC
adds a fixed step count, independent of the matrix size), which is exactly the
quantity being measured — differencing two in-window sizes cancels every
size-independent term.  A production run would page each operand tile into a
256-byte-safe window (the same eviction the streaming driver already does), which
does not change the per-MAC rate.

Only the verified op set is used: IMM / PSH / MUL / DIV / ADD + JSR/ENT/ADJ/LEV
framing + LEA/LI/SI locals + PRTF (op 33).
"""
from __future__ import annotations

from typing import List

#: Fixed-point scale (2^4) — shared with ``_matmul_src.SCALE`` so the kernels are
#: directly comparable.  Small enough that every stored byte stays 0..255.
SCALE = 16


def _decl_rev(name: str, n: int) -> str:
    """Declare ``name{n-1} .. name0`` (reverse index order) so ``&name0`` is the
    last-declared local (lowest address) and ``base + k*4`` walks UPWARD through
    logical indices 0..n-1 — the byte-safe walk pattern (see module header)."""
    return ", ".join(f"{name}{i}" for i in range(n - 1, -1, -1))


def _init_fp(name: str, arr: List[int]) -> str:
    """Init each ``name{i}`` to ``arr[i]*s`` (runtime MUL of two byte literals —
    the fixed-point scaling, NOT constant-folded, so no IMM > 255)."""
    return " ".join(f"{name}{i} = {arr[i]}*s;" for i in range(len(arr)))


def _validate(A: List[int], B: List[int], M: int, K: int, N: int,
              scale: int) -> None:
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    assert len(A) == M * K, f"A must be flat M*K={M*K}, got {len(A)}"
    assert len(B) == K * N, f"B must be flat K*N={K*N}, got {len(B)}"
    for lbl, flat in (("A", A), ("B", B)):
        for v in flat:
            assert 0 <= v <= 255, f"{lbl} entry {v} must be a byte literal (0..255)"


def matmul_general_c(A: List[int], B: List[int], M: int, K: int, N: int,
                     scale: int = SCALE, call_fpmul: bool = True) -> str:
    """Return C source for a fixed-point ``C = A @ B`` with ``A`` a flat row-major
    ``M x K`` and ``B`` a flat row-major ``K x N`` (both length-checked, byte
    entries).  Triple nested loop ``p (rows) / q (cols) / r (contraction)`` — the
    SAME structure as ``onnx_kernel_c4subset.c``'s ``matmul()`` inner loop, so the
    measured per-MAC step count is the honest unit cost of the runtime's matmul.
    Emits the ``M*N`` fixed-point result bytes over PRTF.

    ``call_fpmul=True`` uses the ``fpmul(a,b,s) = a*b/s`` FUNCTION form (the
    faithful one, matching ``_matmul_src``); ``False`` inlines ``*ap * *bp / s`` in
    the accumulate expression (still byte-exact, saves the per-MAC JSR/ENT/LEV
    frame).  Both are byte-exact; the two rates bracket the real per-MAC cost.
    """
    _validate(A, B, M, K, N, scale)
    a_decl = _decl_rev("a", M * K)
    b_decl = _decl_rev("b", K * N)
    c_decl = _decl_rev("c", M * N)
    prelude = (
        "int fpmul(int a, int b, int s) { return a * b / s; }\n"
        if call_fpmul else "")
    mac = ("acc = acc + fpmul(*ap, *bp2, s);" if call_fpmul
           else "acc = acc + *ap * *bp2 / s;")
    return f'''
{prelude}int main() {{
  int s;
  int {a_decl};
  int {b_decl};
  int {c_decl};
  int p, q, r, acc;
  char *ab; char *bb; char *cb;
  int *ap; int *bp2; int *cp;
  s = {scale};
  {_init_fp("a", A)}
  {_init_fp("b", B)}
  ab = &a0; bb = &b0; cb = &c0;
  p = 0;
  while (p < {M}) {{
    q = 0;
    while (q < {N}) {{
      acc = 0; r = 0;
      while (r < {K}) {{
        ap = ab + (p * {K} + r) * 4;
        bp2 = bb + (r * {N} + q) * 4;
        {mac}
        r = r + 1;
      }}
      cp = cb + (p * {N} + q) * 4;
      *cp = acc;
      q = q + 1;
    }}
    p = p + 1;
  }}
  p = 0;
  while (p < {M * N}) {{
    cp = cb + p * 4;
    printf(*cp);
    p = p + 1;
  }}
  return 0;
}}
'''


def matmul_general_reference(A: List[int], B: List[int], M: int, K: int, N: int,
                             scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point ``C = A @ B``: ``C[i,j] = (sum_k
    (A[i,k]*scale * B[k,j]*scale) // scale) & 0xFF`` row-major.  The ``& 0xFF``
    matches the draft VM's ``SI`` store (a local is one byte).  Returns the
    ``M*N`` result bytes."""
    _validate(A, B, M, K, N, scale)
    C: List[int] = []
    for i in range(M):
        for j in range(N):
            acc = sum((A[i * K + k] * scale * B[k * N + j] * scale) // scale
                      for k in range(K))
            C.append(acc & 0xFF)
    return C
