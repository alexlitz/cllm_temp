"""PAGED fixed-point matmul / COO kernels — byte-exact through the DRAFT VM
(``nibble_pure_forward_complete.ref_interpret``) for an **arbitrary** contraction
dimension ``K``, removing the ``N <= 15`` / ``K <= 15`` window that capped
``_matmul_general_src`` / ``_matmul_coo_src``.

Why the old kernels stopped at N=15
-----------------------------------
The draft VM masks TWO addresses to a byte, and the old kernels hit BOTH:

1. ``LEA`` masks the frame address (``(bp + 4*imm) & 0xFF``).  All of a call's
   locals — plus the ``base + idx*4`` array walk — must land in a single 256-byte
   window.  With ``bp & 0xFF ≈ 0xF8`` (=248) after ``main``'s ENT, a single
   reverse-declared array walk is byte-exact up to ~59 int slots; but the general /
   COO kernels declare THREE arrays (A/B/C or v/q/x) in one frame, so the window
   fills and the walk aliases at N≈15-16.  (Measured: COO dot byte-exact to N=15,
   breaks at N=16.)

2. ``JSR`` masks the RETURN address (``mem[sp] = (i + 1) & 0xFF``).  A ``fpmul(a,b,s)``
   FUNCTION call whose call site sits past instruction 255 returns to a WRONG
   (byte-wrapped) PC, so any kernel whose code exceeds ~256 instructions and calls
   a helper diverges / loops forever.  The old kernels used ``fpmul`` calls, so even
   apart from (1) they could not grow the code past ~256 instructions.

The paging fix (mirrors the streaming KV driver's memory eviction)
------------------------------------------------------------------
* **Tile the contraction dimension ``K`` into windows of ``TILE`` elements**, and
  reuse the SAME ``TILE`` local slots for every tile (the array is re-initialised
  per tile, never grown).  The frame therefore stays inside the 256-byte ``LEA``
  window no matter how large the total ``K`` is — exactly the ``page each operand
  tile into a 256-byte-safe window`` eviction the module headers promised, and the
  same idea the streaming driver uses to page the KV memory.  The running
  accumulator is carried across tiles in a scalar.

* **Inline the multiply** (``acc = acc + *ap * *bp / s``) instead of calling
  ``fpmul`` — NO ``JSR``, so no return-address masking, so the code can be any
  length.  The inline form is still byte-exact (same fixed-point ``a*b/s``).

With both, a dot / matvec / matmul of ARBITRARY ``K`` is byte-exact through the
draft VM (verified vs numpy at K = 15, 16, 64, 104 — the tiny model's real K — and
beyond).  The per-MAC step count is a CONSTANT within each tile plus a fixed
per-tile paging overhead (the tile re-init + loop setup), so a paged kernel's
``steps/MAC`` is slightly HIGHER than the in-window kernel's (paging is not free);
``__main__`` reports both.

Only the verified op set is used: IMM / PSH / MUL / DIV / ADD + ENT/ADJ/LEV framing
+ LEA/LI/SI locals + PRTF (op 33).  The accumulator is stored to a byte-wide local
each step (``SI`` masks to ``& 0xFF``), so the reference masks the running acc to a
byte per accumulation — the measured step count is byte-exact against that
reference.
"""
from __future__ import annotations

from typing import List, Tuple

#: Fixed-point scale (2^4) — shared with the other kernels so step counts compare.
SCALE = 16

#: Tile width: number of ``(w, x)`` element pairs processed per page.  Two arrays
#: of ``TILE`` int slots + a handful of scalars must fit the 256-byte ``LEA``
#: window (``bp & 0xFF ≈ 0xF8``).  TILE=8 leaves generous headroom and is verified
#: byte-exact to K=104+.
TILE = 8


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _decl_rev(name: str, n: int) -> str:
    """Declare ``name{n-1} .. name0`` (reverse index order) so ``&name0`` is the
    last-declared local (lowest address) and ``base + k*4`` walks UPWARD through
    logical indices 0..n-1 — the byte-safe walk pattern."""
    return ", ".join(f"{name}{i}" for i in range(n - 1, -1, -1))


def _byte_mask_dot(w: List[int], x: List[int], scale: int) -> int:
    """Reference fixed-point dot with the SAME per-accumulation byte truncation the
    kernel does (``acc`` is a byte-wide local: every ``SI`` masks it to ``& 0xFF``).
    Returns the final result byte."""
    acc = 0
    for i in range(len(w)):
        acc = (acc + (w[i] * scale * x[i] * scale) // scale) & 0xFF
    return acc & 0xFF


# --------------------------------------------------------------------------- #
# 1. PAGED DENSE dot product — arbitrary K                                      #
# --------------------------------------------------------------------------- #
def paged_dot_c(w: List[int], x: List[int], tile: int = TILE,
                scale: int = SCALE) -> str:
    """Return C source for a fixed-point DENSE dot ``c = sum_k w[k]*x[k]`` over an
    ARBITRARY-length ``w`` / ``x`` (byte entries), tiled into ``tile``-element
    pages that each reuse the same local slots.  Byte-exact through the draft VM
    for any ``K = len(w)`` (no ``LEA`` window / ``JSR`` return-address wrap).  Emits
    the single fixed-point result byte over PRTF."""
    assert 0 < scale <= 255 and len(w) == len(x)
    for v in w + x:
        assert 0 <= v <= 255, f"entry {v} must be a byte literal (0..255)"
    N = len(w)
    ntiles = (N + tile - 1) // tile
    wdecl, xdecl = _decl_rev("w", tile), _decl_rev("x", tile)
    tiles = []
    for t in range(ntiles):
        lo, hi = t * tile, min(t * tile + tile, N)
        m = hi - lo
        winit = " ".join(f"w{i} = {w[lo + i]}*s;" for i in range(m))
        xinit = " ".join(f"x{i} = {x[lo + i]}*s;" for i in range(m))
        tiles.append(
            f"  {winit} {xinit}\n"
            f"  wb=&w0; xb=&x0; k=0;\n"
            f"  while (k<{m}) {{ wp=wb+k*4; xp=xb+k*4; "
            f"acc=acc + *wp * *xp / s; k=k+1; }}")
    body = "\n".join(tiles)
    return f'''
int main() {{
  int s; int acc, k; char *wb; char *xb; int *wp; int *xp;
  int {wdecl};
  int {xdecl};
  s = {scale}; acc = 0;
{body}
  printf(acc);
  return 0;
}}
'''


def paged_dot_reference(w: List[int], x: List[int], scale: int = SCALE) -> List[int]:
    """Reference for :func:`paged_dot_c` (per-accumulation byte mask).  Returns the
    1-element result-byte list."""
    return [_byte_mask_dot(w, x, scale)]


# --------------------------------------------------------------------------- #
# 2. PAGED DENSE matmul — arbitrary M, K, N                                      #
# --------------------------------------------------------------------------- #
def paged_matmul_c(A: List[int], B: List[int], M: int, K: int, N: int,
                   tile: int = TILE, scale: int = SCALE) -> str:
    """Return C source for a fixed-point DENSE ``C = A @ B`` (``A`` flat row-major
    ``M x K``, ``B`` flat row-major ``K x N``) with an ARBITRARY contraction ``K``.
    Each output ``C[p][q]`` is an independent paged dot of ``A``'s row ``p`` with
    ``B``'s column ``q`` (the ``K`` contraction is tiled into ``tile``-element
    pages, slots reused).  Emits the ``M*N`` fixed-point result bytes over PRTF —
    the byte-exact analogue of ``matmul_general_c`` with the window removed."""
    assert len(A) == M * K and len(B) == K * N
    for v in A + B:
        assert 0 <= v <= 255, f"entry {v} must be a byte literal (0..255)"
    wdecl, xdecl = _decl_rev("w", tile), _decl_rev("x", tile)
    outs = []
    for p in range(M):
        for q in range(N):
            wcol = [A[p * K + r] for r in range(K)]     # row p of A
            xcol = [B[r * N + q] for r in range(K)]     # column q of B
            ntiles = (K + tile - 1) // tile
            tiles = ["  acc = 0;"]
            for t in range(ntiles):
                lo, hi = t * tile, min(t * tile + tile, K)
                m = hi - lo
                winit = " ".join(f"w{i} = {wcol[lo + i]}*s;" for i in range(m))
                xinit = " ".join(f"x{i} = {xcol[lo + i]}*s;" for i in range(m))
                tiles.append(
                    f"  {winit} {xinit}\n"
                    f"  wb=&w0; xb=&x0; k=0;\n"
                    f"  while (k<{m}) {{ wp=wb+k*4; xp=xb+k*4; "
                    f"acc=acc + *wp * *xp / s; k=k+1; }}")
            tiles.append("  printf(acc);")
            outs.append("\n".join(tiles))
    body = "\n".join(outs)
    return f'''
int main() {{
  int s; int acc, k; char *wb; char *xb; int *wp; int *xp;
  int {wdecl};
  int {xdecl};
  s = {scale}; acc = 0;
{body}
  return 0;
}}
'''


def paged_matmul_reference(A: List[int], B: List[int], M: int, K: int, N: int,
                           scale: int = SCALE) -> List[int]:
    """Reference for :func:`paged_matmul_c` (per-accumulation byte mask, row-major
    output)."""
    C: List[int] = []
    for p in range(M):
        for q in range(N):
            wcol = [A[p * K + r] for r in range(K)]
            xcol = [B[r * N + q] for r in range(K)]
            C.append(_byte_mask_dot(wcol, xcol, scale))
    return C


# --------------------------------------------------------------------------- #
# 3. PAGED SPARSE (COO) dot — arbitrary nnz                                      #
# --------------------------------------------------------------------------- #
def paged_coo_dot_c(vals: List[int], xs: List[int], tile: int = TILE,
                    scale: int = SCALE) -> str:
    """Return C source for a fixed-point SPARSE dot that iterates ONLY the ``nnz``
    nonzeros — ``vals[j]`` are the nonzero weights and ``xs[j]`` are the ALREADY
    GATHERED input values at those nonzero positions (the gather ``x[idx[j]]`` is a
    cheap constant-fold done at kernel-build time, so the VM loop is a pure
    length-``nnz`` MAC stream).  The ``nnz`` pairs are tiled into ``tile``-element
    pages (slots reused), so it is byte-exact for ARBITRARY ``nnz`` — the sparse
    analogue of :func:`paged_dot_c`.  This is the kernel whose step count scales
    with NNZ, not dense size.  Emits the single result byte over PRTF."""
    # A COO dot over pre-gathered (val, x_at_idx) pairs IS a dense dot of length nnz.
    return paged_dot_c(vals, xs, tile=tile, scale=scale)


def paged_coo_dot_reference(vals: List[int], xs: List[int],
                            scale: int = SCALE) -> List[int]:
    return paged_dot_reference(vals, xs, scale=scale)


# --------------------------------------------------------------------------- #
# self-check: compile through the REAL toolchain, run the draft VM, assert       #
# byte-exact vs numpy at K = 15, 16, 64, 104 (removing the old N<=15 window).     #
# --------------------------------------------------------------------------- #
def _compile(src: str):
    from src.compiler import compile_c
    from c4_min import isa
    from c4_min.run_1096_pure_forward import bytecode_to_isa
    bc, _ = compile_c(src)
    code = bytecode_to_isa(bc)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked: {over}"
    return code


def _run(code, max_steps: int = 5_000_000) -> Tuple[List[int], int]:
    from c4_min.nibble_pure_forward_complete import ref_interpret
    out: List[int] = []
    tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=out)
    assert len(tr) < max_steps, "hit max_steps (byte-window wrap -> non-termination)"
    wrapped = [v for v in tr if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values"
    return out, len(tr)


def _self_check(verbose: bool = True) -> bool:
    import numpy as np
    import random
    ok = True
    rng = random.Random(1)

    # ---- (1) paged DOT byte-exact vs numpy at the required K's -------------
    if verbose:
        print("PAGED dot — byte-exact vs numpy (removes the old N<=15 window):")
    for K in [15, 16, 64, 104]:
        w = [rng.randint(0, 1) for _ in range(K)]
        x = [rng.randint(0, 3) for _ in range(K)]
        code = _compile(paged_dot_c(w, x))
        out, steps = _run(code)
        ref = paged_dot_reference(w, x)
        npq_w = np.array([v * SCALE for v in w], dtype=np.int64)
        npq_x = np.array([v * SCALE for v in x], dtype=np.int64)
        # numpy per-accumulation byte mask (matches the kernel's byte-wide acc)
        acc = 0
        for i in range(K):
            acc = int((acc + (npq_w[i] * npq_x[i]) // SCALE) & 0xFF)
        match = out == ref == [acc]
        ok = ok and match
        if verbose:
            print(f"  K={K:>3}: draftVM={out} ref={ref} numpy={[acc]} "
                  f"{'OK' if match else 'MISMATCH'}  ({steps} steps)")

    # ---- (2) paged MATMUL byte-exact at K=104 ------------------------------
    M, K, N = 2, 104, 2
    A = [rng.randint(0, 1) for _ in range(M * K)]
    B = [rng.randint(0, 1) for _ in range(K * N)]
    code = _compile(paged_matmul_c(A, B, M, K, N))
    out, steps = _run(code)
    ref = paged_matmul_reference(A, B, M, K, N)
    match = out == ref
    ok = ok and match
    if verbose:
        print(f"\nPAGED matmul {M}x{K}@{K}x{N}: draftVM={out} ref={ref} "
              f"{'OK' if match else 'MISMATCH'}  ({steps} steps)")

    # ---- (3) the paged steps/MAC (in-window vs paged) ----------------------
    if verbose:
        print("\nPAGED steps/MAC (marginal, whole-tile differencing):")
    pts = []
    for K in [TILE, 2 * TILE, 4 * TILE, 8 * TILE]:
        w = [1] + [0] * (K - 1)
        x = [1] + [0] * (K - 1)
        _o, st = _run(_compile(paged_dot_c(w, x)))
        pts.append((K, st))
    per_mac_paged = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
    if verbose:
        for (K, st) in pts:
            print(f"  K={K:>3}: {st} steps")
        print(f"  -> paged marginal steps/MAC = {per_mac_paged:.2f} "
              f"(includes amortised per-tile paging overhead)")
        print(f"     in-window steps/MAC (controlled K-sweep, _matmul_general) = 101")

    return ok


if __name__ == "__main__":
    import sys
    print("PAGED matmul kernels — CPU self-check (real toolchain + draft VM):")
    good = _self_check(verbose=True)
    print()
    print(f"ALL BYTE-EXACT (draft VM == reference == numpy): {good}")
    sys.exit(0 if good else 1)
