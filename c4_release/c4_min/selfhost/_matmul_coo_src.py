"""C-source generators for the SPARSE (COO) self-emulation kernel — the
fixed-point inner op of a transformer layer emulated over ONLY its nonzero
weights, so the VM step count scales with **NNZ, not dense size**.

This is the sparse companion to ``_matmul_src.py`` (the *dense* dot / matvec /
matmul demo).  The emulated c4_min model is ~99.99% sparse, so emulating a DENSE
forward (``_matmul_src.dot_c`` iterates EVERY entry) is the wrong computation:
a dense length-``N`` dot costs ``O(N)`` VM steps, but only ``k`` of those entries
are nonzero.  The COO (coordinate) kernel here iterates ONLY the ``k`` nonzero
``(val, idx)`` pairs — ``acc += val[j] * x[idx[j]] / scale`` — so its step count
is ``O(k)``.  That is the ~10^4x MAC lever: for a 99.99%-sparse row the sparse
kernel does ``k`` MACs where the dense one does ``N ≈ 10^4 * k``.

Everything that makes ``_matmul_src`` byte-exact through the reference VM
(``nibble_pure_forward_complete.ref_interpret``) is preserved verbatim; read that
module's header first.  The load-bearing rules, and how this kernel honours each:

1. **Every emitted ``IMM`` immediate is <= 255.**  Operands, indices and array
   lengths are byte-sized; scaled operands (``entry * scale``) are built via a
   runtime ``MUL`` of two byte literals; the kernel references ONLY locals (a
   GLOBAL would emit its data-seg ADDRESS > 255 as an IMM).

2. **Every value STORED to a local stays <= 255 and NON-NEGATIVE.**  ``scale`` is
   small (16) and operands are small so every accumulator byte fits 0..255 (a
   wrapped-negative AX corrupts the next ``LEA``); the MAC product ``a*b`` lives
   only in the 32-bit AX and is divided by ``scale`` BEFORE any store, so
   ``fpmul`` is ``return a*b/s`` with NO intermediate local (which would truncate
   to a byte).

3. **Byte-safe indexed local access (the sparse-specific part).**  c4 has NO
   ``int A[n]`` array-decl syntax, and ``malloc`` links a stdlib whose heap base
   leaks as an ``IMM > 255`` (and blows the code-size budget) — so ``vals``,
   ``idxs`` and ``x`` are laid out as **consecutive LOCALS** and walked with a
   hand-built byte-safe address.  Two c4 quirks force the exact pattern:

     * Local ``int``s are **4 bytes apart** (``LEA`` computes ``bp + 4*imm``), but
       the ``p[k]`` subscript multiplies ``k`` by the ``sizeof(int)==8`` element
       stride — a 4-vs-8 mismatch that makes ``p[k]`` skip every other slot.  So
       the walk is done as ``ip = (char*)base + k*4; *ip`` — a ``char*`` cast
       gives the ``+`` a stride of 1, and ``k*4`` (a small, byte-safe product) is
       the literal byte offset between adjacent int locals.
     * Consecutive locals grow toward LOWER addresses (the LAST-declared local
       has the most-negative ``LEA`` offset), so each array is declared in
       **REVERSE index order** (``x{N-1} .. x0``) and ``base = &x0`` — the
       last-declared element, the lowest address — so ``base + k*4`` walks UPWARD
       through logical indices ``0, 1, 2, ...``.

   The indexed load ``x[idx[j]]`` is the double indirection ``ix = *(idxb + j*4);
   *(xb + ix*4)`` — both offsets (``j*4`` and ``ix*4``) are small runtime products
   of byte quantities, never an ``IMM > 255``.

**Byte-safety envelope (honest).**  ``LEA`` masks the frame address to its low
byte (``(bp + 4*imm) & 0xFF``), so when the local frame grows past ~one 256-byte
window the walked addresses ``base + k*4`` alias other slots and the kernel
diverges from the reference VM.  Measured on the reference VM (``SP_INIT =
0x10000``, ``bp_low`` after ``main``'s ENT ≈ ``0xF8``): the COO dot is byte-exact
for a length-``N`` vector up to **N = 15** and breaks at N = 16 (the point the
``x``-array walk crosses the 256-byte boundary).  This bounds the *demo* size, not
the *algorithm*: the step-count scaling ``O(k)`` is exact WITHIN the window (each
extra nonzero costs a fixed ~101 VM steps, independent of N; see the table printed
by ``__main__``), which is the property being demonstrated.  A production run
would page the operand into 256-byte-safe local windows (the same eviction the
streaming driver already does for the KV memory) — the workaround, not a change to
the kernel.  Only the verified op set is used: IMM / PSH / MUL / DIV / ADD +
JSR/ENT/ADJ/LEV framing + LEA/LI/SI/LC locals + PRTF (op 33); the fixed-point
result bytes are emitted over PRTF, exactly like the dense kernels.
"""
from __future__ import annotations

from typing import List, Tuple

#: Fixed-point scale (2^4).  1.0 == ``SCALE``.  Small enough that every stored
#: accumulator byte stays <= 255 for the small sparse operands used here.  Shared
#: with ``_matmul_src.SCALE`` so the two kernels are directly comparable.
SCALE = 16

#: Largest length-N the COO dot stays byte-exact through the reference VM before
#: the LEA frame-byte wrap aliases the indexed-load address (see the module
#: header, "Byte-safety envelope").  A demo array MUST keep ``len(x) <= N_MAX``.
N_MAX = 15


# --------------------------------------------------------------------------- #
# small helpers for laying an array out as consecutive REVERSE-declared locals  #
# --------------------------------------------------------------------------- #
def _decl_rev(name: str, n: int) -> str:
    """Declare ``name{n-1} .. name0`` (reverse index order) so ``&name0`` is the
    last-declared local (lowest address) and ``base + k*4`` walks UPWARD through
    logical indices 0..n-1."""
    return ", ".join(f"{name}{i}" for i in range(n - 1, -1, -1))


def _init_fp(name: str, arr: List[int]) -> str:
    """Init each ``name{i}`` to ``arr[i]*s`` (runtime MUL of two byte literals —
    the fixed-point scaling, NOT constant-folded, so no IMM > 255)."""
    return " ".join(f"{name}{i} = {arr[i]}*s;" for i in range(len(arr)))


def _init_raw(name: str, arr: List[int]) -> str:
    """Init each ``name{i}`` to the raw byte ``arr[i]`` (used for the position
    indices ``idxs`` / row boundaries, which are unscaled)."""
    return " ".join(f"{name}{i} = {arr[i]};" for i in range(len(arr)))


def _validate(vals: List[int], idxs: List[int], x: List[int], scale: int) -> None:
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    assert len(vals) == len(idxs), "vals[] and idxs[] must be parallel (same nnz)"
    assert len(x) <= N_MAX, (
        f"len(x)={len(x)} exceeds the byte-safe demo window N_MAX={N_MAX} "
        f"(LEA frame-byte wrap; see module header)")
    for v in vals:
        assert 0 <= v <= 255, f"vals entry {v} must be a byte literal (0..255)"
    for i in idxs:
        assert 0 <= i < len(x), f"idx {i} out of range for x of length {len(x)}"
    for v in x:
        assert 0 <= v <= 255, f"x entry {v} must be a byte literal (0..255)"


# --------------------------------------------------------------------------- #
# COO sparse dot product                                                        #
# --------------------------------------------------------------------------- #
def coo_dot_c(vals: List[int], idxs: List[int], x: List[int],
              scale: int = SCALE) -> str:
    """Return C source for a fixed-point SPARSE dot product ``c = sum_j vals[j] *
    x[idxs[j]]`` — a weight vector given as parallel ``vals[]`` (the nonzero
    values) + ``idxs[]`` (their positions in the dense length-``N`` row) dotted
    with a dense input ``x[]``.  The loop iterates ONLY the ``nnz = len(vals)``
    nonzeros, so the VM step count is ``O(nnz)`` — independent of ``N`` beyond the
    (cheap) one-time ``x`` setup.  Emits the single fixed-point result byte over
    PRTF.  See the module header for the byte-safe indexed-access pattern."""
    _validate(vals, idxs, x, scale)
    nnz, N = len(vals), len(x)
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s;
  int {_decl_rev("v", nnz)};
  int {_decl_rev("q", nnz)};
  int {_decl_rev("x", N)};
  int acc, k, ix;
  char *vb;
  char *qb;
  char *xb;
  int *vp;
  int *qp;
  int *xp;
  s = {scale};
  {_init_fp("v", vals)}
  {_init_raw("q", idxs)}
  {_init_fp("x", x)}
  vb = &v0; qb = &q0; xb = &x0;
  acc = 0; k = 0;
  while (k < {nnz}) {{
    qp = qb + k * 4; ix = *qp;
    vp = vb + k * 4;
    xp = xb + ix * 4;
    acc = acc + fpmul(*vp, *xp, s);
    k = k + 1;
  }}
  printf(acc);
  return 0;
}}
'''


def coo_dot_reference(vals: List[int], idxs: List[int], x: List[int],
                      scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point SPARSE dot product: ``c = sum_j
    (vals[j]*scale * x[idxs[j]]*scale) // scale`` over the ``nnz`` nonzeros —
    IDENTICAL fixed-point semantics to ``_matmul_src.dot_reference``, restricted
    to the nonzero terms (the zero terms contribute 0, so this equals the dense
    reference on the same row).  Returns the single fixed-point result byte (a
    1-list, matching the PRTF stream).  Asserts it fits in a byte."""
    acc = 0
    for j in range(len(vals)):
        vq = vals[j] * scale
        xq = x[idxs[j]] * scale
        acc += (vq * xq) // scale
    assert 0 <= acc <= 255, (
        f"coo dot result {acc} exceeds a byte — shrink the operands")
    return [acc]


# --------------------------------------------------------------------------- #
# COO sparse matrix-vector product (per-row nonzeros, CSR-style row boundaries)  #
# --------------------------------------------------------------------------- #
def coo_matvec_c(rows: int, N: int, vals: List[int], cols: List[int],
                 row_start: List[int], x: List[int], scale: int = SCALE) -> str:
    """Return C source for a fixed-point SPARSE matrix-vector product ``c = A @ x``
    where ``A`` is a ``rows x N`` sparse matrix in COO/CSR form: ``vals[]`` +
    ``cols[]`` are the flat nonzero values / column indices, and ``row_start[]``
    (length ``rows+1``) marks each output row's slice ``[row_start[i],
    row_start[i+1])``.  The outer loop is over output rows, the inner over that
    row's nonzeros ONLY, so the total inner-loop work is ``O(total_nnz)`` — the
    sparse analogue of ``_matmul_src.matvec_c`` (which does the full ``rows*N``
    MACs).  Emits one fixed-point result byte per row over PRTF."""
    assert 0 < scale <= 255, f"scale={scale} must be a byte literal"
    assert len(row_start) == rows + 1, "row_start must have rows+1 boundaries"
    assert len(vals) == len(cols), "vals[]/cols[] must be parallel"
    assert N <= N_MAX, f"N={N} exceeds byte-safe window N_MAX={N_MAX}"
    nnz = len(vals)
    for v in vals:
        assert 0 <= v <= 255, f"vals entry {v} must be a byte literal"
    for c in cols:
        assert 0 <= c < N, f"col {c} out of range for N={N}"
    for v in x:
        assert 0 <= v <= 255, f"x entry {v} must be a byte literal"
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s;
  int {_decl_rev("v", nnz)};
  int {_decl_rev("c", nnz)};
  int {_decl_rev("r", rows + 1)};
  int {_decl_rev("x", N)};
  int acc, i, k, ix, ks, ke;
  char *vb;
  char *cb;
  char *rb;
  char *xb;
  int *vp;
  int *cp;
  int *rp;
  int *xp;
  s = {scale};
  {_init_fp("v", vals)}
  {_init_raw("c", cols)}
  {_init_raw("r", row_start)}
  {_init_fp("x", x)}
  vb = &v0; cb = &c0; rb = &r0; xb = &x0;
  i = 0;
  while (i < {rows}) {{
    rp = rb + i * 4; ks = *rp;
    rp = rb + (i + 1) * 4; ke = *rp;
    acc = 0; k = ks;
    while (k < ke) {{
      cp = cb + k * 4; ix = *cp;
      vp = vb + k * 4;
      xp = xb + ix * 4;
      acc = acc + fpmul(*vp, *xp, s);
      k = k + 1;
    }}
    printf(acc);
    i = i + 1;
  }}
  return 0;
}}
'''


def coo_matvec_reference(rows: int, N: int, vals: List[int], cols: List[int],
                         row_start: List[int], x: List[int],
                         scale: int = SCALE) -> List[int]:
    """The numpy/reference fixed-point SPARSE matrix-vector product: for each row
    ``i``, ``c[i] = sum_{k in [row_start[i], row_start[i+1])} (vals[k]*scale *
    x[cols[k]]*scale) // scale`` — same fixed-point semantics as
    ``_matmul_src.matvec_reference`` over the row's nonzeros.  Returns one result
    byte per row.  Asserts each fits in a byte."""
    C: List[int] = []
    for i in range(rows):
        acc = 0
        for k in range(row_start[i], row_start[i + 1]):
            vq = vals[k] * scale
            xq = x[cols[k]] * scale
            acc += (vq * xq) // scale
        assert 0 <= acc <= 255, (
            f"c[{i}]={acc} exceeds a byte — shrink the operands")
        C.append(acc)
    return C


# --------------------------------------------------------------------------- #
# the DENSE baseline, as a loop over ALL N entries (for the step-count table)    #
# --------------------------------------------------------------------------- #
def dense_dot_unroll_c(w: List[int], x: List[int], scale: int = SCALE) -> str:
    """Return C source for the DENSE fixed-point dot product UNROLLED over ALL
    ``N`` entries — exactly the ``_matmul_src.dot_c`` form (distinct scalars +
    ``fpmul`` per entry, no loop, no array walk), so it stays byte-exact for
    larger ``N`` (it never hits the array-walk frame-wrap) and its VM step count
    is a clean ``O(N)`` (~82 steps/entry; N=2 == the documented 96-step dot).
    This is the honest ``O(N)`` dense baseline for the step-count table."""
    N = len(w)
    wn = ", ".join(f"w{i}" for i in range(N))
    xn = ", ".join(f"x{i}" for i in range(N))
    wi = " ".join(f"w{i} = {w[i]}*s;" for i in range(N))
    xi = " ".join(f"x{i} = {x[i]}*s;" for i in range(N))
    terms = " + ".join(f"fpmul(w{i}, x{i}, s)" for i in range(N))
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s;
  int {wn};
  int {xn};
  int c;
  s = {scale};
  {wi}
  {xi}
  c = {terms};
  printf(c);
  return 0;
}}
'''


def dense_dot_loop_c(w: List[int], x: List[int], scale: int = SCALE) -> str:
    """Return C source for the DENSE fixed-point dot product as a LOOP over ALL
    ``N`` entries (``acc += w[k]*x[k]/s`` for every ``k``), so its VM step count is
    ``O(N)`` — the sparse kernel's honest ``O(k)`` competitor.  Uses the SAME
    byte-safe local-array walk as the COO kernel (so the comparison isolates
    ``k`` vs ``N``, not the addressing scheme).  NOTE: because it declares ALL
    ``N`` entries as a local array, it hits the SAME frame-byte-wrap window as the
    COO kernel (byte-exact only up to N ~= the COO envelope); for the larger-N
    step-count table use ``dense_dot_unroll_c`` (loop-free, so it stays exact)."""
    N = len(w)
    return f'''
int fpmul(int a, int b, int s) {{
  return a * b / s;
}}
int main() {{
  int s;
  int {_decl_rev("w", N)};
  int {_decl_rev("x", N)};
  int acc, k;
  char *wb;
  char *xb;
  int *wp;
  int *xp;
  s = {scale};
  {_init_fp("w", w)}
  {_init_fp("x", x)}
  wb = &w0; xb = &x0;
  acc = 0; k = 0;
  while (k < {N}) {{
    wp = wb + k * 4;
    xp = xb + k * 4;
    acc = acc + fpmul(*wp, *xp, s);
    k = k + 1;
  }}
  printf(acc);
  return 0;
}}
'''


# =========================================================================== #
# self-check: compile through the REAL toolchain, run the reference/draft VM,   #
# assert byte-exact vs numpy, and report the O(k) vs O(N) step scaling.          #
# =========================================================================== #
def _compile_ok(src: str):
    """Compile via the REAL c4 toolchain, translate to ISA, assert no IMM > 255
    leaked (which would diverge the byte-masking reference from the model)."""
    from src.compiler import compile_c
    from c4_min import isa
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bytecode, _data = compile_c(src)
    code = bytecode_to_isa(bytecode)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from ref): {over}"
    return code


def _run(code) -> Tuple[List[int], int]:
    """Run the DETERMINISTIC draft VM (``ref_interpret``) at 32-bit width; return
    (PRTF bytes, draft step count).  Asserts no wrapped-negative AX."""
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out: List[int] = []
    trace = ref_interpret(code, max_steps=300000, mask=0xFFFFFFFF, out=out)
    wrapped = [v for v in trace if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values (would corrupt LEA)"
    return out, len(trace)


def _dense_from_sparse(vals, idxs, N):
    w = [0] * N
    for v, i in zip(vals, idxs):
        w[i] = v
    return w


def _self_check(verbose: bool = True) -> bool:
    ok = True

    # ---- (1) byte-exact vs numpy for several small sparse cases -------------
    cases = [
        # (vals, idxs, x) — including an all-zero-except-few case
        ([3, 2], [1, 5], [0, 2, 0, 0, 0, 4, 0, 0]),     # N=8, k=2
        ([1], [7], [0, 0, 0, 0, 0, 0, 0, 3]),           # N=8, k=1 (one nonzero)
        ([2, 1, 3], [0, 4, 9], [1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0]),  # N=12, k=3
        ([1, 1], [0, 14], [1] + [0] * 13 + [1]),        # N=15 (envelope edge), k=2
    ]
    for vals, idxs, x in cases:
        code = _compile_ok(coo_dot_c(vals, idxs, x))
        out, steps = _run(code)
        ref = coo_dot_reference(vals, idxs, x)
        # numpy cross-check
        import numpy as np
        wq = np.array([vals[j] * SCALE for j in range(len(vals))], dtype=np.int64)
        xq = np.array([x[idxs[j]] * SCALE for j in range(len(idxs))], dtype=np.int64)
        npv = int((wq * xq // SCALE).sum())
        match = out == ref == [npv]
        ok = ok and match
        if verbose:
            print(f"  coo_dot N={len(x):>2} k={len(vals)}: "
                  f"draft_VM={out} reference={ref} numpy={[npv]} "
                  f"{'OK' if match else 'MISMATCH'}  ({steps} steps)")

    # ---- (2) COO matvec (per-row nonzeros) byte-exact -----------------------
    # A = [[1,0,2,0],[0,3,0,0]] @ x=[2,1,1,0]
    rows, N = 2, 4
    mv_vals, mv_cols, mv_rowstart = [1, 2, 3], [0, 2, 1], [0, 2, 3]
    mv_x = [2, 1, 1, 0]
    code = _compile_ok(coo_matvec_c(rows, N, mv_vals, mv_cols, mv_rowstart, mv_x))
    out, steps = _run(code)
    ref = coo_matvec_reference(rows, N, mv_vals, mv_cols, mv_rowstart, mv_x)
    match = out == ref
    ok = ok and match
    if verbose:
        print(f"  coo_matvec {rows}x{N}: draft_VM={out} reference={ref} "
              f"{'OK' if match else 'MISMATCH'}  ({steps} steps)")

    # ---- (3) the two SLOPES: COO is O(k), dense is O(N) ---------------------
    # Measure each kernel's marginal draft-step cost per MAC by differencing two
    # sizes IN the byte-safe window, then project to any (N, k).  The COO cost is
    # per NONZERO (so it grows with k, NOT N); the dense cost is per ENTRY (grows
    # with N).  Both increments are exactly constant -> clean O(k) vs O(N).

    def _coo_steps(N, idxs, vals):
        x = [0] * N
        for j, i in enumerate(idxs):
            x[i] = vals[j]
        return _run(_compile_ok(coo_dot_c(vals, idxs, x)))[1]

    def _dense_unroll_steps(w, x):
        out, st = _run(_compile_ok(dense_dot_unroll_c(w, x)))
        return out, st

    # COO slope: same N, k=1 vs k=2  ->  steps-per-nonzero
    coo1 = _coo_steps(12, [0], [1])
    coo2 = _coo_steps(12, [0, 3], [1, 1])
    coo_per_nnz = coo2 - coo1
    coo_base = coo1 - 1 * coo_per_nnz            # setup (k=0 intercept)
    # dense slope: N=4 vs N=8 unrolled  ->  steps-per-entry  (byte-safe, exact)
    d4_out, d4 = _dense_unroll_steps([1, 0, 0, 0], [1, 0, 0, 0])
    d8_out, d8 = _dense_unroll_steps([1] + [0] * 7, [1] + [0] * 7)
    dense_per_entry = (d8 - d4) // 4
    dense_base = d4 - 4 * dense_per_entry
    ok = ok and d4_out == [16] and d8_out == [16]  # both unrolled dense byte-exact

    if verbose:
        print()
        print("  measured draft-step SLOPES (byte-exact within the window):")
        print(f"    COO   (sparse): {coo_base:>4} setup + {coo_per_nnz} per NONZERO "
              f"-> O(k)")
        print(f"    dense (unroll): {dense_base:>4} setup + {dense_per_entry} per ENTRY "
              f"-> O(N)")
        print()
        print("  projected dot-product draft steps  (steps = base + slope * count):")
        print(f"    {'N':>7} {'k':>5} | {'COO steps':>10} {'dense steps':>12} "
              f"| {'reduction':>10}")
        for N, k in [(8, 2), (100, 2), (10_000, 4), (872, 9)]:
            coo_p = coo_base + coo_per_nnz * k
            dense_p = dense_base + dense_per_entry * N
            print(f"    {N:>7} {k:>5} | {coo_p:>10} {dense_p:>12} "
                  f"| {dense_p / coo_p:>9.1f}x")

    # ---- (4) marginal cost per nonzero — the O(k) proof (constant slope) -----
    if verbose:
        print()
        print("  O(k) proof: fixed N=12, sweep k -> each nonzero adds a CONSTANT step count:")
    prev = None
    per_nnz = []
    for k in [1, 2, 3, 4]:
        N = 12
        idxs = list(range(k))
        vals = [1] * k
        x = [0] * N
        for i in idxs:
            x[i] = 1
        cs = _run(_compile_ok(coo_dot_c(vals, idxs, x)))[1]
        if prev is not None:
            per_nnz.append(cs - prev)
        if verbose:
            d = "" if prev is None else f"  (+{cs - prev} vs k={k - 1})"
            print(f"    k={k}: {cs} steps{d}")
        prev = cs
    ok = ok and len(set(per_nnz)) == 1  # constant increment == linear in k
    if verbose and per_nnz:
        print(f"    -> constant +{per_nnz[0]} steps/nonzero (independent of N)")

    # ---- (5) projection to a real ~99.99%-sparse forward --------------------
    if verbose:
        DENSE_FORWARD_STEPS = 2_400_000  # the ~2.4M-step dense self-forward wall
        sparsity = 0.9999
        sparse_frac = 1.0 - sparsity
        proj = DENSE_FORWARD_STEPS * sparse_frac
        print()
        print(f"  projection: a ~{sparsity * 100:.2f}%-sparse forward touches "
              f"{sparse_frac * 100:.2f}% of the MACs ->")
        print(f"    dense self-forward ~= {DENSE_FORWARD_STEPS:,} draft steps")
        print(f"    COO  self-forward ~= {proj:,.0f} draft steps "
              f"(~{1 / sparse_frac:,.0f}x fewer MACs)")

    return ok


if __name__ == "__main__":
    import sys

    print("COO sparse self-emulation kernel — CPU self-check "
          "(compile via real toolchain + draft VM, no model build):")
    ok = _self_check(verbose=True)
    print()
    print(f"ALL BYTE-EXACT (draft VM == reference == numpy): {ok}")
    sys.exit(0 if ok else 1)
