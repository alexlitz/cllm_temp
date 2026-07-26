#!/usr/bin/env python3
"""measure_strength_reduced_matmul.py — hand-optimise the emulated matmul c4 source
to cut the draft-VM ``steps/MAC`` below the ~101 the 2-D-indexed kernel pays, and
MEASURE the reduction byte-exactly.

CPU-only: the pure-Python c4 compiler (``src.compiler``) + the byte-masking draft
VM (``c4_min.nibble_pure_forward_complete.ref_interpret``).  NO GPU, NO neural
build, NO model weights.  This touches ONLY the self-emulation matmul C source.

The kernel variants (all M=N=1 dot so a K-sweep gives a pure per-MAC slope by
differencing; every size-independent setup term cancels):

  (a) ORIGINAL 2-D-indexed  — the current ``_matmul_general_src.matmul_general_c``
      inner body: ``ap = ab + (p*K+r)*4; bp2 = bb + (r*N+q)*4`` recomputed each
      inner iteration + ``fpmul(*ap,*bp2,s)`` CALL.  Reproduces the documented
      ~101 (call) / ~88 (inline) steps/MAC.

  (b) FLAT-LINEAR (index recompute) — flat 1-D arrays, single linear index
      ``a_idx = a_row + r`` / ``b_idx = b_off + r`` with the row/col base HOISTED
      out of the inner loop, but the byte address still recomputed from the index
      each iteration (``ap = ab + a_idx*4``).  fpmul inlined.

  (c/d) STRENGTH-REDUCED (+ INLINE-MUL) — running byte pointers ``ap``/``bp``
      carried across the inner loop and incremented by the stride each MAC
      (``ap = ap + 4`` for the contiguous A row; ``bp = bp + N*4`` for the strided
      B column; both +4 at N=1) instead of recomputing ``base + index*4``.  The
      per-MAC index MUL becomes a single ADD.  Deref via ``*(int *)ap`` (a
      byte-running ``char*`` incremented by the raw stride, then loaded as an int).
      fpmul inlined (d) or CALL (c, to isolate the call-frame delta).

Every variant is asserted BYTE-EXACT vs a shared numpy fixed-point reference AND
vs the original kernel's output.

Run:  python -m c4_min.selfhost.measure_strength_reduced_matmul
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from c4_min import isa

SCALE = 16          # 2**4 fixed-point scale (shared with the other kernels)
_MAX = 2_000_000


# --------------------------------------------------------------------------- #
# compile + run helpers (byte-masking draft VM, byte-exactness asserted)       #
# --------------------------------------------------------------------------- #
def _compile(src: str):
    from src.compiler import compile_c
    from c4_min.run_1096_pure_forward import bytecode_to_isa

    bc, _data = compile_c(src)
    code = bytecode_to_isa(bc)
    over = [i.imm for i in code if i.op == isa.IMM and i.imm > 255]
    assert not over, f"IMM>255 leaked (would diverge from draft VM): {over}"
    return code


def _run(code) -> Tuple[List[int], int]:
    from c4_min.nibble_pure_forward_complete import ref_interpret

    out: List[int] = []
    tr = ref_interpret(code, max_steps=_MAX, mask=0xFFFFFFFF, out=out)
    assert len(tr) < _MAX, "hit max_steps (byte-window wrap -> non-termination)"
    wrapped = [v for v in tr if v >= 2 ** 31]
    assert not wrapped, f"{len(wrapped)} wrapped-negative AX values"
    return out, len(tr)


def _run_hist(code) -> Tuple[Counter, int, List[int]]:
    """A ref_interpret clone that also returns the per-OPCODE execution histogram
    (byte-identical semantics to nibble_pure_forward_complete.ref_interpret)."""
    from c4_min.selfhost.ablate_matmul_buckets import _run_hist as _rh
    return _rh(code)


# --------------------------------------------------------------------------- #
# shared reference + array literals                                           #
# --------------------------------------------------------------------------- #
def _AB(K: int):
    return [i % 3 + 1 for i in range(K)], [i % 2 + 1 for i in range(K)]


def _dot_ref(K: int) -> List[int]:
    """Fixed-point dot reference. A[i]=i%3+1, B[i]=i%2+1 (same convention as
    ablate_matmul_buckets so the two scripts are directly comparable)."""
    A, B = _AB(K)
    acc = sum((A[i] * SCALE) * (B[i] * SCALE) // SCALE for i in range(K))
    return [acc & 0xFF]


def _decl_rev(name: str, n: int) -> str:
    return ", ".join(f"{name}{i}" for i in range(n - 1, -1, -1))


def _init(name: str, vals: List[int]) -> str:
    return " ".join(f"{name}{i} = {vals[i]}*s;" for i in range(len(vals)))


# --------------------------------------------------------------------------- #
# (a) ORIGINAL 2-D-indexed kernel (the ~101 / ~88 baseline)                    #
# --------------------------------------------------------------------------- #
def kernel_original(K: int, call_fpmul: bool) -> str:
    """M=N=1 dot in the CURRENT _matmul_general_src form: ap/bp2 recomputed from
    the 2-D index (p*K+r)*4 / (r*N+q)*4 each inner iteration."""
    from c4_min.selfhost._matmul_general_src import matmul_general_c
    A, B = _AB(K)
    return matmul_general_c(A, B, 1, K, 1, call_fpmul=call_fpmul)


# --------------------------------------------------------------------------- #
# (b) FLAT-LINEAR (hoisted base, index recomputed to a byte offset each MAC)   #
# --------------------------------------------------------------------------- #
def kernel_flat_linear(K: int) -> str:
    """Flat 1-D arrays a0..,b0..; single linear index a_idx = a_row + r,
    b_idx = b_off + r with the row/col base HOISTED out of the inner loop.  The
    byte address is still recomputed from the index each MAC (ap = ab + a_idx*4),
    but the *2-D* (p*K+r)*4 / (r*N+q)*4 double-multiply is gone (M=N=1 so a_row=0,
    b_off=0; the general M>1/N>1 form hoists a_row=p*K, b_off=q once per output).
    fpmul inlined."""
    A, B = _AB(K)
    return f'''
int main() {{
  int s;
  int {_decl_rev("a", K)};
  int {_decl_rev("b", K)};
  int c0;
  int r, acc, a_row, b_off, a_idx, b_idx;
  char *ab; char *bb;
  int *ap; int *bp;
  s = {SCALE};
  {_init("a", A)}
  {_init("b", B)}
  ab = &a0; bb = &b0;
  a_row = 0; b_off = 0;
  acc = 0; r = 0;
  while (r < {K}) {{
    a_idx = a_row + r;
    b_idx = b_off + r;
    ap = ab + a_idx * 4;
    bp = bb + b_idx * 4;
    acc = acc + *ap * *bp / s;
    r = r + 1;
  }}
  c0 = acc;
  printf(c0);
  return 0;
}}
'''


# --------------------------------------------------------------------------- #
# (c/d) STRENGTH-REDUCED (running pointers, += stride, no per-MAC index MUL)   #
# --------------------------------------------------------------------------- #
def kernel_strength_reduced(K: int, call_fpmul: bool = False) -> str:
    """Running byte pointers ap/bp carried across the inner loop and incremented
    by the stride each MAC.  A row is contiguous (stride +4); a B column is strided
    by N (stride +N*4); here M=N=1 so both strides are +4.  The per-MAC index
    multiply (k*4 / (k*N)*4) becomes a single running ADD.  Deref via *(int *)ap.
    fpmul inlined (call_fpmul=False, the (d) variant) or CALL (True, the (c)
    variant, to isolate the call-frame delta)."""
    A, B = _AB(K)
    prelude = ("int fpmul(int a, int b, int s) { return a * b / s; }\n"
               if call_fpmul else "")
    mac = ("acc = acc + fpmul(*(int *)ap, *(int *)bp, s);" if call_fpmul
           else "acc = acc + *(int *)ap * *(int *)bp / s;")
    return f'''
{prelude}int main() {{
  int s;
  int {_decl_rev("a", K)};
  int {_decl_rev("b", K)};
  int c0;
  int r, acc;
  char *ap; char *bp;
  s = {SCALE};
  {_init("a", A)}
  {_init("b", B)}
  ap = &a0; bp = &b0;
  acc = 0; r = 0;
  while (r < {K}) {{
    {mac}
    ap = ap + 4;
    bp = bp + 4;
    r = r + 1;
  }}
  c0 = acc;
  printf(c0);
  return 0;
}}
'''


# --------------------------------------------------------------------------- #
# LOOP-BODY-ONLY per-MAC (fixed array, sweep the loop BOUND) — amortises the    #
# per-element init out so the slope is the genuine inner-loop per-MAC floor.    #
# --------------------------------------------------------------------------- #
def _loopbody_original(Kfix: int, L: int, call: bool) -> str:
    """Original 2-D-index inner loop with a FIXED array (Kfix elements) and a
    variable loop BOUND L — sweeping L isolates the loop body (no extra init)."""
    A, B = _AB(Kfix)
    prelude = ("int fpmul(int a, int b, int s) { return a * b / s; }\n"
               if call else "")
    mac = ("acc = acc + fpmul(*ap, *bp2, s);" if call
           else "acc = acc + *ap * *bp2 / s;")
    return f'''
{prelude}int main() {{
  int s; int {_decl_rev("a", Kfix)}; int {_decl_rev("b", Kfix)}; int c0;
  int p, q, r, acc; char *ab; char *bb; int *ap; int *bp2;
  s = {SCALE}; {_init("a", A)} {_init("b", B)}
  ab = &a0; bb = &b0; acc = 0; r = 0; p = 0; q = 0;
  while (r < {L}) {{
    ap = ab + (p * {Kfix} + r) * 4;
    bp2 = bb + (r * 1 + q) * 4;
    {mac}
    r = r + 1;
  }}
  c0 = acc; printf(c0); return 0;
}}
'''


def _loopbody_sr(Kfix: int, L: int, call: bool) -> str:
    """Strength-reduced inner loop with a FIXED array and a variable loop BOUND L."""
    A, B = _AB(Kfix)
    prelude = ("int fpmul(int a, int b, int s) { return a * b / s; }\n"
               if call else "")
    mac = ("acc = acc + fpmul(*(int *)ap, *(int *)bp, s);" if call
           else "acc = acc + *(int *)ap * *(int *)bp / s;")
    return f'''
{prelude}int main() {{
  int s; int {_decl_rev("a", Kfix)}; int {_decl_rev("b", Kfix)}; int c0;
  int r, acc; char *ap; char *bp;
  s = {SCALE}; {_init("a", A)} {_init("b", B)}
  ap = &a0; bp = &b0; acc = 0; r = 0;
  while (r < {L}) {{
    {mac}
    ap = ap + 4; bp = bp + 4; r = r + 1;
  }}
  c0 = acc; printf(c0); return 0;
}}
'''


def _loopbody_slope(gen_bound, kfix=7, bounds=(3, 4, 5, 6)) -> int:
    """Sweep the loop BOUND over a FIXED-size array; return the constant per-MAC
    increment (the pure loop-body cost, init amortised out).  Byte-exactness is
    checked against a per-accumulation reference at each bound."""
    A, B = _AB(kfix)
    incs = set()
    prev = None
    for L in bounds:
        out, steps = _run(_compile(gen_bound(L)))
        ref = [sum((A[i] * SCALE) * (B[i] * SCALE) // SCALE
                   for i in range(L)) & 0xFF]
        assert out == ref, f"loop-body bound L={L}: {out} != {ref}"
        if prev is not None:
            incs.add(steps - prev)
        prev = steps
    assert len(incs) == 1, f"non-constant loop-body increment {incs}"
    return incs.pop()


# --------------------------------------------------------------------------- #
# marginal-slope extractor (difference two sizes; assert byte-exact + constant) #
# --------------------------------------------------------------------------- #
def _slope(gen, sizes, label) -> Tuple[int, List[Tuple[int, int, int]], List[int]]:
    """Compile+run gen(K) for each K in sizes; assert byte-exact vs _dot_ref(K)
    and a CONSTANT step increment; return (per-MAC slope, rows, last-out)."""
    rows = []
    prev = None
    incs = set()
    last_out = None
    for K in sizes:
        code = _compile(gen(K))
        out, steps = _run(code)
        ref = _dot_ref(K)
        assert out == ref, f"{label} K={K}: {out} != {ref} (NOT byte-exact)"
        d = None if prev is None else steps - prev
        if d is not None:
            incs.add(d)
        rows.append((K, steps, d))
        prev = steps
        last_out = out
    assert len(incs) == 1, f"{label}: non-constant increment {incs} for {rows}"
    return incs.pop(), rows, last_out


# --------------------------------------------------------------------------- #
def main(verbose=True):
    sizes = range(2, 8)

    # (a) original 2-D-indexed
    orig_call, _, out_a = _slope(lambda K: kernel_original(K, True), sizes,
                                 "orig-CALL")
    orig_inline, _, _ = _slope(lambda K: kernel_original(K, False), sizes,
                               "orig-INLINE")
    # (b) flat-linear (hoisted base, index recompute)
    flat, _, out_b = _slope(kernel_flat_linear, sizes, "flat-linear")
    # (c) strength-reduced (running pointers) CALL, (d) INLINE
    sr_call, _, _ = _slope(lambda K: kernel_strength_reduced(K, True),
                           sizes, "sr-CALL")
    sr_inline, _, out_d = _slope(lambda K: kernel_strength_reduced(K, False),
                                 sizes, "sr-INLINE")

    # cross-check: every variant produces the identical output at K=7
    assert out_a == out_b == out_d == _dot_ref(7), (out_a, out_b, out_d)

    if verbose:
        print("=" * 78)
        print("STRENGTH-REDUCED matmul c4 source — draft-VM steps/MAC (byte-exact)")
        print("  M=N=1 K-sweep; slope = marginal draft steps per inner-loop MAC")
        print("=" * 78)
        print(f"\n  {'variant':<48}{'steps/MAC':>10}{'x vs 101':>9}")
        print(f"    {'(a) ORIGINAL 2-D index, fpmul CALL':<46}"
              f"{orig_call:>10}{101 / orig_call:>8.2f}x")
        print(f"    {'(a) ORIGINAL 2-D index, fpmul INLINE':<46}"
              f"{orig_inline:>10}{101 / orig_inline:>8.2f}x")
        print(f"    {'(b) FLAT-LINEAR (hoist base, idx recompute)':<46}"
              f"{flat:>10}{101 / flat:>8.2f}x")
        print(f"    {'(c) STRENGTH-REDUCED (running ptr), fpmul CALL':<46}"
              f"{sr_call:>10}{101 / sr_call:>8.2f}x")
        print(f"    {'(d) STRENGTH-REDUCED + INLINE-MUL':<46}"
              f"{sr_inline:>10}{101 / sr_inline:>8.2f}x")

        print("\n  --- reduction chain (vs the 101 baseline) ---")
        print(f"    101 -> {orig_call}   reproduce ORIGINAL 2-D-index CALL")
        print(f"    {orig_call} -> {flat}   flat-linear + hoist base + INLINE mul "
              f"(-{orig_call - flat})")
        print(f"    {flat} -> {sr_inline}   strength-reduce inner address "
              f"(-{flat - sr_inline}: per-MAC index MUL -> running-ptr ADD)")
        print(f"    -> TOTAL {orig_call} -> {sr_inline}  "
              f"= {orig_call / sr_inline:.2f}x fewer draft steps/MAC "
              f"(inline vs inline: {orig_inline} -> {sr_inline} "
              f"= {orig_inline / sr_inline:.2f}x)")

        # per-MAC opcode census of the FINAL (strength-reduced inline) form
        h5, _, _ = _run_hist(_compile(kernel_strength_reduced(5, False)))
        h4, _, _ = _run_hist(_compile(kernel_strength_reduced(4, False)))
        cen = {k: h5[k] - h4[k] for k in (set(h4) | set(h5)) if h5[k] - h4[k]}
        print("\n  --- per-MAC opcode census AFTER optimisation "
              "(strength-reduced INLINE, K=5 - K=4) ---")
        for k in sorted(cen, key=lambda k: -cen[k]):
            print(f"        {k:>5}: +{cen[k]}")
        print(f"        {'TOTAL':>5}: {sum(cen.values())}")

        # side-by-side: what the linear layout removed
        oh5, _, _ = _run_hist(_compile(kernel_original(5, True)))
        oh4, _, _ = _run_hist(_compile(kernel_original(4, True)))
        ocen = {k: oh5[k] - oh4[k] for k in (set(oh4) | set(oh5)) if oh5[k] - oh4[k]}
        print("\n  --- what the linear layout REMOVED "
              "(orig 2-D CALL -> strength-reduced INLINE, per MAC) ---")
        for k in sorted(set(ocen) | set(cen), key=lambda k: -(ocen.get(k, 0))):
            o, n = ocen.get(k, 0), cen.get(k, 0)
            if o - n:
                print(f"        {k:>5}: {o:>3} -> {n:<3}  ({o - n:+d})")

        print("\n  --- what is now IRREDUCIBLE (genuine per-MAC work) ---")
        print("    2 int loads (LI) + 1 MUL + 1 accumulate ADD + 1 rescale DIV,")
        print("    each ALU operand staged with a PSH (stack machine) + 2 pointer")
        print("    LEA/LI, plus the running-ptr +4 ADDs and loop compare/branch")
        print("    (LT/BZ/JMP + r increment).  The linear layout removed the per-MAC")
        print("    index MUL ((k*N)*4) and the hoistable row/col base; it did NOT")
        print("    remove the loads/mul/accumulate/loop-control — the real floor.")

        # ---- LOOP-BODY-ONLY per-MAC (init amortised out; the true floor) ----
        lb_orig_call = _loopbody_slope(lambda L: _loopbody_original(7, L, True))
        lb_orig_inline = _loopbody_slope(lambda L: _loopbody_original(7, L, False))
        lb_sr = _loopbody_slope(lambda L: _loopbody_sr(7, L, False))
        print("\n  --- LOOP-BODY-ONLY steps/MAC "
              "(fixed array, sweep loop bound — init amortised out) ---")
        print(f"    original 2-D CALL   : {lb_orig_call}/MAC")
        print(f"    original 2-D INLINE : {lb_orig_inline}/MAC")
        print(f"    strength-reduced    : {lb_sr}/MAC")
        print(f"    -> loop-body reduction {lb_orig_call} -> {lb_sr} "
              f"= {lb_orig_call / lb_sr:.2f}x (CALL) / "
              f"{lb_orig_inline} -> {lb_sr} = {lb_orig_inline / lb_sr:.2f}x (INLINE)")
        print("    (The K-sweep 101/66 above includes ~16 steps of per-element init")
        print("     the K-sweep attributes to each extra element; the loop-body-only")
        print("     figure is the truer per-MAC floor and both are honest.)")

        # ---- ADDM (C4_CODEGEN_FUSE) foldability of the linear operand ----
        from c4_min import codegen_fuse as CF
        from src.compiler import compile_c
        w_sr, _ = compile_c(kernel_strength_reduced(6, False))
        w_flat, _ = compile_c(kernel_flat_linear(6))
        _, nf_sr = CF.fuse_bytecode(w_sr)
        _, nf_flat = CF.fuse_bytecode(w_flat)
        # the ONE case the linear layout DOES make foldable: an unrolled GLOBAL-array
        # MAC (each operand a compile-time immediate absolute address).
        unrolled = ("int a0; int a1; int a2; int b0; int b1; int b2; int acc; int s;"
                    "int main(){int i; s=16; acc=0;"
                    "a0=1*s;a1=2*s;a2=3*s;b0=1*s;b1=2*s;b2=1*s;"
                    "acc=acc+a0*b0/s; acc=acc+a1*b1/s; acc=acc+a2*b2/s;"
                    "printf(acc); return 0;}")
        w_unr, _ = compile_c(unrolled)
        _, nf_unr = CF.fuse_bytecode(w_unr)
        print("\n  --- ADDM peephole (C4_CODEGEN_FUSE) foldability of the operand ---")
        print(f"    flat-linear (looped, ab+idx*4)        : {nf_flat} folds")
        print(f"    strength-reduced (looped, *(int*)ap)  : {nf_sr} folds")
        print(f"    unrolled GLOBAL-array MAC (a0*b0/s..)  : {nf_unr} folds")
        print("    => the LOOPED linear operand is NOT foldable — the address is a")
        print("       runtime pointer deref (mem[mem[frame]]), not a compile-time")
        print("       immediate, so the ADDM peephole (which folds IMM addr;LI;PSH;OP)")
        print("       still sees the wrong addressing mode.  Only a fully-UNROLLED")
        print("       GLOBAL-array MAC (operand = immediate absolute address) folds.")

    return dict(orig_call=orig_call, orig_inline=orig_inline, flat=flat,
                sr_inline=sr_inline, sr_call=sr_call)


if __name__ == "__main__":
    import sys
    main(verbose=True)
    sys.exit(0)
