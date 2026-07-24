#!/usr/bin/env python3
"""ablate_matmul_buckets.py — ABLATE where the ~101 draft-VM steps/MAC of the
self-emulation matmul go, and PROTOTYPE / analytically-count a native-fp32-scalar
MAC.  CPU-only: the pure-Python c4 compiler (``src.compiler``) + the draft VM
(``c4_min.nibble_pure_forward_complete.ref_interpret``).  NO GPU, NO neural build.

Everything measured here is BYTE-EXACT through ``ref_interpret`` (asserted vs a
numpy fixed-point reference) unless explicitly labelled "analytic".

Part A — ablate the current ~101 marginal steps/MAC into buckets
----------------------------------------------------------------
Each cost bucket is isolated by DIFFERENCING two byte-exact variants that differ
in exactly one dimension (so every size-independent setup term cancels and the
slope is the honest marginal cost):

  * fpmul as CALL vs INLINE (the JSR/ENT/LEV/ADJ call-frame bucket).
  * DIV rescale vs SHR rescale (``/16`` as a DIV vs ``>>4``; scale = 2**4).
  * PAGED / byte-masked array walk vs IN-WINDOW named scalars (the ADDRESSING
    bucket — LEA/LI/PSH pointer arithmetic the native path does NOT remove
    unless the operands also shrink the arrays).
  * the irreducible arithmetic (the ONE real MUL + ONE ADD + rescale + operand
    fetch), read off the in-window unrolled form.

Part B — native-fp32-scalar MAC
-------------------------------
``ref_interpret`` is the byte-faithful INTEGER VM: ``IMM``/``LEA`` mask to a byte,
``SI``/``SC`` store one byte, AX is at most a 32-bit *integer* under
``mask=0xFFFFFFFF``.  There is NO fp32 register and NO fp MUL/ADD opcode, so it
CANNOT run an fp32-scalar MAC — a true native-fp32 VM is a *different interpreter*.
We therefore ANALYTICALLY count the instruction stream of a native-fp32 MAC kernel
(load a, load b, fmul, fadd, loop-control) and report the estimated steps/MAC with
the reasoning + which buckets it removes.

Part C — recompute the payoff
-----------------------------
Fold the best measured/estimated steps/MAC through the sparse self-forward budget
to recompute the step count + wall-clock at 7.2/1/0.1 ms/step, and state the
honest native-fp32 multiplier over the current 101/MAC.

Run:  python -m c4_min.selfhost.ablate_matmul_buckets
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from c4_min import isa

SCALE = 16          # 2**4 fixed-point scale (shared with the matmul kernels)
_MAX = 500_000


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
    """A ref_interpret CLONE that also returns the per-OPCODE execution histogram
    (byte-identical semantics to ``nibble_pure_forward_complete.ref_interpret``)."""
    SP_INIT = 0x10000
    mem: Dict[int, int] = {}
    sp = bp = SP_INIT
    ax = pc = steps = 0
    mask = 0xFFFFFFFF
    hist: Counter = Counter()
    out: List[int] = []
    while 0 <= pc < len(code) and steps < _MAX:
        steps += 1
        ins = code[pc]
        op, imm = ins.op, ins.imm
        i = pc
        pc += 1
        hist[isa.NAMES.get(op, op)] += 1
        if op == isa.IMM:
            ax = imm & 0xFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFF
        elif op == isa.PSH:
            sp -= 4
            mem[sp] = ax & mask
        elif op in (isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD):
            v = mem.get(sp, 0) & mask
            sp += 4
            if op == isa.ADD:
                ax = (v + ax) & mask
            elif op == isa.SUB:
                ax = (v - ax) & mask
            elif op == isa.MUL:
                ax = (v * ax) & mask
            elif op == isa.DIV:
                ax = ((v // ax) if ax else 0) & mask
            else:
                ax = ((v % ax) if ax else 0) & mask
        elif op in (isa.OR, isa.XOR, isa.AND, isa.SHL, isa.SHR):
            if op in (isa.SHL, isa.SHR):
                v = mem.get(sp, 0) & mask
                sp += 4
                ax = (v << ax) & mask if op == isa.SHL else (v >> ax) & mask
            else:
                v = mem.get(sp, 0) & 0xFF
                sp += 4
                ax = ((v | ax) if op == isa.OR else
                      (v ^ ax) if op == isa.XOR else (v & ax)) & 0xFF
        elif op in (isa.EQ, isa.NE, isa.LT, isa.GT, isa.LE, isa.GE):
            v = mem.get(sp, 0) & mask
            av = ax & mask
            sp += 4
            sv = v - (1 << 32) if v & (1 << 31) else v
            sax = av - (1 << 32) if av & (1 << 31) else av
            r = {isa.EQ: v == av, isa.NE: v != av, isa.LT: sv < sax,
                 isa.GT: sv > sax, isa.LE: sv <= sax, isa.GE: sv >= sax}[op]
            ax = 1 if r else 0
        elif op in (isa.LI, isa.LC):
            ax = mem.get(ax, 0) & 0xFF
        elif op in (isa.SI, isa.SC):
            addr = mem.get(sp, 0)
            sp += 4
            mem[addr] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.JSR:
            sp -= 4
            mem[sp] = (i + 1) & 0xFF
            pc = imm
        elif op == isa.ENT:
            mem[sp - 4] = bp & 0xFFFFFFFF
            sp -= 4
            bp = sp
            sp -= 4 * imm
        elif op == isa.ADJ:
            sp += 4 * imm
        elif op == isa.LEV:
            sp = bp
            bp = mem.get(sp, 0)
            pc = mem.get(sp + 4, 0)
            sp += 8
        elif op == isa.PRTF:
            out.append(ax & 0xFF)
        elif op == isa.NOP:
            pass
        elif op == isa.HALT:
            break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
    return hist, steps, out


# --------------------------------------------------------------------------- #
# kernel source generators for the ablation variants                          #
# --------------------------------------------------------------------------- #
def _scalar_dot(K: int, mode: str) -> str:
    """IN-WINDOW dot: K named scalar locals (w0..w{K-1}, x0..x{K-1}), NO array
    walk, NO paging.  ``mode`` in {'call','inline_div','inline_shr'}."""
    wn = ", ".join(f"w{i}" for i in range(K))
    xn = ", ".join(f"x{i}" for i in range(K))
    wi = " ".join(f"w{i} = {i % 3 + 1}*s;" for i in range(K))
    xi = " ".join(f"x{i} = {i % 2 + 1}*s;" for i in range(K))
    if mode == "call":
        prelude = "int fpmul(int a, int b, int s) { return a * b / s; }\n"
        terms = " + ".join(f"fpmul(w{i}, x{i}, s)" for i in range(K))
    elif mode == "inline_div":
        prelude = ""
        terms = " + ".join(f"(w{i}*x{i}/s)" for i in range(K))
    elif mode == "inline_shr":
        prelude = ""
        terms = " + ".join(f"(w{i}*x{i}>>4)" for i in range(K))
    else:
        raise ValueError(mode)
    return (f"{prelude}int main() {{\n"
            f"  int s; int {wn}; int {xn}; int c;\n"
            f"  s = {SCALE}; {wi} {xi}\n"
            f"  c = {terms};\n"
            f"  printf(c); return 0; }}\n")


def _scalar_ref(K: int) -> List[int]:
    w = [i % 3 + 1 for i in range(K)]
    x = [i % 2 + 1 for i in range(K)]
    acc = sum((w[i] * SCALE) * (x[i] * SCALE) // SCALE for i in range(K))
    return [acc & 0xFF]


def _paged_dot(N: int, mode: str) -> str:
    """PAGED / byte-masked array-walk dot over N consecutive REVERSE-declared
    locals (the COO / _matmul_coo_src.dense_dot addressing).  ``mode`` in
    {'call','inline_div','inline_shr'}."""
    def decl_rev(name):
        return ", ".join(f"{name}{i}" for i in range(N - 1, -1, -1))
    wi = " ".join(f"w{i} = {i % 3 + 1}*s;" for i in range(N))
    xi = " ".join(f"x{i} = {i % 2 + 1}*s;" for i in range(N))
    if mode == "call":
        prelude = "int fpmul(int a, int b, int s) { return a * b / s; }\n"
        mac = "acc = acc + fpmul(*wp, *xp, s);"
    elif mode == "inline_div":
        prelude = ""
        mac = "acc = acc + (*wp * *xp / s);"
    elif mode == "inline_shr":
        prelude = ""
        mac = "acc = acc + (*wp * *xp >> 4);"
    else:
        raise ValueError(mode)
    return (f"{prelude}int main() {{\n"
            f"  int s; int {decl_rev('w')}; int {decl_rev('x')};\n"
            f"  int acc, k; char *wb; char *xb; int *wp; int *xp;\n"
            f"  s = {SCALE}; {wi} {xi}\n"
            f"  wb = &w0; xb = &x0; acc = 0; k = 0;\n"
            f"  while (k < {N}) {{\n"
            f"    wp = wb + k * 4; xp = xb + k * 4;\n"
            f"    {mac}\n"
            f"    k = k + 1; }}\n"
            f"  printf(acc); return 0; }}\n")


def _paged_ref(N: int) -> List[int]:
    w = [i % 3 + 1 for i in range(N)]
    x = [i % 2 + 1 for i in range(N)]
    acc = sum((w[i] * SCALE) * (x[i] * SCALE) // SCALE for i in range(N))
    return [acc & 0xFF]


def _general_dot(K: int, call_fpmul: bool):
    """The REAL current kernel (``_matmul_general_src.matmul_general_c``) as a
    length-K dot (M=N=1) — the 2-D-indexed ``A[p*K+r] / B[r*N+q]`` triple loop
    whose K-sweep slope IS the documented 101 (call) / 88 (inline) steps/MAC.
    Returns (src, reference-out)."""
    from c4_min.selfhost._matmul_general_src import (
        matmul_general_c, matmul_general_reference)
    A = [i % 3 + 1 for i in range(K)]
    B = [i % 2 + 1 for i in range(K)]
    return (matmul_general_c(A, B, 1, K, 1, call_fpmul=call_fpmul),
            matmul_general_reference(A, B, 1, K, 1))


# --------------------------------------------------------------------------- #
# marginal-slope extractor (difference two sizes; assert byte-exact + constant) #
# --------------------------------------------------------------------------- #
def _slope(gen, ref, sizes) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Compile+run ``gen(n)`` for each n in ``sizes``; assert byte-exact vs
    ``ref(n)`` and a CONSTANT step increment; return (per-unit slope, rows)."""
    rows = []
    prev = None
    incs = set()
    for n in sizes:
        code = _compile(gen(n))
        out, steps = _run(code)
        assert out == ref(n), f"n={n}: {out} != {ref(n)} (NOT byte-exact)"
        d = None if prev is None else steps - prev
        if d is not None:
            incs.add(d)
        rows.append((n, steps, d))
        prev = steps
    assert len(incs) == 1, f"non-constant increment {incs} for {rows}"
    return incs.pop(), rows


# --------------------------------------------------------------------------- #
# PART A                                                                       #
# --------------------------------------------------------------------------- #
def part_a(verbose=True):
    if verbose:
        print("=" * 76)
        print("PART A — ablate the ~101 marginal draft-VM steps/MAC into buckets")
        print("         (all byte-exact through ref_interpret vs numpy fixed-point)")
        print("=" * 76)

    # --- the REAL current kernel (matmul_general_c, 2-D indexed): the 101/88 -----
    gen_call, _ = _slope(lambda n: _general_dot(n, True)[0],
                         lambda n: _general_dot(n, True)[1], range(2, 8))
    gen_inline, _ = _slope(lambda n: _general_dot(n, False)[0],
                           lambda n: _general_dot(n, False)[1], range(2, 8))

    # --- paged 1-D array-walk (single-index k*4): CALL / DIV / SHR --------------
    paged_call, _ = _slope(lambda n: _paged_dot(n, "call"), _paged_ref, range(2, 8))
    paged_div, _ = _slope(lambda n: _paged_dot(n, "inline_div"), _paged_ref, range(2, 8))
    paged_shr, _ = _slope(lambda n: _paged_dot(n, "inline_shr"), _paged_ref, range(2, 8))

    # --- in-window named scalars (NO array walk, NO paging): CALL / DIV / SHR ---
    scal_call, _ = _slope(lambda n: _scalar_dot(n, "call"), _scalar_ref, range(2, 7))
    scal_div, _ = _slope(lambda n: _scalar_dot(n, "inline_div"), _scalar_ref, range(2, 7))
    scal_shr, _ = _slope(lambda n: _scalar_dot(n, "inline_shr"), _scalar_ref, range(2, 7))

    # --- per-MAC opcode histogram for the paged-CALL form (steady K=3 - K=2) ----
    h3, _, _ = _run_hist(_compile(_paged_dot(3, "call")))
    h2, _, _ = _run_hist(_compile(_paged_dot(2, "call")))
    per_mac_ops = {k: h3[k] - h2[k] for k in (set(h2) | set(h3)) if h3[k] - h2[k]}

    if verbose:
        print("\n  ABLATION TABLE — marginal draft steps/MAC (K-sweep slope, byte-exact):")
        print(f"    {'variant':<40}{'CALL':>7}{'INLINE/DIV':>12}{'INLINE/SHR':>12}")
        print(f"    {'general_c (2-D index, THE CURRENT KERNEL)':<40}"
              f"{gen_call:>7}{gen_inline:>12}{'-':>12}")
        print(f"    {'paged 1-D walk (single k*4 index)':<40}"
              f"{paged_call:>7}{paged_div:>12}{paged_shr:>12}")
        print(f"    {'in-window scalars (NO array walk)':<40}"
              f"{scal_call:>7}{scal_div:>12}{scal_shr:>12}")
        print(f"\n    (1) fpmul CALL->INLINE saves : general {gen_call - gen_inline:+d}/MAC, "
              f"paged {paged_call - paged_div:+d}/MAC, in-window {scal_call - scal_div:+d}/MAC "
              f"(the JSR/ENT/LEV/ADJ frame)")
        print(f"    (2) DIV->SHR rescale saves   : paged {paged_div - paged_shr:+d}/MAC, "
              f"in-window {scal_div - scal_shr:+d}/MAC  (drops the '/s' local-load)")
        print(f"    (3) ADDRESSING:")
        print(f"        2-D-index premium (general - 1-D paged, CALL) = "
              f"{gen_call - paged_call:>2}/MAC  (extra p*K+r / r*N+q index MUL+ADD)")
        print(f"        1-D byte-safe walk (paged - in-window, SHR)   = "
              f"{paged_shr - scal_shr:>2}/MAC  (LEA/LI/PSH pointer walk)")
        print(f"        -> total addressing in the 101 kernel         = "
              f"{gen_call - scal_call:>2}/MAC  (a DRAFT-VM byte-mask artifact)")

        print("\n  (4) per-MAC OPCODE histogram (general_c CALL, steady K=3 - K=2):")
        gh3, _, _ = _run_hist(_compile(_general_dot(3, True)[0]))
        gh2, _, _ = _run_hist(_compile(_general_dot(2, True)[0]))
        gmix = {k: gh3[k] - gh2[k] for k in (set(gh2) | set(gh3)) if gh3[k] - gh2[k]}
        for k in sorted(gmix, key=lambda k: -gmix[k]):
            print(f"        {k:>5}: +{gmix[k]}")

    # --- bucket decomposition of the 101 general-CALL steps/MAC -----------------
    call_frame = gen_call - gen_inline               # JSR/ENT/LEV/ADJ + call args
    div_bucket = paged_div - paged_shr               # the '/s' local-load (rescale)
    index_2d = gen_inline - paged_div                # extra 2-D index MUL+ADD
    walk_1d = paged_shr - scal_shr                   # byte-safe pointer walk
    addressing = index_2d + walk_1d                  # all addressing
    arith_and_loop = scal_shr                        # in-window: real MUL+ADD+SHR+LI

    if verbose:
        print("\n  --- BUCKET BREAKDOWN of the 101 (general-CALL) marginal steps/MAC ---")
        print(f"    fpmul CALL frame (JSR/ENT/LEV/ADJ+args)     : {call_frame:>4}")
        print(f"    DIV rescale (vs SHR, the '/s' load)         : {div_bucket:>4}")
        print(f"    ADDRESSING: 2-D index (p*K+r / r*N+q)       : {index_2d:>4}")
        print(f"    ADDRESSING: 1-D byte-safe pointer walk      : {walk_1d:>4}")
        print(f"    irreducible arithmetic + operand fetch      : {arith_and_loop:>4}")
        print(f"      (in-window SHR: real MUL + ADD + rescale + LIs + loop)")
        print(f"    {'':>48}{'-' * 5}")
        tot = call_frame + div_bucket + index_2d + walk_1d + arith_and_loop
        print(f"    {'SUM':>46}: {tot:>4}  (== general-CALL {gen_call})")

    return dict(gen_call=gen_call, gen_inline=gen_inline,
                paged_call=paged_call, paged_div=paged_div, paged_shr=paged_shr,
                scal_call=scal_call, scal_div=scal_div, scal_shr=scal_shr,
                call_frame=call_frame, div_bucket=div_bucket, index_2d=index_2d,
                walk_1d=walk_1d, addressing=addressing,
                arith_and_loop=arith_and_loop, per_mac_ops=per_mac_ops)


# --------------------------------------------------------------------------- #
# PART B — native-fp32-scalar MAC (analytic; ref_interpret cannot run fp32)     #
# --------------------------------------------------------------------------- #
def part_b(a_rates, verbose=True):
    if verbose:
        print("\n" + "=" * 76)
        print("PART B — native-fp32-scalar MAC")
        print("=" * 76)

    can_run_fp32 = False
    reason = (
        "ref_interpret is the byte-faithful INTEGER VM: IMM & LEA mask to 0xFF, "
        "SI/SC store one byte, AX is a 32-bit INTEGER under mask=0xFFFFFFFF, and "
        "there is NO fp32 register or fp MUL/ADD opcode.  An fp32 scalar (e.g. "
        "1.5f = 0x3FC00000) cannot be an IMM byte, cannot be stored intact, and "
        "a*b would be integer MUL not fmul.  => a native-fp32 MAC needs a "
        "DIFFERENT interpreter; we count it analytically.")

    if verbose:
        print("\n  HONESTY CHECK 1 — can ref_interpret run an fp32-scalar MAC?")
        print(f"    NO.  {reason}")

    # ANALYTIC native-fp32 MAC instruction count.  A native-fp32 VM (fp32 register,
    # native FMUL/FADD, 32-bit word addressing — NO byte mask) does per inner MAC:
    #   TIGHT  : fully register-allocated (a,b in regs, acc in reg):
    #            2 loads + FMUL + FADD + amortised loop-control(~3) ~= 7 instrs/MAC.
    #   STACK  : same stack-machine ISA as c4 but native 32-bit word LEA + fp ops:
    #            per MAC = 2*(LEA+LI) loads(4) + PSH + FMUL + PSH + FADD + loop(4)
    #            ~= 4 + 1 + 1 + 1 + 1 + 4 = 12 instrs/MAC.
    fp32_tight = 7
    fp32_stack = 12

    if verbose:
        print("\n  ANALYTIC native-fp32-scalar MAC steps/MAC:")
        print(f"    TIGHT (register-allocated, native FMUL/FADD)      ~= {fp32_tight}/MAC")
        print(f"    STACK (c4-style stack ISA + native 32-bit LEA/fp) ~= {fp32_stack}/MAC")
        print("\n    buckets REMOVED vs the draft-VM general-CALL 101/MAC:")
        print(f"      fpmul CALL frame  ({a_rates['call_frame']:>3}/MAC)  -> native FMUL opcode          REMOVED")
        print(f"      DIV rescale       ({a_rates['div_bucket']:>3}/MAC)  -> fp32 has no fixed-point /s    REMOVED")
        print(f"      nibble decomp     (n/a: fp32 is 1 value/register)                REMOVED")
        print(f"      2-D + byte walk   ({a_rates['addressing']:>3}/MAC)  -> 32-bit LEA, no paging       MOSTLY REMOVED")
        print(f"      irreducible       ({a_rates['arith_and_loop']:>3}/MAC)  -> 2 loads + MUL + ADD + loop    REMAINS (shrinks)")

    return dict(can_run_fp32=can_run_fp32, reason=reason,
                fp32_tight=fp32_tight, fp32_stack=fp32_stack)


# --------------------------------------------------------------------------- #
# PART C — recompute the payoff at the best rate                              #
# --------------------------------------------------------------------------- #
def part_c(a_rates, b_rates, verbose=True):
    if verbose:
        print("\n" + "=" * 76)
        print("PART C — recompute the sparse self-forward payoff")
        print("=" * 76)

    # Current sparse (COO) self-forward budgets stated in the task, at 101/MAC:
    CUR_WITHOUT_DIVMOD = 9.76e8
    CUR_WITH_DIVMOD = 1.58e9
    CUR_RATE = 101.0            # draft-VM paged-CALL steps/MAC these assume

    ms = {"7.2 ms/step (dense)": 7.2, "1 ms/step (sparse-cond)": 1.0,
          "0.1 ms/step (batched)": 0.1}

    rates = [
        ("draft general-CALL (CURRENT, 2-D index)", a_rates["gen_call"]),
        ("draft general-INLINE (fpmul inlined)", a_rates["gen_inline"]),
        ("draft paged 1-D-CALL", a_rates["paged_call"]),
        ("draft paged 1-D INLINE+SHR", a_rates["paged_shr"]),
        ("draft IN-WINDOW-CALL (no paging)", a_rates["scal_call"]),
        ("draft IN-WINDOW INLINE+SHR (no paging)", a_rates["scal_shr"]),
        ("native-fp32 STACK (analytic)", b_rates["fp32_stack"]),
        ("native-fp32 TIGHT (analytic)", b_rates["fp32_tight"]),
    ]

    if verbose:
        print(f"\n  Anchor (task): sparse self-forward @ {CUR_RATE:.0f}/MAC =")
        print(f"    {CUR_WITHOUT_DIVMOD:.3e} steps (without-divmod) / "
              f"{CUR_WITH_DIVMOD:.3e} steps (with-divmod)")
        print("\n  Rescaled step budgets (linear in steps/MAC) + speedup vs 101:")
        print(f"    {'rate variant':<40}{'steps/MAC':>9}{'x vs 101':>9}"
              f"{'  without-divmod steps':>22}")
        for name, r in rates:
            scale = r / CUR_RATE
            wo = CUR_WITHOUT_DIVMOD * scale
            print(f"    {name:<40}{r:>9.0f}{CUR_RATE / r:>8.1f}x{wo:>22.3e}")

        scale = b_rates["fp32_stack"] / CUR_RATE
        print("\n  Wall-clock (without-divmod) at native-fp32 STACK "
              f"({b_rates['fp32_stack']}/MAC):")
        for label, m in ms.items():
            cur_h = CUR_WITHOUT_DIVMOD * m / 1000 / 3600
            nat_h = CUR_WITHOUT_DIVMOD * scale * m / 1000 / 3600
            print(f"    {label:<24}: current {cur_h:>8.1f} h  ->  native-fp32 "
                  f"{nat_h:>8.1f} h")

        print("\n  Wall-clock (WITH-divmod) at native-fp32 STACK:")
        for label, m in ms.items():
            cur_h = CUR_WITH_DIVMOD * m / 1000 / 3600
            nat_h = CUR_WITH_DIVMOD * scale * m / 1000 / 3600
            print(f"    {label:<24}: current {cur_h:>8.1f} h  ->  native-fp32 "
                  f"{nat_h:>8.1f} h")

        mult_stack = CUR_RATE / b_rates["fp32_stack"]
        mult_tight = CUR_RATE / b_rates["fp32_tight"]
        addr_frac = a_rates["addressing"] / a_rates["gen_call"]
        print("\n  HONEST native-fp32 multiplier:")
        print(f"    STACK estimate: {mult_stack:.1f}x   TIGHT estimate: {mult_tight:.1f}x")
        print(f"    addressing was {a_rates['addressing']}/{a_rates['gen_call']} "
              f"= {100 * addr_frac:.0f}% of the 101 — but that byte-safe paging + 2-D "
              f"index is a DRAFT-VM byte-mask ARTIFACT, not intrinsic to a MAC.  A "
              f"native-fp32 VM (32-bit LEA) removes it AND the fpmul-call AND the DIV, "
              f"so the multiplier is the full ~{mult_stack:.0f}-{mult_tight:.0f}x — NOT "
              f"capped near ~1.5-2x by addressing.")

    return dict(rates=rates)


# --------------------------------------------------------------------------- #
def main():
    a = part_a(verbose=True)
    b = part_b(a, verbose=True)
    part_c(a, b, verbose=True)
    print("\n" + "=" * 76)
    print("HEADLINE")
    print("=" * 76)
    print(f"  draft-VM general-CALL {a['gen_call']}/MAC = {a['call_frame']} fpmul-call-frame + "
          f"{a['div_bucket']} DIV + {a['addressing']} addressing (2-D index+byte walk) + "
          f"{a['arith_and_loop']} arithmetic+fetch+loop")
    print(f"  native-fp32 (analytic) ~= {b['fp32_tight']}-{b['fp32_stack']}/MAC "
          f"-> ~{a['gen_call'] // b['fp32_stack']}-{a['gen_call'] // b['fp32_tight']}x "
          f"fewer steps; removes fpmul-call + DIV + nibbles + the byte-safe paging")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
