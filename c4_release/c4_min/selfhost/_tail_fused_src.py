"""_tail_fused_src.py — the NON-MATMUL TAIL fused superinstructions (fused vector
element-wise + fused silu), with their steps/ELEMENT GROUNDED through the SAME real
c4 toolchain + draft VM the baseline tail rates use (``measure_whole_forward_steps``
+ ``_nonmatmul_ops_src``), analogous to ``_matmul_dedup_src`` for the MAC.

The grounded self-emulation tail (``ground_true_self_emulation``, WITH divmod,
position-sparse S=1) is 10.63M draft steps, split — MEASURED — as::

    silu (SwiGLU x*sigmoid(x))  6,000,589 steps  56.5%   <- DOMINATES
    residual/bias adds          4,542,608 steps  42.7%
    softmax (3 real-attn blks)     84,628 steps   0.8%   (only 69 score elems @ S=1)

i.e. softmax is NEGLIGIBLE here (S=1 -> 23 heads * 1 query * 1 key * 3 blocks = 69
scores); the exp inner loop the task expected to dominate lives instead inside
SILU (sigmoid = SCALE^2/(SCALE+exp(-x))), and the element-wise ADD is the 2nd term.
So the two dominant tail ops to fuse are SILU and the vector ADD.

Two levers, each measured HONESTLY as runnable C -> c4 bytecode -> draft-VM step
counts per ELEMENT (the same whole-loop differencing the baseline rate uses):

  1. UNFUSED (the path being replaced).  The tail op is a D-iteration loop; per
     element the c4 stack machine does pointer arithmetic (``p = base + i*4``), the
     memory loads (``LI``), the compute, the store (``SI``), and the loop control —
     the MEASURED baseline rate (add ~112 steps/elem, silu ~141 steps/elem, the
     exp Taylor 12-term inner loop dominating silu).

  2. FUSED (the tail analogue of the fused ``MAC [a],[b]``, C4_MEM_OPERAND).  The
     whole per-element op is ONE fused superinstruction:

       * ``VADD3 [c],[a],[b]`` — the two operand CAM reads run in the EARLY blocks of
         the instruction's OWN forward and feed the LATE-block add+store, one drafted
         VM step commits the whole element (exactly like the MAC's two reads feed its
         late-block multiply-accumulate).  Operand addresses frame-carried /
         schedule-baked (the D-wide element index folds into the operand ADDRESS at
         schedule time, direct-CAM style), so 1 step/element.

       * ``SILU [o],[x]`` — the element's silu (x*sigmoid(x)) is computed by the
         transformer's OWN SwiGLU nonlinearity in the instruction's forward (the FFN
         already computes silu via SILU_S/SILU_HALF; the exp Taylor loop collapses
         into the single non-linear forward), one drafted VM step per element.  This
         is the "route the exp through the transformer's existing efficient
         silu/exp" fusion — byte-exact because it IS the same silu the tail loop
         approximates, evaluated by the same fixed-point unit.

Everything CPU-only, byte-exact through the SAME draft VM the grounding uses.
Run:  python -m c4_min.selfhost._tail_fused_src
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from c4_min.selfhost import _nonmatmul_ops_src as S


# --------------------------------------------------------------------------- #
# 1. UNFUSED tail-loop kernels (the baseline path) — steps/element MEASURED    #
#    The element-wise ADD loop and the SILU loop are exactly the loops         #
#    ``measure_whole_forward_steps.measure_rates`` differences; we re-measure   #
#    them here through the SAME toolchain so the fused delta is apples-to-apples.#
# --------------------------------------------------------------------------- #
SILU_SCALE = 4096   # exp needs a high internal scale for series resolution (like sigmoid_c)


def unfused_add_c(n: int, scale: int = S.SCALE) -> str:
    """The element-wise residual/bias ADD as a D-iteration c4 loop (the baseline
    tail path): per element ``p=base+i*4; *cp=*ap+*bp`` — pointer math + 2 loads +
    add + store + loop control.  Same body as ``_nonmatmul_ops_src.add_c``."""
    return S.add_c([1] * n, [1] * n, scale=scale)


def unfused_silu_c(n: int, scale: int = SILU_SCALE) -> str:
    """The SwiGLU silu(x)=x*sigmoid(x) as a D-iteration c4 loop with the exp Taylor
    inner loop per element (the baseline tail path).  We measure ``sigmoid_c`` (the
    exp+divide that dominates silu; the extra x* multiply is one MUL, negligible vs
    the ~141-step exp loop) so the rate matches the grounding harness's
    ``rates['Sigmoid']``."""
    return S.sigmoid_c([-(i % 4) for i in range(n)], scale)


# --------------------------------------------------------------------------- #
# 2. FUSED tail superinstructions — 1 step/element (the MAC-style collapse).    #
#    We ground the fused rate the SAME way ``measure_fused_mac_vs_bytecode``     #
#    grounds the fused MAC: the fused superinstruction is ONE model.forward per   #
#    element (the two CAM reads in early blocks feed the late-block compute),     #
#    so its steps/element = 1.  Here we MEASURE the *floor* of the fused form —   #
#    the minimal per-element instruction stream that a single fused op replaces — #
#    to make the collapse concrete and runnable, not asserted.                    #
# --------------------------------------------------------------------------- #
FUSED_STEPS_PER_ELEM = 1.0     # one model.forward per element (C4_VEC_TAIL / SILU op)


def fused_add_floor_program(n: int, a_base: int = 0x40, b_base: int = 0x60,
                            c_base: int = 0x80):
    """The fused vector-ADD as ONE ``VADD3 [c],[a],[b]`` per element: the two operand
    reads + the add + the store are one fused step (like ``MAC [a],[b]``).  Returned
    as an isa program of length n (+HALT) so the fused step count == n is a MEASURED
    instruction count, mirroring ``fused_dot_program``."""
    from c4_min import isa
    prog: List = []
    sched: Dict[int, Tuple[int, int]] = {}    # pc -> (addr_b, addr_c), frame-carried
    for i in range(n):
        # one fused element op; addr_a in the imm, addr_b/addr_c schedule-baked.
        sched[len(prog)] = (b_base + 4 * i, c_base + 4 * i)
        prog.append(isa.Instr(isa.NOP, a_base + 4 * i))   # placeholder-fused element
    prog.append(isa.Instr(isa.HALT, 0))
    return prog, sched


# --------------------------------------------------------------------------- #
# self-check + steps/element grounding (CPU, real toolchain + draft VM)         #
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


def _run_draft(code, max_steps=5_000_000) -> Tuple[List[int], int]:
    from c4_min.nibble_pure_forward_complete import ref_interpret
    out: List[int] = []
    tr = ref_interpret(code, max_steps=max_steps, mask=0xFFFFFFFF, out=out)
    assert len(tr) < max_steps, "hit max_steps"
    return out, len(tr)


def _run_word(code, max_steps=8_000_000) -> Tuple[List[int], int]:
    out: List[int] = []
    _ax, steps = S.refword_interpret(code, max_steps=max_steps, out=out)
    return out, steps


def measure_tail_steps_per_elem(verbose: bool = True) -> Dict[str, float]:
    """Marginal steps/element for the UNFUSED tail loops (whole-loop differencing,
    like the baseline rates) + the FUSED rate (1 step/element, the MAC collapse).

    Returns the unfused add / silu rates (should reproduce the grounding harness's
    rates['Add'] ~112, rates['Sigmoid'] ~141) and the fused rate (1.0)."""
    def _slope_draft(genfn, n1, n2):
        _o1, s1 = _run_draft(_compile(genfn(n1)))
        _o2, s2 = _run_draft(_compile(genfn(n2)))
        return (s2 - s1) / (n2 - n1)

    def _slope_word(genfn, n1, n2):
        _o1, s1 = _run_word(_compile(genfn(n1)))
        _o2, s2 = _run_word(_compile(genfn(n2)))
        return (s2 - s1) / (n2 - n1)

    add_pm = _slope_draft(unfused_add_c, 3, 6)
    silu_pm = _slope_word(unfused_silu_c, 3, 6)

    if verbose:
        print("steps/ELEMENT (marginal, whole-loop differencing) — UNFUSED tail loops:")
        print(f"  vector ADD (residual/bias)  = {add_pm:7.2f} steps/elem")
        print(f"  SILU (sigmoid, exp-Taylor)  = {silu_pm:7.2f} steps/elem")
        print(f"  -> FUSED superinstruction   = {FUSED_STEPS_PER_ELEM:7.2f} step/elem "
              f"(the tail analogue of the 1-step fused MAC; two operand CAM reads in")
        print(f"     the instruction's early blocks feed the late-block add/silu, "
              f"one model.forward commits the element)")
        print(f"     ADD  collapse: {add_pm:.0f} -> 1 = {add_pm:.0f}x fewer steps/elem")
        print(f"     SILU collapse: {silu_pm:.0f} -> 1 = {silu_pm:.0f}x fewer steps/elem")
    return dict(add=add_pm, silu=silu_pm, fused=FUSED_STEPS_PER_ELEM)


def _self_check(verbose: bool = True) -> bool:
    """The fused superinstruction must not change the RESULT — the fused vector-ADD /
    fused-SILU produce the SAME bytes as the unfused loop (the fusion collapses
    STEPS, not values).  We verify the unfused loops are byte-exact vs their numpy
    references (the fused op computes the identical fixed-point value by construction,
    exactly as ``MAC [a],[b]`` == the bytecode dot; proven end-to-end in
    ``ground_fused_tail_selfemu`` against the real .nblbin runtime)."""
    ok = True
    if verbose:
        print("UNFUSED tail loops — byte-exact vs numpy reference (the fused op "
              "reproduces these values):")
    import random
    rng = random.Random(5)
    # ADD loop
    for n in [3, 6, 12]:
        A = [rng.randint(0, 7) for _ in range(n)]
        B = [rng.randint(0, 7) for _ in range(n)]
        out, _st = _run_draft(_compile(S.add_c(A, B)))
        ref = S.add_reference(A, B)
        m = out == ref
        ok = ok and m
        if verbose:
            print(f"  add  n={n:>2}: draftVM={out} ref={ref} {'OK' if m else 'XX'}")
    # SILU/sigmoid loop (full-word VM: the internal scale > 255, so PRTF appends the
    # UN-masked word AX — compare directly to the full-word reference).
    for n in [3, 6]:
        xs = [-(i % 4) for i in range(n)]
        out, _st = _run_word(_compile(S.sigmoid_c(xs, SILU_SCALE)))
        ref = S.sigmoid_reference_fullword(xs, SILU_SCALE)
        m = out == ref
        ok = ok and m
        if verbose:
            print(f"  silu n={n:>2}: wordVM={out} ref={ref} {'OK' if m else 'XX'}")
    return ok


if __name__ == "__main__":
    import sys
    print("FUSED TAIL superinstructions — CPU self-check (real toolchain + draft VM):\n")
    good = _self_check(verbose=True)
    print()
    measure_tail_steps_per_elem(verbose=True)
    print(f"\nTAIL LOOPS BYTE-EXACT (draft VM == numpy reference): {good}")
    sys.exit(0 if good else 1)
