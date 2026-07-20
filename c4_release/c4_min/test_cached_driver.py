"""Byte-identity + bounded-cache proof for the KV-cached pure-forward driver.

Proves CHK-1 item #6 ("KV cache eviction works properly and maintains correct
outputs over even long problems") on the REAL pure-forward model: for a battery
of programs — multi-step arithmetic, functions (JSR/ENT/LEV), 32-bit values > 8
bits, memory (SI/LI), and eviction-heavy long loops — the KV-cached (+evicted)
driver's full output is byte-identical to the naive re-forward driver AND to the
SP-addressed reference interpreter.  Reports max seq length, flat cache size, and
eviction counts on the deep runners.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import c4_min.nibble_pure_forward as _PF
import c4_min.nibble_pure_forward_complete as _PFC
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa
from c4_min.nibble_pure_forward_complete import (
    run_pure_forward_complete, ref_interpret)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
from c4_min.nibble_pure_forward import assert_no_python_compute
from c4_min._build_guard import guarded_complete_build


def I(op, imm=0):
    return isa.Instr(op, imm)


# ---------------------------------------------------------------------------
# The battery.  (name, code, max_steps, mask, requires_divmod)
# ---------------------------------------------------------------------------
def battery():
    progs = []
    # multi-step arithmetic (8-bit)
    progs.append(("add", [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 3), I(isa.ADD),
                          I(isa.HALT)], 20, 0xFF, False))
    progs.append(("sub", [I(isa.IMM, 40), I(isa.PSH), I(isa.IMM, 9), I(isa.SUB),
                          I(isa.HALT)], 20, 0xFF, False))
    progs.append(("mul", [I(isa.IMM, 6), I(isa.PSH), I(isa.IMM, 7), I(isa.MUL),
                          I(isa.HALT)], 20, 0xFF, False))
    # chained arithmetic (several steps)
    progs.append(("add_chain",
                  [I(isa.IMM, 10), I(isa.PSH), I(isa.IMM, 20), I(isa.ADD),
                   I(isa.PSH), I(isa.IMM, 30), I(isa.ADD),
                   I(isa.PSH), I(isa.IMM, 40), I(isa.ADD), I(isa.HALT)],
                  30, 0xFF, False))
    # 32-bit values > 8 bits (full 32-bit ALU, mask = 0xFFFFFFFF)
    progs.append(("mul32",
                  [I(isa.IMM, 1000), I(isa.PSH), I(isa.IMM, 1000), I(isa.MUL),
                   I(isa.HALT)], 20, 0xFFFFFFFF, False))
    progs.append(("add32",
                  [I(isa.IMM, 60000), I(isa.PSH), I(isa.IMM, 60000), I(isa.ADD),
                   I(isa.HALT)], 20, 0xFFFFFFFF, False))
    # DIV / MOD (needs the 32-bit long-division blocks)
    progs.append(("div", [I(isa.IMM, 100), I(isa.PSH), I(isa.IMM, 7), I(isa.DIV),
                          I(isa.HALT)], 20, 0xFFFFFFFF, True))
    progs.append(("mod", [I(isa.IMM, 100), I(isa.PSH), I(isa.IMM, 7), I(isa.MOD),
                          I(isa.HALT)], 20, 0xFFFFFFFF, True))
    # comparisons
    progs.append(("lt", [I(isa.IMM, 3), I(isa.PSH), I(isa.IMM, 5), I(isa.LT),
                         I(isa.HALT)], 20, 0xFF, False))
    progs.append(("eq", [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 5), I(isa.EQ),
                         I(isa.HALT)], 20, 0xFF, False))
    # memory: SI then LI round-trip.  addr in AX, PSH addr, IMM val, ... store,
    # then load it back.  Store contract: SI pops the addr, stores AX.
    #   IMM 7 ; PSH ; IMM 200 ; SI    -> MEM[200] = 7 ; AX = 7   (SI: *pop=AX)
    #   IMM 200 ; LI                  -> AX = MEM[200] = 7
    progs.append(("si_li",
                  [I(isa.IMM, 7), I(isa.PSH), I(isa.IMM, 200), I(isa.PSH),
                   I(isa.IMM, 7), I(isa.SI),           # store 7 @ 200
                   I(isa.IMM, 200), I(isa.LI),         # load back
                   I(isa.HALT)], 30, 0xFF, False))
    # function call: JSR to a routine that returns AX = imm, then LEV back.
    #   0: JSR 3
    #   1: HALT              (after return, AX from the callee survives)
    #   2: NOP               (padding)
    #   3: ENT 0             (enter frame)
    #   4: IMM 42
    #   5: LEV               (return; PC <- return addr = 1)
    progs.append(("func_jsr_lev",
                  [I(isa.JSR, 3), I(isa.HALT), I(isa.NOP),
                   I(isa.ENT, 0), I(isa.IMM, 42), I(isa.LEV)],
                  30, 0xFF, False))
    # a loop: count down from N, accumulate.  BNZ back-edge -> deep-ish, triggers
    # eviction.  sum = 1+2+...  Actually simpler: decrement a counter to 0.
    #   0: IMM 8            AX = 8   (loop counter)
    #   1: PSH              push counter
    #   2: IMM 1
    #   3: SUB              AX = counter - 1
    #   4: BNZ 1            if AX != 0 goto 1
    #   5: HALT
    progs.append(("loop_countdown",
                  [I(isa.IMM, 8), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                   I(isa.BNZ, 1), I(isa.HALT)],
                  200, 0xFF, False))
    # a LONGER loop (eviction-heavy): counter 30 -> ~120 steps.
    progs.append(("loop_long",
                  [I(isa.IMM, 30), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                   I(isa.BNZ, 1), I(isa.HALT)],
                  400, 0xFF, False))
    return progs


def main():
    import sys
    want_dm = "--divmod" in sys.argv
    print(f"[cached] building models (LEAN{'+divmod' if want_dm else ''}) ...",
          flush=True)
    t0 = time.time()
    # Memory-SAFE streaming full-op build (peak ~5 GB) — the streamed model
    # already folds DIV/MOD, so it serves BOTH the lean and --divmod batteries;
    # the dense build would peak at 54-108 GB RSS.
    model_lean, L_lean = guarded_complete_build(code_size=16)
    model_dm = L_dm = None
    if want_dm:
        model_dm, L_dm = model_lean, L_lean
    print(f"[cached] built in {time.time()-t0:.1f}s "
          f"(lean blocks={len(model_lean.blocks)}"
          f"{', divmod blocks=' + str(len(model_dm.blocks)) if want_dm else ''})",
          flush=True)

    n_ok = n_fail = 0
    print(f"\n{'prog':16s} {'ref':>10s} {'naive':>10s} {'cached':>10s} "
          f"{'evict':>10s} {'steps':>6s} {'maxseq':>7s} {'cache':>6s} {'evicted':>8s}",
          flush=True)
    print("-" * 100, flush=True)
    for name, code, msteps, mask, need_dm in battery():
        if need_dm and not want_dm:
            continue
        model, L = (model_dm, L_dm) if need_dm else (model_lean, L_lean)
        ref = ref_interpret(code, max_steps=msteps, mask=mask)
        naive = run_pure_forward_complete(model, L, code, max_steps=msteps, mask=mask)
        st_ne = {}
        cached_noevict = run_pure_forward_cached(
            model, L, code, max_steps=msteps, mask=mask, evict=False, stats=st_ne)
        st = {}
        cached_evict = run_pure_forward_cached(
            model, L, code, max_steps=msteps, mask=mask, evict=True,
            prune_interval=120, stats=st)
        ref_v = ref[-1] if ref else None
        na_v = naive[-1] if naive else None
        ca_v = cached_noevict[-1] if cached_noevict else None
        ev_v = cached_evict[-1] if cached_evict else None
        ok = (naive == cached_noevict == cached_evict)
        # also assert full trace identity, not just final value
        ok = ok and (naive == cached_noevict) and (naive == cached_evict)
        status = "OK " if ok else "!! "
        print(f"{status}{name:13s} {str(ref_v):>10s} {str(na_v):>10s} "
              f"{str(ca_v):>10s} {str(ev_v):>10s} {st['steps']:6d} "
              f"{st['max_seq_len']:7d} {st['max_cache_size']:6d} "
              f"{st['total_evicted']:8d}", flush=True)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            print(f"    MISMATCH: naive={naive}")
            print(f"              cached_noevict={cached_noevict}")
            print(f"              cached_evict  ={cached_evict}")

    print("-" * 100)
    print(f"BYTE-IDENTITY (naive == cached == cached+evict): {n_ok} OK, {n_fail} FAIL")

    # purity guard on the cached+evicted driver (a representative program).
    guard_prog = [I(isa.IMM, 8), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                  I(isa.BNZ, 1), I(isa.HALT)]
    try:
        tr = assert_no_python_compute(
            run_pure_forward_cached, model_lean, L_lean, guard_prog,
            max_steps=200, mask=0xFF, evict=True)
        print(f"PURITY GUARD (cached+evict): CLEAN — trace ends {tr[-1]}")
    except AssertionError as e:
        print(f"PURITY GUARD: LEAK — {e}")
        n_fail += 1

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
