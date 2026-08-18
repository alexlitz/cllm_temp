#!/usr/bin/env python3
"""Benchmark path 3: the local C port of native_c4.py semantics (_fast_native_c4.so).

Verifies byte-exact final AX + step count vs native_c4.run, then times. Single-thread.
This is the "faster native path" ceiling for the c4 reference semantics on this box.
"""
import ctypes
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "id_port", "c90_e2e"))

from c4_min import isa
from src.compiler import compile_c
import native_c4
from cases_ext import CASES

DATA_BASE = 65536

_lib = ctypes.CDLL(os.path.join(HERE, "_fast_native_c4.so"))
_lib.vm_run.restype = ctypes.c_int64
_lib.vm_run.argtypes = [
    ctypes.POINTER(ctypes.c_int32),   # ops
    ctypes.POINTER(ctypes.c_int64),   # imms
    ctypes.c_int64,                   # code_len
    ctypes.POINTER(ctypes.c_int64),   # init_addr
    ctypes.POINTER(ctypes.c_int64),   # init_val
    ctypes.c_int64,                   # init_n
    ctypes.c_int64,                   # max_steps
    ctypes.POINTER(ctypes.c_int64),   # out_steps
]


def compile_prog(src):
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data


def to_c(code, data):
    n = len(code)
    ops = (ctypes.c_int32 * n)(*[c.op for c in code])
    imms = (ctypes.c_int64 * n)(*[c.imm for c in code])
    if data:
        addrs = [DATA_BASE + i for i, b in enumerate(data) if b]
        vals = [b & 0xFF for i, b in enumerate(data) if b]
    else:
        addrs, vals = [], []
    m = len(addrs)
    ia = (ctypes.c_int64 * max(1, m))(*addrs)
    iv = (ctypes.c_int64 * max(1, m))(*vals)
    return ops, imms, n, ia, iv, m


def run_c(code, data, max_steps=500_000_000):
    ops, imms, n, ia, iv, m = to_c(code, data)
    out_steps = ctypes.c_int64(0)
    ax = _lib.vm_run(ops, imms, n, ia, iv, m, max_steps, ctypes.byref(out_steps))
    return ax & 0xFFFFFFFF, out_steps.value


def bench(name, code, data, expect=None, repeats=3, max_steps=500_000_000):
    # verify byte-exact vs python native
    pax, psteps = native_c4.run(code, data=data, max_steps=max_steps)
    cax, csteps = run_c(code, data, max_steps=max_steps)
    veri = "EXACT" if (pax == cax and psteps == csteps) else \
           f"DIVERGE(py ax={pax} steps={psteps} | c ax={cax} steps={csteps})"
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_c(code, data, max_steps=max_steps)
        dt = time.perf_counter() - t0
        if best is None or dt < best:
            best = dt
    ips = csteps / best if best > 0 else float("inf")
    ok = "" if expect is None else (" OK" if (cax & 0xFF) == expect else f" MISMATCH")
    print(f"{name:26s} steps={csteps:>12,}  wall={best*1e3:10.3f} ms  "
          f"{ips/1e6:9.2f} M instr/s  [{veri}]{ok}")
    return ips


def main():
    print("=" * 100)
    print("PATH 3 — C port of native_c4 semantics (ctypes, -O3), single-thread")
    print("=" * 100)
    by_name = {c[0]: c for c in CASES}
    for nm in ["fn_recursion_fib", "while_factorial", "nested_while", "ptr_walk_sum"]:
        if nm in by_name:
            _, cat, src, exp = by_name[nm]
            code, data = compile_prog(src)
            bench(nm, code, data, expect=exp)

    heavy = ("int main(){ int i; int j; int s; i=0; s=0;"
             " while(i<%d){ j=0; while(j<%d){ s=s+((i*7+j)%%13); j=j+1; } i=i+1; }"
             " return s&0xFF; }")
    for (oi, ij) in [(500, 500), (1000, 1000)]:
        src = heavy % (oi, ij)
        code, data = compile_prog(src)
        bench(f"heavy_{oi}x{ij}", code, data)


if __name__ == "__main__":
    main()
