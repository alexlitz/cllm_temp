"""Correctness gate for the rescued A1 native-C port (`_fast_native_c4.c`).

`_fast_native_c4.c` is an exact C port of `id_port/c90_e2e/native_c4.py`
(8-byte stack cell, byte-offset LEA/ENT/ADJ, 32-bit truncating ALU, flat
memory). A1 shipped only a benchmark driver with NO committed correctness
gate — this is that gate: gcc-compile the C source, run a handful of real
compiled C programs through BOTH the native-C VM and the faithful Python
reference `native_c4.run`, and assert the OUTPUT (final AX + step count) is
byte-exact. Mirrors the conversion in the rescued `_bench_fast_native.py`.

Requires gcc. CPU-safe: no model load, tiny programs, RSS negligible.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile

import pytest

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C_SRC = os.path.join(HERE, "_fast_native_c4.c")
DATA_BASE = 65536

# `native_c4` + `src.compiler` + `c4_min` live under the repo root; there is no
# conftest that wires them, so bootstrap sys.path exactly like the A1 driver.
for _p in (HERE, os.path.join(HERE, "id_port", "c90_e2e")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None,
                                reason="gcc required to compile _fast_native_c4.c")


# A few real C programs with a known final AX (return value & 0xFF).
PROGRAMS = [
    ("const_add", "int main(){ return 2+3; }", 5),
    ("while_sum", "int main(){ int i; int s; i=0; s=0;"
                  " while(i<10){ s=s+i; i=i+1; } return s; }", 45),
    ("nested_mul", "int main(){ int i; int j; int s; i=0; s=0;"
                   " while(i<8){ j=0; while(j<8){ s=s+((i*7+j)%13); j=j+1; }"
                   " i=i+1; } return s&0xFF; }", None),
    ("fib_recursion", "int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2); }"
                      " int main(){ return fib(15)&0xFF; }", None),
    ("heap_ptr", "int main(){ int *p; int i; int s; p=malloc(8*4);"
                 " i=0; while(i<8){ *(p+i)=i*i; i=i+1; }"
                 " i=0; s=0; while(i<8){ s=s+*(p+i); i=i+1; }"
                 " return s&0xFF; }", 140),
    ("ptr_deref", "int main(){ int v; int *p; v=99; p=&v; return *p; }", 99),
    ("div_mod", "int main(){ int a; a=100; return (a/7)*10+(a%7); }", 142),
    ("bitops", "int main(){ int x; x=0xF0; x=x|0x0F; x=x^0xAA; x=x&0xFF;"
               " return x; }", 0x55),
]


@pytest.fixture(scope="module")
def native_lib():
    workdir = tempfile.mkdtemp(prefix="test_fast_native_")
    so = os.path.join(workdir, "_fast_native_c4.so")
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-static-libgcc",
                    C_SRC, "-o", so], check=True)
    lib = ctypes.CDLL(so)
    lib.vm_run.restype = ctypes.c_int64
    lib.vm_run.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    yield lib
    shutil.rmtree(workdir, ignore_errors=True)


def _compile_prog(src):
    from c4_min import isa
    from src.compiler import compile_c
    words, data = compile_c(src)
    code = []
    for w in words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append(isa.Instr(op, imm))
    return code, data


def _run_c(lib, code, data, max_steps=500_000_000):
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
    out_steps = ctypes.c_int64(0)
    ax = lib.vm_run(ops, imms, n, ia, iv, m, max_steps, ctypes.byref(out_steps))
    return ax & 0xFFFFFFFF, out_steps.value


@pytest.mark.parametrize("name,src,expect", PROGRAMS)
def test_native_c_port_output_matches_reference(native_lib, name, src, expect):
    import native_c4  # id_port/c90_e2e bootstrapped onto sys.path above
    code, data = _compile_prog(src)
    py_ax, py_steps = native_c4.run(code, data=data, max_steps=500_000_000)
    c_ax, c_steps = _run_c(native_lib, code, data)

    assert (c_ax, c_steps) == (py_ax, py_steps), (
        f"{name}: native-C (ax={c_ax}, steps={c_steps}) != "
        f"faithful native_c4.run (ax={py_ax}, steps={py_steps})"
    )
    if expect is not None:
        assert (c_ax & 0xFF) == expect, (
            f"{name}: got AX&0xFF={c_ax & 0xFF}, expected {expect}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
