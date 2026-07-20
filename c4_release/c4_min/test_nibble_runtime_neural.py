"""The c4_min runtime library (malloc / free / memset / memcmp) THROUGH THE ONE
UNIFIED MODEL — neural verification via ``model.forward``.

The library is pure base-ISA BYTECODE (``nibble_runtime``), so it needs no new
neural op; the unified full-op interpreter already decodes + executes every op it
composes from.  The ONE thing the heap needs that the base corpus does not is a
32-bit LOAD ADDRESS (heap at 0x30008); ``lib_neural.build_lib_model`` supplies the
address-query widening.

These tests run the BAKED bytecode through the KV-cached sparse driver
(``run_pure_forward_cached`` over a ``SparseTransformer``) at ``mask=0xFFFFFFFF``
(word-width memory) and check byte-exact against the word-width reference oracle
``nibble_runtime.ref_interpret_words`` — the library's faithful oracle (SI/LI are
32-bit words, matching the neural memory's 4-byte MEM_VAL cells).

The model build is ~62 GB dense before the sparse conversion; it is memory-heavy.
Run ONLY on a box with headroom (``free -g``), single-process, ``OMP_NUM_THREADS=4``.

Run:
    OMP_NUM_THREADS=4 PYTHONPATH=<repo> python c4_min/test_nibble_runtime_neural.py
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os

from c4_min import isa
from c4_min import nibble_runtime as R


# The KV-cached driver + dense-kernel sparse forward allocate large transient
# CPU tensors per run; glibc's arena keeps those freed blocks resident, so the
# process RSS high-water mark grows monotonically across the shared-model runs
# (each byte-exact + cheap in isolation, but ~5 runs in one pytest process
# accreted to >20 GB).  `malloc_trim(0)` returns the freed arena pages to the OS
# between runs, so the per-run peak (~14 GB) stays flat instead of accumulating.
_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")


def _release_memory():
    """gc + return glibc's freed arenas to the OS (keeps RSS flat across runs)."""
    gc.collect()
    try:
        _LIBC.malloc_trim(0)
    except (AttributeError, OSError):
        pass                                    # non-glibc: gc.collect() alone


# Build the model + sparse wrapper ONCE and share it across the tests — but the
# code overlay has EXACTLY ``code_size`` instruction slots (``L.CODE_OP[k]`` is a
# fixed-length list), so a cached model built for a SHORT program indexes out of
# range / control-flow-desyncs when a LONGER program is run through it later.
# Tests run in file order (zfod=35, malloc_bump=45, memset=30): the ORIGINAL
# `_sparse_model` built the shared model at the FIRST caller's code_size (37 for
# zfod) and never grew it, so `test_malloc_bump_neural`'s 45-instruction program
# then overran `L.CODE_OP` — an IndexError / control-flow desync (the "31 vs 35
# steps + 0xFFFFFFFF garbage" the harness caught).  It was a test-harness sizing
# bug, NOT a model value-path bug: each program is byte-exact through the model
# when the model is built big enough (zfod/malloc_bump/memset all verified equal
# to the word-width reference on one model at code_size=47).
#
# Fix: size the shared build to fit the LARGEST test program up front (one build,
# no rebuild), and still GROW monotonically if a future/larger program is added
# (dropping the old model first so peak RSS stays ~one streaming build, ~4 GB).
# A model built at a LARGER code_size is byte-identical for a SHORTER program, so
# the shared oversize model is safe for every test.
_SPARSE = None
_L = None
_CODE_SIZE = 0

#: Code-segment size the shared model is built at: the max instruction count of
#: any test program (+2 headroom), so ONE streaming build serves every test.
#: The default suite's longest program is malloc_bump (45 -> 47); the opt-in
#: heavy memcmp programs are longer (53 -> 55).  `_sparse_model` still grows past
#: this if a bigger program is ever requested, so 47 is a safe default that the
#: heavy path auto-grows to 55 on demand.
_SHARED_CODE_SIZE = 55 if os.environ.get("C4_LIB_NEURAL_HEAVY") == "1" else 47


def _sparse_model(code_size: int):
    global _SPARSE, _L, _CODE_SIZE
    want = max(code_size, _SHARED_CODE_SIZE)
    if _SPARSE is None or want > _CODE_SIZE:
        # STREAMING sparse build: peak RSS is ~one block, not the ~62 GB dense
        # whole — the memory-safe way to materialise the unified full-op model.
        from c4_min.lib_neural import build_lib_model_streaming
        _SPARSE = _L = None                     # free the old model before rebuild
        _SPARSE, _L, _ = build_lib_model_streaming(
            code_size=want, recurrent_divmod=True, addr32=True)
        _CODE_SIZE = want
    return _SPARSE, _L


def _run_neural(instrs, max_steps=200, evict=False, prune_interval=60):
    """Run baked bytecode through the unified model (KV-cached sparse driver).

    ``evict`` prunes the per-block KV caches on a schedule (``prune_interval``)
    so the cache stays FLAT over deep loops — needed for the longer looping
    subroutines (memcmp), whose stream would otherwise grow the cache without
    bound at this model width (dim≈1725 × 305 blocks).  ``prune_interval=60`` is
    byte-exact (verified full-trace equal to the word-width reference for both
    memcmp branches; the driver default of 120 is byte-exact too but lets the
    cache grow larger).  ``_release_memory`` after the run returns the freed
    transient CPU tensors to the OS so the shared-process peak (~14 GB) stays
    flat across all the tests instead of accreting past the memory budget."""
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    sparse, L = _sparse_model(code_size=len(instrs) + 2)
    try:
        return run_pure_forward_cached(sparse, L, instrs, max_steps=max_steps,
                                       mask=0xFFFFFFFF, evict=evict,
                                       prune_interval=prune_interval)
    finally:
        _release_memory()


def _ref_ax_trace(instrs, max_steps=5000):
    """Per-step AX trace from the word-width reference (matches the neural driver's
    per-step AX emission).  We compare the neural trace against this."""
    from c4_min.nibble_pure_forward import SP_INIT
    mem = {}
    stack = {}
    ax = pc = 0
    sp = bp = SP_INIT
    steps = 0
    trace = []

    def lw(a):
        return sum(mem.get(a + i, 0) << (8 * i) for i in range(4))

    def sw(a, v):
        for i in range(4):
            mem[a + i] = (v >> (8 * i)) & 0xFF

    def push(v):
        nonlocal sp
        sp -= 4; stack[sp] = v & 0xFFFFFFFF

    def pop():
        nonlocal sp
        v = stack.get(sp, 0); sp += 4; return v

    while 0 <= pc < len(instrs) and steps < max_steps:
        steps += 1
        op, imm = instrs[pc].op, instrs[pc].imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFFFFFFFF
        elif op == isa.LEA:
            ax = (bp + 4 * imm) & 0xFFFFFFFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.ADD:
            ax = (pop() + ax) & 0xFFFFFFFF
        elif op == isa.SUB:
            ax = (pop() - ax) & 0xFFFFFFFF
        elif op == isa.MUL:
            ax = (pop() * ax) & 0xFFFFFFFF
        elif op == isa.LT:
            ax = 1 if pop() < ax else 0
        elif op == isa.LI:
            ax = lw(ax)
        elif op == isa.LC:
            ax = mem.get(ax, 0) & 0xFF
        elif op == isa.SI:
            sw(pop(), ax)
        elif op == isa.SC:
            mem[pop()] = ax & 0xFF
        elif op == isa.JMP:
            pc = imm
        elif op == isa.BZ:
            pc = imm if ax == 0 else pc
        elif op == isa.BNZ:
            pc = imm if ax != 0 else pc
        elif op == isa.HALT:
            trace.append(ax & 0xFFFFFFFF); break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
        trace.append(ax & 0xFFFFFFFF)
    return trace


# ---------------------------------------------------------------------------
# 1. ZFOD: malloc -> store -> load -> free -> load-returns-zero, all through the
#    transformer.  This is the headline malloc-on-transformer proof.
# ---------------------------------------------------------------------------
def _zfod_program():
    """malloc(4) -> *(ptr)=0x99 -> AX=*(ptr) -> free(ptr) -> AX=*(ptr) (==0).

    The allocated pointer is spilled into a scratch cell so the store/load/free
    can address it (the bytecode has no register file beyond AX).  We use the
    fixed heap base for the store/load/free (malloc's first allocation is always
    HEAP_BASE), so the program is straight-line and self-contained."""
    a = R.Asm()
    a.splice(R.emit_malloc(4))                          # AX = malloc(4) = HEAP_BASE
    R._store(a, R.HEAP_BASE, lambda a: a.imm(0x99))    # *(ptr) = 0x99
    R._load(a, R.HEAP_BASE)                             # AX = *(ptr)  -> 0x99
    a.splice(R.emit_free(R.HEAP_BASE))                 # free(ptr)
    R._load(a, R.HEAP_BASE)                             # AX = *(ptr)  -> 0 (ZFOD)
    a.exit_()
    return a.instrs()


def test_zfod_malloc_store_load_free_neural():
    prog = _zfod_program()
    tr = _run_neural(prog)
    ref = _ref_ax_trace(prog)
    assert tr == ref, f"ZFOD neural trace != reference\n  neural={tr}\n  ref={ref}"
    # explicit checkpoints in the trace: the store value read back, then 0 after free.
    assert 0x99 in tr, f"stored value not read back: {tr}"
    assert tr[-1] == 0, f"load-after-free must be 0 (ZFOD), got {tr[-1]}: {tr}"


# ---------------------------------------------------------------------------
# 2. malloc returns the aligned heap base + a second alloc is bumped by 4.
# ---------------------------------------------------------------------------
def test_malloc_bump_neural():
    prog = R.chain(R.emit_malloc(4), R.emit_malloc(6))   # AX = 2nd ptr
    tr = _run_neural(prog)
    ref = _ref_ax_trace(prog)
    assert tr == ref, f"malloc-bump neural != reference\n  neural={tr}\n  ref={ref}"
    assert tr[-1] == R.HEAP_BASE + 4, f"2nd ptr {hex(tr[-1])} want {hex(R.HEAP_BASE+4)}"


# ---------------------------------------------------------------------------
# 3. memset: fill n bytes then read one back with LC (through the model).
# ---------------------------------------------------------------------------
def test_memset_neural():
    p, c, n = R.HEAP_BASE, 0xAB, 4
    # emit_memset ends with `IMM p; EXIT` (it halts on its return value), so to
    # observe a filled byte we inline the loop WITHOUT its trailing exit and LC
    # one filled byte back through the model.
    prog = _memset_then_readback(p, c, n, off=2)
    tr = _run_neural(prog)
    ref = _ref_ax_trace(prog)
    assert tr == ref, f"memset neural != reference\n  neural={tr}\n  ref={ref}"
    assert tr[-1] == c, f"filled byte readback {tr[-1]} want {c}"


def _memset_then_readback(p, c, n, off):
    """memset(p,c,n) loop body WITHOUT the trailing EXIT, then LC p[off]."""
    a = R.Asm()
    R._store(a, R._SC_I, lambda a: a.imm(0))
    a.label("top")
    R._load(a, R._SC_I); a.psh(); a.imm(n); a.lt()
    a.bz("done")
    a.imm(p).psh(); R._load(a, R._SC_I); a.add()
    a.psh(); a.imm(c).emit(isa.SC)
    R._add_const(a, R._SC_I, 1)
    a.jmp("top")
    a.label("done")
    a.imm(p + off).emit(isa.LC)     # AX = p[off]
    a.exit_()
    return a.instrs()


# ---------------------------------------------------------------------------
# 4. memcmp: seed two byte buffers via SC, then run the memcmp loop through the
#    model.  Covers BOTH branches — first-differing byte (returns a[i]-b[i]) and
#    all-equal (loops to completion, returns 0).
#
#    memcmp is the DEEPEST looping subroutine (72-88 model steps).  The KV-cached
#    driver over that many steps churns the glibc arena to a large TRANSIENT peak
#    at the full unified model's scale (305 blocks × dim≈1600), well past the
#    ~4 GB streaming-build budget — an inherent property of running a deep loop
#    through the whole model, not a build-time blow-up.  The result is byte-exact
#    (verified full-trace equal to the word-width reference, both branches), so
#    these tests are GATED OPT-IN behind ``C4_LIB_NEURAL_HEAVY=1`` to keep the
#    default suite (malloc/free/memset) within a modest transient footprint.  Run
#    them explicitly with a box that has headroom:
#        C4_LIB_NEURAL_HEAVY=1 OMP_NUM_THREADS=4 pytest -k memcmp <this file>
# ---------------------------------------------------------------------------
import pytest

_HEAVY = pytest.mark.skipif(
    os.environ.get("C4_LIB_NEURAL_HEAVY") != "1",
    reason="memcmp neural runs a deep loop through the full model (large transient "
           "RSS); set C4_LIB_NEURAL_HEAVY=1 to run")


def _memcmp_prog(a_bytes, b_bytes):
    """Store ``a_bytes`` at ``pa`` and ``b_bytes`` at ``pb`` (byte stores), then
    emit_memcmp(pa, pb, n).  Self-contained: the neural memory holds the two
    buffers from the leading SC stores, then the memcmp LC/SUB loop reads them
    back through the model — the "compiled from C, not a tool call" path."""
    pa, pb = R.HEAP_BASE, R.HEAP_BASE + 16
    a = R.Asm()
    for i, bv in enumerate(a_bytes):
        R._store(a, pa + i, lambda a, bv=bv: a.imm(bv), byte=True)
    for i, bv in enumerate(b_bytes):
        R._store(a, pb + i, lambda a, bv=bv: a.imm(bv), byte=True)
    a.splice(R.emit_memcmp(pa, pb, len(a_bytes)))
    return a.instrs()


@_HEAVY
def test_memcmp_mismatch_neural():
    # a = [1, 5], b = [1, 2] -> first differs at index 1: 5 - 2 = 3.
    prog = _memcmp_prog([1, 5], [1, 2])
    tr = _run_neural(prog, max_steps=300, evict=True)
    ref = _ref_ax_trace(prog)
    assert tr == ref, f"memcmp(mismatch) neural != reference\n  neural={tr}\n  ref={ref}"
    assert tr[-1] == 3, f"memcmp first-diff must be 5-2=3, got {tr[-1]}: {tr}"


@_HEAVY
def test_memcmp_equal_neural():
    # a == b -> the loop runs to completion and returns 0.
    prog = _memcmp_prog([7, 7], [7, 7])
    tr = _run_neural(prog, max_steps=300, evict=True)
    ref = _ref_ax_trace(prog)
    assert tr == ref, f"memcmp(equal) neural != reference\n  neural={tr}\n  ref={ref}"
    assert tr[-1] == 0, f"memcmp equal must return 0, got {tr[-1]}: {tr}"


if __name__ == "__main__":
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
