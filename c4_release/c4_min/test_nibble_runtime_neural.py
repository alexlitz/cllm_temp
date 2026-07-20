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

import os

from c4_min import isa
from c4_min import nibble_runtime as R


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
#: `_sparse_model` still grows past this if a bigger program is ever requested.
_SHARED_CODE_SIZE = 48


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


def _run_neural(instrs, max_steps=200):
    """Run baked bytecode through the unified model (KV-cached sparse driver)."""
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    sparse, L = _sparse_model(code_size=len(instrs) + 2)
    return run_pure_forward_cached(sparse, L, instrs, max_steps=max_steps,
                                   mask=0xFFFFFFFF, evict=False)


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
