"""c4_min runtime library (malloc / free / memset / memcmp) — the BAKED bytecode
proven byte-exact against the word-width reference oracle AND the ISA syscall
INTRINSICS (ops 34-37).

Fast, model-free: everything runs on ``nibble_runtime.ref_interpret_words`` (the
word-width VM whose SI/LI are 32-bit — the faithful oracle for the library, since
the unified neural memory carries a 32-bit VALUE per cell).  The equivalence diff
(baked subroutine vs intrinsic) demonstrates the "compiled from C into bytecode,
NOT a tool call" contract (§687, §747).

The through-the-transformer neural proof lives in ``test_nibble_runtime_neural``
(memory-heavy; separate).

Run:  PYTHONPATH=<repo> python -m pytest c4_min/test_nibble_runtime.py
"""
from __future__ import annotations

from c4_min import isa
from c4_min import nibble_runtime as R


def _run(instrs, data=None):
    return R.ref_interpret_words(instrs, data=data)


def _halt(*asms):
    return R.chain(*asms)


# ---------------------------------------------------------------------------
# MALC
# ---------------------------------------------------------------------------
def test_malloc_first_is_heap_base():
    ax, _ = _run(_halt(R.emit_malloc(4)))
    assert ax == R.HEAP_BASE, f"{hex(ax)} != {hex(R.HEAP_BASE)}"


def test_malloc_increasing_aligned():
    ax, _ = _run(_halt(R.emit_malloc(4), R.emit_malloc(6)))
    assert ax == R.HEAP_BASE + 4 and ax % R.ALIGN == 0, hex(ax)


def test_malloc_matches_intrinsic():
    for size in (4, 8, 12):
        baked, _ = _run(_halt(R.emit_malloc(size)))
        intr, _ = _run([isa.Instr(isa.IMM, size), isa.Instr(isa.PSH, 0),
                        isa.Instr(R.MALC, 0), isa.Instr(isa.HALT, 0)])
        assert baked == intr == R.HEAP_BASE, f"size {size}: {hex(baked)} vs {hex(intr)}"


# ---------------------------------------------------------------------------
# FREE — zero-overwrite (ZFOD): load-after-free reads 0.
# ---------------------------------------------------------------------------
def test_free_zeroes_value():
    ptr = R.HEAP_BASE
    a = R.Asm()
    R._store(a, ptr, lambda a: a.imm(0x99))
    a.code += R.emit_free(ptr).code
    R._load(a, ptr)
    a.exit_()
    ax, mem = _run(a.instrs())
    assert ax == 0 and R.ref_interpret_words  # loaded 0
    assert sum(mem.get(ptr + i, 0) << (8 * i) for i in range(4)) == 0


def test_free_matches_intrinsic():
    ptr = R.HEAP_BASE
    # baked: write 0x99, FREE, read.
    a = R.Asm(); R._store(a, ptr, lambda a: a.imm(0x99))
    a.code += R.emit_free(ptr).code; R._load(a, ptr); a.exit_()
    baked, _ = _run(a.instrs())
    # intrinsic: write 0x99, PSH ptr; FREE, read.
    b = R.Asm(); R._store(b, ptr, lambda b: b.imm(0x99))
    b.imm(ptr).psh().emit(R.FREE); R._load(b, ptr); b.exit_()
    intr, _ = _run(b.instrs())
    assert baked == intr == 0


# ---------------------------------------------------------------------------
# MSET
# ---------------------------------------------------------------------------
def test_memset_fills_n_bytes():
    p, c, n = R.HEAP_BASE, 0xAB, 5
    ax, mem = _run(R.emit_memset(p, c, n).instrs())
    assert ax == p
    assert all(mem.get(p + i) == c for i in range(n))
    assert mem.get(p + n, 0) == 0        # n+1th byte untouched


def test_memset_matches_intrinsic():
    p, c, n = R.HEAP_BASE, 0xAB, 5
    bax, bmem = _run(R.emit_memset(p, c, n).instrs())
    iax, imem = _run([isa.Instr(isa.IMM, p), isa.Instr(isa.PSH, 0),
                      isa.Instr(isa.IMM, c), isa.Instr(isa.PSH, 0),
                      isa.Instr(isa.IMM, n), isa.Instr(R.MSET, 0),
                      isa.Instr(isa.HALT, 0)])
    assert bax == iax
    assert [bmem.get(p + i) for i in range(n)] == [imem.get(p + i) for i in range(n)]


# ---------------------------------------------------------------------------
# MCMP
# ---------------------------------------------------------------------------
def _seg(pairs):
    d = {}
    for base, vals in pairs:
        for i, v in enumerate(vals):
            d[base + i] = v
    return d


def test_memcmp_equal_returns_0():
    pa, pb = R.DATA_BASE, R.DATA_BASE + 0x10
    data = _seg([(pa, [5, 6, 7, 8]), (pb, [5, 6, 7, 8])])
    ax, _ = _run(R.emit_memcmp(pa, pb, 4).instrs(), data=data)
    assert ax == 0


def test_memcmp_mismatch_returns_diff():
    pa, pb = R.DATA_BASE, R.DATA_BASE + 0x10
    data = _seg([(pa, [1, 2, 3, 4]), (pb, [1, 2, 9, 4])])
    ax, _ = _run(R.emit_memcmp(pa, pb, 4).instrs(), data=data)
    assert ax == (3 - 9) & 0xFFFFFFFF


def test_memcmp_matches_intrinsic():
    pa, pb = R.DATA_BASE, R.DATA_BASE + 0x10
    data = _seg([(pa, [1, 2, 3, 4]), (pb, [1, 2, 9, 4])])
    bax, _ = _run(R.emit_memcmp(pa, pb, 4).instrs(), data=data)
    iax, _ = _run([isa.Instr(isa.IMM, pa), isa.Instr(isa.PSH, 0),
                   isa.Instr(isa.IMM, pb), isa.Instr(isa.PSH, 0),
                   isa.Instr(isa.IMM, 4), isa.Instr(R.MCMP, 0),
                   isa.Instr(isa.HALT, 0)], data=data)
    assert bax == iax == (3 - 9) & 0xFFFFFFFF


if __name__ == "__main__":
    import sys
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
