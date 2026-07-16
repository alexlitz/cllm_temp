"""Tool-use I/O (OPEN / READ / CLOS / PRTF) through the PURE-FORWARD small model.

CHK-1 item #5 ("Tool use IO works correctly"): the file syscalls run through the
TOOL_CALL protocol on the pure-forward VM — the model emits the tool call, the
runner (``nibble_filesys.FileRunner``) services it, the bytes flow back into the
VM as §Memory KV store frames, and a later LC reads them back byte-exact.

Unlike ``test_nibble_filesys`` (which drives the HYBRID ``blogspec_run._apply_op``
path), these tests run the ops through the actual pure-forward drivers:

  * ``run_pure_forward_complete`` — the naive one-forward-per-step driver.
  * ``run_pure_forward_cached``  — the SPARSE + KV-CACHED driver (the CHK-1
    headline path: ``build_pure_forward_complete_model`` wrapped in
    ``SparseTransformer``, driven at O(cache)/step).

A file op is the ONE class the transformer does NOT compute (§Tool Use Mode); the
driver performs it and overrides the registers.  A READ writes its bytes into the
KV store log so the model can read them back with LC — the bytes literally
re-enter the token stream as memory KV entries.

Addressing note: the pure-forward LI/LC memory CAM keys the query on the LOW BYTE
of the address (``compile_addr_expand`` ``n_bits=8``), so the READ destination
buffer must sit in the low-256-byte window for the model to attend to the read
bytes back — exactly like the existing SI/LI memory tests (0x40 / 0x44 / 0x80).

Run (naive only, fast-ish):
    C4_FILEIO_TEST_MODE=naive PYTHONPATH=<repo> python c4_min/test_fileio_pure_forward.py
Run (cached, the headline path — default):
    PYTHONPATH=<repo> python c4_min/test_fileio_pure_forward.py
"""
from __future__ import annotations

import os

from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.nibble_pure_forward_complete import (
    build_pure_forward_complete_model, run_pure_forward_complete, SP_INIT,
)


HELLO = b"hello, file!\n"
FNAME = "greeting.txt"
NAME_ADDR = 0x20000      # filename lives in the (runner-side) data segment
BUF_ADDR = 0x40          # READ dest buffer: low addr so the LC CAM can read back
FMT_ADDR = 0x22000

# Build the model ONCE (shared across the tests in this module).
_MODEL = None
_L = None
_SPARSE = None


def _seed_cstring(d, addr, s):
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        d[addr + i] = b
    d[addr + len(s)] = 0


def _model():
    global _MODEL, _L
    if _MODEL is None:
        _MODEL, _L = build_pure_forward_complete_model(
            code_size=48, include_bitwise=True, include_divmod=False)
    return _MODEL, _L


def _sparse():
    global _SPARSE
    if _SPARSE is None:
        from c4_min.sparse_forward import SparseTransformer
        m, _ = _model()
        _SPARSE = SparseTransformer(m, compute_mode="dense_kernel")
    return _SPARSE


def _mk_fio(files=None, stdin=b""):
    fs = FS.StubFilesystem(files if files is not None else {FNAME: HELLO})
    return FS.FileOpState(runner=FS.FileRunner(fs=fs, stdin=FS.InputKVStream(stdin)))


def _file_program():
    """OPEN greeting.txt -> READ 13 bytes -> CLOS ; then LC three bytes back out
    of the buffer the file was read into (buf[0], buf[1], buf[12]).  Marshalling
    follows ``dispatch_file_op`` (OPEN: name=pop,flags=imm ; READ: fd=pop,buf=pop,
    n=AX ; CLOS: fd=AX)."""
    return isa.assemble([
        ("IMM", NAME_ADDR), ("PSH", 0), ("OPEN", 0),   # AX = open(name) -> fd (=3)
        ("PSH", 0),                                    # push fd (AX = fd)
        ("IMM", BUF_ADDR), ("PSH", 0),                 # push buf
        ("IMM", len(HELLO)),                           # AX = count
        ("READ", 0),                                   # AX = read(...) -> n_read
        ("IMM", 3), ("CLOS", 0),                       # AX = close(fd) -> 0
        ("IMM", BUF_ADDR + 0), ("LC", 0),              # AX = buf[0]  'h'=104
        ("IMM", BUF_ADDR + 1), ("LC", 0),              # AX = buf[1]  'e'=101
        ("IMM", BUF_ADDR + 12), ("LC", 0),             # AX = buf[12] '\n'=10
        ("HALT", 0),
    ])


def _data_seg():
    d = {}
    _seed_cstring(d, NAME_ADDR, FNAME)
    return d


def _run(code, fio, data_seg, mode=None):
    """Drive ``code`` through the requested pure-forward driver."""
    mode = mode or os.environ.get("C4_FILEIO_TEST_MODE", "cached")
    if mode == "naive":
        m, L = _model()
        return run_pure_forward_complete(
            m, L, code, max_steps=48, mask=0xFFFFFFFF, fio=fio,
            data_seg=dict(data_seg))
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    _, L = _model()
    return run_pure_forward_cached(
        _sparse(), L, code, max_steps=48, mask=0xFFFFFFFF, fio=fio,
        data_seg=dict(data_seg), evict=False)


# ---------------------------------------------------------------------------
# 1. OPEN -> READ -> CLOS through the pure-forward model, bytes read back via LC.
# ---------------------------------------------------------------------------
def test_open_read_close_pure_forward():
    code = _file_program()
    fio = _mk_fio(files={FNAME: HELLO})
    tr = _run(code, fio, _data_seg())

    # OPEN lands fd=3; READ lands n_read=len(HELLO).
    assert tr[2] == 3, f"OPEN fd (got {tr[2]})"
    assert tr[7] == len(HELLO), f"READ n_read (got {tr[7]})"
    # CLOS returns 0.
    assert tr[9] == 0, f"CLOS (got {tr[9]})"
    # the file bytes flowed back into VM memory: LC reads them byte-exact.
    lc = [tr[11], tr[13], tr[15]]
    assert lc == [HELLO[0], HELLO[1], HELLO[12]], f"LC readback (got {lc})"
    # the exact tool-call token stream was emitted.
    assert [c.type for c in fio.calls] == ["open", "read", "close"]
    assert fio.runner.log[0][0].startswith("TOOL_CALL:open:")
    # the read response carried the file bytes byte-for-byte.
    assert HELLO.hex() in fio.runner.log[1][1]


# ---------------------------------------------------------------------------
# 2. Byte-exact vs a plain-python reference (the whole AX trace).
# ---------------------------------------------------------------------------
def _ref_trace(code, files):
    runner = FS.FileRunner(fs=FS.StubFilesystem(files))
    fio = FS.FileOpState(runner=runner)
    mem = {}
    _seed_cstring(mem, NAME_ADDR, FNAME)
    ax, sp, pc, steps = 0, SP_INIT, 0, 0
    stack = {}
    out = []

    def push(v):
        nonlocal sp
        sp -= 4; stack[sp] = v & 0xFFFFFFFF

    def pop():
        nonlocal sp
        v = stack.get(sp, 0); sp += 4; return v

    class _M:
        def load_int(s, a, w=4):
            return sum(mem.get(a + i, 0) << (8 * i) for i in range(w))

        def store_int(s, a, val, w=4):
            for i in range(w):
                mem[a + i] = (val >> (8 * i)) & 0xFF
    M = _M()
    while pc < len(code) and steps < 200:
        steps += 1
        op, imm = code[pc].op, code[pc].imm
        pc += 1
        if op == isa.IMM:
            ax = imm & 0xFFFFFFFF
        elif op == isa.PSH:
            push(ax)
        elif op == isa.LC:
            ax = M.load_int(ax, 1) & 0xFF
        elif op in FS.FILE_OPCODES:
            ax = FS.dispatch_file_op(op, ax, imm, pop, M, fio)
        elif op == isa.HALT:
            out.append(ax); break
        else:
            raise NotImplementedError(isa.NAMES.get(op, op))
        out.append(ax & 0xFFFFFFFF)
    return out


def test_file_trace_byte_exact_vs_reference():
    code = _file_program()
    fio = _mk_fio(files={FNAME: HELLO})
    tr = _run(code, fio, _data_seg())
    ref = _ref_trace(code, {FNAME: HELLO})
    assert tr == ref, f"pure-forward trace != reference\n  pf ={tr}\n  ref={ref}"


# ---------------------------------------------------------------------------
# 3. PRTF: the format string is formatted and the byte count re-enters AX.
# ---------------------------------------------------------------------------
def test_prtf_pure_forward():
    code = isa.assemble([("IMM", FMT_ADDR), ("PSH", 0), ("PRTF", 0), ("HALT", 0)])
    fio = _mk_fio(files={})
    fio.pending_args = [7, ord("Z")]
    data = {}
    _seed_cstring(data, FMT_ADDR, "hi %d %c\n")
    tr = _run(code, fio, data)
    assert bytes(fio.runner.stdout) == b"hi 7 Z\n", bytes(fio.runner.stdout)
    assert tr[2] == len(b"hi 7 Z\n"), f"PRTF n_written (got {tr[2]})"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
