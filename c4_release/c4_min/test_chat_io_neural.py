"""Conversational I/O (READ stdin -> LC -> PRTF) through the ONE UNIFIED model.

Item #3 of the lib-integration: wire ELIZA conversational I/O onto the unified
full-op interpreter and add a BOUNDED full-turn neural test.  A full multi-turn
ELIZA conversation is O(N^2)-token per VM step (the pure-forward VM re-forwards
the whole growing stream every step), so ~15-40-step turns are only marginally
tractable on CPU; this module therefore proves:

  1. the conversational PRIMITIVE — READ(stdin) -> LC(readback) -> PRTF(emit) —
     end-to-end through model.forward (small, always tractable), and
  2. ONE bounded ELIZA turn through the unified model, byte-exact vs the plain-
     python reference, under an explicit step cap (skipped if too slow / the
     model can't be built for memory).

We do NOT claim untested full multi-turn (the task's tractability caveat).

The unified model build is memory-heavy (~62 GB dense before sparse conversion);
run on a box with headroom, single-process, OMP_NUM_THREADS=4.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<repo> python c4_min/test_chat_io_neural.py
"""
from __future__ import annotations

import os

from c4_min import isa
from c4_min import nibble_filesys as FS


BUF = 0x40   # low-256 window so the LC CAM reads the READ-back bytes byte-exact

_MODEL = None
_L = None
_SPARSE = None


def _seed_cstring(d, addr, s):
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        d[addr + i] = b
    d[addr + len(s)] = 0


def _model(code_size):
    """Build the unified model + sparse wrapper once (base build, no addr32 —
    conversational I/O uses low-window buffers, so the base 8-bit query suffices).
    """
    global _MODEL, _L, _SPARSE
    if _SPARSE is None:
        from c4_min.nibble_pure_forward_complete import build_pure_forward_complete_model
        from c4_min.sparse_forward import SparseTransformer
        _MODEL, _L = build_pure_forward_complete_model(
            code_size=code_size, recurrent_divmod=True)
        _SPARSE = SparseTransformer(_MODEL, compute_mode="dense_kernel")
    return _SPARSE, _L


# ---------------------------------------------------------------------------
# 1. The conversational PRIMITIVE: READ(stdin) -> LC(readback) -> PRTF(emit).
# ---------------------------------------------------------------------------
def test_read_lc_prtf_primitive_neural():
    FMT = 0x80
    code = isa.assemble([
        ("IMM", 0), ("PSH", 0),          # push fd=0 (stdin)
        ("IMM", BUF), ("PSH", 0),        # push buf
        ("IMM", 2),                      # AX = n = 2
        ("READ", 0),                     # AX = read(0, buf, 2) -> n_read
        ("IMM", BUF + 0), ("LC", 0),     # AX = buf[0]  'A'
        ("IMM", BUF + 1), ("LC", 0),     # AX = buf[1]  'B'
        ("IMM", FMT), ("PSH", 0), ("PRTF", 0),   # printf("hi\n")
        ("HALT", 0),
    ])
    data = {}
    _seed_cstring(data, FMT, "hi\n")
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    sparse, L = _model(code_size=len(code) + 2)
    fio = FS.FileOpState(
        runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                             stdin=FS.InputKVStream(b"AB")))
    tr = run_pure_forward_cached(sparse, L, code, max_steps=48, mask=0xFF,
                                 fio=fio, data_seg=data, evict=False)
    # the READ delivered 2 bytes; LC read them back byte-exact through the CAM.
    assert tr[5] == 2, f"READ n_read (got {tr[5]})"
    assert tr[7] == ord("A"), f"LC buf[0] (got {tr[7]})"
    assert tr[9] == ord("B"), f"LC buf[1] (got {tr[9]})"
    # PRTF emitted the greeting to stdout.
    assert bytes(fio.runner.stdout) == b"hi\n", bytes(fio.runner.stdout)


# ---------------------------------------------------------------------------
# 2. ONE bounded ELIZA turn through the unified model, byte-exact vs reference.
# ---------------------------------------------------------------------------
def test_eliza_one_turn_bounded_neural():
    import pytest
    from c4_min import chat_eliza as CE

    eliza = CE.build_chat_min()
    message = "yes"                      # a short prefix-match turn (few steps)
    ref = CE.chat_turn_ref(eliza, message)

    # Build once for the ELIZA code size; drive the bounded turn through the model.
    from c4_min.sparse_forward import SparseTransformer
    from c4_min.nibble_pure_forward_complete import build_pure_forward_complete_model
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    model, L = build_pure_forward_complete_model(
        code_size=len(eliza.code) + 2, recurrent_divmod=True)
    sparse = SparseTransformer(model, compute_mode="dense_kernel")

    fio = CE._fresh_fio(message)
    STEP_CAP = 120                        # bounded: fail fast rather than hang
    run_pure_forward_cached(sparse, L, eliza.code, max_steps=STEP_CAP, mask=0xFF,
                            fio=fio, data_seg=dict(eliza.data_seg), evict=False)
    mdl = bytes(fio.runner.stdout).decode("latin-1")
    assert mdl == ref, f"ELIZA turn byte-mismatch\n  ref  ={ref!r}\n  model={mdl!r}"


if __name__ == "__main__":
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    fns = [("test_read_lc_prtf_primitive_neural", test_read_lc_prtf_primitive_neural),
           ("test_eliza_one_turn_bounded_neural", test_eliza_one_turn_bounded_neural)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
