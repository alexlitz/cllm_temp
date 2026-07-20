"""Conversational I/O (READ stdin -> match -> PRTF) through the ONE UNIFIED model.

ELIZA runs end-to-end through ``model.forward``: a user message enters via the
tool-use input-KV (``READ`` fd 0), the ELIZA bytecode pattern-matches it (LC + EQ
+ branches, all in-weights), and the therapist response comes back out via
``PRTF`` — a real read/compute/write chat turn, byte-exact vs the plain-python
reference running the IDENTICAL bytecode.

This module proves, THROUGH THE KV-CACHED SPARSE DRIVER WITH EVICTION:

  1. the conversational PRIMITIVE — READ(stdin) -> LC(readback) -> PRTF(emit) —
     end-to-end through model.forward (small, always tractable);
  2. ONE FULL ELIZA turn (READ the message, prefix-match every rule, PRTF the
     matching response) byte-exact vs the reference — a WHOLE turn, no step cap
     truncation (the previous cap was a fail-fast guard, not a real bound); and
  3. a FEW ELIZA turns of a real conversation (opt-in, see the tractability note),
     each byte-exact, state carried across turns as the growing conversation log.

Tractability (measured, honest — see ``docs`` and the module ``__main__`` bench):
  Each VM step is ONE ``model.forward`` over a fixed ~31-row window through the
  full unified model (305 physical blocks, sparse-CSR CPU) ~= 10 s/step, so a
  turn of N VM steps is ~10N s.  With ``evict=True, prune_interval=60`` the KV
  cache stays FLAT (max_cache_size ~= 157 regardless of turn length — the
  eviction that also holds the peak RSS at ~8 GB), so cost is LINEAR in steps,
  NOT the O(N^2) of the un-evicted growing-stream path.  The per-step 305-block
  forward is the fixed floor; eviction removes the memory blow-up + the O(N^2)
  cache-growth compute, it does not make the per-step forward cheap.

  A compact 2-rule ELIZA turn is ~29-35 VM steps (~5-6 min); the full 8-rule
  table's later rules are ~71-77 steps (~12 min).  So ONE full turn is always
  tractable; a few turns are tractable but slow (opt-in ``C4_CHAT_NEURAL_MULTITURN
  =1``, ~15-18 min for 3 short turns).  We do NOT claim untested-long multi-turn.

The unified model build is memory-heavy DENSE (~62 GB); we always use the
STREAMING SPARSE build (~6-8 GB peak) + KV-cache eviction + ``malloc_trim`` between
runs, single-process, ``OMP_NUM_THREADS=4``.

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<repo> python c4_min/test_chat_io_neural.py
      OMP_NUM_THREADS=4 C4_CHAT_NEURAL_MULTITURN=1 PYTHONPATH=<repo> \
            python c4_min/test_chat_io_neural.py     # + the multi-turn test
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os

from c4_min import isa
from c4_min import nibble_filesys as FS


BUF = 0x40   # low-256 window so the LC CAM reads the READ-back bytes byte-exact

# Eviction cadence proven memory-flat for the looping library routines
# (test_nibble_runtime_neural): prune every 60 tokens keeps the per-block cache
# flat (~157 rows) and the peak RSS ~8 GB even as the token stream grows to ~2k+.
_PRUNE_INTERVAL = 60


# ---------------------------------------------------------------------------
# Memory discipline (mirrors test_nibble_runtime_neural._release_memory): the
# KV-cached dense-kernel sparse forward allocates transient CPU tensors per step;
# glibc keeps freed arenas resident, so ``malloc_trim(0)`` between runs returns
# them to the OS and the shared-process RSS stays flat across the tests.
# ---------------------------------------------------------------------------
_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")


def _release_memory():
    gc.collect()
    try:
        _LIBC.malloc_trim(0)
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------------------
# Shared streaming model (built once, oversized to fit the largest test program,
# grown monotonically if a bigger one is ever requested — the sizing pattern from
# test_nibble_runtime_neural, since the code overlay has a fixed ``code_size``
# instruction slot count).  Conversational I/O uses low-window buffers, so the
# base 8-bit query suffices — addr32=False.
#
# ``_MIN_CODE_SIZE`` floor: the streaming build's OWN value-liveness probe compiles
# a battery of C programs (``compact_alloc._default_probe_programs``) and overlays
# each onto the ``code_size`` code slots, so the shared model must be built at
# least as big as the LONGEST probe program (the func-args probe ~50 instrs) — a
# model built for a tiny program (e.g. the 14-instr READ/LC/PRTF primitive) makes
# the build's own probe overlay index past ``L.CODE_OP`` (IndexError).  A model
# built at a LARGER code_size is byte-identical for a SHORTER program, so this
# floor is safe for every test.  55 is the size the sibling library-neural suite
# (test_nibble_runtime_neural) builds at.
# ---------------------------------------------------------------------------
_MIN_CODE_SIZE = 64

_SPARSE = None
_L = None
_CODE_SIZE = 0


def _model(code_size: int):
    global _SPARSE, _L, _CODE_SIZE
    want = max(code_size, _MIN_CODE_SIZE)
    if _SPARSE is None or want > _CODE_SIZE:
        from c4_min.lib_neural import build_lib_model_streaming
        _SPARSE = _L = None                      # free the old model before rebuild
        _SPARSE, _L, _ = build_lib_model_streaming(
            code_size=want, recurrent_divmod=True, addr32=False)
        _CODE_SIZE = want
    return _SPARSE, _L


def _seed_cstring(d, addr, s):
    if isinstance(s, str):
        s = s.encode("latin-1")
    for i, b in enumerate(s):
        d[addr + i] = b
    d[addr + len(s)] = 0


def _run_turn(sparse, L, eliza, message, max_steps, stats=None):
    """Drive ONE ELIZA turn for ``message`` through the KV-cached sparse model
    with eviction ON; return the PRTF stdout string.  The message enters via the
    input-KV (READ fd 0); the reply comes back via PRTF.  Eviction keeps the KV
    cache flat + the peak RSS bounded, ``malloc_trim`` after returns the arena."""
    from c4_min import chat_eliza as CE
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
    fio = CE._fresh_fio(message)
    try:
        run_pure_forward_cached(
            sparse, L, eliza.code, max_steps=max_steps, mask=0xFF,
            fio=fio, data_seg=dict(eliza.data_seg),
            evict=True, prune_interval=_PRUNE_INTERVAL, stats=stats)
    finally:
        _release_memory()
    return bytes(fio.runner.stdout).decode("latin-1")


# A COMPACT ELIZA rule table (2 rules) keeps a turn to ~29-35 VM steps so the
# bounded + multi-turn tests are as tractable as the model allows while still
# exercising the WHOLE READ -> per-byte match -> PRTF-response loop (and both the
# rule-hit and fallback branches).  Byte-exact against the same-bytecode reference.
_COMPACT_RULES = [("hi", "HELLO!\n"), ("no", "WHY NOT?\n")]


# ---------------------------------------------------------------------------
# 1. The conversational PRIMITIVE: READ(stdin) -> LC(readback) -> PRTF(emit).
#    Small + always tractable; the always-run smoke of the I/O path.
# ---------------------------------------------------------------------------
def test_read_lc_prtf_primitive_neural():
    from c4_min.nibble_pure_forward_cached import run_pure_forward_cached
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
    sparse, L = _model(code_size=len(code) + 2)
    fio = FS.FileOpState(
        runner=FS.FileRunner(fs=FS.StubFilesystem({}),
                             stdin=FS.InputKVStream(b"AB")))
    try:
        tr = run_pure_forward_cached(sparse, L, code, max_steps=48, mask=0xFF,
                                     fio=fio, data_seg=data,
                                     evict=True, prune_interval=_PRUNE_INTERVAL)
    finally:
        _release_memory()
    # the READ delivered 2 bytes; LC read them back byte-exact through the CAM.
    assert tr[5] == 2, f"READ n_read (got {tr[5]})"
    assert tr[7] == ord("A"), f"LC buf[0] (got {tr[7]})"
    assert tr[9] == ord("B"), f"LC buf[1] (got {tr[9]})"
    # PRTF emitted the greeting to stdout.
    assert bytes(fio.runner.stdout) == b"hi\n", bytes(fio.runner.stdout)


# ---------------------------------------------------------------------------
# 2. ONE FULL ELIZA turn through the unified model, byte-exact vs reference.
#    A WHOLE turn — READ the message, prefix-match every rule, PRTF the response —
#    runs to HALT (no truncating step cap), byte-exact against the same bytecode
#    on the plain-python reference.  Eviction keeps the cache flat.
# ---------------------------------------------------------------------------
def test_eliza_one_turn_bounded_neural():
    from c4_min import chat_eliza as CE

    eliza = CE.build_chat_min(rules=_COMPACT_RULES)
    message = "no thanks"                 # rule-1 hit: the WHOLE match+respond turn
    ref = CE.chat_turn_ref(eliza, message)
    assert ref == "WHY NOT?\n", ref       # sanity: the reference itself is right

    sparse, L = _model(code_size=len(eliza.code) + 2)
    stats = {}
    # max_steps generous of the ~35-step turn; the turn HALTs well inside it, so
    # this is a whole turn, not a truncated prefix.
    mdl = _run_turn(sparse, L, eliza, message, max_steps=200, stats=stats)
    assert mdl == ref, f"ELIZA turn byte-mismatch\n  ref  ={ref!r}\n  model={mdl!r}"
    # eviction kept the cache FLAT (bounded) even as the stream grew — the memory
    # lever.  (A ~35-step turn appends ~35*30 tokens; the flat cache proves prune.)
    assert stats["steps"] <= 60, stats            # a whole turn, not a runaway loop
    assert stats["max_cache_size"] < stats["max_seq_len"], stats  # eviction fired


# ---------------------------------------------------------------------------
# 3. A FEW ELIZA turns of a real conversation, byte-exact each turn (opt-in).
#    Each turn is an independent READ/compute/write run (a fresh input-KV carrying
#    that turn's message) — exactly how the reference ``run_chat`` proceeds — and
#    the conversation state is the growing (user, eliza) log carried across turns.
#    ~5-6 min/turn on CPU, so gated behind C4_CHAT_NEURAL_MULTITURN=1.
# ---------------------------------------------------------------------------
def test_eliza_multi_turn_neural():
    if os.environ.get("C4_CHAT_NEURAL_MULTITURN") != "1":
        # ~5-6 min/turn (the fixed 305-block per-step forward); opt-in only.
        import pytest
        pytest.skip("multi-turn neural chat is slow (~5-6 min/turn); "
                    "set C4_CHAT_NEURAL_MULTITURN=1 to run")
    from c4_min import chat_eliza as CE

    eliza = CE.build_chat_min(rules=_COMPACT_RULES)
    conversation = ["hi there", "no way", "whatever"]   # rule0, rule1, fallback
    sparse, L = _model(code_size=len(eliza.code) + 2)

    log = []                                            # conversation state
    for message in conversation:
        ref = CE.chat_turn_ref(eliza, message)
        stats = {}
        mdl = _run_turn(sparse, L, eliza, message, max_steps=200, stats=stats)
        assert mdl == ref, (f"turn {message!r} byte-mismatch\n"
                            f"  ref  ={ref!r}\n  model={mdl!r}")
        assert stats["max_cache_size"] < stats["max_seq_len"], stats  # eviction
        log.append((message, mdl))
    # the conversation exercised BOTH a rule hit and the fallback, byte-exact.
    replies = [r for _, r in log]
    assert replies == ["HELLO!\n", "WHY NOT?\n", "PLEASE GO ON.\n"], log


if __name__ == "__main__":
    import sys
    import time
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    fns = [("test_read_lc_prtf_primitive_neural", test_read_lc_prtf_primitive_neural),
           ("test_eliza_one_turn_bounded_neural", test_eliza_one_turn_bounded_neural),
           ("test_eliza_multi_turn_neural", test_eliza_multi_turn_neural)]
    failed = 0
    skipped = 0
    for name, fn in fns:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name}  ({time.time() - t0:.1f}s)")
        except BaseException as exc:  # noqa: BLE001 (pytest.skip raises BaseException)
            if type(exc).__name__ == "Skipped":
                skipped += 1
                print(f"SKIP {name}: {exc}")
                continue
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {exc!r}")
    passed = len(fns) - failed - skipped
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
