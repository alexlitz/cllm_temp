"""argc/argv reading via READ (§Reading Arguments) — GOLDEN + byte-exact NEURAL.

Proves the READ-based ``__argv_setup`` (``c4_min.nibble_argv``) — the re-plumbing
of argv onto the neural-stdin READ opcode after GETCHAR was removed from the ISA
— three ways:

  1. GOLDEN (clean-room reference VM): the FULL 145-instruction ``__argv_setup``
     reader run on ``isa.interpret`` (READ served from the ARGV-block
     ``InputKVStream``) lays out the argv table + strings and returns
     argc / argv[i][0] byte-for-byte correctly, and matches ``ref_argv_setup``
     (the intended image, computed independently of any bytecode).

  2. NEURAL (the model): the READ-based stdin path run through the pure-forward
     transformer — every VM step is one ``model.forward``; READ(fd=0) is serviced
     by the TOOL_CALL runner from the ARGV-block stdin, and the read byte
     re-enters the token stream as a §Memory KV frame the model reads back with
     LC.  Validated with the COMPACT straight-line reader
     (``emit_read_nth_stdin_byte`` — the exact same READ->memory->LC mechanism as
     the full loop reader, but short enough to run on a LEAN model), byte-exact vs
     the reference VM.

  3. GCC: the model result equals the corresponding byte of the ARGV block gcc
     would hand the program (== gcc's view of ``argv[i][k]`` / the argc bytes).

Run:  OMP_NUM_THREADS=4 PYTHONPATH=<repo> python c4_min/test_argv_read.py
   or: OMP_NUM_THREADS=4 PYTHONPATH=<repo> python -m pytest c4_min/test_argv_read.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import pytest

# Small stack base so the argv region (low bytes) is clear of the stack, which
# grows down from SP_INIT.  Set BEFORE the driver import (it pins SP_INIT).
import c4_min.nibble_pure_forward as _PF
_PF.SP_INIT = 0xF8

from c4_min import isa
from c4_min import nibble_argv as AV
from c4_min import nibble_filesys as FS
from c4_min.nibble_pure_forward import build_pure_forward_model, run_pure_forward


# ---------------------------------------------------------------------------
# Shared LEAN model (built once).  No 32-bit muldiv lookup (the 160k-row select
# block), so the build stays small (~0.7GB) — the compact READ->memory->LC proof
# needs only IMM/PSH/READ/LC.
# ---------------------------------------------------------------------------
_MODELS = {}


def _model(code_size=64):
    """A LEAN model sized to the program.  The per-forward overlay cost scales with
    ``code_size``, so the fast (low-offset) cases use a small model; the argv[1]
    slow cases pass a bigger ``code_size`` to fit their longer reader."""
    if code_size not in _MODELS:
        _MODELS[code_size] = build_pure_forward_model(
            code_size=code_size, include_muldiv=False, include_bitwise=False,
            include_memory=True, include_cmp=True)
    return _MODELS[code_size]


def _neural_read_byte(n, args, buf=0x40, max_steps=70, code_size=64):
    """Read the n-th ARGV-block byte through the pure-forward MODEL (READ<-stdin)."""
    m, L = _model(code_size)
    code = AV.emit_read_nth_stdin_byte(n, buf=buf)
    assert len(code) <= code_size, f"program {len(code)} > code_size {code_size}"
    fio = FS.FileOpState(runner=FS.FileRunner(stdin=AV.argv_stdin(args)))
    return run_pure_forward(m, L, code, max_steps=max_steps, fio=fio, mask=0xFF)[-1] & 0xFF


def _ref_run(code, args, max_steps=8000):
    """The clean-room reference VM (READ from the argv InputKVStream)."""
    return isa.interpret(code, mem_size=256, max_steps=max_steps,
                         stdin=AV.argv_stdin(args))


# ---------------------------------------------------------------------------
# gcc oracle: compile the equivalent C and run it with the EXACT argv the model
# sees (§753-761: argc + argv[0..]).  We invoke via ``os.execv`` with a custom
# argv[0] == ``args[0]`` so gcc's ``argv[0]`` is the program name from the ARGV
# block, NOT the executable path (a bare ``subprocess`` would make argv[0] the
# exe path — '/...').  gcc's exit code is then its view of the requested byte.
# (``-static`` because the sandbox's dynamic loader lacks -lgcc_s.)
# ---------------------------------------------------------------------------
_GCC = shutil.which("gcc") or shutil.which("cc")


def _run_with_argv(exe, args):
    """Run ``exe`` with argv EXACTLY ``args`` (incl. argv[0]); return exit code.

    A clean ``python -c 'os.execv(...)'`` subprocess sets the child's argv[0] to
    ``args[0]`` (the program name from the ARGV block) — a bare ``subprocess.run``
    can't override argv[0].  Done in a fresh subprocess so we never ``fork()`` a
    multi-threaded (torch) interpreter."""
    launcher = (
        "import os,sys; os.execv(sys.argv[1], sys.argv[2:])"
    )
    p = subprocess.run([sys.executable, "-c", launcher, exe, *args],
                       capture_output=True)
    return p.returncode & 0xFF


def _gcc_run(c_source, args):
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "p.c"), os.path.join(d, "p")
        with open(src, "w") as fh:
            fh.write(c_source)
        subprocess.run([_GCC, "-w", "-static", "-o", exe, src], check=True)
        return _run_with_argv(exe, args)


def _gcc_argc(args):
    """gcc's exit code for ``int main(int c,char**v){return c;}`` with args."""
    return _gcc_run("int main(int c,char**v){return c;}", args)


def _gcc_argv_char(args, i, k):
    """gcc's exit code for ``return (unsigned char) argv[i][k];`` with args."""
    return _gcc_run(
        "int main(int c,char**v){return (unsigned char)v[%d][%d];}" % (i, k), args)


# ===========================================================================
# 1. GOLDEN: the FULL __argv_setup reader on the reference VM.
# ===========================================================================
@pytest.mark.parametrize("args", [
    ["prog"], ["prog", "hello"], ["prog", "hello", "World"],
    ["a", "bb", "ccc", "dddd"],
])
def test_golden_full_reader_argc(args):
    tr = _ref_run(AV.emit_return_argc(), args)
    assert tr[-1] == len(args), f"argc: got {tr[-1]} want {len(args)} for {args}"


@pytest.mark.parametrize("args,i", [
    (["prog", "hello", "World"], 0),
    (["prog", "hello", "World"], 1),
    (["prog", "hello", "World"], 2),
    (["a", "bb", "ccc", "dddd"], 3),
])
def test_golden_full_reader_argv_char0(args, i):
    want = ord(args[i][0])
    tr = _ref_run(AV.emit_print_argv_i_char0(i), args)
    assert tr[-1] == want, f"argv[{i}][0]: got {tr[-1]} want {want}"


def test_golden_full_reader_matches_ref_image():
    """The full reader on the reference VM lays out exactly ref_argv_setup's
    argv[i] bytes (each argv[i] round-trips to its arg)."""
    args = ["prog", "hi", "there!"]
    img = AV.ref_argv_setup(args)
    assert img.argc == len(args)
    for i, a in enumerate(args):
        assert img.argv(i) == a.encode("latin-1")
        want = ord(a[0])
        tr = _ref_run(AV.emit_print_argv_i_char0(i), args)
        assert tr[-1] == want, (i, tr[-1], want)


# ===========================================================================
# 2. NEURAL: the READ->memory->LC path through model.forward, byte-exact.
# ===========================================================================
# byte 0-3 = argc (LE uint32); byte 4 = argv[0][0]; then the arg bytes, back to
# back with null terminators.  Each case reads a KNOWN target byte of the block.
# (The pure-forward per-step cost grows with the stream length, so the FAST cases
# read low offsets; the argv[1] case is marked ``slow``.)
@pytest.mark.parametrize("args,n,label", [
    (["A"], 4, "argv[0][0]='A'"),
    (["prog", "hi"], 0, "argc byte0"),
    (["xyz"], 5, "argv[0][1]='y'"),
])
def test_neural_read_byte_byte_exact(args, n, label):
    want = AV.argv_block(args)[n]
    ref = _ref_run(AV.emit_read_nth_stdin_byte(n), args)[-1]
    neu = _neural_read_byte(n, args, max_steps=70)
    assert ref == want, f"reference block[{n}]={ref} want {want} ({label})"
    assert neu == want, f"NEURAL block[{n}]={neu} want {want} ({label})"
    assert neu == ref, f"neural {neu} != reference {ref} ({label})"


@pytest.mark.slow
@pytest.mark.parametrize("args,n,label", [
    (["cat", "f"], 9, "argv[1][0]='f'"),
])
def test_neural_read_argv1_byte_exact(args, n, label):
    """argv[1] indexing through the model (slow: higher offset => more steps)."""
    want = AV.argv_block(args)[n]
    ref = _ref_run(AV.emit_read_nth_stdin_byte(n), args)[-1]
    neu = _neural_read_byte(n, args, max_steps=120, code_size=112)
    assert neu == want == ref, f"argv1 block[{n}] ref={ref} neural={neu} want={want}"


# ===========================================================================
# 3. GCC: the neural result == gcc's view of that argv/argc byte, run with the
# EXACT argv (custom argv[0]) the model reads.
# ===========================================================================
@pytest.mark.skipif(_GCC is None, reason="no gcc")
def test_neural_argc_matches_gcc():
    for args in (["prog"], ["p", "q", "r"], ["x", "y"]):
        neu = _neural_read_byte(0, args)          # argc LE byte 0 == argc (argc<256)
        gcc = _gcc_argc(args)
        assert neu == gcc == len(args), f"argc neural={neu} gcc={gcc} args={args}"


@pytest.mark.skipif(_GCC is None, reason="no gcc")
@pytest.mark.parametrize("args", [["prog", "hi"], ["Xanadu"]])
def test_neural_argv0_char0_matches_gcc(args):
    """model(argv[0][0]) == gcc(argv[0][0]) with argv[0] == args[0]."""
    off = 4                                        # argv[0][0] is block byte 4
    neu = _neural_read_byte(off, args)
    gcc = _gcc_argv_char(args, 0, 0)
    assert neu == gcc == ord(args[0][0]), f"argv[0][0] neural={neu} gcc={gcc} args={args}"


@pytest.mark.slow
@pytest.mark.skipif(_GCC is None, reason="no gcc")
@pytest.mark.parametrize("args,i,k", [
    (["prog", "hello"], 1, 0),      # argv[1][0] = 'h'
    (["prog", "hello"], 1, 2),      # argv[1][2] = 'l'
])
def test_neural_matches_gcc_argv1_char(args, i, k):
    off = 4 + sum(len(a) + 1 for a in args[:i]) + k
    neu = _neural_read_byte(off, args, max_steps=120, code_size=112)
    gcc = _gcc_argv_char(args, i, k)
    assert neu == gcc, f"neural {neu} != gcc {gcc} for argv[{i}][{k}] args={args}"


@pytest.mark.skipif(_GCC is None, reason="no gcc")
def test_neural_argc_byte_matches_gcc():
    """The argc low byte read neurally == gcc's argc (for small argc)."""
    args = ["prog", "a", "b"]
    neu = _neural_read_byte(0, args)         # argc byte 0 (LE) == argc for argc<256
    gcc = _gcc_argc(args)
    assert neu == gcc == len(args), f"neural {neu} gcc {gcc} want {len(args)}"


if __name__ == "__main__":
    failed = 0
    # 1. reference-VM GOLDEN — the FULL 145-instruction __argv_setup reader.
    for args in (["prog"], ["prog", "hello"], ["prog", "hello", "World"],
                 ["a", "bb", "ccc", "dddd"]):
        got = _ref_run(AV.emit_return_argc(), args)[-1]
        ok = got == len(args); failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} GOLDEN argc {args} -> {got}", flush=True)
    for args, i in ((["prog", "hello", "World"], 1), (["prog", "hello", "World"], 2),
                    (["a", "bb", "ccc", "dddd"], 3)):
        want = ord(args[i][0])
        got = _ref_run(AV.emit_print_argv_i_char0(i), args)[-1]
        ok = got == want; failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} GOLDEN argv[{i}][0] {args} -> {got} "
              f"(want {want})", flush=True)
    # 2+3. NEURAL (model.forward) + GCC — argc & argv[0][0] byte-exact.
    print("building LEAN model for neural check ...", flush=True)
    # argc: neural read of block byte 0 (== argc for argc<256) vs gcc(return argc).
    for args in (["prog"], ["p", "q", "r"], ["x", "y"]):
        neu = _neural_read_byte(0, args)
        g = _gcc_argc(args) if _GCC else len(args)
        ok = neu == len(args) == g; failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} NEURAL==gcc argc {args} -> "
              f"neural={neu} gcc={g}", flush=True)
    # argv[0][0]: neural read of block byte 4 vs gcc(return argv[0][0]) with argv[0]=args[0].
    for args in (["prog", "hi"], ["Xanadu"], ["zzz"]):
        want = AV.argv_block(args)[4]
        ref = _ref_run(AV.emit_read_nth_stdin_byte(4), args)[-1]
        neu = _neural_read_byte(4, args)
        g = _gcc_argv_char(args, 0, 0) if _GCC else want
        ok = ref == want == neu == g; failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} NEURAL==gcc argv[0][0]={args[0][0]!r} "
              f"{args} -> ref={ref} neural={neu} gcc={g}", flush=True)
    print(f"\n{'ALL PASS' if not failed else str(failed) + ' FAILED'}", flush=True)
    sys.exit(1 if failed else 0)
