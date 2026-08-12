#!/usr/bin/env python3
"""Transpiler C90 conformance harness (SOURCE-path, CPU-only, no model load).

Runs each C90 case through TWO pipelines and compares the observable (exit code
mod 256 + stdout):

  (A) gcc -m32 -std=c90            -> reference exit/stdout          (x86 ground truth)
  (B) transpile.py -> compile_c -> c4vm.C4VM.run  -> exit/stdout     (the PORT path)

This is the pipeline the Doom port uses (id_port/transpile.py flattens vanilla
linuxdoom C90 into the c4 subset, then src.compiler.compile_c -> c4 bytecode ->
c4vm).  Testing it against a BROAD C90 corpus surfaces the transpiler's
systematic bug CLASSES at the source, so they can be fixed once (a clean
"vanilla linuxdoom -> generic transpiler -> byte-exact Doom" path) rather than
hand-patched per-bug in doom_run.c.

Stage outcomes per case:
  gcc_fail          : gcc rejected the source (should not happen for valid C90)
  transpile_error   : transpile.py raised (a transpiler crash bug)
  compile_error     : compile_c rejected the transpiled output (subset limit / bug)
  vm_error          : the c4vm raised / did not halt
  PASS              : observable matches gcc
  MISMATCH          : observable differs (a MIS-LOWERING bug -- the interesting class)

sizeof(int) gap: gcc -m32 has sizeof(int)==4; the c4 VM word is 8.  Cases whose
result legitimately depends on sizeof(int) are tagged `c4_x86_sizeof_gap` and are
NOT counted as failures (documented, expected).

Pure ADDITION under id_port/c90_e2e/ + a c4_doom sibling; touches no model-build
file (golden 174ece66 unchanged).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- import paths: transpile.py (c4_doom) + src.compiler + c4vm (c4_doom) -----
# Override with C4_DOOM_IDPORT (points at c4_doom/id_port: transpile.py + c4vm.py
# + struct_engine.py) and C4_RELEASE_ROOT (points at c4_release: src/compiler.py).
# The transpiler baseline is the ``task-transpiler-c90-b1b8`` branch (95.7% on the
# 117-case hand corpus); set both env vars to that branch's worktrees.
DOOM_IDPORT = os.environ.get("C4_DOOM_IDPORT",
                             "/tmp/wt_c90_doom_mine/id_port")
C4_RELEASE = os.environ.get("C4_RELEASE_ROOT",
                            "/tmp/wt_c90_rel_mine/c4_release")
# this harness's own directory (holds the CORPUS-SCOPED corpus_libc.c4)
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (DOOM_IDPORT, C4_RELEASE):
    if p not in sys.path:
        sys.path.insert(0, p)

import transpile as T                      # c4_doom/id_port/transpile.py
from src.compiler import Compiler          # c4_release/src/compiler.py
import c4vm                                # c4_doom/id_port/c4vm.py (faithful word VM)

try:
    import struct_engine as _SE            # c4_doom/id_port/struct_engine.py
    from struct_lower import Contract as _Contract
except Exception:                          # pragma: no cover
    _SE = None
    _Contract = None


def build_file_contract(src: str):
    """Build a per-FILE struct/union layout contract from THIS program's own
    struct/union definitions (the same struct_engine path the doom whole-tree
    build uses, but scoped to one standalone file).  Returns a Contract or None.

    The transpiler needs a struct layout to lower ``s.field`` -> ``base[off]``;
    the shipped ``struct_layout.json`` only carries DOOM's structs, so a generic
    program with its OWN structs must supply its own layout.  This isolates a
    real struct-LOWERING bug from mere absence-of-contract noise."""
    if _SE is None or _Contract is None:
        return None
    if "struct" not in src and "union" not in src:
        return None
    try:
        local, _lay, _defs = _SE.build_layout_from_sources(
            {"__case__.c": _SE._scrub(src)})
        return _Contract(local)
    except Exception:
        return None

SYSCALL_OPS = {c4vm.OPEN, c4vm.READ, c4vm.CLOS, c4vm.PRTF,
               c4vm.PUTCHAR, c4vm.GETCHAR, 40, 41}
ADJ = 7


def _decode(bytecode: List[int]) -> List[Tuple[int, int]]:
    code = []
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        if imm >= (1 << 55):
            imm -= (1 << 56)
        code.append((op, imm))
    return code


def _compute_argc(code):
    argc_at = {}
    for i, (op, imm) in enumerate(code):
        if op in SYSCALL_OPS and i + 1 < len(code) and code[i + 1][0] == ADJ:
            argc_at[i] = code[i + 1][1] // 8
    return argc_at


# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    category: str
    stage: str = ""              # PASS / MISMATCH / transpile_error / compile_error / vm_error / gcc_fail
    gcc_exit: Optional[int] = None
    gcc_stdout: str = ""
    port_exit: Optional[int] = None
    port_stdout: str = ""
    detail: str = ""
    bug_class: str = ""          # attributed class for MISMATCH/error
    transpiled: str = ""         # kept only for failing cases (debug)


def run_gcc(src: str, stdin: bytes, timeout: int = 20) -> Tuple[Optional[int], str, str]:
    """gcc -m32 -std=c90 compile+run.  Returns (exit_mod256, stdout, err)."""
    with tempfile.TemporaryDirectory() as td:
        cpath = Path(td) / "case.c"
        bpath = Path(td) / "case"
        cpath.write_text(src)
        cp = subprocess.run(
            ["gcc", "-m32", "-std=c90", "-w", "-O0", str(cpath), "-o", str(bpath)],
            capture_output=True, timeout=timeout)
        if cp.returncode != 0:
            return None, "", "gcc-compile: " + cp.stderr.decode("latin-1", "replace")[:400]
        try:
            rp = subprocess.run([str(bpath)], input=stdin, capture_output=True,
                                timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "", "gcc-run: TIMEOUT"
        # strip the harmless ld.so preload warning that leaks to child stdout? it's stderr; keep stdout clean
        return rp.returncode & 0xFF, rp.stdout.decode("latin-1", "replace"), ""


import re as _re


def _inject_static_init_call(tsrc: str) -> str:
    """Emulate the doom link step: if the transpiled output defines a NON-empty
    ``c4_static_init()`` (deferred global/array/static-value initializers) and
    ``main`` does not already call it, insert ``c4_static_init();`` as main's
    first statement.  Byte-neutral when the init body is empty."""
    m = _re.search(r"int\s+c4_static_init\s*\(\s*\)\s*\{(.*?)\}", tsrc, _re.S)
    if not m or not m.group(1).strip():
        return tsrc               # no deferred inits -> nothing to wire
    if _re.search(r"\bc4_static_init\s*\(\s*\)\s*;", tsrc):
        return tsrc               # already called
    # find `... main ( ... ) {` and insert the init call AFTER main's leading
    # local declarations (c4 requires all locals at the top of the function
    # BEFORE any statement; injecting at the brace would put a call before them).
    mm = _re.search(r"(\bmain\s*\([^)]*\)\s*\{)", tsrc)
    if not mm:
        return tsrc
    i = mm.end()
    n = len(tsrc)
    # skip over leading declaration lines: whitespace, then `int|char [*] name ;`
    decl = _re.compile(r"\s*(?:int|char)[ \t\*]+[A-Za-z_]\w*\s*;")
    while True:
        m2 = decl.match(tsrc, i)
        if not m2:
            break
        i = m2.end()
    return tsrc[:i] + " c4_static_init();" + tsrc[i:]


def run_port(src: str, stdin: bytes, timeout_cycles: int = 40_000_000) -> Result:
    """transpile -> compile_c -> c4vm.  Returns a partially-filled Result (stage,
    port_exit, port_stdout, detail, transpiled)."""
    r = Result(name="", category="")
    # -- stage 1: transpile (with a per-file struct/union contract) --
    try:
        contract = build_file_contract(src)
        # preprocess=True: run the transpiler's OPT-IN full C90 macro
        # preprocessor (object + function-like macros with #/##, + stdint/EOF/
        # NULL prelude).  DEFAULT-OFF for the DOOM build (byte-identity), ON here
        # because the c-testsuite cases are self-contained programs that expect
        # real cpp semantics.
        tsrc = T.transpile(src, contract=contract, preprocess=True) \
            if contract is not None else T.transpile(src, preprocess=True)
    except Exception as e:
        r.stage = "transpile_error"
        r.detail = f"{type(e).__name__}: {e}"
        return r
    # LINK-STEP EMULATION: the transpiler DEFERS global/array/static-value inits
    # into a generated ``c4_static_init()`` helper that the STANDALONE per-file
    # output never calls -- the doom LINK step (link_doom_transpiled.py) injects
    # ``c4_static_init();`` at startup.  Mirror that here so we test the
    # transpiler's LOWERING, not the (separately-owned) link wiring.  If the init
    # helper is empty this is a no-op.
    tsrc = _inject_static_init_call(tsrc)
    r.transpiled = tsrc
    # -- stage 2: compile_c (via Compiler so we can measure globals for the heap) --
    try:
        # match run_doom_c4: raise the bump-heap base above the globals region so
        # a global (e.g. a static array) is not overwritten by the heap.
        stdlib_path = Path(C4_RELEASE) / "src" / "compiler" / "stdlib" / "memory.c4"
        if not stdlib_path.exists():
            stdlib_path = Path(C4_RELEASE) / "src" / "stdlib" / "memory.c4"
        stdlib = stdlib_path.read_text() if stdlib_path.exists() else ""
        # CORPUS-SCOPED C90 libc: standard-library functions (calloc/strcpy/
        # strcmp/strlen/sprintf/...) the c-testsuite cases call that memory.c4
        # does NOT provide.  This is appended ONLY here in the harness -- it is
        # NOT part of the shared c4 build stdlib and is NOT linked into the Doom
        # build (Doom supplies its own libc via id_port/doom_libc.c), so the
        # byte-exact Doom path (golden 174ece66) is untouched.  It defines only
        # functions memory.c4 lacks (no malloc/free/memset/memcmp duplicates).
        corpus_libc_path = Path(HARNESS_DIR) / "corpus_libc.c4"
        corpus_libc = corpus_libc_path.read_text() if corpus_libc_path.exists() else ""
        if corpus_libc:
            # concat order: memory.c4 first (owns __heap_ptr, rewritten below),
            # then the corpus libc.  The heap-base .replace() targets memory.c4.
            stdlib = stdlib + "\n" + corpus_libc
        full = tsrc + ("\n" + stdlib if stdlib else "")
        if stdlib:
            probe = Compiler()
            probe.compile(full)
            gtop = max((s.value for s in probe.symbols.values()
                        if getattr(s, "sclass", "") == "Glo"), default=0x20000)
            heap_base = ((gtop + 0x10000) + 0xFFFF) & ~0xFFFF
            heap_base = max(heap_base, 0x100000)
            stdlib2 = stdlib.replace("__heap_ptr = 0x20000;",
                                     f"__heap_ptr = {hex(heap_base)};")
            full = tsrc + "\n" + stdlib2
        bytecode, data = Compiler().compile(full)
    except Exception as e:
        r.stage = "compile_error"
        r.detail = f"{type(e).__name__}: {str(e)[:300]}"
        return r
    # -- stage 3: c4vm run --
    code = _decode(bytecode)
    argc_at = _compute_argc(code)
    try:
        vm = c4vm.C4VM(code, data, stdin=stdin)
        vm.run(max_cycles=timeout_cycles, syscall_argc=argc_at)
    except Exception as e:
        r.stage = "vm_error"
        r.detail = f"{type(e).__name__}: {str(e)[:200]}"
        r.port_stdout = vm.stdout.decode("latin-1", "replace") if 'vm' in dir() else ""
        return r
    if not vm.halted:
        r.stage = "vm_error"
        r.detail = f"did not halt within {timeout_cycles:,} cycles (cyc={vm.cycle:,})"
        r.port_stdout = bytes(vm.stdout).decode("latin-1", "replace")
        return r
    r.port_exit = vm.exit_code & 0xFF
    r.port_stdout = bytes(vm.stdout).decode("latin-1", "replace")
    r.stage = "_ran"
    return r


def run_case(name, category, src, expect_mod256, stdin=b"", sizeof_gap=False,
             timeout_cycles=40_000_000) -> Result:
    gexit, gout, gerr = run_gcc(src, stdin)
    if gexit is None:
        r = Result(name, category, stage="gcc_fail", detail=gerr)
        return r
    r = run_port(src, stdin, timeout_cycles=timeout_cycles)
    r.name = name
    r.category = category
    r.gcc_exit = gexit
    r.gcc_stdout = gout
    if r.stage in ("transpile_error", "compile_error", "vm_error"):
        return r
    # r.stage == "_ran": compare observable
    exit_match = (r.port_exit == gexit)
    out_match = (r.port_stdout == gout)
    if sizeof_gap and not exit_match:
        # legitimate c4(word=8) vs x86(int=4) divergence
        r.stage = "PASS"
        r.bug_class = "c4_x86_sizeof_gap"
        r.detail = f"sizeof gap: gcc(m32)={gexit} port(word8)={r.port_exit} (EXPECTED)"
        return r
    if exit_match and out_match:
        r.stage = "PASS"
        return r
    r.stage = "MISMATCH"
    parts = []
    if not exit_match:
        parts.append(f"exit gcc={gexit} port={r.port_exit}")
    if not out_match:
        parts.append(f"stdout gcc={gout!r} port={r.port_stdout!r}")
    r.detail = "; ".join(parts)
    return r
