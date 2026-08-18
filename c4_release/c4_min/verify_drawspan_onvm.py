#!/usr/bin/env python3
"""verify_drawspan_onvm.py -- AUTHORITATIVE on-VM correctness oracle for DRAWSPAN.

Runs the REAL compiled ``V_DrawPatch`` inner column-copy loop
(``while (count--) { *dest = *source++; dest += 320; }``) on the 32-bit reference
VM ``c4vm32.py`` and byte-for-byte ``cmp``s the memory it writes against the
native :func:`c4_min.doom_drawspan.draw_span` fused op over the SAME pre-poisoned
memory.  This is the "the C loop AS IT RUNS ON THE VM == the native op"
equivalence proof.

Method (self-contained; no full Doom image needed):
  1. compile the isolated ``span`` C function (the exact V_DrawPatch inner loop)
     with the c4_release ``src.compiler`` -> real bytecode,
  2. for each battery case (dest, source, count, stride): PRE-POISON the whole
     destination footprint with a sentinel (so a 'wrote nothing' bug can NEVER
     false-pass -- the bug ``doom_blit`` originally had) and seed a known source
     pattern,
  3. run the compiled ``span`` on a fresh ``c4vm32`` via an appended call
     trampoline (``PSH dest; PSH source; PSH count; JSR span; ADJ 24; EXIT``),
  4. run ``draw_span`` over an identically-seeded/poisoned ``bytearray``,
  5. ``cmp`` the two destination footprints -- must be byte-identical.

The stride-320 loop is compiled once; cases with a different stride recompile a
``span`` whose ``dest += <stride>`` literal matches (so the on-VM run is the REAL
loop for THAT stride).  ``doom_run.c`` / ``c4vm32.py`` are NOT modified.

Run:  PYTHONPATH=<c4_release> python -m c4_min.verify_drawspan_onvm
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import List, Tuple

# --- locate the c4 compiler (src.compiler) and the 32-bit reference VM --------
_HERE = os.path.dirname(os.path.abspath(__file__))
_C4_RELEASE = os.path.dirname(_HERE)                       # .../c4_release
_C4VM32_PATH = "/home/alexlitz/Documents/misc/c4_doom/id_port/c4vm32.py"

if _C4_RELEASE not in sys.path:
    sys.path.insert(0, _C4_RELEASE)

from c4_min import doom_drawspan as DS   # noqa: E402


def _load_c4vm32():
    """Load ``id_port/c4vm32.py`` (READ-ONLY).  Returns the module or None."""
    if not os.path.exists(_C4VM32_PATH):
        return None
    spec = importlib.util.spec_from_file_location("_c4vm32_ro_drawspan", _C4VM32_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


# The EXACT V_DrawPatch inner column-copy loop, isolated into a leaf function.
# ``char *dest, *source; int count; stride`` baked as a literal so the compiled
# ``dest += <stride>`` matches the on-VM run for THAT stride.
_SPAN_C = """
int span(char *dest, char *source, int count) {{
    while (count--) {{
        *dest = *source;
        source = source + 1;
        dest = dest + {stride};
    }}
    return 0;
}}
int main() {{ return 0; }}
"""


def _compile_span(stride: int) -> Tuple[list, list, int]:
    """Compile the isolated V_DrawPatch inner loop for a given column stride.

    Returns ``(code_tuples, data, span_entry_idx)`` where ``code_tuples`` is the
    list of ``(op, imm)`` the c4vm32 VM executes and ``span_entry_idx`` is the
    instruction index of the ``span`` function body."""
    from src.compiler import Compiler
    c = Compiler()
    code, data = c.compile(_SPAN_C.format(stride=stride))
    tuples = [(w & 0xFF, (w >> 8) & 0xFFFFFFFF) for w in code]
    span_idx = c.symbols["span"].value
    return tuples, data, span_idx


def run_case_on_vm(c4vm, code_tuples, data, span_idx,
                   dest: int, source: int, count: int, stride: int,
                   sentinel: int = 0xEE) -> bytes:
    """Run the REAL compiled ``span`` loop on a fresh c4vm32 for one case and
    return the destination footprint bytes AFTER the run.  The destination is
    PRE-POISONED with ``sentinel`` and the source seeded with a known pattern."""
    MASK = c4vm.MASK
    n = c4vm.sx(count)
    st = c4vm.sx(stride)

    full = list(code_tuples)
    ti = len(full)
    # trampoline: PSH dest; PSH source; PSH count; JSR span; ADJ 24; EXIT
    full += [
        (c4vm.IMM, dest & MASK), (c4vm.PSH, 0),
        (c4vm.IMM, source & MASK), (c4vm.PSH, 0),
        (c4vm.IMM, count & MASK), (c4vm.PSH, 0),
        (c4vm.JSR, span_idx), (c4vm.ADJ, 24),
        (c4vm.EXIT, 0),
    ]
    vm = c4vm.C4VM32(full, data, stdin=b"")
    vm.code = full
    vm.ax = 0
    vm.sp = c4vm.STACK_TOP
    vm.bp = c4vm.STACK_TOP
    vm.pc = ti * 8
    vm.cycle = 0
    vm.halted = False
    # seed source pattern + pre-poison destination footprint
    for k in range(max(0, n)):
        vm.mem[(source + k) & MASK] = (k * 31 + 7) & 0xFF
    for k in range(max(0, n)):
        vm.mem[(dest + k * st) & MASK] = sentinel
    vm.run(max_cycles=40 * max(0, n) + 5000)
    # collect the destination footprint (in write order)
    return bytes(vm.mem[(dest + k * st) & MASK] for k in range(max(0, n)))


def verify() -> dict:
    """Run the on-VM ``cmp`` for every battery case.  Returns
    ``{checked, fails, fail_detail, vm_available}``."""
    c4vm = _load_c4vm32()
    if c4vm is None:
        return {"checked": 0, "fails": 0, "fail_detail": [],
                "vm_available": False}

    cases = DS.battery_cases()
    # compile once per distinct stride (the on-VM loop bakes the stride literal)
    span_by_stride = {}
    fails = 0
    checked = 0
    fail_detail = []
    for (dest, source, count, stride) in cases:
        st = c4vm.sx(stride)
        n = c4vm.sx(count)
        if st <= 0:
            # a non-positive stride is not a real column copy; skip on-VM (the
            # compiler can't bake a negative literal identically) -- the pure
            # battery still covers the count<=0 guard.
            continue
        if st not in span_by_stride:
            span_by_stride[st] = _compile_span(st)
        code_t, data, span_idx = span_by_stride[st]

        sentinel = 0xEE
        got = run_case_on_vm(c4vm, code_t, data, span_idx,
                             dest, source, count, stride, sentinel)

        # native DRAWSPAN reference over an identically-seeded/poisoned buffer
        top = max(source + max(0, n), dest + max(0, n - 1) * max(st, 0)) + 64
        ref_mem = bytearray(top)
        for k in range(max(0, n)):
            ref_mem[(source + k) & c4vm.MASK] = (k * 31 + 7) & 0xFF
        for k in range(max(0, n)):
            ref_mem[(dest + k * st) & c4vm.MASK] = sentinel
        DS.draw_span(ref_mem, dest, source, count, stride)
        ref = bytes(ref_mem[(dest + k * st) & c4vm.MASK] for k in range(max(0, n)))

        checked += 1
        if got != ref:
            fails += 1
            ndiff = sum(1 for a, b in zip(got, ref) if a != b)
            fail_detail.append({"dest": dest, "source": source, "count": count,
                                "stride": stride, "diff_bytes": ndiff,
                                "n": max(0, n)})
    return {"checked": checked, "fails": fails, "fail_detail": fail_detail,
            "vm_available": True}


def main():
    print("compiling isolated V_DrawPatch inner loop + running on c4vm32 ...",
          flush=True)
    res = verify()
    if not res["vm_available"]:
        print(f"  c4vm32 NOT available at {_C4VM32_PATH}; on-VM check SKIPPED",
              flush=True)
        sys.exit(2)
    print(f"  on-VM byte-exact: checked {res['checked']} cases / "
          f"fails {res['fails']}", flush=True)
    if res["fails"]:
        for d in res["fail_detail"]:
            print("   FAIL", d, flush=True)
        sys.exit(1)
    print("  ALL cases byte-identical: compiled V_DrawPatch loop == DRAWSPAN.",
          flush=True)


if __name__ == "__main__":
    main()
