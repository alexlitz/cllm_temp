#!/usr/bin/env python3
"""verify_drawcol_onvm.py -- AUTHORITATIVE on-VM correctness oracle for DRAWCOL /
DRAWSPANF (the gameplay texture-mapped fill render-macros).

Runs the REAL compiled ``R_DrawColumn`` / ``R_DrawSpan`` inner fill loops on the
32-bit reference VM ``c4vm32.py`` and byte-for-byte ``cmp``s the framebuffer they
write against the native :func:`c4_min.doom_drawcol.draw_column` /
:func:`c4_min.doom_drawcol.draw_span` fused ops over the SAME pre-poisoned
memory.  This is the "the C loop AS IT RUNS ON THE VM == the native op"
equivalence proof (the analog of ``verify_drawspan_onvm.py`` for the title).

Method (self-contained; no full Doom image needed):
  1. compile the isolated ``R_DrawColumn`` / ``R_DrawSpan`` C loops (the exact
     ported ``doom_run.c`` bodies, driven by globals so no LUTs are needed) with
     the c4_release ``src.compiler`` -> real bytecode,
  2. for each battery case: PRE-POISON the whole destination footprint with a
     sentinel (so a 'wrote nothing' bug can NEVER false-pass), seed a known
     texture pattern + a NON-identity colormap, and set the driver globals,
  3. run the compiled fn on a fresh ``c4vm32`` via an appended trampoline,
  4. run the native ``draw_column`` / ``draw_span`` over an identically
     seeded/poisoned ``bytearray``,
  5. ``cmp`` the destination footprints -- must be byte-identical.

``doom_run.c`` / ``c4vm32.py`` are NOT modified.

Run:  PYTHONPATH=<c4_release> python -m c4_min.verify_drawcol_onvm
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_C4_RELEASE = os.path.dirname(_HERE)                       # .../c4_release
_C4VM32_PATH = "/home/alexlitz/Documents/misc/c4_doom/id_port/c4vm32.py"

if _C4_RELEASE not in sys.path:
    sys.path.insert(0, _C4_RELEASE)

from c4_min import doom_drawcol as DC   # noqa: E402


def _load_c4vm32():
    """Load ``id_port/c4vm32.py`` (READ-ONLY).  Returns the module or None."""
    if not os.path.exists(_C4VM32_PATH):
        return None
    spec = importlib.util.spec_from_file_location("_c4vm32_ro_drawcol", _C4VM32_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


# The EXACT ported R_DrawColumn fill loop (doom_run.c:28254), driven entirely by
# global operands so no ylookup/columnofs LUTs are needed.  ``stride`` is baked as
# a literal so the compiled ``dest += <stride>`` matches the on-VM run for THAT
# stride.  The colormap index uses the same ``& 255`` mask the port emits.
_COL_C = """
int g_dest, g_source, g_colormap, g_count, g_frac, g_fracstep;
int drawcol() {{
    int __do_once0;
    int count; char *dest; char *cmap; char *src; int frac; int fracstep;
    count = g_count;
    if (count < 0) return 0;
    dest = (char *)g_dest;
    cmap = (char *)g_colormap;
    src  = (char *)g_source;
    frac = g_frac;
    fracstep = g_fracstep;
    __do_once0 = 1;
    while (__do_once0 || (count--)) {{ __do_once0 = 0;
        dest[0] = cmap[src[(frac>>(16))&127] & 255];
        dest = (dest) + (({stride}));
        frac = (frac) + (fracstep);
    }}
    return 0;
}}
int main() {{ return 0; }}
"""

# The EXACT ported R_DrawSpan fill loop (doom_run.c:28503), driven by globals.
_SPAN_C = """
int g_dest, g_source, g_colormap, g_count, g_xfrac, g_yfrac, g_xstep, g_ystep;
int drawspanf() {
    int __do_once0;
    int xfrac; int yfrac; char *dest; char *cmap; char *src; int count; int spot;
    xfrac = g_xfrac;
    yfrac = g_yfrac;
    dest = (char *)g_dest;
    cmap = (char *)g_colormap;
    src  = (char *)g_source;
    count = g_count;
    __do_once0 = 1;
    while (__do_once0 || (count--)) { __do_once0 = 0;
        spot = ((yfrac>>(16-6))&(63*64)) + ((xfrac>>16)&63);
        *dest++ = cmap[src[spot] & 255];
        xfrac = (xfrac) + (g_xstep);
        yfrac = (yfrac) + (g_ystep);
    }
    return 0;
}
int main() { return 0; }
"""


def _compile(src: str, entry: str):
    """Compile a driver snippet; return ``(code_tuples, data, symbols)``."""
    from src.compiler import Compiler
    c = Compiler()
    code, data = c.compile(src)
    tuples = [(w & 0xFF, (w >> 8) & 0xFFFFFFFF) for w in code]
    return tuples, data, c.symbols


def _run_col_on_vm(c4vm, code_t, data, syms, dest, source, colormap,
                   count, frac, fracstep, stride, sentinel=0xEE):
    MASK = c4vm.MASK
    st = c4vm.sx(stride)
    n = c4vm.sx(count)
    iters = n + 1 if n >= 0 else 0
    full = list(code_t)
    ti = len(full)
    entry = syms["drawcol"].value
    full += [(c4vm.JSR, entry), (c4vm.EXIT, 0)]
    vm = c4vm.C4VM32(full, data, stdin=b"")
    vm.code = full
    vm.ax = 0
    vm.sp = c4vm.STACK_TOP
    vm.bp = c4vm.STACK_TOP
    vm.pc = ti * 8
    vm.cycle = 0
    vm.halted = False
    # seed operand globals
    for name, val in (("g_dest", dest), ("g_source", source),
                      ("g_colormap", colormap), ("g_count", count),
                      ("g_frac", frac), ("g_fracstep", fracstep)):
        a = syms[name].value
        vm.mem[a:a + 4] = (val & MASK).to_bytes(4, "little")
    # texture column + non-identity colormap + poison
    for k in range(128):
        vm.mem[(source + k) & MASK] = (k * 13 + 5) & 0xFF
    for k in range(256):
        vm.mem[(colormap + k) & MASK] = (k * 7 + 13) & 0xFF
    for k in range(iters):
        vm.mem[(dest + k * st) & MASK] = sentinel
    # ~60 decoded cycles per wall pixel (LEA/PSH-heavy body) + prologue slack.
    vm.run(max_cycles=64 * max(0, iters) + 16000)
    return bytes(vm.mem[(dest + k * st) & MASK] for k in range(iters))


def _run_span_on_vm(c4vm, code_t, data, syms, dest, source, colormap,
                    count, xfrac, yfrac, xstep, ystep, sentinel=0xEE):
    MASK = c4vm.MASK
    n = c4vm.sx(count)
    iters = n + 1 if n >= 0 else 0
    full = list(code_t)
    ti = len(full)
    entry = syms["drawspanf"].value
    full += [(c4vm.JSR, entry), (c4vm.EXIT, 0)]
    vm = c4vm.C4VM32(full, data, stdin=b"")
    vm.code = full
    vm.ax = 0
    vm.sp = c4vm.STACK_TOP
    vm.bp = c4vm.STACK_TOP
    vm.pc = ti * 8
    vm.cycle = 0
    vm.halted = False
    for name, val in (("g_dest", dest), ("g_source", source),
                      ("g_colormap", colormap), ("g_count", count),
                      ("g_xfrac", xfrac), ("g_yfrac", yfrac),
                      ("g_xstep", xstep), ("g_ystep", ystep)):
        a = syms[name].value
        vm.mem[a:a + 4] = (val & MASK).to_bytes(4, "little")
    for k in range(4096):
        vm.mem[(source + k) & MASK] = (k * 17 + 3) & 0xFF
    for k in range(256):
        vm.mem[(colormap + k) & MASK] = (k * 7 + 13) & 0xFF
    for k in range(iters):
        vm.mem[(dest + k) & MASK] = sentinel
    # ~72 decoded cycles per floor pixel (the 2D-uv spot fold is heavier) + slack.
    vm.run(max_cycles=80 * max(0, iters) + 16000)
    return bytes(vm.mem[(dest + k) & MASK] for k in range(iters))


def _ref_col_bytes(c4vm, dest, source, colormap, count, frac, fracstep, stride, sentinel=0xEE):
    MASK = c4vm.MASK
    st = c4vm.sx(stride)
    n = c4vm.sx(count)
    iters = n + 1 if n >= 0 else 0
    top = max(source + 128, colormap + 256, dest + max(0, iters - 1) * max(st, 1)) + 64
    mem = bytearray(top)
    for k in range(128):
        mem[(source + k) & MASK] = (k * 13 + 5) & 0xFF
    for k in range(256):
        mem[(colormap + k) & MASK] = (k * 7 + 13) & 0xFF
    for k in range(iters):
        mem[(dest + k * st) & MASK] = sentinel
    DC.draw_column(mem, dest, source, colormap, count, frac, fracstep, stride)
    return bytes(mem[(dest + k * st) & MASK] for k in range(iters))


def _ref_span_bytes(c4vm, dest, source, colormap, count, xfrac, yfrac, xstep, ystep, sentinel=0xEE):
    MASK = c4vm.MASK
    n = c4vm.sx(count)
    iters = n + 1 if n >= 0 else 0
    top = max(source + 4096, colormap + 256, dest + max(0, iters)) + 64
    mem = bytearray(top)
    for k in range(4096):
        mem[(source + k) & MASK] = (k * 17 + 3) & 0xFF
    for k in range(256):
        mem[(colormap + k) & MASK] = (k * 7 + 13) & 0xFF
    for k in range(iters):
        mem[(dest + k) & MASK] = sentinel
    DC.draw_span(mem, dest, source, colormap, count, xfrac, yfrac, xstep, ystep)
    return bytes(mem[(dest + k) & MASK] for k in range(iters))


def verify() -> dict:
    """Run the on-VM ``cmp`` for every battery case (both ops)."""
    c4vm = _load_c4vm32()
    if c4vm is None:
        return {"checked": 0, "fails": 0, "fail_detail": [], "vm_available": False}

    fails = 0
    checked = 0
    fail_detail = []

    # ---- R_DrawColumn: compile once per distinct stride ----
    col_by_stride = {}
    for (dest, source, colormap, count, frac, fracstep, stride) in DC.battery_cases_col():
        st = c4vm.sx(stride)
        if st <= 0:
            continue
        if st not in col_by_stride:
            col_by_stride[st] = _compile(_COL_C.format(stride=st), "drawcol")
        code_t, data, syms = col_by_stride[st]
        got = _run_col_on_vm(c4vm, code_t, data, syms, dest, source, colormap,
                             count, frac, fracstep, stride)
        ref = _ref_col_bytes(c4vm, dest, source, colormap, count, frac, fracstep, stride)
        checked += 1
        if got != ref:
            fails += 1
            ndiff = sum(1 for a, b in zip(got, ref) if a != b)
            fail_detail.append({"op": "DRAWCOL", "count": count, "frac": frac,
                                "fracstep": fracstep, "diff_bytes": ndiff})

    # ---- R_DrawSpan ----
    span_code = _compile(_SPAN_C, "drawspanf")
    for (dest, source, colormap, count, xfrac, yfrac, xstep, ystep) in DC.battery_cases_span():
        code_t, data, syms = span_code
        got = _run_span_on_vm(c4vm, code_t, data, syms, dest, source, colormap,
                              count, xfrac, yfrac, xstep, ystep)
        ref = _ref_span_bytes(c4vm, dest, source, colormap, count, xfrac, yfrac, xstep, ystep)
        checked += 1
        if got != ref:
            fails += 1
            ndiff = sum(1 for a, b in zip(got, ref) if a != b)
            fail_detail.append({"op": "DRAWSPANF", "count": count, "xfrac": xfrac,
                                "yfrac": yfrac, "diff_bytes": ndiff})

    return {"checked": checked, "fails": fails, "fail_detail": fail_detail,
            "vm_available": True}


def main():
    print("compiling isolated R_DrawColumn / R_DrawSpan loops + running on c4vm32 ...",
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
    print("  ALL cases byte-identical: compiled R_DrawColumn/R_DrawSpan == "
          "DRAWCOL/DRAWSPANF.", flush=True)


if __name__ == "__main__":
    main()
