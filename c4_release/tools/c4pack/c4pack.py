#!/usr/bin/env python3
"""c4pack -- package a C4 program into ONE self-contained, gcc-compiled,
byte-exact standalone C binary.

Pipeline:
    C source  --(src.compiler.compile_c)-->  (bytecode, data)
                                              |
    OR: existing bytecode (--bytecode a.json / a.npz)
                                              |
                    embed bytecode + data + the c4 VM runtime
                    (tools/c4pack/c4_runtime_template.c)  into ONE .c file
                                              |
                         gcc -O2 [-static]  ->  ONE standalone executable

The runtime is a non-neural, non-torch, full-C4-ISA interpreter that is
byte-for-byte identical to the Python reference VM
(c4_min.libprog_corpus.RefVM): 32-bit words, instruction-index PC, signed
char loads, arithmetic SHR, peek-not-pop syscalls, and the c4 printf subset
(%d %u %x %c %s %%).  The full ISA is covered via the interpreter, so ANY
program src.compiler can compile packages -- this is NOT a min-param subset.

Usage:
    python -m tools.c4pack.c4pack PROGRAM.c -o PROGRAM        # C source
    python -m tools.c4pack.c4pack PROGRAM.c --emit-c out.c    # just the .c
    python tools/c4pack/c4pack.py PROGRAM.c -o PROGRAM
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).resolve().parent
_RELEASE_ROOT = _HERE.parent.parent            # <root>/c4_release
_TEMPLATE = _HERE / "c4_runtime_template.c"


def _compile_source(src_text: str) -> Tuple[List[Tuple[int, int]], bytes]:
    """C source -> ([(op, imm)], data_bytes) via the real repo compiler."""
    if str(_RELEASE_ROOT) not in sys.path:
        sys.path.insert(0, str(_RELEASE_ROOT))
    from src.compiler import compile_c            # the real C4 compiler + stdlib
    code_words, data = compile_c(src_text)
    instrs: List[Tuple[int, int]] = []
    for w in code_words:
        op = w & 0xFF
        imm = w >> 8
        if imm >= (1 << 55):                      # sign-extend negative offsets
            imm -= (1 << 56)
        instrs.append((op, imm))
    return instrs, bytes(data)


def _emit_c(instrs: List[Tuple[int, int]], data: bytes, prog_name: str) -> str:
    """Concatenate the runtime template + the embedded bytecode/data arrays."""
    template = _TEMPLATE.read_text()

    lines: List[str] = []
    lines.append(f"\n/* ==== embedded program: {prog_name} ==== */")
    lines.append(f"const int PROG_CODE_LEN = {len(instrs)};")
    lines.append("const long PROG_CODE[][2] = {")
    for op, imm in instrs:
        lines.append(f"    {{{op}L, {imm}L}},")
    if not instrs:
        lines.append("    {0L, 0L},")             # avoid empty-array UB
    lines.append("};")
    lines.append(f"const int PROG_DATA_LEN = {len(data)};")
    lines.append("const unsigned char PROG_DATA[] = {")
    row: List[str] = []
    for i, b in enumerate(data):
        row.append(str(b))
        if len(row) == 16:
            lines.append("    " + ",".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ",".join(row) + ",")
    lines.append("    0")                          # trailing 0, avoids empty array
    lines.append("};")
    return template + "\n".join(lines) + "\n"


def _gcc(c_path: Path, out_path: Path, static: bool) -> Tuple[bool, str]:
    base = ["gcc", "-O2", "-w"]
    flags = base + (["-static"] if static else [])
    cmd = flags + [str(c_path), "-o", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return True, " ".join(cmd)
    # fall back to dynamic link if -static failed (e.g. no static libc)
    if static:
        cmd2 = base + [str(c_path), "-o", str(out_path)]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode == 0:
            return True, " ".join(cmd2) + "  (dynamic fallback: -static unavailable)"
        return False, r2.stderr
    return False, r.stderr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Package a C4 program into a standalone C binary.")
    ap.add_argument("source", help="C source file (.c)")
    ap.add_argument("-o", "--output", help="output binary path")
    ap.add_argument("--emit-c", help="write the fused .c to this path (no gcc)")
    ap.add_argument("--no-static", action="store_true", help="dynamic link (default tries -static first)")
    args = ap.parse_args(argv)

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"c4pack: no such file: {src_path}", file=sys.stderr)
        return 2

    instrs, data = _compile_source(src_path.read_text())
    prog_name = src_path.name
    c_text = _emit_c(instrs, data, prog_name)

    if args.emit_c:
        Path(args.emit_c).write_text(c_text)
        print(f"c4pack: wrote fused C -> {args.emit_c} "
              f"({len(instrs)} instrs, {len(data)} data bytes)")
        if not args.output:
            return 0

    out_path = Path(args.output) if args.output else src_path.with_suffix("")
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tf:
        tf.write(c_text)
        tmp_c = Path(tf.name)
    try:
        ok, info = _gcc(tmp_c, out_path, static=not args.no_static)
    finally:
        tmp_c.unlink(missing_ok=True)

    if not ok:
        print(f"c4pack: gcc failed:\n{info}", file=sys.stderr)
        return 1
    size = out_path.stat().st_size
    print(f"c4pack: {prog_name} -> {out_path}  "
          f"({len(instrs)} instrs, {len(data)} data bytes, binary {size} bytes)")
    print(f"        {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
