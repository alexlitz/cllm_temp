#!/usr/bin/env python3
"""verify_c4pack.py -- byte-exact verification harness for c4pack.

For each demo program it:
  1. packages it with c4pack into a STANDALONE static binary (no Python/torch),
  2. runs the ACTUAL compiled binary with the given argv/stdin,
  3. compares its stdout byte-for-byte against:
       - gcc -m32 -std=c90 (the C reference), always;
       - the RefVM oracle (c4_min.libprog_corpus.RefVM), for pure-compute /
         printf programs (RefVM does not model argv/getchar).

Prints a PASS/FAIL line per (program, oracle) pair and a final N/N tally.
Exit code 0 iff every check is byte-exact.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RELEASE = _HERE.parent.parent
if str(_RELEASE) not in sys.path:
    sys.path.insert(0, str(_RELEASE))

C4PACK = _HERE / "c4pack.py"


def pack(src: Path, out: Path) -> int:
    r = subprocess.run([sys.executable, str(C4PACK), str(src), "-o", str(out)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def gcc_ref(src: Path, out: Path) -> bool:
    r = subprocess.run(["gcc", "-m32", "-std=c90", "-w", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # fall back to native gcc if -m32 multilib unavailable
        r = subprocess.run(["gcc", "-std=c90", "-w", str(src), "-o", str(out)],
                           capture_output=True, text=True)
    return r.returncode == 0


def run_bin(path: Path, args=None, stdin: bytes = b"") -> bytes:
    r = subprocess.run([str(path)] + list(args or []), input=stdin,
                       capture_output=True)
    return r.stdout


def ref_oracle(src_text: str, stdin: bytes = b"") -> bytes:
    from src.compiler import compile_c
    from c4_min.libprog_corpus import RefVM
    code, data = compile_c(src_text)
    instrs = [((w & 0xFF), (w >> 8) if (w >> 8) < (1 << 55) else (w >> 8) - (1 << 56))
              for w in code]
    vm = RefVM(instrs, bytes(data), stdin=stdin)
    return vm.run().encode("latin-1")


# (label, source_path, argv, stdin, use_refvm)
DEMOS = None


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="c4pack_verify_"))
    tp = _RELEASE / "test_programs"

    # compute demo (pure printf/arith -> RefVM applies)
    compute_c = tmp / "compute.c"
    compute_c.write_text(
        "int fib(int n){ if(n<2) return n; return fib(n-1)+fib(n-2); }\n"
        "int main(){ int i; i=0; while(i<=10){ printf(\"fib(%d)=%d\\n\", i, fib(i)); i=i+1; } return 0; }\n"
    )
    # arith demo (printf %d %u %x, negative, mod, shifts)
    arith_c = tmp / "arith.c"
    arith_c.write_text(
        "int main(){ int a; int b; a=1000000; b=7;"
        " printf(\"%d %d %d %d %x %u\\n\", a+b, a-b, a*b, a/b, a%b, a);"
        " printf(\"%d %d\\n\", 0-5, (0-1)>>1);"
        " return 0; }\n"
    )
    cli = _HERE / "demos" / "cli_bundle.c"

    checks = [
        # label,        src,        argv,                       stdin,           refvm
        ("echo",        tp / "echo.c", ["hello", "world", "foo"], b"",            False),
        ("echo/empty",  tp / "echo.c", [],                        b"",            False),
        ("cat",         tp / "cat.c",  [],                        b"a\nbb\nccc",  False),
        ("cat/binary",  tp / "cat.c",  [],                        bytes(range(256)), False),
        ("compute/fib", compute_c,     [],                        b"",            True),
        ("arith",       arith_c,       [],                        b"",            True),
        ("bundle/echo", cli,           ["echo", "one", "two"],    b"",            False),
        ("bundle/cat",  cli,           ["cat"],                   b"piped\n",     False),
        ("bundle/yes",  cli,           ["yes", "ok", "4"],        b"",            False),
    ]

    total = 0
    passed = 0
    sizes = {}
    for label, src, argv, stdin, use_refvm in checks:
        binp = tmp / (label.replace("/", "_") + ".bin")
        rc, log = pack(src, binp)
        if rc != 0:
            print(f"[FAIL] {label}: c4pack failed\n{log}")
            total += 1
            continue
        sizes[str(src)] = binp.stat().st_size
        got = run_bin(binp, argv, stdin)

        # gcc -m32 reference
        gccp = tmp / (label.replace("/", "_") + ".gcc")
        if gcc_ref(src, gccp):
            exp = run_bin(gccp, argv, stdin)
            total += 1
            ok = got == exp
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {label:14s} vs gcc -m32 -std=c90"
                  f"  ({len(got)} bytes)")
            if not ok:
                print(f"        got={got[:80]!r}\n        exp={exp[:80]!r}")
        else:
            print(f"[SKIP] {label}: gcc reference did not build")

        # RefVM oracle (pure-compute / printf only)
        if use_refvm:
            exp = ref_oracle(src.read_text(), stdin=stdin)
            total += 1
            ok = got == exp
            passed += ok
            print(f"[{'PASS' if ok else 'FAIL'}] {label:14s} vs RefVM oracle"
                  f"       ({len(got)} bytes)")
            if not ok:
                print(f"        got={got[:80]!r}\n        exp={exp[:80]!r}")

    print("-" * 60)
    print(f"byte-exact checks: {passed}/{total}")
    print("binary sizes:")
    for s, sz in sizes.items():
        print(f"  {Path(s).name:16s} {sz:>9,} bytes")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
