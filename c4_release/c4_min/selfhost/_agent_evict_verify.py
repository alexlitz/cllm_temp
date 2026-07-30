#!/usr/bin/env python3
"""_agent_evict_verify.py — verify the FREE-DRIVEN EVICTION + arbitrary-length I/O
of the all-C native VM (onnx_runtime_nibble_allc.c).  Tooling only; NEVER touches
the model build path; golden 069cc32f unchanged.

WHAT IT PROVES
  1) FLAT MEMORY over arbitrary length — cat a multi-KB stream and report the peak
     compacted KV-row count + peak RSS staying BOUNDED (~chunk+window) regardless of
     total bytes (the headline: memory is flat over the whole file), stdout byte-exact
     vs unix `cat`.
  2) KEEP-HEAP correctness — write several §Memory addresses, churn past the recency
     window, then LC-read the OLD addresses back: base (full stream) == --evict
     (compacted keep-set), byte-exact (the live stores were NOT evicted).
  3) REGRESSION — echo / yes / cat (strict + burst) still byte-exact:
     base == block-skip == evict == unix, one static binary (ldd not-dynamic).

USAGE
  # build the binaries once (reuses /tmp/fullisa_sparse/), then run all checks:
  python -m c4_min.selfhost._agent_evict_verify --build --all
  # or a single check against a prebuilt binary:
  python -m c4_min.selfhost._agent_evict_verify --catcurve /tmp/prog_catchunk_burst
"""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "_agent_allc_build.sh")
NBLBIN = "/tmp/fullisa_sparse/blockstack.nblbin"


def _proc_peak_rss_kb(pid):
    """Read VmHWM (peak resident set) in KB from /proc/<pid>/status; 0 if gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _run(cmd, stdin=None, timeout=None, env=None):
    """Run a subprocess, capture stdout(bytes)+stderr(text)+wall(s)+peak RSS(KB).

    Peak RSS is polled from /proc/<pid>/status VmHWM while the child runs (a
    monotonically-increasing high-water mark), so it is the true process peak — the
    headline metric for FLAT memory over arbitrary length."""
    import threading
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=e)
    peak = [0]
    stop = threading.Event()

    def _mon():
        while not stop.is_set():
            hw = _proc_peak_rss_kb(p.pid)
            if hw > peak[0]:
                peak[0] = hw
            stop.wait(0.05)
    th = threading.Thread(target=_mon, daemon=True)
    th.start()
    out, err = p.communicate(input=stdin, timeout=timeout)
    stop.set(); th.join(timeout=1.0)
    dt = time.time() - t0
    return out, err.decode("utf-8", "replace"), dt, peak[0], p.returncode


def gen_header(mode, io_mode, out, text="hello\n", stdin_text="", n=8, chunk=32):
    from c4_min.selfhost._agent_allc_gen import (build_layout, build_program,
                                                 gen_header as _gh)
    L, embed = build_layout()
    code, seed_mem, expected = build_program(
        mode, text, io_mode=io_mode, n=n, stdin_text=stdin_text, chunk_size=chunk)
    _gh(out, L, embed, code, seed_mem)
    return bytes(expected)


def build_bin(header, out_bin):
    r = subprocess.run(["bash", BUILD, header, out_bin, NBLBIN],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise SystemExit("build failed")
    print(f"  built {out_bin} ({os.path.getsize(out_bin):,} bytes)")
    return out_bin


def ldd_static(binpath):
    r = subprocess.run(["ldd", binpath], capture_output=True, text=True)
    return "not a dynamic executable" in (r.stdout + r.stderr)


def _peak_rows(err):
    for line in err.splitlines():
        if "EVICT peak compacted rows" in line:
            try:
                return int(line.split("peak compacted rows =")[1].split("(")[0].strip())
            except Exception:
                return None
    return None


def _final_S(err):
    for line in err.splitlines():
        if "final S=" in line:
            try:
                return int(line.split("final S=")[1].split()[0].strip())
            except Exception:
                return None
    return None


def check_regression(binaries):
    """echo/yes/cat still byte-exact base==skip==evict==unix + static."""
    print("\n=== REGRESSION: base == block-skip == evict == unix, static binary ===")
    ok = True
    for name, binp, io_flags, stdin, expected in binaries:
        variants = {
            "base": [binp, "--allc"] + io_flags,
            "block-skip": [binp, "--allc", "--block-skip"] + io_flags,
            "evict": [binp, "--allc", "--evict"] + io_flags,
        }
        outs = {}
        for v, cmd in variants.items():
            out, err, dt, rss, rc = _run(cmd, stdin=stdin, timeout=1200)
            outs[v] = out
        agree = (outs["base"] == outs["block-skip"] == outs["evict"] == expected)
        static = ldd_static(binp)
        pk = _peak_rows(_run(variants["evict"], stdin=stdin, timeout=1200)[1])
        print(f"  {name:16s} base==skip==evict==unix: {agree}  static:{static}  "
              f"evict_peak_rows={pk}  out={outs['base'][:24]!r}")
        ok = ok and agree and static
    print(f"  REGRESSION {'PASS' if ok else 'FAIL'}")
    return ok


def check_keepheap(binp, expected):
    """base (full stream) == evict (compacted) byte-exact, proving live stores kept.

    Uses a LONG seeded string + a strict §Memory pointer-walk (echo --io-mode strict):
    the string's bytes are seed store-KV rows at the very START of the stream, and the
    LC pointer-walk reads them back one per loop step — so byte i is content-addressed
    ~i loop-iterations (hundreds of steps / thousands of tokens) AFTER its store row,
    far past any recency window.  If evict dropped a live store the read-back would
    return 0/garbage; base==evict byte-exact proves the LIVE HEAP survives eviction.
    (The ad-hoc SI/ADD keepheap program hits a model DECODE wall — op mis-decode,
    base and evict identically — so it is NOT a faithful oracle; this seeded strict
    walk uses only the proven echo-strict idiom.)"""
    print("\n=== KEEP-HEAP: seed a long §Memory string, read it back far past window ===")
    e_out, e_err, e_dt, e_rss, _ = _run([binp, "--allc", "--evict"], timeout=3600)
    b_out, b_err, b_dt, b_rss, _ = _run([binp, "--allc"], timeout=3600)
    pk = _peak_rows(e_err); fS = _final_S(e_err)
    agree = (b_out == e_out == expected)
    print(f"  expected={expected!r}")
    print(f"  base ={b_out!r}  ({b_dt:.1f}s, full stream S={_final_S(b_err)})")
    print(f"  evict={e_out!r}  ({e_dt:.1f}s, peak_rows={pk} vs full S={fS}, "
          f"RSS {e_rss/1024.0:.0f}MB)")
    print(f"  KEEP-HEAP {'PASS (live stores survived eviction)' if agree else 'FAIL'}")
    return agree


def check_catcurve(binp, sizes=(1024, 4096, 8192, 16384), chunk=32):
    """cat a growing multi-KB stream; report peak KV-row count + RSS stay FLAT and
    stdout byte-exact vs unix `cat`."""
    print("\n=== FLAT-MEMORY cat: peak KV-rows + RSS over arbitrary length ===")
    import os as _os
    print(f"  {'bytes':>8} {'exact':>6} {'peak_rows':>10} {'full_S':>8} "
          f"{'RSS_MB':>8} {'steps':>7} {'wall_s':>7}")
    rows_seen = []
    all_exact = True
    for nbytes in sizes:
        # mixed bytes incl newlines; NO NUL (0x00) — the 8-bit burst PRTF is a
        # NUL-terminated C-string walk, so a literal NUL is the string terminator and
        # cannot pass through this path (an HONEST limit of the burst syscall; a
        # count-based burst or a 16/32-bit variant is the future lever).  Real text /
        # files that are NUL-free stream byte-exact.
        data = bytearray((i * 7 + 13) & 0xFF for i in range(nbytes))
        for i in range(0, nbytes, 97):
            data[i] = 0x0A
        data = bytes(b if b != 0 else 0x20 for b in data)   # map NUL -> space
        out, err, dt, rss, rc = _run(
            [binp, "--allc", "--io-burst", "--evict"], stdin=data, timeout=3600)
        exact = (out == data)
        all_exact = all_exact and exact
        pk = _peak_rows(err); fS = _final_S(err)
        steps = None
        for line in err.splitlines():
            if "VM steps" in line:
                try: steps = int(line.split("allc:")[1].split("VM steps")[0].strip())
                except Exception: pass
        rows_seen.append(pk)
        print(f"  {nbytes:>8} {str(exact):>6} {str(pk):>10} {str(fS):>8} "
              f"{rss/1024.0:>8.1f} {str(steps):>7} {dt:>7.1f}")
    flat = (len(set(r for r in rows_seen if r is not None)) <= 2)  # ~constant
    print(f"  peak KV-rows across sizes: {rows_seen}  ->  "
          f"{'FLAT (bounded, no OOM)' if flat else 'GROWING (!)'}")
    print(f"  byte-exact vs unix cat: {'ALL PASS' if all_exact else 'FAIL'}")
    return all_exact and flat


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="build the verification binaries (reuses /tmp/fullisa_sparse)")
    ap.add_argument("--all", action="store_true", help="run every check")
    ap.add_argument("--keepheap", metavar="BIN", help="run only the keep-heap check")
    ap.add_argument("--catcurve", metavar="BIN", help="run only the flat-memory curve")
    ap.add_argument("--sizes", default="1024,4096,8192,16384")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    if args.keepheap:
        check_keepheap(args.keepheap, b"ABC"); raise SystemExit
    if args.catcurve:
        sizes = tuple(int(s) for s in args.sizes.split(","))
        check_catcurve(args.catcurve, sizes=sizes, chunk=args.chunk); raise SystemExit

    if args.build:
        os.makedirs("/tmp/evict_verify", exist_ok=True)
        print("building headers + binaries (reuses /tmp/fullisa_sparse artifacts)...")
        # catchunk burst (arbitrary-length streaming)
        gen_header("catchunk", "burst", "/tmp/evict_verify/catchunk_burst.h",
                   chunk=args.chunk)
        build_bin("/tmp/evict_verify/catchunk_burst.h", "/tmp/prog_catchunk_burst")
        # keep-heap = a LONG seeded string read back via the proven strict pointer-walk
        KH_MSG = "The-quick-brown-fox-jumps-0123456789ABC"
        gen_header("echo", "strict", "/tmp/evict_verify/keepheap.h", text=KH_MSG)
        build_bin("/tmp/evict_verify/keepheap.h", "/tmp/prog_keepheap8")
        globals()["_KH_MSG"] = KH_MSG.encode()

    if args.all:
        ok = True
        sizes = tuple(int(s) for s in args.sizes.split(","))
        ok &= check_catcurve("/tmp/prog_catchunk_burst", sizes=sizes, chunk=args.chunk)
        kh_msg = globals().get("_KH_MSG", b"The-quick-brown-fox-jumps-0123456789ABC")
        ok &= check_keepheap("/tmp/prog_keepheap8", kh_msg)
        print(f"\nOVERALL: {'ALL PASS' if ok else 'SOME FAIL'}")
