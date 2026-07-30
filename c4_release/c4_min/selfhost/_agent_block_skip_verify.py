#!/usr/bin/env python3
"""_agent_block_skip_verify.py — VERIFY + MEASURE the all-C VM per-op BLOCK-SKIP
(#738 port into onnx_runtime_nibble_allc.c).

Builds an all-C binary per utility (echo / yes / cat), runs it BOTH ways
(``--allc`` full-242-block forward  vs  ``--allc --block-skip`` per-op live-set)
and:
  * asserts the STDOUT is BYTE-EXACT between the two modes (the block-skipped
    forward is decode-identical to the full forward),
  * reports the wall time of each mode (block-count-driven speedup),
  * confirms ONE static binary (ldd: not a dynamic executable).

The per-op live-set table lives in allc_gen.h (emitted by _agent_allc_gen.py's
build_live_masks, resolved from step_block_skip.build_live_index).  The C forward
run_full_skip runs ONLY the decoded op's live blocks, passing the residual straight
through the skipped ones — the identical semantics of
step_block_skip.StepBlockSkipRunner (which the torch harness
_step_block_skip_verify proves byte-exact over the full opcode corpus).

Tooling only; no model-build side effects; golden untouched.

Usage:
  python -m c4_min.selfhost._agent_block_skip_verify \
      --binp /tmp/fullisa_sparse/blockstack.nblbin --out-dir /tmp/bs_agent
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "_agent_allc_build.sh")


def gen_header(mode, out_h, io_mode="literal", text="hello\n", stdin_text=""):
    args = [sys.executable, "-m", "c4_min.selfhost._agent_allc_gen",
            "--mode", mode, "--io-mode", io_mode, "--out", out_h, "--text", text]
    if stdin_text:
        args += ["--stdin", stdin_text]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.path.dirname(C4MIN))
    subprocess.run(args, check=True, env=env)


def build(out_h, exe, binp):
    subprocess.run(["bash", BUILD, out_h, exe, binp], check=True)


def run(exe, block_skip, stdin=b"", threads=8, timeout=1800):
    argv = [exe, "--allc"] + (["--block-skip"] if block_skip else [])
    env = dict(os.environ, INCR_THREADS=str(threads))
    t0 = time.time()
    p = subprocess.run(argv, input=stdin, capture_output=True, env=env,
                       timeout=timeout)
    return p.stdout, time.time() - t0


def is_static(exe):
    p = subprocess.run(["ldd", exe], capture_output=True, text=True)
    return "not a dynamic executable" in (p.stdout + p.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binp", default="/tmp/fullisa_sparse/blockstack.nblbin")
    ap.add_argument("--out-dir", default="/tmp/bs_agent")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-cat", action="store_true",
                    help="skip the long cat run (89 steps, minutes)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    utils = [
        ("echo", dict(mode="echo", io_mode="literal", text="hello\n"), b"", b"hello\n"),
        ("yes",  dict(mode="yes",  io_mode="literal", text="y\n"),     b"", b"y\n"),
    ]
    if not args.skip_cat:
        utils.append(("cat", dict(mode="cat", io_mode="strict",
                                  stdin_text="meow\n"), b"meow\n", b"meow\n"))

    all_ok = True
    for name, gargs, stdin, expected in utils:
        out_h = os.path.join(args.out_dir, f"allc_gen_{name}.h")
        exe = os.path.join(args.out_dir, f"allc_{name}")
        print(f"\n=== {name} ===", flush=True)
        gen_header(out_h=out_h, **gargs)
        build(out_h, exe, args.binp)
        static = is_static(exe)
        base, tb = run(exe, False, stdin, args.threads)
        skip, ts = run(exe, True, stdin, args.threads)
        exact = (base == skip == expected)
        all_ok = all_ok and exact and static
        print(f"  static-binary : {static}  ({os.path.getsize(exe):,} bytes)")
        print(f"  base  out={base!r}  {tb:.2f}s")
        print(f"  skip  out={skip!r}  {ts:.2f}s")
        print(f"  BYTE-EXACT (base==skip==expected): {exact}")
        if tb > 0 and ts > 0:
            print(f"  wall speedup (base/skip): {tb/ts:.2f}x")
    print(f"\n{'ALL OK' if all_ok else 'FAILURE'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
