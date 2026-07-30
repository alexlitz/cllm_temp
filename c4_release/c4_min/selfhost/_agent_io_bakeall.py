#!/usr/bin/env python3
"""_agent_io_bakeall.py — bake + build + run + VERIFY + MEASURE the MODE-1 (strict
per-byte §Memory pointer-walk) and MODE-2 (burst / runtime §Memory syscall) I/O
programs (echo / yes / cat [/ eliza]) in the ALL-C native VM.

The compact full-ISA model + embed table + layout are built ONCE (the slow part);
every program then reuses that layout to emit its own allc_gen.h in-process, so the
whole batch costs ONE model build.  For each (program, mode) it:
  1. builds the isa code + seed_mem (§Memory data segment) + expected bytes,
  2. emits the header, gcc -O3 -static the all-C binary,
  3. runs it (strict: no --io-burst; burst: --io-burst) with the program's stdin,
  4. asserts the output == the clean-room reference (_agent_io_reference) AND the
     unix tool where applicable,
  5. times the wall (strict vs burst) and counts VM steps (fewer forwards = faster).

Report:  the two modes, byte-exact (mode1 == mode2 == reference == unix), the
wall-time speedup of burst over strict, one static binary (size + ldd).
Tooling only; golden model unchanged; additive.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
C4MIN = os.path.dirname(HERE)

DEFAULT_BLOB = "/tmp/fullisa_sparse/blockstack.nblbin"
BUILD_SH = os.path.join(HERE, "_agent_allc_build.sh")


def _fmt(b: bytes) -> str:
    return repr(b)


def run_binary(exe, io_burst, stdin_bytes=b"", trace=False, timing=False,
               threads=4, timeout=600):
    env = dict(os.environ)
    env["INCR_THREADS"] = str(threads)
    if io_burst:
        env["C4_IO_BURST"] = "1"
    if trace:
        env["ALLC_TRACE"] = "1"
    if timing:
        env["ALLC_TIMING"] = "1"
    t0 = time.time()
    try:
        p = subprocess.run([exe, "--allc"], input=stdin_bytes,
                           capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, None, b"TIMEOUT"
    wall = time.time() - t0
    steps = None
    m = re.search(rb"allc: (\d+) VM steps", p.stderr)
    if m:
        steps = int(m.group(1))
    return p.stdout, wall, steps, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default=DEFAULT_BLOB)
    ap.add_argument("--work", default="/tmp/io_bakeall")
    ap.add_argument("--progs", default="echo,yes,cat",
                    help="comma list of echo,yes,cat")
    ap.add_argument("--text", default="hello\n")
    ap.add_argument("--n", type=int, default=6, help="yes repeat count")
    ap.add_argument("--stdin", default="cat me\n", help="cat/eliza stdin")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--layout-pkl", default="/tmp/io_layout.pkl",
                    help="cached pickled layout L (skips the slow model build)")
    ap.add_argument("--embed-npy", default="/tmp/io_embed.npy")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    from c4_min.selfhost import _agent_allc_gen as G
    from c4_min.selfhost import _agent_io_progs as IOP
    from c4_min.selfhost import _agent_io_reference as REF

    if os.path.exists(args.layout_pkl) and os.path.exists(args.embed_npy):
        import pickle
        import numpy as np
        print(f"loading cached layout {args.layout_pkl} (no model build)...",
              flush=True)
        with open(args.layout_pkl, "rb") as f:
            cache = pickle.load(f)
        L = cache["L"]
        embed = np.load(args.embed_npy)
        # The 8-bit model sets PFC.SP_INIT=252 as a build side-effect; the cache
        # captured that value.  Restore it into the module global so gen_header's
        # layout_consts() reproduces the SAME SP_INIT the model expects (not the
        # 65536 import-time default the fresh module carries).
        if "consts" in cache:
            import c4_min.nibble_pure_forward as _PF
            import c4_min.nibble_pure_forward_complete as _PFC
            _PF.SP_INIT = _PFC.SP_INIT = int(cache["consts"]["SP_INIT"])
            print(f"  restored SP_INIT = {_PFC.SP_INIT}", flush=True)
        print(f"  loaded layout  D={L.D}  embed={embed.shape}", flush=True)
    else:
        print("building compact full-ISA model + layout (ONCE) ...", flush=True)
        t0 = time.time()
        L, embed = G.build_layout()
        print(f"  built layout in {time.time()-t0:.1f}s  D={L.D}", flush=True)

    prog_names = [p.strip() for p in args.progs.split(",") if p.strip()]

    def build_one(name, io_mode):
        if name == "echo":
            return IOP.build("echo", io_mode, text=args.text)
        if name == "yes":
            return IOP.build("yes", io_mode, text=args.text, n=args.n)
        if name == "cat":
            return IOP.build("cat", io_mode, stdin_text=args.stdin)
        raise ValueError(name)

    results = []
    for name in prog_names:
        row = {"name": name}
        for io_mode in ("strict", "burst"):
            prog = build_one(name, io_mode)
            code = G.assemble(prog.code)
            hdr = os.path.join(args.work, f"gen_{name}_{io_mode}.h")
            G.gen_header(hdr, L, embed, code, prog.seed_mem)
            exe = os.path.join(args.work, f"allc_{name}_{io_mode}")
            r = subprocess.run(["bash", BUILD_SH, hdr, exe, args.blob],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"BUILD FAILED {name}/{io_mode}:\n{r.stderr}")
                sys.exit(2)
            # reference oracle
            ref_out, ref_steps = REF.run_reference(prog, io_burst=(io_mode == "burst"))
            # run the binary
            out, wall, steps, stderr = run_binary(
                exe, io_burst=(io_mode == "burst"),
                stdin_bytes=prog.stdin, trace=args.trace,
                threads=args.threads, timeout=args.timeout)
            ok_ref = out == ref_out
            ok_exp = out == prog.expected
            row[io_mode] = dict(out=out, wall=wall, steps=steps,
                                ref_out=ref_out, ok_ref=ok_ref, ok_exp=ok_exp,
                                exe=exe, size=os.path.getsize(exe),
                                stderr=stderr)
            print(f"  [{name:5s} {io_mode:6s}] out={_fmt(out):30s} "
                  f"steps={steps} wall={wall:.2f}s  ref_ok={ok_ref} exp_ok={ok_exp}",
                  flush=True)
        results.append(row)

    # ---- summary + verification + measurement ----
    print("\n================ VERIFY + MEASURE ================")
    all_ok = True
    for row in results:
        name = row["name"]
        s, b = row["strict"], row["burst"]
        mode_eq = s["out"] == b["out"]           # MODE1 == MODE2
        ref_eq = s["ok_ref"] and b["ok_ref"]     # both == reference
        unix = None
        if name == "echo":
            # `printf '%s' text` writes text verbatim to stdout (no trailing NL add)
            unix = subprocess.run(["printf", "%s", args.text],
                                  capture_output=True).stdout
        elif name == "yes":
            # unix `yes STR | head -n N` == STR repeated N times
            u = subprocess.run(
                f"yes {args.text.rstrip(chr(10))!r} | head -n {args.n}",
                shell=True, capture_output=True).stdout
            unix = u
        elif name == "cat":
            # cat is identity: unix `cat` of the same stdin == the stdin bytes
            unix = subprocess.run(["cat"], input=args.stdin.encode(),
                                  capture_output=True).stdout
        unix_ok = (unix is None) or (s["out"] == unix)
        speedup = (s["wall"] / b["wall"]) if b["wall"] > 0 else float("nan")
        step_ratio = ((s["steps"] / b["steps"])
                      if (s.get("steps") and b.get("steps")) else float("nan"))
        ok = mode_eq and ref_eq and s["ok_exp"] and b["ok_exp"] and unix_ok
        all_ok = all_ok and ok
        ss = "TIMEOUT" if s["steps"] is None else f"{s['steps']:3d}"
        bs = "TIMEOUT" if b["steps"] is None else f"{b['steps']:3d}"
        print(f"{name:6s}: MODE1==MODE2={mode_eq}  both==ref={ref_eq}  "
              f"unix_ok={unix_ok}  out={_fmt(s['out'])}")
        print(f"        strict: {ss} steps  {s['wall']:6.2f}s   |  "
              f"burst: {bs} steps  {b['wall']:6.2f}s   "
              f"=> burst {speedup:.1f}x faster wall, {step_ratio:.1f}x fewer forwards")
    # static-binary facts (one representative)
    rep = results[0]["burst"]["exe"]
    ldd = subprocess.run(["ldd", rep], capture_output=True, text=True)
    print(f"\nstatic binary: {rep}  ({results[0]['burst']['size']:,} bytes)")
    print(f"  ldd: {ldd.stdout.strip() or ldd.stderr.strip()}")
    print("\n==> ALL BYTE-EXACT (mode1==mode2==reference)"
          if all_ok else "\n==> SOME MISMATCH — see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
