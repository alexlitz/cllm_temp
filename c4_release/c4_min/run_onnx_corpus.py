#!/usr/bin/env python3
"""CHK-2 — run the corpus through onnxruntime, byte-identically to torch.

Three phases, all against the COMPACT whole-VM model exported to a vanilla ONNX
graph (``export_onnx_compact``):

  1. **Byte-identity battery** (``--battery``): a per-op battery (add / sub / mul,
     32-bit, cmp, memory SI/LI, functions JSR/ENT/LEV, loops) driven through the
     SAME KV-cached driver, once with the torch model and once with the ONNX model
     as the forward.  Reports the emitted trace equality + the per-step hidden max
     logit diff and any argmax flips.

  2. **Corpus** (default): drive every corpus program through the KV-cached driver
     with the ONNX model as the forward, and compare the PASS/FAIL verdict to the
     torch pure-forward run (built here in the same process so the comparison is
     apples-to-apples).  Reports the ONNX pass fraction and the torch-vs-ONNX
     agreement.  The deep-loop tail has the same throughput caveat as the torch
     run (``--step-cap`` bounds it); ``--exclude-deep`` skips the >N-step tail.

The ONNX model is driven by ``run_pure_forward_cached`` UNCHANGED — the
``OnnxCachedModel`` wrapper exposes the same ``.forward_hidden_cached`` /
``.embed`` / ``.blocks`` API, so the block-stack compute runs entirely in
onnxruntime while the register decode + softmax1/ALiBi eviction stay in the
caller.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # ORT is CPU; keep torch on CPU

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import c4_min.nibble_pure_forward as _PF  # noqa: E402
import c4_min.nibble_pure_forward_complete as _PFC  # noqa: E402
_PF.SP_INIT = 0xF0
_PFC.SP_INIT = 0xF0

from c4_min import isa  # noqa: E402
from c4_min.export_onnx_compact import (  # noqa: E402
    build_compact, export_cached_onnx, to_sparse_onnx, op_inventory,
    assert_vanilla, OnnxCachedModel, _fmt_bytes,
)
from c4_min.nibble_pure_forward_cached import run_pure_forward_cached  # noqa: E402
from c4_min.nibble_pure_forward_complete import ref_interpret  # noqa: E402


# ===========================================================================
# 1. Byte-identity battery.
# ===========================================================================
def I(op, imm=0):
    return isa.Instr(op, imm)


def battery(include_divmod: bool):
    """(name, code, max_steps, mask) — one program per op family."""
    b = [
        ("add",  [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 3), I(isa.ADD), I(isa.HALT)], 20, 0xFF),
        ("sub",  [I(isa.IMM, 40), I(isa.PSH), I(isa.IMM, 9), I(isa.SUB), I(isa.HALT)], 20, 0xFF),
        ("mul",  [I(isa.IMM, 6), I(isa.PSH), I(isa.IMM, 7), I(isa.MUL), I(isa.HALT)], 20, 0xFF),
        ("add32", [I(isa.IMM, 60000), I(isa.PSH), I(isa.IMM, 60000), I(isa.ADD), I(isa.HALT)], 20, 0xFFFFFFFF),
        ("mul32", [I(isa.IMM, 1000), I(isa.PSH), I(isa.IMM, 1000), I(isa.MUL), I(isa.HALT)], 20, 0xFFFFFFFF),
        ("lt",   [I(isa.IMM, 3), I(isa.PSH), I(isa.IMM, 5), I(isa.LT), I(isa.HALT)], 20, 0xFF),
        ("eq",   [I(isa.IMM, 5), I(isa.PSH), I(isa.IMM, 5), I(isa.EQ), I(isa.HALT)], 20, 0xFF),
        ("si_li", [I(isa.IMM, 7), I(isa.PSH), I(isa.IMM, 200), I(isa.PSH),
                   I(isa.IMM, 7), I(isa.SI), I(isa.IMM, 200), I(isa.LI), I(isa.HALT)], 30, 0xFF),
        ("func_jsr_lev", [I(isa.JSR, 3), I(isa.HALT), I(isa.NOP),
                          I(isa.ENT, 0), I(isa.IMM, 42), I(isa.LEV)], 30, 0xFF),
        ("loop_countdown", [I(isa.IMM, 8), I(isa.PSH), I(isa.IMM, 1), I(isa.SUB),
                            I(isa.BNZ, 1), I(isa.HALT)], 200, 0xFF),
    ]
    if include_divmod:
        b += [
            ("div", [I(isa.IMM, 100), I(isa.PSH), I(isa.IMM, 7), I(isa.DIV), I(isa.HALT)], 20, 0xFFFFFFFF),
            ("mod", [I(isa.IMM, 100), I(isa.PSH), I(isa.IMM, 7), I(isa.MOD), I(isa.HALT)], 20, 0xFFFFFFFF),
        ]
    return b


class _LoggingOnnx(OnnxCachedModel):
    """OnnxCachedModel that records the max |hidden_onnx - hidden_torch| and any
    argmax flip per step, by re-running the torch block-stack on the same input."""

    def __init__(self, torch_model, onnx_path, torch_ref, **kw):
        super().__init__(torch_model, onnx_path, **kw)
        self._torch = torch_ref
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.argmax_flips = 0
        self.steps = 0

    def forward_hidden_cached(self, x, past_key_values=None, q_positions=None,
                              use_cache=True):
        hidden_o, new_o = super().forward_hidden_cached(
            x, past_key_values=past_key_values, q_positions=q_positions,
            use_cache=use_cache)
        with torch.no_grad():
            hidden_t, _ = self._torch.forward_hidden_cached(
                x, past_key_values=past_key_values, q_positions=q_positions,
                use_cache=use_cache)
        diff = (hidden_o - hidden_t).abs()
        d = float(diff.max())
        self.max_abs = max(self.max_abs, d)
        # relative diff (the c4 residual bands carry huge-magnitude arithmetic
        # values, so the absolute max is dominated by a ~1e-6-relative fp residue
        # on a ~1e6-magnitude dim; the relative max is the honest fp gap).
        denom = hidden_t.abs().clamp(min=1e-6)
        self.max_rel = max(self.max_rel, float((diff / denom).max()))
        # argmax over the residual dims of the query (last) row.
        a_o = int(hidden_o[0, -1].argmax()); a_t = int(hidden_t[0, -1].argmax())
        if a_o != a_t:
            self.argmax_flips += 1
        self.steps += 1
        return hidden_o, new_o


def run_battery(torch_model, L, onnx_path, include_divmod: bool,
                intra_threads: int) -> Tuple[int, int, float, int]:
    n_ok = n_fail = 0
    global_max_abs = 0.0
    global_max_rel = 0.0
    global_flips = 0
    print(f"\n{'prog':16s} {'ref':>10s} {'torch':>10s} {'onnx':>10s} "
          f"{'trace==':>8s} {'maxabs':>10s} {'maxrel':>10s} {'flips':>6s}")
    print("-" * 88)
    for name, code, msteps, mask in battery(include_divmod):
        ref = ref_interpret(code, max_steps=msteps, mask=mask)
        tr_t = run_pure_forward_cached(torch_model, L, code, max_steps=msteps,
                                       mask=mask, evict=True, prune_interval=120)
        onnx_m = _LoggingOnnx(torch_model, onnx_path, torch_model,
                              intra_threads=intra_threads)
        tr_o = run_pure_forward_cached(onnx_m, L, code, max_steps=msteps,
                                       mask=mask, evict=True, prune_interval=120)
        ident = (tr_t == tr_o)
        ok = ident
        n_ok += int(ok)
        n_fail += int(not ok)
        global_max_abs = max(global_max_abs, onnx_m.max_abs)
        global_max_rel = max(global_max_rel, onnx_m.max_rel)
        global_flips += onnx_m.argmax_flips
        rv = ref[-1] if ref else None
        tv = tr_t[-1] if tr_t else None
        ov = tr_o[-1] if tr_o else None
        print(f"{'OK ' if ok else '!! '}{name:13s} {str(rv):>10s} {str(tv):>10s} "
              f"{str(ov):>10s} {str(ident):>8s} {onnx_m.max_abs:10.2e} "
              f"{onnx_m.max_rel:10.2e} {onnx_m.argmax_flips:6d}")
        if not ok:
            print(f"    torch={tr_t}\n    onnx ={tr_o}")
    print("-" * 88)
    print(f"BYTE-IDENTITY (torch trace == onnxruntime trace): {n_ok} OK, {n_fail} FAIL")
    print(f"  worst per-step hidden max ABS diff (torch vs ORT): {global_max_abs:.3e} "
          f"(large only on the huge-magnitude arithmetic bands)")
    print(f"  worst per-step hidden max REL diff (torch vs ORT): {global_max_rel:.3e} "
          f"(fp accumulation-order residue)")
    print(f"  total per-step register-decode argmax flips: {global_flips}")
    return n_ok, n_fail, global_max_abs, global_flips


# ===========================================================================
# 2. Corpus.
# ===========================================================================
def run_corpus(torch_model, L, onnx_path, args) -> int:
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    from c4_min.run_1096_pure_forward import bytecode_to_isa, cluster_of

    all_tests = generate_test_programs()
    indexed = list(enumerate(all_tests))
    if args.clusters:
        keep = {c.strip() for c in args.clusters.split(",") if c.strip()}
        indexed = [(i, tp) for i, tp in indexed if cluster_of(tp[2]) in keep]
    if args.per_cluster is not None:
        seen: Counter = Counter()
        sampled = []
        for idx, tp in indexed:
            cl = cluster_of(tp[2])
            if seen[cl] < args.per_cluster:
                seen[cl] += 1
                sampled.append((idx, tp))
        indexed = sampled
    if args.shard_of > 1:
        indexed = [it for i, it in enumerate(indexed)
                   if i % args.shard_of == args.shard_idx]
    if args.limit is not None:
        indexed = indexed[:args.limit]

    onnx_model = OnnxCachedModel(torch_model, onnx_path,
                                 intra_threads=args.intra_threads)

    print(f"\n[corpus] scoring {len(indexed)}/{len(all_tests)} programs through "
          f"onnxruntime (step_cap={args.step_cap}, evict=ON)")
    t0 = time.monotonic()
    rows: List[dict] = []
    n_onnx_pass = n_torch_pass = 0
    n_agree = n_disagree = 0
    onnx_steps = 0
    for k, (idx, (source, expected, description)) in enumerate(indexed):
        exp = expected & 0xFFFFFFFF
        cluster = cluster_of(description)
        try:
            code = bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(idx=idx, cluster=cluster, status="ERROR",
                             detail=f"compile: {exc!r}"))
            continue
        # ref step count -> cap (same sizing as the torch runner).
        try:
            ref_tr = ref_interpret(code, max_steps=200000, mask=0xFFFFFFFF)
            ref_steps = len(ref_tr)
        except Exception:  # noqa: BLE001
            ref_steps = None
        cap = args.step_cap if ref_steps is None else min(args.step_cap, ref_steps + 6)
        if args.exclude_deep and ref_steps is not None and ref_steps > args.deep_threshold:
            continue

        st_t: dict = {}
        st_o: dict = {}
        try:
            tr_o = run_pure_forward_cached(onnx_model, L, code, max_steps=cap,
                                           mask=0xFFFFFFFF, evict=True,
                                           prune_interval=120, stats=st_o)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(idx=idx, cluster=cluster, status="ERROR",
                             detail=f"onnx run: {exc!r}"))
            continue
        onnx_steps += st_o.get("steps", 0)

        def verdict(tr):
            if not tr:
                return None, "ERROR"
            steps = len(tr)
            if steps >= cap:
                return int(tr[-1]) & 0xFFFFFFFF, "TIMEOUT"
            got = int(tr[-1]) & 0xFFFFFFFF
            return got, ("PASS" if got == exp else "FAIL")

        got_o, stat_o = verdict(tr_o)
        n_onnx_pass += int(stat_o == "PASS")

        row = dict(idx=idx, cluster=cluster, expected=exp,
                   onnx_status=stat_o, onnx_got=got_o,
                   steps=st_o.get("steps"), max_cache=st_o.get("max_cache_size"))
        if args.torch_crosscheck:
            try:
                tr_t = run_pure_forward_cached(torch_model, L, code, max_steps=cap,
                                               mask=0xFFFFFFFF, evict=True,
                                               prune_interval=120, stats=st_t)
            except Exception:  # noqa: BLE001
                tr_t = None
            got_t, stat_t = verdict(tr_t)
            n_torch_pass += int(stat_t == "PASS")
            trace_ident = (tr_t == tr_o)
            agree = (stat_o == stat_t) and trace_ident
            n_agree += int(agree)
            n_disagree += int(not agree)
            row.update(torch_status=stat_t, torch_got=got_t,
                       trace_ident=trace_ident, agree=agree)
        rows.append(row)
        if args.progress and (k + 1) % args.progress == 0:
            el = time.monotonic() - t0
            print(f"[corpus] {k+1}/{len(indexed)} | onnx_pass={n_onnx_pass} "
                  f"torch_pass={n_torch_pass} agree={n_agree} disagree={n_disagree} "
                  f"| {el:.0f}s {onnx_steps/max(el,1e-6):.0f} st/s", flush=True)
            if args.output:
                _dump(args.output + ".partial", rows, n_onnx_pass, n_torch_pass,
                      n_agree, n_disagree, onnx_steps, time.monotonic() - t0,
                      onnx_path, len(indexed), partial=True)
    wall = time.monotonic() - t0

    total = sum(1 for r in rows if "onnx_status" in r)
    disagreements = [r for r in rows if "agree" in r and not r["agree"]]
    n_crosschecked = sum(1 for r in rows if "agree" in r)
    print("\n" + "=" * 74)
    print("CHK-2 — corpus through onnxruntime (compact whole-VM, KV-cached)")
    print("=" * 74)
    print(f"  scored (non-error): {total}")
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)  onnx steps: {onnx_steps} "
          f"({onnx_steps/max(wall,1e-6):.0f} st/s)")
    print(f"  ONNX  pass: {n_onnx_pass}/{total} "
          f"({100.0*n_onnx_pass/max(total,1):.2f}%)")
    if n_crosschecked:
        print(f"  torch pass (cross-checked {n_crosschecked}): {n_torch_pass}")
        print(f"  torch-vs-ONNX agreement (verdict AND full trace): "
              f"{n_agree}/{n_crosschecked} "
              f"({100.0*n_agree/max(n_crosschecked,1):.2f}%)  "
              f"disagreements: {n_disagree}")
        if disagreements:
            print("\n  DISAGREEMENTS (torch != onnx):")
            for r in disagreements[:40]:
                print(f"    id={r['idx']} [{r['cluster']}] onnx={r['onnx_status']}"
                      f"({r['onnx_got']}) torch={r['torch_status']}({r['torch_got']}) "
                      f"trace_ident={r['trace_ident']}")
        else:
            print("\n  torch and onnxruntime AGREE on every cross-checked program "
                  "(verdict + full byte trace).")
    else:
        print("  (torch cross-check skipped this run — --torch-crosscheck to enable)")

    # per-cluster ONNX pass table.
    cl_tot: Counter = Counter()
    cl_pass: Counter = Counter()
    for r in rows:
        if "onnx_status" not in r:
            continue
        cl_tot[r["cluster"]] += 1
        if r["onnx_status"] == "PASS":
            cl_pass[r["cluster"]] += 1
    print("-" * 74)
    print(f"  {'cluster':28s} {'onnx_pass':>10s} {'total':>7s}")
    for cl in sorted(cl_tot, key=lambda c: (-cl_pass[c], c)):
        print(f"  {cl:28s} {cl_pass[cl]:10d} {cl_tot[cl]:7d}")
    print("=" * 74)
    if args.output:
        _dump(args.output, rows, n_onnx_pass, n_torch_pass, n_agree,
              n_disagree, onnx_steps, wall, onnx_path, len(indexed))
        print(f"\n[corpus] wrote {args.output}")
    return 0 if n_disagree == 0 else 1


def _dump(path, rows, n_onnx_pass, n_torch_pass, n_agree, n_disagree,
          onnx_steps, wall, onnx_path, n_scored, partial=False):
    total = sum(1 for r in rows if "onnx_status" in r)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "partial": partial,
            "onnx_model": onnx_path,
            "onnx_model_bytes": os.path.getsize(onnx_path),
            "n_scheduled": n_scored,
            "scored": total,
            "wall_seconds": wall,
            "onnx_steps": onnx_steps,
            "onnx_pass": n_onnx_pass,
            "torch_pass": n_torch_pass,
            "agree": n_agree,
            "disagree": n_disagree,
            "rows": rows,
        }, fh, indent=2)


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--battery", action="store_true",
                    help="run the per-op byte-identity battery (and exit).")
    ap.add_argument("--divmod", action="store_true",
                    help="build the divmod (compact) model instead of LEAN.")
    ap.add_argument("--onnx-path", type=str, default=None,
                    help="reuse an existing exported ONNX (skip re-export).")
    ap.add_argument("--out-dir", type=str, default="/tmp/c4_compact_onnx")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--step-cap", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--clusters", type=str, default=None)
    ap.add_argument("--per-cluster", type=int, default=None,
                    help="stratified sample <= N programs per cluster.")
    ap.add_argument("--shard-of", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--exclude-deep", action="store_true",
                    help="skip programs whose reference step count exceeds "
                         "--deep-threshold (the deep-loop throughput tail).")
    ap.add_argument("--deep-threshold", type=int, default=400)
    ap.add_argument("--torch-crosscheck", action="store_true",
                    help="ALSO run the torch model per program and assert the "
                         "byte trace + verdict match (doubles wall time).")
    ap.add_argument("--intra-threads", type=int, default=4)
    ap.add_argument("--progress", type=int, default=20)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args(argv)

    include_divmod = args.divmod
    print(f"[build] building compact pure-forward model "
          f"(divmod={include_divmod}) ...", flush=True)
    t = time.time()
    compact, L, stats = build_compact(code_size=args.code_size,
                                      include_divmod=include_divmod)
    compact.eval()
    print(f"[build] compact model in {time.time()-t:.0f}s | "
          f"dim {stats.dim_before}->{stats.dim_after} blocks={stats.n_blocks} "
          f"heads={stats.n_heads} nnz={stats.nonzero_params}", flush=True)

    # export (or reuse).
    onnx_path = args.onnx_path
    if onnx_path is None:
        os.makedirs(args.out_dir, exist_ok=True)
        dense_path = os.path.join(args.out_dir, "compact_vm.onnx")
        onnx_path = os.path.join(args.out_dir, "compact_vm_sparse.onnx")
        print("[export] tracing KV-cached forward to ONNX ...", flush=True)
        te = time.time()
        export_cached_onnx(compact, dense_path)
        ss = to_sparse_onnx(dense_path, onnx_path)
        ok, notes = assert_vanilla(onnx_path)
        print(f"[export] done in {time.time()-te:.0f}s | "
              f"dense {_fmt_bytes(ss['dense_file_bytes'])} -> "
              f"sparse {_fmt_bytes(ss['sparse_file_bytes'])} | "
              f"VANILLA={'YES' if ok else 'NO'}", flush=True)
        for n in notes:
            print(f"         {n}")
        try:
            os.remove(dense_path)     # keep only the small sparse file
        except OSError:
            pass

    if args.battery:
        n_ok, n_fail, mx, flips = run_battery(
            compact, L, onnx_path, include_divmod, args.intra_threads)
        return 0 if n_fail == 0 else 1

    return run_corpus(compact, L, onnx_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
