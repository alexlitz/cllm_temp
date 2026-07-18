#!/usr/bin/env python3
"""CHECKPOINT-RESUMABLE full-1096 pure-forward measurement.

Scores every program in the 1096 corpus through the SPARSE divmod pure-forward
whole-VM model via the VALIDATED block-verify path
(:func:`pf_speculative.speculative_run`) — draft with the reference VM (zero
forwards), then verify block-wise on the model (a handful of batched forwards),
PASS iff the model ACCEPTS the whole drafted stream AND the decoded final AX ==
expected.  This is byte-identical to the token-by-token KV-cached driver
(``spotcheck_vs_cached``) and reproduced the 6/6 deep-loop sanity
(loop_sum 171/153, gcd 2/4, rec_fib 1/2) with ``got != 0``.

CRASH-RESILIENCE (the whole point):
  * results are written to a PERSISTENT JSON in the worktree AS WE GO — every
    program is flushed (atomic temp+rename) the instant it is scored, so a
    process death loses at most the in-flight program;
  * on (re)start the JSON is LOADED and already-scored ids are SKIPPED, so the
    multi-hour sweep RESUMES from where it died instead of restarting.

MEMORY: builds ONLY the SPARSE divmod model (peak RSS ~15 GB during the compact
build, then ~tens of MB VRAM) — NEVER the 106 GB dense-compact path.

Shard across the two idle A5000s (one process per GPU, INDEPENDENT resumable
checkpoints), then merge:
    python -m c4_min.run_1096_measure_resumable --device cuda:0 \\
        --shard-of 2 --shard-idx 0 --checkpoint _measure_1096_progress.s0.json
    python -m c4_min.run_1096_measure_resumable --device cuda:1 \\
        --shard-of 2 --shard-idx 1 --checkpoint _measure_1096_progress.s1.json
    python -m c4_min.run_1096_measure_resumable --merge \\
        _measure_1096_progress.s0.json _measure_1096_progress.s1.json \\
        --checkpoint _measure_1096_progress.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

_STATUSES = ("PASS", "FAIL", "TIMEOUT", "ERROR", "DEEP")

_WORD = 8


# ---------------------------------------------------------------------------
# Local bytecode->isa + cluster key (do NOT import run_1096_pure_forward — it
# blanks CUDA_VISIBLE_DEVICES at import time).
# ---------------------------------------------------------------------------
def _sign32(imm: int) -> int:
    return imm if imm < (1 << 31) else imm - (1 << 32)


def _cluster_of(description: str) -> str:
    import re
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


# ---------------------------------------------------------------------------
# Persistent checkpoint I/O (atomic temp+rename so a crash never leaves a
# half-written JSON).
# ---------------------------------------------------------------------------
def _load_checkpoint(path: str) -> Dict[int, dict]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # a partial/corrupt file (killed mid non-atomic write) -> try the .bak.
        bak = path + ".bak"
        if os.path.exists(bak):
            try:
                with open(bak, encoding="utf-8") as fh:
                    blob = json.load(fh)
            except (OSError, json.JSONDecodeError):
                return {}
        else:
            return {}
    out: Dict[int, dict] = {}
    for r in blob.get("results", []):
        out[int(r["idx"])] = r
    return out


def _flush_checkpoint(path: str, results_by_idx: Dict[int, dict], meta: dict):
    if not path:
        return
    counts = Counter(r["status"] for r in results_by_idx.values())
    ordered = [results_by_idx[i] for i in sorted(results_by_idx)]
    blob = {
        "meta": meta,
        "summary": {s: counts.get(s, 0) for s in _STATUSES}
        | {"total": len(ordered), "pass": counts.get("PASS", 0)},
        "clusters": _cluster_table_dict(ordered),
        "results": ordered,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


def _cluster_table_dict(results: List[dict]) -> "OrderedDict[str, dict]":
    table: "OrderedDict[str, dict]" = OrderedDict()
    for r in results:
        row = table.setdefault(r["cluster"], {"n": 0, **{s: 0 for s in _STATUSES}})
        row["n"] += 1
        row[r["status"]] = row.get(r["status"], 0) + 1
    return table


def _print_report(results_by_idx: Dict[int, dict], wall: float, coverage: str):
    ordered = [results_by_idx[i] for i in sorted(results_by_idx)]
    counts = Counter(r["status"] for r in ordered)
    total = len(ordered)
    n_pass = counts.get("PASS", 0)
    print("\n" + "=" * 74)
    print("c4_min SPARSE PURE-FORWARD VM — FULL 1096 (checkpoint-resumable)")
    print("=" * 74)
    print(f"  coverage: {coverage}")
    print(f"  wall: {wall:.0f}s ({wall/60:.1f} min)")
    print("-" * 74)
    for s in _STATUSES:
        print(f"  {s:10s} {counts.get(s, 0):5d}")
    print("-" * 74)
    print(f"  SCORE:  {n_pass}/{total}  ({100.0*n_pass/max(total,1):.2f}%)")
    print("=" * 74)
    table = _cluster_table_dict(ordered)
    print("\nPER-CLUSTER BREAKDOWN")
    hdr = (f"  {'cluster':18s} {'n':>4s} {'PASS':>5s} {'FAIL':>5s} "
           f"{'TMOUT':>6s} {'ERR':>4s} {'DEEP':>5s} {'pass%':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cl, row in sorted(table.items()):
        n = row["n"]
        pct = (100.0 * row["PASS"] / n) if n else 0.0
        print(f"  {cl:18s} {n:4d} {row['PASS']:5d} {row['FAIL']:5d} "
              f"{row['TIMEOUT']:6d} {row['ERROR']:4d} {row['DEEP']:5d} {pct:6.1f}")


# ---------------------------------------------------------------------------
# MERGE mode: fold N shard JSONs into one.
# ---------------------------------------------------------------------------
def _merge(shard_paths: List[str], out_path: str) -> int:
    merged: Dict[int, dict] = {}
    total_wall = 0.0
    for p in shard_paths:
        part = _load_checkpoint(p)
        merged.update(part)
        try:
            with open(p, encoding="utf-8") as fh:
                total_wall += json.load(fh).get("meta", {}).get("wall_seconds", 0.0)
        except (OSError, json.JSONDecodeError):
            pass
    _flush_checkpoint(out_path, merged,
                      {"merged_from": shard_paths, "wall_seconds": total_wall})
    _print_report(merged, total_wall, f"MERGED {len(shard_paths)} shards")
    print(f"\n[merge] wrote {out_path} ({len(merged)} programs)")
    return 0


# ---------------------------------------------------------------------------
# THE SWEEP.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--shard-of", type=int, default=1)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--checkpoint", type=str, required=False,
                    help="persistent progress JSON (loaded on start, flushed per program)")
    ap.add_argument("--block-steps", type=int, default=48)
    ap.add_argument("--prune-interval", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="draft/verify step budget (raise for the deep rec_fib tail)")
    ap.add_argument("--only-ids", type=str, default=None,
                    help="comma list of program ids to run (default: whole shard)")
    ap.add_argument("--deep-min-steps", type=int, default=None,
                    help="only run programs whose ref draft exceeds this many steps")
    ap.add_argument("--commit-every", type=int, default=50,
                    help="git-commit the checkpoint every N newly-scored programs")
    ap.add_argument("--no-git", action="store_true")
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--merge", nargs="+", default=None,
                    help="MERGE mode: fold these shard JSONs into --checkpoint")
    args = ap.parse_args(argv)

    if args.merge:
        return _merge(args.merge, args.checkpoint)

    ckpt = args.checkpoint
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(_HERE, ckpt)

    import torch
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.zeros(1).to(device)         # init CUDA before the long CPU build

    import c4_min.nibble_pure_forward as _PF
    import c4_min.nibble_pure_forward_complete as _PFC
    _PF.SP_INIT = 0xF0
    _PFC.SP_INIT = 0xF0
    from c4_min.isa import Instr, LEA, ENT, ADJ
    from c4_min.compact_alloc import build_compact_pure_forward_model
    from c4_min.sparse_forward import SparseTransformer
    from c4_min.pf_speculative import speculative_run, draft_pf_program
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs

    _SLOT_SCALED = frozenset({LEA, ENT, ADJ})

    def _bytecode_to_isa(bc):
        out = []
        for word in bc:
            op = int(word) & 0xFF
            imm = int(word) >> 8
            if op in _SLOT_SCALED:
                out.append(Instr(op, _sign32(imm) // _WORD))
            else:
                out.append(Instr(op, imm & 0xFFFFFFFF))
        return out

    all_tests = generate_test_programs()

    # select this shard's ids (stride shard so each GPU gets an even depth mix).
    if args.only_ids:
        want_ids = {int(x) for x in args.only_ids.split(",") if x.strip()}
        shard_ids = [i for i in range(len(all_tests)) if i in want_ids]
    else:
        shard_ids = [i for i in range(len(all_tests))
                     if args.shard_of <= 1 or (i % args.shard_of == args.shard_idx)]

    results_by_idx = _load_checkpoint(ckpt) if ckpt else {}
    already = set(results_by_idx)
    todo = [i for i in shard_ids if i not in already]
    coverage = (f"shard {args.shard_idx+1}/{args.shard_of}: {len(shard_ids)} ids, "
                f"{len(already & set(shard_ids))} already scored, {len(todo)} todo")
    print(f"[measure:{device}] {coverage}", file=sys.stderr, flush=True)

    if not todo:
        print(f"[measure:{device}] nothing to do — all scored.", file=sys.stderr)
        _print_report({i: results_by_idx[i] for i in shard_ids if i in results_by_idx},
                      0.0, coverage)
        return 0

    t_build = time.monotonic()
    print(f"[measure:{device}] building SPARSE divmod model ...", file=sys.stderr, flush=True)
    base, L, cstats = build_compact_pure_forward_model(
        code_size=64, include_bitwise=False, include_divmod=True)
    sparse = SparseTransformer(base, compute_mode="dense_kernel")
    st = sparse.stats()
    del base
    sparse = sparse.to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
        vram = torch.cuda.memory_allocated(torch.device(device)) / 1e6
    else:
        vram = 0.0
    print(f"[measure:{device}] built in {time.monotonic()-t_build:.0f}s | "
          f"blocks={len(sparse.blocks)} storage={st.sparse_mb:.1f}MB "
          f"VRAM={vram:.0f}MB", file=sys.stderr, flush=True)

    def _git_commit(msg):
        if args.no_git or not ckpt:
            return
        try:
            import subprocess
            subprocess.run(["git", "add", ckpt], cwd=_PKG_PARENT,
                           check=False, capture_output=True)
            subprocess.run(["git", "commit", "-q", "-m", msg,
                            "--author", "Alex Litzenberger <litzenbergeralex@gmail.com>"],
                           cwd=_PKG_PARENT, check=False, capture_output=True)
        except Exception:  # noqa: BLE001
            pass

    meta = {"device": device, "shard_of": args.shard_of, "shard_idx": args.shard_idx,
            "block_steps": args.block_steps, "prune_interval": args.prune_interval,
            "max_steps": args.max_steps, "storage_mb": st.sparse_mb,
            "vram_load_mb": vram, "wall_seconds": 0.0}

    t0 = time.monotonic()
    n_new = 0
    n_since_commit = 0
    for k, idx in enumerate(todo):
        source, expected, description = all_tests[idx]
        cluster = _cluster_of(description)
        exp = expected & 0xFFFFFFFF
        rec = {"idx": idx, "cluster": cluster, "description": description,
               "expected": exp}
        try:
            code = _bytecode_to_isa(compile_c(source)[0])
        except Exception as exc:  # noqa: BLE001
            rec.update(status="ERROR", got=None, steps=None,
                       detail=f"compile/translate: {exc!r}")
            results_by_idx[idx] = rec
            n_new += 1
            n_since_commit += 1
            if ckpt:
                _flush_checkpoint(ckpt, results_by_idx, {**meta,
                                  "wall_seconds": time.monotonic() - t0})
            continue

        # optional depth filter (deep-tail runs): draft cheaply to size the program.
        if args.deep_min_steps is not None:
            d = draft_pf_program(code, max_steps=args.max_steps, mask=0xFFFFFFFF)
            if not d.halted or d.step_count <= args.deep_min_steps:
                continue

        t1 = time.monotonic()
        try:
            r = speculative_run(sparse, L, code, exp, block_steps=args.block_steps,
                                max_steps=args.max_steps, device=device,
                                evict=True, prune_interval=args.prune_interval)
            rec.update(status=r.status, got=r.decoded_final_ax, steps=r.step_count,
                       forwards=r.forwards, speedup=round(r.speedup, 1),
                       max_cache=r.max_cache_size, wall=round(time.monotonic()-t1, 1),
                       detail=r.detail)
        except Exception as exc:  # noqa: BLE001
            rec.update(status="ERROR", got=None, steps=None,
                       wall=round(time.monotonic()-t1, 1), detail=f"run: {exc!r}")
        results_by_idx[idx] = rec
        n_new += 1
        n_since_commit += 1
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        if ckpt:
            _flush_checkpoint(ckpt, results_by_idx,
                              {**meta, "wall_seconds": time.monotonic() - t0})
        if args.progress and (k + 1) % args.progress == 0:
            npass = sum(1 for v in results_by_idx.values()
                        if v.get("idx") in set(shard_ids) and v["status"] == "PASS")
            el = time.monotonic() - t0
            print(f"[measure:{device}] {k+1}/{len(todo)} new (pass~{npass}) "
                  f"[{el:.0f}s] last id={idx} {rec['status']} "
                  f"got={rec.get('got')} steps={rec.get('steps')} "
                  f"fwds={rec.get('forwards')}", file=sys.stderr, flush=True)
        if n_since_commit >= args.commit_every:
            _git_commit(f"chk1-measure: checkpoint shard {args.shard_idx} "
                        f"+{n_since_commit} progs ({device})")
            n_since_commit = 0

    wall = time.monotonic() - t0
    if ckpt:
        _flush_checkpoint(ckpt, results_by_idx, {**meta, "wall_seconds": wall})
        _git_commit(f"chk1-measure: checkpoint shard {args.shard_idx} DONE ({device})")
    shard_results = {i: results_by_idx[i] for i in shard_ids if i in results_by_idx}
    _print_report(shard_results, wall, coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
