#!/usr/bin/env python3
"""FAST FAITHFUL CORPUS GATE — predict the full-1096 pass-delta in ~10-15 min.

The gap this closes
-------------------
Every behaviour-changing campaign fix this cycle passed the NARROW checks
(``test_smoke``, ``cpu_full_trace`` on a handful of ids, the flat-interp gate)
and then REGRESSED at the 2-hour full-1096 run: ``C4_DERIVE_ADDSUB`` netted
**-34** (580 -> 546) while its own smoke sample was 13/13 clean, and the
existing ``tools/flag_regression_gate.py`` (72-prog CPU sample) did NOT catch
that -34. The 72-prog sample was too small AND unrepresentative: 4 add + 4 sub
is not enough to see a -34 that is SPREAD across add/sub AND the multi-byte
consumers (mul/div/mod/var/expr/nested/absdiff) that share the L10 OUTPUT
byte-1 bus a ``C4_DERIVE_ADDSUB``-style fix reroutes.

What this tool does
-------------------
It runs a STRATIFIED ~300-id sample (``tools/fast_gate_sample.json``) that
(a) covers ALL 56 clusters, (b) OVER-weights the regression-prone clusters the
ADD/SUB miss exposed (add/sub/mul/div/mod = the shared-OUTPUT-bus ALU families)
+ the MEM-SMOKE ``var_*`` clusters + the multi-byte consumers
(expr/nested/absdiff), and (c) samples BOTH passing and failing programs per
cluster so a regression (ok->fail) OR a gain (fail->ok) is visible in either
direction. It scores each program with the SAME per-program ``full_trace``
verdict as the 2-hour ``tools/run_1096_canonical.py --criterion full_trace
--spec-k 0`` gate (identical pass criterion), in BOTH the flag-OFF and flag-ON
(or base vs HEAD) build, then reports:

  * the PREDICTED full-1096 delta (the sample net scaled to the full corpus by
    each cluster's population), and
  * a per-cluster OFF->ON delta table (regressions + gains, MEM-SMOKE tagged).

On GPU (the ``run_1096_canonical`` batched path) the ~300-id sample is ~8-10
min per state; with two GPUs the two states run CONCURRENTLY (OFF on one, ON on
the other) so total wall is ~10 min. On a single GPU the two states run back to
back (~18 min). See ``docs/FAST_GATE_2026_07_09.md`` for the sample design and
the calibration table (it CATCHES the ADD/SUB -34).

Modes
-----
``--flag C4_MY_FIX``
    OFF state = the flag UNSET; ON state = ``C4_MY_FIX=1``. Both builds happen
    from the current working tree. The fix's build code must consult the flag
    (``os.environ.get("C4_MY_FIX")``). Clean, no git operations.

``--base <commit>``
    OFF state = the model built from ``<commit>``'s source (checked out into a
    throwaway ``git worktree`` — never touches the working tree, no stash). ON
    state = the current HEAD working tree. Use for a branch not behind a flag.

Usage
-----
    # Gate a fix behind its kill-switch flag (predict the full-1096 delta):
    python tools/fast_gate.py --flag C4_DERIVE_ADDSUB

    # Gate HEAD vs a base commit (branch-level, no single flag):
    python tools/fast_gate.py --base 59a9de19

    # Narrow to specific clusters while iterating (faster):
    python tools/fast_gate.py --flag C4_MY_FIX --clusters add,sub,mul,var_mul

    # Regenerate the stratified sample (needs OFF-baseline verdicts to pick
    # pass+fail representatives; pass a full-1096 --output json):
    python tools/fast_gate.py --regen-sample --baseline-json /tmp/off_full.json

Tooling only: this script never writes weights except through the normal
``run_1096_canonical`` subprocess, so the golden model is byte-identical
(golden ``e50521f3`` unchanged).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

_SAMPLE_PATH = os.path.join(_HERE, "fast_gate_sample.json")

# The campaign config is the DEFAULT build now (C4_NO_STACK0_EMIT=1 +
# C4_OPERAND_FROM_MEMSP=1 both default "1"), so the gate runs in the bare
# environment — no explicit campaign env needed. Kept as an override hook.
_CAMPAIGN_ENV: Dict[str, str] = {}

# The four memory-smoke clusters (SI/LI store-load + store/load-cell var_*
# families). Reported with an explicit MEM-SMOKE tag.
_MEM_SMOKE_CLUSTERS = {"var_simple", "var_mul", "var_three", "var_update"}

# Regression-prone cluster families the ADD/SUB miss exposed. A fix that
# reroutes the shared L10 OUTPUT byte-1 bus (as C4_DERIVE_ADDSUB does) perturbs
# EVERY multi-byte consumer, not just the op it targets. These get the heaviest
# sampling so a spread -34 is visible.
#   * SHARED_OUTPUT_BUS: the ALU families that write/read OUTPUT byte-1
#     (add/sub are the direct target; mul/div/mod share the same bus).
#   * MULTIBYTE_CONSUMER: clusters whose results flow through the same byte-1
#     cascade (var multi-byte stores, expr chains, nested, absdiff).
_SHARED_OUTPUT_BUS = {"add", "sub", "mul", "div", "mod"}
_MULTIBYTE_CONSUMER = {
    "expr_add_mul", "expr_paren", "expr_mul_div", "expr_mod",
    "nested_quad", "nested_sumsq", "absdiff",
}

# Step-depth cap. Programs deeper than this are SKIPPED (free, tracked but not
# decoded) — matches run_1096_canonical's short-only fast mode. All 846
# passable programs are <=39 steps; the deep gcd/loop/rec band (which the model
# fails anyway) only costs O(steps^2) wall for no verdict signal, so we cap at
# 40 and let those clusters contribute a tracked sentinel id only.
_GATE_RUN_CAP = 40

# Total sample budget knobs (used by --regen-sample). Sized so the sample runs
# in ~8-10 min per state on one GPU (~1.6s/program amortised + ~40s bake).
_N_HEAVY = 12   # per shared-output-bus cluster (add/sub/mul/div/mod) — CHEAP (<=8 steps)
_N_MEM = 10     # per MEM-SMOKE var_* cluster
# Multi-byte consumers: expr_* are CHEAP (<=8 steps) so keep 8; nested_* are the
# expensive 31/39-step width-2 band, so cap those tighter (see _N_CONSUMER_DEEP)
# to keep the gate's wall time near ~10 min while still covering the cluster.
_N_CONSUMER = 8       # per cheap consumer cluster (expr_*, absdiff ~24 steps)
_N_CONSUMER_DEEP = 5  # per EXPENSIVE consumer cluster (nested_quad 31, nested_sumsq 39)
_N_MEDIUM = 6   # per remaining 25-member cluster (if_*, func_*, bool_and)
_N_DEEP = 2     # per deep cluster (loop_*/rec_*/gcd) — tracked sentinels only
_N_EDGE_SMALL = 1  # per tiny edge_* cluster (1-15 members)

# The expensive-consumer clusters (31/39-step, width-2 band). Sampled at
# _N_CONSUMER_DEEP so the gate does not blow past ~12 min on the deep band.
_CONSUMER_DEEP = {"nested_quad", "nested_sumsq"}


# ---------------------------------------------------------------------------
# Cluster mapping (identical to run_1096_canonical.cluster_of).
# ---------------------------------------------------------------------------
def _cluster_of(description: str) -> str:
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def _cluster_families() -> "collections.OrderedDict[str, List[Tuple[int, str]]]":
    """cluster -> [(id, description), ...] over the whole 1096 corpus."""
    from tests.test_suite_1000 import generate_test_programs

    progs = generate_test_programs()
    fam: "collections.OrderedDict[str, List[Tuple[int, str]]]" = (
        collections.OrderedDict()
    )
    for i, (_src, _exp, desc) in enumerate(progs):
        fam.setdefault(_cluster_of(desc), []).append((i, desc))
    return fam


# ---------------------------------------------------------------------------
# Stratified sample construction.
# ---------------------------------------------------------------------------
def _n_for_cluster(cluster: str, pop: int) -> int:
    """How many representatives to draw from a cluster of population ``pop``."""
    if cluster in _SHARED_OUTPUT_BUS:
        n = _N_HEAVY
    elif cluster in _MEM_SMOKE_CLUSTERS:
        n = _N_MEM
    elif cluster in _CONSUMER_DEEP:
        n = _N_CONSUMER_DEEP
    elif cluster in _MULTIBYTE_CONSUMER:
        n = _N_CONSUMER
    elif cluster.startswith(("loop_", "rec_")) or cluster == "gcd":
        n = _N_DEEP
    elif pop >= 20:
        n = _N_MEDIUM
    else:
        n = _N_EDGE_SMALL
    return min(n, pop)


def _stratified_pick(
    ids_descs: List[Tuple[int, str]],
    n: int,
    steps: Dict[int, int],
    off_status: Optional[Dict[int, str]],
) -> List[int]:
    """Pick ``n`` ids from a cluster, favouring SHORT programs and, when the
    OFF baseline verdicts are available, a BALANCED pass/fail split.

    Without ``off_status`` (no baseline yet): take the ``n`` cheapest (shortest)
    in-cap members, evenly spread across the cluster's index range so we don't
    cluster on the first few near-identical cases.

    With ``off_status``: split the budget ~half pass / ~half fail (so BOTH a
    fail->ok gain and an ok->fail regression are observable), preferring short
    members within each half. Deep (>cap) members are eligible only to fill a
    half that would otherwise be empty (they decode as 'skipped' and cost
    nothing, keeping the cluster tracked).
    """
    in_cap = [(i, d) for (i, d) in ids_descs if steps.get(i, 10**9) <= _GATE_RUN_CAP]
    over_cap = [(i, d) for (i, d) in ids_descs if steps.get(i, 10**9) > _GATE_RUN_CAP]

    def by_short(seq):
        return sorted(seq, key=lambda x: (steps.get(x[0], 10**9), x[0]))

    if off_status is None:
        pool = by_short(in_cap) or by_short(over_cap)
        if not pool:
            return []
        # Evenly spread across the (short-sorted) pool so we don't take only
        # the very cheapest near-duplicate first cases.
        if n >= len(pool):
            chosen = pool
        else:
            step = len(pool) / n
            chosen = [pool[int(k * step)] for k in range(n)]
        return sorted({i for i, _ in chosen})

    passing = by_short([(i, d) for (i, d) in in_cap if off_status.get(i) == "ok"])
    failing = by_short([(i, d) for (i, d) in in_cap if off_status.get(i) not in ("ok", None)])
    n_pass = n // 2
    n_fail = n - n_pass
    # If one side is short, spill the remainder to the other side.
    take_pass = passing[:n_pass]
    take_fail = failing[:n_fail]
    deficit = n - len(take_pass) - len(take_fail)
    if deficit > 0:
        extra_pass = passing[len(take_pass):]
        extra_fail = failing[len(take_fail):]
        spill = by_short(extra_pass + extra_fail)[:deficit]
        take = take_pass + take_fail + spill
    else:
        take = take_pass + take_fail
    chosen = {i for i, _ in take}
    # Ensure the cluster is tracked even if it had no in-cap member sampled.
    if not chosen and over_cap:
        chosen = {by_short(over_cap)[0][0]}
    return sorted(chosen)


def _build_sample(baseline_json: Optional[str]) -> Dict[str, object]:
    """(Re)build the stratified sample json.

    ``baseline_json`` (optional): a full-1096 ``run_1096_canonical
    --criterion full_trace`` OFF-baseline ``--output`` file. When present its
    per-program verdicts drive a balanced pass/fail split per cluster (so a
    regression AND a gain are both observable). Without it, the sample is a
    deterministic short-member stratification (still covers every cluster and
    over-weights the regression-prone families).
    """
    from tools.run_1096_fast import _compile_and_oracle

    fam = _cluster_families()

    # Step counts for every program (cheap: compile + declarative oracle only).
    all_sel = [(i, "", 0, d) for ids in fam.values() for (i, d) in ids]
    # Rebuild proper (idx, src, exp, desc) tuples from the suite.
    from tests.test_suite_1000 import generate_test_programs

    progs = generate_test_programs()
    sel = [(i, progs[i][0], progs[i][1], progs[i][2]) for i in range(len(progs))]
    prepared, _errs = _compile_and_oracle(sel)
    steps = {e[0]: e[4] for e in prepared}

    off_status: Optional[Dict[int, str]] = None
    if baseline_json and os.path.exists(baseline_json):
        data = json.load(open(baseline_json))
        off_status = {r["idx"]: r["status"] for r in data["results"]}

    per_cluster: Dict[str, List[int]] = {}
    for cluster, ids_descs in fam.items():
        n = _n_for_cluster(cluster, len(ids_descs))
        per_cluster[cluster] = _stratified_pick(ids_descs, n, steps, off_status)

    all_ids = sorted(i for v in per_cluster.values() for i in v)
    return {
        "note": (
            "fast_gate stratified sample. Covers ALL 56 clusters; OVER-weights "
            "the regression-prone shared-OUTPUT-bus ALU families "
            "(add/sub/mul/div/mod, %d each), the MEM-SMOKE var_* clusters (%d "
            "each), and the multi-byte consumers (expr/nested/absdiff, %d each) "
            "that a C4_DERIVE_ADDSUB-style byte-1 reroute perturbs. Balanced "
            "pass/fail split per cluster when a baseline verdict json is "
            "supplied. Deep loop_/rec_/gcd clusters get %d sentinel ids "
            "(decoded as SKIPPED at --max-steps-cap %d, free). Regenerate with "
            "`python tools/fast_gate.py --regen-sample --baseline-json <off_full.json>`."
            % (_N_HEAVY, _N_MEM, _N_CONSUMER, _N_DEEP, _GATE_RUN_CAP)
        ),
        "gate_max_steps_cap": _GATE_RUN_CAP,
        "baseline_json": baseline_json or None,
        "n_ids": len(all_ids),
        "per_cluster": per_cluster,
        "ids": all_ids,
    }


def _load_sample(
    clusters_filter: Optional[List[str]],
) -> Tuple[List[int], Dict[int, str]]:
    """Load the sample ids + an id->cluster map (filtered to ``clusters_filter``)."""
    if not os.path.exists(_SAMPLE_PATH):
        print(f"[fast-gate] sample missing — generating {_SAMPLE_PATH} ...",
              file=sys.stderr, flush=True)
        spec = _build_sample(None)
        json.dump(spec, open(_SAMPLE_PATH, "w"), indent=1)
    else:
        spec = json.load(open(_SAMPLE_PATH))

    per_cluster = spec["per_cluster"]
    if clusters_filter:
        want = set(clusters_filter)
        per_cluster = {c: v for c, v in per_cluster.items() if c in want}
        missing = want - set(per_cluster)
        if missing:
            print(f"[fast-gate] WARNING: unknown clusters ignored: "
                  f"{sorted(missing)}", file=sys.stderr, flush=True)

    id_cluster: Dict[int, str] = {}
    for cluster, ids in per_cluster.items():
        for i in ids:
            id_cluster[i] = cluster
    return sorted(id_cluster), id_cluster


# ---------------------------------------------------------------------------
# GPU discovery + per-state run via run_1096_canonical.
# ---------------------------------------------------------------------------
def _visible_gpus() -> List[str]:
    """Return the list of usable GPU indices (as strings).

    Honours an explicit CUDA_VISIBLE_DEVICES; else probes nvidia-smi. Returns
    [] if no GPU is found (caller falls back to a single CPU-less run, which
    run_1096_canonical still handles — just slower)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [x.strip() for x in env.split(",") if x.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return []


def _run_state_cmd(
    *,
    label: str,
    ids: List[int],
    env: Dict[str, str],
    cwd: str,
    gpu: Optional[str],
    out_json: str,
    max_steps_cap: int,
    mem_step_scale: float,
) -> subprocess.Popen:
    """Spawn a run_1096_canonical full_trace subprocess for ``ids``.

    Returns the Popen (non-blocking) so the caller can run OFF and ON
    CONCURRENTLY on separate GPUs. The subprocess writes ``out_json``.
    """
    idstr = ",".join(map(str, ids))
    full_env = dict(os.environ)
    full_env.update(env)
    if gpu is not None:
        full_env["CUDA_VISIBLE_DEVICES"] = gpu
    # Keep the CPU thread pools capped per the memory discipline.
    full_env.setdefault("OMP_NUM_THREADS", "6")
    full_env.setdefault("MKL_NUM_THREADS", "6")
    full_env["PYTHONPATH"] = cwd + os.pathsep + full_env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        os.path.join(cwd, "tools", "run_1096_canonical.py"),
        "--ids", idstr,
        "--criterion", "full_trace",
        "--spec-k", "0",
        "--max-steps-cap", str(max_steps_cap),
        "--mem-step-scale", str(mem_step_scale),
        "--output", out_json,
    ]
    log_path = out_json + ".log"
    log_fh = open(log_path, "w")
    print(f"[fast-gate] launching state {label.upper()} on GPU "
          f"{gpu if gpu is not None else '<default>'} "
          f"({len(ids)} ids, cache={full_env.get('C4_VM_CACHE_DIR')}); "
          f"log -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, env=full_env, cwd=cwd,
                            stdout=log_fh, stderr=subprocess.STDOUT)
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    proc._out_json = out_json  # type: ignore[attr-defined]
    proc._label = label  # type: ignore[attr-defined]
    return proc


def _await(proc: subprocess.Popen) -> Dict[int, str]:
    """Wait for a state subprocess; return id->status. Aborts on non-zero rc."""
    rc = proc.wait()
    try:
        proc._log_fh.close()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    label = proc._label  # type: ignore[attr-defined]
    out_json = proc._out_json  # type: ignore[attr-defined]
    if rc != 0:
        print(f"[fast-gate] FATAL: state {label} run exited {rc}; see "
              f"{out_json}.log", file=sys.stderr, flush=True)
        raise SystemExit(3)
    data = json.load(open(out_json))
    status = {r["idx"]: r["status"] for r in data["results"]}
    n_pass = sum(1 for v in status.values() if v == "ok")
    n_skip = sum(1 for v in status.values() if v == "skipped")
    print(f"[fast-gate]   state {label} done "
          f"(wall={data.get('wall_seconds', 0):.0f}s, "
          f"pass={n_pass}/{len(status)}, skipped={n_skip}).", flush=True)
    return status


# ---------------------------------------------------------------------------
# Base-commit checkout (git-worktree, never touches the working tree).
# ---------------------------------------------------------------------------
def _git_root() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=_PKG
    ).decode().strip()


def _add_base_worktree(base: str) -> Tuple[str, str]:
    root = _git_root()
    wt = tempfile.mkdtemp(prefix="fastgate_base_")
    print(f"[fast-gate] checking out base {base} into worktree {wt} "
          f"(working tree untouched, no stash) ...", flush=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt, base],
        cwd=root, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    rel = os.path.relpath(_PKG, root)
    pkg_cwd = os.path.normpath(os.path.join(wt, rel))
    if not os.path.isdir(os.path.join(pkg_cwd, "tools")):
        pkg_cwd = wt
    return wt, pkg_cwd


def _remove_base_worktree(wt: str) -> None:
    root = _git_root()
    print(f"[fast-gate] removing base worktree {wt} ...", flush=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    shutil.rmtree(wt, ignore_errors=True)


# ---------------------------------------------------------------------------
# Predicted full-1096 delta + per-cluster report.
# ---------------------------------------------------------------------------
def _report(
    off: Dict[int, str],
    on: Dict[int, str],
    id_cluster: Dict[int, str],
    *,
    off_label: str,
    on_label: str,
) -> Tuple[int, int]:
    """Diff OFF->ON; print the per-cluster table + predicted full-1096 delta.

    Returns ``(rc, predicted_full_delta)``. rc=1 iff any cluster regressed.

    The full-1096 prediction scales each cluster's SAMPLE net (flips-regs) by
    that cluster's population / sampled-count, so a -k spread across N heavily
    sampled clusters extrapolates to the real full-corpus magnitude — this is
    what the 72-prog sample could not do (too few + unweighted).
    """
    from tests.test_suite_1000 import generate_test_programs

    progs = generate_test_programs()
    cluster_pop: "collections.Counter[str]" = collections.Counter(
        _cluster_of(d) for (_s, _e, d) in progs
    )

    per_cluster_sampled: "collections.Counter[str]" = collections.Counter(
        id_cluster.values()
    )
    reg: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    flip: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    for i, cluster in id_cluster.items():
        a, b = off.get(i, "MISSING"), on.get(i, "MISSING")
        if a == "ok" and b != "ok":
            reg[cluster].append((i, b))
        elif a != "ok" and b == "ok":
            flip[cluster].append((i, a))

    n_reg = sum(len(v) for v in reg.values())
    n_flip = sum(len(v) for v in flip.values())

    # Predicted full-1096 delta: per-cluster net scaled by population/sampled.
    predicted = 0.0
    per_cluster_pred: Dict[str, float] = {}
    for cluster in sorted(per_cluster_sampled):
        sampled = per_cluster_sampled[cluster]
        # Exclude 'skipped'/'MISSING' from the scoring denominator so the
        # scale-up reflects the DECODED fraction of the cluster.
        decoded = sum(
            1 for i in id_cluster
            if id_cluster[i] == cluster and off.get(i) in ("ok", "fail", "error")
            and on.get(i) in ("ok", "fail", "error")
        )
        net = len(flip.get(cluster, [])) - len(reg.get(cluster, []))
        if decoded > 0:
            scale = cluster_pop[cluster] / decoded
        else:
            scale = 0.0
        pred = net * scale
        per_cluster_pred[cluster] = pred
        predicted += pred

    def _tag(c: str) -> str:
        t = ""
        if c in _MEM_SMOKE_CLUSTERS:
            t += " [MEM-SMOKE]"
        if c in _SHARED_OUTPUT_BUS:
            t += " [ALU-BUS]"
        return t

    print("\n" + "=" * 84)
    print(f"FAST GATE  ({off_label}  ->  {on_label})  campaign/default config")
    print("=" * 84)

    print(f"\nREGRESSIONS (ok -> fail) [{n_reg}]  -- BLOCK the change:")
    if not reg:
        print("    (none)")
    for c in sorted(reg):
        print(f"    {c}{_tag(c)}: {reg[c]}")

    print(f"\nGAINS (fail -> ok) [{n_flip}]:")
    if not flip:
        print("    (none)")
    for c in sorted(flip):
        print(f"    {c}{_tag(c)}: {len(flip[c])} {flip[c]}")

    print("\nPER-CLUSTER SAMPLE net (flips - regs) and full-1096 prediction:")
    print(f"  {'cluster':22s} {'pop':>4s} {'smpl':>4s} {'flip':>4s} "
          f"{'reg':>4s} {'net':>4s} {'pred_full':>9s}")
    for c in sorted(per_cluster_pred):
        net = len(flip.get(c, [])) - len(reg.get(c, []))
        if net == 0 and per_cluster_pred[c] == 0:
            continue
        print(f"  {c:22s} {cluster_pop[c]:4d} {per_cluster_sampled[c]:4d} "
              f"{len(flip.get(c, [])):4d} {len(reg.get(c, [])):4d} "
              f"{net:4d} {per_cluster_pred[c]:+9.1f}{_tag(c)}")

    mem_reg = {c: reg[c] for c in _MEM_SMOKE_CLUSTERS if c in reg}
    print(f"\nMEM-SMOKE clusters (var_simple/var_mul/var_three/var_update): "
          f"{'REGRESSED ' + str(mem_reg) if mem_reg else 'clean'}")

    print(f"\nSAMPLE net: {n_flip - n_reg:+d}  (flips={n_flip} regs={n_reg})")
    print(f"PREDICTED full-1096 delta: {predicted:+.0f}  "
          f"({'REGRESSION — do NOT land' if predicted < -1 or n_reg else 'clean/gain'})")
    print("=" * 84, flush=True)
    return (1 if n_reg else 0), round(predicted)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--flag",
                      help="the fix's KILL-SWITCH env flag, e.g. C4_MY_FIX. OFF "
                           "leaves it unset; ON sets it to --flag-on-value.")
    mode.add_argument("--base",
                      help="compare HEAD's working tree (ON) against this base "
                           "commit (OFF), via a throwaway git worktree (no stash).")
    mode.add_argument("--regen-sample", action="store_true",
                      help="regenerate tools/fast_gate_sample.json and exit "
                           "(pass --baseline-json for a pass/fail-balanced sample).")
    ap.add_argument("--baseline-json",
                    help="a full-1096 run_1096_canonical --criterion full_trace "
                         "OFF-baseline --output json; used by --regen-sample to "
                         "balance pass/fail per cluster.")
    ap.add_argument("--flag-on-value", default="1",
                    help="value to set --flag to in the ON state (default '1').")
    ap.add_argument("--clusters",
                    help="comma-separated cluster names to restrict to.")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU indices to use (default: "
                         "auto-detect via nvidia-smi). With >=2 GPUs the OFF "
                         "and ON states run CONCURRENTLY (one per GPU).")
    ap.add_argument("--max-steps-cap", type=int, default=_GATE_RUN_CAP,
                    help=f"skip (count as 'skipped') any program deeper than this "
                         f"(default {_GATE_RUN_CAP}, the short-only fast mode).")
    ap.add_argument("--mem-step-scale", type=float, default=1.0,
                    help="scale on run_1096_canonical's per-bucket batch widths "
                         "(0.5 to co-exist with another GPU job).")
    ap.add_argument("--cache-dir", default=None,
                    help="base C4_VM_CACHE_DIR for the builds (default a fresh "
                         "/tmp/fastgate_cache_<pid> dir; off/on get subdirs).")
    ap.add_argument("--campaign-env", default="",
                    help="comma-separated K=V env both states run in (default "
                         "empty — the campaign config is already the DEFAULT).")
    args = ap.parse_args(argv)

    if args.regen_sample:
        spec = _build_sample(args.baseline_json)
        json.dump(spec, open(_SAMPLE_PATH, "w"), indent=1)
        print(f"[fast-gate] wrote {_SAMPLE_PATH}: {spec['n_ids']} ids across "
              f"{len(spec['per_cluster'])} clusters "
              f"(baseline={'yes' if args.baseline_json else 'no'}).")
        return 0

    if not (args.flag or args.base):
        ap.error("one of --flag / --base / --regen-sample is required")

    clusters_filter = (
        [c.strip() for c in args.clusters.split(",") if c.strip()]
        if args.clusters else None
    )
    ids, id_cluster = _load_sample(clusters_filter)
    if not ids:
        print("[fast-gate] no ids selected — nothing to do.", file=sys.stderr)
        return 2

    campaign = dict(_CAMPAIGN_ENV)
    if args.campaign_env:
        for kv in args.campaign_env.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                campaign[k.strip()] = v.strip()

    base_cache = os.path.abspath(
        args.cache_dir or os.path.join(tempfile.gettempdir(),
                                       f"fastgate_cache_{os.getpid()}")
    )
    os.makedirs(base_cache, exist_ok=True)

    gpus = (
        [g.strip() for g in args.gpus.split(",") if g.strip()]
        if args.gpus else _visible_gpus()
    )
    concurrent = len(gpus) >= 2

    print(f"[fast-gate] sample: {len(ids)} ids across "
          f"{len(set(id_cluster.values()))} cluster(s); "
          f"gpus={gpus or ['<default>']} "
          f"({'CONCURRENT off||on' if concurrent else 'sequential'}).")

    t_all = time.monotonic()
    base_wt = None
    off_json = os.path.join(base_cache, "off_results.json")
    on_json = os.path.join(base_cache, "on_results.json")
    try:
        if args.flag:
            off_env = dict(campaign)
            off_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "off")
            off_env[args.flag] = "0"
            on_env = dict(campaign)
            on_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "on")
            on_env[args.flag] = args.flag_on_value
            off_label = f"OFF (no {args.flag})"
            on_label = f"ON ({args.flag}={args.flag_on_value})"
            off_cwd = on_cwd = _PKG
        else:
            base_wt, base_cwd = _add_base_worktree(args.base)
            off_env = dict(campaign)
            off_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "base")
            on_env = dict(campaign)
            on_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "head")
            off_label = f"OFF (base {args.base[:8]})"
            on_label = "ON (HEAD)"
            off_cwd, on_cwd = base_cwd, _PKG

        gpu_off = gpus[0] if gpus else None
        gpu_on = gpus[1] if concurrent else (gpus[0] if gpus else None)

        off_proc = _run_state_cmd(
            label="off", ids=ids, env=off_env, cwd=off_cwd, gpu=gpu_off,
            out_json=off_json, max_steps_cap=args.max_steps_cap,
            mem_step_scale=args.mem_step_scale,
        )
        if concurrent:
            on_proc = _run_state_cmd(
                label="on", ids=ids, env=on_env, cwd=on_cwd, gpu=gpu_on,
                out_json=on_json, max_steps_cap=args.max_steps_cap,
                mem_step_scale=args.mem_step_scale,
            )
            off = _await(off_proc)
            on = _await(on_proc)
        else:
            off = _await(off_proc)
            on_proc = _run_state_cmd(
                label="on", ids=ids, env=on_env, cwd=on_cwd, gpu=gpu_on,
                out_json=on_json, max_steps_cap=args.max_steps_cap,
                mem_step_scale=args.mem_step_scale,
            )
            on = _await(on_proc)
    finally:
        if base_wt:
            _remove_base_worktree(base_wt)

    rc, predicted = _report(off, on, id_cluster,
                            off_label=off_label, on_label=on_label)
    print(f"[fast-gate] total wall {time.monotonic() - t_all:.0f}s. "
          f"predicted_full_delta={predicted:+d} exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
