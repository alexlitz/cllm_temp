#!/usr/bin/env python3
"""FLAG-ON REGRESSION GATE — the fast cross-cluster gate for the *campaign* config.

The gap this closes
-------------------
The existing byte-identity gates (``compare_symbolic_to_lowered_ffn``,
``tools/_isa_golden_hash.py``) verify only the **flag-OFF** model (the golden
35-token build). A change can be byte-identical OFF yet silently **regress a
whole cluster FLAG-ON** (the 30-token *campaign* config
``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``). This exact blind spot let a
mul ``l14`` fix pass golden byte-identity while crushing add/sub/div ~-60 in
the campaign config. There was NO fast structural gate for "does turning flag
X on (or applying branch Y) regress any cluster in the campaign config."

What this tool does
-------------------
Given a fix's KILL-SWITCH flag (``--flag C4_MY_FIX``) *or* a base commit
(``--base <sha>`` — HEAD vs base), it runs a REPRESENTATIVE per-cluster sample
(``tools/flag_regression_sample.json``: ~1-4 ids across ALL 56 clusters,
spanning passing + failing programs, reusing the gpu_tripwire sampling idea +
``tools/tripwire_baseline.json`` ids) in BOTH states (fix ON vs OFF) **within
the campaign config**, using the VALIDATED bit-exact ``cpu_full_trace`` verdict
(``--spec-k 0``, ``--workers 2``), and reports any cluster that goes
``ok -> fail`` (a REGRESSION — block) or ``fail -> ok`` (a flip — gain).

It is the automated form of the "cross-cluster verify" agents do by hand, on
CPU, in a few minutes. Exit code is non-zero on any real ok->fail regression.

The four memory-smoke clusters (the SI/LI store/load + the store-cell / load-cell
``var_*`` families: ``var_simple`` / ``var_mul`` / ``var_three`` /
``var_update``) are reported with an explicit ``[MEM-SMOKE]`` tag so a campaign
fix can never silently break the memory path.

Modes
-----
``--flag C4_MY_FIX``
    OFF state = campaign env with the fix flag UNSET. ON state = campaign env
    with ``C4_MY_FIX=1``. The fix's build code must consult the flag (e.g.
    ``os.environ.get("C4_MY_FIX")``) so flag-OFF rebuilds the pre-fix model.
    This is the clean, fast path (no git operations) and what the demo uses.

``--base <commit>``
    OFF state = the model built from ``<commit>``'s source (checked out into a
    throwaway ``git worktree`` so the working tree is never touched / stashed).
    ON state = the model built from the current HEAD working tree. Use this to
    gate a branch whose fix is NOT behind a single env flag.

Both states always run inside the campaign env
(``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``) unless ``--campaign-env`` is
overridden.

Memory discipline
-----------------
``--workers 2`` (cpu_full_trace default here) and a dedicated cache dir
(``C4_VM_CACHE_DIR=/tmp/c4cache_reggate`` by default) keep the bake count low
and avoid the multiprocessing thrash. Each state bakes the campaign model at
most once (warmed in the parent, workers ``torch.load``). The spawned
``cpu_full_trace`` subprocess is the only child; it is reaped before the gate
returns.

Runtime
-------
The CPU full_trace decode is ``O(steps^2)`` per program and ``--workers 2`` is
the memory-safe cap, so wall time scales with the number AND depth of the
sampled programs. With WARM ``off``/``on`` caches (re-runs / iterating) a
single arith cluster (``--clusters sub``, 4 ids) is ~6 min both-states
(measured ~355 s, validated regression demo); ``--clusters add,sub,div,mul``
(16 ids) is ~25 min. The full 56-cluster default sample (72 ids,
``--max-steps-cap 18`` skips the deep loop/gcd/rec band) is ~30-40 min at
``--workers 2`` — a CI/pre-merge gate, not an inner-loop tool. While ITERATING
on one fix, pass ``--clusters <the few you touched>`` for the few-minute path;
run the full sample once before landing. (The first run cold-bakes each state
~40 s; warm re-runs ``torch.load`` in ~2-3 s.)

Usage
-----
    # Gate a fix behind its kill-switch flag (the campaign cross-cluster check):
    python tools/flag_regression_gate.py --flag C4_MUL_BLK33_CLAWBACK

    # Gate HEAD vs a base commit (branch-level, no single flag):
    python tools/flag_regression_gate.py --base 8e37b4e1

    # Narrow to specific clusters / ids while iterating:
    python tools/flag_regression_gate.py --flag C4_MY_FIX --clusters add,sub,mul

    # Regenerate the per-cluster sample (after a test-suite change):
    python tools/flag_regression_gate.py --regen-sample

Tooling only: this script never imports a build path that writes weights other
than through the normal ``cpu_full_trace`` subprocess, so the golden model is
byte-identical (golden ``4958b35b`` unchanged).
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

_SAMPLE_PATH = os.path.join(_HERE, "flag_regression_sample.json")
_TRIPWIRE_PATH = os.path.join(_HERE, "tripwire_baseline.json")

# The campaign config: the 30-token frame the flag-ON gate must defend.
_CAMPAIGN_ENV = {"C4_NO_STACK0_EMIT": "1", "C4_OPERAND_FROM_MEMSP": "1"}

# The four memory-smoke clusters (SI/LI store-load + store/load-cell var_*
# families). Reported with an explicit MEM-SMOKE tag per the gate spec.
_MEM_SMOKE_CLUSTERS = {"var_simple", "var_mul", "var_three", "var_update"}

# Step-depth cap used when BUILDING the sample (so deep diverging clusters
# still contribute a tracked sentinel id) — matches the GPU gate's default.
_SAMPLE_BUILD_CAP = 40

# Step-depth cap the GATE RUN defaults to. Programs deeper than this are
# decoded as 'skipped' (free, still tracked) rather than fully O(steps^2)
# decoded, so the gate stays a few-minute CPU tripwire. A program deeper than
# 18 steps (var_three / func_max-min / nested / deep loops/rec/gcd) is skipped
# by default; pass ``--max-steps-cap 40 --clusters <c>`` to deep-verify one.
_GATE_RUN_CAP = 18


# ---------------------------------------------------------------------------
# Cluster mapping (identical to run_1096_canonical.cluster_of).
# ---------------------------------------------------------------------------
def _cluster_of(description: str) -> str:
    base = description.split(":", 1)[0].strip()
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


# ---------------------------------------------------------------------------
# Per-cluster sample selection.
# ---------------------------------------------------------------------------
def _build_sample() -> Dict[str, object]:
    """(Re)build the per-cluster sample json from the live test suite.

    One CHEAPEST representative per cluster across ALL 56 clusters, EXCEPT:
      * the arith demo clusters (add/sub/mul/div/mod) get 4 each (cheap 5-step
        programs — these are the regression-detection demo targets);
      * ``var_simple`` (the cheapest memory-smoke cluster) gets 2.
    The four memory-smoke ``var_*`` clusters are always present (1+ id) so a
    campaign fix can never silently break the SI/LI store-load path. Prefers
    ``tripwire_baseline.json`` ids (they already span pass+fail). A cluster
    whose cheapest member exceeds the gate run cap is still tracked (its id is
    decoded as 'skipped' by default, runnable via ``--max-steps-cap 40``).
    """
    from tests.test_suite_1000 import generate_test_programs
    from tools.run_1096_fast import _compile_and_oracle

    progs = generate_test_programs()
    clusters: "collections.OrderedDict[str, List[int]]" = collections.OrderedDict()
    for i, (_src, _exp, desc) in enumerate(progs):
        clusters.setdefault(_cluster_of(desc), []).append(i)

    sel = [(i, progs[i][0], progs[i][1], progs[i][2]) for i in range(len(progs))]
    prepared, _errs = _compile_and_oracle(sel)
    steps = {e[0]: e[4] for e in prepared}

    trip_set: set = set()
    if os.path.exists(_TRIPWIRE_PATH):
        trip = json.load(open(_TRIPWIRE_PATH))
        trip_set = {int(k) for k in trip.get("expected", {})}

    hot = {"add", "sub", "mul", "div", "mod"}  # arith demo clusters: 4 each

    def n_for(cluster: str) -> int:
        if cluster in hot:
            return 4
        if cluster == "var_simple":
            return 2
        return 1

    sample: Dict[str, List[int]] = {}
    for cluster, ids in clusters.items():
        n = n_for(cluster)
        # Cheapest first; among equal-cost prefer a tripwire id (spans pass+fail).
        order = sorted(
            ids, key=lambda x: ((steps.get(x) or 0), 0 if x in trip_set else 1, x)
        )
        sample[cluster] = sorted(order[:n])

    all_ids = sorted(i for v in sample.values() for i in v)
    return {
        "note": (
            "flag_regression_gate per-cluster sample (1 representative per "
            "cluster, 4 for the arith demo clusters add/sub/mul/div/mod, 2 for "
            "var_simple). The four memory-smoke var_* clusters "
            "(var_simple/var_mul/var_three/var_update) are always included so a "
            "campaign fix cannot silently break the SI/LI store-load path. "
            "Cheapest member per cluster, preferring tripwire_baseline.json ids "
            "(which span pass+fail). Deep clusters whose cheapest member exceeds "
            "the gate --max-steps-cap (default 18) are decoded as SKIPPED (free, "
            "still tracked); pass --max-steps-cap 40 --clusters <c> to "
            "deep-verify one. Regenerate with "
            "`python tools/flag_regression_gate.py --regen-sample`."
        ),
        "sample_build_cap": _SAMPLE_BUILD_CAP,
        "gate_default_max_steps_cap": _GATE_RUN_CAP,
        "per_cluster": sample,
        "ids": all_ids,
    }


def _load_sample(clusters_filter: Optional[List[str]]) -> Tuple[List[int], Dict[int, str]]:
    """Load the sample ids + an id->cluster map (filtered to ``clusters_filter``)."""
    if not os.path.exists(_SAMPLE_PATH):
        print(f"[reg-gate] sample missing — generating {_SAMPLE_PATH} ...",
              file=sys.stderr, flush=True)
        spec = _build_sample()
        json.dump(spec, open(_SAMPLE_PATH, "w"), indent=1)
    else:
        spec = json.load(open(_SAMPLE_PATH))

    per_cluster = spec["per_cluster"]
    if clusters_filter:
        want = set(clusters_filter)
        per_cluster = {c: v for c, v in per_cluster.items() if c in want}
        missing = want - set(per_cluster)
        if missing:
            print(f"[reg-gate] WARNING: unknown clusters ignored: {sorted(missing)}",
                  file=sys.stderr, flush=True)

    id_cluster: Dict[int, str] = {}
    for cluster, ids in per_cluster.items():
        for i in ids:
            id_cluster[i] = cluster
    return sorted(id_cluster), id_cluster


# ---------------------------------------------------------------------------
# Running one state via the cpu_full_trace subprocess.
# ---------------------------------------------------------------------------
def _run_state(
    *,
    label: str,
    ids: List[int],
    env: Dict[str, str],
    workers: int,
    cwd: str,
    max_steps_cap: int,
) -> Dict[int, str]:
    """Run cpu_full_trace for ``ids`` in ``env``/``cwd``; return id -> status.

    The subprocess is the ONLY child; it is fully awaited (and so reaped)
    before this returns. On a non-zero rc we surface stderr and abort the gate
    (a build failure in one state is itself a hard regression signal).
    """
    out_json = os.path.join(
        tempfile.gettempdir(), f"reggate_{label}_{os.getpid()}.json"
    )
    idstr = ",".join(map(str, ids))
    full_env = dict(os.environ)
    full_env.update(env)
    cmd = [
        sys.executable,
        os.path.join(cwd, "tools", "cpu_full_trace.py"),
        "--ids", idstr,
        "--spec-k", "0",
        "--workers", str(workers),
        "--max-steps-cap", str(max_steps_cap),
        "--criterion", "full_trace",
        "--output", out_json,
    ]
    print(f"\n[reg-gate] === state {label.upper()} === "
          f"({len(ids)} ids, workers={workers}, "
          f"cache={full_env.get('C4_VM_CACHE_DIR')}, cwd={cwd})", flush=True)
    print(f"[reg-gate]   campaign+fix env: "
          f"{ {k: env[k] for k in sorted(env)} }", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=full_env, cwd=cwd)
    secs = time.monotonic() - t0
    if proc.returncode != 0:
        print(f"[reg-gate] FATAL: cpu_full_trace ({label}) exited "
              f"{proc.returncode} — cannot produce a verdict for this state.",
              file=sys.stderr, flush=True)
        raise SystemExit(3)
    data = json.load(open(out_json))
    status = {r["idx"]: r["status"] for r in data["results"]}
    n_pass = sum(1 for v in status.values() if v == "ok")
    print(f"[reg-gate]   state {label} done in {secs:.0f}s "
          f"(pass={n_pass}/{len(status)}, skipped="
          f"{sum(1 for v in status.values() if v == 'skipped')}).", flush=True)
    try:
        os.remove(out_json)
    except OSError:
        pass
    return status


# ---------------------------------------------------------------------------
# Base-commit checkout (git-worktree, never touches the working tree).
# ---------------------------------------------------------------------------
def _git_root() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=_PKG
    ).decode().strip()


def _add_base_worktree(base: str) -> Tuple[str, str]:
    """Create a throwaway git worktree at ``base``; return (worktree_root, pkg_cwd).

    Uses ``git worktree add`` (NOT stash) so the live working tree is untouched.
    The caller MUST call ``_remove_base_worktree`` to clean up.
    """
    root = _git_root()
    wt = tempfile.mkdtemp(prefix="reggate_base_")
    print(f"[reg-gate] checking out base {base} into worktree {wt} "
          f"(working tree untouched, no stash) ...", flush=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt, base],
        cwd=root, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    # The package dir inside the worktree mirrors this repo's layout
    # (c4_release/ under the git root, or the root itself if it IS the package).
    rel = os.path.relpath(_PKG, root)
    pkg_cwd = os.path.normpath(os.path.join(wt, rel))
    if not os.path.isdir(os.path.join(pkg_cwd, "tools")):
        pkg_cwd = wt  # fallback: package == git root
    return wt, pkg_cwd


def _remove_base_worktree(wt: str) -> None:
    root = _git_root()
    print(f"[reg-gate] removing base worktree {wt} ...", flush=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    shutil.rmtree(wt, ignore_errors=True)


# ---------------------------------------------------------------------------
# Verdict diff + report.
# ---------------------------------------------------------------------------
def _report(
    off: Dict[int, str],
    on: Dict[int, str],
    id_cluster: Dict[int, str],
    *,
    off_label: str,
    on_label: str,
) -> int:
    """Diff OFF->ON per id; print the per-cluster report; return rc (1 if reg)."""
    regressions: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    flips: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    for i, cluster in id_cluster.items():
        a, b = off.get(i, "MISSING"), on.get(i, "MISSING")
        if a == "ok" and b != "ok":
            regressions[cluster].append((i, b))
        elif a != "ok" and b == "ok":
            flips[cluster].append((i, a))

    n_reg = sum(len(v) for v in regressions.values())
    n_flip = sum(len(v) for v in flips.values())

    def _tag(c: str) -> str:
        return "  [MEM-SMOKE]" if c in _MEM_SMOKE_CLUSTERS else ""

    print("\n" + "=" * 78)
    print(f"FLAG-ON REGRESSION GATE  ({off_label}  ->  {on_label})  campaign config")
    print("=" * 78)

    print(f"\nREGRESSIONS (ok -> fail) [{n_reg}]  -- these BLOCK the change:")
    if not regressions:
        print("    (none)")
    for c in sorted(regressions):
        print(f"    {c}{_tag(c)}: {regressions[c]}")

    print(f"\nFLIPS (fail -> ok) [{n_flip}]  -- gains from the change:")
    if not flips:
        print("    (none)")
    for c in sorted(flips):
        print(f"    {c}{_tag(c)}: {len(flips[c])} {flips[c]}")

    # Explicit memory-smoke summary line (always printed, even if clean).
    mem_reg = {c: regressions[c] for c in _MEM_SMOKE_CLUSTERS if c in regressions}
    print(f"\nMEM-SMOKE clusters (var_simple/var_mul/var_three/var_update): "
          f"{'REGRESSED ' + str(mem_reg) if mem_reg else 'clean'}")

    print(f"\nnet on sample: {n_flip - n_reg:+d}  "
          f"({'CLEAN — no cluster regressed' if n_reg == 0 else 'REGRESSION — do NOT land'})")
    print("=" * 78, flush=True)
    return 1 if n_reg else 0


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
                           "state leaves it unset; ON state sets it to 1 (or "
                           "--flag-on-value). Both inside the campaign env.")
    mode.add_argument("--base",
                      help="compare HEAD's working tree (ON) against this base "
                           "commit (OFF), via a throwaway git worktree (no "
                           "stash). Use for branch-level fixes not behind a flag.")
    mode.add_argument("--regen-sample", action="store_true",
                      help="regenerate tools/flag_regression_sample.json from "
                           "the live test suite and exit.")
    ap.add_argument("--flag-on-value", default="1",
                    help="value to set --flag to in the ON state (default '1').")
    ap.add_argument("--clusters",
                    help="comma-separated cluster names to restrict to (e.g. "
                         "add,sub,mul). Default: all 56.")
    ap.add_argument("--workers", type=int, default=2,
                    help="CPU workers per cpu_full_trace run (default 2 — the "
                         "memory-safe cap; do NOT raise on a shared machine).")
    ap.add_argument("--max-steps-cap", type=int, default=_GATE_RUN_CAP,
                    help=f"decode programs deeper than this as 'skipped' (free, "
                         f"still tracked) instead of full O(steps^2) decode. "
                         f"Default {_GATE_RUN_CAP} keeps the gate a few-minute "
                         f"CPU tripwire; raise to 40 (the GPU-gate cap) to "
                         f"deep-verify a specific cluster.")
    ap.add_argument("--cache-dir", default="/tmp/c4cache_reggate",
                    help="C4_VM_CACHE_DIR for the builds (default "
                         "/tmp/c4cache_reggate — a dedicated dir per the memory "
                         "discipline).")
    ap.add_argument("--campaign-env", default="C4_NO_STACK0_EMIT=1,C4_OPERAND_FROM_MEMSP=1",
                    help="comma-separated K=V campaign env both states run in "
                         "(default the 30-token campaign config).")
    args = ap.parse_args(argv)

    if args.regen_sample:
        spec = _build_sample()
        json.dump(spec, open(_SAMPLE_PATH, "w"), indent=1)
        print(f"[reg-gate] wrote {_SAMPLE_PATH}: "
              f"{len(spec['ids'])} ids across {len(spec['per_cluster'])} clusters.")
        return 0

    if not (args.flag or args.base):
        ap.error("one of --flag / --base / --regen-sample is required")

    clusters_filter = (
        [c.strip() for c in args.clusters.split(",") if c.strip()]
        if args.clusters else None
    )
    ids, id_cluster = _load_sample(clusters_filter)
    if not ids:
        print("[reg-gate] no ids selected — nothing to do.", file=sys.stderr)
        return 2

    campaign = dict(_CAMPAIGN_ENV)
    if args.campaign_env:
        campaign = {}
        for kv in args.campaign_env.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                campaign[k.strip()] = v.strip()

    base_cache = os.path.abspath(args.cache_dir)
    os.makedirs(base_cache, exist_ok=True)

    print(f"[reg-gate] sample: {len(ids)} ids across "
          f"{len(set(id_cluster.values()))} cluster(s).")
    print(f"[reg-gate] campaign env: { {k: campaign[k] for k in sorted(campaign)} }")

    t_all = time.monotonic()
    base_wt = None
    try:
        if args.flag:
            # OFF: campaign env, fix flag UNSET. ON: campaign env + flag=value.
            # Separate cache dirs so the two models never collide on disk.
            off_env = dict(campaign)
            off_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "off")
            off_env.pop(args.flag, None)
            on_env = dict(campaign)
            on_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "on")
            on_env[args.flag] = args.flag_on_value
            # Make sure a stray flag in the parent env can't pollute OFF.
            off_env.setdefault(args.flag, "0")

            off_label = f"OFF (no {args.flag})"
            on_label = f"ON ({args.flag}={args.flag_on_value})"
            off = _run_state(label="off", ids=ids, env=off_env,
                             workers=args.workers, cwd=_PKG,
                             max_steps_cap=args.max_steps_cap)
            on = _run_state(label="on", ids=ids, env=on_env,
                            workers=args.workers, cwd=_PKG,
                            max_steps_cap=args.max_steps_cap)
        else:
            # --base: OFF = base worktree source, ON = HEAD working tree.
            base_wt, base_cwd = _add_base_worktree(args.base)
            off_env = dict(campaign)
            off_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "base")
            on_env = dict(campaign)
            on_env["C4_VM_CACHE_DIR"] = os.path.join(base_cache, "head")
            off_label = f"OFF (base {args.base[:8]})"
            on_label = "ON (HEAD)"
            off = _run_state(label="base", ids=ids, env=off_env,
                             workers=args.workers, cwd=base_cwd,
                             max_steps_cap=args.max_steps_cap)
            on = _run_state(label="head", ids=ids, env=on_env,
                            workers=args.workers, cwd=_PKG,
                            max_steps_cap=args.max_steps_cap)
    finally:
        if base_wt:
            _remove_base_worktree(base_wt)

    rc = _report(off, on, id_cluster, off_label=off_label, on_label=on_label)
    print(f"[reg-gate] total wall {time.monotonic() - t_all:.0f}s. "
          f"exit={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
