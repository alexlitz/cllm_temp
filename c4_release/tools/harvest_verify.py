#!/usr/bin/env python3
"""HARVEST VERIFY — the automated COMBINED-regression gate.

WHY THIS EXISTS
---------------
This is the tool that would have saved the round we lost to the var_mul TRAP:
three var_mul fixes each measured "+N" in ISOLATION under fleet contention, but
COMBINED + uncontended they were ALL net-negative (var_mul 9 -> 0, cmp
regressed). The truth was only recovered by a manual bisect — toggling each flag
and re-measuring per cluster by hand — which cost hours.

``harvest_verify.py`` automates that bisect's verification step: given a SET of
flags (or a base commit vs HEAD) and an id set, it runs the affected programs in
BOTH states, UNCONTENDED, with the same degenerate-build protections as
``flip_measure.py``, and prints exactly which clusters go ``ok -> fail``
(a COMBINED regression — non-zero exit, BLOCK) or ``fail -> ok`` (a gain). It is
the GPU ``full_trace`` counterpart to the CPU ``flag_regression_gate.py``: this
one drives the production ``run_1096_canonical`` batched-neural path on GPU0, so
the verdict is the exact one the campaign is scored on.

THE THREE PROTECTIONS (all inherited from flip_measure.py)
----------------------------------------------------------
1. CANARY GATE — before trusting any sweep number for a state, build that
   state's model and run four guaranteed-pass ids (0=add, 250=var_simple,
   550=func_identity, 875=expr_mod). If not 4/4 the build is degenerate and we
   abort with "BUILD CORRUPT, INVALID" (exit 2) — NEVER report a fake number off
   a starved build. (This is precisely how the contention traps manifested:
   pass=0 across clusters that pass 25-50/50 in isolation.)
2. MEMORY HEADROOM — wait for ``free -g >= --min-free-gb`` before each build, so
   a build is never started under the starvation that silently corrupts weights.
3. DEDICATED FRESH CACHE PER STATE — the disk-cache key is NOT flag-aware (it
   hashes source + kwargs, not the campaign env flags), so two flag states
   collide on the key and hand back the wrong model. Each state gets its own
   freshly-cleared ``C4_VM_CACHE_DIR`` subdir.

MODES
-----
``--flags C4_FOO,C4_BAR``
    State A = campaign env with EVERY listed flag forced OFF (``=0``).
    State B = campaign env with EVERY listed flag set ON (``=1``).
    Use this to verify a SET of fixes COMBINED (the var_mul trap): does turning
    the whole set on, together, regress any cluster vs the whole set off?

``--flags ... --base <commit>``
    State A = the model built from ``<commit>``'s source (a throwaway
    ``git worktree`` — the live working tree is NEVER touched, NEVER stashed),
    with the listed flags applied. State B = HEAD's working tree with the same
    flags. Use this to gate a BRANCH whose combined fix is not behind one flag.
    (``--flags`` may be empty here to compare base-vs-HEAD purely structurally.)

Both states always run inside the campaign env
(``C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1``) unless ``--campaign-env`` is
overridden (pass ``--campaign-env ""`` to score the plain golden 35-token frame).

GATES / DEMO
------------
    # The campaign si/li discriminator's effect on var_simple (should be clean):
    python tools/harvest_verify.py --flags C4_SILI_CAM_B1 --ids "250-254"

    # Verify a SET of fixes COMBINED over the clusters they touch:
    python tools/harvest_verify.py --flags C4_FOO,C4_BAR --ids "275-299,100-149"

    # Gate a branch (HEAD) against a base commit, flags applied to both:
    python tools/harvest_verify.py --flags C4_FOO --ids "0-99" --base 524899c9

Exit codes: 0 = both builds sound AND no cluster regressed (gains allowed);
1 = a real ``ok -> fail`` COMBINED regression (BLOCK — do NOT land the set);
2 = a build was degenerate (a canary failed) — the number is INVALID;
3 = a runner subprocess crashed.

GPU full_trace, ``--workers 1`` (one canonical run at a time), GPU0 only by
default. Tooling-only: no model bake side effects (it only spawns
``run_1096_canonical`` subprocesses), so the golden model is byte-identical.
"""
from __future__ import annotations

import argparse
import collections
import os
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

# Reuse flip_measure's hardened helpers + canary contract verbatim — this tool
# is the multi-flag / base-vs-HEAD generalization of the same gate.
from tools.flip_measure import (  # noqa: E402
    CANARY_IDS,
    CANARY_LABEL,
    _free_gb,
    _run_canonical,
    _wait_for_memory,
)

# The campaign config the gate defends by default (the 30-token frame). Both
# states run inside it so a flag's effect is measured where it is meant to act.
_CAMPAIGN_ENV = {"C4_NO_STACK0_EMIT": "1", "C4_OPERAND_FROM_MEMSP": "1"}

# The four memory-smoke clusters (SI/LI store-load + store/load-cell var_*
# families). Tagged in the report so a fix can never silently break the path.
_MEM_SMOKE_CLUSTERS = {"var_simple", "var_mul", "var_three", "var_update"}


# ---------------------------------------------------------------------------
# GPU-VRAM headroom — the OTHER contention failure mode flip_measure can't see.
# flip_measure._wait_for_memory only watches HOST RAM (MemAvailable); but a GPU
# full_trace run is killed by a CO-TENANT ballooning GPU0 mid-build/sweep
# (observed: a fleet job grabbing 12.79 GiB left 154 MiB free → CUDA OOM). So we
# also block on free VRAM of the target GPU before each build, by the same
# wait-then-warn contract as the host-RAM gate.
# ---------------------------------------------------------------------------
def _gpu_free_mib(gpu: str) -> Optional[float]:
    """Free VRAM (MiB) on physical GPU index ``gpu`` via nvidia-smi, or None if
    nvidia-smi is unavailable (then the VRAM gate is a no-op)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        if out:
            return float(out[0].strip())
    except Exception:
        pass
    return None


def _wait_for_gpu(gpu: str, min_free_mib: float, timeout_s: float) -> None:
    """Block until the target GPU has >= ``min_free_mib`` free VRAM (or timeout).

    Same proceed-with-a-warning contract as the host-RAM gate: this is a
    contention guard, not a hard precondition (the canary then catches any build
    that DID go degenerate). A no-op when nvidia-smi is unavailable."""
    if _gpu_free_mib(gpu) is None:
        print(f"[harvest] (no nvidia-smi for gpu {gpu} — skipping VRAM gate)",
              flush=True)
        return
    t0 = time.monotonic()
    while True:
        free = _gpu_free_mib(gpu)
        if free is None or free >= min_free_mib:
            print(f"[harvest] GPU{gpu} VRAM OK: {0.0 if free is None else free:.0f}MiB "
                  f"free (>= {min_free_mib:.0f}MiB)", flush=True)
            return
        waited = time.monotonic() - t0
        if waited >= timeout_s:
            print(f"[harvest] WARNING: GPU{gpu} only {free:.0f}MiB free < "
                  f"{min_free_mib:.0f}MiB after {waited:.0f}s — proceeding anyway "
                  f"(sweep may OOM under contention; canary still gates the build).",
                  flush=True)
            return
        print(f"[harvest] waiting for GPU{gpu} VRAM: {free:.0f}MiB free < "
              f"{min_free_mib:.0f}MiB ({waited:.0f}/{timeout_s:.0f}s) — fleet "
              f"contention; UNCONTENDED is required for a trustworthy number",
              flush=True)
        time.sleep(20)


# ---------------------------------------------------------------------------
# id-set + cluster helpers (reuse the canonical runner's exact mapping).
# ---------------------------------------------------------------------------
def _parse_ids(spec: str) -> List[int]:
    from tools.run_1096_fast import _parse_ids as _p
    return _p(spec)


def _cluster_map(ids: List[int]) -> Dict[int, str]:
    """id -> cluster name, using run_1096_canonical.cluster_of (the SAME
    clustering the campaign is scored by)."""
    from tools.run_1096_canonical import cluster_of
    from tests.test_suite_1000 import generate_test_programs

    progs = generate_test_programs()
    out: Dict[int, str] = {}
    for i in ids:
        if 0 <= i < len(progs):
            out[i] = cluster_of(progs[i][2])
        else:
            out[i] = f"id{i}"
    return out


# ---------------------------------------------------------------------------
# git-worktree base checkout (never touches / stashes the live working tree).
# ---------------------------------------------------------------------------
def _git_root() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=_PKG
    ).decode().strip()


def _add_base_worktree(base: str) -> Tuple[str, str]:
    """Create a throwaway detached worktree at ``base``; return (wt_root, pkg_cwd).

    Uses ``git worktree add`` (NOT stash) per the workflow constraint, so the
    live working tree is untouched. Caller MUST ``_remove_base_worktree``."""
    root = _git_root()
    wt = tempfile.mkdtemp(prefix="harvest_base_")
    print(f"[harvest] checking out base {base} into worktree {wt} "
          f"(working tree untouched, NO stash) ...", flush=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", wt, base],
        cwd=root, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    rel = os.path.relpath(_PKG, root)
    pkg_cwd = os.path.normpath(os.path.join(wt, rel))
    if not os.path.isdir(os.path.join(pkg_cwd, "tools")):
        pkg_cwd = wt  # fallback: package == git root
    return wt, pkg_cwd


def _remove_base_worktree(wt: str) -> None:
    root = _git_root()
    print(f"[harvest] removing base worktree {wt} ...", flush=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", wt],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    shutil.rmtree(wt, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run ONE state: wait for memory, fresh cache, CANARY gate, then the sweep.
# ---------------------------------------------------------------------------
def _run_state(
    *,
    label: str,
    ids: str,
    env: Dict[str, str],
    cwd: str,
    cache_dir: str,
    gpu: str,
    chunk: int,
    min_free_gb: float,
    min_free_mib_gpu: float,
    mem_timeout: float,
    skip_canary: bool,
    retries: int,
) -> Dict[int, str]:
    """Build + canary-gate + sweep one state. Returns {idx: status}.

    A subprocess CRASH (e.g. a co-tenant ballooning GPU0 → CUDA OOM during the
    build window, the exact fleet race observed) is RETRIED up to ``retries``
    times, re-waiting for VRAM headroom first — a crash is transient contention,
    NOT a verdict, so we never let it abort or (worse) fake a number. A canary
    FAILURE (a built-but-degenerate model) is NOT retried: it is a hard
    ``SystemExit(2)`` (INVALID) because re-running an identically-starved build
    won't fix it. Exhausting the retries on crashes → ``SystemExit(3)``.
    """
    full_env = dict(os.environ)
    full_env.update(env)
    full_env["CUDA_VISIBLE_DEVICES"] = gpu
    full_env["C4_VM_CACHE_DIR"] = cache_dir
    full_env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir + "_inductor"

    print("\n" + "-" * 78)
    print(f"[harvest] === STATE {label.upper()} ===  cwd={cwd}")
    flagview = {k: env[k] for k in sorted(env)}
    print(f"[harvest]   env: {flagview}")
    print(f"[harvest]   cache: {cache_dir}  gpu={gpu}")
    print("-" * 78, flush=True)

    last_exc: Optional[str] = None
    wipe_cache = True  # wipe on attempt 1; on a transient-OOM retry the warm
    #                    (sound, canary-verified) build is REUSED → cheap re-run.
    for attempt in range(1, retries + 2):
        if attempt > 1:
            print(f"[harvest]   state {label}: RETRY {attempt - 1}/{retries} "
                  f"after a transient OOM (re-waiting for VRAM"
                  f"{'' if wipe_cache else ', reusing the sound warm build'}) ...",
                  flush=True)
        if wipe_cache:
            # Fresh, flag-consistent caches — never reuse a possibly-corrupt
            # build, and never collide with the OTHER state's (the disk-cache key
            # is NOT flag-aware). A crashed half-build is discarded by this.
            shutil.rmtree(cache_dir, ignore_errors=True)
            shutil.rmtree(cache_dir + "_inductor", ignore_errors=True)

        # Don't start a build under starvation (THE silent-corruption mode).
        # Both gates: host RAM (degenerate weights) AND GPU VRAM (CUDA OOM).
        _wait_for_memory(min_free_gb, mem_timeout)
        _wait_for_gpu(gpu, min_free_mib_gpu, mem_timeout)

        try:
            # ---- CANARY GATE (builds + caches this state's model) -----------
            if not skip_canary:
                canary_ids = ",".join(str(i) for i in CANARY_IDS)
                out = os.path.join(
                    tempfile.gettempdir(),
                    f"harvest_canary_{label}_{os.getpid()}.json",
                )
                st = _run_canonical(canary_ids, out, full_env,
                                    chunk=len(CANARY_IDS), tag=f"CANARY[{label}]")
                for i in CANARY_IDS:
                    print(f"[harvest]   canary {CANARY_LABEL[i]:14s} (id {i}): "
                          f"{st.get(i, 'MISSING')}")
                # An 'error' canary id is a transient OOM (no real verdict) →
                # retry. A 'fail' (or MISSING) is a degenerate BUILD → INVALID.
                canary_errored = [i for i in CANARY_IDS if st.get(i) == "error"]
                if canary_errored:
                    last_exc = (f"canary ids {canary_errored} OOM'd (no verdict)")
                    print(f"[harvest]   canary {[CANARY_LABEL[i] for i in canary_errored]}"
                          f" returned 'error' (transient OOM) — attempt {attempt}/"
                          f"{retries + 1} UNTRUSTWORTHY, retrying.", flush=True)
                    # Canary OOM'd before fully verifying the build → rebuild fresh.
                    wipe_cache = True
                    continue
                bad = [i for i in CANARY_IDS if st.get(i) != "ok"]
                if bad:
                    print("=" * 78)
                    print(f"❌ BUILD CORRUPT, INVALID — state {label}: canary "
                          f"FAILED for {[CANARY_LABEL[i] for i in bad]}.")
                    print("   A sound build passes all four. This is a degenerate "
                          "BUILD (not transient contention) — re-run in a quieter "
                          "window / raise --min-free-gb. Number INVALID.")
                    print("=" * 78, flush=True)
                    raise SystemExit(2)
                print(f"[harvest]   ✅ canary PASSED ({label}) — build sound, "
                      f"sweeping.", flush=True)

            # ---- FULL SWEEP (cache-hits the canary-verified build) ----------
            out = os.path.join(
                tempfile.gettempdir(), f"harvest_sweep_{label}_{os.getpid()}.json"
            )
            st = _run_canonical(ids, out, full_env, chunk=chunk,
                                tag=f"SWEEP[{label}]")

            # A program with status 'error' did NOT get a real verdict — it
            # crashed (almost always a solo CUDA-OOM under fleet contention,
            # "too deep for GPU even at B=1"). An 'error' is NOT a 'fail': counting
            # it as one would fabricate a spurious ok->fail / fail->ok in the diff.
            # So any 'error' in the sweep makes this attempt UNTRUSTWORTHY → retry
            # (re-wait for VRAM), exactly like a hard subprocess crash. This is
            # transient contention, never a verdict.
            errored = sorted(i for i, s in st.items() if s == "error")
            if errored:
                last_exc = (f"{len(errored)} program(s) returned 'error' "
                            f"(transient OOM under contention): {errored[:8]}"
                            f"{' ...' if len(errored) > 8 else ''}")
                print(f"[harvest]   state {label}: sweep had {len(errored)} "
                      f"'error' result(s) (transient OOM, not a verdict) on ids "
                      f"{errored[:8]} — attempt {attempt}/{retries + 1} "
                      f"UNTRUSTWORTHY, retrying.", flush=True)
                # The canary already verified the build is SOUND — only the
                # forward OOM'd. Keep the warm build; just re-run the sweep.
                wipe_cache = False
                continue

            # Post-hoc degenerate-build re-check: a canary id inside the sweep
            # that got a real 'fail' (NOT an 'error') means the build went
            # degenerate where the standalone canary did not — INVALID, NOT a
            # transient OOM, so do not retry (the same build will re-fail).
            sweep_canary_failed = [i for i in CANARY_IDS
                                   if st.get(i) == "fail"]
            if sweep_canary_failed:
                print("=" * 78)
                print(f"❌ BUILD CORRUPT, INVALID — state {label}: canary ids "
                      f"{[CANARY_LABEL[i] for i in sweep_canary_failed]} got a "
                      f"real FAIL verdict in the sweep (build degenerate mid-run, "
                      f"not a transient OOM).")
                print("=" * 78, flush=True)
                raise SystemExit(2)
            return st

        except RuntimeError as exc:
            # A runner subprocess crash (CUDA OOM under contention, etc.) — this
            # is transient, NOT a verdict. Retry after re-waiting for VRAM. The
            # crash may have died mid-bake → rebuild fresh next attempt.
            last_exc = str(exc)
            wipe_cache = True
            print(f"[harvest]   state {label}: runner crashed (attempt {attempt}/"
                  f"{retries + 1}) — likely fleet contention. Tail:\n"
                  f"{last_exc.splitlines()[-1] if last_exc else ''}", flush=True)
            continue

    print(f"[harvest] FATAL: state {label} crashed on all {retries + 1} attempts "
          f"(persistent contention — GPU0 never stayed free through a build). "
          f"Last error:\n{last_exc}", file=sys.stderr, flush=True)
    raise SystemExit(3)


# ---------------------------------------------------------------------------
# Diff + report.
# ---------------------------------------------------------------------------
def _report(
    a: Dict[int, str],
    b: Dict[int, str],
    id_cluster: Dict[int, str],
    *,
    a_label: str,
    b_label: str,
) -> int:
    """Diff state A -> state B per id, per cluster; print BLOCK/gain tables.

    Returns rc: 1 if ANY cluster regressed (ok -> fail), else 0.
    """
    regressions: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    gains: "collections.defaultdict[str, list]" = collections.defaultdict(list)
    # Per-cluster pass counts for a net-delta summary.
    a_pass: "collections.Counter[str]" = collections.Counter()
    b_pass: "collections.Counter[str]" = collections.Counter()

    for i, cluster in id_cluster.items():
        sa, sb = a.get(i, "MISSING"), b.get(i, "MISSING")
        if sa == "ok":
            a_pass[cluster] += 1
        if sb == "ok":
            b_pass[cluster] += 1
        if sa == "ok" and sb != "ok":
            regressions[cluster].append((i, sb))
        elif sa != "ok" and sb == "ok":
            gains[cluster].append((i, sa))

    n_reg = sum(len(v) for v in regressions.values())
    n_gain = sum(len(v) for v in gains.values())

    def _tag(c: str) -> str:
        return "  [MEM-SMOKE]" if c in _MEM_SMOKE_CLUSTERS else ""

    print("\n" + "=" * 78)
    print(f"HARVEST VERIFY  (A: {a_label}   ->   B: {b_label})")
    print("=" * 78)

    print(f"\nCOMBINED REGRESSIONS (ok -> fail) [{n_reg}]  -- these BLOCK the set:")
    if not regressions:
        print("    (none)")
    for c in sorted(regressions):
        print(f"    {c}{_tag(c)}: {regressions[c]}")

    print(f"\nGAINS (fail -> ok) [{n_gain}]:")
    if not gains:
        print("    (none)")
    for c in sorted(gains):
        print(f"    {c}{_tag(c)}: {len(gains[c])} {gains[c]}")

    # Per-cluster net delta (B pass - A pass) — the var_mul-trap signal: a
    # cluster that nets negative across the COMBINED set is the regression.
    print("\nPER-CLUSTER NET (B.pass - A.pass)  [< 0 = combined-negative]:")
    any_neg = False
    for c in sorted(set(id_cluster.values())):
        d = b_pass[c] - a_pass[c]
        if d != 0:
            mark = "  <== NEGATIVE" if d < 0 else ""
            if d < 0:
                any_neg = True
            print(f"    {c:18s}{_tag(c)}  A={a_pass[c]:3d}  B={b_pass[c]:3d}  "
                  f"net={d:+d}{mark}")
    if not any_neg and n_reg == 0:
        print("    (no cluster net-negative)")

    mem_reg = {c: regressions[c] for c in _MEM_SMOKE_CLUSTERS if c in regressions}
    print(f"\nMEM-SMOKE (var_simple/var_mul/var_three/var_update): "
          f"{'REGRESSED ' + str(mem_reg) if mem_reg else 'clean'}")

    print(f"\nnet on id set: {n_gain - n_reg:+d}  "
          f"({'CLEAN — no cluster regressed' if n_reg == 0 else 'REGRESSION — do NOT land this set'})")
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
    ap.add_argument("--flags", default="",
                    help="comma-separated campaign flags. State A forces every "
                         "flag OFF (=0); state B sets every flag ON (=1). May be "
                         "empty when --base is given (pure base-vs-HEAD).")
    ap.add_argument("--ids", required=True,
                    help="comma-separated ids/ranges to verify, e.g. "
                         "'275-299,100-149'. These are the clusters the set touches.")
    ap.add_argument("--base", default=None,
                    help="compare HEAD's working tree (state B) against this base "
                         "commit (state A), via a throwaway git worktree (no "
                         "stash). --flags are applied to BOTH states.")
    ap.add_argument("--flag-on-value", default="1",
                    help="value the ON state sets each flag to (default '1').")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES (default 0).")
    ap.add_argument("--chunk", type=int, default=0,
                    help="programs per neural batch (default 0 = memory-safe "
                         "length-aware bucketing in run_1096_canonical).")
    ap.add_argument("--min-free-gb", type=float, default=20.0,
                    help="wait for at least this much free RAM before each build "
                         "(degenerate builds under contention are THE failure mode).")
    ap.add_argument("--min-free-mib-gpu", type=float, default=8000.0,
                    help="wait for at least this much free VRAM (MiB) on the target "
                         "GPU before each build/sweep (default 8000 ~ one model + "
                         "forward headroom). Guards the CUDA-OOM-mid-run contention "
                         "the host-RAM gate cannot see. 0 disables the VRAM gate.")
    ap.add_argument("--mem-timeout", type=float, default=900.0,
                    help="give up waiting for memory after this many seconds "
                         "(then proceed with a warning).")
    ap.add_argument("--retries", type=int, default=3,
                    help="retry a state this many times on a TRANSIENT runner "
                         "crash (a co-tenant ballooning GPU0 → CUDA OOM during the "
                         "build window), re-waiting for VRAM first. A degenerate "
                         "BUILD (canary fail) is NOT retried — that is INVALID. "
                         "Default 3.")
    ap.add_argument("--cache-dir", default="/tmp/c4cache_hv",
                    help="base C4_VM_CACHE_DIR; each state gets a fresh subdir "
                         "(default /tmp/c4cache_hv per the brief).")
    ap.add_argument("--campaign-env", default="C4_NO_STACK0_EMIT=1,C4_OPERAND_FROM_MEMSP=1",
                    help="comma-separated K=V campaign env both states run in "
                         "(default the 30-token campaign config; pass '' for the "
                         "plain golden 35-token frame).")
    ap.add_argument("--skip-canary", action="store_true",
                    help="(debug only) skip the per-state canary gate. NEVER use "
                         "for a real verdict — a degenerate build will be reported "
                         "as a fake number.")
    args = ap.parse_args(argv)

    flags = [f.strip() for f in args.flags.split(",") if f.strip()]
    if not flags and not args.base:
        ap.error("nothing to compare: pass --flags and/or --base.")

    try:
        id_list = _parse_ids(args.ids)
    except ValueError as exc:
        ap.error(f"bad --ids: {exc}")
    if not id_list:
        ap.error("--ids selected no programs.")

    campaign: Dict[str, str] = {}
    if args.campaign_env.strip():
        for kv in args.campaign_env.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                campaign[k.strip()] = v.strip()

    base_cache = os.path.abspath(args.cache_dir)
    os.makedirs(base_cache, exist_ok=True)

    id_cluster = _cluster_map(id_list)
    n_clusters = len(set(id_cluster.values()))

    print("=" * 78)
    print("[harvest] HARVEST VERIFY — combined-regression gate (GPU full_trace)")
    print(f"[harvest] flags          : {flags or '(none)'}")
    print(f"[harvest] ids            : {args.ids}  ({len(id_list)} programs, "
          f"{n_clusters} clusters)")
    print(f"[harvest] campaign env   : {campaign or '(plain golden frame)'}")
    print(f"[harvest] mode           : "
          f"{'base ' + args.base[:8] + ' vs HEAD' if args.base else 'flags OFF vs ON'}")
    print(f"[harvest] gpu            : {args.gpu}   workers=1 (one run at a time)")
    print(f"[harvest] min-free-gb    : {args.min_free_gb}   "
          f"(now: {_free_gb():.1f}GB free)")
    print("=" * 78, flush=True)

    ids_str = args.ids
    t_all = time.monotonic()
    base_wt = None
    try:
        if args.base:
            # State A = base worktree source (flags applied); B = HEAD (flags).
            base_wt, base_cwd = _add_base_worktree(args.base)
            env_a = dict(campaign)
            env_b = dict(campaign)
            for f in flags:
                env_a[f] = args.flag_on_value
                env_b[f] = args.flag_on_value
            a_label = f"base {args.base[:8]}" + (f" +{','.join(flags)}" if flags else "")
            b_label = "HEAD" + (f" +{','.join(flags)}" if flags else "")
            a = _run_state(
                label="base", ids=ids_str, env=env_a, cwd=base_cwd,
                cache_dir=os.path.join(base_cache, "base"), gpu=args.gpu,
                chunk=args.chunk, min_free_gb=args.min_free_gb,
                min_free_mib_gpu=args.min_free_mib_gpu,
                mem_timeout=args.mem_timeout, skip_canary=args.skip_canary,
                retries=args.retries,
            )
            b = _run_state(
                label="head", ids=ids_str, env=env_b, cwd=_PKG,
                cache_dir=os.path.join(base_cache, "head"), gpu=args.gpu,
                chunk=args.chunk, min_free_gb=args.min_free_gb,
                min_free_mib_gpu=args.min_free_mib_gpu,
                mem_timeout=args.mem_timeout, skip_canary=args.skip_canary,
                retries=args.retries,
            )
        else:
            # State A = all flags OFF (=0); state B = all flags ON (=value).
            env_a = dict(campaign)
            env_b = dict(campaign)
            for f in flags:
                env_a[f] = "0"
                env_b[f] = args.flag_on_value
            a_label = "flags OFF (" + ",".join(f + "=0" for f in flags) + ")"
            b_label = "flags ON (" + ",".join(f + "=" + args.flag_on_value
                                              for f in flags) + ")"
            a = _run_state(
                label="off", ids=ids_str, env=env_a, cwd=_PKG,
                cache_dir=os.path.join(base_cache, "off"), gpu=args.gpu,
                chunk=args.chunk, min_free_gb=args.min_free_gb,
                min_free_mib_gpu=args.min_free_mib_gpu,
                mem_timeout=args.mem_timeout, skip_canary=args.skip_canary,
                retries=args.retries,
            )
            b = _run_state(
                label="on", ids=ids_str, env=env_b, cwd=_PKG,
                cache_dir=os.path.join(base_cache, "on"), gpu=args.gpu,
                chunk=args.chunk, min_free_gb=args.min_free_gb,
                min_free_mib_gpu=args.min_free_mib_gpu,
                mem_timeout=args.mem_timeout, skip_canary=args.skip_canary,
                retries=args.retries,
            )
    finally:
        if base_wt:
            _remove_base_worktree(base_wt)

    rc = _report(a, b, id_cluster, a_label=a_label, b_label=b_label)
    print(f"[harvest] total wall {time.monotonic() - t_all:.0f}s.  exit={rc}",
          flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
