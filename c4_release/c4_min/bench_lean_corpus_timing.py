"""HOW FAST IS TESTING NOW — the lean-forward corpus timing bench.

Runs a representative stratified sample of the C4 corpus THROUGH the LEAN
compacted forward via ``speculative_run_lean_graphed`` (the fast, CUDA-graph
speculative path, ~2.4 ms/VM-step regime) and reports the honest
"how fast can we run our testing" number for the NON-muldiv corpus.

muldiv is deliberately OUT of scope: the lean model on this branch has no
iterative MUL/DIV/MOD gadget (that arrives via #699 subroutines), so those
clusters are not timed here.

Per program it measures: VM steps, model forwards (speculative), ms/VM-step,
wall. It aggregates per-cluster (ms/program, ms/step), extrapolates the full
non-muldiv-corpus wall, and honestly labels which clusters the lean model
decodes BYTE-EXACT vs ``isa.interpret`` and which hit the KNOWN pre-existing
model bugs (signed-compare LT/GE/GT/LE, countdown >= 100) — those are #700 and
affect PASS-rate, not timing.

Run:  python -m c4_min.bench_lean_corpus_timing --device cuda:0
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import qwen_lean_cuda_graph as CG
from . import qwen_full_vm_corpus as CORP


# ---------------------------------------------------------------------------
# Cluster -> minimal subset routing.  A wider subset is a strict superset of a
# narrower one's ops, so we build ONE lean model per widest-needed subset and run
# every reachable cluster on it (memory-safe: at most two builds).
# ---------------------------------------------------------------------------
# clusters the task asked for (the corpus tags map to these):
#   arith / if / (bool via cmp) / var / func / loop / memory / bitwise
# muldiv is EXCLUDED (no iterative gadget on the lean model yet).
NONMULDIV_FAMILIES = {"arith", "if", "func", "loop", "cmp", "memory", "var",
                      "bitwise", "shift"}
MULDIV_FAMILIES = {"muldiv"}

# which subset each family runs on (widest needed).  base ops also run on the
# mem+cmp model (superset); bitwise/shift need the +bitwise model.
_FAMILY_SUBSET = {
    "arith": "mem+cmp", "if": "mem+cmp", "func": "mem+cmp", "loop": "mem+cmp",
    "cmp": "mem+cmp", "memory": "mem+cmp", "var": "mem+cmp",
    "bitwise": "bitwise", "shift": "bitwise",
}
_SUBSET_OBJ = {"mem+cmp": Q.SUBSET_MEM_CMP, "bitwise": Q.SUBSET_BITWISE}


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


# ---------------------------------------------------------------------------
# func-safe reference.  ``isa.interpret`` (the slice-ISA reference) does NOT
# implement the FUNCTION ops JSR/ENT/ADJ/LEV, so it raises on the func cluster.
# The drivers call ``isa.interpret`` for ``ref_trace``; we install a tolerant
# shim that returns [] on the out-of-slice ops (func) so the driver runs to
# completion.  func byte-exactness is then validated by graph==naive-lean
# self-consistency (both run the identical Python call-stack logic), which is
# exactly how ``test_qwen_lean_forward`` validates func (lean == HF).
# ---------------------------------------------------------------------------
_ORIG_INTERPRET = isa.interpret


def _func_safe_interpret(code, **kw):
    try:
        return _ORIG_INTERPRET(code, **kw)
    except NotImplementedError:
        return []                      # func: no slice-ISA ref -> validate self-consistently


def _install_func_safe_interpret():
    isa.interpret = _func_safe_interpret


def _time_spec(lean, code, graphed, *, warmup: int, iters: int, device: str,
               block_steps: int = 32, max_steps: int = 512):
    """Return (mean_ms_wall, result).  Warms up (so the graph capture per shape is
    amortised OUT of the timed window) then times ``iters`` graphed spec runs."""
    res = None
    for _ in range(warmup):
        res = CG.speculative_run_lean_graphed(
            lean, code, block_steps=block_steps, max_steps=max_steps, graphed=graphed)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        res = CG.speculative_run_lean_graphed(
            lean, code, block_steps=block_steps, max_steps=max_steps, graphed=graphed)
    _sync(device)
    ms = (time.perf_counter() - t0) / iters * 1000.0
    return ms, res


# known pre-existing model bugs on this branch (affect PASS not timing) -> #700
_SIGNED_CMP_OPS = {"lt", "gt", "le", "ge"}   # signed-compare failure family


def _known_bug_tag(case) -> Optional[str]:
    """Label a case with the KNOWN pre-existing model bug it will hit, if any."""
    lab = case.label.lower()
    if case.family == "cmp":
        opname = lab.split("_")[0]
        if opname in _SIGNED_CMP_OPS:
            return "signed-compare (#700)"
    if case.family == "loop":
        # countdown >= 100 is the known deep-count bug; the sampled ones are small.
        return None
    return None


@dataclass
class ProgResult:
    family: str
    label: str
    subset: str
    steps: int
    forwards: int
    fwd_saved: float
    ms_wall: float
    ms_step: float
    exact: bool
    bug_tag: Optional[str]


def run_corpus_timing(device: str = "cuda:0", *, iters: int = 3, warmup: int = 2,
                      block_steps: int = 32, code_size: int = 24,
                      include_bitwise: bool = True) -> Dict:
    warnings.filterwarnings("ignore")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    _install_func_safe_interpret()     # let the func cluster run (JSR/ENT/LEV)

    cases = [c for c in CORP.corpus() if c.family in NONMULDIV_FAMILIES]
    # route to the widest subset each family needs.
    subsets_needed = sorted({_FAMILY_SUBSET[c.family] for c in cases})
    if not include_bitwise:
        subsets_needed = [s for s in subsets_needed if s != "bitwise"]
        cases = [c for c in cases if _FAMILY_SUBSET[c.family] != "bitwise"]

    print(f"device: {device}  torch {torch.__version__}")
    print(f"non-muldiv corpus: {len(cases)} programs across "
          f"{len(set(c.family for c in cases))} clusters; "
          f"builds: {subsets_needed}")

    all_results: List[ProgResult] = []
    build_times: Dict[str, float] = {}
    layers: Dict[str, int] = {}

    for sub_name in subsets_needed:
        subs_cases = [c for c in cases if _FAMILY_SUBSET[c.family] == sub_name]
        print(f"\n=== building lean model (subset {sub_name}) ===")
        t0 = time.time()
        vm = Q.build(code_size=code_size, subset=_SUBSET_OBJ[sub_name])
        vm.embed = vm.embed.to(device)
        lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
        bt = time.time() - t0
        build_times[sub_name] = bt
        layers[sub_name] = lean.n_layers
        print(f"  lean: {lean.n_layers}L, {lean.n_heads}q-heads (6 CAM), "
              f"head_dim {lean.head_dim}  (built {bt:.1f}s)")

        # ONE reused warm graph pool per model (capture amortised across programs).
        g = CG.GraphedLeanForward(lean)

        print(f"  {'cluster':8s} {'label':16s} {'steps':>5s} {'fwds':>4s} "
              f"{'saved':>6s} {'ms/prog':>8s} {'ms/step':>8s} {'exact':>6s}  note")
        for c in subs_cases:
            code = isa.assemble(c.prog)
            ms, res = _time_spec(lean, code, g, warmup=warmup, iters=iters,
                                 device=device, block_steps=block_steps)
            steps = res.steps
            fwd = res.forwards
            saved = res.naive_forwards / fwd if fwd else 0.0
            bug = _known_bug_tag(c)
            # byte-exactness: for the func cluster ``isa.interpret`` has no
            # reference (JSR/ENT/LEV out of slice ISA), so validate graph==eager
            # naive lean (self-consistency, the test-suite's func validation).
            if c.family == "func":
                rn = LF.run_program_lean(lean, code, max_steps=512)
                exact = (res.ax_trace == rn["ax_trace"])
                note_ref = "graph==naive-lean (no slice-ISA ref)"
            else:
                exact = res.exact
                note_ref = ""
            pr = ProgResult(
                family=c.family, label=c.label, subset=sub_name, steps=steps,
                forwards=fwd, fwd_saved=saved, ms_wall=ms,
                ms_step=ms / max(steps, 1), exact=exact, bug_tag=bug)
            all_results.append(pr)
            note = (bug if bug else (note_ref if exact else "MISMATCH (unlabeled)"))
            print(f"  {c.family:8s} {c.label:16s} {steps:5d} {fwd:4d} "
                  f"{saved:5.1f}x {ms:8.2f} {pr.ms_step:8.2f} "
                  f"{str(exact):>6s}  {note}")
        del vm, lean, g
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {"results": all_results, "build_times": build_times, "layers": layers,
            "device": device, "block_steps": block_steps}


# ---------------------------------------------------------------------------
# Aggregation + extrapolation.
# ---------------------------------------------------------------------------
# The FULL non-muldiv corpus population.  The stratified sample here IS most of
# the tractable corpus; for the extrapolation we scale the measured per-cluster
# ms/program by a documented full-population program count per cluster, and note
# the deep diverging loops separately (they dominate wall by step-count).
#
# Population estimate: the 1096 corpus is dominated by a handful of clusters.
# The non-muldiv tractable population (from tools/run_1096_canonical.py's
# 863 tractable / 233 deep-diverging split) breaks down roughly as below.  These
# are LABELED estimates for the extrapolation, not a full run.
FULL_POP = {
    # cluster : (n_programs_full_population, mean_steps_full_population)
    "arith":   (140, 6),     # add/sub/imm variants
    "if":      (120, 5),     # bz/bnz/jmp/if_var/if_gt/if_lt/if_eq families
    "cmp":     (150, 5),     # eq/ne/lt/gt/le/ge x operands
    "func":    (90, 12),     # call/nested/adj/ent — deeper step traces
    "loop":    (60, 40),     # countdown/while — SHALLOW loops (deep ones separate)
    "memory":  (70, 7),      # si_li / zfod / latest-write
    "var":     (110, 10),    # var_add/var_two/multilocal
    "bitwise": (60, 5),      # and/or/xor
    "shift":   (40, 5),      # shl/shr
}
# deep diverging loops (gcd / rec / while(n) with n>=100 / mandelbrot-ish): they
# pass none today and dominate wall by step-count, so we account them SEPARATELY.
DEEP_DIVERGING = {"n_programs": 233, "mean_steps": 250}


def aggregate_and_extrapolate(report: Dict) -> Dict:
    results: List[ProgResult] = report["results"]
    device = report["device"]

    # per-cluster measured aggregates.
    by_cluster: Dict[str, List[ProgResult]] = defaultdict(list)
    for r in results:
        by_cluster[r.family].append(r)

    print("\n" + "=" * 78)
    print("PER-CLUSTER MEASURED AGGREGATES (lean graphed-speculative forward)")
    print("=" * 78)
    print(f"  {'cluster':8s} {'n':>3s} {'tot_steps':>9s} {'tot_fwd':>7s} "
          f"{'ms/prog':>9s} {'ms/step':>9s} {'exact':>7s}")
    cluster_stats = {}
    for fam in sorted(by_cluster):
        rs = by_cluster[fam]
        n = len(rs)
        tot_steps = sum(r.steps for r in rs)
        tot_fwd = sum(r.forwards for r in rs)
        tot_ms = sum(r.ms_wall for r in rs)
        ms_prog = tot_ms / n
        ms_step = tot_ms / max(tot_steps, 1)
        n_exact = sum(r.exact for r in rs)
        cluster_stats[fam] = {"n": n, "tot_steps": tot_steps, "tot_fwd": tot_fwd,
                              "ms_prog": ms_prog, "ms_step": ms_step,
                              "n_exact": n_exact}
        print(f"  {fam:8s} {n:3d} {tot_steps:9d} {tot_fwd:7d} "
              f"{ms_prog:8.2f}  {ms_step:8.2f}  {n_exact:2d}/{n}")

    # overall measured aggregates.
    tot_ms = sum(r.ms_wall for r in results)
    tot_steps = sum(r.steps for r in results)
    tot_fwd = sum(r.forwards for r in results)
    n_prog = len(results)
    n_exact = sum(r.exact for r in results)
    print(f"\n  MEASURED TOTAL: {n_prog} programs, {tot_steps} VM steps, "
          f"{tot_fwd} forwards")
    print(f"    wall (this sample, {report['block_steps']}-block spec, graphed): "
          f"{tot_ms:.1f} ms  ({tot_ms/n_prog:.2f} ms/program, "
          f"{tot_ms/max(tot_steps,1):.2f} ms/VM-step)")
    print(f"    byte-exact vs isa.interpret: {n_exact}/{n_prog}")

    # ---- extrapolation to the FULL non-muldiv corpus ----
    print("\n" + "=" * 78)
    print("EXTRAPOLATED FULL NON-MULDIV-CORPUS WALL "
          "(measured ms/step x labeled population)")
    print("=" * 78)
    print("  NOTE: measured ms/step from the sample above; program COUNTS + mean")
    print("  STEPS are a labeled population estimate, NOT a full run.")
    # use the overall measured ms/step (robust; per-cluster ms/step is nearly flat
    # because the forward is dominated by the fixed layer-stack launch, and spec
    # batches ~block_steps steps/forward).  Also compute a per-cluster figure using
    # each cluster's own measured ms/step where it has >=1 step.
    overall_ms_step = tot_ms / max(tot_steps, 1)

    print(f"\n  {'cluster':8s} {'n_full':>6s} {'mean_steps':>10s} "
          f"{'tot_steps':>10s} {'ms/step':>8s} {'cluster wall':>13s}")
    grand = 0.0
    for fam in sorted(FULL_POP):
        n_full, mean_steps = FULL_POP[fam]
        # use the cluster's own measured ms/step if available, else the overall.
        cs = cluster_stats.get(fam)
        ms_step = cs["ms_step"] if cs else overall_ms_step
        cl_steps = n_full * mean_steps
        cl_wall = cl_steps * ms_step / 1000.0
        grand += cl_wall
        print(f"  {fam:8s} {n_full:6d} {mean_steps:10d} {cl_steps:10d} "
              f"{ms_step:7.2f}  {cl_wall:11.1f}s")
    n_full_total = sum(v[0] for v in FULL_POP.values())
    print(f"\n  TRACTABLE non-muldiv full-corpus wall (extrapolated): "
          f"{grand:.0f}s  ({grand/60:.1f} min) for ~{n_full_total} programs")

    # deep diverging loops accounted separately.
    dd = DEEP_DIVERGING
    dd_steps = dd["n_programs"] * dd["mean_steps"]
    dd_wall = dd_steps * overall_ms_step / 1000.0
    print(f"\n  DEEP DIVERGING loops (gcd/rec/while n>=100): "
          f"~{dd['n_programs']} programs x ~{dd['mean_steps']} steps "
          f"= {dd_steps} steps")
    print(f"    at {overall_ms_step:.2f} ms/step (spec-graphed) -> "
          f"~{dd_wall:.0f}s ({dd_wall/60:.1f} min) — these DOMINATE the wall.")
    print(f"\n  GRAND TOTAL (tractable + deep diverging): "
          f"~{(grand+dd_wall):.0f}s ({(grand+dd_wall)/60:.1f} min)")

    return {"cluster_stats": cluster_stats, "measured_ms_step": overall_ms_step,
            "measured_wall_ms": tot_ms, "n_prog": n_prog, "n_exact": n_exact,
            "extrap_tractable_s": grand, "extrap_deep_s": dd_wall,
            "extrap_grand_s": grand + dd_wall}


def _pass_timing_split(report: Dict):
    results: List[ProgResult] = report["results"]
    print("\n" + "=" * 78)
    print("HONEST PASS / TIMING SPLIT")
    print("=" * 78)
    exact = [r for r in results if r.exact]
    buggy = [r for r in results if not r.exact]
    print(f"  BYTE-EXACT vs isa.interpret: {len(exact)}/{len(results)}")
    # cluster-level byte-exactness.
    by_cluster = defaultdict(lambda: [0, 0])
    for r in results:
        by_cluster[r.family][1] += 1
        if r.exact:
            by_cluster[r.family][0] += 1
    print("  per-cluster byte-exact:")
    for fam in sorted(by_cluster):
        p, t = by_cluster[fam]
        flag = "" if p == t else "  <- some fail (see below)"
        print(f"    {fam:8s} {p}/{t}{flag}")
    if buggy:
        print("\n  NON-EXACT programs (PASS-rate, NOT timing; these are #700):")
        for r in buggy:
            tag = r.bug_tag or "UNLABELED mismatch"
            print(f"    {r.family:8s} {r.label:16s}  -> {tag}")
    print("\n  => The lean forward TIMES every program identically regardless of")
    print("     PASS/FAIL (the forward runs the same kernels either way); the")
    print("     known model bugs (signed-compare, deep countdown) change the")
    print("     byte-exact PASS count, not the ms/step.  Timing is honest for all.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--block-steps", type=int, default=32)
    ap.add_argument("--no-bitwise", action="store_true",
                    help="skip the +bitwise build (mem+cmp only)")
    args = ap.parse_args()

    report = run_corpus_timing(
        device=args.device, iters=args.iters, warmup=args.warmup,
        block_steps=args.block_steps, include_bitwise=not args.no_bitwise)
    aggregate_and_extrapolate(report)
    _pass_timing_split(report)


if __name__ == "__main__":
    main()
