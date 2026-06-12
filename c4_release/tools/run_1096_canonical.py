#!/usr/bin/env python3
"""Canonical fast runner for the 1096 corpus.

This is the FAST (~10-15 min) equivalent of the pytest suite
``tests/test_suite_1096_pure_neural_pytest.py``. It reports the SAME
per-program pass/fail criterion as that suite, so the count it prints is the
canonical 1096 score (the real ~62/1096 number) rather than a ``--limit 128``
slice artifact.

Why a separate entry point from ``run_1096_fast.py``
---------------------------------------------------
``run_1096_fast.py`` already applies the same pass *criterion* (neural
exit-code == declarative oracle exit-code, masked to 32 bits). The "~1%"
numbers attributed to that tool historically came from running only a small
leading slice (``--limit 128`` = the add/sub/mul band, whose declarative
results the model gets wrong on the high-byte carry path), NOT from a
criterion mismatch. See ``docs/1096_CANONICAL_RUNNER_2026_06_11.md`` for the
full reconciliation.

This wrapper:

  * applies the SUITE's exact pass criterion (no KV cache, declarative halt
    horizon as ``max_steps``, neural-vs-declarative exit-code compare);
  * defaults to ``spec_k=0`` (raw one-token-per-forward) instead of the
    suite's literal ``spec_k=adaptive``. DraftVM is verifier-arbitrated and
    byte-identity with spec_k=0, so the pass SET is identical, but spec_k=0 is
    FASTER on the heavily-rejecting clusters (var/if/func/loop/rec) where
    adaptive K wastes a draft-and-rollback per rejected step — that adaptive
    waste is the main reason the full pytest run takes >5h. Pass ``--spec-k -1``
    to reproduce the suite's literal path (same passes, slower);
  * runs the FULL corpus by default (all 1096 programs);
  * emits a per-cluster breakdown (var / func / if / loop / rec / expr / ...)
    so we can see which clusters move as fixes land.

Memory robustness (2026-06-12)
------------------------------
The full corpus contains a handful of ultra-deep recursion programs
(``rec_fib`` ``fib(10..12)`` => 3185..8369 VM steps). At 35 tokens/step a
single such program produces a forward tensor of ~110k..290k tokens, which
OOMs a 24 GB GPU before any score is printed. This runner now:

  1. sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (at import,
     before torch allocates) to cut fragmentation OOM;
  2. groups prepared programs into MEMORY-SAFE LENGTH BUCKETS (deep programs
     run nearly solo, short ones batch wide) so the per-forward tensor stays
     bounded and short programs still run fast;
  3. frees the CUDA allocator cache + drops per-chunk references AFTER every
     chunk, so memory does not accumulate / fragment across chunks;
  4. streams each chunk's per-program results to a ``.jsonl`` checkpoint
     sidecar next to ``--output`` and prints a running cumulative tally to
     stderr, so a late OOM still yields a partial score on disk;
  5. supports ``--max-steps-cap N`` (default 40): any program whose
     DECLARATIVE oracle step count exceeds the cap is SKIPPED (counted
     separately as ``skipped`` — never as pass) and its id/cluster is logged
     (and recorded in ``--output``). This is an explicit, logged cap (no
     silent truncation). All 846 PASSABLE (non-diverging) programs are <=39
     steps, so the default 40 keeps EVERY potential pass while skipping the
     deeper diverging ``loop_*`` / ``gcd`` / ``rec_*`` band (~210 programs)
     that runs its full horizon at ``O(steps^2)`` — minutes each, OOM-prone
     when batched, and failing anyway. Raise the cap (e.g. 150, 2000) to fold
     more of that band back in — each step up is much slower (those run solo).

Pass criterion (identical to the suite's ``test_program``)
----------------------------------------------------------
A program counts as PASS iff ALL hold:

  1. ``compile_c`` succeeds, AND
  2. the declarative oracle halts and its exit code equals the suite's
     ``expected`` value masked to 32 bits
     (``decl_exit == suite_expected & 0xFFFFFFFF``), AND
  3. the pure-neural batched decode halts with the same exit code as the
     declarative oracle (``neural_exit & 0xFFFFFFFF == decl_exit & 0xFFFFFFFF``).

Anything else is FAIL. (In the pytest suite, 2 and 3 are two ``assert``
statements and 1 surfaces as ``err``; xfail(strict=False) means XPASS=pass,
xfail=fail, so PASS here == XPASS there.)

This is the PURE-NEURAL path (the vanilla thesis), exactly like the suite. It
is NOT the production handler path.

Usage
-----

    # Full canonical 1096 score + per-cluster breakdown (GPU 0):
    CUDA_VISIBLE_DEVICES=0 python tools/run_1096_canonical.py \
        --output /tmp/canonical_1096.json

    # A validation sample (compare these exact ids against the pytest suite):
    CUDA_VISIBLE_DEVICES=0 python tools/run_1096_canonical.py \
        --ids 0,49,250,550,975 --output /tmp/canonical_sample.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

# Cut allocator fragmentation OOM. MUST be set before torch is first imported
# (torch reads PYTORCH_CUDA_ALLOC_CONF when its CUDA caching allocator is
# initialised). ``setdefault`` so an explicit caller override is honoured.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Make ``import neural_vm`` / ``import tools`` work from any cwd / worktree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tools.run_1096_fast import (  # noqa: E402
    ProgramResult,
    _build_neural_runner,
    _compile_and_oracle,
    _parse_ids,
)

# Speculation default for the CANONICAL fast runner.
#
# The pytest suite's literal default is C4_SPEC_K="adaptive" -> -1 (per-element
# adaptive DraftVM starting at K=32). DraftVM is verifier-arbitrated and
# byte-identity with spec_k=0 (the model is the final arbiter), so the pass SET
# is identical for any spec_k. We therefore default this fast runner to
# spec_k=0 (raw one-token-per-forward batched decode) NOT to -1, because:
#
#   * Correctness is identical (byte-identity, documented in
#     batched_pure_neural.run_batch / project_probe_path_spec_k_not_hooks).
#   * spec_k=0 is FASTER on the heavily-rejecting clusters (var / if / func /
#     loop / rec). Adaptive K=32 wastes a full draft-and-rollback cycle on
#     every rejected step; on those clusters (~600 of 1096) that turns each
#     step into many wasted forwards. The suite's adaptive default is the main
#     reason the full pytest run takes >5h — it is throughput-optimal only on
#     clean single-step programs.
#
# spec_k=0 is also the smoke-gate ground-truth path
# (project_probe_path_spec_k_not_hooks). Pass --spec-k -1 to reproduce the
# suite's literal adaptive path (same passes, slower).
_SUITE_SPEC_K = 0

# Default cap (in DECLARATIVE oracle VM steps) above which a program is
# SKIPPED rather than run through the neural decode.
#
# WHY a cap (memory AND wall time, measured 2026-06-12, d_model=872, 37
# physical blocks, spec_k=0, CSR inference). The batched runner decodes with
# the FULL (un-windowed) context: every forward re-processes
# ``[B, prefix + tok_so_far]``, so a single program costs ``O(horizon^2)``
# forward FLOPs and ``O(horizon)`` activation memory.
#
#   * When the neural decode HALTS EARLY (model emits EXIT) the horizon is
#     short and the program is cheap (a 1205-step program that early-halts ran
#     solo in ~17 s at ~4 GB).
#   * When the neural decode DIVERGES from the declarative semantics it runs
#     the WHOLE declarative horizon one token at a time. The corpus's deep
#     ``loop_*`` / ``gcd`` / ``rec_*`` programs all diverge (the model does not
#     yet do deep loops/recursion), so a ~115-step ``loop_sum`` takes minutes
#     and a ~450-step one takes >14 min and peaks at ~22 GB of reserved memory.
#     Divergence is NOT predictable from the source, so a too-deep program is
#     both a wall-time sink (it fails anyway) and a solo OOM risk.
#
# The crucial corpus fact (measured): the corpus splits cleanly by cluster.
#   * The 846 NON-diverging programs (add/sub/mul/var/if/expr/edge/...) — the
#     ONLY clusters the model can pass — are ALL <= 39 declarative steps.
#   * The 250 DIVERGING programs (``loop_*`` / ``gcd`` / ``rec_*``) span 16..
#     8369 steps and the model passes NONE of them; under the neural decode
#     they run their full horizon (O(steps^2)) and the deeper ones OOM.
# So any cap >= 39 keeps EVERY potential pass. The cap only decides how many
# guaranteed-FAIL diverging programs get a real ``fail`` verdict vs ``skipped``.
#
# The default cap 40 keeps all 846 passable programs PLUS the few shallow
# diverging programs (16..40 steps), and skips the deeper diverging band
# (~210 programs > 40 steps) that is slow (minutes each) and OOM-prone when
# batched. With the conservative bucket widths below, the runnable set is both
# fast (~few minutes) and memory-safe. The skipped ids/clusters are printed
# and recorded in ``--output``, NEVER counted as pass.
#
# Raise ``--max-steps-cap`` to fold more of the diverging tail back in — but
# those programs run their full horizon at ``O(steps^2)`` (MINUTES each) and
# must run nearly SOLO (the bucket table forces width 1 above 80 steps) because
# batching them OOMs a 24 GB GPU (a 12-wide ~45-step batch already hit 24 GB;
# a 6-wide ~150-step batch climbs past 23 GB). Useful higher caps add slow,
# all-failing programs only. Pass 0 to disable the cap (WILL OOM on the deepest
# rec_fib, 8369 steps ~= a 290k-token forward tensor).
_DEFAULT_MAX_STEPS_CAP = 40

# Memory-safety scale on the per-bucket batch widths below. The widths were
# tuned for a dedicated 24 GB GPU; ``--mem-step-scale 0.5`` halves every width
# (rounding up) so the run co-exists with another job on a shared GPU. 1.0 =
# the tuned widths.
_DEFAULT_MEM_STEP_SCALE = 1.0

# Length buckets (upper bound on declarative steps) -> per-chunk batch width.
#
# The memory driver is the per-block attention score tensor ``[B, H, S, S]``
# (O(B * S^2) in the context length S). When a program's neural decode
# DIVERGES it runs to the full declarative horizon, so S grows to
# ``steps * 35`` and the score tensor co-grows for EVERY batch slot. Measured:
# a 48-wide batch of ~45-step diverging programs hit 24 GB; a 12-wide one still
# OOM'd; a 6-wide ~150-step batch climbs past 23 GB. So only the genuinely
# trivial programs (<=15 steps) batch wide; deeper bands narrow sharply and
# anything that can diverge to a large S runs solo.
#
# Tuples are ``(step_upper_bound, width)`` checked in order; first whose bound
# >= the chunk's deepest member wins. Deliberately conservative: a diverging
# member runs its FULL horizon, so the score tensor ``[B, H, S, S]`` co-grows
# for every batch slot. A 48-wide batch of ~45-step diverging programs hit
# 24 GB and a 12-wide one still OOM'd (measured), so the deeper bands narrow
# sharply and anything > 80 steps runs solo.
_LENGTH_BUCKET_CHUNKS: Tuple[Tuple[int, int], ...] = (
    (15, 32),    # trivially shallow (S<=525): batch wide for throughput.
    (25, 8),
    (40, 2),     # 25..40 steps: mostly diverging loop/gcd -> width 2.
    (80, 1),
    (1 << 30, 1),  # >80 steps: SOLO (a diverging member fills the horizon).
)


def _chunk_size_for_steps(steps: int, mem_step_scale: float) -> int:
    """Batch width for a chunk whose deepest member is ``steps`` deep.

    Wide for shallow programs, narrow / solo for deep ones (see the
    ``_LENGTH_BUCKET_CHUNKS`` rationale). ``mem_step_scale`` < 1.0 shrinks
    every width for a shared GPU.
    """
    steps = max(1, int(steps))
    for bound, width in _LENGTH_BUCKET_CHUNKS:
        if steps <= bound:
            scaled = max(1, int(round(width * max(0.0, mem_step_scale))))
            return scaled
    return 1


def cluster_of(description: str) -> str:
    """Derive a stable cluster key from a program's description.

    Descriptions look like ``add_0: 654 + 114`` / ``var_simple_3: x = 7`` /
    ``rec_fib_12: fib(9)`` / ``edge_literal_4`` / ``gcd_17: ...``. The cluster
    is the alphabetic prefix with the trailing per-case index stripped. This
    reproduces the cluster families in docs/1096_TRIAGE_2026_06_11.md:
    add, sub, mul, div, mod, var_*, if_*, loop_*, func_*, rec_*, expr_*,
    gcd, nested_*, edge*, absdiff, bool_and.
    """
    base = description.split(":", 1)[0].strip()
    # Strip a trailing _<digits> (per-case index), then any trailing digits.
    base = re.sub(r"_\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    base = base.rstrip("_")
    return base or "misc"


def _select(all_tests, ids_spec: Optional[str], offset: int, limit: Optional[int]):
    enumerated = list(enumerate(all_tests))
    if ids_spec:
        wanted = set(_parse_ids(ids_spec))
        return [
            (idx, src, exp, desc)
            for idx, (src, exp, desc) in enumerated
            if idx in wanted
        ]
    windowed = enumerated[offset:]
    if limit is not None:
        windowed = windowed[:limit]
    return [(idx, src, exp, desc) for idx, (src, exp, desc) in windowed]


def _cluster_breakdown(
    results: List[ProgramResult],
) -> "OrderedDict[str, Dict[str, int]]":
    """Aggregate pass/fail/error/skipped per cluster, in first-seen order."""
    table: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in results:
        key = cluster_of(r.description)
        row = table.setdefault(
            key, {"n": 0, "pass": 0, "fail": 0, "error": 0, "skipped": 0}
        )
        row["n"] += 1
        if r.status == "ok":
            row["pass"] += 1
        elif r.status == "error":
            row["error"] += 1
        elif r.status == "skipped":
            row["skipped"] += 1
        else:
            row["fail"] += 1
    return table


def _print_cluster_table(table, fh=sys.stderr) -> None:
    print("\n[1096-canonical] per-cluster breakdown", file=fh, flush=True)
    print(
        f"  {'cluster':22s} {'n':>5s} {'pass':>5s} {'fail':>5s} {'err':>5s} "
        f"{'skip':>5s} {'pass%':>6s}",
        file=fh,
    )
    for key, row in table.items():
        n = row["n"]
        pct = (100.0 * row["pass"] / n) if n else 0.0
        print(
            f"  {key:22s} {n:5d} {row['pass']:5d} {row['fail']:5d} "
            f"{row['error']:5d} {row.get('skipped', 0):5d} {pct:6.1f}",
            file=fh,
        )


def _empty_cuda_cache() -> None:
    """Release cached-but-unused CUDA blocks back to the driver.

    The model weights stay resident (they are referenced by the runner); this
    only frees the decode-time activation / KV scratch the chunk just used. We
    call this after every chunk so memory does not accumulate or fragment
    across the 1096-program run.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _split_over_cap(
    prepared: List[Tuple[int, int, str, int, int, list, bytes]],
    max_steps_cap: Optional[int],
) -> Tuple[
    List[Tuple[int, int, str, int, int, list, bytes]],
    List[ProgramResult],
]:
    """Partition prepared programs into (runnable, skipped-over-cap).

    A program is skipped iff its DECLARATIVE oracle step count exceeds the
    cap. Skipped programs become ``ProgramResult(status="skipped")`` — they
    are counted separately in the score, NEVER as pass. The skip list is
    logged by the caller (no silent truncation).
    """
    if not max_steps_cap or max_steps_cap <= 0:
        return list(prepared), []
    runnable: List[Tuple[int, int, str, int, int, list, bytes]] = []
    skipped: List[ProgramResult] = []
    for entry in prepared:
        idx, expected, description, decl_exit, decl_steps, _bc, _data = entry
        if decl_steps is not None and decl_steps > max_steps_cap:
            skipped.append(
                ProgramResult(
                    idx=idx,
                    description=description,
                    suite_expected=expected,
                    declarative_exit=decl_exit,
                    declarative_steps=decl_steps,
                    neural_exit=None,
                    status="skipped",
                    error=(
                        f"skipped: declarative steps {decl_steps} "
                        f"> --max-steps-cap {max_steps_cap}"
                    ),
                )
            )
        else:
            runnable.append(entry)
    return runnable, skipped


def _length_bucketed_chunks(
    runnable: List[Tuple[int, int, str, int, int, list, bytes]],
    *,
    mem_step_scale: float,
    fixed_chunk: Optional[int] = None,
) -> List[List[Tuple[int, int, str, int, int, list, bytes]]]:
    """Group runnable programs into memory-safe, length-aware chunks.

    Sort by declarative step count (descending) so deep programs lead, then
    emit chunks whose batch width is the per-bucket width for the chunk's
    FIRST (longest) member (see ``_LENGTH_BUCKET_CHUNKS``). Deep programs run
    nearly solo while shallow programs batch wide, bounding the worst-case
    forward tensor regardless of whether a member's neural decode diverges and
    runs its full horizon. ``mem_step_scale`` < 1.0 shrinks every width for a
    shared GPU.

    ``fixed_chunk`` (a positive int) overrides the bucket logic with a fixed
    width for every chunk — used only for parity testing.
    """
    ordered = sorted(
        runnable,
        key=lambda e: (e[4] if e[4] is not None else 0),
        reverse=True,
    )
    chunks: List[List[Tuple[int, int, str, int, int, list, bytes]]] = []
    i = 0
    n = len(ordered)
    while i < n:
        lead_steps = ordered[i][4] or 0
        if fixed_chunk and fixed_chunk > 0:
            size = int(fixed_chunk)
        else:
            size = _chunk_size_for_steps(lead_steps, mem_step_scale)
        chunk = ordered[i : i + size]
        chunks.append(chunk)
        i += len(chunk)
    return chunks


def _run_chunks_streaming(
    neural_runner,
    prepared: List[Tuple[int, int, str, int, int, list, bytes]],
    *,
    spec_k: int,
    max_context_window: int,
    mem_step_scale: float,
    fixed_chunk: Optional[int],
    checkpoint_path: Optional[str],
) -> List[ProgramResult]:
    """Run prepared programs through the neural runner in length-aware chunks.

    Differs from ``run_1096_fast._run_chunks``:

      * chunk batch width is MEMORY-SAFE + length-aware (deep programs run
        nearly solo, shallow ones batch wide) instead of a fixed 32;
      * the CUDA cache is freed + per-chunk tensors are dropped AFTER each
        chunk;
      * each chunk's per-program results are appended to ``checkpoint_path``
        (a ``.jsonl`` sidecar) immediately, so a late OOM crash still leaves
        a partial score on disk.

    Returns the list of ``ProgramResult`` (input order is restored by the
    caller via ``sort(key=idx)``).
    """
    import traceback

    results: List[ProgramResult] = []
    total = len(prepared)
    if total == 0:
        return results

    chunks = _length_bucketed_chunks(
        prepared,
        mem_step_scale=mem_step_scale,
        fixed_chunk=fixed_chunk,
    )

    cum_pass = 0
    cum_fail = 0
    cum_error = 0
    done = 0

    ckpt_fh = None
    if checkpoint_path:
        ckpt_fh = open(checkpoint_path, "w", encoding="utf-8")

    try:
        for chunk_no, chunk in enumerate(chunks, start=1):
            bytecodes = [entry[5] for entry in chunk]
            data_list = [entry[6] for entry in chunk]
            expected_steps = [entry[4] for entry in chunk]
            chunk_ids = [entry[0] for entry in chunk]

            chunk_t0 = time.monotonic()
            chunk_results: List[ProgramResult] = []
            try:
                neural_results = neural_runner.run_batch(
                    bytecodes,
                    data_list=data_list,
                    max_steps=None,
                    expected_steps_list=expected_steps,
                    max_context_window=max_context_window,
                    spec_k=spec_k,
                )
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc(limit=3)
                for entry in chunk:
                    idx, expected, description, decl_exit, decl_steps, _, _ = entry
                    chunk_results.append(
                        ProgramResult(
                            idx=idx,
                            description=description,
                            suite_expected=expected,
                            declarative_exit=decl_exit,
                            declarative_steps=decl_steps,
                            neural_exit=None,
                            status="error",
                            error=f"neural batch error: {exc!r}",
                        )
                    )
                    cum_error += 1
                print(
                    f"[1096-canonical] chunk {chunk_no}/{len(chunks)} "
                    f"ids={min(chunk_ids):04d}-{max(chunk_ids):04d} "
                    f"CRASHED: {exc!r}\n{tb}",
                    file=sys.stderr,
                    flush=True,
                )
                neural_results = None
            else:
                for entry, (_neural_output, neural_exit) in zip(
                    chunk, neural_results
                ):
                    idx, expected, description, decl_exit, decl_steps, _, _ = entry
                    if neural_exit is None:
                        status = "error"
                        cum_error += 1
                        err = "neural exit is None (no halt within horizon)"
                        ne_masked = None
                    else:
                        ne = int(neural_exit) & 0xFFFFFFFF
                        de = int(decl_exit) & 0xFFFFFFFF
                        ne_masked = ne
                        if ne == de:
                            status = "ok"
                            cum_pass += 1
                            err = None
                        else:
                            status = "fail"
                            cum_fail += 1
                            err = None
                    chunk_results.append(
                        ProgramResult(
                            idx=idx,
                            description=description,
                            suite_expected=expected,
                            declarative_exit=decl_exit,
                            declarative_steps=decl_steps,
                            neural_exit=ne_masked,
                            status=status,
                            error=err,
                        )
                    )

            chunk_elapsed = time.monotonic() - chunk_t0
            results.extend(chunk_results)
            done += len(chunk)

            # Stream this chunk's verdicts to the checkpoint sidecar so a
            # later OOM still leaves a partial score on disk.
            if ckpt_fh is not None:
                for r in chunk_results:
                    ckpt_fh.write(json.dumps(asdict(r)) + "\n")
                ckpt_fh.flush()
                os.fsync(ckpt_fh.fileno())

            # Drop references to this chunk's tensors / decode state and free
            # the CUDA allocator cache so memory does not accumulate across
            # chunks. The model weights are held by ``neural_runner`` and are
            # NOT freed.
            del bytecodes, data_list, neural_results, chunk
            _empty_cuda_cache()

            print(
                f"[1096-canonical] chunk {chunk_no}/{len(chunks)} "
                f"{done}/{total} done "
                f"ids={min(chunk_ids):04d}-{max(chunk_ids):04d} "
                f"size={len(chunk_results)} "
                f"max_steps={max(expected_steps)} "
                f"chunk_wall={chunk_elapsed:.1f}s | "
                f"cum: pass={cum_pass} fail={cum_fail} err={cum_error}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()

    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical fast runner for the 1096 corpus. Reports the SAME "
            "per-program pass/fail as test_suite_1096_pure_neural_pytest.py."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N tests (default: all 1096).",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first M tests before applying --limit.",
    )
    parser.add_argument(
        "--ids", type=str, default=None,
        help="Comma-separated ids/ranges (e.g. '0,5-9,42'). Overrides offset/limit.",
    )
    parser.add_argument(
        "--chunk", type=int,
        default=int(os.environ.get("C4_BATCH_CHUNK", "0")),
        help="Fixed programs per neural batch. Default 0 = MEMORY-SAFE "
             "length-aware bucketing (deep programs run nearly solo, shallow "
             "ones batch wide; see --mem-step-scale). Pass a positive value "
             "to force a fixed chunk size for every batch (parity testing).",
    )
    parser.add_argument(
        "--spec-k", type=int, default=_SUITE_SPEC_K,
        help="Speculative-decode K (default: 0 = raw one-token-per-forward). "
             "DraftVM is byte-identity with spec_k=0 so this does NOT change "
             "the pass set vs the suite's adaptive default; 0 is chosen for "
             "speed on the rejecting clusters. Pass -1 to reproduce the "
             "suite's literal adaptive path (same passes, slower).",
    )
    parser.add_argument(
        "--max-steps-cap", type=int, default=_DEFAULT_MAX_STEPS_CAP,
        help=f"Skip (count as 'skipped', NEVER pass) any program whose "
             f"declarative oracle step count exceeds this cap. Default "
             f"{_DEFAULT_MAX_STEPS_CAP} keeps the run FAST and memory-safe: all "
             f"846 passable (non-diverging) programs are <=39 steps, so this "
             f"drops NO pass; it only skips the deeper diverging loop/gcd/rec "
             f"band that runs its full horizon at O(steps^2) (minutes each, "
             f"OOM-prone when batched) and fails anyway. Raise it to fold more "
             f"of that band in (slower; runs solo). Pass 0 to disable (WILL OOM).",
    )
    parser.add_argument(
        "--mem-step-scale", type=float, default=_DEFAULT_MEM_STEP_SCALE,
        help=f"Scale on the per-bucket batch widths (deep=solo, shallow=wide). "
             f"Default {_DEFAULT_MEM_STEP_SCALE} is tuned for a DEDICATED "
             f"24 GB GPU. Pass e.g. 0.5 to halve every width so the run "
             f"co-exists with another job on a shared GPU.",
    )
    parser.add_argument(
        "--max-context-window", type=int,
        default=int(os.environ.get("C4_BATCH_CONTEXT_WINDOW", "512")),
        help="Tail context window (default: 512, matching the suite).",
    )
    parser.add_argument(
        "--model-max-seq-len", type=int,
        default=int(os.environ.get("C4_BATCH_MODEL_MAX_SEQ_LEN", "4096")),
        help="Model max seq len for the runner (default: 4096).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to dump full per-program results + cluster breakdown as JSON.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a JSONL checkpoint sidecar (one result per line, "
             "flushed per chunk). Defaults to '<output>.jsonl' when --output "
             "is set so a late OOM still leaves a partial score on disk.",
    )
    parser.add_argument(
        "--print-failures", action="store_true",
        help="Stream each non-passing program row to stdout.",
    )
    args = parser.parse_args(argv)

    # Match run_1096_fast: only set a default device if the caller hasn't.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    overall_t0 = time.monotonic()

    from tests.test_suite_1000 import generate_test_programs

    all_tests = generate_test_programs()
    selected = _select(all_tests, args.ids, args.offset, args.limit)

    print(
        f"[1096-canonical] selected={len(selected)} (of {len(all_tests)}) "
        f"chunk={'mem-safe' if args.chunk <= 0 else args.chunk} "
        f"mem_step_scale={args.mem_step_scale} "
        f"spec_k={args.spec_k} max_steps_cap={args.max_steps_cap} "
        f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')} "
        f"(suite-exact criterion; pure-neural path) "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        file=sys.stderr,
        flush=True,
    )

    # Phase 1: compile + declarative oracle (CPU). Oracle is called with
    # suite_expected so an oracle-vs-suite disagreement becomes an oracle
    # error == suite assert #2 failing == FAIL. Identical to the suite.
    prepared, oracle_errors = _compile_and_oracle(selected)

    # Phase 1b: partition out over-cap (too-deep) programs BEFORE the model
    # build, so they never enter a forward. They are counted as 'skipped'.
    runnable, skipped = _split_over_cap(prepared, args.max_steps_cap)
    if skipped:
        skipped_by_cluster: "OrderedDict[str, List[ProgramResult]]" = OrderedDict()
        for r in skipped:
            skipped_by_cluster.setdefault(cluster_of(r.description), []).append(r)
        print(
            f"[1096-canonical] SKIPPING {len(skipped)} program(s) over "
            f"--max-steps-cap={args.max_steps_cap} (counted as 'skipped', "
            f"NOT pass):",
            file=sys.stderr,
            flush=True,
        )
        for cluster, rows in skipped_by_cluster.items():
            detail = ", ".join(
                f"id={r.idx}(steps={r.declarative_steps})" for r in rows
            )
            print(
                f"    cluster={cluster}: {detail}",
                file=sys.stderr,
                flush=True,
            )

    # Phase 2: bake the pure-neural model once.
    neural_runner = _build_neural_runner(model_max_seq_len=args.model_max_seq_len)

    # Determine the checkpoint sidecar path (default '<output>.jsonl').
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.output:
        checkpoint_path = args.output + ".jsonl"

    # Phase 3: batched pure-neural decode in memory-budgeted, length-aware
    # chunks, with CUDA cache freed between chunks and per-chunk results
    # streamed to the checkpoint sidecar. A positive --chunk forces a fixed
    # batch width for every chunk (parity testing only).
    fixed_chunk = int(args.chunk) if args.chunk and args.chunk > 0 else None
    neural_results = _run_chunks_streaming(
        neural_runner,
        runnable,
        spec_k=int(args.spec_k),
        max_context_window=int(args.max_context_window),
        mem_step_scale=float(args.mem_step_scale),
        fixed_chunk=fixed_chunk,
        checkpoint_path=checkpoint_path,
    )

    all_results: List[ProgramResult] = (
        oracle_errors + skipped + neural_results
    )
    all_results.sort(key=lambda r: r.idx)

    total_elapsed = time.monotonic() - overall_t0

    pass_n = sum(1 for r in all_results if r.status == "ok")
    fail_n = sum(1 for r in all_results if r.status == "fail")
    err_n = sum(1 for r in all_results if r.status == "error")
    skip_n = sum(1 for r in all_results if r.status == "skipped")
    total_n = len(all_results)

    # The suite counts FAIL and ERROR identically (both are "not a pass" /
    # xfail). The canonical headline number is PASS / total. Skipped programs
    # are reported separately and are NEVER counted as pass.
    print(
        f"\n[1096-canonical] CANONICAL SCORE: {pass_n}/{total_n} PASS "
        f"({100.0 * pass_n / total_n:.2f}%)  "
        f"[fail={fail_n} error={err_n} skipped(>cap)={skip_n}]  "
        f"wall={total_elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    table = _cluster_breakdown(all_results)
    _print_cluster_table(table)

    if args.print_failures:
        for r in all_results:
            if r.status != "ok":
                print(r.to_row(), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "wall_seconds": total_elapsed,
                    "criterion": (
                        "suite-exact: compile_c ok AND "
                        "decl_exit==(suite_expected & 0xFFFFFFFF) AND "
                        "neural_exit==decl_exit (both masked 0xFFFFFFFF); "
                        "pure-neural batched path"
                    ),
                    "spec_k": args.spec_k,
                    "chunk": (
                        "mem-safe" if args.chunk <= 0 else args.chunk
                    ),
                    "mem_step_scale": args.mem_step_scale,
                    "max_steps_cap": args.max_steps_cap,
                    "summary": {
                        "total": total_n,
                        "pass": pass_n,
                        "fail": fail_n,
                        "error": err_n,
                        "skipped": skip_n,
                    },
                    "skipped_ids": [
                        {
                            "idx": r.idx,
                            "cluster": cluster_of(r.description),
                            "declarative_steps": r.declarative_steps,
                            "description": r.description,
                        }
                        for r in all_results
                        if r.status == "skipped"
                    ],
                    "clusters": {
                        k: v for k, v in table.items()
                    },
                    "results": [asdict(r) for r in all_results],
                },
                fh,
                indent=2,
            )
        print(f"[1096-canonical] wrote {args.output}", file=sys.stderr, flush=True)
        if checkpoint_path:
            print(
                f"[1096-canonical] streamed per-chunk checkpoint to "
                f"{checkpoint_path}",
                file=sys.stderr,
                flush=True,
            )

    # Exit 0 always: this is a measurement tool, not a gate. (The suite's
    # gate semantics are xfail; a non-zero pass count is success here.)
    return 0


if __name__ == "__main__":
    sys.exit(main())
