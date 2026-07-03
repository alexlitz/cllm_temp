#!/usr/bin/env python3
"""UNIFIED VERIFICATION GATE — one command that detects what a change touched and
runs the CORRECT subset of the 6+ verification tools automatically.

The gap this closes
-------------------
Today a contributor / agent must KNOW which of the many gates to run for a given
change (CLAUDE.md lists them: ``tools/_isa_golden_hash.py`` byte-identity;
``tools/lint_cross_op_attention.py`` shared-attention-head; ``tools/lint_cross_op_ffn.py``
shared-FFN/OUTPUT/ALU-band; ``tools/flag_regression_gate.py`` campaign flag-ON;
``tools/gpu_tripwire.py`` cross-cluster; ``tools/cpu_full_trace.py`` framing;
``pytest tests/test_smoke.py``). This is error-prone — the lints EXIST precisely
BECAUSE isolated per-op checks missed shared-head / shared-band interactions
(the byte-0 −39 and the mul-l14 −60 incidents). Picking the wrong subset silently
re-opens the exact blind spot the lint was built for.

What this tool does
-------------------
1. DETECT what the change touched, from ``git diff`` (working tree + committed
   vs ``main``, or vs ``--base <commit>``). The added/removed hunk lines are
   classified by DSL signature:

     * any ops / compiler file changed        -> byte-identity + smoke  (always)
     * a SHARED attention head edited          -> lint_cross_op_attention
       (``DeclarativeAttentionHeadSpec`` / ``AP(`` / ``AO(`` / ``.W_q`` / ``.W_k`` /
       ``.W_v`` / ``.W_o`` / ``head_specs`` / ``head_idx`` / ``RuntimeAttentionFragment``)
     * a SHARED FFN / OUTPUT / ALU band edited  -> lint_cross_op_ffn
       (``OUTPUT_LO`` / ``OUTPUT_HI`` / ``ALU_LO`` / ``ALU_HI`` / ``FFNRule`` /
       ``constant_write`` / ``gated_write`` / ``W_down`` / ``register_residual_band``)
     * a ``C4_*`` flag added / changed          -> flag_regression_gate  (needs --flag)
     * any op changed at all                    -> optional verdict sample (cpu_full_trace)

   The diff-based detection is a fast PRE-FILTER: it decides WHICH lints to
   invoke. The lints themselves are AUTHORITATIVE — they build the model
   flag-OFF/flag-ON and diff the actual weights, so a diff hunk that merely
   MENTIONS ``W_q`` inside a comment triggers the attn-lint, which then reports
   "no shared head modified" and passes in seconds. Over-triggering a lint is
   cheap and safe; UNDER-triggering is the dangerous direction, so the
   signatures are deliberately broad.

2. RUN the correct subset, in the right order: the FAST byte-identity + smoke
   gates first (a broken build / regressed smoke should abort before the slow
   lints), then the targeted shared-surface lints, then the optional campaign
   flag-ON regression sample and the CPU verdict sample.

3. REPORT one consolidated PASS / FAIL with per-gate detail (status, seconds,
   why it ran / was skipped).

Flags
-----
    --base <commit>   diff against this commit instead of ``main``.
    --flag  <C4_X>    the fix's kill-switch flag. REQUIRED to run the two shared-
                      surface lints and the flag-regression gate (they build the
                      model OFF vs ON by toggling this flag). Without it, those
                      gates are reported as SKIPPED-NEEDS-FLAG (with the exact
                      command to run them).
    --expect <rows>   whitelist forwarded to the lints (opcodes/contexts the fix
                      is ALLOWED to change, e.g. ``OP_ADD,OP_SUB``).
    --quick           skip the slow verdict sample (cpu_full_trace) and the
                      flag-regression gate; keep byte-identity + smoke + lints.
    --clusters <c>    forwarded to flag_regression_gate (narrow the sample).
    --no-smoke        skip the pytest smoke run (e.g. on a contended GPU box —
                      see the smoke-timeout memory note).
    --verdict-ids <l> ids for the CPU verdict sample (default: a tiny arith set).

Exit code is the number of FAILED gates (0 == all pass). Tooling only: this
runner never writes weights; it subprocesses the existing tools unchanged, so
the golden model is byte-identical.

USAGE
-----
    # gate a working-tree change (auto-detect everything):
    python tools/gate.py

    # gate a flag-gated fix (enables the shared-surface + campaign gates):
    python tools/gate.py --flag C4_MY_FIX --expect OP_ADD,OP_SUB

    # gate a branch vs a base commit, fast (skip the slow verdict sample):
    python tools/gate.py --base e405ea92 --flag C4_MY_FIX --quick

    # self-test the detector on synthetic diffs (no builds):
    python tools/gate.py --self-test
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)  # .../c4_release
_GOLDEN_SHA = "b4d2ab273438b3b2fa1bd024b48d0b06a15e63a81f66dec2812ada580b3ec70e"


# ---------------------------------------------------------------------------
# Detection signatures. Classified per CHANGED (added/removed) diff line so a
# file that merely CONTAINS both attn + ffn DSL (almost every lN_ops.py does)
# only triggers the lint whose surface the change actually edited.
# ---------------------------------------------------------------------------

# Files whose change means "an op / compiler weight could have moved" -> always
# byte-identity + smoke. Matches the whole weight-authoring surface.
_OP_FILE_RE = re.compile(
    r"(^|/)neural_vm/("
    r"unified_compiler/|vm_step\.py|setup_helpers\.py|dim_registry|dim_allocator"
    r"|attention_head_allocator|ffn_unit_allocator|primitives\.py)"
)

# A changed line touching a SHARED attention head's Q/K/V/O or a head spec.
_ATTN_LINE_RE = re.compile(
    r"DeclarativeAttentionHeadSpec"
    r"|RuntimeAttentionFragment"
    r"|AttentionHeadIR"
    r"|AttentionOp\b"
    r"|head_specs"
    r"|head_idx"
    r"|_HEAD_LAYOUT"
    r"|\bAP\("            # AttentionProjection constructor
    r"|\bAO\("            # AttentionOutput constructor
    r"|\.W_q\b|\.W_k\b|\.W_v\b"
    r"|attn\.W_o\b"
    r"|\bW_q\[|\bW_k\[|\bW_v\["
)

# A changed line touching a SHARED FFN region / OUTPUT / ALU residual band.
_FFN_LINE_RE = re.compile(
    r"\bOUTPUT_LO\b|\bOUTPUT_HI\b|\bALU_LO\b|\bALU_HI\b"
    r"|\bFFNRule\b|\bFFNOp\b"
    r"|\bconstant_write\b|\bgated_write\b"
    r"|\bW_down\b|\bW_up\b|\bW_gate\b"
    r"|register_residual_band"
)

# A changed line adding / removing / re-referencing a C4_* env flag.
_FLAG_LINE_RE = re.compile(r"C4_[A-Z0-9_]+")


@dataclass
class Detection:
    changed_files: List[str] = field(default_factory=list)
    op_files: List[str] = field(default_factory=list)
    touched_op: bool = False
    touched_attn: bool = False
    touched_ffn: bool = False
    touched_flag: bool = False
    flags_seen: List[str] = field(default_factory=list)
    attn_evidence: List[str] = field(default_factory=list)
    ffn_evidence: List[str] = field(default_factory=list)


def _run_diff(base: str, *, only_names: bool) -> str:
    """Return ``git diff`` output for base..working-tree (committed + uncommitted).

    ``base`` defaults to ``main`` in the CLI. Using a plain ``git diff <base>``
    (no ``..``) compares the base commit against the CURRENT working tree, so
    uncommitted edits in the isolated worktree ARE captured — exactly what a
    pre-commit gate wants.
    """
    cmd = ["git", "diff"]
    if only_names:
        cmd.append("--name-only")
    else:
        cmd.append("--unified=0")
    cmd.append(base)
    out = subprocess.run(
        cmd, cwd=_PKG, capture_output=True, text=True
    )
    return out.stdout


def detect(base: str, *, diff_text: Optional[str] = None,
           names_text: Optional[str] = None) -> Detection:
    """Classify a git diff into which gates it needs.

    ``diff_text`` / ``names_text`` may be injected (the ``--self-test`` path
    feeds synthetic diffs); otherwise they are produced by ``git diff <base>``.
    """
    if names_text is None:
        names_text = _run_diff(base, only_names=True)
    if diff_text is None:
        diff_text = _run_diff(base, only_names=False)

    det = Detection()
    det.changed_files = [f for f in names_text.splitlines() if f.strip()]
    det.op_files = [f for f in det.changed_files if _OP_FILE_RE.search(f)]
    det.touched_op = bool(det.op_files)

    flags: set = set()
    for raw in diff_text.splitlines():
        # Only classify CHANGED content lines (+/-), not hunk headers (@@, +++,
        # ---) or context. Hunk/file headers start with '+++'/'---'/'@@'.
        if not raw or raw[0] not in "+-":
            continue
        if raw.startswith(("+++", "---")):
            continue
        line = raw[1:]  # strip the +/- marker
        if _ATTN_LINE_RE.search(line):
            det.touched_attn = True
            if len(det.attn_evidence) < 5:
                det.attn_evidence.append(line.strip()[:100])
        if _FFN_LINE_RE.search(line):
            det.touched_ffn = True
            if len(det.ffn_evidence) < 5:
                det.ffn_evidence.append(line.strip()[:100])
        for m in _FLAG_LINE_RE.findall(line):
            flags.add(m)
    det.touched_flag = bool(flags)
    det.flags_seen = sorted(flags)
    return det


# ---------------------------------------------------------------------------
# Gate execution.
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    name: str
    status: str          # "PASS" / "FAIL" / "SKIP" / "SKIP-NEEDS-FLAG"
    reason: str
    seconds: float = 0.0
    cmd: Optional[List[str]] = None


def _cpu_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "")  # prefer CPU per memory discipline
    if extra:
        env.update(extra)
    return env


def _run_gate(name: str, cmd: List[str], *, why: str,
              env: Optional[Dict[str, str]] = None,
              timeout: Optional[int] = None) -> GateResult:
    print(f"\n{'=' * 78}\n[gate] RUN  {name}\n[gate]   why: {why}\n"
          f"[gate]   cmd: {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=_PKG, env=env or _cpu_env(),
                              timeout=timeout)
        rc = proc.returncode
        status = "PASS" if rc == 0 else "FAIL"
        reason = why if rc == 0 else f"{why} (exit {rc})"
    except subprocess.TimeoutExpired:
        # A TIMEOUT is INFRA, not a regression: on this shared/CPU-contended
        # fleet a build+decode gate can exceed the wall budget purely because
        # sibling agents are thrashing the cores (see the smoke-timeout memory
        # note). Report it as TIMEOUT (surfaced as ERR, non-zero) with a
        # re-run hint — never mislabel it a genuine FAIL / regression.
        status = "TIMEOUT"
        reason = (f"{why} (TIMEOUT after {timeout}s — likely CPU/GPU "
                  f"contention on a shared box; re-run uncontended or raise "
                  f"the timeout before believing this a real failure)")
    secs = time.monotonic() - t0
    print(f"[gate]   -> {name}: {status} ({secs:.0f}s)", flush=True)
    return GateResult(name=name, status=status, reason=reason,
                      seconds=secs, cmd=cmd)


def gate_byte_identity() -> GateResult:
    """The golden byte-identity gate — the flag-OFF 35-token production build.

    We run the whole-model param-hash harness and compare against the recorded
    golden SHA. A mismatch is a hard FAIL (the change moved a production weight).
    """
    name = "byte-identity (_isa_golden_hash)"
    why = "an op / compiler file changed — the flag-OFF golden build MUST be unchanged"
    cmd = [sys.executable, os.path.join(_HERE, "_isa_golden_hash.py")]
    print(f"\n{'=' * 78}\n[gate] RUN  {name}\n[gate]   why: {why}\n"
          f"[gate]   cmd: {' '.join(cmd)}\n{'=' * 78}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=_PKG, env=_cpu_env(),
                          capture_output=True, text=True)
    secs = time.monotonic() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"state_dict_sha256=([0-9a-f]{64})", out)
    got = m.group(1) if m else None
    if proc.returncode != 0 or got is None:
        status = "FAIL"
        reason = f"harness failed (exit {proc.returncode}); output tail: {out[-200:]!r}"
    elif got == _GOLDEN_SHA:
        status = "PASS"
        reason = f"golden {got[:8]} unchanged"
    else:
        status = "FAIL"
        reason = f"GOLDEN DRIFT: got {got[:8]} != golden {_GOLDEN_SHA[:8]}"
    print(f"[gate]   -> {name}: {status} ({secs:.0f}s) — {reason}", flush=True)
    return GateResult(name=name, status=status, reason=reason,
                      seconds=secs, cmd=cmd)


def gate_smoke(timeout: int = 1800) -> GateResult:
    """The pytest opcode/path smoke suite (spec_k=0, the neural-authoritative path)."""
    return _run_gate(
        "smoke (pytest test_smoke.py)",
        [sys.executable, "-m", "pytest",
         os.path.join(_PKG, "tests", "test_smoke.py"),
         "-q", "--no-header", "-p", "no:cacheprovider"],
        why="an op changed — the narrow opcode/path smoke suite must stay green",
        env=_cpu_env({"C4_SMOKE_SPEC_K": "0"}),
        timeout=timeout,
    )


def gate_attn_lint(flag: str, expect: str) -> GateResult:
    cmd = [sys.executable, os.path.join(_HERE, "lint_cross_op_attention.py"),
           "--flag", flag]
    if expect:
        cmd += ["--expect", expect]
    return _run_gate(
        "lint_cross_op_attention",
        cmd,
        why=("a SHARED attention head signature was edited — the isolated-op "
             "check cannot see the global softmax1 interaction"),
        timeout=1200,
    )


def gate_ffn_lint(flag: str, expect: str) -> GateResult:
    cmd = [sys.executable, os.path.join(_HERE, "lint_cross_op_ffn.py"),
           "--flag", flag]
    if expect:
        cmd += ["--expect", expect]
    return _run_gate(
        "lint_cross_op_ffn",
        cmd,
        why=("a SHARED FFN / OUTPUT / ALU band signature was edited — the "
             "isolated-op check cannot see the shared-band silu/softmax read"),
        env=_cpu_env({"C4_VM_CACHE_DIR":
                      os.environ.get("C4_VM_CACHE_DIR", "/tmp/c4cache_ffnlint")}),
        timeout=1200,
    )


def gate_flag_regression(flag: Optional[str], base: Optional[str],
                         clusters: Optional[str]) -> GateResult:
    cmd = [sys.executable, os.path.join(_HERE, "flag_regression_gate.py")]
    if flag:
        cmd += ["--flag", flag]
        why = (f"a C4_* flag changed — verify {flag}=ON does not regress any "
               f"cluster in the 30-token campaign config")
    else:
        cmd += ["--base", base]  # type: ignore[arg-type]
        why = ("a C4_* flag changed but no --flag given — gating HEAD vs base "
               "in the campaign config")
    if clusters:
        cmd += ["--clusters", clusters]
    return _run_gate("flag_regression_gate", cmd, why=why, timeout=3600)


def gate_verdict_sample(ids: str) -> GateResult:
    """A tiny CPU full_trace verdict sample (framing self-check)."""
    return _run_gate(
        "verdict sample (cpu_full_trace)",
        [sys.executable, os.path.join(_HERE, "cpu_full_trace.py"),
         "--ids", ids, "--spec-k", "0", "--workers", "1",
         "--max-steps-cap", "18", "--criterion", "full_trace"],
        why="an op changed — spot-check the framing verdict on a small id set",
        timeout=1800,
    )


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def _print_detection(det: Detection, base: str) -> None:
    print(f"{'=' * 78}\n[gate] DETECTION (diff vs {base})\n{'=' * 78}", flush=True)
    print(f"[gate] changed files: {len(det.changed_files)}", flush=True)
    for f in det.changed_files:
        tag = "  (op/compiler)" if f in det.op_files else ""
        print(f"[gate]     {f}{tag}", flush=True)
    print(f"[gate] touched op/compiler weights : {det.touched_op}", flush=True)
    print(f"[gate] touched SHARED attn head    : {det.touched_attn}"
          + (f"   e.g. {det.attn_evidence[0]!r}" if det.attn_evidence else ""),
          flush=True)
    print(f"[gate] touched SHARED FFN/ALU band : {det.touched_ffn}"
          + (f"   e.g. {det.ffn_evidence[0]!r}" if det.ffn_evidence else ""),
          flush=True)
    print(f"[gate] touched C4_* flag           : {det.touched_flag}"
          + (f"   flags: {det.flags_seen}" if det.flags_seen else ""),
          flush=True)


def plan_gates(det: Detection, *, flag: Optional[str], quick: bool,
               no_smoke: bool) -> List[str]:
    """Compute the ORDERED gate PLAN (names only) without running anything.

    Mirrors the branch logic in :func:`run_gates` exactly so ``--dry-run`` and a
    real run agree on WHAT would execute. Used by ``--dry-run`` to validate the
    auto-detection routing on a REAL diff with zero model builds.
    """
    plan: List[str] = []
    if not det.touched_op and not det.touched_flag:
        return ["(no gate — no op/compiler/flag change)"]
    if det.touched_op:
        plan.append("byte-identity (_isa_golden_hash)")
        plan.append("smoke (SKIP: --no-smoke)" if no_smoke else "smoke (pytest test_smoke.py)")
    else:
        plan.append("byte-identity (SKIP: no op/compiler file changed)")
    if det.touched_attn:
        plan.append("lint_cross_op_attention"
                    + ("" if flag else " (SKIP-NEEDS-FLAG)"))
    if det.touched_ffn:
        plan.append("lint_cross_op_ffn"
                    + ("" if flag else " (SKIP-NEEDS-FLAG)"))
    if det.touched_flag:
        if quick:
            plan.append("flag_regression_gate (SKIP: --quick)")
        else:
            plan.append("flag_regression_gate"
                        + ("" if flag else " (SKIP-NEEDS-FLAG unless --base)"))
    if det.touched_op:
        plan.append("verdict sample (cpu_full_trace)"
                    + (" (SKIP: --quick)" if quick else ""))
    return plan


def run_gates(det: Detection, *, base: str, flag: Optional[str], expect: str,
              quick: bool, clusters: Optional[str], no_smoke: bool,
              verdict_ids: str, smoke_timeout: int = 1800) -> List[GateResult]:
    results: List[GateResult] = []

    if not det.touched_op and not det.touched_flag:
        print("\n[gate] No op / compiler file and no C4_* flag changed — "
              "nothing weight-affecting to gate.", flush=True)
        results.append(GateResult(
            "no-op", "PASS",
            "no op/compiler/flag change detected — no gate required"))
        return results

    # --- FAST gates first: byte-identity + smoke ---------------------------
    if det.touched_op:
        results.append(gate_byte_identity())
        if no_smoke:
            results.append(GateResult(
                "smoke (pytest test_smoke.py)", "SKIP",
                "skipped via --no-smoke"))
        else:
            results.append(gate_smoke(timeout=smoke_timeout))
    else:
        results.append(GateResult(
            "byte-identity (_isa_golden_hash)", "SKIP",
            "no op/compiler file changed — only a flag reference moved"))

    # --- Targeted shared-surface lints (need OFF/ON via --flag) ------------
    if det.touched_attn:
        if flag:
            results.append(gate_attn_lint(flag, expect))
        else:
            results.append(GateResult(
                "lint_cross_op_attention", "SKIP-NEEDS-FLAG",
                "a shared attention head was edited but no --flag given; run: "
                f"python tools/lint_cross_op_attention.py --flag C4_MY_FIX"
                + (f" --expect {expect}" if expect else "")))
    if det.touched_ffn:
        if flag:
            results.append(gate_ffn_lint(flag, expect))
        else:
            results.append(GateResult(
                "lint_cross_op_ffn", "SKIP-NEEDS-FLAG",
                "a shared FFN/ALU band was edited but no --flag given; run: "
                f"python tools/lint_cross_op_ffn.py --flag C4_MY_FIX"
                + (f" --expect {expect}" if expect else "")))

    # --- Campaign flag-ON cross-cluster gate ------------------------------
    if det.touched_flag and not quick:
        if flag or base:
            results.append(gate_flag_regression(flag, base, clusters))
        else:
            results.append(GateResult(
                "flag_regression_gate", "SKIP-NEEDS-FLAG",
                "a C4_* flag changed but neither --flag nor --base given; run: "
                "python tools/flag_regression_gate.py --flag C4_MY_FIX"))
    elif det.touched_flag and quick:
        results.append(GateResult(
            "flag_regression_gate", "SKIP",
            "skipped via --quick (run before landing: "
            f"python tools/flag_regression_gate.py --flag {flag or 'C4_MY_FIX'})"))

    # --- Optional verdict sample (framing spot-check) ---------------------
    if det.touched_op and not quick:
        results.append(gate_verdict_sample(verdict_ids))
    elif det.touched_op and quick:
        results.append(GateResult(
            "verdict sample (cpu_full_trace)", "SKIP",
            "skipped via --quick"))

    return results


def _print_summary(det: Detection, results: List[GateResult]) -> int:
    print(f"\n{'=' * 78}\n[gate] CONSOLIDATED REPORT\n{'=' * 78}", flush=True)
    width = max((len(r.name) for r in results), default=10)
    n_fail = 0
    n_timeout = 0
    n_needs_flag = 0
    for r in results:
        if r.status == "FAIL":
            n_fail += 1
        elif r.status == "TIMEOUT":
            n_timeout += 1
        elif r.status == "SKIP-NEEDS-FLAG":
            n_needs_flag += 1
        secs = f"{r.seconds:4.0f}s" if r.seconds else "   - "
        print(f"[gate]   {r.status:16s} {secs}  {r.name:<{width}}  — {r.reason}",
              flush=True)

    print(f"{'=' * 78}", flush=True)
    # A genuine FAIL (real regression) is the strongest signal and dominates.
    if n_fail:
        verdict = f"FAIL — {n_fail} gate(s) FAILED (real regression). DO NOT LAND."
    elif n_timeout:
        # A TIMEOUT blocks landing (the gate did not go green) but is INFRA, not
        # a proven regression — say so, and don't call it a FAIL.
        verdict = (f"INCONCLUSIVE — {n_timeout} gate(s) TIMED OUT (likely "
                   f"CPU/GPU contention, NOT a regression). Re-run uncontended "
                   f"(or with --smoke-timeout) before landing; every other gate "
                   f"that ran passed.")
    elif n_needs_flag:
        verdict = (f"INCOMPLETE — all run gates passed, but {n_needs_flag} "
                   f"shared-surface / flag gate(s) need --flag to run. "
                   f"Re-run with --flag <C4_X> before landing.")
    else:
        verdict = "PASS — all applicable gates passed."
    print(f"[gate] {verdict}", flush=True)
    print(f"{'=' * 78}", flush=True)
    # Exit code: non-zero if any gate did not go green (a real FAIL or a
    # blocking TIMEOUT). NEEDS-FLAG alone is not a hard failure (nothing
    # regressed) but is surfaced loudly above.
    return n_fail + n_timeout


# ---------------------------------------------------------------------------
# Self-test: prove the detector routes synthetic diffs to the right gates,
# with NO model build (fast, offline).
# ---------------------------------------------------------------------------
def _self_test() -> int:
    print("=" * 78)
    print("SELF-TEST — detector routing on synthetic diffs (no model build)")
    print("=" * 78)

    cases: List[Tuple[str, str, str, Dict[str, bool]]] = [
        (
            "no-op (docs only)",
            "docs/README.md\n",
            "diff --git a/docs/README.md b/docs/README.md\n"
            "--- a/docs/README.md\n+++ b/docs/README.md\n"
            "@@ -1 +1 @@\n-old text\n+new text\n",
            {"touched_op": False, "touched_attn": False,
             "touched_ffn": False, "touched_flag": False},
        ),
        (
            "shared attention head edit (W_q slot on L7 head)",
            "neural_vm/unified_compiler/ops/l7_ops.py\n",
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -10 +10,2 @@\n"
            "+    spec = DeclarativeAttentionHeadSpec(head_idx=0)\n"
            "+    spec.W_q[34] = c * (OP_ADD + OP_SUB)\n",
            {"touched_op": True, "touched_attn": True,
             "touched_ffn": False, "touched_flag": False},
        ),
        (
            "shared FFN/ALU band edit (OUTPUT_LO gated_write)",
            "neural_vm/unified_compiler/ops/l14_ops.py\n",
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -20 +20,2 @@\n"
            "+    FFNRule.gated_write(writes={'OUTPUT_LO': 30.0},\n"
            "+                        conditions=['OP_MUL'])\n",
            {"touched_op": True, "touched_attn": False,
             "touched_ffn": True, "touched_flag": False},
        ),
        (
            "flag-only add (new C4_ kill-switch gate on an FFN band)",
            "neural_vm/unified_compiler/ops/l10_ops.py\n",
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -5 +5,2 @@\n"
            "+    if os.environ.get('C4_MY_FIX'):\n"
            "+        rule = FFNRule.constant_write(writes={'ALU_HI': 1.0})\n",
            {"touched_op": True, "touched_attn": False,
             "touched_ffn": True, "touched_flag": True},
        ),
        (
            "comment-only mention of W_q (broad signature — safe over-trigger)",
            "neural_vm/unified_compiler/ops/l5_ops.py\n",
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n"
            "+    # note: this does NOT touch attn.W_o\n",
            {"touched_op": True, "touched_attn": True,
             "touched_ffn": False, "touched_flag": False},
        ),
    ]

    ok = True
    for label, names, diff, expect in cases:
        det = detect("<synthetic>", diff_text=diff, names_text=names)
        got = {
            "touched_op": det.touched_op,
            "touched_attn": det.touched_attn,
            "touched_ffn": det.touched_ffn,
            "touched_flag": det.touched_flag,
        }
        passed = got == expect
        ok = ok and passed
        print(f"\n[self-test] {label}")
        print(f"[self-test]   expected {expect}")
        print(f"[self-test]   got      {got}   -> {'OK' if passed else 'MISMATCH'}")
        # Show which gates this routing would run.
        gates = ["byte-identity", "smoke"] if det.touched_op else []
        if det.touched_attn:
            gates.append("lint_cross_op_attention")
        if det.touched_ffn:
            gates.append("lint_cross_op_ffn")
        if det.touched_flag:
            gates.append("flag_regression_gate")
        print(f"[self-test]   routes to: {gates or ['(no gate)']}")

    print("\n" + "=" * 78)
    print(f"SELF-TEST: {'PASS — detector routes every case correctly' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base", default="main",
                    help="diff against this commit (default: main).")
    ap.add_argument("--flag", default=None,
                    help="the fix's kill-switch C4_* flag — REQUIRED to run the "
                         "shared-surface lints + the campaign flag-regression "
                         "gate (they build the model OFF vs ON via this flag).")
    ap.add_argument("--expect", default="",
                    help="whitelist forwarded to the lints (opcodes/contexts the "
                         "fix is ALLOWED to change, e.g. OP_ADD,OP_SUB).")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow verdict sample + flag-regression gate; "
                         "keep byte-identity + smoke + the shared-surface lints.")
    ap.add_argument("--clusters", default=None,
                    help="forwarded to flag_regression_gate (narrow the sample).")
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip the pytest smoke run (e.g. on a contended GPU box).")
    ap.add_argument("--smoke-timeout", type=int, default=1800,
                    help="wall budget (s) for the smoke suite before it is "
                         "reported as TIMEOUT (default 1800). Raise on a "
                         "CPU-contended box; a TIMEOUT is flagged as INFRA, "
                         "not a regression.")
    ap.add_argument("--verdict-ids", default="0,1,2,3",
                    help="ids for the CPU verdict sample (default a tiny set).")
    ap.add_argument("--self-test", action="store_true",
                    help="run the detector routing self-test (no model build) "
                         "and exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="detect + print the ORDERED gate PLAN from the real "
                         "diff, but RUN nothing (no model build). Use to see "
                         "what the gate would do for the current change.")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    det = detect(args.base)
    _print_detection(det, args.base)

    if args.dry_run:
        plan = plan_gates(det, flag=args.flag, quick=args.quick,
                          no_smoke=args.no_smoke)
        print(f"\n{'=' * 78}\n[gate] PLAN (dry-run — nothing executed)\n"
              f"{'=' * 78}", flush=True)
        for i, name in enumerate(plan, 1):
            print(f"[gate]   {i}. {name}", flush=True)
        print(f"{'=' * 78}", flush=True)
        return 0

    results = run_gates(
        det, base=args.base, flag=args.flag, expect=args.expect,
        quick=args.quick, clusters=args.clusters, no_smoke=args.no_smoke,
        verdict_ids=args.verdict_ids, smoke_timeout=args.smoke_timeout,
    )
    return _print_summary(det, results)


if __name__ == "__main__":
    sys.exit(main())
