"""Triage the single-id long-tail residual after wide-byte ALU and
stack/JSR/LEV clusters are accounted for.

Read-only. Subtracts the wide-byte ALU prefix set (from
``triage_wide_alu.py``) and the stack/JSR/LEV prefix set (from
``triage_stack_jsr_lev.py``) from the diverging rows of the 06-01
sweep, then characterises whatever remains as the "single-id residual"
failure category referenced in ``docs/BUG_CATALOG.md`` as the final
untracked tail.

Usage:
    python tools/triage_single_id.py \\
        --sweep-dir .agent-logs/sweep-2026-06-01 \\
        --audit-glob '.agent-logs/lowering-audit/**/*.log' \\
        --out .agent-logs/single_id_longtail_triage_2026_06_01.md
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools.attribute_failures import (  # noqa: E402
    parse_diag_rows,
    _load_audit_index,
    _default_audit_paths,
)
from tools.triage_wide_alu import classify_program as classify_wide_alu  # noqa: E402
from tools.triage_stack_jsr_lev import classify_program as classify_stack  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402


# Slots covered by numbered bugs #27/#29/#30/#31/#28.
TRACKED_SLOT_RE = re.compile(
    r"^(?:SP_byte\d+|PC_byte\d+|STACK0_byte2)$"
)


def collect_sweep_rows(sweep_dir: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(sweep_dir, "shard_*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for r in parse_diag_rows(text):
            rows.setdefault(r.test_idx, {
                "status": r.status,
                "description": r.description,
                "expected": r.expected,
                "decl": r.declarative,
                "neural": r.neural,
            })
    return rows


def desc_prefix(desc: str) -> str:
    """Short prefix like 'edge_overflow' from desc 'edge_overflow_3: ...'."""
    base = desc.split(":", 1)[0].strip()
    m = re.match(r"^([a-z_]+?)_\d+$", base)
    if m:
        return m.group(1)
    return base


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=".agent-logs/sweep-2026-06-01")
    ap.add_argument("--audit-glob", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sweep_dir = args.sweep_dir
    if not os.path.isabs(sweep_dir):
        if not os.path.isdir(sweep_dir):
            cand = os.path.normpath(os.path.join(REPO, "..", sweep_dir))
            if os.path.isdir(cand):
                sweep_dir = cand

    # ---- 1. classify every id.
    programs = generate_test_programs()
    desc_by_id: Dict[int, str] = {
        i: desc for i, (_src, _exp, desc) in enumerate(programs)
    }
    expected_by_id: Dict[int, int] = {
        i: exp for i, (_src, exp, _desc) in enumerate(programs)
    }

    wide_alu_ids: Set[int] = set()
    stack_ids: Set[int] = set()
    for i, desc in desc_by_id.items():
        if classify_wide_alu(desc) is not None:
            wide_alu_ids.add(i)
        if classify_stack(desc) is not None:
            stack_ids.add(i)

    tracked_union = wide_alu_ids | stack_ids

    # ---- 2. sweep status per id.
    sweep = collect_sweep_rows(sweep_dir)
    diverging_ids = {i for i, r in sweep.items() if r["status"] not in {"ok", "strict-ok"}}

    # ---- 3. residual = diverging ids not in wide-ALU union with stack/JSR/LEV.
    residual_ids = sorted(diverging_ids - tracked_union)

    # ---- 4. audit (first-fatal slot) per residual id.
    if args.audit_glob:
        audit_paths = sorted(glob.glob(args.audit_glob, recursive=True))
    else:
        audit_paths = _default_audit_paths()
    audit = _load_audit_index(audit_paths)

    # ---- 5. group residual rows by failure pattern.
    by_prefix: Counter = Counter()
    by_status: Counter = Counter()
    slot_hist: Counter = Counter()
    tracked_slot_count = 0
    untracked_slot_count = 0
    no_audit_count = 0

    by_prefix_examples: Dict[str, List[Tuple[int, str, str, str]]] = defaultdict(list)
    per_row_records: List[Tuple[int, str, str, str, str, str]] = []
    # (id, prefix, desc, expected, neural, slot)

    for i in residual_ids:
        s = sweep[i]
        desc = s["description"] or desc_by_id.get(i, "")
        pfx = desc_prefix(desc)
        by_prefix[pfx] += 1
        by_status[s["status"]] += 1

        a = audit.get(i)
        if a is None:
            no_audit_count += 1
            slot_label = "<no-audit>"
        else:
            slot_label = f"step{a.step}:{a.slot}"
            slot_hist[slot_label] += 1
            if TRACKED_SLOT_RE.match(a.slot):
                tracked_slot_count += 1
            else:
                untracked_slot_count += 1

        if len(by_prefix_examples[pfx]) < 5:
            by_prefix_examples[pfx].append(
                (i, desc, s["expected"], s["neural"], slot_label)
            )
        per_row_records.append((i, pfx, desc, s["expected"], s["neural"], slot_label))

    # ---- 6. emit markdown.
    buf: List[str] = []
    w = buf.append

    w("# Single-id long-tail residual triage (2026-06-01)\n")
    w("Branch: `speedup-cache-and-buckets` @ HEAD ~`e18885a`.\n")
    w("Sweep logs: `.agent-logs/sweep-2026-06-01/shard_*.log`.\n")
    w("Catalog claim (`docs/BUG_CATALOG.md:339`): \"long-tail single-id "
      "regressions (~100-150 ids, no enumeration)\".\n")
    w("")
    w("This triage closes the final untracked failure category. It is the "
      "residual of the 864 diverging rows after the wide-byte ALU cluster "
      "(525 classified / 386 diverging) and the stack/JSR/LEV cluster "
      "(375 classified / 371 diverging) are subtracted. The two clusters "
      "overlap heavily (function-call wide-MUL families appear in both), "
      "so the actual residual is much smaller than 864 - 525 - 375.\n")
    w("")
    w("---\n")

    n_wide = len(wide_alu_ids)
    n_stack = len(stack_ids)
    n_union = len(tracked_union)
    n_overlap = n_wide + n_stack - n_union
    n_diverging = len(diverging_ids)
    n_sweep = len(sweep)
    n_ok = sum(1 for r in sweep.values() if r["status"] in {"ok", "strict-ok"})
    n_error = sum(1 for r in sweep.values() if r["status"] == "error")
    n_residual = len(residual_ids)

    w("## 1. Headline counts\n")
    w(f"- Total rows in 06-01 sweep: **{n_sweep}**\n")
    w(f"  - ok / strict-ok: **{n_ok}**\n")
    w(f"  - error: **{n_error}**\n")
    w(f"  - diverging (status != ok/strict-ok): **{n_diverging}**\n")
    w(f"- Wide-byte ALU classified rows: **{n_wide}**\n")
    w(f"- Stack/JSR/LEV classified rows: **{n_stack}**\n")
    w(f"- Overlap (rows in BOTH clusters): **{n_overlap}**\n")
    w(f"- Union (wide-ALU OR stack/JSR/LEV): **{n_union}**\n")
    w(f"- **Single-id residual = diverging - union = "
      f"{n_diverging} - {n_union} (cap to sweep coverage) = "
      f"{n_residual} rows**\n")
    w("")
    w("> Note: The brief's back-of-envelope subtraction (864 - 525 - 375 = "
      "~-36) is negative because wide-ALU and stack/JSR/LEV double-count "
      f"the {n_overlap} overlap rows (function-call wide-MUL families: "
      "`func_mul`, `func_square`, `nested_quad`, `nested_sumsq`, "
      "`rec_factorial`, `rec_power`, `gcd`). The set-theoretic residual "
      f"is the correct number: **{n_residual} rows**.\n")
    w("")

    # ok / diverge / error within the residual
    w("## 2. Residual status breakdown\n")
    for st, n in by_status.most_common():
        w(f"- status `{st}`: **{n}**\n")
    w("")

    # Failure pattern by prefix family
    w("## 3. Failure patterns (by test-family prefix)\n")
    w("| prefix | count | examples |\n")
    w("|---|---:|---|\n")
    for pfx, n in by_prefix.most_common():
        ex_lines = []
        for row_id, desc, exp, neural, slot in by_prefix_examples[pfx][:3]:
            d = desc.replace("|", "\\|")
            ex_lines.append(
                f"id={row_id:04d} {d} | exp={exp} neural={neural} slot=`{slot}`"
            )
        ex_md = " <br> ".join(ex_lines)
        w(f"| `{pfx}` | {n} | {ex_md} |\n")
    w("")

    # First-fatal slot histogram
    w("## 4. First-fatal slot histogram (residual rows with audit)\n")
    w(f"- residual rows with audit data: **{tracked_slot_count + untracked_slot_count}**\n")
    w(f"  - slot in SP_byte*/PC_byte*/STACK0_byte2 (subsumed by bugs #27/29/30/31/28): **{tracked_slot_count}**\n")
    w(f"  - slot elsewhere (new/unaddressed): **{untracked_slot_count}**\n")
    w(f"- residual rows without audit data: **{no_audit_count}**\n")
    w("")
    w("| slot | rows |\n|---|---:|\n")
    for slot, n in slot_hist.most_common():
        w(f"| `{slot}` | {n} |\n")
    w("")

    # Top sub-clusters: cross-tab prefix x slot
    w("## 5. Top 5 sub-clusters (prefix x first-fatal slot)\n")
    pair_counts: Counter = Counter()
    for row_id, pfx, desc, exp, neural, slot in per_row_records:
        pair_counts[(pfx, slot)] += 1
    w("| rank | prefix | first_fatal | count |\n|---:|---|---|---:|\n")
    for rank, ((pfx, slot), n) in enumerate(pair_counts.most_common(10), start=1):
        w(f"| {rank} | `{pfx}` | `{slot}` | {n} |\n")
    w("")

    # Per-row dump
    w("## 6. Per-row residual dump\n")
    w("| id | prefix | desc | expected | neural | first_fatal |\n")
    w("|---:|---|---|---|---|---|\n")
    for row_id, pfx, desc, exp, neural, slot in per_row_records[:200]:
        d = desc.replace("|", "\\|")
        w(f"| {row_id:04d} | `{pfx}` | {d} | {exp} | {neural} | `{slot}` |\n")
    if len(per_row_records) > 200:
        w(f"\n... +{len(per_row_records) - 200} more residual rows\n")
    w("")

    # Bug catalog cross-reference
    w("## 7. BUG_CATALOG.md cross-reference\n")
    w("The 32 numbered bugs (`docs/BUG_CATALOG.md`) cover the following "
      "residual rows:\n")
    if tracked_slot_count > 0:
        w(f"- **{tracked_slot_count} rows** have first-fatal in "
          "`SP_byte*` / `PC_byte*` / `STACK0_byte2` -- subsumed by "
          "Bug #27 (L10 soft ADDR_B0 misfire), #28 (`var_three` "
          "`STACK0_byte2`), #29 (function-call `SP_byte0` step 2), "
          "#30 (`PC_byte0` slice 548-821), #31 (`PC_byte1` slice "
          "822-1095).\n")
    if untracked_slot_count > 0:
        w(f"- **{untracked_slot_count} rows** have first-fatal elsewhere "
          "(AX/BP/MEM/STACK0 byte0,1,3 lanes). These are NOT covered by "
          "any numbered bug.\n")
    if no_audit_count > 0:
        w(f"- **{no_audit_count} rows** have no first-fatal audit data; "
          "they fail in the sweep but no `[1096-lowering]` line was "
          "captured. Attribution is unavailable.\n")
    w("")

    # Should a new bug be added?
    w("## 8. Should this open Bug #33+?\n")
    if untracked_slot_count + no_audit_count >= 25 and untracked_slot_count + no_audit_count <= 80:
        w(f"- **Likely yes**: the residual carries {untracked_slot_count + no_audit_count} "
          "rows of net-new failure modes not covered by bugs #1-32. The "
          "top sub-cluster(s) in section 5 should become Bug #33 (and "
          "potentially #34, #35) per their structural homogeneity.\n")
    elif untracked_slot_count + no_audit_count < 25:
        w(f"- **Likely no**: only {untracked_slot_count + no_audit_count} "
          "rows of net-new failure (slot outside SP_byte*/PC_byte*/"
          "STACK0_byte2 or no audit). These are best handled per-row "
          "rather than as a numbered cluster.\n")
    else:
        w(f"- **Maybe**: {untracked_slot_count + no_audit_count} rows of "
          "net-new failure. Whether to bug-number depends on whether they "
          "cluster by op family (section 5 should reveal that).\n")
    w("")

    # Recommendation
    w("## 9. Recommended fix approach\n")
    if untracked_slot_count == 0 and tracked_slot_count == n_residual:
        w(f"- Every residual row is subsumed by an existing numbered bug. "
          "**No new fix work** is needed for this category beyond "
          "landing the bug #27/29/30/31/28 fixes already in flight.\n")
    elif untracked_slot_count > 0 and untracked_slot_count <= 30:
        w(f"- **Surgical per-row**: {untracked_slot_count} rows is small "
          "enough that per-row attribution via `tools/attribute_failures.py` "
          "should identify each row's candidate rule individually. Avoid "
          "single-rule fixes (zero-sum per "
          "`feedback_single_rule_fixes_are_zero_sum.md`) -- batch into a "
          "single cluster edit per first-fatal slot.\n")
    else:
        w(f"- **Await per-row attribution improvement**: {untracked_slot_count} "
          "rows is large enough that surgical per-row attempts will be "
          "zero-sum. The attribution tool (`tools/attribute_failures.py`) "
          "may need an op-suspect cross-reference enrichment before this "
          "tail is addressable. Recommend deferring until the existing "
          "bug #27/29 fixes land and the residual is re-measured.\n")
    w("")

    # Estimated row impact
    w("## 10. Estimated row impact if addressable subset is fixed\n")
    if untracked_slot_count > 0:
        top_pair_count = pair_counts.most_common(1)[0][1] if pair_counts else 0
        w(f"- Top sub-cluster (section 5) has {top_pair_count} rows; "
          f"if that cluster is addressable as a single fix, "
          f"~{top_pair_count} rows would unblock.\n")
        w(f"- Total untracked residual: {untracked_slot_count} rows; "
          f"upper-bound row recovery is **~{untracked_slot_count} rows** "
          f"(less in practice due to downstream-tolerance overlap).\n")
    else:
        w("- Untracked residual is zero; row impact is fully covered by "
          "existing bug #27/29/30/31/28 fixes.\n")
    if no_audit_count > 0:
        w(f"- {no_audit_count} no-audit residual rows: row impact is "
          "uncertain. These rows likely require an audit-log re-run to "
          "be attributable; treat as 'unknown' for projection purposes.\n")
    w("")

    # Closeout block
    w("---\n")
    w("## Closeout\n")
    w(f"- Residual size: **{n_residual} rows** (after wide-ALU "
      f"+ stack/JSR/LEV subtraction with proper set arithmetic; the "
      f"overlap of {n_overlap} rows was the gap in the brief's "
      f"back-of-envelope estimate).\n")
    if tracked_slot_count > 0:
        pct_tracked = 100.0 * tracked_slot_count / max(1, n_residual)
        w(f"- {tracked_slot_count} of {n_residual} ({pct_tracked:.0f}%) "
          "residual rows are subsumed by existing numbered bugs.\n")
    if untracked_slot_count > 0:
        w(f"- {untracked_slot_count} rows are net-new and the candidate "
          "for Bug #33+ if they cluster (section 5).\n")
    if no_audit_count > 0:
        w(f"- {no_audit_count} rows lack audit data; re-run "
          "lowering-audit to attribute.\n")
    w("")

    report = "".join(buf)
    if args.out:
        out_path = args.out
        if not os.path.isabs(out_path):
            cwd_parent = os.path.dirname(os.path.abspath(out_path))
            if not os.path.isdir(cwd_parent):
                out_path = os.path.normpath(os.path.join(REPO, "..", out_path))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        sys.stderr.write(f"[triage_single_id] wrote -> {out_path}\n")
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
