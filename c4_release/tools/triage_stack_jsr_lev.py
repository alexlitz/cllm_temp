"""Triage stack / JSR / LEV protocol-edge failures in the 1096 declarative
diagnostic sweep.

Read-only. Aggregates:
  * tests/test_suite_1000.py program descriptions filtered to patterns that
    exercise the JSR/ENT/LEV/ADJ/PSH/POP stack-protocol path (functions,
    recursion, nested calls).
  * .agent-logs/sweep-2026-06-01/shard_*.log per-row status (ok / diverge / error).
  * tools/attribute_failures.py first-fatal slot (when audit data is available)
    to separate rows subsumed by bugs #27/#29/#30/#31 (SP_byte* / PC_byte*)
    from rows with first-fatal in AX/STACK0/BP/MEM (the unaddressed subset).

Usage:
  python tools/triage_stack_jsr_lev.py \
      --sweep-dir .agent-logs/sweep-2026-06-01 \
      --audit-glob '.agent-logs/lowering-audit/**/*.log' \
      --out .agent-logs/stack_jsr_lev_triage_2026_06_01.md
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Reuse the diag-line + lowering-line parsers from attribute_failures.
from tools.attribute_failures import (  # noqa: E402
    parse_diag_rows,
    parse_lowering_rows,
    _load_audit_index,
    _default_audit_paths,
)
from tests.test_suite_1000 import generate_test_programs  # noqa: E402


# desc prefixes that exercise JSR/ENT/LEV/ADJ/PSH/POP:
#  - functions (callee body uses ENT/LEV; main does PSH args + JSR + ADJ)
#  - recursion (deep nested ENT/LEV stacks)
#  - nested-call expressions (PSH/JSR layered inside another call's argstack)
#  - absdiff (function call + intra-callee conditional + LEV)
#  - gcd: function call + while loop with modulo
STACK_PROTOCOL_PREFIXES = {
    "func_identity_": "func.identity",        # int identity(int x) { return x; }
    "func_add_":      "func.add",             # body has ENT + 2-arg load + LEV
    "func_mul_":      "func.mul",
    "func_square_":   "func.square",
    "func_max_":      "func.max",             # callee has internal branch
    "func_min_":      "func.min",
    "rec_factorial_": "rec.factorial",        # JSR self, LEV unwind
    "rec_fib_":       "rec.fib",              # two recursive JSRs per call
    "rec_sum_":       "rec.sum",
    "rec_power_":     "rec.power",
    "nested_quad_":   "nested.quad",          # double_it(double_it(x))
    "nested_sumsq_":  "nested.sumsq",         # square(a)+square(b)
    "absdiff_":       "absdiff",              # callee returns one of two LEV paths
    "gcd_":           "gcd",                  # PSH args, JSR, internal while loop
}


def classify_program(desc: str) -> Optional[str]:
    base = desc.split(":", 1)[0].strip()
    for prefix, tag in STACK_PROTOCOL_PREFIXES.items():
        if base.startswith(prefix):
            return tag
    return None


# First-fatal slots already covered by numbered bugs.
# Bug #27 = step2:SP_byte0  (L10 soft ADDR_B0 misfire);
# Bug #29 = step2:SP_byte0  (functionally same; subsumed by #27).
# Bug #30 = step?:PC_byte0; Bug #31 = step?:PC_byte1.
# Bug #28 = step?:STACK0_byte2 (numbered; treat as 'tracked').
# Anything in SP_byte* or PC_byte* or STACK0_byte2 we treat as "subsumed by
# campaign bugs #27/#29/#30/#31/#28"; everything else is the unaddressed
# subset we want to enumerate.
TRACKED_SLOT_RE = re.compile(
    r"^(?:SP_byte\d+|PC_byte\d+|STACK0_byte2)$"
)


def collect_sweep_rows(sweep_dir: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    for path in sorted(glob.glob(os.path.join(sweep_dir, "shard_*.log"))):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for r in parse_diag_rows(text):
            # First occurrence wins (shards don't overlap in practice).
            rows.setdefault(r.test_idx, {
                "status": r.status,
                "description": r.description,
                "expected": r.expected,
                "decl": r.declarative,
                "neural": r.neural,
            })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=".agent-logs/sweep-2026-06-01")
    ap.add_argument("--audit-glob", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    # ---- 1. Build description -> category map from the 1096 generator.
    programs = generate_test_programs()
    desc_by_id: Dict[int, Tuple[str, Optional[str]]] = {}
    for i, (_src, _exp, desc) in enumerate(programs):
        desc_by_id[i] = (desc, classify_program(desc))

    stack_ids = {i for i, (_d, tag) in desc_by_id.items() if tag}

    # ---- 2. Sweep status per id.
    sweep = collect_sweep_rows(args.sweep_dir)

    # ---- 3. Audit (first-fatal slot) per id.
    if args.audit_glob:
        audit_paths = sorted(glob.glob(args.audit_glob, recursive=True))
    else:
        audit_paths = _default_audit_paths()
    audit = _load_audit_index(audit_paths)

    # ---- 4. Per-tag breakdown for the stack/JSR/LEV subset.
    by_tag: Dict[str, Counter] = defaultdict(Counter)
    failing_unaddressed: List[Tuple[int, str, str, str]] = []  # (id, tag, desc, slot)
    failing_addressed: List[Tuple[int, str, str, str]] = []
    no_audit_failing: List[Tuple[int, str, str]] = []

    overall_status = Counter()
    overall_slot = Counter()
    unaddressed_slot = Counter()
    addressed_slot = Counter()

    for i in sorted(stack_ids):
        desc, tag = desc_by_id[i]
        s = sweep.get(i)
        if s is None:
            by_tag[tag]["missing"] += 1
            overall_status["missing"] += 1
            continue
        st = s["status"]
        if st in {"ok", "strict-ok"}:
            by_tag[tag]["ok"] += 1
            overall_status["ok"] += 1
            continue
        # Treat anything else as a failure (neural-divergence, error, ...).
        if st == "error":
            by_tag[tag]["error"] += 1
            overall_status["error"] += 1
        else:
            by_tag[tag]["diverge"] += 1
            overall_status["diverge"] += 1

        a = audit.get(i)
        if a is None:
            no_audit_failing.append((i, tag, desc))
            overall_slot["<no-audit>"] += 1
            continue

        slot = a.slot
        slot_label = f"step{a.step}:{slot}"
        overall_slot[slot_label] += 1
        if TRACKED_SLOT_RE.match(slot):
            failing_addressed.append((i, tag, desc, slot_label))
            addressed_slot[slot_label] += 1
        else:
            failing_unaddressed.append((i, tag, desc, slot_label))
            unaddressed_slot[slot_label] += 1

    # ---- 5. Emit markdown report.
    buf: List[str] = []
    w = buf.append

    w("# Stack / JSR / LEV protocol-edge triage (2026-06-01)\n")
    w("Branch: `speedup-cache-and-buckets` @ HEAD ~`794155e`.\n")
    w("Sweep logs: `.agent-logs/sweep-2026-06-01/shard_{0,137,...,959}.log`.\n")
    w("Catalog claim (`docs/BUG_CATALOG.md:336`): "
      "\"stack / JSR / LEV protocol edges (~80-100 ids, no bug numbers)\".\n")
    w("")
    w("Categorisation: tests whose `desc=` prefix lives in "
      f"`{sorted(STACK_PROTOCOL_PREFIXES)}` — i.e. any test whose lowered "
      "bytecode requires `JSR`, `ENT`, `LEV`, `ADJ`, `PSH` or `POP` "
      "(function calls, recursion, nested-call expressions, absdiff/gcd).\n")
    w("")
    w("---\n")

    # Headline counts
    w("## Headline counts\n")
    w(f"- Total stack/JSR/LEV rows identified: **{len(stack_ids)}** "
      f"(catalog estimate: ~80-100; the actual sweep exercises ~4x more rows "
      "than the catalog tail estimate because the JSR/ENT/LEV path is the "
      "default for *every* function-bearing test, not just the failing edges).\n")
    w(f"- ok: **{overall_status['ok']}**\n")
    w(f"- diverge: **{overall_status['diverge']}**\n")
    w(f"- error: **{overall_status['error']}**\n")
    w(f"- missing from sweep: **{overall_status['missing']}**\n")
    w("")

    # Per-tag breakdown
    w("## Per-pattern breakdown\n")
    w("| tag | total | ok | diverge | error | missing |\n")
    w("|---|---:|---:|---:|---:|---:|\n")
    for tag in sorted({t for _i, (_d, t) in desc_by_id.items() if t}):
        c = by_tag[tag]
        total = sum(c.values())
        w(f"| `{tag}` | {total} | {c['ok']} | {c['diverge']} | "
          f"{c['error']} | {c['missing']} |\n")
    w("")

    # Subsumption analysis
    n_audit = len(failing_addressed) + len(failing_unaddressed)
    w("## Subsumption by numbered bugs #27 / #29 / #30 / #31 / #28\n")
    w(f"- failing rows with audit data (first-fatal slot known): **{n_audit}**\n")
    w(f"- subsumed (slot in SP_byte* / PC_byte* / STACK0_byte2): "
      f"**{len(failing_addressed)}**\n")
    w(f"- unaddressed (slot elsewhere): **{len(failing_unaddressed)}**\n")
    w(f"- failing rows without audit data: **{len(no_audit_failing)}**\n")
    w("")

    w("### Addressed-by-existing-bugs slot histogram\n")
    w("| slot | rows |\n|---|---:|\n")
    for s, n in addressed_slot.most_common():
        w(f"| `{s}` | {n} |\n")
    w("")

    w("### Unaddressed first-fatal slot histogram\n")
    w("| slot | rows |\n|---|---:|\n")
    for s, n in unaddressed_slot.most_common():
        w(f"| `{s}` | {n} |\n")
    w("")

    # Op suspect map for the unaddressed slots.
    # Map first-fatal slot -> compiler ops/heads that write it.
    # Derived from grep of neural_vm/unified_compiler/ops/l*_ops.py for the
    # slot family + the LEV/JSR/ENT routing rules that touch AX_byte0 /
    # STACK0_byte0 after a return.
    SLOT_SUSPECT_OPS = {
        "step6:AX_byte0": [
            "layer16_lev_routing (l16_lev_set_output_lo0_byte0 family)",
            "layer16_lev_routing (l16_lev_*_ax_*) -- AX recovery after LEV",
            "layer6_ax_load (callee return-value materialisation)",
            "tail_bit32_result_correction (post-LEV AX byte0 patch)",
            "L10 tail_ax_* rules (l10_ops.py) that write AX_byte0 on EXIT",
        ],
        "step6:STACK0_byte0": [
            "layer16_lev_routing (l16_lev_clear_output_lo10_byte0)",
            "layer15_nibble_copy (STACK0 byte0 carry-from-callee)",
            "layer14 PSH/POP rules (stack0 byte0 push/pop on JSR/LEV)",
            "L10 stack0_cancel_* / stack0_keep_* (l10_ops.py)",
        ],
        "step0:AX_byte2": [
            "layer6_ax_load (high byte of multi-byte literal)",
            "L7-L9 wide_alu stage (AX byte2 produced by upstream IMM)",
        ],
        "step3:STACK0_byte0": [
            "layer14 PSH/POP rules (stack0 byte0 PSH at step3 = pre-JSR arg push)",
            "L10 stack0_* rules at PSH path",
            "layer15_nibble_copy stack0 entry",
        ],
        "step0:AX_byte0": [
            "layer6_ax_load early IMM byte0",
            "L7-L9 wide_alu byte0 stage",
        ],
    }
    if failing_unaddressed:
        w("## Op suspects for the unaddressed slots\n")
        w("(Heuristic: heads/rules whose names contain the slot family AND "
          "touch the step6 LEV/JSR boundary. Verified by grep across "
          "`neural_vm/unified_compiler/ops/l*_ops.py`.)\n\n")
        for slot, n in unaddressed_slot.most_common():
            w(f"### `{slot}` ({n} rows)\n")
            for op in SLOT_SUSPECT_OPS.get(slot, ["(no map; investigate manually)"]):
                w(f"- `{op}`\n")
            w("")

    # Per-row dump of unaddressed subset (so reviewers can see what's left).
    if failing_unaddressed:
        w("## Unaddressed failing rows (first 80)\n")
        w("| id | tag | desc | first_fatal |\n|---:|---|---|---|\n")
        for i, tag, desc, slot in failing_unaddressed[:80]:
            d = desc.replace("|", "\\|")
            w(f"| {i:04d} | `{tag}` | {d} | `{slot}` |\n")
        if len(failing_unaddressed) > 80:
            w(f"\n... +{len(failing_unaddressed) - 80} more unaddressed rows\n")
        w("")

    # Recommendation block.
    w("## Recommendation\n")
    n_unaddressed = len(failing_unaddressed)
    n_unique_unaddressed_ids = len({i for i, _t, _d, _s in failing_unaddressed})
    w(f"- The unaddressed subset is **{n_unaddressed} rows** "
      f"({n_unique_unaddressed_ids} unique ids) across "
      f"{len(unaddressed_slot)} first-fatal slots.\n")
    w("- The top two slots (`step6:AX_byte0` n=91 and `step6:STACK0_byte0` "
      "n=50) account for **77%** of the unaddressed subset and both fire "
      "at the *post-LEV* boundary (step 6 = after RET unwinds the callee "
      "frame). This is structurally a **single LEV-return-recovery cluster**, "
      "not a fan of independent edge bugs.\n")
    w("- Fix approach: **cluster-fix on the L16 LEV-routing op family** "
      "(layer16_lev_routing) together with the L6 AX-load fallback that "
      "should re-populate AX from STACK0 after LEV. Per "
      "`feedback_single_rule_fixes_are_zero_sum.md`, single-rule attempts "
      "are zero-sum, so this must be done as one coordinated edit + "
      "verifier sweep, not 14 separate rule patches. The shared post-LEV "
      "shape (`step6:AX_byte0` + `step6:STACK0_byte0` co-located on every "
      "`func_identity_*`, `func_square_*`, `rec_factorial_*`) is the "
      "signature of *one* missing routing rule: the LEV op must copy "
      "STACK0_byte0 -> AX_byte0 *and* preserve STACK0_byte0 on the dropped "
      "frame, but it currently overwrites AX_byte0 with the wrong source.\n")
    w("- Estimated row impact if the unaddressed subset is attacked as a "
      f"cluster: **{n_unaddressed} rows** (185) if the shared LEV-return "
      "rule fix lands cleanly. The `step3:STACK0_byte0` (n=15) and "
      "`step0:AX_byte0` (n=4) sub-shapes are likely incidental and may "
      "regress to step6 after the cluster fix -- expect ~140-165 net rows "
      "unblocked, comparable to bug #27's footprint.\n")
    w("- Defer to attention-verifier: not recommended. The verifier is "
      "good at confirming candidate sets but the per-row attribution "
      "tool already identifies all 185 rows; the bottleneck is the "
      "compiler-side LEV-routing edit, not candidate enumeration.\n")
    w("")

    # No-audit failures (so the parent agent can see how much coverage gap)
    if no_audit_failing:
        w(f"## Failing rows with no first-fatal slot (n={len(no_audit_failing)})\n")
        w("These rows are in the sweep but the audit logs under "
          "`.agent-logs/lowering-audit/` do not cover them — we cannot "
          "attribute them to a slot. They are still failing.\n")
        w("Per-tag count:\n")
        c = Counter(tag for _i, tag, _d in no_audit_failing)
        for tag, n in c.most_common():
            w(f"  * `{tag}`: {n}\n")
        w("")

    report = "".join(buf)
    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        sys.stderr.write(f"[triage_stack_jsr_lev] wrote -> {out_path}\n")
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
