"""Triage F-7 scope_violations for L10 make_tail_bit32_result_correction_op.

Pure analysis: classifies each violation by likely cause into four
categories (A/B/C/D) using simple heuristics over the rule name,
declared scope, and the effective predicate computed by F-5.

Categories:
  A  scope-annotation error: effective and name agree on a marker, but
     scope declares a DIFFERENT one (label mistake).
  B  genuine E5-class bug: conditions admit positions outside intent;
     scope is correct, conditions are too loose.
  C  false positive from imprecise dim semantics: registry semantics are
     placeholders (e.g., `is_byte OR NOT is_byte`) and entailment is
     defeated by vacuity.
  D  unsatisfiable effective predicate: conditions contradict (e.g.,
     positive `MEM_STORE` requires `mark == MEM`, but `MARK_MEM` is a
     hard blocker requiring `NOT mark == MEM`). Structural F-5
     over-approximation issue; commonly hides a gated rule.

Writes:
  L10_summary.json
  L10_summary.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make c4_release importable regardless of cwd.
HERE = Path(__file__).resolve()
C4_RELEASE = HERE.parents[2]
if str(C4_RELEASE) not in sys.path:
    sys.path.insert(0, str(C4_RELEASE))

from neural_vm.dim_registry import build_default_registry  # noqa: E402
from neural_vm.unified_compiler.ir import FFNOp  # noqa: E402
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    make_tail_bit32_result_correction_op,
)
from neural_vm.verification.decl_verifier import (  # noqa: E402
    verify_rule_scopes,
)
from neural_vm.verification.predicates import (  # noqa: E402
    parse,
    satisfiable,
)

# Canonical short-name -> registry role-name.
MARKER_CANON = {
    "stack0": "STACK0",
    "mem": "MEM",
    "sp": "SP",
    "ax": "AX",
    "pc": "PC",
    "bp": "BP",
    "se": "SE",
}
MARKER_WORDS = list(MARKER_CANON.keys())

# Placeholder-vacuous semantics phrases. If the effective predicate
# contains any of these as a literal substring, we count it as
# "vacuous-leaning" — useful evidence for Category C.
VACUOUS_PHRASES = (
    "is_byte OR NOT is_byte",
    "NOT (is_byte OR NOT is_byte)",
)


# --- IR walk ----------------------------------------------------------------


def collect_rules_from_compiler_ir(op):
    """Walk an op's CompilerIR layers to gather every FFNRule.

    The stock decl_verifier `_collect_ffn_rules_from_op` only handles
    FFNOp / list / tuple shapes; `make_tail_bit32_result_correction_op`
    uses `CompilerIR` with `layer(idx).ffn.rules`, which falls through.
    """
    ir = op.compiler_ir
    if hasattr(ir, "layer"):
        rules = []
        # Try all layer indices; stop on the first missing layer.
        for idx in range(64):
            try:
                layer = ir.layer(idx)
            except Exception:
                break
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and getattr(ffn, "rules", None):
                rules.extend(ffn.rules)
        return rules
    return []


class _FakeOpWrapper:
    """Wraps a flat list of FFNRules into an FFNOp the verifier will walk."""

    def __init__(self, rules):
        self.compiler_ir = FFNOp(rules=list(rules))


# --- Marker extraction ------------------------------------------------------

_MARK_EQ_RE = re.compile(r"mark == (\w+)")
_NOT_MARK_EQ_RE = re.compile(r"NOT mark == (\w+)")


def positive_markers(text: str) -> set[str]:
    """Marker roles that the predicate text claims `mark == X` for, *outside*
    of `NOT mark == X` occurrences.

    We strip `NOT mark == X` substrings first, then find remaining
    `mark == X`. This is a heuristic — adequate for the simple
    conjunctive shapes F-5 emits.
    """
    stripped = _NOT_MARK_EQ_RE.sub("", text)
    return set(_MARK_EQ_RE.findall(stripped))


def negative_markers(text: str) -> set[str]:
    """Marker roles that the predicate text claims `NOT mark == X` for."""
    return set(_NOT_MARK_EQ_RE.findall(text))


def name_marker_tokens(rule_name: str) -> set[str]:
    """Marker roles implied by tokens in the rule name (stack0/mem/sp/...)."""
    lower = rule_name.lower()
    return {MARKER_CANON[w] for w in MARKER_WORDS if w in lower}


# --- Classification ---------------------------------------------------------


def classify(violation: dict) -> tuple[str, dict]:
    """Apply heuristics in priority order. Returns (category, debug_info).

    Priority (first match wins; downstream still recorded for debugging):
      1. Effective is unsatisfiable per S-3 -> D
      2. Effective's NOT-blockers include scope's marker -> D
      3. Effective's positive marker differs from scope's marker -> A
      4. Rule name marker matches effective marker but not scope -> A
      5. Effective contains placeholder/vacuous semantics phrases -> C
      6. Fall-through -> B
    """
    rule = violation.get("rule", "")
    scope_str = violation.get("scope", "")
    eff_str = violation.get("effective", "")

    eff_pos = positive_markers(eff_str)
    eff_neg = negative_markers(eff_str)
    scope_pos = positive_markers(scope_str)
    name_markers = name_marker_tokens(rule)

    # 1. Effective unsatisfiable structurally?
    eff_sat = True
    try:
        eff_pred = parse(eff_str)
        eff_sat = satisfiable(eff_pred)
    except Exception:
        # Treat parse failure conservatively as "unknown sat" (do NOT
        # mark D solely on parse failure); proceed with other heuristics.
        eff_sat = True

    info = {
        "rule": rule,
        "scope": scope_str,
        "effective_pos_markers": sorted(eff_pos),
        "effective_neg_markers": sorted(eff_neg),
        "scope_pos_markers": sorted(scope_pos),
        "name_markers": sorted(name_markers),
        "effective_satisfiable": eff_sat,
    }

    if not eff_sat:
        info["why"] = "effective predicate is unsatisfiable per S-3"
        return "D", info

    # 2. NOT-blockers include the scope's marker -> contradictory.
    if scope_pos and (scope_pos & eff_neg):
        info["why"] = (
            f"scope marker {sorted(scope_pos & eff_neg)} appears as a "
            f"hard-blocker in the effective predicate"
        )
        return "D", info

    # 3. Effective marker != scope marker.
    #    Scope mentions a marker; effective mentions a different one.
    if scope_pos and eff_pos and not (scope_pos & eff_pos):
        info["why"] = (
            f"effective marker {sorted(eff_pos)} differs from scope "
            f"marker {sorted(scope_pos)}"
        )
        # Refine: if the rule name agrees with the SCOPE (not effective),
        # then it really IS a scope-vs-conditions mismatch but the
        # author's INTENT lines up with the scope, suggesting Category B
        # (genuine E5 bug — conditions are wrong). If name agrees with
        # the EFFECTIVE, scope is mislabeled -> Category A.
        name_matches_eff = bool(name_markers & eff_pos)
        name_matches_scope = bool(name_markers & scope_pos)
        if name_matches_eff and not name_matches_scope:
            info["why"] += (
                "; rule name marker matches effective, not scope "
                "(scope was mislabeled)"
            )
            return "A", info
        if name_matches_scope and not name_matches_eff:
            info["why"] += (
                "; rule name marker matches scope (conditions appear "
                "to broaden firing outside intent)"
            )
            # This is closer to genuine E5 (B). But for the common
            # gated_write/hard-blocker shape, it usually reflects that
            # F-5 ignores the gate. Still classify as A by the task
            # rubric ("effective marker DIFFERENT from scope's marker"),
            # but note the nuance for review.
            info["nuance"] = "may be F-5 gate-ignoring artefact"
            return "A", info
        return "A", info

    # 4. Name agrees with effective, not with scope.
    if name_markers and (name_markers & eff_pos) and not (name_markers & scope_pos):
        info["why"] = (
            "rule name marker matches effective but not scope"
        )
        return "A", info

    # 5. Vacuous placeholder semantics leak into effective.
    if any(phrase in eff_str for phrase in VACUOUS_PHRASES):
        info["why"] = (
            "effective predicate contains vacuous-placeholder semantics "
            "(e.g. `is_byte OR NOT is_byte`)"
        )
        return "C", info

    # 6. Otherwise: probable genuine E5 bug.
    info["why"] = (
        "no marker mismatch / unsat detected; conditions appear to admit "
        "positions outside declared scope"
    )
    return "B", info


# --- Main -------------------------------------------------------------------


def main() -> int:
    out_dir = HERE.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = build_default_registry()
    op = make_tail_bit32_result_correction_op()
    rules = collect_rules_from_compiler_ir(op)

    fake = _FakeOpWrapper(rules)
    issues = verify_rule_scopes(fake, reg)
    violations = [i for i in issues if i.get("kind") == "scope_violation"]

    total = len(violations)

    classified = []
    counts = Counter()
    examples = defaultdict(list)
    family_counts = defaultdict(Counter)

    for v in violations:
        cat, info = classify(v)
        counts[cat] += 1
        classified.append({"violation": v, "category": cat, "info": info})

        # Build trim example payloads (limit string sizes for JSON sanity).
        if len(examples[cat]) < 25:
            examples[cat].append(
                {
                    "rule": v.get("rule"),
                    "scope": v.get("scope"),
                    "effective_pos_markers": info["effective_pos_markers"],
                    "effective_neg_markers": info["effective_neg_markers"],
                    "scope_pos_markers": info["scope_pos_markers"],
                    "name_markers": info["name_markers"],
                    "effective_satisfiable": info["effective_satisfiable"],
                    "why": info["why"],
                    "reason": v.get("reason", "")[:240],
                }
            )

        # Track rule-family counts (collapse trailing byte hex into XX).
        fam = re.sub(r"_[0-9a-f]{2}(_|$)", r"_XX\1", v.get("rule", ""))
        family_counts[cat][fam] += 1

    summary = {
        "op": "make_tail_bit32_result_correction_op",
        "total_violations": total,
        "by_category": {
            "A_scope_annotation_error": counts["A"],
            "B_genuine_e5_bug": counts["B"],
            "C_imprecise_semantics": counts["C"],
            "D_unsatisfiable": counts["D"],
        },
        "rule_families_per_category": {
            cat: dict(family_counts[cat])
            for cat in ("A", "B", "C", "D")
        },
        "examples_per_category": {
            cat: examples[cat][:10] for cat in ("A", "B", "C", "D")
        },
    }

    json_path = out_dir / "L10_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    md_path = out_dir / "L10_summary.md"
    write_markdown(md_path, summary)

    # Also write a full classified-rules dump so follow-up code can
    # bulk-fix without re-running this triage.
    detail_path = out_dir / "L10_classified_violations.json"
    with detail_path.open("w") as f:
        json.dump(
            [
                {
                    "rule": c["violation"]["rule"],
                    "scope": c["violation"]["scope"],
                    "category": c["category"],
                    "why": c["info"]["why"],
                    "effective_pos_markers": c["info"]["effective_pos_markers"],
                    "effective_neg_markers": c["info"]["effective_neg_markers"],
                    "scope_pos_markers": c["info"]["scope_pos_markers"],
                    "name_markers": c["info"]["name_markers"],
                    "effective_satisfiable": c["info"]["effective_satisfiable"],
                    "effective": c["violation"]["effective"][:600],
                }
                for c in classified
            ],
            f,
            indent=2,
        )

    print(f"total: {total}")
    print(f"A scope_annotation_error: {counts['A']}")
    print(f"B genuine_e5_bug:         {counts['B']}")
    print(f"C imprecise_semantics:    {counts['C']}")
    print(f"D unsatisfiable:          {counts['D']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {detail_path}")
    return 0


# --- Markdown formatter -----------------------------------------------------


def write_markdown(path: Path, summary: dict) -> None:
    total = summary["total_violations"]
    counts = summary["by_category"]
    families = summary["rule_families_per_category"]
    examples = summary["examples_per_category"]

    lines = []
    lines.append("# L10 scope_violations triage")
    lines.append("")
    lines.append(
        f"Op: `{summary['op']}` — total scope_violations: **{total}**"
    )
    lines.append("")
    lines.append("## Category totals")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for k, label in [
        ("A_scope_annotation_error", "A — scope-annotation error"),
        ("B_genuine_e5_bug", "B — genuine E5 bug"),
        ("C_imprecise_semantics", "C — imprecise semantics"),
        ("D_unsatisfiable", "D — unsatisfiable effective"),
    ]:
        lines.append(f"| {label} | {counts.get(k, 0)} |")
    lines.append("")

    lines.append("## Rule families per category")
    lines.append("")
    for cat in ("A", "B", "C", "D"):
        cat_fams = families.get(cat, {})
        if not cat_fams:
            continue
        lines.append(f"### Category {cat}")
        for fam, n in sorted(cat_fams.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{fam}`: {n}")
        lines.append("")

    lines.append("## Top examples per category")
    lines.append("")
    for cat in ("A", "B", "C", "D"):
        ex = examples.get(cat, [])[:5]
        if not ex:
            continue
        lines.append(f"### Category {cat}")
        lines.append("")
        for e in ex:
            lines.append(f"- **rule**: `{e['rule']}`")
            lines.append(f"  - scope: `{e['scope']}`")
            lines.append(
                f"  - effective markers: pos={e['effective_pos_markers']} "
                f"neg={e['effective_neg_markers']}"
            )
            lines.append(f"  - name markers: {e['name_markers']}")
            lines.append(
                f"  - effective_satisfiable: {e['effective_satisfiable']}"
            )
            lines.append(f"  - why: {e['why']}")
            if e.get("reason"):
                lines.append(f"  - verifier reason: {e['reason']}")
        lines.append("")

    lines.append("## Recommended next step per category")
    lines.append("")
    lines.append(
        "- **A — scope-annotation error**: bulk-fix by scanning each "
        "rule's effective marker and replacing the declared `scope=` "
        "with the matching marker. Safe automated rewrite once the "
        "name/effective agreement is confirmed."
    )
    lines.append(
        "- **B — genuine E5 bug**: surgical per-rule review. The "
        "rule's conditions admit positions outside intent; tighten "
        "the conditions (add a marker term, narrow byte_index, etc.)."
    )
    lines.append(
        "- **C — imprecise semantics**: tighten the registry semantics "
        "(`build_default_registry`) for the dims with placeholder "
        "`is_byte OR NOT is_byte` semantics so F-5 doesn't drop to "
        "vacuous tautologies."
    )
    lines.append(
        "- **D — unsatisfiable effective**: usually means F-5 is "
        "ignoring a gated_write `gate=` argument that supplies the "
        "true firing marker. Two complementary fixes: "
        "(1) teach F-5 to AND in the gate's semantics so the effective "
        "predicate is no longer contradictory; "
        "(2) revisit which conditions list `MARK_MEM, -1e6` as a hard "
        "blocker when the rule actually fires at MEM positions (the "
        "blocker is double-counted with the scope/gate). "
        "Either fix collapses the entire D bucket without per-rule "
        "edits."
    )
    lines.append("")

    biggest_label, biggest_n = max(
        counts.items(), key=lambda kv: kv[1]
    )
    lines.append(
        f"**Biggest bucket:** {biggest_label} ({biggest_n}/{total}). "
        "See the corresponding recommendation above."
    )
    lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
