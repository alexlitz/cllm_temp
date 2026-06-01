"""L10 solo-strength scan only (cross-op skipped for runtime).

Writes .agent-logs/violation_catalog_2026_06_01/per_op/tail_bit32_result_correction.txt.
"""
import time
import traceback
from pathlib import Path

OUT_DIR = Path(".agent-logs/violation_catalog_2026_06_01/per_op")
OUT_DIR.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print(f"[t={time.time():.1f}] start", flush=True)

from neural_vm.dim_registry import build_default_registry
from neural_vm.unified_compiler.decl_verifier import (
    verify_rule_strength,
    verify_rule_scopes,
    _collect_ffn_rules_from_op,
)
from neural_vm.unified_compiler.ops.l10_ops import (
    make_tail_bit32_result_correction_op,
)

try:
    from neural_vm.unified_compiler.backbone_bounds import load_default_bounds
    bb_obj = load_default_bounds()
    bb = bb_obj.as_strength_bound if bb_obj is not None else None
except Exception:
    bb = None

registry = build_default_registry()
op = make_tail_bit32_result_correction_op()
n_rules = len(_collect_ffn_rules_from_op(op))
print(f"[t={time.time():.1f}] op built, {n_rules} rules", flush=True)

print(f"[t={time.time():.1f}] verify_rule_strength solo...", flush=True)
t1 = time.time()
try:
    s_issues = verify_rule_strength(op, registry, backbone_bounds=bb)
except Exception:
    traceback.print_exc()
    s_issues = []
t_solo = time.time() - t1
print(f"[t={time.time():.1f}] solo done in {t_solo:.1f}s — {len(s_issues)} issues", flush=True)

print(f"[t={time.time():.1f}] verify_rule_scopes...", flush=True)
t1 = time.time()
try:
    sc_issues = verify_rule_scopes(op, registry, require_scope=False)
except Exception:
    traceback.print_exc()
    sc_issues = []
t_scope = time.time() - t1
print(f"[t={time.time():.1f}] scope done in {t_scope:.1f}s — {len(sc_issues)} issues", flush=True)

# Group by rule family (prefix before last underscore-number)
import re

def family_of(rule_name):
    m = re.match(r"^(.*?)(?:_lo|_hi)?_\d+$", rule_name)
    return m.group(1) if m else rule_name

from collections import Counter
sv = [i for i in s_issues if i.get("kind") == "strength_violation"]
nd = [i for i in s_issues if i.get("kind") == "no_dominates_at"]
other = [i for i in s_issues if i.get("kind") not in ("strength_violation", "no_dominates_at")]
fam_sv = Counter(family_of(i.get("rule", "")) for i in sv)
fam_sc = Counter(family_of(i.get("rule", "")) for i in sc_issues)
fam_nd = Counter(family_of(i.get("rule", "")) for i in nd)

lines = [
    f"# Op: tail_bit32_result_correction (L10)",
    f"Total rules: {n_rules}",
    f"Solo strength runtime: {t_solo:.1f}s",
    f"Scope runtime: {t_scope:.1f}s",
    f"Note: cross-op skipped (too slow for L10's rule count)",
    "",
    "## STRENGTH (solo)",
    f"strength_violation: {len(sv)}",
    f"no_dominates_at: {len(nd)}",
    f"other: {len(other)}",
    "",
    "### strength_violation per family",
]
for fam, n in fam_sv.most_common():
    lines.append(f"  {n:>5}  {fam}")
lines.append("")
lines.append("### no_dominates_at per family")
for fam, n in fam_nd.most_common():
    lines.append(f"  {n:>5}  {fam}")
lines.append("")
lines.append("--- strength_violation top 50 ---")
for i in sv[:50]:
    lines.append(
        f"  rule={i.get('rule')!r} dim={i.get('output_dim')!r} "
        f"my={i.get('my_contribution', 0):.1f} comp={i.get('competing_max', 0):.1f} "
        f"shortfall={i.get('shortfall', 0):.1f} top={i.get('top_competitor')!r}"
    )
if len(sv) > 50:
    lines.append(f"  ... and {len(sv)-50} more")
lines.append("")
lines.append("## SCOPE")
lines.append(f"scope_violation: {len(sc_issues)}")
lines.append("")
lines.append("### scope_violation per family")
for fam, n in fam_sc.most_common():
    lines.append(f"  {n:>5}  {fam}")
lines.append("")
lines.append("--- scope_violation top 50 ---")
for i in sc_issues[:50]:
    reason = str(i.get("reason", ""))
    if len(reason) > 250:
        reason = reason[:250] + "..."
    lines.append(f"  rule={i.get('rule')!r} reason={reason}")

out_path = OUT_DIR / "tail_bit32_result_correction.txt"
out_path.write_text("\n".join(lines))
print(f"[t={time.time():.1f}] wrote {out_path}", flush=True)
print(f"DONE total={time.time()-t0:.1f}s", flush=True)
