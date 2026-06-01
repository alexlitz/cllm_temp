"""Run verify_claims_static and produce a per-op drift catalog.

Outputs:
  .agent-logs/claims_catalog_2026_06_01.md
"""
import time
import traceback
from pathlib import Path

OUT_DIR = Path(".agent-logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[t={time.time():.1f}] start", flush=True)

from neural_vm.unified_compiler.decl_verifier import verify_claims_static
print(f"[t={time.time():.1f}] import done; calling verify_claims_static()", flush=True)

t0 = time.time()
try:
    report = verify_claims_static()
    err = None
except Exception as e:
    traceback.print_exc()
    report = None
    err = e

t_total = time.time() - t0
print(f"[t={time.time():.1f}] verify_claims_static done in {t_total:.1f}s", flush=True)

if report is None:
    print(f"FAILED: {err!r}", flush=True)
    raise SystemExit(2)

results = report.results
ok_n = sum(1 for r in results if getattr(r, "ok", False))
problem = [r for r in results if not getattr(r, "ok", False)]
inert_n = sum(1 for r in results if getattr(r, "inert", False))
print(f"total: {len(results)}, ok: {ok_n}, problems: {len(problem)}, inert: {inert_n}", flush=True)
print(f"has_errors: {report.has_errors() if hasattr(report, 'has_errors') else '?'}", flush=True)


def _issue_count(r):
    dn = len(getattr(r, "declared_but_not_written", []) or [])
    nd = len(getattr(r, "written_but_not_declared", []) or [])
    return dn, nd


sorted_problems = sorted(
    problem,
    key=lambda r: -(sum(_issue_count(r))),
)

lines = [
    "# Static claims catalog — 2026-06-01",
    "",
    "## Summary",
    f"- total ops verified: {len(results)}",
    f"- ok: {ok_n}",
    f"- with issues: {len(problem)}",
    f"- inert (bake fired but no observable diff): {inert_n}",
    f"- runtime: {t_total:.1f}s",
    f"- has_errors: {report.has_errors() if hasattr(report, 'has_errors') else 'unknown'}",
    "",
    "## Top 30 problem ops",
    "",
    "| Op | declared_but_not_written | written_but_not_declared | inert |",
    "|---|---:|---:|---|",
]
for r in sorted_problems[:30]:
    dn, nd = _issue_count(r)
    lines.append(
        f"| {getattr(r, 'op_name', '?')} | {dn} | {nd} | "
        f"{getattr(r, 'inert', False)} |"
    )
lines += ["", "## Per-op detail (top 20)", ""]
for r in sorted_problems[:20]:
    lines.append(f"### {getattr(r, 'op_name', '?')}")
    dn = list(getattr(r, "declared_but_not_written", []) or [])
    nd = list(getattr(r, "written_but_not_declared", []) or [])
    lines.append(f"- declared_but_not_written: {len(dn)}")
    for cell in dn[:10]:
        lines.append(f"  - `{cell}`")
    if len(dn) > 10:
        lines.append(f"  - ... and {len(dn)-10} more")
    lines.append(f"- written_but_not_declared: {len(nd)}")
    for cell in nd[:10]:
        lines.append(f"  - `{cell}`")
    if len(nd) > 10:
        lines.append(f"  - ... and {len(nd)-10} more")
    lines.append("")

out_path = OUT_DIR / "claims_catalog_2026_06_01.md"
out_path.write_text("\n".join(lines))
print(f"[t={time.time():.1f}] wrote {out_path}", flush=True)
print("DONE", flush=True)
