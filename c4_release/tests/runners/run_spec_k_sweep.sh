#!/usr/bin/env bash
# Sweep the 1096 declarative neural diagnostic across speculation depths to
# detect spec_k-correlated divergences (speculator / correction bugs).
#
# Runs a 32-id representative subset for each C4_SPEC_K in {0, 8, 64, 128},
# one id per pytest invocation (C4_1096_OFFSET/LIMIT only supports contiguous
# ranges), then aggregates ok-counts into a Markdown table.
#
# Per-id timeout bumped to 180s (from 120s) -- prior 120s caused many ERRs
# under GPU contention with sister agents running concurrently.
#
# Usage (from the repo root, i.e. the directory containing c4_release/):
#   c4_release/tests/runners/run_spec_k_sweep.sh [LOG_DIR]
#
# Outputs:
#   ${LOG_DIR}/k<K>.log     -- one "[1096-summary]" line per id, per K
#   ${LOG_DIR}/SUMMARY.md   -- 32 x 4 Markdown table of ok counts

set -euo pipefail

LOG_DIR="${1:-.agent-logs/spec-k-sweep}"
mkdir -p "${LOG_DIR}"

IDS=(
  5 32 75 150 210 260 310 360 410 425 470 520 555 580 620 660
  700 740 780 810 830 855 880 905 925 955 975 1000 1020 1055 1075 1090
)
SPEC_KS=(0 8 64 128)

TEST_NODE="c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice"

for K in "${SPEC_KS[@]}"; do
  : > "${LOG_DIR}/k${K}.log"
  for ID in "${IDS[@]}"; do
    line=$(
      C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 \
      C4_1096_OFFSET="${ID}" C4_1096_LIMIT=1 \
      C4_1096_TRACE_FAILURES=0 C4_BATCH_CHUNK=1 \
      C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K="${K}" C4_BATCH_USE_KV_CACHE=0 \
      PYTHONPATH=.:c4_release \
      timeout 180 python -m pytest -q -s "${TEST_NODE}" --tb=no 2>&1 \
        | grep "1096-summary" || echo "[1096-summary] mode=final-output selected=0 ok=ERR divergences=0 errors=1 suite_mismatches=0"
    )
    printf 'id=%s %s\n' "${ID}" "${line}" >> "${LOG_DIR}/k${K}.log"
  done
done

python3 - "${LOG_DIR}" "${IDS[@]}" <<'PY'
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
ids = [int(x) for x in sys.argv[2:]]
spec_ks = [0, 8, 64, 128]

pat = re.compile(r"id=(\d+).*ok=(\S+)\s+divergences=(\S+)\s+errors=(\S+)")

# results[id][k] = (ok, div, err)
results: dict[int, dict[int, tuple[str, str, str]]] = {i: {} for i in ids}
totals: dict[int, dict[str, int]] = {k: {"ok": 0, "div": 0, "err": 0} for k in spec_ks}

for k in spec_ks:
    log = log_dir / f"k{k}.log"
    if not log.exists():
        continue
    for line in log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        i, ok, div, err = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        results[i][k] = (ok, div, err)
        if ok.isdigit():
            totals[k]["ok"] += int(ok)
        if div.isdigit():
            totals[k]["div"] += int(div)
        if err.isdigit():
            totals[k]["err"] += int(err)

out = []
out.append("# spec_k sweep -- 1096 declarative neural diagnostic")
out.append("")
out.append("Per-id ok count (1 = match, 0 = divergence, ERR = pytest failure)")
out.append("for each `C4_SPEC_K` value. 32-id representative subset.")
out.append("")
header = "| id | " + " | ".join(f"k={k}" for k in spec_ks) + " |"
sep = "|----|" + "|".join(["------"] * len(spec_ks)) + "|"
out.append(header)
out.append(sep)
for i in ids:
    cells = []
    for k in spec_ks:
        v = results[i].get(k)
        cells.append("-" if v is None else v[0])
    out.append(f"| {i} | " + " | ".join(cells) + " |")

out.append("")
out.append("## Totals")
out.append("")
out.append("| metric | " + " | ".join(f"k={k}" for k in spec_ks) + " |")
out.append("|--------|" + "|".join(["------"] * len(spec_ks)) + "|")
for metric, key in (("ok", "ok"), ("divergences", "div"), ("errors", "err")):
    cells = [str(totals[k][key]) for k in spec_ks]
    out.append(f"| {metric} | " + " | ".join(cells) + " |")

out.append("")
ok_counts = {k: totals[k]["ok"] for k in spec_ks}
baseline = ok_counts[0]
spikes = [k for k in spec_ks if k != 0 and ok_counts[k] < baseline]
if spikes:
    out.append(
        "**Divergence spike vs k=0 baseline:** "
        + ", ".join(f"k={k} (ok={ok_counts[k]} vs {baseline})" for k in spikes)
    )
else:
    out.append("**No spec_k spike detected** vs k=0 baseline "
              f"(ok counts: {ok_counts}).")

(log_dir / "SUMMARY.md").write_text("\n".join(out) + "\n")
print(f"Wrote {log_dir / 'SUMMARY.md'}")
PY
