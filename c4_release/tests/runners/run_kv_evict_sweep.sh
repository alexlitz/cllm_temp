#!/usr/bin/env bash
# Sweep the 1096 declarative neural diagnostic across kv_cache_max_tokens
# ceilings to detect off-by-one eviction bugs or thresholds where divergences
# appear (eviction-correlated correctness).
#
# Runs:
#   1. test_batched_kv_eviction_validation.py once (unit eviction-window parity)
#   2. The 1096 diagnostic over a 32-id representative subset for each ceiling
#      in {32, 64, 128, 256, unbounded}, one id per pytest invocation since
#      C4_1096_OFFSET/LIMIT only supports contiguous ranges.
#
# Ceiling knob: C4_BATCH_KV_MAX_TOKENS (see batched_pure_neural.py:264-275).
# The "unbounded" entry omits the env var so the runner uses its 65_536-token
# default, which is effectively unbounded for 1096 program traces.
#
# Usage (from the repo root, i.e. the directory containing c4_release/):
#   c4_release/tests/runners/run_kv_evict_sweep.sh [LOG_DIR]
#
# Outputs:
#   ${LOG_DIR}/eviction_validation.log -- unit test pass/fail
#   ${LOG_DIR}/kvmax<C>.log            -- per-ceiling [1096-summary] per id
#   ${LOG_DIR}/SUMMARY.md              -- per-id x per-ceiling ok grid + totals

set -euo pipefail

LOG_DIR="${1:-.agent-logs/kv-evict-sweep}"
mkdir -p "${LOG_DIR}"

IDS=(
  5 32 75 150 210 260 310 360 410 425 470 520 555 580 620 660
  700 740 780 810 830 855 880 905 925 955 975 1000 1020 1055 1075 1090
)
CEILINGS=(32 64 128 256 unbounded)

TEST_NODE="c4_release/tests/test_1096_neural_declarative_diagnostic.py::test_1096_neural_declarative_diagnostic_slice"

PYTHONPATH=.:c4_release timeout 120 python -m pytest -q -s \
  c4_release/tests/test_batched_kv_eviction_validation.py --tb=short \
  > "${LOG_DIR}/eviction_validation.log" 2>&1

for CEIL in "${CEILINGS[@]}"; do
  LOG_FILE="${LOG_DIR}/kvmax${CEIL}.log"
  # Resume mode: don't truncate existing logs, instead skip IDs already done.
  touch "${LOG_FILE}"
  for ID in "${IDS[@]}"; do
    if grep -q "^id=${ID} " "${LOG_FILE}"; then
      continue
    fi
    kv_env=()
    if [[ "${CEIL}" != "unbounded" ]]; then
      kv_env=(C4_BATCH_KV_MAX_TOKENS="${CEIL}")
    fi
    line=$(
      env \
        C4_1096_DIAG=1 C4_1096_DIAG_ASSERT=0 \
        C4_1096_OFFSET="${ID}" C4_1096_LIMIT=1 \
        C4_1096_TRACE_FAILURES=0 C4_BATCH_CHUNK=1 \
        C4_DECLARATIONS_ONLY_BAKE=1 C4_SPEC_K=0 \
        C4_BATCH_USE_KV_CACHE=1 \
        "${kv_env[@]}" \
        PYTHONPATH=.:c4_release \
        timeout 240 python -m pytest -q -s "${TEST_NODE}" --tb=no 2>&1 \
        | grep "1096-summary" \
        || echo "[1096-summary] mode=final-output selected=0 ok=ERR divergences=0 errors=1 suite_mismatches=0"
    )
    printf 'id=%s %s\n' "${ID}" "${line}" >> "${LOG_FILE}"
  done
done

python3 - "${LOG_DIR}" "${IDS[@]}" <<'PY'
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
ids = [int(x) for x in sys.argv[2:]]
ceilings = ["32", "64", "128", "256", "unbounded"]

pat = re.compile(r"id=(\d+).*ok=(\S+)\s+divergences=(\S+)\s+errors=(\S+)")

results: dict[int, dict[str, tuple[str, str, str]]] = {i: {} for i in ids}
totals: dict[str, dict[str, int]] = {
    c: {"ok": 0, "div": 0, "err": 0} for c in ceilings
}

for ceil in ceilings:
    log = log_dir / f"kvmax{ceil}.log"
    if not log.exists():
        continue
    for line in log.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        i = int(m.group(1))
        ok, div, err = m.group(2), m.group(3), m.group(4)
        results[i][ceil] = (ok, div, err)
        for label, val in (("ok", ok), ("div", div), ("err", err)):
            if val.isdigit():
                totals[ceil][label] += int(val)

out = []
out.append("# KV eviction max_tokens sweep -- 1096 declarative neural diagnostic")
out.append("")
out.append("Per-id ok count (1 = match, 0 = divergence, ERR = pytest failure)")
out.append("for each `C4_BATCH_KV_MAX_TOKENS` ceiling. `unbounded` omits the env")
out.append("var so the runner falls back to its 65_536-token default.")
out.append("")

val_log = log_dir / "eviction_validation.log"
if val_log.exists():
    tail = [l for l in val_log.read_text().splitlines() if "passed" in l or "failed" in l]
    out.append(f"Eviction unit tests: `{tail[-1] if tail else 'no result line'}`")
    out.append("")

header = "| id | " + " | ".join(ceilings) + " |"
sep = "|----|" + "|".join(["------"] * len(ceilings)) + "|"
out.append(header)
out.append(sep)
for i in ids:
    cells = [results[i].get(c, ("-",))[0] for c in ceilings]
    out.append(f"| {i} | " + " | ".join(cells) + " |")

out.append("")
out.append("## Totals")
out.append("")
out.append("| metric | " + " | ".join(ceilings) + " |")
out.append("|--------|" + "|".join(["------"] * len(ceilings)) + "|")
for metric, key in (("ok", "ok"), ("divergences", "div"), ("errors", "err")):
    cells = [str(totals[c][key]) for c in ceilings]
    out.append(f"| {metric} | " + " | ".join(cells) + " |")

out.append("")
ok_counts = {c: totals[c]["ok"] for c in ceilings}
baseline = ok_counts["unbounded"]
regressions = [c for c in ceilings if c != "unbounded" and ok_counts[c] < baseline]
if regressions:
    out.append(
        "**Eviction-correlated regression vs unbounded baseline:** "
        + ", ".join(
            f"kvmax={c} (ok={ok_counts[c]} vs {baseline})" for c in regressions
        )
    )
else:
    out.append(
        "**No eviction-ceiling regression** vs unbounded baseline "
        f"(ok counts: {ok_counts})."
    )

(log_dir / "SUMMARY.md").write_text("\n".join(out) + "\n")
print(f"Wrote {log_dir / 'SUMMARY.md'}")
PY
