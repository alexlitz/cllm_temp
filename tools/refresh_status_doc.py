#!/usr/bin/env python3
"""Refresh c4_release/docs/STATUS.md.

Auto-updatable handoff doc that captures:
  - What works
  - What's broken
  - What's in flight
  - Priority next
  - Recently landed (last N commits)

The "what works / broken / in flight / priority" prose lives in a curated
front-matter block at the top of STATUS.md between BEGIN_CURATED /
END_CURATED markers. The tool preserves that block and rewrites only the
auto-generated tail (recent commits + scanned in-flight docs + footer).

Usage:
  python3 tools/refresh_status_doc.py            # update in place
  python3 tools/refresh_status_doc.py --check    # exit 1 if stale
  python3 tools/refresh_status_doc.py --stdout   # print, don't write

This is intentionally side-effect light (no compile, no model load).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "c4_release" / "docs"
STATUS_PATH = DOCS_DIR / "STATUS.md"

# Curated section delimiters. The tool never touches text between these
# markers; everything outside is regenerated.
CURATED_BEGIN = "<!-- BEGIN_CURATED -->"
CURATED_END = "<!-- END_CURATED -->"
AUTOGEN_BEGIN = "<!-- BEGIN_AUTOGEN -->"
AUTOGEN_END = "<!-- END_AUTOGEN -->"

# Investigation docs scanned for "in flight" surface.
IN_FLIGHT_PATTERNS = [
    re.compile(r"^SMOKE_.*\.md$"),
    re.compile(r"^CROSS_STEP_SSA.*\.md$"),
    re.compile(r"^.*ROOT_CAUSE.*\.md$"),
    re.compile(r"^VAR_CLUSTER.*\.md$"),
    re.compile(r"^EFFICIENT_MODE_FIX_GAP\.md$"),
    re.compile(r"^IR_INCREMENTAL_IMPROVEMENTS\.md$"),
    re.compile(r"^HANDOFF_\d{4}_\d{2}_\d{2}\.md$"),
]

DEFAULT_CURATED = f"""# c4_release STATUS

Auto-refreshed by `tools/refresh_status_doc.py`. The curated narrative
between the markers below is hand-maintained; the recent-commits and
in-flight scan below the markers are regenerated.

A new session should be able to read this single file and skip the 30+
other docs in `c4_release/docs/`.

{CURATED_BEGIN}

## 1. What works

- **Structural compiler**: V1 imperative cells fully migrated
  (`imperative_trivial = 0`, `imperative_heavy = 0`), V1 no_ir = 0.
- **Compile path**: `compile_full_vm_dynamic(alu_mode='efficient',
  alu_mode='lookup', n_heads=8, ffn_hidden=4096, max_seq_len=8192)`
  works after the L10 d_model fix (`24cca5ee`, sibling time bomb fix
  `bda17245`).
- **Mixtral RMSNorm**: end-to-end through padded + non-padded HF export
  paths, regression-pinned (`24967832`, `e0b01183`).
- **Byte-identity gates**: declarative bakes (`null_terminator_detection`,
  `_l10_comparison_combine_rules`, etc.) are byte-identical to the
  imperative versions they replaced.
- **Cache hygiene**: LRU evictor for `compiled_vm/` disk cache landed
  (`cb52e54e`); safetensors warm-disk hit <0.5s (`f43463a2`).
- **V2 dynamic heads / GQA**: done.
- **V3 dim_ref corpus migration**: mostly done; CARRY+0 collapsed in
  L9 (`0d0012a3`).
- **Phase 7 / 8 / 9**: closed per phase-status table in
  `HANDOFF_2026_06_03.md`.

## 2. What's broken

- **Smoke pass rate**: 12 pass / 39 fail. 6 of the failures
  (`TestSmokeComparison::test_*_true`) trace to a missing OP_<NAME>
  decode in the L5 opcode-decode FFN — `OP_EQ`, `OP_NE`, `OP_LT`, etc.
  are zero at every block position. The default-leak / MARK_PC frame
  was wrong; see `SMOKE_COMPARISON_OP_DECODE_MISSING.md`.
- **1096 lookup-mode baseline**: ~11% pass (real declarations-only
  baseline). 5 distinct failure clusters identified.
- **L17.post_ops[0] block=27**: 58 of 384 1096 failures cluster here;
  margin uniformly -16.00, OUT_HI[0] = +9.4e6 runaway.
- **edge_pow2 sign-extension**: `2^3 = -24` — 3-bit value read as 8-bit
  signed.
- **if_eq branch collapse**: always takes one branch.
- **absdiff halt-horizon miss**: runs past the expected halt step.
- **L5 FFN slot collision**: `block.5.ffn.W_up` has 129 suppressor
  units reading OPCODE_BYTE_LO+1 / OPCODE_BYTE_HI+1 with magnitude
  ~1e3, but no writer to `OP_*` dims (187..217) exists outside the
  embedding heads.
- **Efficient-mode fix gap**: declarative IR fixes to L8/L10/L11/L13
  are inert for smoke because efficient mode replaces those FFNs
  wholesale.

## 3. In flight (10-step incremental plan)

Per `IR_INCREMENTAL_IMPROVEMENTS.md`:

| Step | Description | Status |
|--:|---|---|
| 1 | Unblock smoke (find OP_* zeroer between L7-L14) | in flight |
| 2 | Strengthen one rule at a time (dominates_at) | not started |
| 3 | Bare-literal lint | not started |
| 4 | SSA value distinction (step=0 vs step=-1) | scaffolding (`ef6ef561`) |
| 5 | Drop produces/consumes_fresh as separate fields | in flight (waves 1-7 landed) |
| 6 | Auto-attribute 1096 failures | not started |
| 7 | Strict-mode compile gate | not started |
| 8 | Smoke pass-count CI metric | not started |
| 9 | Worktree + cache hygiene | partial (LRU landed; worktree cleanup pending) |
| 10 | STATUS.md + auto-refresh | **this commit** |

## 4. Priority next (top 3)

1. **Find the OP_* zeroer between L7 and L14** (Step 1). Smoke 12/39 and
   the 89% 1096 mass should move together once this lands. The
   blocker-1 cluster agent is targeting the cross-step rename in
   `l5_ops.py:527` (`OPCODE_BYTE_LO.*.-1`).
2. **Run the cross-step SSA rename audit corpus-wide** as a parallel
   root-cause sweep — Tier A revert ~30 records already documented
   (`979094ed`).
3. **Wire NormSpec / PositionalEncodingSpec / activation-spec
   dataclasses into `compile_full_vm_dynamic`** once runtime baseline
   stabilizes. Scaffolding landed (`9068b6b4`, `a563f925`); the four
   imperative kwargs can then retire.

{CURATED_END}
"""


def _run_git(args: list[str]) -> str:
    """Run a git command from the repo root and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def get_recent_commits(n: int = 20) -> list[tuple[str, str]]:
    """Return [(sha, subject), ...] for the most recent N commits."""
    out = _run_git(["log", f"-{n}", "--pretty=format:%h %s"])
    commits: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        commits.append((sha, subject))
    return commits


def scan_in_flight_docs() -> list[tuple[str, str]]:
    """Return [(name, first heading), ...] for active investigation docs."""
    if not DOCS_DIR.exists():
        return []
    hits: list[tuple[str, str]] = []
    for entry in sorted(DOCS_DIR.iterdir()):
        if not entry.is_file():
            continue
        name = entry.name
        if not any(p.match(name) for p in IN_FLIGHT_PATTERNS):
            continue
        # Read first heading as the one-line summary.
        title = name
        try:
            with entry.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if raw.startswith("#"):
                        title = raw.lstrip("#").strip()
                        break
        except OSError:
            pass
        hits.append((name, title))
    return hits


def render_autogen_block(commits: list[tuple[str, str]],
                         in_flight: list[tuple[str, str]],
                         when: str) -> str:
    lines: list[str] = []
    lines.append(AUTOGEN_BEGIN)
    lines.append("")
    lines.append("## 5. Recently landed (auto-generated)")
    lines.append("")
    lines.append(f"Last {len(commits)} commits as of {when}:")
    lines.append("")
    for sha, subj in commits:
        lines.append(f"- `{sha}` {subj}")
    lines.append("")
    lines.append("## 6. Active investigation docs (auto-scanned)")
    lines.append("")
    if in_flight:
        for name, title in in_flight:
            lines.append(f"- `c4_release/docs/{name}` — {title}")
    else:
        lines.append("- (none found)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"_Auto-refreshed by `tools/refresh_status_doc.py` at {when}. "
        f"Edit curated sections between BEGIN_CURATED / END_CURATED; "
        f"recent commits + in-flight scan are regenerated._"
    )
    lines.append("")
    lines.append(AUTOGEN_END)
    return "\n".join(lines)


def split_existing(text: str) -> tuple[str, str]:
    """Return (curated_block, _) — preserves curated front matter.

    If the file does not yet exist or lacks markers, return the default
    curated template and an empty placeholder.
    """
    if CURATED_BEGIN not in text or CURATED_END not in text:
        return DEFAULT_CURATED, ""
    # Keep everything up to and including CURATED_END verbatim.
    head_end = text.index(CURATED_END) + len(CURATED_END)
    return text[:head_end] + "\n", ""


def build_doc(existing: str | None) -> str:
    when = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ")
    commits = get_recent_commits(20)
    in_flight = scan_in_flight_docs()
    autogen = render_autogen_block(commits, in_flight, when)
    if existing is None:
        curated = DEFAULT_CURATED
    else:
        curated, _ = split_existing(existing)
    return curated + "\n" + autogen + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if STATUS.md is stale")
    parser.add_argument("--stdout", action="store_true",
                        help="print to stdout instead of writing")
    args = parser.parse_args(argv)

    existing = STATUS_PATH.read_text(encoding="utf-8") if STATUS_PATH.exists() else None
    new_doc = build_doc(existing)

    if args.stdout:
        sys.stdout.write(new_doc)
        return 0

    if args.check:
        if existing != new_doc:
            sys.stderr.write(
                f"STATUS.md is stale; run tools/refresh_status_doc.py\n"
            )
            return 1
        return 0

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(new_doc, encoding="utf-8")
    sys.stdout.write(f"wrote {STATUS_PATH}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
