#!/usr/bin/env python3
"""Aggregate a 1096 teacher-forced lowering audit log into a structured summary."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROW_RE = re.compile(r"\[1096-lowering\] id=(\d{4}) status=(\S+) desc='([^']*)' .*$")
SUMMARY_RE = re.compile(r"\[1096-lowering-summary\] (.*)$")
RUNTIME_RE = re.compile(r"^=+\s*(\d+) passed in ([0-9.]+s.*)$")
STEP_SLOT_TOKEN_RE = re.compile(r"\bstep(\d+):([A-Za-z0-9_\[\]]+)\b")


def parse_kv_segment(text: str) -> Dict[str, str]:
    """Parse a flat 'k=v k=v ...' segment. Stops at unmatched grouping if any."""
    out: Dict[str, str] = {}
    # naive split on whitespace, since values are simple tokens here.
    i = 0
    tokens = text.split()
    for tok in tokens:
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def extract_first_segment(row: str, anchor: str, terminator_re: re.Pattern) -> Optional[str]:
    """Return substring starting just after `anchor=` up to next major key marker."""
    idx = row.find(anchor + "=")
    if idx < 0:
        return None
    start = idx + len(anchor) + 1
    rest = row[start:]
    m = terminator_re.search(rest)
    end = m.start() if m else len(rest)
    return rest[:end].strip()


# Boundaries delineating the major top-level sections in each row.
SECTION_BOUNDARY = re.compile(
    r"\s(?=(?:first_fatal|first_info|detail|final|first_loss_after_support|band_contracts|violations|projection_diag)=)"
)

# Inside a first_fatal / first_info segment, the keys appear flat; stop the
# segment when we hit the next top-level key.
NEXT_TOP_KEY = re.compile(
    r"\s(?=(?:first_fatal|first_info|detail|final|first_loss_after_support)=)"
)

KIND_KEYS = ("wrong_argmax", "wrong_output_byte", "low_head_margin",
             "output_band_contract", "wrong_token")

# Final block extraction.
FINAL_BLOCK_RE = re.compile(r"final=block=(\d+) layer=(\d+) ")
FLAS_BLOCK_RE = re.compile(r"first_loss_after_support=block=(\d+) layer=(\d+) ")


def parse_fatal_or_info(row: str, key: str) -> Optional[Dict[str, str]]:
    """Extract `key=...` section from a row, returning parsed k/v map.

    Also captures the bare `stepN:SLOT` token into the dict under the synthetic
    key `__step_slot__` (e.g. `step0:SP_byte0`).
    """
    idx = row.find(key + "=")
    if idx < 0:
        return None
    rest = row[idx + len(key) + 1:]
    m = NEXT_TOP_KEY.search(rest)
    seg = rest[:m.start()] if m else rest
    kv = parse_kv_segment(seg)
    sm = STEP_SLOT_TOKEN_RE.search(seg)
    if sm:
        kv["__step_slot__"] = f"step{sm.group(1)}:{sm.group(2)}"
        kv["__step__"] = f"step{sm.group(1)}"
        kv["__slot__"] = sm.group(2)
    return kv


def parse_family(desc: str) -> str:
    # examples: add_0: ..., var_simple_24: ..., mod_49: ...
    head = desc.split(":", 1)[0]
    # strip trailing digits/underscore-digits
    m = re.match(r"^([a-zA-Z_]+?)_\d+$", head)
    if m:
        return m.group(1)
    return head


def short_desc(desc: str, idx: str) -> str:
    return f"{idx}:{desc}"


def parse_log(log_path: Path) -> Dict:
    rows: List[Dict] = []
    summary_line: Optional[str] = None
    runtime_line: Optional[str] = None
    visible_segments = 0

    with log_path.open("r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = ROW_RE.search(line)
            if m:
                idx, status, desc = m.group(1), m.group(2), m.group(3)
                row: Dict = {"idx": idx, "status": status, "desc": desc, "line": line}
                row["first_fatal"] = parse_fatal_or_info(line, "first_fatal")
                row["first_info"] = parse_fatal_or_info(line, "first_info")
                # Count visible segments (fatal + info appearances).
                vis = 0
                if row["first_fatal"]:
                    vis += 1
                if row["first_info"]:
                    vis += 1
                visible_segments += vis
                row["family"] = parse_family(desc)

                # Final / FLAS blocks (for support analysis).
                fm = FINAL_BLOCK_RE.search(line)
                if fm:
                    row["final_block"] = f"block{fm.group(1)}/layer{fm.group(2)}"
                fsm = FLAS_BLOCK_RE.search(line)
                if fsm:
                    row["flas_block"] = f"block{fsm.group(1)}/layer{fsm.group(2)}"
                rows.append(row)
                continue

            sm = SUMMARY_RE.search(line)
            if sm:
                summary_line = line
                continue
            rm = RUNTIME_RE.match(line)
            if rm:
                runtime_line = line
                continue

    return {
        "rows": rows,
        "summary_line": summary_line,
        "runtime_line": runtime_line,
        "visible_segments": visible_segments,
    }


def kind_set_from_segment(seg: Dict[str, str]) -> Tuple[str, ...]:
    kind_csv = seg.get("kind", "")
    if not kind_csv:
        return tuple()
    return tuple(sorted(set(k for k in kind_csv.split(",") if k)))


def fmt_counter(counter: Counter, joiner: str = ", ") -> str:
    return joiner.join(f"{k}:{v}" for k, v in counter.most_common())


def fmt_counter_keyed(counter: Counter, prefix: str, joiner: str = ", ") -> str:
    return joiner.join(f"{prefix}:{k}:{v}" for k, v in counter.most_common())


def build_aggregate(log_path: Path, parsed: Dict) -> str:
    rows = parsed["rows"]
    out: List[str] = []

    # Recipe-format heading first, so awk extractor finds it.
    out.append("SUMMARY")
    status_counter: Counter = Counter(r["status"] for r in rows)
    family_counter: Counter = Counter(r["family"] for r in rows)
    fatal_rows = sum(1 for r in rows if r["status"].startswith("fatal"))
    info_rows = sum(1 for r in rows if r["status"].startswith("info"))
    clean_rows = sum(1 for r in rows if r["status"] == "clean")
    out.append(f"log={log_path}")
    out.append(f"row_count={len(rows)} visible_failure_segments={parsed['visible_segments']}")
    if parsed["summary_line"]:
        out.append(parsed["summary_line"])
    if parsed["runtime_line"]:
        out.append(parsed["runtime_line"])
    out.append(f"status_summary={fmt_counter(status_counter)}")
    out.append(
        f"row_severity=fatal_rows:{fatal_rows} info_only_rows:{info_rows} clean_rows:{clean_rows}"
    )
    out.append(f"families={fmt_counter(family_counter)}")
    out.append("")

    # FAILURE_KIND_COUNTS – severity x kind for visible first_fatal/first_info segments.
    out.append("FAILURE_KIND_COUNTS")
    combined_counter: Counter = Counter()
    atomic_counter: Counter = Counter()
    combined_by_sev: Counter = Counter()
    atomic_by_sev: Counter = Counter()
    for r in rows:
        for sev_label, key in (("fatal", "first_fatal"), ("info", "first_info")):
            seg = r.get(key)
            if not seg:
                continue
            kinds = kind_set_from_segment(seg)
            if not kinds:
                continue
            csv_kinds = ",".join(kinds)
            combined_counter[csv_kinds] += 1
            combined_by_sev[(sev_label, csv_kinds)] += 1
            for k in kinds:
                atomic_counter[k] += 1
                atomic_by_sev[(sev_label, k)] += 1
    out.append(f"combined={fmt_counter(combined_counter)}")
    out.append(f"atomic={fmt_counter(atomic_counter)}")
    out.append(
        "combined_by_severity="
        + ", ".join(f"{sev}:{kc}:{cnt}" for (sev, kc), cnt in combined_by_sev.most_common())
    )
    out.append(
        "atomic_by_severity="
        + ", ".join(f"{sev}:{k}:{cnt}" for (sev, k), cnt in atomic_by_sev.most_common())
    )
    out.append("")

    # FIRST_FATAL_SLOT_COUNTS – the histogram requested in the recipe.
    out.append("FIRST_FATAL_SLOT_COUNTS")
    slot_counter: Counter = Counter()
    step_slot_counter: Counter = Counter()
    family_step_slot_counter: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        seg = r.get("first_fatal")
        if not seg:
            continue
        step_slot = seg.get("__step_slot__")
        if not step_slot:
            continue
        slot = seg.get("__slot__", step_slot.split(":", 1)[-1])
        slot_counter[slot] += 1
        step_slot_counter[step_slot] += 1
        family_step_slot_counter[(r["family"], step_slot)] += 1
        examples[step_slot].append(short_desc(r["desc"], r["idx"]))
    out.append(f"slots={fmt_counter(slot_counter)}")
    out.append(f"step_slots={fmt_counter(step_slot_counter)}")
    out.append(
        "family_step_slots="
        + ", ".join(
            f"{fam}->{ss}:{cnt}" for (fam, ss), cnt in family_step_slot_counter.most_common()
        )
    )
    for ss, lst in examples.items():
        out.append(f"first_fatal_examples[{ss}]=" + "; ".join(lst[:4]))
    out.append("")

    # FIRST_INFO_SLOT_COUNTS – useful sibling histogram.
    out.append("FIRST_INFO_SLOT_COUNTS")
    info_slot_counter: Counter = Counter()
    info_step_slot_counter: Counter = Counter()
    for r in rows:
        seg = r.get("first_info")
        if not seg:
            continue
        step_slot = seg.get("__step_slot__")
        if not step_slot:
            continue
        slot = seg.get("__slot__", step_slot.split(":", 1)[-1])
        info_slot_counter[slot] += 1
        info_step_slot_counter[step_slot] += 1
    out.append(f"slots={fmt_counter(info_slot_counter)}")
    out.append(f"step_slots={fmt_counter(info_step_slot_counter)}")
    out.append("")

    # SUPPORT / final-block summary.
    out.append("FINAL_BLOCK_COUNTS")
    final_blocks = Counter(r.get("final_block") for r in rows if r.get("final_block"))
    flas_blocks = Counter(r.get("flas_block") for r in rows if r.get("flas_block"))
    out.append(f"final_blocks={fmt_counter(final_blocks)}")
    out.append(f"flas_blocks={fmt_counter(flas_blocks)}")
    out.append("")

    # PER_ID status (compact one-line list).
    out.append("PER_ID_STATUS")
    out.append("# id status family desc(short)")
    for r in rows:
        short = r["desc"]
        if len(short) > 60:
            short = short[:57] + "..."
        out.append(f"{r['idx']} {r['status']} {r['family']} {short}")
    return "\n".join(out) + "\n"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    parsed = parse_log(args.log)
    agg = build_aggregate(args.log, parsed)
    args.out.write_text(agg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
