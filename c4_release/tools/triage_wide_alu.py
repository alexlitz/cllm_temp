"""Triage wide-byte ALU (MUL/DIV/MOD/SHL/SHR/large-numeric) failures
in the 1096 declarative diagnostic sweep.

Read-only. Aggregates:
  * test_suite_1000.py program descriptions filtered to wide-byte ALU patterns
  * .agent-logs/sweep-2026-06-01/shard_*.log per-row ok / diverge / error

Outputs a markdown table on stdout (or to --out path) grouping diverging rows
by failure shape.

Usage:
  python tools/triage_wide_alu.py --sweep-dir .agent-logs/sweep-2026-06-01 \\
      --out .agent-logs/wide_alu_triage_2026_06_01.md
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the package importable without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.test_suite_1000 import generate_test_programs  # noqa: E402


# Category buckets keyed by `desc` prefix in test_suite_1000.
# Per generate_test_programs(), categories appear in this order with these
# counts (id offsets stable due to fixed Random(42) seed):
#   add (50)  sub (50)  mul (50)  div (50)         -> ids 0..199    arith
#   mod (50)                                       -> ids 200..249
#   var_simple (25) var_mul (25) var_three (25) var_update (25)
#                                                  -> ids 250..349
#   if_gt/lt/eq/var (25 ea)                        -> ids 350..449
#   loop_sum/countdown/mul/pow2 (25 ea)            -> ids 450..549
#   func_identity/add/mul/square/max/min (25 ea)   -> ids 550..699
#   rec_factorial/fib/sum/power (25 ea)            -> ids 700..799
#   expr_add_mul/paren/mul_div/mod (25 ea)         -> ids 800..899
#   gcd (50)                                       -> ids 900..949
#   nested_quad (25) nested_sumsq (25)             -> ids 950..999
#   edge_* (50)                                    -> ids 1000..1049
#   absdiff (25)                                   -> ids 1050..1074
#   bool_and (25)                                  -> ids 1075..1099


WIDE_ALU_PREFIXES = {
    # direct wide-byte ALU on multi-digit literals
    "mul_": "MUL_direct",
    "div_": "DIV_direct",
    "mod_": "MOD_direct",
    "var_mul_": "MUL_via_var",
    "func_mul_": "MUL_via_func",
    "func_square_": "MUL_square",
    "loop_mul_": "MUL_via_loop_add",       # implements * by repeated addition
    "loop_pow2_": "SHL_via_repeated_mul2", # result = result * 2 in loop
    "expr_add_mul_": "MUL_in_expression",
    "expr_paren_": "MUL_in_expression",    # (a+b)*c
    "expr_mul_div_": "MUL_DIV_combined",
    "expr_mod_": "MOD_in_expression",
    "gcd_": "MOD_iterative",                # while (b != 0) b = a % b
    "rec_power_": "MUL_recursive",
    "rec_factorial_": "MUL_recursive",
    "nested_quad_": "MUL_via_nested",       # *2 twice
    "nested_sumsq_": "MUL_via_nested",      # x*x + y*y
}


def classify_program(desc: str) -> Optional[str]:
    """Return a sub-cluster tag if `desc` belongs to the wide-byte ALU set."""
    base = desc.split(":", 1)[0].strip()
    for prefix, tag in WIDE_ALU_PREFIXES.items():
        if base.startswith(prefix):
            return tag
    return None


def parse_desc_args(desc: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Pull up to 3 integer operands out of a description string."""
    # Examples we want to match operands from:
    #   "mul_3: 17 * 92"  -> (17, 92, None)
    #   "div_5: 100 / 4"  -> (100, 4, None)
    #   "mod_7: 154 % 8"  -> (154, 8, None)
    #   "expr_mod_2: 89%10+3" -> (89, 10, 3)
    #   "expr_mul_div_4: 30*40/5" -> (30, 40, 5)
    #   "rec_power_1: 3^4"  -> (3, 4, None)
    #   "gcd_0: gcd(120, 75)" -> (120, 75, None)
    #   "func_mul_2: mul(7, 11)" -> (7, 11, None)
    #   "loop_pow2_4: 2^7" -> (2, 7, None)
    nums = [int(x) for x in re.findall(r"-?\d+", desc)]
    # Drop the leading "<prefix>_<i>" index when present so the operand stream
    # starts with the program's actual inputs.
    if ":" in desc:
        # Operands appear after the colon — re-extract from the right side.
        right = desc.split(":", 1)[1]
        nums = [int(x) for x in re.findall(r"-?\d+", right)]
    if not nums:
        return (None, None, None)
    a = nums[0] if len(nums) >= 1 else None
    b = nums[1] if len(nums) >= 2 else None
    c = nums[2] if len(nums) >= 3 else None
    return (a, b, c)


_ROW_RE = re.compile(
    r"id=(?P<id>\d+)\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"status=(?P<status>\S+)\s+"
    r"suite_decl=(?P<suite_decl>\S+)\s+"
    r"desc='(?P<desc>[^']*)'\s+"
    r"expected=(?P<expected>\S+)\s+"
    r"decl=(?P<decl>\S+)(?:\s+decl_steps=(?P<decl_steps>\S+))?"
    r"(?:\s+neural=(?P<neural>\S+))?"
)


def parse_sweep_logs(sweep_dir: str) -> Dict[int, Dict[str, str]]:
    """Parse all shard_*.log files. Return {id -> {status, expected, decl, neural, desc}}."""
    rows: Dict[int, Dict[str, str]] = {}
    paths = sorted(glob.glob(os.path.join(sweep_dir, "shard_*.log")))
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if "1096-diag" not in line:
                    continue
                m = _ROW_RE.search(line)
                if not m:
                    continue
                row_id = int(m.group("id"))
                rows[row_id] = {
                    "status": m.group("status"),
                    "expected": m.group("expected"),
                    "decl": m.group("decl"),
                    "neural": m.group("neural") or "MISSING",
                    "desc": m.group("desc"),
                    "shard": os.path.basename(path),
                }
    return rows


def safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def failure_shape(tag: str, expected: int, neural: Optional[int],
                  a: Optional[int], b: Optional[int]) -> str:
    """Heuristic grouping of (cluster, expected, neural) -> human-readable shape."""
    if neural is None:
        return f"{tag}::neural_None"
    if expected == neural:
        return f"{tag}::correct"

    # Bit-width analysis: did the neural answer at least fit in the byte?
    exp_hi = (expected >> 8) & 0xFF if expected is not None else 0
    exp_lo = expected & 0xFF if expected is not None else 0
    neu_lo = neural & 0xFF

    if expected is not None and expected > 255 and neural <= 255:
        return f"{tag}::high_byte_lost"
    if expected is not None and neural == exp_lo:
        return f"{tag}::low_byte_only"
    if expected is not None and (expected ^ neural) & 0xFF == 0 and neural != expected:
        return f"{tag}::high_byte_corrupt_low_ok"
    if tag.startswith("MUL") and b is not None and a is not None and a * b <= 255 and neural != a * b:
        return f"{tag}::single_byte_mul_wrong"
    if tag.startswith("MUL") and a is not None and b is not None and a * b > 255:
        return f"{tag}::multi_byte_mul_wrong"
    if tag.startswith("MOD") and b is not None and (b & (b - 1)) == 0:
        return f"{tag}::pow2_modulus_wrong"
    if tag.startswith("MOD"):
        return f"{tag}::nonpow2_modulus_wrong"
    if tag.startswith("DIV") and b is not None and (b & (b - 1)) == 0:
        return f"{tag}::pow2_divisor_wrong"
    if tag.startswith("DIV"):
        return f"{tag}::nonpow2_divisor_wrong"
    if tag == "SHL_via_repeated_mul2" and b is not None:
        if b < 8:
            return f"{tag}::shift_lt8_wrong"
        return f"{tag}::shift_ge8_wrong"
    return f"{tag}::other_diverge"


def render(programs: List[Tuple[str, int, str]],
           sweep: Dict[int, Dict[str, str]]) -> str:
    lines: List[str] = []
    lines.append("# Wide-Byte ALU Triage (1096 sweep, 2026-06-01)\n")
    lines.append(
        "Read-only triage of MUL/DIV/MOD/SHL/SHR + multi-byte-numeric rows in the "
        "1096 declarative diagnostic suite. Source data: "
        "`tests/test_suite_1000.py` (Random(42)-seeded) and "
        "`.agent-logs/sweep-2026-06-01/shard_*.log`.\n"
    )

    # 1. Membership counts.
    total_wide = 0
    per_tag: Dict[str, List[int]] = defaultdict(list)
    for row_id, (src, expected, desc) in enumerate(programs):
        tag = classify_program(desc)
        if tag is None:
            continue
        total_wide += 1
        per_tag[tag].append(row_id)

    lines.append(f"## 1. Membership\n")
    lines.append(f"- **Total wide-byte ALU rows**: {total_wide} / {len(programs)}\n")
    lines.append("- **Sub-clusters (by source pattern)**:\n")
    for tag, ids in sorted(per_tag.items(), key=lambda kv: -len(kv[1])):
        rng = f"{min(ids)}-{max(ids)}"
        lines.append(f"  - `{tag}` : {len(ids):3d} rows (ids {rng})")
    lines.append("")

    # 2. Sweep coverage / status.
    swept_ids = set(sweep.keys())
    wide_ids = {i for ids in per_tag.values() for i in ids}
    covered = wide_ids & swept_ids
    missing = wide_ids - swept_ids
    status_counts: Counter[str] = Counter()
    for row_id in covered:
        status_counts[sweep[row_id]["status"]] += 1

    lines.append("## 2. Sweep status breakdown\n")
    lines.append(f"- wide-byte ALU rows present in sweep logs: {len(covered)} / {len(wide_ids)}")
    lines.append(f"- wide-byte ALU rows missing from sweep:    {len(missing)}")
    for st, n in status_counts.most_common():
        lines.append(f"  - status `{st}`: {n}")

    # Are any rows actually "correct" (decl == expected AND neural == expected)?
    correct = []
    diverge_decl_neural = []
    decl_wrong = []
    for row_id in covered:
        row = sweep[row_id]
        exp = safe_int(row["expected"])
        decl = safe_int(row["decl"])
        neu = safe_int(row["neural"])
        if exp is not None and decl is not None and exp == decl and neu == exp:
            correct.append(row_id)
        elif exp is not None and decl is not None and exp == decl and neu != exp:
            diverge_decl_neural.append(row_id)
        else:
            decl_wrong.append(row_id)
    lines.append("")
    lines.append(f"- end-to-end correct (decl == expected == neural): **{len(correct)}**")
    lines.append(f"- declarative bake correct, neural wrong:           **{len(diverge_decl_neural)}**")
    lines.append(f"- declarative bake itself wrong (rare):             **{len(decl_wrong)}**")
    lines.append("")

    # 3. Failure shapes.
    shapes: Counter[str] = Counter()
    per_shape_examples: Dict[str, List[str]] = defaultdict(list)
    for row_id in diverge_decl_neural:
        row = sweep[row_id]
        tag = classify_program(row["desc"])
        if tag is None:
            continue
        exp = safe_int(row["expected"])
        neu = safe_int(row["neural"])
        a, b, _c = parse_desc_args(row["desc"])
        shape = failure_shape(tag, exp if exp is not None else 0, neu, a, b)
        shapes[shape] += 1
        if len(per_shape_examples[shape]) < 3:
            per_shape_examples[shape].append(
                f"id={row_id:04d} {row['desc']} | exp={row['expected']} neural={row['neural']}"
            )

    lines.append("## 3. Failure-shape grouping (decl-correct / neural-wrong rows)\n")
    lines.append("| count | shape | examples |")
    lines.append("|------:|-------|----------|")
    for shape, n in shapes.most_common():
        examples = " <br> ".join(per_shape_examples[shape])
        lines.append(f"| {n} | `{shape}` | {examples} |")
    lines.append("")

    # 4. Per-cluster breakdown to show tag x status.
    # 3b. Coverage gap (missing ids = test ran but no log line / skipped)
    if missing:
        lines.append("### Coverage gaps\n")
        lines.append(
            f"- {len(missing)} wide-byte ALU ids do not appear in any shard "
            "log line (likely pytest skipped or row crashed before printing). "
            "Re-running the sweep with `-x` removed should fill these in.\n"
        )
        miss_per_tag: Counter[str] = Counter()
        for row_id in missing:
            desc_row = programs[row_id][2]
            tag = classify_program(desc_row)
            if tag is not None:
                miss_per_tag[tag] += 1
        for tag, n in miss_per_tag.most_common():
            lines.append(f"  - `{tag}`: {n} ids missing")
        lines.append("")

    lines.append("## 4. Per-sub-cluster status\n")
    lines.append("| cluster | total | swept | correct | div-neural | top failure shape |")
    lines.append("|---------|------:|------:|--------:|----------:|-------------------|")
    for tag, ids in sorted(per_tag.items(), key=lambda kv: -len(kv[1])):
        sw = [i for i in ids if i in swept_ids]
        cor = [i for i in sw if safe_int(sweep[i]["expected"]) == safe_int(sweep[i]["neural"])]
        div = [i for i in sw if safe_int(sweep[i]["expected"]) != safe_int(sweep[i]["neural"])]
        # Most common shape inside this cluster
        local_shapes: Counter[str] = Counter()
        for row_id in div:
            row = sweep[row_id]
            exp = safe_int(row["expected"])
            neu = safe_int(row["neural"])
            a, b, _c = parse_desc_args(row["desc"])
            local_shapes[failure_shape(tag, exp or 0, neu, a, b)] += 1
        top = local_shapes.most_common(1)[0][0] if local_shapes else "(none)"
        lines.append(f"| {tag} | {len(ids)} | {len(sw)} | {len(cor)} | {len(div)} | `{top}` |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir",
                   default=".agent-logs/sweep-2026-06-01",
                   help="Directory containing shard_*.log files.")
    p.add_argument("--out", default="-",
                   help="Output path; '-' for stdout.")
    args = p.parse_args()

    sweep_dir = args.sweep_dir
    if not os.path.isabs(sweep_dir):
        # Try CWD first; fall back to repo-root parent (one above c4_release/).
        if not os.path.isdir(sweep_dir):
            candidate = os.path.normpath(os.path.join(REPO, "..", sweep_dir))
            if os.path.isdir(candidate):
                sweep_dir = candidate
    sweep = parse_sweep_logs(sweep_dir)

    programs = generate_test_programs()
    body = render(programs, sweep)

    if args.out == "-":
        sys.stdout.write(body)
    else:
        out_path = args.out
        if not os.path.isabs(out_path):
            # If parent doesn't exist relative to CWD, try repo-root parent.
            cwd_parent = os.path.dirname(os.path.abspath(out_path))
            if not os.path.isdir(cwd_parent):
                out_path = os.path.normpath(os.path.join(REPO, "..", out_path))
        Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(body)
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
