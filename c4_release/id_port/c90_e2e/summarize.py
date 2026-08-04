"""Summarize the extended C90 conformance matrix into the coverage report.

Reads CONFORMANCE_MATRIX_EXT.json and prints: total cases, native-c4 pass rate,
transformer-run rate, transformer-byte-exact rate, per-category breakdown, and the
per-divergence-class fix backlog.  Read-only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "CONFORMANCE_MATRIX_EXT.json")


def main():
    d = json.load(open(MATRIX))
    res = d["results"]
    n = len(res)
    n_native = sum(1 for r in res if r["native_ok"])
    n_ran = sum(1 for r in res if r["transformer"] is not None)
    n_exact = sum(1 for r in res if r["tf_exact"])

    print(f"=== EXTENDED C90 CONFORMANCE MATRIX ({MATRIX}) ===")
    print(f"cases evaluated       : {n}")
    print(f"native ./c4 == gcc    : {n_native}/{n}")
    print(f"transformer ran       : {n_ran}/{n}")
    print(f"transformer byte-exact: {n_exact}/{n}")
    print()

    # per-category
    cats = {}
    for r in res:
        c = r["category"]
        cats.setdefault(c, [0, 0])
        cats[c][0] += 1
        cats[c][1] += r["tf_exact"]
    print("per-category (transformer byte-exact / total):")
    for c in sorted(cats):
        tot, ex = cats[c][0], cats[c][1]
        print(f"  {c:10s} {ex:3d}/{tot:<3d}")
    print()

    # divergence classes (fix backlog)
    classes = {}
    for r in res:
        if not r["tf_exact"]:
            classes.setdefault(r["class"], []).append(r["name"])
    print("divergence classes (fix backlog):")
    for cls, names in sorted(classes.items(), key=lambda x: -len(x[1])):
        print(f"  {len(names):3d}  {cls}")
        print(f"        {', '.join(names)}")
    print()
    print(f"progress: {d.get('done', n)}/{d.get('n_cases', n)} cases run through the battery")


if __name__ == "__main__":
    main()
