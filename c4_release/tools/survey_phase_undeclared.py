"""B12 prep: survey every ``phase_required_but_undeclared`` op.

Reads the canonical op factory list (``all_core_ops`` +
``all_alu_postop_attach_ops``) and emits a per-op summary suitable for
authoring the B12 backfill spec. The script is read-only; nothing is
mutated. It is the source of truth for the table in
``c4_release/docs/B12_BACKFILL_SPEC.md`` and is committed alongside that
doc so the doc can be regenerated when the underlying ops shift.

Usage::

    python c4_release/tools/survey_phase_undeclared.py
    python c4_release/tools/survey_phase_undeclared.py --json out.json
    python c4_release/tools/survey_phase_undeclared.py --names-only

For every op flagged ``phase_required_but_undeclared`` by
``analyze_scheduler.py``'s categoriser, the survey reports:

  * factory file:line (resolved via ``inspect.getsourcefile`` /
    ``inspect.getsourcelines`` on the constructor function pulled out of
    ``all_core_ops.py`` and ``alu_ops.py``);
  * current declared ``phase`` / ``layer_idx`` / ``kind``;
  * declared ``reads`` / ``writes`` / ``produces`` / ``consumes_fresh`` /
    ``requires``;
  * the dep-graph in-degree (predecessors) and out-degree (successors);
  * the *successor* ops that anchor it (what currently has to wait on
    it via reads/produces). When in_degree == 0 the op is "totally free
    in DAG" so we report successors only.
  * a heuristic recommendation for the B12 backfill (one of:
    ``op-name-requires-only``, ``dim-read-addition``, ``both``,
    ``needs-upstream-change``, ``manual``).

The recommendation is a HINT, not authoritative — the per-op factory
docstrings and bake helpers still need a human review.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Reuse analyze_scheduler's collection + categorisation so we are
# byte-identical to the canonical report.
import tools.analyze_scheduler as az  # noqa: E402
from neural_vm.unified_compiler.layer_compiler import Operation  # noqa: E402
from neural_vm.unified_compiler.ops import all_core_ops as _aco  # noqa: E402
from neural_vm.unified_compiler.ops import alu_ops as _alu  # noqa: E402


def _factory_index() -> Dict[str, Any]:
    """Map op.name -> factory function by invoking every ``make_*`` in
    the per-layer modules under a probe context manager that records the
    name field of each constructed Operation.

    This is fragile against factories that return more than one op or
    that wrap construction in a helper. For B12 we only need the
    *file:line* of the factory; if a name can't be resolved here we
    fall back to grepping the source tree (see ``_locate_by_grep``).
    """
    idx: Dict[str, Any] = {}
    # Walk all module attributes that look like factories.
    modules: List[Any] = []
    for mod_name in (
        "l0_ops", "l1_ops", "l2_ops", "l3_ops", "l4_ops", "l5_ops",
        "l6_ops", "l7_ops", "l8_ops", "l9_ops", "l10_ops", "l11_ops",
        "l12_ops", "l13_ops", "l14_ops", "l15_ops", "l16_ops",
        "alu_ops", "flag_gated_ops", "model_ops", "user_input_ops",
    ):
        mod = __import__(
            f"neural_vm.unified_compiler.ops.{mod_name}",
            fromlist=["*"],
        )
        modules.append(mod)
    for mod in modules:
        for attr in dir(mod):
            if not attr.startswith("make_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            # Try to invoke with broad-default kwargs.
            try:
                op = _invoke_factory(fn)
            except Exception:
                continue
            if isinstance(op, Operation):
                idx[op.name] = fn
            elif isinstance(op, list):
                for sub in op:
                    if isinstance(sub, Operation):
                        idx[sub.name] = fn
    return idx


def _invoke_factory(fn: Any) -> Any:
    """Best-effort factory invocation with permissive defaults."""
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if p.default is not inspect.Parameter.empty:
            continue
        # Required arg — guess.
        if "enable" in pname or pname.startswith("with_"):
            kwargs[pname] = True
        elif "mode" in pname:
            kwargs[pname] = "lookup"
        else:
            return None
    return fn(**kwargs)


def _locate_by_grep(op_name: str) -> Optional[Tuple[str, int]]:
    """Fallback: grep for ``name=<op_name>"`` or ``name='<op_name>'`` in
    the ops package. Returns the first hit as (file, line)."""
    ops_dir = os.path.join(_REPO, "neural_vm/unified_compiler/ops")
    needle_d = f'name="{op_name}"'
    needle_s = f"name='{op_name}'"
    needle_str_d = f'"{op_name}"'
    needle_str_s = f"'{op_name}'"
    for root, _, files in os.walk(ops_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as fh:
                    for i, line in enumerate(fh, start=1):
                        if needle_d in line or needle_s in line:
                            return (path, i)
            except OSError:
                continue
    # Second pass — maybe the name is a positional arg.
    for root, _, files in os.walk(ops_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as fh:
                    for i, line in enumerate(fh, start=1):
                        if needle_str_d in line or needle_str_s in line:
                            return (path, i)
            except OSError:
                continue
    return None


def _factory_loc(fn: Any) -> Optional[Tuple[str, int]]:
    try:
        src = inspect.getsourcefile(fn)
        _, lineno = inspect.getsourcelines(fn)
    except (TypeError, OSError):
        return None
    if src is None:
        return None
    return (src, lineno)


def _recommend(op: Operation, in_edges, out_edges, depth) -> str:
    """Heuristic recommendation for what B12 should add."""
    in_deg = len(in_edges[op.name])
    out_deg = len(out_edges[op.name])
    if op.name.startswith("l") and op.name.endswith("_alu_postop_attach"):
        return "op-name-requires-only"  # use requires["same_layer_as"]
    if in_deg == 0 and out_deg == 0:
        # Either flag-gated stub or a truly free op. Marking
        # ``freely_placeable`` is a one-line opt-in.
        return "op-name-requires-only"
    if in_deg == 0 and out_deg > 0:
        # Has consumers; the consumer side already creates an edge
        # FROM this op. The "undeclared" gap is that this op runs
        # later than dep_depth permits. Likely a single requires["after"]
        # ties it to its current pin (e.g. layer1_threshold_attn after
        # layer0_threshold_attn).
        return "op-name-requires-only"
    # Has predecessors but still gapped — the missing edge is probably
    # a dim read that would tie it to its current layer (e.g. the L14
    # cleanup ops that read OUTPUT_HI which is in the cycle).
    cur = az._current_layer(op)
    der = depth.get(op.name, -1)
    if cur is not None and der >= 0 and (cur - der) >= 5:
        # Big gap; almost certainly waiting on a dim split in B9.
        return "needs-upstream-change"
    return "dim-read-addition"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=None, help="write JSON summary here")
    p.add_argument("--names-only", action="store_true",
                   help="just print the undeclared op names, one per line")
    args = p.parse_args()

    ops = az.collect_ops()
    in_edges, out_edges, _ = az.build_dep_graph(ops)
    depth, cycle_members = az.topo_depth(ops, in_edges, out_edges)
    cats = az.categorise(ops, depth, cycle_members, in_edges, out_edges)

    target = [op for op in ops
              if cats[op.name] == "phase_required_but_undeclared"]
    target.sort(key=lambda o: (-(az._current_layer(o) or 0),
                               o.phase or 0.0, o.name))

    if args.names_only:
        for op in target:
            print(op.name)
        return 0

    fac_idx = _factory_index()
    out_records: List[Dict[str, Any]] = []
    for op in target:
        rec: Dict[str, Any] = {
            "name": op.name,
            "kind": op.kind,
            "phase": op.phase,
            "layer_idx": op.layer_idx,
            "current_layer": az._current_layer(op),
            "dep_depth": depth.get(op.name, -1),
            "in_degree": len(in_edges[op.name]),
            "out_degree": len(out_edges[op.name]),
            "predecessors": sorted(in_edges[op.name]),
            "successors": sorted(out_edges[op.name]),
            "reads": sorted(op.reads),
            "writes": sorted(op.writes),
            "produces": dict(op.produces),
            "consumes_fresh": dict(op.consumes_fresh),
            "requires": dict(op.requires),
            "recommendation": _recommend(op, in_edges, out_edges, depth),
        }
        loc = None
        fn = fac_idx.get(op.name)
        if fn is not None:
            loc = _factory_loc(fn)
        if loc is None:
            loc = _locate_by_grep(op.name)
        if loc is not None:
            rec["factory_file"] = os.path.relpath(loc[0], _REPO)
            rec["factory_line"] = loc[1]
        out_records.append(rec)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out_records, fh, indent=2, default=list)
        print(f"wrote {args.json}")

    # Always also print a human-readable summary.
    print(f"phase_required_but_undeclared: {len(target)} ops")
    print()
    for rec in out_records:
        loc = ""
        if "factory_file" in rec:
            loc = f"  ({rec['factory_file']}:{rec['factory_line']})"
        print(f"- {rec['name']} phase={rec['phase']} kind={rec['kind']}"
              f" dep_depth={rec['dep_depth']} current={rec['current_layer']}"
              f" in={rec['in_degree']} out={rec['out_degree']}"
              f" rec={rec['recommendation']}{loc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
