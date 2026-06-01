"""Survey ops with empty / missing static claims.

Diagnostic only (read-only). Companion to verify_claims_static (Mode A) in
decl_verifier.py.  Goal: enumerate the ops that declare no ``claims=`` so
they are invisible to the static-claims verifier and bucket them as
``legitimate_empty`` / ``flag_gated`` / ``topology_anchor`` /
``should_backfill`` / ``unclear``.

Output goes to ``.agent-logs/no_claims_survey_2026_06_01.md`` (gitignored).
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "c4_release"))

from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402


PARTIAL_FIELDS = ("reads", "writes", "produces", "consumes_fresh")


def _claims_empty(op) -> bool:
    c = getattr(op, "claims", None)
    if c is None:
        return True
    try:
        return len(c) == 0
    except TypeError:
        return False


def _bake_source(op) -> str:
    fn = getattr(op, "bake_fn", None) or getattr(op, "declarative_bake_fn", None)
    if fn is None:
        return ""
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ""


def _bake_module(op) -> str:
    fn = getattr(op, "bake_fn", None) or getattr(op, "declarative_bake_fn", None)
    if fn is None:
        return ""
    mod = inspect.getmodule(fn)
    if mod is None:
        return ""
    try:
        return inspect.getsourcefile(fn) or mod.__name__
    except TypeError:
        return mod.__name__


def _judge(op, src: str, module_path: str = "") -> str:
    authority = getattr(op, "declarative_authority", None)
    if authority == "topology_anchor":
        return "topology_anchor"
    if authority == "structural_model":
        # Mutates model topology (e.g. _right_size_ffns, _expand_wrapper_blocks),
        # not weights -- legitimate to have no claims.
        return "legitimate_empty"

    body = src.strip()
    if not body:
        return "unclear"

    # Ops sourced from ``flag_gated_ops.py`` follow the pattern
    # ``if <flag>: def bake(): <writes>; else: def bake(): return``.
    # When the harness picks up the closed-over no-op branch, the inner body
    # is a pure ``return`` but the op is semantically flag-gated.
    is_flag_gated_module = "flag_gated_ops.py" in module_path

    # Look at the inner bake function body (after first def line)
    inner = body
    m = re.search(r"def\s+\w+\([^)]*\)[^:]*:\s*\n([\s\S]*)", body)
    if m:
        inner = m.group(1)

    inner_stripped = inner.strip()

    # Bare `return  # disabled when enable_<X>=False` / `# stub` style:
    # hardcoded no-op for the flag-off branch (the factory builds a different
    # closure when the flag is on). Treat as flag_gated. Must run BEFORE the
    # pure no-op check, otherwise these get mis-bucketed as legitimate_empty.
    disabled_comment = re.compile(
        r"return\s*(?:None\s*)?#\s*(?:disabled|no-op|noop|stub|staged|phase\s*\d)",
        re.IGNORECASE,
    )
    if disabled_comment.search(inner):
        return "flag_gated"

    # Pattern: pure no-op (only `return` / `return None` / `pass` and docstring)
    no_op = True
    for line in inner_stripped.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith('"""') or s.startswith("'''") or s.endswith('"""') or s.endswith("'''"):
            continue
        # Strip trailing inline comment before classification
        code = s.split("#", 1)[0].strip()
        if code in ("pass", "return", "return None", ""):
            continue
        # If we hit anything else, it's not pure no-op
        no_op = False
        break
    if no_op:
        # In flag_gated_ops.py the no-op closure is the disabled branch.
        if is_flag_gated_module:
            return "flag_gated"
        return "legitimate_empty"

    # Flag-gated: top-level guard with early return.
    # Heuristic: any early `if not <name>: return` where `<name>` looks
    # like an enable flag (``enable_*`` or bare `enable`), OR a return whose
    # comment explicitly mentions a disable/flag.
    flag_re = re.compile(
        r"^\s*if\s+not\s+(enable\w*|.*flag.*)\s*:\s*\n\s+(?:return\s*(?:None|\b)|pass|raise\s+NotImplementedError)\b",
        re.MULTILINE,
    )
    flag_match = flag_re.search(inner)
    if flag_match:
        return "flag_gated"


    # Look for unconditional write patterns
    writes_pattern = re.compile(
        r"weights\b|set_weight|set_attn_weight|set_ffn_weight|add_weight|"
        r"\.weight|setter|residual_stream|set_ffn|\.W\[|\.W_\w|cancel_token_logits",
    )
    if writes_pattern.search(inner):
        return "should_backfill"

    # Indirect bake: body just dispatches to a helper that does the writing
    # (e.g. `_bake_layer4_ffn(block.ffn, S, proxy)`). Treat as backfill
    # candidate -- the op writes weights, it just doesn't document them.
    helper_call_pattern = re.compile(
        r"\b(_bake_\w+|_set_layer\w+|_lower_\w+_ir|_add_\w+_rules|"
        r"_install_\w+|_apply_\w+|_register_\w+|_emit_\w+)\s*\(",
    )
    if helper_call_pattern.search(inner):
        return "should_backfill"

    # Common shape: bake delegates to block.ffn / block.attn or to lowering helpers
    block_write = re.compile(r"block\.(ffn|attn)\b|S\.\w+\s*=|proxy\.|setdim_proxy")
    if block_write.search(inner):
        return "should_backfill"

    return "unclear"


def _partials(op) -> list[str]:
    out = []
    for field in PARTIAL_FIELDS:
        v = getattr(op, field, None)
        if v is None:
            continue
        try:
            if len(v) == 0:
                continue
        except TypeError:
            pass
        out.append(field)
    if getattr(op, "kind", None):  # scope analogue
        pass
    return out


def main() -> None:
    ops = all_core_ops()
    no_claims = [o for o in ops if _claims_empty(o)]
    print(f"total ops: {len(ops)}")
    print(f"no-claims ops: {len(no_claims)}")

    rows = []
    buckets = {
        "legitimate_empty": [],
        "flag_gated": [],
        "topology_anchor": [],
        "should_backfill": [],
        "unclear": [],
    }
    for op in no_claims:
        src = _bake_source(op)
        mod = _bake_module(op)
        # Strip repo prefix
        try:
            mod_rel = str(Path(mod).relative_to(ROOT))
        except Exception:
            mod_rel = mod
        authority = getattr(op, "declarative_authority", None) or ""
        partials = _partials(op)
        judgment = _judge(op, src, mod_rel)
        buckets[judgment].append((op.name, mod_rel))
        rows.append((op.name, mod_rel, authority, partials, judgment))

    # Markdown output
    out_dir = ROOT / ".agent-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "no_claims_survey_2026_06_01.md"

    lines: list[str] = []
    lines.append("# No-claims op survey (2026-06-01)")
    lines.append("")
    lines.append(
        f"Total ops: **{len(ops)}**.  Ops with empty / missing `claims`: "
        f"**{len(no_claims)}**.",
    )
    lines.append("")
    lines.append("## Bucket summary")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---|")
    for k in ("legitimate_empty", "flag_gated", "topology_anchor", "should_backfill", "unclear"):
        lines.append(f"| `{k}` | {len(buckets[k])} |")
    lines.append("")
    lines.append("## Top `should_backfill` candidates")
    lines.append("")
    if buckets["should_backfill"]:
        lines.append("| # | Op | Module |")
        lines.append("|---|---|---|")
        for i, (name, mod) in enumerate(buckets["should_backfill"][:10], 1):
            lines.append(f"| {i} | `{name}` | `{mod}` |")
    else:
        lines.append("_(none)_")
    lines.append("")
    lines.append("## Full survey")
    lines.append("")
    lines.append("| Op | Module | `declarative_authority` | Partial decls | Judgment |")
    lines.append("|---|---|---|---|---|")
    for name, mod, authority, partials, judgment in rows:
        parts = ",".join(partials) if partials else "-"
        auth = f"`{authority}`" if authority else "-"
        lines.append(f"| `{name}` | `{mod}` | {auth} | {parts} | `{judgment}` |")
    lines.append("")
    lines.append("## Structural observation")
    lines.append("")
    # Group should_backfill by module
    from collections import Counter

    backfill_modules = Counter(mod for _, mod in buckets["should_backfill"])
    if backfill_modules:
        lines.append("`should_backfill` ops by module:")
        lines.append("")
        for mod, count in backfill_modules.most_common():
            lines.append(f"- `{mod}` x {count}")
    else:
        lines.append("_(no `should_backfill` ops to group)_")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")

    # Print buckets to stdout for quick visibility
    for k, v in buckets.items():
        print(f"{k}: {len(v)}")
    print("--- should_backfill top 10 ---")
    for name, mod in buckets["should_backfill"][:10]:
        print(f"  {name}  ({mod})")


if __name__ == "__main__":
    main()
