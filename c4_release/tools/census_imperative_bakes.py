"""Phase 6 Wave 1A — imperative bake_fn census.

For every op returned by ``all_core_ops()``, classify the bake_fn body as:

- ``declarative``         -- has ``compiler_ir`` and the bake_fn body just
                             lowers it via ``lower_ffn`` / ``lower_attention``
                             / a ``_lower_*_via_compiler_ir`` helper
                             (no direct ``W_*`` / ``.data[...]`` writes)
- ``imperative_trivial``  -- bake_fn writes <= 10 non-zero weight cells
- ``imperative_medium``   -- bake_fn writes 11..100 non-zero weight cells
- ``imperative_heavy``    -- bake_fn writes > 100 non-zero weight cells
- ``no_op``               -- bake_fn writes nothing (flag-gated off, anchors,
                             diagnostic-only ops, etc.)
- ``unknown``             -- could not execute bake_fn cleanly (model ops with
                             special model-shape requirements, missing optional
                             attributes the stub does not provide, etc.)

Cells are counted by running each bake_fn against a fresh ``StubBlock``
(``c4_release/tests/_per_op_audit.py``) for kind in ("attn","ffn","block")
and against a ``StubModel`` (built here) for kind="model".

Outputs:

- ``c4_release/.agent-logs/imperative_bake_census_phase6.md`` (human-readable)
- ``c4_release/.agent-logs/imperative_bake_census_phase6.json`` (structured)

Run:

    cd c4_release && python -m c4_release.tools.census_imperative_bakes
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Path bootstrap (run as ``python -m c4_release.tools.census_imperative_bakes``
# or directly via ``python c4_release/tools/census_imperative_bakes.py``).
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
# .../c4_release/c4_release/tools/census_imperative_bakes.py
#   parents[0] = tools/
#   parents[1] = c4_release/  (inner)
#   parents[2] = c4_release/  (outer, contains the inner package)
_PROJECT_PARENT = _HERE.parents[2]
if str(_PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_PARENT))


# ---------------------------------------------------------------------------
# Stub model for kind="model" ops
# ---------------------------------------------------------------------------

class _StubEmbed:
    def __init__(self, d_model: int, vocab_size: int = 512):
        # Mimics NeuralVMEmbedding.embed (an nn.Embedding-like with .weight)
        class _E:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.zeros(vocab_size, d_model))
        self.embed = _E()


class _StubHead:
    def __init__(self, d_model: int, vocab_size: int = 512):
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, d_model))
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))


def _build_stub_model(d_model: int, ffn_hidden: int, n_blocks: int = 20,
                      num_heads: int = 16):
    from tests._per_op_audit import StubBlock

    class _M:
        pass

    m = _M()
    m.blocks = [
        StubBlock(d_model=d_model, num_heads=num_heads, ffn_hidden=ffn_hidden)
        for _ in range(n_blocks)
    ]
    m.embed = _StubEmbed(d_model=d_model)
    m.head = _StubHead(d_model=d_model)
    # Some bakes look at model.config / model.num_blocks etc.; expose a small
    # surface so attribute access doesn't AttributeError.
    m.config = {}
    m.num_blocks = n_blocks
    m.d_model = d_model
    return m


# ---------------------------------------------------------------------------
# Cell counting
# ---------------------------------------------------------------------------

def _count_block_cells(block) -> Dict[str, int]:
    """Count non-zero cells across attn + ffn parameter tensors of a block."""
    counts: Dict[str, int] = {}
    if hasattr(block, "attn"):
        attn = block.attn
        for name in ("W_q", "W_k", "W_v", "W_o"):
            t = getattr(attn, name, None)
            if t is not None:
                counts[f"attn.{name}"] = int((t != 0).sum().item())
        slopes = getattr(attn, "alibi_slopes", None)
        if slopes is not None:
            counts["attn.alibi_slopes"] = int((slopes != 0).sum().item())
    if hasattr(block, "ffn"):
        ffn = block.ffn
        for name in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
            t = getattr(ffn, name, None)
            if t is not None:
                counts[f"ffn.{name}"] = int((t != 0).sum().item())
    return counts


def _count_model_cells(model) -> Dict[str, int]:
    """Count non-zero cells across all blocks and head/embed of a stub model."""
    counts: Dict[str, int] = {}
    for i, blk in enumerate(model.blocks):
        for k, v in _count_block_cells(blk).items():
            if v:
                counts[f"block{i}.{k}"] = v
    # Embed
    try:
        ew = model.embed.embed.weight
        nz = int((ew != 0).sum().item())
        if nz:
            counts["embed.weight"] = nz
    except Exception:
        pass
    # Head
    try:
        for name in ("weight", "bias"):
            t = getattr(model.head, name, None)
            if t is not None:
                nz = int((t != 0).sum().item())
                if nz:
                    counts[f"head.{name}"] = nz
    except Exception:
        pass
    return counts


# ---------------------------------------------------------------------------
# Bake_fn source-inspection helpers
# ---------------------------------------------------------------------------

# Patterns that look like the op is just lowering its compiler_ir to weights.
_DECLARATIVE_LOWER_RE = re.compile(
    r"\b("
    r"lower_ffn"
    r"|lower_attention"
    r"|_?lower_[a-zA-Z0-9_]*_via_(compiler_)?ir"
    r"|_?lower_[a-zA-Z0-9_]*_ir"
    r"|Primitives\.lower_ffn_rules"
    r"|Primitives\.generate_attention_heads?"
    r"|Primitives\.generate_threshold_attention_heads"
    r")\b"
)

# Patterns that look imperative.
_IMPERATIVE_RE = re.compile(
    r"\b("
    r"W_up\.data\["
    r"|W_down\.data\["
    r"|W_gate\.data\["
    r"|b_up\.data\["
    r"|b_gate\.data\["
    r"|W_q\.data\["
    r"|W_k\.data\["
    r"|W_v\.data\["
    r"|W_o\.data\["
    r"|alibi_slopes\.data\["
    r")"
)

# Helper-call patterns to map an op to its underlying setup_helpers /
# vm_step helper. Captures `_set_*`, `setup_*`, generate_*` calls.
_HELPER_CALL_RE = re.compile(
    r"\b("
    r"_set_[a-zA-Z0-9_]+"
    r"|_lower_[a-zA-Z0-9_]+"
    r"|setup_[a-zA-Z0-9_]+"
    r"|Primitives\.[a-zA-Z0-9_]+"
    r"|_suppress_[a-zA-Z0-9_]+"
    r"|_right_size_ffns"
    r"|_expand_wrapper_blocks"
    r"|_set_layer[0-9]+_[a-zA-Z0-9_]+"
    r"|setup_head_weights"
    r"|setup_token_embeddings"
    r")\b"
)


def _bake_source(bake_fn) -> str:
    try:
        return inspect.getsource(bake_fn)
    except (OSError, TypeError):
        return ""


def _looks_declarative(source: str, has_compiler_ir: bool) -> bool:
    """True if bake_fn body suggests it just lowers compiler_ir.

    Heuristic: it (a) calls lower_ffn / lower_attention / a
    _lower_*_via_compiler_ir helper, and (b) has NO direct W_*.data[...] /
    b_*.data[...] writes anywhere in the source.
    """
    if not has_compiler_ir:
        return False
    if not _DECLARATIVE_LOWER_RE.search(source):
        return False
    if _IMPERATIVE_RE.search(source):
        return False
    return True


def _extract_helpers(source: str) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    for m in _HELPER_CALL_RE.finditer(source):
        name = m.group(1)
        # Filter trivial / internal names that aren't really weight-setters.
        if name.startswith("Primitives.") and name in (
            "Primitives.dim_positions_from_bd",
            "Primitives.ffn_rule_dim_names",
        ):
            continue
        if name in ("_set_", "_lower_", "setup_"):
            continue
        if name in seen_set:
            continue
        seen_set.add(name)
        seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@dataclass
class OpCensusRow:
    name: str
    kind: str
    phase: Optional[float]
    layer_idx: Optional[int]
    has_compiler_ir: bool
    has_compiler_ir_factory: bool
    declarative_authority: Optional[str]
    migrated: bool
    bake_module: str
    bake_line: int
    classification: str         # declarative / imperative_* / no_op / unknown
    cells_written: int
    cell_breakdown: Dict[str, int] = field(default_factory=dict)
    helpers: List[str] = field(default_factory=list)
    enabled_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    # Best-effort target layer for grouping (uses layer_idx, or parses name).
    layer_group: str = "?"


_LAYER_NAME_RE = re.compile(r"l(?:ayer)?[_]?(\d+)|(?:^|_)l(\d+)(?:_|$)")


def _infer_layer_group(name: str, layer_idx: Optional[int]) -> str:
    if layer_idx is not None:
        return f"L{layer_idx}"
    m = _LAYER_NAME_RE.search(name.lower())
    if m:
        digits = m.group(1) or m.group(2)
        return f"L{int(digits)}"
    # Specific known names
    if name in ("phase_a_ffn",):
        return "L0"
    if name in (
        "head_bake",
        "embedding_bake",
        "initial_pc_bake",
        "right_size_ffns",
        "expand_wrapper_blocks",
        "branch_override_patch",
        "contract_validation",
    ):
        return "model"
    return "?"


def _classify_cells(cells: int) -> str:
    if cells == 0:
        return "no_op"
    if cells <= 10:
        return "imperative_trivial"
    if cells <= 100:
        return "imperative_medium"
    return "imperative_heavy"


_NUM_HEADS = 16  # wide enough to cover L15 head_idx=13


def _augment_stub(block) -> None:
    """Patch in nn.Module-like helpers some bakes call.

    L15's ``l15_attention_resize`` and a couple of model-ops call
    ``attn.register_buffer`` / ``ffn.named_children`` etc. Adding minimal
    shims here lets those bakes run against the stub.
    """
    attn = getattr(block, "attn", None)
    ffn = getattr(block, "ffn", None)
    for mod in (attn, ffn):
        if mod is None:
            continue
        if not hasattr(mod, "register_buffer"):
            def _register_buffer(self, name, tensor, persistent=True):
                setattr(self, name, tensor)
            mod.register_buffer = _register_buffer.__get__(mod, type(mod))
        if not hasattr(mod, "named_children"):
            mod.named_children = lambda: iter(())
        if not hasattr(mod, "named_parameters"):
            mod.named_parameters = lambda: iter(())


def _exercise_op(op, dim_positions: Dict[str, int], d_model: int,
                 ffn_hidden: int) -> Tuple[int, Dict[str, int], Optional[str]]:
    """Run op.bake_fn against a fresh stub and return (cells, breakdown, error)."""
    from tests._per_op_audit import StubBlock

    kind = getattr(op, "kind", "block")
    try:
        if kind == "model":
            model = _build_stub_model(
                d_model=d_model, ffn_hidden=ffn_hidden, n_blocks=20,
                num_heads=_NUM_HEADS,
            )
            for blk in model.blocks:
                _augment_stub(blk)
            op.bake_fn(model, dict(dim_positions), 100.0)
            counts = _count_model_cells(model)
        elif kind == "attn":
            block = StubBlock(d_model=d_model, num_heads=_NUM_HEADS,
                              ffn_hidden=ffn_hidden)
            _augment_stub(block)
            op.bake_fn(block.attn, dict(dim_positions), 100.0)
            counts = _count_block_cells(block)
        elif kind == "ffn":
            block = StubBlock(d_model=d_model, num_heads=_NUM_HEADS,
                              ffn_hidden=ffn_hidden)
            _augment_stub(block)
            op.bake_fn(block.ffn, dict(dim_positions), 100.0)
            counts = _count_block_cells(block)
        else:  # block
            block = StubBlock(d_model=d_model, num_heads=_NUM_HEADS,
                              ffn_hidden=ffn_hidden)
            _augment_stub(block)
            op.bake_fn(block, dict(dim_positions), 100.0)
            counts = _count_block_cells(block)
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return 0, {}, err
    total = sum(counts.values())
    # Strip zero entries for compactness in output.
    counts = {k: v for k, v in counts.items() if v}
    return total, counts, None


def _classify_op(op, dim_positions: Dict[str, int], d_model: int,
                 ffn_hidden: int, enabled_flags: List[str]) -> OpCensusRow:
    source = _bake_source(op.bake_fn)
    has_ir = op.compiler_ir is not None
    has_ir_factory = op.compiler_ir_factory is not None
    try:
        bake_module = op.bake_fn.__code__.co_filename
        bake_line = op.bake_fn.__code__.co_firstlineno
    except AttributeError:
        bake_module = "?"
        bake_line = 0
    declarative = _looks_declarative(source, has_ir or has_ir_factory)
    helpers = _extract_helpers(source)

    cells, breakdown, err = _exercise_op(op, dim_positions, d_model, ffn_hidden)

    if declarative and cells > 0:
        classification = "declarative"
    elif declarative and cells == 0:
        # Declarative shape but produced no cells -- could be flag-off no-op
        classification = "declarative_no_op"
    elif err is not None:
        classification = "unknown"
    elif cells == 0:
        classification = "no_op"
    else:
        classification = _classify_cells(cells)

    return OpCensusRow(
        name=op.name,
        kind=op.kind,
        phase=op.phase,
        layer_idx=op.layer_idx,
        has_compiler_ir=has_ir,
        has_compiler_ir_factory=has_ir_factory,
        declarative_authority=getattr(op, "declarative_authority", None),
        migrated=bool(getattr(op, "migrated", False)),
        bake_module=bake_module,
        bake_line=bake_line,
        classification=classification,
        cells_written=cells,
        cell_breakdown=breakdown,
        helpers=helpers,
        enabled_flags=list(enabled_flags),
        error=err,
        layer_group=_infer_layer_group(op.name, op.layer_idx),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_census() -> List[OpCensusRow]:
    from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops
    from tests._per_op_audit import compile_compact_layout, _DEFAULT_FFN_HIDDEN

    layout = compile_compact_layout()
    dim_positions = layout.dim_positions
    # Use a wider model than `_per_op_audit`'s 512/8 default so L15 attention
    # ops (which target head indices up to 13) and L15 attention-resize have
    # enough head/column room.
    d_model = max(layout.d_model, 1024)
    ffn_hidden = _DEFAULT_FFN_HIDDEN

    # Pull ops twice: default flags AND with all conversational-I/O / tool /
    # neural-think flags enabled. We merge by op-name, preferring the
    # configuration with more cells written -- that exposes the "real" bake
    # cost for flag-gated ops (which would otherwise be permanent no_ops).
    flag_modes = [
        ("default", {}),
        ("all_flags_on", {
            "enable_conversational_io": True,
            "enable_tool_calling": True,
            "enable_neural_io_think_protocol": True,
        }),
    ]

    rows_by_name: Dict[str, OpCensusRow] = {}
    seen_in_default: Set[str] = set()
    for label, kwargs in flag_modes:
        ops = all_core_ops(**kwargs)
        if label == "default":
            seen_in_default = {op.name for op in ops}
        for op in ops:
            enabled = [k for k, v in kwargs.items() if v]
            row = _classify_op(op, dim_positions, d_model, ffn_hidden, enabled)
            prior = rows_by_name.get(op.name)
            # Always prefer the row with more cells (real bake cost) but keep
            # the default-mode info in enabled_flags so we can see which ops
            # only fire under flags.
            if prior is None or row.cells_written > prior.cells_written:
                rows_by_name[op.name] = row

    # Tag ops that don't appear under default flags (composite-shift ops, etc.)
    for name, row in rows_by_name.items():
        if name not in seen_in_default:
            row.enabled_flags = sorted(
                set(row.enabled_flags) | {"requires_flag"}
            )

    # Stable ordering by phase then name.
    def _key(r: OpCensusRow):
        return (r.phase if r.phase is not None else 1e9, r.name)

    return sorted(rows_by_name.values(), key=_key)


def aggregate(rows: List[OpCensusRow]) -> Dict[str, Any]:
    total_per_class = Counter(r.classification for r in rows)
    per_kind = Counter(r.kind for r in rows)
    per_layer_per_class: Dict[str, Counter] = {}
    for r in rows:
        per_layer_per_class.setdefault(r.layer_group, Counter())[r.classification] += 1

    # Biggest 10 helpers by cells.
    top10 = sorted(
        (r for r in rows if r.classification.startswith("imperative") or
         r.classification == "declarative"),
        key=lambda r: r.cells_written,
        reverse=True,
    )[:10]

    # Cells totals per class.
    cells_per_class = Counter()
    for r in rows:
        cells_per_class[r.classification] += r.cells_written

    # Op count by has_compiler_ir.
    ir_status = Counter(
        ("has_ir" if r.has_compiler_ir or r.has_compiler_ir_factory
         else "no_ir") for r in rows
    )

    return {
        "total_ops": len(rows),
        "per_class": dict(total_per_class),
        "per_kind": dict(per_kind),
        "per_layer_per_class": {
            k: dict(v) for k, v in sorted(per_layer_per_class.items())
        },
        "cells_per_class": dict(cells_per_class),
        "ir_status": dict(ir_status),
        "top10_biggest": [
            {
                "name": r.name,
                "cells": r.cells_written,
                "classification": r.classification,
                "helpers": r.helpers,
                "layer": r.layer_group,
            }
            for r in top10
        ],
    }


_MD_HEADER = (
    "# Phase 6 Wave 1A — imperative bake_fn census\n"
    "\n"
    "Generated by `c4_release/tools/census_imperative_bakes.py`.\n"
    "\n"
    "Each op is classified by running its `bake_fn` against a fresh\n"
    "`StubBlock` (or `StubModel` for `kind=\"model\"`) and counting non-zero\n"
    "cells written into the standard parameter tensors\n"
    "(`attn.W_{q,k,v,o}`, `attn.alibi_slopes`, `ffn.W_{up,gate,down}`,\n"
    "`ffn.b_{up,gate,down}`, plus `model.embed.embed.weight` /\n"
    "`model.head.{weight,bias}` for model-kind ops).\n"
    "\n"
    "`declarative` = `compiler_ir` (or `compiler_ir_factory`) is populated\n"
    "AND the `bake_fn` source contains a `lower_ffn` / `lower_attention` /\n"
    "`_lower_*_via_compiler_ir` call AND no direct `W_*.data[...]` writes.\n"
    "\n"
    "Cell buckets: trivial ≤10, medium 11–100, heavy >100.\n"
    "\n"
)


def render_markdown(rows: List[OpCensusRow], agg: Dict[str, Any]) -> str:
    lines: List[str] = [_MD_HEADER]

    lines.append("## Aggregates\n")
    lines.append(f"- Total ops in `all_core_ops()`: **{agg['total_ops']}**")
    lines.append("- Per classification:")
    for k, v in sorted(agg['per_class'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: **{v}**")
    lines.append("- Per kind:")
    for k, v in sorted(agg['per_kind'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: **{v}**")
    lines.append("- IR status:")
    for k, v in sorted(agg['ir_status'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: **{v}**")
    lines.append("- Cells written per class (sum of nonzero param cells):")
    for k, v in sorted(agg['cells_per_class'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: **{v}**")
    lines.append("")

    lines.append("## Per-layer breakdown\n")
    cols = ["layer", "declarative", "declarative_no_op", "imperative_trivial",
            "imperative_medium", "imperative_heavy", "no_op", "unknown"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for layer, counts in agg['per_layer_per_class'].items():
        row = [layer] + [str(counts.get(c, 0)) for c in cols[1:]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Top 10 heaviest bakes (by cell count)\n")
    lines.append("| Op | Layer | Cells | Class | Helpers (first 3) |")
    lines.append("|---|---|---:|---|---|")
    for row in agg['top10_biggest']:
        helpers = ", ".join(f"`{h}`" for h in row['helpers'][:3]) or "-"
        lines.append(
            f"| `{row['name']}` | {row['layer']} | {row['cells']:,} "
            f"| {row['classification']} | {helpers} |"
        )
    lines.append("")

    lines.append("## All ops (sorted by phase)\n")
    lines.append(
        "| Op | Layer | Kind | Phase | IR | Class | Cells | Helpers |"
    )
    lines.append("|---|---|---|---:|:-:|---|---:|---|")
    for r in rows:
        ir_flag = "Y" if (r.has_compiler_ir or r.has_compiler_ir_factory) else "n"
        helpers = ", ".join(f"`{h}`" for h in r.helpers[:3]) or "-"
        phase = f"{r.phase:g}" if r.phase is not None else "-"
        flags = ""
        if "requires_flag" in r.enabled_flags:
            flags = " (flag-on)"
        lines.append(
            f"| `{r.name}`{flags} | {r.layer_group} | {r.kind} | {phase} "
            f"| {ir_flag} | {r.classification} | {r.cells_written:,} "
            f"| {helpers} |"
        )
    lines.append("")

    # Verdict on the plan's 99 imperative / 89 attention estimate.
    imp_total = sum(
        v for k, v in agg['per_class'].items() if k.startswith("imperative")
    )
    decl = agg['per_class'].get("declarative", 0)
    decl_no_op = agg['per_class'].get("declarative_no_op", 0)
    no_op = agg['per_class'].get("no_op", 0)
    unknown = agg['per_class'].get("unknown", 0)
    attn_ops = sum(1 for r in rows if r.kind == "attn")
    block_attn_ops = sum(
        1 for r in rows
        if r.kind == "block" and any(b.startswith("attn.") for b in r.cell_breakdown)
    )
    model_attn_ops = sum(
        1 for r in rows
        if r.kind == "model" and any(
            "attn" in b for b in r.cell_breakdown
        )
    )
    attn_writing_ops_total = attn_ops + block_attn_ops + model_attn_ops

    lines.append("## Verdict on plan estimate\n")
    lines.append(
        f"- Plan doc estimates **99 imperative ops** + **89 legacy attention bakes**."
    )
    lines.append(
        f"- Observed imperative-classed ops (trivial+medium+heavy): "
        f"**{imp_total}**"
    )
    lines.append(
        f"- Observed declarative ops (lower-via-IR): **{decl}** "
        f"(+ {decl_no_op} declarative-shape but flag-off no_op)"
    )
    lines.append(
        f"- Pure no_op ops: **{no_op}**; unknown (stub-incompatible): "
        f"**{unknown}**"
    )
    lines.append(
        f"- Ops whose bake actually writes to attention parameters: "
        f"**{attn_writing_ops_total}** "
        f"(kind=attn: {attn_ops}; kind=block touching attn: {block_attn_ops}; "
        f"kind=model touching attn: {model_attn_ops})"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = _PROJECT_PARENT / "c4_release" / ".agent-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_census()
    agg = aggregate(rows)

    md_path = out_dir / "imperative_bake_census_phase6.md"
    json_path = out_dir / "imperative_bake_census_phase6.json"
    md_path.write_text(render_markdown(rows, agg))
    json_path.write_text(json.dumps(
        {"aggregate": agg, "rows": [asdict(r) for r in rows]},
        indent=2,
    ))

    # Brief console summary so the agent can confirm at a glance.
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"total ops: {agg['total_ops']}")
    print("per_class:", agg['per_class'])
    print("per_kind:", agg['per_kind'])


if __name__ == "__main__":
    main()
