"""Phase 6 Wave 1A v2 — imperative bake_fn census with helper-source walking.

Like v1 (``census_imperative_bakes.py``) but with two correctness fixes:

1. **Recursive helper walking.** v1's classifier only inspected the bake_fn's
   own source. v1 misclassified ops like ``phase_a_ffn`` whose bake_fn calls
   a module-local helper ``_bake_phase_a_ffn`` that itself calls
   ``Primitives.lower_ffn_rules``. v2 resolves helper names referenced in
   the bake_fn body against the bake_fn's module globals (also walking
   class-method ``self._foo`` callees via the bound module) and collects
   their sources transitively (depth-bounded) before pattern-matching.

2. **New classification ``declarative_via_helper``.** Distinct from
   ``declarative`` (lower call appears directly in bake_fn source) and
   ``imperative_*`` (bake_fn writes ``W_*.data`` directly). The v2 row also
   carries a list of helper names that contained the lowering call, so the
   reader can audit the chain.

Additionally v2 flags ops where ``compiler_ir=`` is attached but neither the
bake_fn nor its (transitive) helpers contain a lowering call — these are
"informational-only" IR declarations whose bake still writes weights
imperatively.

Outputs:

- ``c4_release/.agent-logs/imperative_bake_census_phase6_v2.md`` (human)
- ``c4_release/.agent-logs/imperative_bake_census_phase6_v2.json`` (structured)

Run:

    cd c4_release && python -m c4_release.tools.census_imperative_bakes_v2
"""
from __future__ import annotations

import inspect
import json
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_PARENT = _HERE.parents[2]
if str(_PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_PARENT))


# ---------------------------------------------------------------------------
# Stub model (unchanged from v1)
# ---------------------------------------------------------------------------

class _StubEmbed:
    def __init__(self, d_model: int, vocab_size: int = 512):
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
    m.config = {}
    m.num_blocks = n_blocks
    m.d_model = d_model
    return m


# ---------------------------------------------------------------------------
# Cell counting (unchanged from v1)
# ---------------------------------------------------------------------------

def _count_block_cells(block) -> Dict[str, int]:
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
    counts: Dict[str, int] = {}
    for i, blk in enumerate(model.blocks):
        for k, v in _count_block_cells(blk).items():
            if v:
                counts[f"block{i}.{k}"] = v
    try:
        ew = model.embed.embed.weight
        nz = int((ew != 0).sum().item())
        if nz:
            counts["embed.weight"] = nz
    except Exception:
        pass
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
# Source inspection — pattern set (same as v1)
# ---------------------------------------------------------------------------

_DECLARATIVE_LOWER_RE = re.compile(
    r"\b("
    r"lower_ffn"
    r"|lower_attention"
    r"|lower_token_embeddings"
    r"|_?lower_[a-zA-Z0-9_]*_via_(compiler_)?ir"
    r"|_?lower_[a-zA-Z0-9_]*_ir"
    r"|Primitives\.lower_ffn_rules"
    r"|Primitives\.generate_attention_heads?"
    r"|Primitives\.generate_threshold_attention_heads"
    r")\b"
)

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

# "Benign" residual write patterns. These don't disqualify an op as
# declarative-via-helper:
#   - ``alibi_slopes.data[<scalar idx>] = <number>``: one-shot per-head slope
#     bookkeeping written alongside an otherwise declarative head spec.
#     The :class:`AttentionHeadIR` doesn't carry slope values, so a 1-line
#     scalar assignment is the canonical residual; see L10
#     ``_bake_layer10_*_passthrough_head`` for the pattern.
#   - ``<param>.data[...].zero_()``: zero-fill range cleanups that
#     ``_clear_ffn_unit_band`` and similar helpers issue before the IR
#     lowering writes into the same range. Pure zeroing has no semantic
#     content and is not a target for IR migration on its own.
_BENIGN_RESIDUAL_RES = (
    # ``alibi_slopes.data[<idx>] = <scalar number>`` (require a non-word
    # char after the number so we don't strip ``= 0`` from ``= 0.5``).
    re.compile(
        r"alibi_slopes\.data\[\d+\] = -?\d+(?:\.\d+)?(?![\.\w])"
    ),
    # ``<param>.data[...].zero_()`` — single contiguous slice zero-fill.
    re.compile(
        r"\b(?:W_up|W_down|W_gate|W_q|W_k|W_v|W_o|b_up|b_gate|b_down)"
        r"\.data\[[^\]]*\]\.zero_\(\)"
    ),
    # ``<param>.data[<slice>] = 0`` / ``= 0.0`` — slice zero-fill via
    # assignment. The slice must contain ``:`` (a true slice, not an
    # individual cell), so this only matches band-style clears such as
    # ``ffn.W_up.data[start:end, :] = 0`` from ``_clear_ffn_unit_band``.
    # The trailing negative lookahead avoids stripping ``= 0`` from
    # ``= 0.5`` / ``= 0e3``.
    re.compile(
        r"\b(?:W_up|W_down|W_gate|W_q|W_k|W_v|W_o|b_up|b_gate|b_down)"
        r"\.data\[[^\]]*:[^\]]*\] = 0(?:\.0)?(?![\.\w])"
    ),
)


def _strip_benign_residuals(source: str) -> str:
    """Remove benign residual write patterns from source for imperative scan."""
    if not source:
        return source
    out = source
    for pat in _BENIGN_RESIDUAL_RES:
        out = pat.sub("", out)
    return out

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

# All callable identifiers, not just the helper-naming convention above.
# Catches things like `_bake_phase_a_ffn`, `_apply_*`, `_attach_*`, etc.
# We deliberately match a broad word followed by `(` to find function calls.
_ANY_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Identifiers we never want to chase (builtins, common stdlib, tensor methods)
_CALL_BLACKLIST: Set[str] = {
    # builtins
    "len", "range", "list", "dict", "set", "tuple", "int", "float", "str",
    "bool", "bytes", "abs", "min", "max", "sum", "any", "all", "enumerate",
    "zip", "map", "filter", "sorted", "reversed", "iter", "next", "isinstance",
    "issubclass", "hasattr", "getattr", "setattr", "delattr", "print",
    "repr", "id", "type", "open", "Exception", "ValueError", "TypeError",
    "RuntimeError", "AssertionError", "AttributeError", "KeyError",
    "IndexError", "NotImplementedError", "StopIteration", "ZeroDivisionError",
    "OSError", "round", "divmod", "pow", "hash", "vars", "dir", "callable",
    "frozenset", "format", "ord", "chr", "hex", "oct", "bin", "complex",
    # torch / numpy common
    "tensor", "zeros", "ones", "arange", "stack", "cat", "where",
    "Tensor", "Parameter", "nn", "torch", "np", "numpy", "Module",
    # dataclasses / typing
    "dataclass", "field", "Counter", "deque", "defaultdict",
    # operation-shape locals
    "Operation", "CompilerIR", "set", "tuple",
}


def _get_source(fn) -> str:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return ""


def _looks_declarative_in_source(source: str) -> bool:
    """Source contains a lowering call AND no direct W_*.data writes.

    Benign residual patterns (per-head ``alibi_slopes`` scalars, zero-fill
    range cleanups) are stripped before scanning so a mostly-declarative
    helper isn't penalized for a 1-line slope setter or a band zero-fill
    that precedes the IR lowering.
    """
    if not source:
        return False
    if not _DECLARATIVE_LOWER_RE.search(source):
        return False
    if _IMPERATIVE_RE.search(_strip_benign_residuals(source)):
        return False
    return True


def _has_lower_call(source: str) -> bool:
    return bool(source and _DECLARATIVE_LOWER_RE.search(source))


def _has_imperative_writes(source: str) -> bool:
    """Detect direct W_*.data writes that are NOT benign residuals."""
    if not source:
        return False
    return bool(_IMPERATIVE_RE.search(_strip_benign_residuals(source)))


# Threshold for the ``declarative_with_residual`` classification: an op
# whose bake or helper-chain contains a declarative lower call AND no
# more than this many distinct ``W_*.data[`` / ``b_*.data[`` /
# ``alibi_slopes.data[`` write statements (after stripping benign
# residuals) is classified as "declarative with a small imperative
# residual" rather than imperative_heavy. The residual is intentionally
# kept by the op (e.g., cross-head boundary spillover for
# ``layer6_attn_bake``) but the bulk of cells is IR-driven.
_DECLARATIVE_WITH_RESIDUAL_MAX_IMP_WRITES = 10


def _count_imperative_writes(source: str) -> int:
    """Count direct ``W_*.data[`` / ``b_*.data[`` / ``alibi_slopes.data[``
    write statements in ``source`` after benign residuals are stripped.
    """
    if not source:
        return 0
    return len(_IMPERATIVE_RE.findall(_strip_benign_residuals(source)))


# ---------------------------------------------------------------------------
# Recursive helper walking
# ---------------------------------------------------------------------------


def _resolve_call_in_module(name: str, module: types.ModuleType) -> Optional[Any]:
    """Resolve a bare identifier (e.g. ``_bake_phase_a_ffn``) at module scope.

    Returns the callable, or None if not present in module globals.
    """
    if module is None:
        return None
    obj = getattr(module, name, None)
    if obj is None:
        return None
    if callable(obj):
        return obj
    return None


def _module_of(fn) -> Optional[types.ModuleType]:
    if fn is None:
        return None
    mod_name = getattr(fn, "__module__", None)
    if mod_name is None:
        return None
    return sys.modules.get(mod_name)


@dataclass
class _HelperWalkResult:
    visited: List[Tuple[str, str]] = field(default_factory=list)
    declarative_helpers: List[Tuple[str, str]] = field(default_factory=list)
    imperative_helpers: List[Tuple[str, str]] = field(default_factory=list)
    # Any helper had a lower call (even if accompanied by direct writes).
    any_lower_call: bool = False
    # Any helper had direct W_*.data writes.
    any_imperative_writes: bool = False
    # Sum of imperative-write statements (post benign-residual strip) across
    # the bake_fn's source + every reached helper's source. Used by the
    # ``declarative_with_residual`` classification to distinguish
    # "mostly-declarative with a tiny boundary residual" from a genuinely
    # imperative helper chain.
    total_imperative_writes: int = 0


def _walk_helpers(
    bake_fn,
    max_depth: int = 3,
) -> _HelperWalkResult:
    """Walk bake_fn's source, then each helper's source, up to ``max_depth``.

    Returns a ``_HelperWalkResult`` summarizing:
      - which helpers were reached (visited),
      - which contain a lowering call with NO imperative writes (purely declarative),
      - which contain direct ``W_*.data`` writes (imperative),
      - whether ANY reached helper contained a lower call,
      - whether ANY reached helper contained direct writes.

    Cross-module helpers (e.g. ``from ...vm_step import _set_*``) are reached
    by following the ``from ... import name`` form in the source: we look the
    name up in the caller's module globals after import.
    """
    result = _HelperWalkResult()

    seen: Set[int] = set()
    queue: List[Tuple[Any, int]] = [(bake_fn, 0)]

    while queue:
        fn, depth = queue.pop(0)
        if fn is None:
            continue
        try:
            fid = id(fn)
        except Exception:
            continue
        if fid in seen:
            continue
        seen.add(fid)

        src = _get_source(fn)
        if not src:
            continue

        qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))
        # Always count imperative-write statements (including the bake_fn's
        # own source at depth 0). The bake_fn body is the most common host
        # for residual writes alongside an inline declarative lower call,
        # so we must include it in the running total used by
        # ``declarative_with_residual``.
        result.total_imperative_writes += _count_imperative_writes(src)
        if depth > 0:  # don't list the bake_fn itself
            result.visited.append(
                (getattr(fn, "__name__", "?"), qualname)
            )
            if _has_lower_call(src):
                result.any_lower_call = True
                if not _has_imperative_writes(src):
                    result.declarative_helpers.append(
                        (getattr(fn, "__name__", "?"), qualname)
                    )
            if _has_imperative_writes(src):
                result.any_imperative_writes = True
                result.imperative_helpers.append(
                    (getattr(fn, "__name__", "?"), qualname)
                )

        if depth >= max_depth:
            continue

        mod = _module_of(fn)
        if mod is None:
            continue

        # Find bare-name calls in fn source and try to resolve in fn's module.
        called_names: Set[str] = set()
        for m in _ANY_CALL_RE.finditer(src):
            name = m.group(1)
            if name in _CALL_BLACKLIST:
                continue
            if name.startswith("__"):
                continue
            if name in {"if", "for", "while", "return", "yield", "lambda",
                        "with", "def", "class", "raise", "assert", "elif",
                        "else", "try", "except", "finally", "from", "import",
                        "as", "in", "is", "not", "and", "or", "True", "False",
                        "None", "pass", "break", "continue", "global",
                        "nonlocal", "del"}:
                continue
            called_names.add(name)

        # Pre-execute any ``from X import Y`` inside the source so the
        # imported names land in the module globals (some bakes do their
        # imports lazily inside the function body — e.g. L6 routing pulls
        # ``_set_layer6_routing_ffn`` from ``vm_step`` inside the helper).
        # We do this by scanning for ``from X import Y[, Z, ...]`` lines and
        # running ``importlib`` on X then attaching attributes to ``mod``.
        try:
            import importlib
            for im in re.finditer(
                r"^\s*from\s+([\w\.]+)\s+import\s+([\w,\s]+)", src, re.M
            ):
                mod_path = im.group(1)
                imported = [s.strip() for s in im.group(2).split(",")]
                # Resolve relative imports against fn's package.
                if mod_path.startswith("."):
                    fn_pkg = getattr(mod, "__package__", None) or ""
                    leading = len(mod_path) - len(mod_path.lstrip("."))
                    parts = fn_pkg.split(".")
                    if leading <= len(parts):
                        base = ".".join(parts[: len(parts) - (leading - 1)])
                    else:
                        base = ""
                    tail = mod_path.lstrip(".")
                    full = f"{base}.{tail}" if tail else base
                else:
                    full = mod_path
                try:
                    src_mod = importlib.import_module(full)
                except Exception:
                    continue
                for name in imported:
                    if not name or name in called_names:
                        # Already in our set; nothing to attach.
                        pass
                    obj = getattr(src_mod, name, None)
                    if obj is not None and not hasattr(mod, name):
                        try:
                            setattr(mod, name, obj)
                        except Exception:
                            pass
        except Exception:
            pass

        for name in called_names:
            resolved = _resolve_call_in_module(name, mod)
            if resolved is None:
                continue
            queue.append((resolved, depth + 1))

    return result


# ---------------------------------------------------------------------------
# Bake_fn helper-name extraction (for display; mirrors v1 helpers column)
# ---------------------------------------------------------------------------

def _extract_helpers(source: str) -> List[str]:
    seen: List[str] = []
    seen_set: Set[str] = set()
    for m in _HELPER_CALL_RE.finditer(source):
        name = m.group(1)
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
    classification: str
    cells_written: int
    cell_breakdown: Dict[str, int] = field(default_factory=dict)
    helpers: List[str] = field(default_factory=list)
    declarative_helpers: List[str] = field(default_factory=list)
    walked_helpers_count: int = 0
    informational_ir: bool = False
    enabled_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    layer_group: str = "?"


_LAYER_NAME_RE = re.compile(r"l(?:ayer)?[_]?(\d+)|(?:^|_)l(\d+)(?:_|$)")


def _infer_layer_group(name: str, layer_idx: Optional[int]) -> str:
    if layer_idx is not None:
        return f"L{layer_idx}"
    m = _LAYER_NAME_RE.search(name.lower())
    if m:
        digits = m.group(1) or m.group(2)
        return f"L{int(digits)}"
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


_NUM_HEADS = 16


def _augment_stub(block) -> None:
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
        else:
            block = StubBlock(d_model=d_model, num_heads=_NUM_HEADS,
                              ffn_hidden=ffn_hidden)
            _augment_stub(block)
            op.bake_fn(block, dict(dim_positions), 100.0)
            counts = _count_block_cells(block)
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return 0, {}, err
    total = sum(counts.values())
    counts = {k: v for k, v in counts.items() if v}
    return total, counts, None


def _classify_op(op, dim_positions: Dict[str, int], d_model: int,
                 ffn_hidden: int, enabled_flags: List[str]) -> OpCensusRow:
    source = _get_source(op.bake_fn)
    has_ir = op.compiler_ir is not None
    has_ir_factory = op.compiler_ir_factory is not None
    try:
        bake_module = op.bake_fn.__code__.co_filename
        bake_line = op.bake_fn.__code__.co_firstlineno
    except AttributeError:
        bake_module = "?"
        bake_line = 0

    # ---- declarative detection -------------------------------------------
    inline_declarative = _looks_declarative_in_source(source)

    walk = _walk_helpers(op.bake_fn, max_depth=3)
    helpers_with_lower = [h[1] for h in walk.declarative_helpers]

    bake_has_imperative_writes = _has_imperative_writes(source)

    # Helper-declarative: bake_fn has no direct writes, no helper in the
    # transitive chain has direct writes, and at least one helper contains
    # a lowering call.
    helper_declarative = (
        not inline_declarative
        and not bake_has_imperative_writes
        and not walk.any_imperative_writes
        and walk.any_lower_call
    )

    # Declarative-with-residual: at least one declarative lower call is
    # reached (inline in bake_fn OR via helper) AND the total imperative-
    # write count is small. This covers ops whose bulk is IR-driven but
    # that keep a tiny residual of direct writes — for example
    # ``layer6_attn_bake`` uses ``Primitives.generate_attention_heads``
    # for ~306 cells and 6 cross-head boundary-spillover writes to
    # ``attn.W_v/W_o.data[...]``. We exclude truly imperative chains
    # (e.g. ``function_call_weights`` which dispatches to
    # ``vm_step._set_function_call_weights`` and ``l15_attention_resize``
    # whose helpers carry ~232 imperative writes).
    #
    # NOTE: ``_looks_declarative_in_source`` returns False when the bake_fn
    # has any imperative writes (even residuals); to recognize the inline
    # lower call we check the regex directly on ``source`` here so that
    # ``layer6_attn_bake`` (inline ``Primitives.generate_attention_heads``
    # + 2 residual ``attn.W_v.data[...]`` writes in a 6-cell loop) lands
    # in this bucket.
    bake_has_lower_call = _has_lower_call(source)
    inline_imperative_count = _count_imperative_writes(source)
    declarative_with_residual = (
        (inline_declarative or bake_has_lower_call or walk.any_lower_call)
        and (bake_has_imperative_writes or walk.any_imperative_writes)
        and walk.total_imperative_writes <= _DECLARATIVE_WITH_RESIDUAL_MAX_IMP_WRITES
    )

    helpers = _extract_helpers(source)

    cells, breakdown, err = _exercise_op(op, dim_positions, d_model, ffn_hidden)

    # ---- classification --------------------------------------------------
    if inline_declarative and cells > 0:
        classification = "declarative"
    elif inline_declarative and cells == 0:
        classification = "declarative_no_op"
    elif helper_declarative and cells > 0:
        classification = "declarative_via_helper"
    elif helper_declarative and cells == 0:
        classification = "declarative_via_helper_no_op"
    elif declarative_with_residual and cells > 0:
        classification = "declarative_with_residual"
    elif err is not None:
        classification = "unknown"
    elif cells == 0:
        classification = "no_op"
    else:
        classification = _classify_cells(cells)

    # ---- informational-IR flag ------------------------------------------
    # IR attached but the bake (transitively through helpers) doesn't lower
    # it — bake still writes weights through legacy/imperative helpers.
    # ``declarative_with_residual`` ops DO lower their IR (the residual is
    # a small auxiliary patch), so they're not informational.
    informational_ir = bool(
        (has_ir or has_ir_factory)
        and not inline_declarative
        and not helper_declarative
        and not declarative_with_residual
    )

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
        declarative_helpers=helpers_with_lower,
        walked_helpers_count=len(walk.visited),
        informational_ir=informational_ir,
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
    d_model = max(layout.d_model, 1024)
    ffn_hidden = _DEFAULT_FFN_HIDDEN

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
            if prior is None or row.cells_written > prior.cells_written:
                rows_by_name[op.name] = row

    for name, row in rows_by_name.items():
        if name not in seen_in_default:
            row.enabled_flags = sorted(
                set(row.enabled_flags) | {"requires_flag"}
            )

    def _key(r: OpCensusRow):
        return (r.phase if r.phase is not None else 1e9, r.name)

    return sorted(rows_by_name.values(), key=_key)


def aggregate(rows: List[OpCensusRow]) -> Dict[str, Any]:
    total_per_class = Counter(r.classification for r in rows)
    per_kind = Counter(r.kind for r in rows)
    per_layer_per_class: Dict[str, Counter] = {}
    for r in rows:
        per_layer_per_class.setdefault(r.layer_group, Counter())[r.classification] += 1

    top10 = sorted(
        (r for r in rows if r.classification.startswith("imperative") or
         r.classification.startswith("declarative")),
        key=lambda r: r.cells_written,
        reverse=True,
    )[:10]

    cells_per_class = Counter()
    for r in rows:
        cells_per_class[r.classification] += r.cells_written

    ir_status = Counter(
        ("has_ir" if r.has_compiler_ir or r.has_compiler_ir_factory
         else "no_ir") for r in rows
    )

    informational_ir_count = sum(1 for r in rows if r.informational_ir)
    informational_ir_ops = sorted(
        (r.name for r in rows if r.informational_ir)
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
        "informational_ir_count": informational_ir_count,
        "informational_ir_ops": informational_ir_ops,
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
    "# Phase 6 Wave 1A v2 — imperative bake_fn census (helper-aware)\n"
    "\n"
    "Generated by `c4_release/tools/census_imperative_bakes_v2.py`.\n"
    "\n"
    "Same shape as the v1 census but with a recursive classifier that walks\n"
    "module-local helpers referenced from each bake_fn body (depth-bounded).\n"
    "This catches ops like `phase_a_ffn` whose `bake_fn` delegates to a\n"
    "module helper (e.g. `_bake_phase_a_ffn`) that internally calls a\n"
    "lowering primitive (`Primitives.lower_ffn_rules`).\n"
    "\n"
    "Classifications:\n"
    "\n"
    "- `declarative`              — lower call appears directly in `bake_fn`.\n"
    "- `declarative_via_helper`   — lower call appears in a module-local helper\n"
    "                               reached from `bake_fn`; bake_fn itself has\n"
    "                               no direct `W_*.data` writes.\n"
    "- `declarative_with_residual` — bake/helpers issue a declarative lower call\n"
    "                               (bulk of cells come from IR lowering) AND a\n"
    "                               small number (≤ 10) of direct `W_*.data`\n"
    "                               writes that remain as documented residuals\n"
    "                               (e.g. boundary spillover, cross-head fixups).\n"
    "- `declarative_no_op`        — declarative shape but produced 0 cells (flag-off).\n"
    "- `declarative_via_helper_no_op` — same but via helper.\n"
    "- `imperative_{trivial,medium,heavy}` — bake (or its helper chain) writes\n"
    "                               weights cells directly. Buckets: ≤10, 11–100, >100.\n"
    "- `no_op`                    — no cells written and no lower call.\n"
    "- `unknown`                  — bake_fn raised against the stub.\n"
    "\n"
    "Cell counts are non-zero entries of the standard parameter tensors\n"
    "(`attn.W_{q,k,v,o}`, `attn.alibi_slopes`, `ffn.W_{up,gate,down}`,\n"
    "`ffn.b_{up,gate,down}`, plus `model.embed.embed.weight` /\n"
    "`model.head.{weight,bias}` for model-kind ops).\n"
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
    lines.append(
        f"- Informational-only IR (has compiler_ir, bake doesn't lower): "
        f"**{agg['informational_ir_count']}**"
    )
    if agg['informational_ir_ops']:
        for n in agg['informational_ir_ops']:
            lines.append(f"  - `{n}`")
    lines.append("- Cells written per class (sum of nonzero param cells):")
    for k, v in sorted(agg['cells_per_class'].items(), key=lambda kv: -kv[1]):
        lines.append(f"  - `{k}`: **{v}**")
    lines.append("")

    lines.append("## Per-layer breakdown\n")
    cols = ["layer", "declarative", "declarative_via_helper",
            "declarative_with_residual",
            "declarative_no_op", "declarative_via_helper_no_op",
            "imperative_trivial", "imperative_medium", "imperative_heavy",
            "no_op", "unknown"]
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
        "| Op | Layer | Kind | Phase | IR | Class | Cells | Decl helpers | Helpers |"
    )
    lines.append("|---|---|---|---:|:-:|---|---:|---|---|")
    for r in rows:
        ir_flag = "Y" if (r.has_compiler_ir or r.has_compiler_ir_factory) else "n"
        if r.informational_ir:
            ir_flag += "!"  # has IR but bake doesn't lower it
        helpers = ", ".join(f"`{h}`" for h in r.helpers[:3]) or "-"
        decl_helpers = ", ".join(
            f"`{h}`" for h in r.declarative_helpers[:2]
        ) or "-"
        phase = f"{r.phase:g}" if r.phase is not None else "-"
        flags = ""
        if "requires_flag" in r.enabled_flags:
            flags = " (flag-on)"
        lines.append(
            f"| `{r.name}`{flags} | {r.layer_group} | {r.kind} | {phase} "
            f"| {ir_flag} | {r.classification} | {r.cells_written:,} "
            f"| {decl_helpers} | {helpers} |"
        )
    lines.append("")

    imp_total = sum(
        v for k, v in agg['per_class'].items() if k.startswith("imperative")
    )
    decl_inline = (
        agg['per_class'].get("declarative", 0)
        + agg['per_class'].get("declarative_no_op", 0)
    )
    decl_helper = (
        agg['per_class'].get("declarative_via_helper", 0)
        + agg['per_class'].get("declarative_via_helper_no_op", 0)
    )
    no_op = agg['per_class'].get("no_op", 0)
    unknown = agg['per_class'].get("unknown", 0)

    lines.append("## Verdict on plan estimate\n")
    lines.append(
        f"- Observed imperative-classed ops (trivial+medium+heavy): "
        f"**{imp_total}**"
    )
    lines.append(
        f"- Observed declarative ops (inline lower call): **{decl_inline}**"
    )
    lines.append(
        f"- Observed declarative-via-helper ops: **{decl_helper}**"
    )
    lines.append(f"- Pure no_op ops: **{no_op}**; unknown: **{unknown}**")
    lines.append(
        f"- Informational-only IR (has compiler_ir but bake doesn't lower): "
        f"**{agg['informational_ir_count']}**"
    )
    lines.append("")

    # Migration targeting hints
    lines.append("## Migration targeting hints (v1 vs v2)\n")
    lines.append(
        "Ops newly recognized as declarative-via-helper (i.e. already done\n"
        "and should be DROPPED from Wave 2/3/4 imperative-migration targeting):\n"
    )
    helper_decl = [
        r for r in rows
        if r.classification in ("declarative_via_helper",
                                 "declarative_via_helper_no_op")
    ]
    for r in sorted(helper_decl, key=lambda r: (r.layer_group, r.name)):
        decl_helpers = ", ".join(r.declarative_helpers[:2]) or "?"
        lines.append(
            f"- `{r.name}` (L={r.layer_group}, cells={r.cells_written:,}) — "
            f"helper: {decl_helpers}"
        )
    lines.append("")
    lines.append(
        "Ops with informational-only IR (carry `compiler_ir=` but bake still\n"
        "writes weights imperatively — these are GOOD imperative-migration\n"
        "targets, the IR shows intent but the bake hasn't been switched yet):\n"
    )
    informational = [r for r in rows if r.informational_ir]
    for r in sorted(informational, key=lambda r: (r.layer_group, r.name)):
        lines.append(
            f"- `{r.name}` (L={r.layer_group}, kind={r.kind}, "
            f"cells={r.cells_written:,}, class={r.classification})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = _PROJECT_PARENT / "c4_release" / ".agent-logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_census()
    agg = aggregate(rows)

    md_path = out_dir / "imperative_bake_census_phase6_v2.md"
    json_path = out_dir / "imperative_bake_census_phase6_v2.json"
    md_path.write_text(render_markdown(rows, agg))
    json_path.write_text(json.dumps(
        {"aggregate": agg, "rows": [asdict(r) for r in rows]},
        indent=2,
    ))

    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"total ops: {agg['total_ops']}")
    print("per_class:", agg['per_class'])
    print("per_kind:", agg['per_kind'])
    print(f"informational_ir_count: {agg['informational_ir_count']}")


if __name__ == "__main__":
    main()
