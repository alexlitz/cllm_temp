"""End-to-end compile path: declare dims+ops, build model, bake all weights.

This is THE production entry point for building a Neural VM. The compiler is
the bake authority: every weight that goes into the model is set by an
Operation registered with `LayerCompiler`.

Pipeline:
  1. Declare all dims (positions pinned to `_SetDim` for backward-compat)
  2. Add per-layer ops from `all_core_ops()` — these drive layout (d_model,
     n_layers) via dependency analysis
  3. Add `legacy_bake` model-level op — bridges the migration: while individual
     ops are being split out of `set_vm_weights` into their own `Operation`
     instances, this op runs the legacy pipeline. As ops migrate out, the
     legacy pipeline shrinks.
  4. `build_model_from_layout` constructs the model and dispatches all ops in
     dependency / phase order

Output: an AutoregressiveVM with all weights baked via the compiler. No
direct call to `set_vm_weights` from outside the compiler module.

On-disk cache
-------------
``compile_full_vm`` is deterministic (see ``tests/test_compile_determinism``),
so an on-disk cache keyed on source bytes + kwargs lets pytest processes (and
any other short-lived caller) skip the ~40-70 s bake on cache hit. The cache
file holds the post-bake model (including the right-sized FFNs, attached
post_ops, and any wrapper modules such as ``FlattenedALUMul`` /
``ALUDivMod`` that the bake pipeline swaps in), so loading reproduces the
compiled model byte-identically without re-running any bake_fn. Pass
``disk_cache=False`` to bypass.
"""

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .layer_compiler import (
    LayerCompiler,
    ModelLayout,
    Operation,
    build_model_from_layout,
    dispatch_operation_bake,
    validate_declarations_only_ops,
)
from ..kv_eviction import (
    KVEvictionPolicy,
    KVEvictionState,
    build_state_from_report,
)

_logger = logging.getLogger(__name__)


_REQUIRE_DECLARATIVE_BAKE_ENV = "C4_REQUIRE_DECLARATIVE_BAKE"
_DECLARATIONS_ONLY_BAKE_ENV = "C4_DECLARATIONS_ONLY_BAKE"
_ENABLE_MOE_ROUTING_ENV = "C4_ENABLE_MOE_ROUTING"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "strict", "error"})


class DeclarativeBakeRequirementError(RuntimeError):
    """Raised by the opt-in authoritative declarative-bake gate."""


@dataclass(frozen=True)
class DeclarativeBakeAuthorityReport:
    """Diagnostic report for the authoritative declarative-bake gate."""

    legacy_model_ops: Sequence[str]
    legacy_wrapper_ops: Sequence[str]
    non_migrated_layer_ops: Sequence[str]
    non_migrated_block_ops: Sequence[str]
    unowned_wrapper_model_ops: Sequence[str]

    @property
    def ok(self) -> bool:
        return not (
            self.legacy_model_ops
            or self.legacy_wrapper_ops
            or self.non_migrated_layer_ops
            or self.non_migrated_block_ops
            or self.unowned_wrapper_model_ops
        )

    def format(self) -> str:
        lines = [
            "Authoritative declarative bake is not yet available.",
            "The default compile path is unchanged; this report only appears "
            f"when {_REQUIRE_DECLARATIVE_BAKE_ENV}=1 or "
            "require_declarative_bake=True.",
        ]
        sections = [
            ("legacy model ops", self.legacy_model_ops),
            ("legacy wrapper ops", self.legacy_wrapper_ops),
            ("non-migrated layer ops", self.non_migrated_layer_ops),
            ("non-migrated block ops", self.non_migrated_block_ops),
            ("unowned wrapper/model bakes", self.unowned_wrapper_model_ops),
        ]
        for label, names in sections:
            if not names:
                continue
            sample = ", ".join(names[:12])
            if len(names) > 12:
                sample += f", ... (+{len(names) - 12} more)"
            lines.append(f"- {label}: {len(names)} [{sample}]")
        return "\n".join(lines)


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in _TRUTHY_ENV_VALUES


_AUTHORITATIVE_DECLARATIVE_SOURCES = frozenset({
    "declarative",
    "spec_generated",
    "structural_model",
    "topology_anchor",
})


def _looks_like_wrapper_model_op(name: str) -> bool:
    lowered = name.lower()
    return "wrapper" in lowered or "wrap" in lowered


def inspect_declarative_bake_authority(
    layout: ModelLayout,
) -> DeclarativeBakeAuthorityReport:
    """Return remaining blockers for an authoritative declarative bake.

    This is intentionally conservative and diagnostic-only by default. The
    gate treats ``legacy_bake`` as a hard blocker, any non-migrated per-layer
    or block op as still relying on skipped/legacy ownership, and wrapper
    model ops as still structurally unowned unless they carry an explicit
    compiler-owned authority marker. Explicit legacy wrappers are always
    blockers, regardless of their name.
    """
    legacy_model_ops = sorted(
        op.name for op in layout.model_ops if op.name == "legacy_bake"
    )
    all_ops = [
        *(op for ops_at_layer in layout.ops_per_layer for op in ops_at_layer),
        *layout.block_ops,
        *layout.model_ops,
    ]
    legacy_wrapper_ops = sorted(
        op.name for op in all_ops if op.declarative_authority == "legacy_wrapper"
    )
    non_migrated_layer_ops = sorted({
        op.name
        for ops_at_layer in layout.ops_per_layer
        for op in ops_at_layer
        if not op.migrated
    })
    non_migrated_block_ops = sorted(
        op.name for op in layout.block_ops if not op.migrated
    )
    unowned_wrapper_model_ops = sorted(
        op.name
        for op in layout.model_ops
        if (
            _looks_like_wrapper_model_op(op.name)
            and op.declarative_authority not in _AUTHORITATIVE_DECLARATIVE_SOURCES
        )
    )
    return DeclarativeBakeAuthorityReport(
        legacy_model_ops=legacy_model_ops,
        legacy_wrapper_ops=legacy_wrapper_ops,
        non_migrated_layer_ops=non_migrated_layer_ops,
        non_migrated_block_ops=non_migrated_block_ops,
        unowned_wrapper_model_ops=unowned_wrapper_model_ops,
    )


def enforce_declarative_bake_authority(layout: ModelLayout) -> DeclarativeBakeAuthorityReport:
    """Raise if the layout still needs non-authoritative bake paths."""
    report = inspect_declarative_bake_authority(layout)
    if not report.ok:
        raise DeclarativeBakeRequirementError(report.format())
    return report

# Re-exported analyzer entry points so callers don't need to reach into
# `layer_compiler` directly. The compiler also runs both scans automatically
# from `LayerCompiler.compile()`; these helpers exist for tests / debugging
# tools that want to inspect the registries without rebuilding the model.


def detect_staleness_violations(compiler: LayerCompiler):
    """Run the staleness-invariant scan on an already-populated compiler.

    Returns the list of warning messages produced. See
    `LayerCompiler._detect_staleness_violations` for the algorithm and
    `c4_release/docs/STALENESS_INVARIANTS.md` for the bake-author API.
    """
    return compiler._detect_staleness_violations()


def build_staleness_registry(compiler: LayerCompiler):
    """Return (producers, consumers) registries for inspection.

    Each registry maps ``(dim_name, register_name)`` -> list of
    ``(op_name, phase)`` tuples across attn / ffn / block / model ops.
    """
    return compiler.build_staleness_registry()
from .migrated_ops import (
    all_core_ops,
    all_alu_postop_attach_ops,
    declare_setdim_compat_dims,
    make_alu_divmod_composite_ops,
    make_contract_validation_op,
    make_efficient_l8_addsub_wrap_op,
    make_efficient_l10_andorxor_wrap_op,
    make_efficient_l11_alumul_wrap_op,
    make_l10_post_op_attach_op,
    make_l11_alu_mul_bdtoge_op,
    make_l11_alu_mul_carrypass1_op,
    make_l11_alu_mul_carrypass2_op,
    make_l11_alu_mul_carrypass3_op,
    make_l11_alu_mul_schoolbook_op,
    make_l12_alu_mul_binarylookahead_op,
    make_l12_alu_mul_finalcorrection_op,
    make_l12_alu_mul_genprop_op,
    make_l12_alu_mul_getobd_op,
    make_layer8_op_imm_relay_op,
    make_layer10_residual_alibi_slopes_op,
    make_residual_alibi_slopes_op,
)


def derive_layout(num_heads: int = 8):
    """Run the LayerCompiler over `all_core_ops` to produce a ModelLayout.

    Returns a layout whose d_model is divisible by `num_heads`, padding via
    a synthetic `_pad` dim if needed.
    """
    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler)
    for op in all_core_ops():
        compiler.add_op(op)
    layout = compiler.compile()
    if layout.d_model % num_heads != 0:
        pad = num_heads - (layout.d_model % num_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()
    return layout


_CACHE_FORMAT_VERSION = 4

# Operation fields that hold callables / closures captured at op-construction
# time (typically inner functions inside the per-op factory). These are not
# picklable in the general case and are not needed at cache-hit time (a cache
# hit short-circuits the bake pipeline), so we strip them before serialising
# the operation list and rely on the metadata-only view for introspection
# (``layout.block_ops``, ``layout.ops_per_layer``, ``layout.model_ops``).
_UNPICKLABLE_OP_FIELDS: Sequence[str] = (
    "bake_fn",
    "declarative_bake_fn",
    "compiler_ir_factory",
)


def _strip_op_for_cache(op: Operation) -> Operation:
    """Return a copy of ``op`` with unpicklable callable fields cleared.

    ``Operation.bake_fn`` and its declarative siblings are typically inner
    closures created inside per-op factory functions and so cannot be
    pickled. Cache-hit consumers only read metadata (name / kind /
    layer_idx / phase / claims / writes / migrated / ...), so dropping the
    bakes is safe: any code path that actually needs to bake should bypass
    the cache (``disk_cache=False``) or recompile.
    """
    return dataclasses.replace(
        op,
        **{name: None for name in _UNPICKLABLE_OP_FIELDS},
    )


def _strip_ops_per_layer_for_cache(
    ops_per_layer: Sequence[Sequence[Operation]],
) -> List[List[Operation]]:
    return [[_strip_op_for_cache(op) for op in layer] for layer in ops_per_layer]


def _strip_op_list_for_cache(ops: Sequence[Operation]) -> List[Operation]:
    return [_strip_op_for_cache(op) for op in ops]


def _cache_dir() -> pathlib.Path:
    """Return the disk-cache directory, honoring ``C4_VM_CACHE_DIR``."""
    env = os.environ.get("C4_VM_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "c4_release" / "compiled_vm"


def _hash_source_bytes() -> str:
    """SHA256 of every compiler source file that affects bake output.

    Includes every ``.py`` file under ``neural_vm/``. The compiler still calls
    helper modules outside ``unified_compiler/`` during bake and post-bake
    transforms (for example MoE routing and efficient ALU composites), so
    hashing only the compiler package can leave stale compiled-model caches
    after a helper implementation changes. Files are read in sorted-path order
    so the hash is stable across hosts.
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent
    repo_neural_vm = pkg_dir.parent
    sources = sorted(repo_neural_vm.rglob("*.py"))
    h = hashlib.sha256()
    for path in sources:
        h.update(str(path.relative_to(repo_neural_vm)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _cache_key(kwargs_snapshot: dict) -> str:
    """Build the cache key = SHA256(source bytes + kwargs JSON + format ver)."""
    h = hashlib.sha256()
    h.update(_hash_source_bytes().encode("utf-8"))
    h.update(b"\0")
    h.update(
        json.dumps(kwargs_snapshot, sort_keys=True, default=repr).encode("utf-8")
    )
    h.update(b"\0")
    h.update(str(_CACHE_FORMAT_VERSION).encode("utf-8"))
    return h.hexdigest()


def _try_load_cached(path: pathlib.Path, kwargs_snapshot: dict):
    """Load a cached compile from ``path``. Returns ``(model, layout)`` or None.

    On any load failure (missing, corrupt, key collision, version mismatch)
    returns None and the caller falls through to the recompile path. A bad
    file is deleted so the next run won't keep tripping over it.
    """
    if not path.exists():
        return None
    import torch as _torch

    try:
        payload = _torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:
        _logger.warning(
            "compile_full_vm: failed to load cache %s (%s); recompiling",
            path, exc,
        )
        try:
            path.unlink()
        except OSError:
            pass
        return None

    saved_kwargs = payload.get("kwargs_snapshot")
    if saved_kwargs != kwargs_snapshot:
        # Hash collision (vanishingly unlikely) — fall through to recompile
        # without deleting; a future call with the original kwargs may still
        # want this entry.
        _logger.warning(
            "compile_full_vm: cache %s kwargs_snapshot mismatch "
            "(saved=%r, requested=%r); recompiling",
            path, saved_kwargs, kwargs_snapshot,
        )
        return None
    if payload.get("format_version") != _CACHE_FORMAT_VERSION:
        return None

    # Backwards-compat guard: format v3 and earlier didn't serialise
    # ``block_ops`` / ``ops_per_layer`` / ``model_ops`` / ``ffn_widths``, so
    # loading would silently return a layout with empty op lists and
    # introspecting tests (e.g. ``tests/test_addr_key_neural_decode.py``) would
    # break. The format-version check above already invalidates those entries;
    # this defensive check covers any future field-renaming slip-ups.
    required_keys = (
        "model", "d_model", "n_layers", "dim_positions", "dim_sizes",
        "ops_per_layer", "block_ops", "model_ops", "ffn_widths",
    )
    for key in required_keys:
        if key not in payload:
            _logger.warning(
                "compile_full_vm: cache %s missing field %r; recompiling",
                path, key,
            )
            return None

    model = payload["model"]
    layout = ModelLayout(
        d_model=payload["d_model"],
        n_layers=payload["n_layers"],
        ops_per_layer=payload["ops_per_layer"],
        dim_positions=payload["dim_positions"],
        dim_sizes=payload["dim_sizes"],
        block_ops=payload["block_ops"],
        model_ops=payload["model_ops"],
        ffn_widths=payload["ffn_widths"],
    )
    return model, layout


def _try_save_cached(path: pathlib.Path, model, layout, kwargs_snapshot: dict):
    """Save the compiled model to ``path`` atomically. Best-effort.

    Saves the full model object so loading reproduces the post-bake state
    (including right-sized FFN shapes and wrapper modules attached during
    the bake) without re-running any bake_fn.
    """
    import torch as _torch

    payload = {
        "format_version": _CACHE_FORMAT_VERSION,
        "model": model,
        "d_model": layout.d_model,
        "n_layers": layout.n_layers,
        "dim_positions": dict(layout.dim_positions),
        "dim_sizes": dict(layout.dim_sizes),
        # Persist the per-layer / block / model op metadata so layouts
        # restored from cache match the freshly-compiled ones for tests
        # and tools that introspect ``layout.block_ops`` / ``ops_per_layer``
        # / ``model_ops``. Inner-closure ``bake_fn`` / ``declarative_bake_fn``
        # / ``compiler_ir_factory`` fields are stripped before pickling
        # because they are not picklable in the general case and aren't
        # needed when the cache short-circuits the bake.
        "ops_per_layer": _strip_ops_per_layer_for_cache(layout.ops_per_layer),
        "block_ops": _strip_op_list_for_cache(layout.block_ops),
        "model_ops": _strip_op_list_for_cache(layout.model_ops),
        "ffn_widths": dict(layout.ffn_widths),
        "kwargs_snapshot": kwargs_snapshot,
    }
    tmp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: save to a temp file alongside the target, then replace.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        tmp_path = pathlib.Path(tmp_name)
        # The default zip writer has been observed to fail near the end of
        # this large model payload with a small "unexpected pos" mismatch on
        # some hosts. The legacy stream format is slower to write but has been
        # reliable for this cache, and cache reads remain transparent through
        # torch.load(..., weights_only=False).
        _torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
        tmp_path = None  # replaced; nothing to clean up
    except Exception as exc:
        _logger.warning(
            "compile_full_vm: failed to save cache %s (%s); returning "
            "in-memory model anyway",
            path, exc,
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _attach_kv_eviction_state(
    model,
    layout: ModelLayout,
    *,
    kv_eviction_policy: KVEvictionPolicy,
    n_steps: int,
) -> None:
    """Phase 7.F.2: attach per-layer ``KVEvictionState`` to each block's attn.

    The policy is the single switch. ``KVEvictionPolicy.OFF`` clears any
    previously-attached state on every block (so a cached model built with
    STATIC_LIVENESS but reloaded with OFF reverts to byte-identity with the
    baseline) and returns without running the analyzer.

    ``KVEvictionPolicy.STATIC_LIVENESS`` runs
    :func:`neural_vm.kv_liveness_analyzer.analyze_kv_liveness` once over
    the layout's full op corpus, then projects the report onto each
    attention layer via :func:`build_state_from_report` and attaches the
    resulting :class:`KVEvictionState` to ``block.attn``. The decisions are
    deterministic functions of the static IR, so spec-decode and main-decode
    paths see identical eviction sets when given the same step index.
    """

    # Bottom-out: OFF clears any prior attached state and skips analysis.
    if kv_eviction_policy is KVEvictionPolicy.OFF:
        for block in getattr(model, "blocks", ()):
            attn = getattr(block, "attn", None)
            if attn is not None:
                # Use object.__setattr__ to defeat any custom __setattr__ traps;
                # nn.Module sets attributes via plain attribute assignment but
                # we go through setattr for safety with sparse / compact wrappers.
                setattr(attn, "eviction_state", None)
        return

    # Collect every op the compiler placed into the layout. The analyzer is
    # read-only and tolerates ops without compiler_ir (no IR -> contributes
    # only declared op.reads / op.writes / op.step_idx).
    ops: List[Operation] = []
    for ops_at_layer in layout.ops_per_layer:
        ops.extend(ops_at_layer)
    ops.extend(layout.block_ops)
    ops.extend(layout.model_ops)

    # Lazy-import the analyzer to keep ``compile_full_vm`` import-time light
    # and to avoid pulling its IR-walking helpers into the OFF path.
    from ..kv_liveness_analyzer import analyze_kv_liveness

    report = analyze_kv_liveness(ops, n_steps=n_steps)

    # Pass dim_positions / dim_sizes to the state builder so the
    # Phase 7.F.5 per-(position, dim_slice) eviction map gets populated.
    # Falls back gracefully when a layout omits either map (older cached
    # payloads): the builder only populates the new map when both are
    # provided.
    dim_positions = getattr(layout, "dim_positions", None)
    dim_sizes = getattr(layout, "dim_sizes", None)

    # When the analyzer can't identify per-layer head sets (because all
    # attention ops in this build are imperative — they have no
    # ``compiler_ir`` with ``AttentionHeadIR`` entries), the report's
    # entries are all stamped layer=0 (the analyzer's fallback). In that
    # case the per-layer projection would only populate layer 0's state,
    # leaving the other 30+ attention modules with empty maps. Broadcast
    # the same state to every layer instead — the dim-name liveness
    # semantics are intrinsically layer-independent for the dim slices
    # the slice path zeros (TEMP / *_PREV_STEP / safe cycle dims).
    analyzer_saw_specific_heads = bool(
        {
            (int(getattr(op, "layer_idx", 0) or 0),)
            for op in ops
            if any(True for _ in _iter_attention_heads_in_op(op))
        }
    )

    for layer_idx, block in enumerate(getattr(model, "blocks", ())):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        # Per-layer projection; head=None means "every head in the layer
        # shares the same evictable set". The state is a small dataclass
        # so attaching one per attention is O(layers) memory.
        state = build_state_from_report(
            report,
            # When the analyzer didn't see per-layer head specs, drop
            # the layer filter so every layer inherits the same dim
            # slice map. This is sound because the dim names the slice
            # path operates on (TEMP / *_PREV_STEP / safe cycle dims)
            # are residual-stream identifiers that every layer shares.
            layer_idx=layer_idx if analyzer_saw_specific_heads else None,
            head_idx=None,
            policy=kv_eviction_policy,
            dim_positions=dim_positions,
            dim_sizes=dim_sizes,
            num_heads=getattr(attn, "num_heads", None),
            head_dim=getattr(attn, "head_dim", None),
        )
        setattr(attn, "eviction_state", state)
        # Make sure the runtime step counter exists so PureAttention's
        # ``run_eviction_hook`` finds the attribute it expects.
        if not hasattr(attn, "_eviction_step_idx"):
            setattr(attn, "_eviction_step_idx", 0)


def _iter_attention_heads_in_op(op):
    """Yield ``AttentionHeadIR`` entries from an op's ``compiler_ir``.

    Returns an empty iterator when the op is imperative (no ``compiler_ir``)
    or when the IR doesn't carry attention-head specs. Used by
    :func:`_attach_kv_eviction_state` to decide whether the analyzer's
    layer-specific entries are usable as a layer filter.
    """

    ir = getattr(op, "compiler_ir", None)
    if ir is None:
        return
    layers = getattr(ir, "layers", ())
    for layer in layers:
        attn = getattr(layer, "attention", None)
        if attn is None:
            continue
        for head in getattr(attn, "rules", ()):
            yield head


def compile_full_vm(
    S: float = 100.0,
    *,
    enable_conversational_io: bool = False,
    enable_tool_calling: bool = False,
    enable_neural_io_think_protocol: bool = False,
    alu_mode: str = "lookup",
    n_heads: int = 8,
    ffn_hidden: int = 4096,
    max_seq_len: int = 8192,
    pin_io_only: bool = True,
    disk_cache: bool = True,
    use_dynamic_ffn: bool = True,
    enable_moe_routing: Optional[bool] = None,
    positional_encoding: Optional[str] = None,
    attention_normalization: Optional[str] = None,
    rope_base: Optional[float] = None,
    use_rms_norm: Optional[bool] = None,
    rms_norm_eps: Optional[float] = None,
    require_declarative_bake: Optional[bool] = None,
    declarations_only: bool = False,
    kv_eviction_policy: KVEvictionPolicy = KVEvictionPolicy.OFF,
    kv_eviction_n_steps: int = 64,
):
    """Compile and bake a full Neural VM model via the compiler.

    The compiler is the single bake authority. Internally, the legacy
    `set_vm_weights` pipeline is wrapped as one model-level Operation
    (`legacy_bake`) that the compiler dispatches. As individual ops migrate
    out of `set_vm_weights` into their own per-layer Operation instances, the
    legacy_bake op shrinks and eventually disappears.

    Args:
        pin_io_only: when True, only IO-required dims (the externally-
            observable ones read/written by token embedding, the head, and
            `_inject_*` runtime injectors) are pinned, and they are pinned to
            a *compact contiguous block* starting at position 0. Every other
            dim is bump-pointer-allocated above the IO block, shrinking
            d_model relative to the legacy `_SetDim`-pinned layout. See
            `declare_setdim_compat_dims` for details. Defaults to False for
            backward compatibility.
        disk_cache: when True (default), look up / write a persistent
            on-disk cache so successive Python processes (e.g. pytest
            workers) skip the bake pipeline. See module docstring for the
            cache-key construction and the ``C4_VM_CACHE_DIR`` override.
            On cache hit the returned layout's ``ops_per_layer`` /
            ``block_ops`` / ``model_ops`` are populated with metadata-only
            copies of the operations (the inner-closure ``bake_fn`` /
            ``declarative_bake_fn`` / ``compiler_ir_factory`` fields are
            stripped because they aren't picklable in the general case;
            all other metadata — name, kind, layer_idx, phase, claims,
            reads/writes, migrated, etc. — is preserved). Tests that
            introspect the layout work transparently across cold and warm
            cache runs; only callers that need to *bake* via the cached
            ops should pass ``disk_cache=False`` to bypass.
        use_dynamic_ffn: when True (default), use ``layout.ffn_widths`` to
            pre-size each block's PureFFN to the exact unit count its ops
            need, avoiding the allocate-4096-then-trim overhead. Blocks
            without any FFN-annotated op fall back to ``ffn_hidden`` (4096)
            and are trimmed by ``_right_size_ffns`` post-bake. Set to False
            to force the legacy allocate-4096-everywhere path (used for
            byte-identity comparison).
        enable_moe_routing: when true, emit the compiler-built model with
            FFN blocks converted to standard top-1 ``StandardMoEFFN`` experts.
            ``None`` (default) follows ``C4_ENABLE_MOE_ROUTING``. This is a
            compiler-owned structural transform and is included in the
            persistent compile cache key.
        positional_encoding: optional architecture override. ``None`` follows
            ``VMConfig`` / ``NEURAL_VM_POS_ENCODING``. Default remains ALiBi.
        attention_normalization: optional architecture override. ``None``
            follows ``VMConfig`` / ``NEURAL_VM_ATTENTION_NORMALIZATION``.
            Default remains ``"softmax1"``; pass ``"softmax"`` for standard
            decoder attention normalization.
        rope_base: optional RoPE base override. ``None`` follows ``VMConfig``.
        use_rms_norm: optional architecture override. ``None`` follows
            ``VMConfig`` / ``NEURAL_VM_USE_RMS_NORM``. Default remains off.
        rms_norm_eps: optional RMSNorm epsilon. ``None`` follows ``VMConfig``.
        require_declarative_bake: opt-in enforcement gate for the migration
            endpoint. ``None`` (default) follows ``C4_REQUIRE_DECLARATIVE_BAKE``.
            When true, compile fails before model bake if the layout still
            contains ``legacy_bake``, non-migrated layer/block ops, or wrapper
            model bakes that are not owned by declarative per-layer ops.
        declarations_only: opt-in prototype runner for the end-state bake.
            When true, compile refuses to call imperative ``Operation.bake_fn``
            bodies and only dispatches ``declarative_bake_fn`` generators (plus
            no-op topology anchors). This also enforces declarative authority
            and bypasses the disk cache so unsupported ops are visible.
        kv_eviction_policy: Phase 7.F.2 runtime KV-eviction policy. Default
            :data:`KVEvictionPolicy.OFF` preserves byte-identity with all
            historical baselines (no eviction state attached). When set to
            :data:`KVEvictionPolicy.STATIC_LIVENESS`, the compiler runs
            :func:`neural_vm.kv_liveness_analyzer.analyze_kv_liveness` against
            the registered op corpus, projects the report onto each attention
            layer via :func:`neural_vm.kv_eviction.build_state_from_report`,
            and attaches the resulting :class:`KVEvictionState` to every
            ``model.blocks[i].attn`` instance. The forward-pass step boundary
            hook in :class:`PureAttention.forward` consults the state
            deterministically (same decisions across spec-decode and
            main-decode paths). The flag is included in the disk-cache key
            so OFF and STATIC_LIVENESS builds don't collide.
        kv_eviction_n_steps: Upper bound on VM steps the liveness analyzer
            reasons about when ``kv_eviction_policy`` is on. The default of
            64 covers the smoke / 1096 test corpus; larger values widen the
            evictable window at the cost of analysis time. Ignored when the
            policy is OFF.

    Environment variables:
        C4_VALIDATE_ON_COMPILE: when set to ``"1"``, run Mode A of the
            declarative verifier (``verify_claims_static``) at the end of
            compile as a warn-only sanity check. Drift is surfaced via
            ``warnings.warn`` and never fails the compile -- the verifier
            is opt-in diagnostic, not a production gate. Default off.
        C4_VALIDATE_VERBOSE: when set to ``"1"`` alongside
            ``C4_VALIDATE_ON_COMPILE``, print the full verifier report to
            stdout instead of just the summary warning.
        NEURAL_VM_POS_ENCODING: ``alibi`` (default), ``rope``, or ``hybrid``.
        NEURAL_VM_ATTENTION_NORMALIZATION: ``softmax1`` (default) or
            ``softmax``.
        NEURAL_VM_USE_RMS_NORM: truthy values enable RMSNorm pre-norm blocks.

    Returns:
        (model, layout) where:
        - model is an AutoregressiveVM with all weights baked
        - layout is the ModelLayout (d_model, n_layers, dim_positions)
    """
    if not declarations_only:
        declarations_only = _env_flag_enabled(_DECLARATIONS_ONLY_BAKE_ENV)
    if enable_moe_routing is None:
        enable_moe_routing = _env_flag_enabled(_ENABLE_MOE_ROUTING_ENV)
    if require_declarative_bake is None:
        require_declarative_bake = _env_flag_enabled(_REQUIRE_DECLARATIVE_BAKE_ENV)
    if declarations_only:
        require_declarative_bake = True

    from ..config import get_config
    vm_config = get_config()
    if positional_encoding is None:
        positional_encoding = vm_config.positional_encoding
    if attention_normalization is None:
        attention_normalization = vm_config.attention_normalization
    if rope_base is None:
        rope_base = vm_config.rope_base
    if use_rms_norm is None:
        use_rms_norm = vm_config.use_rms_norm
    if rms_norm_eps is None:
        rms_norm_eps = vm_config.rms_norm_eps

    kwargs_snapshot = {
        "S": S,
        "enable_conversational_io": enable_conversational_io,
        "enable_tool_calling": enable_tool_calling,
        "enable_neural_io_think_protocol": enable_neural_io_think_protocol,
        "alu_mode": alu_mode,
        "n_heads": n_heads,
        "ffn_hidden": ffn_hidden,
        "max_seq_len": max_seq_len,
        "pin_io_only": pin_io_only,
        "enable_moe_routing": bool(enable_moe_routing),
        "positional_encoding": positional_encoding,
        "attention_normalization": attention_normalization,
        "rope_base": float(rope_base),
        "use_rms_norm": bool(use_rms_norm),
        "rms_norm_eps": float(rms_norm_eps),
        "require_declarative_bake": bool(require_declarative_bake),
        "declarations_only": bool(declarations_only),
        # Phase 7.F.2: include eviction policy in the cache key so OFF and
        # STATIC_LIVENESS builds don't share a serialised model.
        "kv_eviction_policy": KVEvictionPolicy(kv_eviction_policy).value,
        "kv_eviction_n_steps": int(kv_eviction_n_steps),
    }
    cache_path = None
    cache_key = _cache_key(kwargs_snapshot) if disk_cache else None
    if disk_cache and not require_declarative_bake and not declarations_only:
        cache_path = _cache_dir() / f"{cache_key}.pt"
        cached = _try_load_cached(cache_path, kwargs_snapshot)
        if cached is not None:
            # Phase 7.F.2: re-attach KV eviction state on cache hit. The
            # cache key includes the policy so cached models already match
            # the requested policy; the re-attach is a defensive idempotent
            # step that protects against stale caches built before the flag
            # existed (analyzer + attach are cheap relative to a full bake).
            cached_model, cached_layout = cached
            _attach_kv_eviction_state(
                cached_model,
                cached_layout,
                kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
                n_steps=int(kv_eviction_n_steps),
            )
            return cached_model, cached_layout

    compiler = LayerCompiler()
    declare_setdim_compat_dims(compiler, pin_io_only=pin_io_only)

    # Per-layer ops drive the layout (d_model, n_layers, dim_positions).
    # Forward alu_mode so SHL/SHR (and any future alu_mode-aware migrated op)
    # can branch between the legacy lookup-table bake and the efficient
    # neural-ALU bake. Forward enable_conversational_io / enable_tool_calling
    # to flag-gated ops (registered unconditionally to keep the dep graph
    # stable) so they fire their bakes when the corresponding flag is on.
    for op in all_core_ops(
        alu_mode=alu_mode,
        enable_conversational_io=enable_conversational_io,
        enable_tool_calling=enable_tool_calling,
        enable_neural_io_think_protocol=enable_neural_io_think_protocol,
    ):
        compiler.add_op(op)

    # L10 post_op attach: runs as a migrated block op (phase=10.7) before the
    # legacy bake pipeline. Replaces the inline post_op appends previously
    # done in set_vm_weights. alu_mode-dependent, so it lives outside
    # all_core_ops().
    compiler.add_op(make_l10_post_op_attach_op(alu_mode=alu_mode))

    # L11/L12 MUL ALU flattening: 9 phase-ordered block ops at L11 install
    # the FlattenedALUMul wrapper. Phases 11.0..12.3 split the previous
    # monolithic `model.blocks[11].ffn = ALUMul(...)` assignment in
    # set_vm_weights into discrete BD↔GE conversion + 7 sub-FFN MUL pipeline
    # stages. All bind to layer_idx=11 (the runtime forward collapses them
    # into one block call). Only registered in efficient mode — lookup mode
    # keeps the `_set_layer11_mul_partial` / `_set_layer12_mul_combine`
    # lookup tables; the L11 ALUMul module is attached to ``block.post_ops``
    # by ``make_l11_alu_postop_attach_op`` and split out by
    # ``_expand_wrapper_blocks``.
    if alu_mode == "efficient":
        compiler.add_op(make_l11_alu_mul_bdtoge_op())
        compiler.add_op(make_l11_alu_mul_schoolbook_op())
        compiler.add_op(make_l11_alu_mul_carrypass1_op())
        compiler.add_op(make_l11_alu_mul_carrypass2_op())
        compiler.add_op(make_l11_alu_mul_carrypass3_op())
        compiler.add_op(make_l12_alu_mul_genprop_op())
        compiler.add_op(make_l12_alu_mul_binarylookahead_op())
        compiler.add_op(make_l12_alu_mul_finalcorrection_op())
        compiler.add_op(make_l12_alu_mul_getobd_op())

        # Efficient-mode ALU wrapper installs — replace the inline
        # ``model.blocks[N].ffn = ...`` assignments previously in legacy_bake's
        # efficient branch. See migrated_ops for ordering notes.
        compiler.add_op(make_efficient_l8_addsub_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l10_andorxor_wrap_op(alu_mode=alu_mode))
        compiler.add_op(make_efficient_l11_alumul_wrap_op(alu_mode=alu_mode))

    # L10 DIV/MOD ALU flattening: 4 cooperating block ops install the
    # FlattenedDivMod composite (BD→GE, long-division pipeline, GE→BD,
    # plus an install op that appends to model.blocks[10].post_ops).
    # Replaces the previous EfficientDivMod_Neural runtime instantiations
    # (lookup-mode override at vm_step.py and efficient-mode append in
    # make_l10_post_op_attach_op). All 4 ops share a builder so the install
    # op (phase=10.8) gets the fully-assembled composite. Both alu_modes
    # use the flattened composite — its forward is byte-identical to the
    # previous EfficientDivMod_Neural.
    for op in make_alu_divmod_composite_ops():
        compiler.add_op(op)

    # Residual ALiBi-slope bakes (previously inline in set_vm_weights):
    #   phase=999 :  L6/L8/L14/L15 alibi_slopes (mode-agnostic)
    #   phase=999.1: L10 alibi_slopes (mode-conditional: lookup vs efficient)
    # phase=8.4 (block op): L8 head 4 OP_IMM relay (previously inline)
    # phase=1199 (model op): contract validation diagnostic
    # The alu_postop_attach_ops are added below in the lookup branch.
    compiler.add_op(make_residual_alibi_slopes_op())
    compiler.add_op(make_layer10_residual_alibi_slopes_op(alu_mode=alu_mode))
    compiler.add_op(make_layer8_op_imm_relay_op())
    compiler.add_op(make_contract_validation_op())

    # ALU post-op attach ops (lookup mode only): attach a structural neural
    # ALU to each L8-L13 block's ``post_ops`` so it runs on top of the baked
    # lookup-table FFN. Must run AFTER the L8-L13 FFN bakes complete, which
    # is naturally the case here because phase=L+0.5 sits AFTER the block-op
    # phases (8.0-8.5, 9, 10.0-10.85, 11, 12, 13) and the phase-999
    # residuals.
    if alu_mode == 'lookup':
        for op in all_alu_postop_attach_ops():
            compiler.add_op(op)

    layout = compiler.compile()
    if layout.d_model % n_heads != 0:
        pad = n_heads - (layout.d_model % n_heads)
        compiler.declare_dim("_pad", pad)
        layout = compiler.compile()

    if require_declarative_bake:
        enforce_declarative_bake_authority(layout)

    # Run all model-level ops via the compiler dispatch. Per-layer ops are
    # skipped (by default) because legacy_bake is present and owns them.
    # Per-layer ops with migrated=True are dispatched before legacy_bake so
    # their bakes run on the freshly-built model before set_vm_weights edits.
    # Block ops with migrated=True are dispatched similarly.
    import torch as _torch
    per_layer_dispatch = [
        (layer_idx, op)
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer)
        for op in ops_at_layer
        if op.migrated
    ]
    block_dispatch = [
        op for op in sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        )
        if op.migrated
    ]
    model_dispatch = sorted(layout.model_ops, key=lambda o: (o.phase or 0))
    if declarations_only:
        validate_declarations_only_ops(
            [op for _, op in per_layer_dispatch]
            + block_dispatch
            + model_dispatch
        )
        if disk_cache:
            # Declarations-only is the authoritative endpoint, but it must
            # still validate ownership before a cache hit can bypass the bake.
            # The layout compile + validation above is cheap; loading here
            # keeps unsupported imperative ops visible while avoiding repeated
            # 40s+ model bakes in smoke/1096 test runs.
            cache_path = _cache_dir() / f"{cache_key}.pt"
            cached = _try_load_cached(cache_path, kwargs_snapshot)
            if cached is not None:
                # Phase 7.F.2: same defensive re-attach as in the main cache
                # hit path above.
                cached_model, cached_layout = cached
                _attach_kv_eviction_state(
                    cached_model,
                    cached_layout,
                    kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
                    n_steps=int(kv_eviction_n_steps),
                )
                return cached_model, cached_layout

    # Build the model with d_model/n_layers from the compiler. We override
    # the default size from build_model_from_layout to set ffn_hidden,
    # n_heads, and max_seq_len that AutoregressiveVM expects.
    from ..vm_step import AutoregressiveVM

    # Per-block FFN hidden_dim from layout when use_dynamic_ffn is on.
    # Blocks without an annotated op are absent from ``layout.ffn_widths``
    # and fall back to ``ffn_hidden`` inside AutoregressiveVM.__init__'s
    # dict-dispatch path; those still get trimmed by ``_right_size_ffns``
    # post-bake. As ops accumulate ``ffn_units_used`` annotations, more
    # blocks land in the dict and skip the allocate-then-trim overhead.
    if use_dynamic_ffn and layout.ffn_widths:
        ffn_hidden_arg = layout.ffn_widths
    else:
        ffn_hidden_arg = ffn_hidden

    model = AutoregressiveVM(
        d_model=layout.d_model,
        n_layers=layout.n_layers,
        n_heads=n_heads,
        ffn_hidden=ffn_hidden_arg,
        max_seq_len=max_seq_len,
        dim_positions=layout.dim_positions,
        positional_encoding=positional_encoding,
        attention_normalization=attention_normalization,
        rope_base=rope_base,
        use_rms_norm=use_rms_norm,
        rms_norm_eps=rms_norm_eps,
    )

    with _torch.no_grad():
        # Migrated per-layer ops fire before block/model ops.
        for layer_idx, op in per_layer_dispatch:
            block = model.blocks[layer_idx]
            if op.kind == "attn":
                target = block.attn
            elif op.kind == "ffn":
                target = block.ffn
            else:
                raise ValueError(
                    f"Op {op.name!r} in ops_per_layer has kind={op.kind!r}; "
                    "expected 'attn' or 'ffn'"
                )
            dispatch_operation_bake(
                op, target, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

        # Migrated block ops fire before model ops (which include legacy_bake).
        # `_n_layers_hint` lets block bake_fns gate on total layer count (e.g.
        # the L15 attention resize only fires for >=17-layer models).
        # Block op binding via target_op_name (resolved from layout) takes
        # precedence over the legacy layer_idx field.
        for op in block_dispatch:
            block = model.blocks[layout.resolve_block_op_layer(op)]
            block._n_layers_hint = len(model.blocks)
            dispatch_operation_bake(
                op, block, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

        for op in model_dispatch:
            dispatch_operation_bake(
                op, model, layout.dim_positions, S,
                declarations_only=declarations_only,
            )

    if enable_moe_routing:
        # Compiler-owned MoE emission: callers receive a model whose routed
        # FFNs are already represented as tensor-native top-1 experts. Keep
        # this after all bakes so the partition observes final weights, and
        # before cache save so MoE artifacts are reused across test runs.
        model.compact(block_size=32)
        model.compact_moe()

    # Phase 7.F.2: attach per-attention :class:`KVEvictionState` artifacts.
    # The hook short-circuits to a no-op when policy is OFF (preserves
    # byte-identity with all historical baselines). When STATIC_LIVENESS,
    # the analyzer runs against the full op corpus to compute the per-step
    # evictable set; the state is then projected onto every block's attn
    # (both PureAttention and AutoregressiveAttention; the latter relies
    # on the caller invoking ``run_eviction_hook`` at the step boundary).
    _attach_kv_eviction_state(
        model,
        layout,
        kv_eviction_policy=KVEvictionPolicy(kv_eviction_policy),
        n_steps=int(kv_eviction_n_steps),
    )

    if cache_path is not None:
        _try_save_cached(cache_path, model, layout, kwargs_snapshot)

    # Opt-in declarative verifier hook. Gated on C4_VALIDATE_ON_COMPILE
    # so it never runs in production builds; tests / CI flip it on to
    # surface declaration drift at compile time. Always warn-only --
    # the verifier is diagnostic, never a hard fail.
    if os.environ.get("C4_VALIDATE_ON_COMPILE") == "1":
        import warnings
        try:
            from .decl_verifier import verify_claims_static
            report = verify_claims_static()
            if report.has_errors():
                warnings.warn(
                    f"C4_VALIDATE_ON_COMPILE=1: declarative verifier "
                    f"surfaced {len(report.results)} drift entries. "
                    f"Set C4_VALIDATE_VERBOSE=1 for details.",
                    stacklevel=2,
                )
                if os.environ.get("C4_VALIDATE_VERBOSE") == "1":
                    print(report.format())
        except Exception as e:
            # Validator is opt-in diagnostic -- never fail compile.
            warnings.warn(
                f"C4_VALIDATE_ON_COMPILE: verifier raised {e}; continuing"
            )

    return model, layout
