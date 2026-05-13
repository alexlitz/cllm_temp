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

import hashlib
import json
import logging
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence

from .layer_compiler import LayerCompiler, ModelLayout, build_model_from_layout

_logger = logging.getLogger(__name__)


_REQUIRE_DECLARATIVE_BAKE_ENV = "C4_REQUIRE_DECLARATIVE_BAKE"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on", "strict", "error"})


class DeclarativeBakeRequirementError(RuntimeError):
    """Raised by the opt-in authoritative declarative-bake gate."""


@dataclass(frozen=True)
class DeclarativeBakeAuthorityReport:
    """Diagnostic report for the authoritative declarative-bake gate."""

    legacy_model_ops: Sequence[str]
    non_migrated_layer_ops: Sequence[str]
    non_migrated_block_ops: Sequence[str]
    unowned_wrapper_model_ops: Sequence[str]

    @property
    def ok(self) -> bool:
        return not (
            self.legacy_model_ops
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


def _is_unowned_wrapper_model_op(name: str) -> bool:
    lowered = name.lower()
    return (
        "wrapper" in lowered
        or "wrap" in lowered
        or lowered == "expand_wrapper_blocks"
    )


def inspect_declarative_bake_authority(
    layout: ModelLayout,
) -> DeclarativeBakeAuthorityReport:
    """Return remaining blockers for an authoritative declarative bake.

    This is intentionally conservative and diagnostic-only by default. The
    gate treats ``legacy_bake`` as a hard blocker, any non-migrated per-layer
    or block op as still relying on skipped/legacy ownership, and wrapper
    model ops as still structurally unowned by declarative per-layer bakes.
    """
    legacy_model_ops = sorted(
        op.name for op in layout.model_ops if op.name == "legacy_bake"
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
        if _is_unowned_wrapper_model_op(op.name)
    )
    return DeclarativeBakeAuthorityReport(
        legacy_model_ops=legacy_model_ops,
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


_CACHE_FORMAT_VERSION = 1


def _cache_dir() -> pathlib.Path:
    """Return the disk-cache directory, honoring ``C4_VM_CACHE_DIR``."""
    env = os.environ.get("C4_VM_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "c4_release" / "compiled_vm"


def _hash_source_bytes() -> str:
    """SHA256 of every compiler source file that affects bake output.

    Includes every ``.py`` file under ``unified_compiler/`` (this module's
    package) plus ``neural_vm/vm_step.py`` (the helpers some bake_fns still
    call into directly). Files are read in sorted-path order so the hash is
    stable across hosts.
    """
    pkg_dir = pathlib.Path(__file__).resolve().parent
    repo_neural_vm = pkg_dir.parent
    sources = sorted(pkg_dir.rglob("*.py"))
    extra = repo_neural_vm / "vm_step.py"
    if extra.exists():
        sources.append(extra)
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

    model = payload["model"]
    layout = ModelLayout(
        d_model=payload["d_model"],
        n_layers=payload["n_layers"],
        ops_per_layer=[[] for _ in range(payload["n_layers"])],
        dim_positions=payload["dim_positions"],
        dim_sizes=payload["dim_sizes"],
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
        "kwargs_snapshot": kwargs_snapshot,
    }
    tmp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: save to a temp file alongside the target, then replace.
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        tmp_path = pathlib.Path(tmp_name)
        _torch.save(payload, tmp_path)
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
    require_declarative_bake: Optional[bool] = None,
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
            ``block_ops`` / ``model_ops`` are empty (the bake ran in the
            producer process); only ``d_model``, ``n_layers``,
            ``dim_positions``, and ``dim_sizes`` are populated. Runtime
            callers (``run_vm``, batch runners) only read those four
            fields, so the cache is transparent to them; tests that
            inspect the per-op placement should pass ``disk_cache=False``.
        use_dynamic_ffn: when True (default), use ``layout.ffn_widths`` to
            pre-size each block's PureFFN to the exact unit count its ops
            need, avoiding the allocate-4096-then-trim overhead. Blocks
            without any FFN-annotated op fall back to ``ffn_hidden`` (4096)
            and are trimmed by ``_right_size_ffns`` post-bake. Set to False
            to force the legacy allocate-4096-everywhere path (used for
            byte-identity comparison).
        require_declarative_bake: opt-in enforcement gate for the migration
            endpoint. ``None`` (default) follows ``C4_REQUIRE_DECLARATIVE_BAKE``.
            When true, compile fails before model bake if the layout still
            contains ``legacy_bake``, non-migrated layer/block ops, or wrapper
            model bakes that are not owned by declarative per-layer ops.

    Returns:
        (model, layout) where:
        - model is an AutoregressiveVM with all weights baked
        - layout is the ModelLayout (d_model, n_layers, dim_positions)
    """
    if require_declarative_bake is None:
        require_declarative_bake = _env_flag_enabled(_REQUIRE_DECLARATIVE_BAKE_ENV)

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
    }
    cache_path = None
    if disk_cache and not require_declarative_bake:
        cache_path = _cache_dir() / f"{_cache_key(kwargs_snapshot)}.pt"
        cached = _try_load_cached(cache_path, kwargs_snapshot)
        if cached is not None:
            return cached

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
    )

    # Run all model-level ops via the compiler dispatch. Per-layer ops are
    # skipped (by default) because legacy_bake is present and owns them.
    # Per-layer ops with migrated=True are dispatched before legacy_bake so
    # their bakes run on the freshly-built model before set_vm_weights edits.
    # Block ops with migrated=True are dispatched similarly.
    import torch as _torch
    with _torch.no_grad():
        # Migrated per-layer ops fire before block/model ops.
        for layer_idx, ops_at_layer in enumerate(layout.ops_per_layer):
            block = model.blocks[layer_idx]
            for op in ops_at_layer:
                if not op.migrated:
                    continue
                if op.kind == "attn":
                    target = block.attn
                elif op.kind == "ffn":
                    target = block.ffn
                else:
                    raise ValueError(
                        f"Op {op.name!r} in ops_per_layer has kind={op.kind!r}; "
                        "expected 'attn' or 'ffn'"
                    )
                op.bake_fn(target, layout.dim_positions, S)

        # Migrated block ops fire before model ops (which include legacy_bake).
        # `_n_layers_hint` lets block bake_fns gate on total layer count (e.g.
        # the L15 attention resize only fires for >=17-layer models).
        # Block op binding via target_op_name (resolved from layout) takes
        # precedence over the legacy layer_idx field.
        for op in sorted(
            layout.block_ops,
            key=lambda o: (layout.resolve_block_op_layer(o), o.phase or 0),
        ):
            if not op.migrated:
                continue
            block = model.blocks[layout.resolve_block_op_layer(op)]
            block._n_layers_hint = len(model.blocks)
            op.bake_fn(block, layout.dim_positions, S)

        for op in sorted(layout.model_ops, key=lambda o: (o.phase or 0)):
            op.bake_fn(model, layout.dim_positions, S)

    if cache_path is not None:
        _try_save_cached(cache_path, model, layout, kwargs_snapshot)

    return model, layout
