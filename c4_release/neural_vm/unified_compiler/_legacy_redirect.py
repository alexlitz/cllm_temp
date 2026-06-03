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
``compile_full_vm_dynamic`` is deterministic (see ``tests/test_compile_determinism``),
so an on-disk cache keyed on source bytes + kwargs lets pytest processes (and
any other short-lived caller) skip the ~40-70 s bake on cache hit. Each entry
is a pair of sibling files: ``<key>.pt`` (pickled module shell with meta
tensors + layout metadata) and ``<key>.safetensors`` (the real weights). The
shell records the post-bake model structure (right-sized FFNs, attached
post_ops, wrapper modules such as ``FlattenedALUMul`` / ``ALUDivMod``), and
the safetensors file holds essentially all of the bytes and mmap-loads in
milliseconds on the warm-disk path. Loading reproduces the compiled model
byte-identically without re-running any bake_fn. Pass ``disk_cache=False``
to bypass.
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


# Cache format version. Bump invalidates every entry on disk.
#
# v4 → v5: switched model weights serialisation from pickle (``torch.save``)
# to ``safetensors`` for the warm-disk hit. The cache is now a two-file pair:
#
#   <key>.pt           — pickled shell (model with parameters/buffers swapped
#                        to ``meta`` tensors) + layout metadata + kwargs
#                        snapshot. Tiny (sub-MB).
#   <key>.safetensors  — ``model.state_dict()`` written via
#                        ``safetensors.torch.save_file``. Holds the ~830 MB of
#                        weights; loaded via mmap so the warm hit is
#                        sub-half-second.
#
# On load the shell is unpickled, then ``load_state_dict(..., assign=True)``
# swaps the safetensors-backed real tensors back in over the meta
# placeholders. This produces the same post-bake model the pickle path used
# to (including FFN width / wrapper module swaps recorded in the shell's
# class structure), without paying torch.save's pickle-based tensor IO cost.
_CACHE_FORMAT_VERSION = 5

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


def _weights_path_for(shell_path: pathlib.Path) -> pathlib.Path:
    """Return the ``.safetensors`` sibling for a ``.pt`` shell cache path."""
    return shell_path.with_suffix(".safetensors")


def _dedupe_state_dict_for_safetensors(state_dict: dict):
    """Drop aliased entries from ``state_dict`` for ``safetensors`` save.

    Returns ``(deduped_state_dict, alias_map)`` where ``alias_map`` is a dict
    ``{alias_name: keep_name}`` covering every dropped entry. ``safetensors``
    refuses to save tensors that share storage; the compiler's wrapper
    expansion (``_stages`` / ``pipeline`` parallel branches on the same
    submodule) creates these shared-storage aliases for many FFN weights.

    Strategy mirrors ``safetensors.torch.save_model``: group entries by
    storage identity, pick one "keep" name per group (alphabetically first,
    matching ``save_model``'s choice), drop the rest. The alias map is
    persisted in the shell pickle so ``_try_load_cached`` can re-establish
    the sharing after ``load_state_dict``.
    """
    from safetensors.torch import _find_shared_tensors

    groups = _find_shared_tensors(state_dict)
    alias_map: dict = {}
    drop: set = set()
    for group in groups:
        names = sorted(group)
        keep = names[0]
        for name in names[1:]:
            alias_map[name] = keep
            drop.add(name)
    deduped = {k: v for k, v in state_dict.items() if k not in drop}
    return deduped, alias_map


def _reshare_aliased_params(model, alias_map: dict) -> None:
    """Re-establish shared storage between params/buffers that were deduped.

    For each ``alias -> keep`` mapping, locate the underlying
    parameter/buffer attribute on both modules and assign the alias slot to
    reference the same tensor object as the keep slot. This restores the
    bit-identical post-bake module graph that the old pickle path produced
    (where the same module instance was attached to two different parent
    paths, so both names naturally aliased the same tensor).
    """
    import torch as _torch

    # Build a name → (module, attr_name, kind) index so we can look entries
    # up by their dotted ``state_dict`` key. We walk the full module tree
    # once instead of repeatedly resolving paths per alias.
    index: dict = {}
    for mod_name, mod in model.named_modules():
        prefix = mod_name + "." if mod_name else ""
        for name, p in mod._parameters.items():
            if p is None:
                continue
            index[prefix + name] = (mod, name, "param")
        for name, b in mod._buffers.items():
            if name in mod._non_persistent_buffers_set or b is None:
                continue
            index[prefix + name] = (mod, name, "buffer")

    for alias_name, keep_name in alias_map.items():
        alias_entry = index.get(alias_name)
        keep_entry = index.get(keep_name)
        if alias_entry is None or keep_entry is None:
            # The shell graph no longer has one of these names; skip silently
            # rather than failing the whole load. This shouldn't happen in
            # practice for v5+ entries but the loader is defensive.
            continue
        alias_mod, alias_attr, alias_kind = alias_entry
        keep_mod, keep_attr, _keep_kind = keep_entry
        if alias_kind == "param":
            keep_param = keep_mod._parameters[keep_attr]
            alias_mod._parameters[alias_attr] = keep_param
        else:
            alias_mod._buffers[alias_attr] = keep_mod._buffers[keep_attr]


def _strip_module_to_meta(module) -> None:
    """Replace a module's persistent parameters/buffers with meta tensors.

    Meta tensors have shape + dtype but no storage, so they pickle in a few
    hundred bytes regardless of original size. Non-persistent buffers (those
    excluded from ``state_dict``) are left intact — they're not reloaded from
    safetensors and the model needs them at runtime (e.g. compact-routing
    index buffers).
    """
    import torch as _torch

    for name in list(module._parameters.keys()):
        p = module._parameters[name]
        if p is None:
            continue
        module._parameters[name] = _torch.nn.Parameter(
            _torch.empty(p.shape, dtype=p.dtype, device="meta"),
            requires_grad=p.requires_grad,
        )
    for name in list(module._buffers.keys()):
        if name in module._non_persistent_buffers_set:
            continue
        b = module._buffers[name]
        if b is None:
            continue
        module._buffers[name] = _torch.empty(
            b.shape, dtype=b.dtype, device="meta"
        )


def _strip_model_to_meta(model) -> None:
    """Strip every submodule of ``model`` to meta tensors (in place)."""
    for mod in model.modules():
        _strip_module_to_meta(mod)


def _try_load_cached(path: pathlib.Path, kwargs_snapshot: dict):
    """Load a cached compile from ``path``. Returns ``(model, layout)`` or None.

    The cache is a pair of files:
      - ``path`` (``.pt``): pickled shell (model with meta tensors) + layout
        + kwargs snapshot. Tiny.
      - ``path.with_suffix(".safetensors")``: real weights, loaded via mmap.

    On any load failure (missing, corrupt, key collision, version mismatch)
    returns None and the caller falls through to the recompile path. A bad
    pair is deleted so the next run won't keep tripping over it.
    """
    weights_path = _weights_path_for(path)
    if not path.exists() or not weights_path.exists():
        return None
    import pickle as _pickle
    from safetensors.torch import load_file as _safetensors_load

    try:
        with open(path, "rb") as f:
            payload = _pickle.load(f)
    except Exception as exc:
        _logger.warning(
            "compile_full_vm_dynamic: failed to load cache shell %s (%s); "
            "recompiling",
            path, exc,
        )
        for p in (path, weights_path):
            try:
                p.unlink()
            except OSError:
                pass
        return None

    saved_kwargs = payload.get("kwargs_snapshot")
    if saved_kwargs != kwargs_snapshot:
        # Hash collision (vanishingly unlikely) — fall through to recompile
        # without deleting; a future call with the original kwargs may still
        # want this entry.
        _logger.warning(
            "compile_full_vm_dynamic: cache %s kwargs_snapshot mismatch "
            "(saved=%r, requested=%r); recompiling",
            path, saved_kwargs, kwargs_snapshot,
        )
        return None
    if payload.get("format_version") != _CACHE_FORMAT_VERSION:
        return None

    # Defensive: format-version mismatches are already filtered above. This
    # guards against future renames where the shell pickle is written by a
    # newer code path than the loader.
    required_keys = (
        "model", "d_model", "n_layers", "dim_positions", "dim_sizes",
        "ops_per_layer", "block_ops", "model_ops", "ffn_widths",
    )
    for key in required_keys:
        if key not in payload:
            _logger.warning(
                "compile_full_vm_dynamic: cache %s missing field %r; recompiling",
                path, key,
            )
            return None

    model = payload["model"]

    try:
        state_dict = _safetensors_load(str(weights_path), device="cpu")
    except Exception as exc:
        _logger.warning(
            "compile_full_vm_dynamic: failed to load weights %s (%s); "
            "recompiling",
            weights_path, exc,
        )
        for p in (path, weights_path):
            try:
                p.unlink()
            except OSError:
                pass
        return None

    alias_map = payload.get("alias_map") or {}

    try:
        # ``assign=True`` replaces the model's meta-tensor placeholders with
        # the real tensors from safetensors without any shape-coercing copy.
        # ``strict=False`` because the safetensors file only carries the
        # dedup'd (keep) names — the alias slots are still meta and will be
        # rebound to the kept tensors by ``_reshare_aliased_params`` below.
        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        # Every "missing" key should be an alias we know how to re-share.
        unaccounted = [m for m in missing if m not in alias_map]
        if unaccounted or unexpected:
            raise RuntimeError(
                f"load_state_dict produced unexpected={unexpected}, "
                f"unaccounted missing={unaccounted}"
            )
        _reshare_aliased_params(model, alias_map)
    except Exception as exc:
        _logger.warning(
            "compile_full_vm_dynamic: load_state_dict failed for %s (%s); "
            "recompiling",
            path, exc,
        )
        for p in (path, weights_path):
            try:
                p.unlink()
            except OSError:
                pass
        return None

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


_CACHE_MAX_BYTES_ENV = "C4_VM_CACHE_MAX_BYTES"
_CACHE_MAX_ENTRIES_ENV = "C4_VM_CACHE_MAX_ENTRIES"
_CACHE_DEFAULT_MAX_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB
_CACHE_DEFAULT_MAX_ENTRIES = 16


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive int env var; fall back to ``default`` on bad values.

    Returning <= 0 disables the corresponding LRU threshold.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _logger.warning(
            "compile_full_vm_dynamic: %s=%r is not an integer; using default %d",
            name, raw, default,
        )
        return default


def _evict_cache_lru(
    cache_dir: pathlib.Path,
    *,
    keep_path: Optional[pathlib.Path] = None,
    max_bytes: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> List[pathlib.Path]:
    """Evict oldest ``*.pt`` entries from ``cache_dir`` to satisfy LRU bounds.

    The disk cache is unbounded by default — each entry is ~830 MB and a
    handful of compile variants (n_heads / ffn_hidden / alu_mode / ...) can
    push the directory past 30 GB quickly. This helper enforces two caps:

    - ``max_bytes`` (env ``C4_VM_CACHE_MAX_BYTES``, default 10 GiB): total
      on-disk size across ``*.pt`` files.
    - ``max_entries`` (env ``C4_VM_CACHE_MAX_ENTRIES``, default 16): count of
      ``*.pt`` files in the directory.

    Set either env var to ``0`` to disable that cap independently. Eviction
    is best-effort and never raises; the caller (cache writer) is unaffected
    by failures here. ``keep_path`` is never evicted (used to protect the
    entry just written by the current call). Returns the list of evicted
    paths (mostly for tests).
    """
    if max_bytes is None:
        max_bytes = _parse_positive_int_env(
            _CACHE_MAX_BYTES_ENV, _CACHE_DEFAULT_MAX_BYTES
        )
    if max_entries is None:
        max_entries = _parse_positive_int_env(
            _CACHE_MAX_ENTRIES_ENV, _CACHE_DEFAULT_MAX_ENTRIES
        )

    evicted: List[pathlib.Path] = []
    try:
        if not cache_dir.exists():
            return evicted

        entries: List[tuple] = []
        # Each cache entry is the pair (``<key>.pt``, ``<key>.safetensors``);
        # size is the combined on-disk bytes so caps account for the real
        # footprint. The ``.pt`` mtime drives LRU ordering since the loader
        # touches both files together.
        for p in cache_dir.glob("*.pt"):
            try:
                st = p.stat()
            except OSError:
                continue
            weights = _weights_path_for(p)
            size = st.st_size
            try:
                size += weights.stat().st_size
            except OSError:
                pass
            entries.append((st.st_mtime, size, p))

        # Oldest first; LRU = evict from the front.
        entries.sort(key=lambda e: e[0])

        total_bytes = sum(e[1] for e in entries)
        count = len(entries)

        keep_resolved = None
        if keep_path is not None:
            try:
                keep_resolved = keep_path.resolve()
            except OSError:
                keep_resolved = keep_path

        for mtime, size, p in entries:
            over_bytes = max_bytes > 0 and total_bytes > max_bytes
            over_count = max_entries > 0 and count > max_entries
            if not (over_bytes or over_count):
                break
            try:
                p_resolved = p.resolve()
            except OSError:
                p_resolved = p
            if keep_resolved is not None and p_resolved == keep_resolved:
                # Don't evict the file we just wrote even if it's also the
                # oldest (e.g. a single-entry over-budget cache).
                continue
            try:
                p.unlink()
            except OSError as exc:
                _logger.warning(
                    "compile_full_vm_dynamic: failed to evict cache entry %s (%s)",
                    p, exc,
                )
                continue
            # Also drop the safetensors sibling; an orphan would otherwise
            # leak ~830 MB per evicted entry and confuse cache_dir size
            # accounting on the next eviction pass.
            sibling = _weights_path_for(p)
            try:
                sibling.unlink()
            except OSError:
                pass
            evicted.append(p)
            total_bytes -= size
            count -= 1
    except Exception as exc:  # never let LRU break the cache write
        _logger.warning(
            "compile_full_vm_dynamic: LRU eviction in %s failed (%s); skipping",
            cache_dir, exc,
        )
    return evicted


def _try_save_cached(path: pathlib.Path, model, layout, kwargs_snapshot: dict):
    """Save the compiled model to ``path`` atomically. Best-effort.

    Splits the cache into two atomically-replaced files:

      - ``path`` (``.pt``): pickled shell. The model's persistent
        parameters/buffers are swapped to ``meta`` tensors so the shell
        carries only the post-bake module graph (FFN shapes, wrapper module
        classes, compact-routing buffers' non-persistent values) and weighs
        well under a megabyte.
      - ``path.with_suffix(".safetensors")``: the real ``state_dict``,
        written via ``safetensors.torch.save_file``. This holds essentially
        all of the on-disk size; mmap'd on load.

    The model object passed in is *not* mutated — the meta-swap runs on an
    in-memory shallow clone (the original module instances are restored
    immediately afterward). Callers continue to use the returned model.

    After a successful write the disk cache is trimmed via
    ``_evict_cache_lru`` so it does not grow unbounded across compile-variant
    kwargs (see the env knobs ``C4_VM_CACHE_MAX_BYTES`` /
    ``C4_VM_CACHE_MAX_ENTRIES``).
    """
    from safetensors.torch import save_file as _safetensors_save
    import pickle as _pickle

    weights_path = _weights_path_for(path)

    # Capture the live state_dict before stripping. The wrapper-expansion
    # phase attaches the same FFN submodule under both ``_stages.*`` and
    # ``pipeline.*`` paths, so many entries share storage. ``safetensors``
    # refuses shared storages; we dedupe to one keep-name per group and
    # record the alias map in the shell so the loader can re-share.
    raw_state_dict = dict(model.state_dict())
    state_dict, alias_map = _dedupe_state_dict_for_safetensors(raw_state_dict)
    # ``save_file`` requires contiguous tensors; ``raw_state_dict`` entries
    # are usually contiguous but ``.contiguous()`` is a no-op when so.
    state_dict = {
        k: (v.contiguous() if not v.is_contiguous() else v)
        for k, v in state_dict.items()
    }

    # Snapshot the real parameters/buffers per-module so we can restore them
    # after pickling the meta-only shell. We mutate ``model`` in place to
    # avoid a 830 MB deepcopy; the restore step at the end puts everything
    # back exactly as it was, so callers see no change.
    saved_params: List[tuple] = []  # (module, name, value)
    saved_buffers: List[tuple] = []
    import torch as _torch
    for mod in model.modules():
        for name, p in list(mod._parameters.items()):
            if p is None:
                continue
            saved_params.append((mod, name, p))
            mod._parameters[name] = _torch.nn.Parameter(
                _torch.empty(p.shape, dtype=p.dtype, device="meta"),
                requires_grad=p.requires_grad,
            )
        for name, b in list(mod._buffers.items()):
            if name in mod._non_persistent_buffers_set:
                continue
            if b is None:
                continue
            saved_buffers.append((mod, name, b))
            mod._buffers[name] = _torch.empty(
                b.shape, dtype=b.dtype, device="meta"
            )

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
        # ``alias_map`` records {alias_name: keep_name} pairs that were
        # dropped from the safetensors file to satisfy the no-shared-storage
        # rule. The loader uses it to re-establish the aliasing after
        # ``load_state_dict``.
        "alias_map": dict(alias_map),
    }

    tmp_shell = None
    tmp_weights = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: weights first, then shell. Both go to .tmp siblings
        # and are os.replace'd into place. On a partial write the loader
        # checks both files exist before consuming either.
        fd, tmp_weights_name = tempfile.mkstemp(
            dir=str(path.parent), suffix=".weights.tmp"
        )
        os.close(fd)
        tmp_weights = pathlib.Path(tmp_weights_name)
        _safetensors_save(state_dict, str(tmp_weights))

        fd, tmp_shell_name = tempfile.mkstemp(
            dir=str(path.parent), suffix=".shell.tmp"
        )
        os.close(fd)
        tmp_shell = pathlib.Path(tmp_shell_name)
        with open(tmp_shell, "wb") as f:
            _pickle.dump(payload, f, protocol=_pickle.HIGHEST_PROTOCOL)

        os.replace(tmp_weights, weights_path)
        tmp_weights = None
        os.replace(tmp_shell, path)
        tmp_shell = None
    except Exception as exc:
        _logger.warning(
            "compile_full_vm_dynamic: failed to save cache %s (%s); returning "
            "in-memory model anyway",
            path, exc,
        )
    finally:
        for tmp in (tmp_weights, tmp_shell):
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

        # Restore the real tensors onto the live model so the caller sees no
        # change. This runs even when the save raised.
        for mod, name, p in saved_params:
            mod._parameters[name] = p
        for mod, name, b in saved_buffers:
            mod._buffers[name] = b

    # Best-effort: trim cache directory after a successful write. Pass the
    # just-written path as ``keep_path`` so it's protected even if the cap is
    # somehow below a single entry's size (e.g. a misconfigured tiny cap).
    _evict_cache_lru(path.parent, keep_path=path)


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

    ``KVEvictionPolicy.OVERWRITE_BASED`` (Phase 8.E.3) instead builds the
    declarative-IR :class:`~neural_vm.kv_overwrite_map.OverwriteMap` and
    projects it onto each attention via
    :func:`neural_vm.kv_eviction.build_state_from_overwrite_map`. The map
    is a static function of the IR's per-step writers, so spec-decode and
    main-decode reach the same eviction decisions at any step index.
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

    # Phase 8.E.3: OVERWRITE_BASED uses the declarative-IR overwrite map
    # builder. The path is parallel to STATIC_LIVENESS but consumes the
    # static (position, dim) -> overwrite_step table instead of the
    # per-(layer, head) liveness report. Same dim-slice safety semantics,
    # different upstream data source.
    if kv_eviction_policy is KVEvictionPolicy.OVERWRITE_BASED:
        from ..kv_overwrite_map import build_overwrite_map
        from ..kv_eviction import build_state_from_overwrite_map

        overwrite_map = build_overwrite_map(ops, n_steps=n_steps)

        dim_positions = getattr(layout, "dim_positions", None)
        dim_sizes = getattr(layout, "dim_sizes", None)

        for layer_idx, block in enumerate(getattr(model, "blocks", ())):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            state = build_state_from_overwrite_map(
                overwrite_map,
                layer_idx=layer_idx,
                policy=kv_eviction_policy,
                dim_positions=dim_positions,
                dim_sizes=dim_sizes,
                num_heads=getattr(attn, "num_heads", None),
                head_dim=getattr(attn, "head_dim", None),
            )
            setattr(attn, "eviction_state", state)
            if not hasattr(attn, "_eviction_step_idx"):
                setattr(attn, "_eviction_step_idx", 0)
        return

    # Lazy-import the analyzer to keep ``compile_full_vm_dynamic`` import-time light
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

