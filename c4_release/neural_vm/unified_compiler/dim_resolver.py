"""Canonical residual-dim resolution — the ONE blessed way to turn a dim
NAME into its residual-stream column.

Why this module exists
-----------------------
The widen-repack (host of over-width residual bands + head-dim-preserving
auto-widen) MOVES almost every dim relative to the *static* registry
(``build_default_registry_dynamic``). Concretely, on the current default
build (d_model 920 static → 1221 built):

    OP_PSH          275 → 197
    PSH_AT_SP       467 → 225
    OPCODE_BYTE_LO   12 → 474
    OUTPUT_LO       174 →  69
    AX_CARRY_LO     328 → 362

~93% of named dims move. Any tool/probe that resolves a dim through the
static registry (``build_default_registry_dynamic().slots[name].start`` or a
``+N`` offset off a static start) reads the WRONG residual cell → the value
it prints belongs to some *other* dim → it concludes "signal is dead /
constant" and a bogus "documented wall" gets written. This exact trap cost
four failed byte-1 fix attempts (see memory note
``feedback_probe_dims_use_built_layout_not_static_registry``).

The fix is discipline: resolve dims through the BUILT layout
(``layout.dim_positions``) ONLY. This module packages that discipline in an
ergonomic, cached, guarded API so there is no reason to ever touch the
static registry for a live-model probe again.

Usage
-----
    from neural_vm.unified_compiler.dim_resolver import DimResolver

    # Resolve against an already-built layout (preferred — reuses the model
    # you already compiled):
    r = DimResolver.from_layout(layout)
    col = r.resolve("OP_PSH")                 # -> 197 (built), not 275
    cols = r.resolve_many(["OUTPUT_LO", "AX_CARRY_LO"])
    lo, hi = r.resolve("ALU_LO"), r.resolve("ALU_LO", offset=15)

    # Or let the resolver compile a default model for you (process-cached so
    # a battery of probes shares one bake):
    r = DimResolver.default()

    # Slice helpers:
    r.dim_slice("ALU_LO")          # -> slice(330, 346)  (uses built size)
    r.range("ALU_LO")              # -> range(330, 346)

Unknown names raise ``UnknownDimError`` with the closest known names, so a
typo fails loud instead of silently reading column 0.

This module is TOOLING-ONLY. Importing it or resolving a name does not bake
or mutate any weights — the model is byte-identical whether or not this file
exists.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


class UnknownDimError(KeyError):
    """Raised when a dim name is not present in the BUILT layout.

    Carries the closest known names so a typo (or a stale static-registry
    name that the build renamed / dropped) fails with an actionable message
    instead of a bare ``KeyError``.
    """


@dataclass(frozen=True)
class DimResolver:
    """Resolve residual-dim NAMEs to columns via the BUILT layout.

    Construct with :meth:`from_layout` (you already have a compiled model),
    :meth:`from_positions` (you have a raw ``dim_positions`` mapping), or
    :meth:`default` (let the resolver compile + process-cache a default
    build). Never construct from the static registry — that is the whole
    point.
    """

    positions: Mapping[str, int]
    sizes: Mapping[str, int]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_layout(cls, layout) -> "DimResolver":
        """Build a resolver from a compiled ``ModelLayout`` (the object
        returned as the SECOND element of ``compile_full_vm_dynamic()``).

        Reads ``layout.dim_positions`` (and ``layout.dim_sizes`` when
        present) — the authoritative post-widen layout.
        """
        positions = getattr(layout, "dim_positions", None)
        if positions is None:
            # Some callers hand the raw dict straight through.
            if isinstance(layout, Mapping):
                positions = layout
            else:
                raise TypeError(
                    "DimResolver.from_layout expected a ModelLayout with a "
                    "`.dim_positions` attribute (or a mapping); got "
                    f"{type(layout).__name__}. Pass the SECOND element of "
                    "compile_full_vm_dynamic() -> (model, layout)."
                )
        sizes = getattr(layout, "dim_sizes", {}) or {}
        return cls(positions=dict(positions), sizes=dict(sizes))

    @classmethod
    def from_positions(
        cls,
        positions: Mapping[str, int],
        sizes: Optional[Mapping[str, int]] = None,
    ) -> "DimResolver":
        """Build from a raw ``dim_positions`` mapping (and optional sizes).

        Use this only when you already extracted ``layout.dim_positions``;
        the mapping MUST come from a built layout, not the static registry.
        """
        return cls(positions=dict(positions), sizes=dict(sizes or {}))

    @classmethod
    def default(cls, **compile_kwargs) -> "DimResolver":
        """Compile (or reuse a process-cached) default build and resolve
        against ITS layout.

        A battery of probes calling ``DimResolver.default()`` shares one
        bake per (frozen) kwargs signature — the underlying
        ``compile_full_vm_dynamic`` is invoked once and memoised for the
        life of the process. ``disk_cache`` defaults to True so the on-disk
        cache is also honoured.
        """
        layout = _compile_default_layout(tuple(sorted(compile_kwargs.items())))
        return cls.from_layout(layout)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        """True iff ``name`` is a resolvable dim in the built layout."""
        return name in self.positions

    def resolve(self, name: str, offset: int = 0) -> int:
        """Return the BUILT residual column for ``name`` (+ optional
        ``offset`` into the band).

        Raises :class:`UnknownDimError` (with close-match suggestions) for
        an unknown name — a typo or a stale static-registry name fails loud
        instead of silently returning column ``0``.
        """
        try:
            base = self.positions[name]
        except KeyError:
            raise self._unknown(name) from None
        if offset:
            size = self.sizes.get(name)
            if size is not None and not (0 <= offset < size):
                raise UnknownDimError(
                    f"offset {offset} out of range for dim {name!r} "
                    f"(size {size}; valid 0..{size - 1})"
                )
        return base + offset

    def resolve_many(self, names: Iterable[str]) -> Dict[str, int]:
        """Resolve a batch of names; raises on the FIRST unknown one.

        Returns an ordered ``{name: column}`` dict. Use this to grab a
        battery of dims for a probe in one call.
        """
        out: Dict[str, int] = {}
        for n in names:
            out[n] = self.resolve(n)
        return out

    def size(self, name: str) -> int:
        """Return the built band width for ``name`` (defaults to 1 when the
        layout carried no explicit size)."""
        if name not in self.positions:
            raise self._unknown(name)
        return int(self.sizes.get(name, 1))

    def range(self, name: str) -> range:
        """Return ``range(start, start+size)`` over the band's built cells."""
        start = self.resolve(name)
        return range(start, start + self.size(name))

    def dim_slice(self, name: str) -> slice:
        """Return ``slice(start, start+size)`` for tensor indexing."""
        start = self.resolve(name)
        return slice(start, start + self.size(name))

    def names(self) -> List[str]:
        """All resolvable dim names, sorted by built column."""
        return sorted(self.positions, key=lambda n: self.positions[n])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _unknown(self, name: str) -> UnknownDimError:
        close = difflib.get_close_matches(name, self.positions.keys(), n=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return UnknownDimError(
            f"Unknown dim name {name!r} in the BUILT layout "
            f"({len(self.positions)} known dims).{hint} "
            "NOTE: names/positions come from the BUILT layout "
            "(layout.dim_positions), NOT the static registry — the widen "
            "repack moves ~93% of dims, so a static-registry name/offset is "
            "the wrong cell."
        )


# ----------------------------------------------------------------------
# Process-level bake cache for DimResolver.default()
# ----------------------------------------------------------------------
_LAYOUT_CACHE: Dict[Tuple[Tuple[str, object], ...], object] = {}


def _compile_default_layout(kwargs_key: Tuple[Tuple[str, object], ...]):
    """Compile-or-reuse a default build's layout, keyed on frozen kwargs."""
    if kwargs_key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[kwargs_key]
    # Local import so importing this module never triggers a bake.
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    kwargs = dict(kwargs_key)
    kwargs.setdefault("disk_cache", True)
    _model, layout = compile_full_vm_dynamic(**kwargs)
    _LAYOUT_CACHE[kwargs_key] = layout
    return layout


# ----------------------------------------------------------------------
# Convenience free functions (thin wrappers over DimResolver.default())
# ----------------------------------------------------------------------
def resolve(name: str, offset: int = 0, **compile_kwargs) -> int:
    """One-shot resolve against a process-cached default build.

    Convenience for ``DimResolver.default(**kw).resolve(name, offset)``.
    Prefer constructing a :class:`DimResolver` once and reusing it when you
    resolve many names.
    """
    return DimResolver.default(**compile_kwargs).resolve(name, offset)


def resolve_many(names: Iterable[str], **compile_kwargs) -> Dict[str, int]:
    """One-shot batch resolve against a process-cached default build."""
    return DimResolver.default(**compile_kwargs).resolve_many(names)
