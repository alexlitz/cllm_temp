"""Op-local residual-band declarations + auto-collection.

Replaces the centralized ``_PRODUCTION_EXTRA_RESIDUAL_DIMS`` dict that every
band-adding op had to hand-edit (a cross-lane merge-conflict magnet). An op
now declares the over-width residual band(s) it needs LOCALLY, right next to
the op that uses them, via :func:`register_residual_band`. The compiler
(:func:`collect_registered_residual_bands`, called from
``compile_full_vm_dynamic``) AUTO-COLLECTS every registered band and feeds the
union — together with any caller-passed ``extra_residual_dims`` and the
``C4_EXTRA_RESIDUAL_DIMS`` env override — into the SINGLE head-dim-preserving
auto-widen path. The collected set flows into the disk + in-proc cache keys and
the ``_LIVENESS_NEVER_SHARE`` set automatically.

Why a module-level registry (vs. a field on the op factory)
-----------------------------------------------------------
Residual bands must be known BEFORE op collection / scheduling runs: they grow
``d_model`` (so they drive the head-dim-preserving widen and the cache key) and
they must be forward-declared as pending dim names so the ops that reference
them validate at ``add_op`` time. An op factory's return value is only seen
DURING collection, too late. A module-level registry populated at IMPORT time
of the op module (``all_core_ops`` wildcard-imports every ``lN_ops`` module, so
every band declaration runs) is read by the compiler up front.

Flag-gated bands
----------------
A band whose presence depends on a runtime env flag (mul ``C4_MUL_WIDTH2``,
Root 2 ``C4_STACK0_B0_DUMP``, AX ``C4_AX_BYTE1_DUMP``) passes a zero-arg
``flag`` predicate. The predicate is evaluated FRESH at every
:func:`collect_registered_residual_bands` call (compile time), so a flag-off
build stays byte-identical (the band is simply not collected → smaller
d_model). Registration itself is always unconditional (import time), so the
registry is deterministic regardless of env state at import.

``never_share``
---------------
Most carry/dump bands hold cross-step or cross-op state and must keep a PRIVATE
residual slot (a dim-liveness merge onto a same-width donor whose lifetime
"ended" would clobber the carried one-hot). Such a band passes
``never_share=True`` and its name is threaded into the compiler's
``_LIVENESS_NEVER_SHARE`` set automatically (see
``LayerCompiler.add_never_share_names``). Bands that are safe to liveness-merge
(e.g. the flag-gated MUL result band) pass ``never_share=False`` (the default).

Public API
----------
    register_residual_band(name, size, *, owner, flag=None, never_share=False)
    collect_registered_residual_bands() -> Dict[str, int]
    collect_never_share_band_names() -> Set[str]

See ``docs/RESIDUAL_BAND_REGISTRY_2026_06_13.md`` and ``c4_release/CLAUDE.md``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class _BandSpec:
    name: str
    size: int
    owner: str
    # Zero-arg predicate evaluated at collect time. ``None`` => always present.
    flag: Optional[Callable[[], bool]]
    # True => name is threaded into the compiler's _LIVENESS_NEVER_SHARE set.
    never_share: bool


# Insertion-ordered registry. Import-time population from the op modules means
# the order is the op-module import order (l0..l16, alu, model_ops, ...), which
# is deterministic. The auto-widen declares bands in this order (the historical
# order — AX bands, then Root 2 bands, then the flag-gated MUL band — is
# preserved by the registration call sites so dim_positions stay byte-identical).
_REGISTRY: "List[_BandSpec]" = []
_NAMES_SEEN: Set[str] = set()


def register_residual_band(
    name: str,
    size: int,
    *,
    owner: str,
    flag: Optional[Callable[[], bool]] = None,
    never_share: bool = False,
) -> None:
    """Declare an over-width residual band owned by ``owner``.

    Call this at MODULE IMPORT TIME (top level of the ``lN_ops`` module that
    owns the band, next to the op factory that reads/writes it). The compiler
    auto-collects every registered band at compile time and feeds the union
    into the head-dim-preserving auto-widen.

    Args:
        name: residual band dim name (referenced by op rules via
            ``layout.dim_positions[name]``). Must be a non-empty str.
        size: number of residual dims (positive int).
        owner: the op / op-family name that owns the band (for diagnostics
            and ``register_residual_band`` collision messages).
        flag: optional zero-arg predicate; when supplied, the band is
            collected only on compiles where ``flag()`` is truthy. Evaluated
            FRESH at each collect (so an env flag flip takes effect without a
            re-import). ``None`` => always present.
        never_share: when True, the band name is threaded into the compiler's
            ``_LIVENESS_NEVER_SHARE`` set so the dim-liveness allocator keeps
            it in a private slot (required for carry / dump bands that hold
            cross-step or cross-op state).

    Re-registering the SAME (name, size, owner) is idempotent (supports module
    re-import under test reload). Registering the same NAME with a different
    size or owner raises ``ValueError`` (a genuine cross-op collision — the
    very class of bug this registry exists to surface early).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"register_residual_band: name must be a non-empty str (got {name!r})"
        )
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(
            f"register_residual_band({name!r}): size must be a positive int "
            f"(got {size!r})"
        )
    if not isinstance(owner, str) or not owner:
        raise ValueError(
            f"register_residual_band({name!r}): owner must be a non-empty str "
            f"(got {owner!r})"
        )
    for existing in _REGISTRY:
        if existing.name == name:
            if (existing.size == size and existing.owner == owner
                    and existing.never_share == never_share):
                # Idempotent re-registration (e.g. module reload under test).
                return
            raise ValueError(
                f"register_residual_band: band {name!r} already registered by "
                f"{existing.owner!r} (size={existing.size}, "
                f"never_share={existing.never_share}); cannot re-register with "
                f"owner={owner!r} size={size} never_share={never_share}. Two "
                f"ops must not claim the same residual band name."
            )
    _REGISTRY.append(
        _BandSpec(
            name=name, size=size, owner=owner, flag=flag,
            never_share=never_share,
        )
    )
    _NAMES_SEEN.add(name)


def collect_registered_residual_bands() -> Dict[str, int]:
    """Return ``{name: size}`` for every band whose flag is active right now.

    Evaluates each band's ``flag`` predicate fresh, so flag-off builds omit the
    corresponding band (smaller d_model). Insertion order is preserved (the
    returned dict's iteration order matches registration order), which is what
    keeps the auto-widen's tail dim_positions byte-identical to the historical
    central-dict ordering.
    """
    out: Dict[str, int] = {}
    for spec in _REGISTRY:
        if spec.flag is not None and not spec.flag():
            continue
        out[spec.name] = spec.size
    return out


def collect_never_share_band_names() -> Set[str]:
    """Return the names of all ACTIVE bands that requested a private slot.

    Mirrors :func:`collect_registered_residual_bands` flag-gating: a flag-off
    band contributes no never-share name. Threaded into the compiler's
    ``_LIVENESS_NEVER_SHARE`` set so carry/dump bands keep private slots.
    """
    out: Set[str] = set()
    for spec in _REGISTRY:
        if not spec.never_share:
            continue
        if spec.flag is not None and not spec.flag():
            continue
        out.add(spec.name)
    return out


def registered_band_specs() -> "List[_BandSpec]":
    """Return the raw registered specs (diagnostics / tests only)."""
    return list(_REGISTRY)
