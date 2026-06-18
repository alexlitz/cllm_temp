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
    register_band_category(name, category, role)        # Phase 7.E.0 Bug-B
    collect_registered_band_categories() -> List[Tuple[str, str, str]]

See ``docs/RESIDUAL_BAND_REGISTRY_2026_06_13.md`` and ``c4_release/CLAUDE.md``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple
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


def collect_alibi_base_residual_bands() -> Dict[str, int]:
    """Return ``{name: size}`` for every band active at its flag's DEFAULT state.

    This is the band set that defines the ALiBi-SLOPE BASE head count -- the
    head count of the GOLDEN default build (every env flag at its default).
    The head-dim-preserving auto-widen derives ``n_heads`` from the residual
    width, and the default ALiBi slope of head ``i`` is ``2**(-8/N*(i+1))``
    with ``N == n_heads``. If a non-default over-width flag is toggled on
    (e.g. ``C4_AX_BYTE1_FULL_WIDTH=1`` appends the 256-cell
    ``AX_BYTE1_FULL_WIDE`` band -> n_heads 10 -> 13), every EXISTING head's
    slope would shift and silently perturb every globally-sized attention
    block. Pinning the slope base to THIS set (flags at default) keeps the
    slope base equal to the golden n_heads on every compile, regardless of
    which over-width flag is toggled, so the widen's trailing padding heads
    are inert (the toggled band's own emission columns still light up; only
    the slope geometry is held constant).

    Each flag predicate is re-evaluated with the ``C4_*`` environment cleared
    so the DEFAULT state is observed (predicates read ``os.environ.get(NAME,
    DEFAULT)``); the environment is restored before returning. Insertion order
    is preserved (matches :func:`collect_registered_residual_bands`).
    """
    import os as _os

    saved = {k: v for k, v in _os.environ.items() if k.startswith("C4_")}
    for k in saved:
        del _os.environ[k]
    try:
        out: Dict[str, int] = {}
        for spec in _REGISTRY:
            if spec.flag is not None and not spec.flag():
                continue
            out[spec.name] = spec.size
        return out
    finally:
        for k, v in saved.items():
            _os.environ[k] = v


def registered_band_specs() -> "List[_BandSpec]":
    """Return the raw registered specs (diagnostics / tests only)."""
    return list(_REGISTRY)


# ============================================================================
# Phase 7.E.0 Bug-B — (category, role) tags for op-local / over-width bands
# ----------------------------------------------------------------------------
# ``_register_default_categories`` (dim_registry.py) only tags the SHARED
# base-registry slots. Op-local over-width bands (``STACK0_BYTE_VAL_*`` and
# the ``register_residual_band`` families) live OUTSIDE that block, so they
# have no ``(category, role)`` and therefore cannot be ``dim_ref``'d. This
# thin wrapper lets a band owner attach a semantic tag NEXT TO the band /
# dim it owns; ``build_default_registry`` collects every tag and applies it
# to the default registry (after the static bindings) so ``dim_ref`` can
# resolve the band by family.
#
# This is a NAME-TABLE-ONLY registration (it never moves a dim or touches a
# weight), so it is byte-identity-safe. A tag whose NAME is absent from the
# built layout would be caught by ``verify_categories_resolve_in_built_layout``
# (tests/test_dim_allocator.py) — the same audit that guards Bug A.
# ============================================================================
@dataclass(frozen=True)
class _BandCategory:
    name: str
    category: str
    role: str


# Insertion-ordered so the applied tags are deterministic. Populated at
# import time of this module (the STACK0_BYTE_VAL tags below) and of any op
# module that calls :func:`register_band_category`.
_CATEGORY_REGISTRY: "List[_BandCategory]" = []
_CATEGORY_KEYS_SEEN: Dict[Tuple[str, str], str] = {}


def register_band_category(name: str, category: str, role: str) -> None:
    """Tag an op-local band / dim ``name`` with a ``(category, role)`` pair.

    Call at MODULE IMPORT TIME (next to the band/dim the tag describes).
    :func:`build_default_registry` collects every tag via
    :func:`collect_registered_band_categories` and applies it to the default
    registry through ``DimRegistry.register_category``, so rule authors can
    then resolve the band with ``dim_ref(category, role, offset)``.

    Pure name-table registration — no dim is allocated or moved, so this is
    byte-identity-safe. The tagged ``name`` MUST be a dim that survives into
    the BUILT layout (the ``verify_categories_resolve_in_built_layout`` audit
    enforces this); tagging a liveness-merged alias would re-introduce the
    Bug-A landmine this whole change exists to remove.

    Re-registering the SAME (name, category, role) triple is idempotent
    (supports module reload under test). Mapping the same (category, role) to
    a DIFFERENT name, or re-tagging the same name's pair to a different one,
    raises ``ValueError`` — a genuine collision (the very class of bug the
    registry surfaces early).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"register_band_category: name must be a non-empty str (got {name!r})"
        )
    if not isinstance(category, str) or not category:
        raise ValueError(
            f"register_band_category({name!r}): category must be a non-empty "
            f"str (got {category!r})"
        )
    if not isinstance(role, str) or not role:
        raise ValueError(
            f"register_band_category({name!r}): role must be a non-empty str "
            f"(got {role!r})"
        )
    key = (category, role)
    existing_name = _CATEGORY_KEYS_SEEN.get(key)
    if existing_name is not None:
        if existing_name == name:
            return  # Idempotent re-registration.
        raise ValueError(
            f"register_band_category: (category={category!r}, role={role!r}) "
            f"already maps to {existing_name!r}; cannot re-map to {name!r}."
        )
    _CATEGORY_REGISTRY.append(_BandCategory(name=name, category=category, role=role))
    _CATEGORY_KEYS_SEEN[key] = name


def collect_registered_band_categories() -> "List[Tuple[str, str, str]]":
    """Return ``[(name, category, role), ...]`` for every registered tag.

    Insertion order is preserved so the tags apply deterministically.
    Consumed by ``build_default_registry`` (dim_registry.py).
    """
    return [(c.name, c.category, c.role) for c in _CATEGORY_REGISTRY]


# ---- STACK0_BYTE_VAL_h_LO/HI (Wave-1 A1 family, dim_registry.py:734..829) ----
# Read by 100+ L10/L13/L14 rules; the highest-value band for the STACK0
# campaign (see ``project_var_fulltrace_stack0_frame_desync`` memory note).
# These are value-bus nibbles broadcast from MARK_AX to the matching STACK0
# byte row during PSH, so they belong to the ``memory_lo`` / ``memory_hi``
# value-bus families (alongside ``MEM_VAL_B*``). Role names the STACK0 byte
# index the band carries.
for _h in (1, 2, 3):
    register_band_category(f"STACK0_BYTE_VAL_{_h}_LO", "memory_lo", f"stack0_b{_h}")
    register_band_category(f"STACK0_BYTE_VAL_{_h}_HI", "memory_hi", f"stack0_b{_h}")
del _h
